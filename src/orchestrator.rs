//! Drive an agent driver through a corpus with a follow-up strategy. Ports
//! `orchestrator.py`. Proxy-agnostic: the caller wires capture context via the
//! `set_ctx` callback (the proxy stamps it onto each capture).

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::drivers::{AgentDriver, DriverError};
use crate::followups::FollowUp;

pub struct TaskTurnResult {
    pub turn: i64,
    pub returncode: i32,
    pub duration_s: f64,
}

pub struct TaskResult {
    pub task_id: String,
    pub prompt: String,
    pub session_id: Option<String>,
    pub turns: Vec<TaskTurnResult>,
}

impl TaskResult {
    pub fn completed_turns(&self) -> usize {
        self.turns.iter().filter(|t| t.returncode == 0).count()
    }
}

/// Read a tasks file: one prompt per line, `#` comments + blanks skipped.
pub fn read_tasks_txt(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(str::to_string)
        .collect())
}

/// Run every task for `turns` turns. `set_ctx(task_id, turn)` is called before
/// each turn so captures are stamped.
pub fn run_corpus(
    driver: &dyn AgentDriver,
    followup: &dyn FollowUp,
    tasks: &[String],
    turns: i64,
    timeout: Option<Duration>,
    sessions_dir: Option<&Path>,
    set_ctx: &dyn Fn(Option<&str>, Option<i64>),
) -> Vec<TaskResult> {
    tasks
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let task_id = format!("task_{:02}", i + 1);
            run_task(
                driver,
                followup,
                prompt,
                &task_id,
                turns,
                timeout,
                sessions_dir,
                set_ctx,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_task(
    driver: &dyn AgentDriver,
    followup: &dyn FollowUp,
    prompt: &str,
    task_id: &str,
    turns: i64,
    timeout: Option<Duration>,
    sessions_dir: Option<&Path>,
    set_ctx: &dyn Fn(Option<&str>, Option<i64>),
) -> TaskResult {
    let mut result = TaskResult {
        task_id: task_id.to_string(),
        prompt: prompt.to_string(),
        session_id: None,
        turns: Vec::new(),
    };

    // Turn 1: open the session.
    set_ctx(Some(task_id), Some(1));
    let t0 = Instant::now();
    let first = match driver.start(prompt, timeout) {
        Ok(t) => t,
        Err(e) => {
            abort(task_id, "initial-turn", &e);
            return result;
        }
    };
    let dur = t0.elapsed().as_secs_f64();
    log_turn(sessions_dir, task_id, 1, &first.stdout, &first.stderr);
    result.session_id = first.session_id.clone();
    result.turns.push(TaskTurnResult {
        turn: 1,
        returncode: first.returncode,
        duration_s: dur,
    });
    eprintln!(
        "  [turn_done] task_id={task_id} turn=1 rc={} dur={dur:.3}",
        first.returncode
    );

    if first.returncode != 0 {
        eprintln!("  [task_aborted] task_id={task_id} reason=initial-turn-failed");
        return result;
    }
    let mut sid = match &first.session_id {
        Some(s) => s.clone(),
        None if turns > 1 => {
            eprintln!("  [task_aborted] task_id={task_id} reason=no-session-id");
            return result;
        }
        None => return result,
    };

    // Follow-up turns.
    let mut last_response = first.response_text;
    for turn in 2..=turns {
        let next_prompt = followup.next(prompt, &last_response, turn);
        set_ctx(Some(task_id), Some(turn));
        let t0 = Instant::now();
        let fu = match driver.resume(&next_prompt, &sid, timeout) {
            Ok(t) => t,
            Err(e) => {
                abort_turn(task_id, turn, &e);
                break;
            }
        };
        let dur = t0.elapsed().as_secs_f64();
        log_turn(sessions_dir, task_id, turn, &fu.stdout, &fu.stderr);
        result.turns.push(TaskTurnResult {
            turn,
            returncode: fu.returncode,
            duration_s: dur,
        });
        eprintln!(
            "  [turn_done] task_id={task_id} turn={turn} rc={} dur={dur:.3}",
            fu.returncode
        );
        if fu.returncode != 0 {
            break;
        }
        if let Some(s) = fu.session_id {
            sid = s;
        }
        last_response = fu.response_text;
    }
    result
}

fn abort(task_id: &str, phase: &str, e: &DriverError) {
    let reason = match e {
        DriverError::Timeout => format!("{phase}-timeout"),
        DriverError::ResumeUnsupported => "resume-not-supported".to_string(),
        DriverError::Other(err) => {
            eprintln!("  [driver_error] task_id={task_id}: {err:#}");
            format!("{phase}-error")
        }
    };
    eprintln!("  [task_aborted] task_id={task_id} reason={reason}");
}

fn abort_turn(task_id: &str, turn: i64, e: &DriverError) {
    match e {
        DriverError::Timeout => {
            eprintln!("  [task_aborted] task_id={task_id} turn={turn} reason=follow-up-turn-timeout")
        }
        DriverError::ResumeUnsupported => eprintln!("  [task_aborted] task_id={task_id} reason=resume-not-supported"),
        DriverError::Other(err) => eprintln!("  [task_aborted] task_id={task_id} turn={turn} error={err:#}"),
    }
}

fn log_turn(sessions_dir: Option<&Path>, task_id: &str, turn: i64, stdout: &str, stderr: &str) {
    let Some(dir) = sessions_dir else { return };
    let base = dir.join(format!("{task_id}_turn_{turn:02}"));
    let _ = std::fs::write(base.with_extension("out"), stdout);
    let _ = std::fs::write(base.with_extension("err"), stderr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::AgentTurn;
    use std::cell::RefCell;

    struct FakeDriver {
        ok: bool,
    }
    impl AgentDriver for FakeDriver {
        fn name(&self) -> &str {
            "fake"
        }
        fn start(&self, _p: &str, _t: Option<Duration>) -> Result<AgentTurn, DriverError> {
            Ok(AgentTurn {
                session_id: Some("sid1".into()),
                response_text: "r1".into(),
                returncode: if self.ok { 0 } else { 1 },
                stdout: "out".into(),
                stderr: String::new(),
                tool_errors: vec![],
            })
        }
        fn resume(&self, _p: &str, _s: &str, _t: Option<Duration>) -> Result<AgentTurn, DriverError> {
            Ok(AgentTurn {
                session_id: Some("sid1".into()),
                response_text: "r2".into(),
                returncode: 0,
                stdout: "out2".into(),
                stderr: String::new(),
                tool_errors: vec![],
            })
        }
    }

    #[test]
    fn read_tasks_skips_comments_and_blanks() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("tasks.txt");
        std::fs::write(&f, "# header\n\n  do a thing  \n# c\nsecond\n").unwrap();
        assert_eq!(read_tasks_txt(&f).unwrap(), vec!["do a thing", "second"]);
    }

    #[test]
    fn multi_turn_records_turns_and_sets_context() {
        let driver = FakeDriver { ok: true };
        let fu = crate::followups::get_followup("continue", None, None, None).unwrap();
        let ctx: RefCell<Vec<(Option<String>, Option<i64>)>> = RefCell::new(Vec::new());
        let set = |tid: Option<&str>, turn: Option<i64>| ctx.borrow_mut().push((tid.map(str::to_string), turn));
        let results = run_corpus(&driver, fu.as_ref(), &["task".to_string()], 3, None, None, &set);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].completed_turns(), 3);
        assert_eq!(results[0].session_id.as_deref(), Some("sid1"));
        // ctx set once per turn: (task_01,1),(task_01,2),(task_01,3)
        assert_eq!(ctx.borrow().len(), 3);
        assert_eq!(ctx.borrow()[2], (Some("task_01".to_string()), Some(3)));
    }

    #[test]
    fn failed_first_turn_aborts() {
        let driver = FakeDriver { ok: false };
        let fu = crate::followups::get_followup("continue", None, None, None).unwrap();
        let set = |_: Option<&str>, _: Option<i64>| {};
        let results = run_corpus(&driver, fu.as_ref(), &["task".to_string()], 3, None, None, &set);
        assert_eq!(results[0].turns.len(), 1);
        assert_eq!(results[0].completed_turns(), 0);
    }
}
