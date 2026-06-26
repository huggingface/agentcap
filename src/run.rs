//! The `run` command: drive an agent CLI through a corpus, capture every
//! chat-completion, and summarise.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::drivers::{get_driver, traces_dump_argv_for};
use crate::export::workspace_root;
use crate::followups::get_followup;
use crate::orchestrator::{read_tasks_txt, run_corpus, TaskResult};
use crate::provider::{hostname_fallback, refine_for_sub_provider};
use crate::proxy::serve_in_thread;
use crate::sandbox::require_sandbox;

#[allow(clippy::too_many_arguments)]
pub fn run(
    agent: String,
    model: Option<String>,
    upstream: String,
    api_key: Option<String>,
    sandbox_dir: Option<String>,
    skills_dir: Option<String>,
    tool_dir: Option<String>,
    tasks_file: String,
    turns: i64,
    followup: String,
    timeout: f64,
) -> Result<()> {
    let model = model.context("--model is required")?;
    let (api_key, api_key_source) = resolve_api_key(&upstream, api_key);
    if let Some(src) = &api_key_source {
        if is_hf_router(&upstream) {
            eprintln!("  [auth] HF Router token source={src}");
        }
    }

    let fu = if followup == "synthesized" {
        get_followup("synthesized", Some(&upstream), Some(&model), api_key.as_deref())?
    } else {
        get_followup(&followup, None, None, None)?
    };

    let provider_slug = refine_for_sub_provider(&hostname_fallback(&upstream), Some(&model));
    eprintln!("  [provider] {provider_slug}");

    let utc = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let workdir = workspace_root().join(format!("{agent}-{}-{utc}", provider_slug.replace('/', "-")));
    let captures = workdir.join("captures");
    let sessions = workdir.join("sessions");
    let traces = workdir.join("traces");
    let state = workdir.join("state");
    for d in [&captures, &sessions, &traces, &state] {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    eprintln!("  [workdir] {}", workdir.display());

    // Stub run.json so ls/inspect/export can discover this run while in flight.
    write_run_json(
        &workdir,
        &agent,
        &model,
        &provider_slug,
        &upstream,
        turns,
        &followup,
        &[],
    )?;

    let sandbox_cwd = match &sandbox_dir {
        Some(d) => abs(Path::new(d)),
        None => {
            let d = workdir.join("sandbox");
            std::fs::create_dir_all(&d)?;
            abs(&d)
        }
    };

    let tasks = read_tasks_txt(Path::new(&tasks_file))?;
    if tasks.is_empty() {
        bail!("no tasks found in {tasks_file}");
    }

    let proxy = serve_in_thread(&upstream, &captures, "0.0.0.0")?;
    eprintln!("  [proxy] {}", proxy.proxy_url());

    let mut env: BTreeMap<String, String> = BTreeMap::from([
        ("AGENTCAP_PROXY_URL".into(), proxy.proxy_url()),
        ("AGENTCAP_MODEL".into(), model.clone()),
        ("AGENTCAP_PROVIDER".into(), provider_slug.clone()),
        ("AGENTCAP_TRACES_DIR".into(), abs(&traces)),
        ("AGENTCAP_STATE_DIR".into(), abs(&state)),
    ]);
    if let Some(k) = &api_key {
        env.insert("AGENTCAP_API_KEY".into(), k.clone());
    }
    let mut readonly: Vec<PathBuf> = Vec::new();
    if let Some(s) = &skills_dir {
        let skills_abs = abs(Path::new(s));
        env.insert("AGENTCAP_SKILLS_DIR".into(), skills_abs.clone());
        readonly.push(PathBuf::from(skills_abs));
    }
    // A self-contained toolchain dir, mounted read-only at its host path.
    if let Some(t) = &tool_dir {
        let (tool_bin, mount) = tool_dir_wiring(t);
        env.insert("AGENTCAP_TOOL_BIN".into(), tool_bin);
        readonly.push(mount);
    }
    let writable: Vec<PathBuf> = vec![
        PathBuf::from(abs(&traces)),
        PathBuf::from(abs(&state)),
        PathBuf::from(&sandbox_cwd),
    ];

    // First call per agent builds/boots the image; can take minutes.
    let sandbox = Arc::new(require_sandbox(&agent, env.clone(), readonly, writable, &|m| {
        eprintln!("  [sandbox] {m}")
    })?);
    let driver = get_driver(&agent, sandbox.clone(), Some(sandbox_cwd.clone()), Some(model.clone()))?;

    eprintln!(
        "agentcap run: {} tasks × {turns} turns through {agent} -> {upstream}",
        tasks.len()
    );
    let to = Some(Duration::from_secs_f64(timeout));
    let results = run_corpus(
        driver.as_ref(),
        fu.as_ref(),
        &tasks,
        turns,
        to,
        Some(&sessions),
        &|tid, turn| proxy.set_context(tid, turn),
    );

    // Dump SQLite-stored sessions for agents whose images ship `dump-traces`.
    if let Some(argv) = traces_dump_argv_for(&agent) {
        match sandbox.run(&argv, &env, Some(&sandbox_cwd), Some(Duration::from_secs(600))) {
            Ok(o) if o.code != 0 => eprintln!("  [traces] dump-traces rc={}", o.code),
            Ok(_) => {}
            Err(_) => eprintln!("  [traces] dump-traces failed"),
        }
    }
    proxy.shutdown();

    write_run_json(
        &workdir,
        &agent,
        &model,
        &provider_slug,
        &upstream,
        turns,
        &followup,
        &results,
    )?;
    let n_ok = results.iter().filter(|r| r.completed_turns() as i64 == turns).count();
    eprintln!(
        "agentcap run: {n_ok}/{} tasks completed all {turns} turns; summary -> {}",
        results.len(),
        workdir.join("run.json").display()
    );
    Ok(())
}

fn abs(p: &Path) -> String {
    std::path::absolute(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Sandbox wiring for `--tool-dir`: the `AGENTCAP_TOOL_BIN` value (the bundle's
/// `bin/`) and the read-only mount. The mount is the bundle *root*, not `bin/`,
/// so the interpreter and libs that `bin/` shebangs into come too; both are
/// absolute so the src==dst bind keeps those shebangs valid in-container.
fn tool_dir_wiring(tool_dir: &str) -> (String, PathBuf) {
    let root = abs(Path::new(tool_dir));
    let bin = abs(&Path::new(&root).join("bin"));
    (bin, PathBuf::from(root))
}

fn is_hf_router(upstream: &str) -> bool {
    url::Url::parse(upstream)
        .ok()
        .and_then(|u| u.host_str().map(str::to_lowercase))
        .as_deref()
        == Some("router.huggingface.co")
}

/// Explicit `--api-key`/`AGENTCAP_API_KEY`; for HF Router only, fall back to
/// `HF_TOKEN` then the cached token file.
fn resolve_api_key(upstream: &str, explicit: Option<String>) -> (Option<String>, Option<String>) {
    if let Some(k) = explicit.filter(|s| !s.is_empty()) {
        return (Some(k), Some("--api-key / AGENTCAP_API_KEY".into()));
    }
    if !is_hf_router(upstream) {
        return (None, None);
    }
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let t = t.trim();
        if !t.is_empty() {
            return (Some(t.to_string()), Some("HF_TOKEN".into()));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".cache/huggingface/token");
        if let Ok(t) = std::fs::read_to_string(p) {
            let t = t.trim();
            if !t.is_empty() {
                return (Some(t.to_string()), Some("~/.cache/huggingface/token".into()));
            }
        }
    }
    (None, None)
}

#[allow(clippy::too_many_arguments)]
fn write_run_json(
    workdir: &Path,
    agent: &str,
    model: &str,
    provider: &str,
    upstream: &str,
    turns: i64,
    followup: &str,
    results: &[TaskResult],
) -> Result<()> {
    let tasks: Vec<_> = results
        .iter()
        .map(|r| {
            json!({
                "task_id": r.task_id,
                "prompt": r.prompt,
                "session_id": r.session_id,
                "completed_turns": r.completed_turns(),
                "turns": r.turns.iter().map(|t| json!({
                    "turn": t.turn,
                    "returncode": t.returncode,
                    "duration_s": (t.duration_s * 1000.0).round() / 1000.0,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let summary = json!({
        "agent": agent, "model": model, "provider": provider, "upstream": upstream,
        "turns_per_task": turns, "followup": followup, "tasks": tasks,
    });
    std::fs::write(workdir.join("run.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_dir_wiring_points_at_bin_and_mounts_root() {
        let (bin, mount) = tool_dir_wiring("/opt/toolbox");
        // bin/ is on PATH; the whole bundle (interpreter + libs, not just bin/) is mounted.
        assert_eq!(bin, "/opt/toolbox/bin");
        assert_eq!(mount, PathBuf::from("/opt/toolbox"));
    }

    #[test]
    fn tool_dir_wiring_absolutizes_relative_paths() {
        // Relocatable src==dst mount needs absolute paths even for a relative arg.
        let (bin, mount) = tool_dir_wiring("toolbox");
        assert!(mount.is_absolute(), "mount not absolute: {mount:?}");
        assert!(Path::new(&bin).is_absolute(), "bin not absolute: {bin}");
        assert!(bin.ends_with("toolbox/bin"), "bin not under toolbox/bin: {bin}");
    }
}
