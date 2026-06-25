//! Hermes driver: `hermes chat -q "<prompt>"` non-interactively. The proxy URL
//! + config are baked into the image.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use regex::Regex;

use super::{AgentDriver, AgentTurn, DriverError};
use crate::sandbox::PodmanSandbox;

fn session_id_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"session_id:\s*([a-zA-Z0-9_\-]+)").unwrap())
}

pub fn parse_session_id(output: &str) -> Option<String> {
    session_id_re().captures(output).map(|c| c[1].to_string())
}

/// Extract the assistant body: for a resumed session, slice after the last
/// `↻ Resumed` marker; then drop bare `session_id:` lines and trim.
pub fn parse_response_text(stdout: &str) -> String {
    let lines: Vec<&str> = stdout.lines().collect();
    let mut last = None;
    for (i, l) in lines.iter().enumerate() {
        if l.contains("Resumed") && l.contains('↻') {
            last = Some(i);
        }
    }
    let body = match last {
        Some(i) => &lines[i + 1..],
        None => &lines[..],
    };
    let re = session_id_re();
    body.iter()
        .filter(|l| re.find(l.trim()).is_none_or(|m| m.start() != 0))
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

pub struct HermesDriver {
    sandbox: Arc<PodmanSandbox>,
    cwd: Option<String>,
    model: Option<String>,
}

impl HermesDriver {
    pub fn new(sandbox: Arc<PodmanSandbox>, cwd: Option<String>, model: Option<String>) -> Self {
        HermesDriver { sandbox, cwd, model }
    }

    fn argv(&self, prompt: &str, session_id: Option<&str>) -> Vec<String> {
        let mut argv = vec![
            "hermes".into(),
            "chat".into(),
            "-q".into(),
            prompt.into(),
            "-Q".into(),
            "--yolo".into(),
            "--accept-hooks".into(),
        ];
        if let Some(m) = &self.model {
            argv.extend(["-m".into(), m.clone()]);
        }
        match session_id {
            None => argv.push("--pass-session-id".into()),
            Some(sid) => argv.extend(["--resume".into(), sid.into()]),
        }
        argv
    }

    fn run(
        &self,
        argv: &[String],
        session_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<AgentTurn, DriverError> {
        let out = self.sandbox.run(argv, &BTreeMap::new(), self.cwd.as_deref(), timeout)?;
        let combined = format!("{}\n{}", out.stdout, out.stderr);
        Ok(AgentTurn {
            session_id: session_id.map(str::to_string).or_else(|| parse_session_id(&combined)),
            response_text: parse_response_text(&out.stdout),
            returncode: out.code,
            stdout: out.stdout,
            stderr: out.stderr,
            tool_errors: Vec::new(),
        })
    }
}

impl AgentDriver for HermesDriver {
    fn name(&self) -> &str {
        "hermes"
    }
    fn start(&self, prompt: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        self.run(&self.argv(prompt, None), None, timeout)
    }
    fn resume(&self, prompt: &str, session_id: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        self.run(&self.argv(prompt, Some(session_id)), Some(session_id), timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_extracted() {
        assert_eq!(
            parse_session_id("noise\nsession_id: abc-12_3\nmore").as_deref(),
            Some("abc-12_3")
        );
        assert_eq!(parse_session_id("no id here"), None);
    }

    #[test]
    fn response_initial_drops_session_line() {
        assert_eq!(parse_response_text("session_id: x\nHello\nWorld"), "Hello\nWorld");
    }

    #[test]
    fn response_resumed_slices_after_marker() {
        let out = "old turn\n↻ Resumed x\nsession_id: y\nReply line";
        assert_eq!(parse_response_text(out), "Reply line");
    }
}
