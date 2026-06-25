//! OpenCode driver: `opencode run --format json`.
//! OpenCode emits NDJSON events on stdout: `text` events carry assistant chunks,
//! every event has `sessionID`, `tool_use` events carry error states.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use super::{AgentDriver, AgentTurn, DriverError};
use crate::sandbox::PodmanSandbox;

fn events(stdout: &str) -> impl Iterator<Item = Value> + '_ {
    stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
}

pub fn parse_response_text(stdout: &str) -> String {
    let mut parts = String::new();
    for obj in events(stdout) {
        if obj.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(t) = obj.get("text").and_then(Value::as_str) {
                parts.push_str(t);
            }
        }
    }
    parts.trim().to_string()
}

pub fn parse_session_id(stdout: &str) -> Option<String> {
    for obj in events(stdout) {
        if let Some(s) = obj.get("sessionID").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
        if let Some(s) = obj
            .get("part")
            .and_then(|p| p.get("sessionID"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return Some(s.to_string());
        }
    }
    None
}

pub fn parse_tool_errors(stdout: &str) -> Vec<String> {
    let mut errors = Vec::new();
    for obj in events(stdout) {
        if obj.get("type").and_then(Value::as_str) != Some("tool_use") {
            continue;
        }
        let part = obj.get("part").cloned().unwrap_or(Value::Null);
        let state = part.get("state").cloned().unwrap_or(Value::Null);
        if state.get("status").and_then(Value::as_str) != Some("error") {
            continue;
        }
        let tool = part.get("tool").and_then(Value::as_str).unwrap_or("<unknown>");
        let msg = state
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("(no error message)");
        errors.push(format!("{tool}: {msg}"));
    }
    errors
}

pub struct OpenCodeDriver {
    sandbox: Arc<PodmanSandbox>,
    cwd: Option<String>,
    model: Option<String>,
    provider_name: String,
}

impl OpenCodeDriver {
    pub fn new(sandbox: Arc<PodmanSandbox>, cwd: Option<String>, model: Option<String>) -> Self {
        OpenCodeDriver {
            sandbox,
            cwd,
            model,
            provider_name: "local".into(),
        }
    }

    fn argv(&self, prompt: &str, session_id: Option<&str>) -> Vec<String> {
        let mut argv = vec!["opencode".into(), "run".into(), "--format".into(), "json".into()];
        if let Some(m) = &self.model {
            argv.extend(["--model".into(), format!("{}/{m}", self.provider_name)]);
        }
        if let Some(sid) = session_id {
            argv.extend(["--session".into(), sid.into()]);
        }
        argv.push(prompt.into());
        argv
    }

    fn run(
        &self,
        argv: &[String],
        fallback_sid: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<AgentTurn, DriverError> {
        let out = self.sandbox.run(argv, &BTreeMap::new(), self.cwd.as_deref(), timeout)?;
        Ok(AgentTurn {
            session_id: parse_session_id(&out.stdout).or_else(|| fallback_sid.map(str::to_string)),
            response_text: parse_response_text(&out.stdout),
            returncode: out.code,
            tool_errors: parse_tool_errors(&out.stdout),
            stdout: out.stdout,
            stderr: out.stderr,
        })
    }
}

impl AgentDriver for OpenCodeDriver {
    fn name(&self) -> &str {
        "opencode"
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

    const NDJSON: &str = r#"{"type":"text","text":"Hel","sessionID":"s1"}
{"type":"tool_use","part":{"tool":"edit","state":{"status":"error","error":"boom"}}}
not json
{"type":"text","text":"lo"}"#;

    #[test]
    fn response_concatenates_text_events() {
        assert_eq!(parse_response_text(NDJSON), "Hello");
    }

    #[test]
    fn session_id_from_first_event() {
        assert_eq!(parse_session_id(NDJSON).as_deref(), Some("s1"));
        let nested = r#"{"part":{"sessionID":"s2"}}"#;
        assert_eq!(parse_session_id(nested).as_deref(), Some("s2"));
    }

    #[test]
    fn tool_errors_surfaced() {
        assert_eq!(parse_tool_errors(NDJSON), vec!["edit: boom"]);
    }
}
