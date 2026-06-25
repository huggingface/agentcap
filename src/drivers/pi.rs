//! pi-mono driver: `pi -p "<prompt>" --provider local`.
//! pi mints its own session UUID on start and resumes the latest via `--continue`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::{AgentDriver, AgentTurn, DriverError};
use crate::sandbox::PodmanSandbox;

pub struct PiDriver {
    sandbox: Arc<PodmanSandbox>,
    cwd: Option<String>,
    model: Option<String>,
}

impl PiDriver {
    pub fn new(sandbox: Arc<PodmanSandbox>, cwd: Option<String>, model: Option<String>) -> Self {
        PiDriver { sandbox, cwd, model }
    }

    fn argv(&self, prompt: &str, resume: bool) -> Vec<String> {
        let mut argv = vec![
            "pi".into(),
            "-p".into(),
            prompt.into(),
            "--provider".into(),
            "local".into(),
        ];
        if let Some(m) = &self.model {
            argv.extend(["--model".into(), m.clone()]);
        }
        if resume {
            argv.push("--continue".into());
        }
        argv
    }

    fn run(&self, argv: &[String], session_id: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        let out = self.sandbox.run(argv, &BTreeMap::new(), self.cwd.as_deref(), timeout)?;
        Ok(AgentTurn {
            session_id: Some(session_id.to_string()),
            response_text: out.stdout.trim().to_string(),
            returncode: out.code,
            stdout: out.stdout,
            stderr: out.stderr,
            tool_errors: Vec::new(),
        })
    }
}

impl AgentDriver for PiDriver {
    fn name(&self) -> &str {
        "pi"
    }
    fn start(&self, prompt: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        // pi mints its own UUID; resume picks the latest via --continue, so we
        // hand the orchestrator a synthetic marker.
        self.run(&self.argv(prompt, false), "latest", timeout)
    }
    fn resume(&self, prompt: &str, session_id: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        self.run(&self.argv(prompt, true), session_id, timeout)
    }
}
