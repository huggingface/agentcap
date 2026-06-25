//! Goose driver: `goose run -t "<prompt>"`. Ports `drivers/goose.py`. The proxy
//! URL + provider are baked into the image ENV; the driver sets `GOOSE_MODEL`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::{AgentDriver, AgentTurn, DriverError};
use crate::sandbox::PodmanSandbox;

pub struct GooseDriver {
    sandbox: Arc<PodmanSandbox>,
    cwd: Option<String>,
    model: Option<String>,
}

impl GooseDriver {
    pub fn new(sandbox: Arc<PodmanSandbox>, cwd: Option<String>, model: Option<String>) -> Self {
        GooseDriver { sandbox, cwd, model }
    }

    fn argv(&self, prompt: &str, session_name: &str, resume: bool) -> Vec<String> {
        let mut argv = vec![
            "goose".into(),
            "run".into(),
            "-t".into(),
            prompt.into(),
            "--name".into(),
            session_name.into(),
        ];
        if resume {
            argv.push("--resume".into());
        }
        argv
    }

    fn env(&self) -> BTreeMap<String, String> {
        let mut e = BTreeMap::new();
        if let Some(m) = &self.model {
            e.insert("GOOSE_MODEL".into(), m.clone());
        }
        e
    }

    fn run(&self, argv: &[String], session_name: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        let out = self.sandbox.run(argv, &self.env(), self.cwd.as_deref(), timeout)?;
        Ok(AgentTurn {
            session_id: Some(session_name.to_string()),
            response_text: out.stdout.trim().to_string(),
            returncode: out.code,
            stdout: out.stdout,
            stderr: out.stderr,
            tool_errors: Vec::new(),
        })
    }
}

impl AgentDriver for GooseDriver {
    fn name(&self) -> &str {
        "goose"
    }
    fn start(&self, prompt: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        let session_name = format!("agentcap-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        self.run(&self.argv(prompt, &session_name, false), &session_name, timeout)
    }
    fn resume(&self, prompt: &str, session_id: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError> {
        self.run(&self.argv(prompt, session_id, true), session_id, timeout)
    }
}
