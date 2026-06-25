//! Agent driver adapters. Ports `drivers/__init__.py` + the four agent modules.
//!
//! A driver wraps an agent CLI so the orchestrator can `start` a session,
//! `resume` it, and read back the final response text. Drivers shell out via the
//! sandbox; configuring the agent to point at the capture proxy is baked into the
//! per-agent image.

mod goose;
mod hermes;
mod opencode;
mod pi;

use std::sync::Arc;
use std::time::Duration;

use crate::sandbox::{PodmanSandbox, SandboxError};

/// One turn of agent execution.
#[derive(Debug, Clone)]
pub struct AgentTurn {
    pub session_id: Option<String>,
    pub response_text: String,
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
    pub tool_errors: Vec<String>,
}

/// Why a turn didn't complete normally.
pub enum DriverError {
    Timeout,
    ResumeUnsupported,
    Other(anyhow::Error),
}

impl From<SandboxError> for DriverError {
    fn from(e: SandboxError) -> Self {
        match e {
            SandboxError::Timeout => DriverError::Timeout,
            SandboxError::Other(e) => DriverError::Other(e),
        }
    }
}

pub trait AgentDriver {
    fn name(&self) -> &str;
    fn start(&self, prompt: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError>;
    fn resume(&self, prompt: &str, session_id: &str, timeout: Option<Duration>) -> Result<AgentTurn, DriverError>;
}

/// Driver names, in registration order (feeds `run --agent` validation).
pub const KNOWN_DRIVERS: &[&str] = &["hermes", "opencode", "goose", "pi"];

pub fn get_driver(
    name: &str,
    sandbox: Arc<PodmanSandbox>,
    cwd: Option<String>,
    model: Option<String>,
) -> anyhow::Result<Box<dyn AgentDriver>> {
    Ok(match name {
        "hermes" => Box::new(hermes::HermesDriver::new(sandbox, cwd, model)),
        "opencode" => Box::new(opencode::OpenCodeDriver::new(sandbox, cwd, model)),
        "goose" => Box::new(goose::GooseDriver::new(sandbox, cwd, model)),
        "pi" => Box::new(pi::PiDriver::new(sandbox, cwd, model)),
        _ => anyhow::bail!("unknown driver: {name:?}; known: {}", KNOWN_DRIVERS.join(", ")),
    })
}

/// Agents whose images ship a `dump-traces` script (SQLite session stores).
/// Symlink-style agents (pi) surface traces directly and return `None`.
pub fn traces_dump_argv_for(agent: &str) -> Option<Vec<String>> {
    matches!(agent, "hermes" | "goose" | "opencode").then(|| vec!["dump-traces".to_string()])
}
