//! Podman container sandbox. Ports `sandbox/podman.py` + `sandbox/__init__.py`.
//!
//! Each [`PodmanSandbox::run`] is a fresh `podman run --rm` against a pre-built
//! per-agent image. Host paths in `writable_paths` / `readonly_paths` are
//! bind-mounted at the same path so the agent sees identical paths in and out.

pub mod provisioning;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Result;
use wait_timeout::ChildExt;

/// Result of a sandboxed run.
pub struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Why a run didn't return a normal exit.
pub enum SandboxError {
    /// The per-turn timeout elapsed; the child was killed.
    Timeout,
    Other(anyhow::Error),
}

impl From<std::io::Error> for SandboxError {
    fn from(e: std::io::Error) -> Self {
        SandboxError::Other(e.into())
    }
}

fn abs(p: &Path) -> String {
    std::path::absolute(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Assemble a `podman run --rm … <image> <argv>` invocation. Pure (testable).
pub fn build_command(
    argv: &[String],
    image: &str,
    writable_paths: &[PathBuf],
    readonly_paths: &[PathBuf],
    deny_network: bool,
    env: &BTreeMap<String, String>,
    cwd: Option<&str>,
) -> Vec<String> {
    let mut cmd = vec!["podman".to_string(), "run".to_string(), "--rm".to_string()];
    if deny_network {
        cmd.push("--network=none".to_string());
    }
    if let Some(c) = cwd {
        cmd.push("--workdir".to_string());
        cmd.push(c.to_string());
    }

    let mut bound = std::collections::HashSet::new();
    let mut all_writable: Vec<String> = writable_paths.iter().map(|p| abs(p)).collect();
    if let Some(c) = cwd {
        all_writable.push(abs(Path::new(c)));
    }
    for resolved in all_writable {
        if bound.insert(resolved.clone()) {
            cmd.push("--mount".to_string());
            cmd.push(format!("type=bind,src={resolved},dst={resolved}"));
        }
    }
    for p in readonly_paths {
        let resolved = abs(p);
        if bound.insert(resolved.clone()) {
            cmd.push("--mount".to_string());
            cmd.push(format!("type=bind,src={resolved},dst={resolved},ro"));
        }
    }
    for (k, v) in env {
        cmd.push("-e".to_string());
        cmd.push(format!("{k}={v}"));
    }
    cmd.push(image.to_string());
    cmd.extend(argv.iter().cloned());
    cmd
}

/// Image-based sandbox using `podman run --rm`.
pub struct PodmanSandbox {
    image: String,
    extra_env: BTreeMap<String, String>,
    readonly_paths: Vec<PathBuf>,
    writable_paths: Vec<PathBuf>,
}

impl PodmanSandbox {
    pub fn run(
        &self,
        argv: &[String],
        env: &BTreeMap<String, String>,
        cwd: Option<&str>,
        timeout: Option<Duration>,
    ) -> std::result::Result<Output, SandboxError> {
        let mut full_env = self.extra_env.clone();
        full_env.extend(env.iter().map(|(k, v)| (k.clone(), v.clone())));
        let mut wrapped = build_command(
            argv,
            &self.image,
            &self.writable_paths,
            &self.readonly_paths,
            false,
            &full_env,
            cwd,
        );
        // `--rm` only fires on a clean exit; tag every run with a unique name so
        // we can force-remove an orphan no matter how the child returned.
        let name = format!("agentcap-{}", uuid::Uuid::new_v4().simple().to_string().split_at(12).0);
        wrapped.insert(2, "--name".to_string());
        wrapped.insert(3, name.clone());

        let outcome = run_child(&wrapped, timeout);
        // Best-effort cleanup; never shadow the primary outcome.
        let _ = Command::new("podman")
            .args(["rm", "-f", &name])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        outcome
    }
}

/// Spawn the child, draining stdout/stderr on threads (so a chatty agent can't
/// deadlock on a full pipe) and enforcing `timeout`.
fn run_child(wrapped: &[String], timeout: Option<Duration>) -> std::result::Result<Output, SandboxError> {
    let mut child = Command::new(&wrapped[0])
        .args(&wrapped[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = so.read_to_end(&mut b);
        b
    });
    let t_err = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = se.read_to_end(&mut b);
        b
    });

    let status = match timeout {
        Some(d) => child.wait_timeout(d).map_err(SandboxError::from)?,
        None => Some(child.wait().map_err(SandboxError::from)?),
    };
    let timed_out = status.is_none();
    if timed_out {
        let _ = child.kill();
        let _ = child.wait();
    }
    let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
    if timed_out {
        return Err(SandboxError::Timeout);
    }
    Ok(Output {
        code: status.and_then(|s| s.code()).unwrap_or(-1),
        stdout,
        stderr,
    })
}

/// Provision (build the image if needed) and return a sandbox, or an error with
/// an install hint. Ports `require_sandbox_or_die`.
pub fn require_sandbox(
    agent: &str,
    env: BTreeMap<String, String>,
    readonly_paths: Vec<PathBuf>,
    writable_paths: Vec<PathBuf>,
    log: &dyn Fn(&str),
) -> Result<PodmanSandbox> {
    let os = std::env::consts::OS;
    if os != "linux" && os != "macos" {
        anyhow::bail!("agentcap sandboxing is only supported on Linux and macOS; host is {os:?}.");
    }
    if which_podman().is_none() {
        anyhow::bail!(
            "podman is required.\n    Install with: brew install podman (macOS) or apt install podman (Linux)"
        );
    }
    provisioning::ensure_machine_running(log)?;
    provisioning::ensure_image(agent, log)?;
    Ok(PodmanSandbox {
        image: provisioning::image_tag(agent),
        extra_env: env,
        readonly_paths,
        writable_paths,
    })
}

pub(crate) fn which_podman() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join("podman"))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn build_command_minimal() {
        let cmd = build_command(
            &["echo".into(), "hi".into()],
            "img:latest",
            &[],
            &[],
            false,
            &env(&[]),
            None,
        );
        assert_eq!(cmd, vec!["podman", "run", "--rm", "img:latest", "echo", "hi"]);
    }

    #[test]
    fn build_command_mounts_env_cwd_network() {
        let cmd = build_command(
            &["agent".into()],
            "img",
            &[PathBuf::from("/data/rw")],
            &[PathBuf::from("/data/ro")],
            true,
            &env(&[("K", "V")]),
            Some("/work"),
        );
        let joined = cmd.join(" ");
        assert!(joined.contains("--network=none"));
        assert!(joined.contains("--workdir /work"));
        assert!(joined.contains("type=bind,src=/data/rw,dst=/data/rw"));
        assert!(joined.contains("type=bind,src=/data/ro,dst=/data/ro,ro"));
        assert!(joined.contains("type=bind,src=/work,dst=/work")); // cwd auto-mounted rw
        assert!(joined.contains("-e K=V"));
        assert!(joined.ends_with("img agent"));
    }

    #[test]
    fn build_command_dedups_paths() {
        // cwd == a writable path → mounted once.
        let cmd = build_command(
            &["a".into()],
            "img",
            &[PathBuf::from("/work")],
            &[],
            false,
            &env(&[]),
            Some("/work"),
        );
        let mounts = cmd.iter().filter(|s| s.starts_with("type=bind")).count();
        assert_eq!(mounts, 1);
    }
}
