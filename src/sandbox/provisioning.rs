//! Per-agent podman image lifecycle.
//!
//! The Containerfile is the source of truth: its SHA256 (plus any sibling
//! context dir) is baked into the built image as a label; a mismatch on a later
//! run forces a rebuild. The hash format is fixed so images built by earlier
//! agentcap versions are reused on upgrade, not needlessly rebuilt.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

const HASH_LABEL: &str = "agentcap.containerfile-hash";
const BUILD_TIMEOUT: Duration = Duration::from_secs(1800);

/// Where the Containerfiles live: `$AGENTCAP_CONTAINERS_DIR`, else the repo's
/// `containers/` (resolved at compile time).
pub fn containers_dir() -> PathBuf {
    std::env::var_os("AGENTCAP_CONTAINERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("containers"))
}

pub fn containerfile_path(agent: &str) -> PathBuf {
    containers_dir().join(format!("agentcap-{agent}.Containerfile"))
}

pub fn image_tag(agent: &str) -> String {
    format!("localhost/agentcap-{agent}:latest")
}

/// sha256(Containerfile bytes) then, for each file in the sibling `<stem>/`
/// context dir (sorted by path): `relpath + \0 + bytes + \0`.
fn containerfile_hash(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    h.update(std::fs::read(path).with_context(|| format!("reading {}", path.display()))?);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ctx = path.parent().unwrap_or(Path::new(".")).join(stem);
    if ctx.is_dir() {
        let mut files = Vec::new();
        collect_files(&ctx, &mut files);
        files.sort();
        for f in files {
            let rel = f.strip_prefix(&ctx).unwrap_or(&f).to_string_lossy();
            h.update(rel.as_bytes());
            h.update([0]);
            h.update(std::fs::read(&f).with_context(|| format!("reading {}", f.display()))?);
            h.update([0]);
        }
    }
    Ok(hex::encode(h.finalize()))
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_files(&p, out);
        } else if p.is_file() {
            out.push(p);
        }
    }
}

fn image_info(tag: &str) -> Option<Value> {
    let out = Command::new("podman").args(["image", "inspect", tag]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice::<Value>(&out.stdout)
        .ok()?
        .as_array()?
        .first()
        .cloned()
}

fn image_stored_hash(info: &Value) -> Option<String> {
    info.get("Labels")
        .or_else(|| info.get("Config").and_then(|c| c.get("Labels")))
        .and_then(|l| l.get(HASH_LABEL))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Build the per-agent image from its Containerfile if absent or stale; return
/// the image tag.
pub fn ensure_image(agent: &str, log: &dyn Fn(&str)) -> Result<String> {
    if super::which_podman().is_none() {
        bail!("podman not on $PATH (brew install podman / apt install podman)");
    }
    let cf = containerfile_path(agent);
    if !cf.is_file() {
        bail!("Containerfile not found: {}", cf.display());
    }
    let tag = image_tag(agent);
    let want = containerfile_hash(&cf)?;

    let existing = image_info(&tag);
    if existing.as_ref().and_then(image_stored_hash).as_deref() == Some(want.as_str()) {
        log(&format!("{tag} ready (Containerfile hash match)"));
        return Ok(tag);
    }
    if existing.is_some() {
        log(&format!("{tag} is stale; rebuilding…"));
        let _ = Command::new("podman").args(["rmi", "--force", &tag]).output();
    } else {
        log(&format!("{tag} not built; building (cold build can take minutes)…"));
    }

    let mut child = Command::new("podman")
        .args(["build", "-f"])
        .arg(&cf)
        .args(["-t", &tag, "--label", &format!("{HASH_LABEL}={want}")])
        .arg(cf.parent().unwrap_or(Path::new(".")))
        .stdin(Stdio::null())
        .spawn()
        .context("spawning podman build")?;
    use wait_timeout::ChildExt;
    let status = child.wait_timeout(BUILD_TIMEOUT).context("waiting on podman build")?;
    let code = match status {
        Some(s) => s.code().unwrap_or(-1),
        None => {
            let _ = child.kill();
            bail!("podman build for {tag} timed out after {}s", BUILD_TIMEOUT.as_secs());
        }
    };
    if code != 0 {
        bail!("podman build failed for {tag} (rc={code}); see streamed output above.");
    }
    log(&format!("{tag} built"));
    Ok(tag)
}

/// macOS only: ensure `podman machine` is up. No-op on Linux.
pub fn ensure_machine_running(log: &dyn Fn(&str)) -> Result<()> {
    if std::env::consts::OS != "macos" {
        return Ok(());
    }
    if super::which_podman().is_none() {
        bail!("podman not on $PATH (brew install podman)");
    }
    let out = Command::new("podman")
        .args(["machine", "list", "--format", "json"])
        .output()?;
    let machines: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    let list = machines.as_array().cloned().unwrap_or_default();
    if list.is_empty() {
        bail!("no podman machine found. Initialise one first:\n    podman machine init\n    podman machine start");
    }
    let default = list
        .iter()
        .find(|m| m.get("Default") == Some(&Value::Bool(true)))
        .unwrap_or(&list[0]);
    if default.get("Running") == Some(&Value::Bool(true)) {
        return Ok(());
    }
    log("podman machine is not running; starting…");
    let r = Command::new("podman").args(["machine", "start"]).output()?;
    if !r.status.success() {
        bail!(
            "podman machine start failed: {}",
            String::from_utf8_lossy(&r.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_changes_with_containerfile_and_context() {
        let tmp = tempfile::tempdir().unwrap();
        let cf = tmp.path().join("agentcap-x.Containerfile");
        std::fs::write(&cf, b"FROM scratch\n").unwrap();
        let h1 = containerfile_hash(&cf).unwrap();

        // Add a sibling context dir → hash changes and is stable.
        let ctx = tmp.path().join("agentcap-x");
        std::fs::create_dir(&ctx).unwrap();
        std::fs::write(ctx.join("init.sh"), b"echo hi\n").unwrap();
        let h2 = containerfile_hash(&cf).unwrap();
        assert_ne!(h1, h2);
        assert_eq!(h2, containerfile_hash(&cf).unwrap());
        assert_eq!(h1.len(), 64);
    }
}
