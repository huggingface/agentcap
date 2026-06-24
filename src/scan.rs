//! Secret scan over a capture run, gating `export`.
//!
//! Shells out to `trufflehog filesystem` and parses its JSON. Policy (matching
//! the Python `scan.py`): a single **verified** hit aborts the export;
//! **unverified** hits are reported but non-blocking (pattern matchers have a
//! real false-positive rate). Results are cached to `<run_dir>/scan.json`; the
//! cache is invalidated by `rescan` or when a pattern-only cache can't satisfy a
//! verify request.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCAN_CACHE_NAME: &str = "scan.json";

const SCAN_SUBDIRS: &[&str] = &["captures", "traces", "sessions"];

const INSTALL_HINT: &str = "trufflehog is required for the pre-export secret scan but was not found on PATH. \
Install with:\n    curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
| sh -s -- -b ~/.local/bin\nOr pass --no-scan to `agentcap export` to skip the scan.";

/// One potential secret trufflehog flagged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanHit {
    pub detector: String,
    pub file: String,
    pub verified: bool,
    /// Trufflehog's redacted `Raw` field, capped for context.
    pub raw: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanResult {
    #[serde(default)]
    pub bytes_scanned: i64,
    #[serde(default)]
    pub chunks_scanned: i64,
    #[serde(default)]
    pub verified: Vec<ScanHit>,
    #[serde(default)]
    pub unverified: Vec<ScanHit>,
}

/// Locate `trufflehog`: PATH first, then `~/.local/bin` (the installer's default
/// target). Errors with an install hint if absent.
pub fn find_trufflehog() -> Result<PathBuf> {
    find_in(
        std::env::var("PATH").ok(),
        std::env::var("HOME").ok(),
        is_executable_file,
    )
}

/// Pure core of [`find_trufflehog`], testable without touching real env/fs.
fn find_in(path: Option<String>, home: Option<String>, exists_exec: impl Fn(&Path) -> bool) -> Result<PathBuf> {
    if let Some(path) = path {
        for dir in std::env::split_paths(&path) {
            let cand = dir.join("trufflehog");
            if exists_exec(&cand) {
                return Ok(cand);
            }
        }
    }
    if let Some(home) = home {
        let cand = PathBuf::from(home).join(".local/bin/trufflehog");
        if exists_exec(&cand) {
            return Ok(cand);
        }
    }
    bail!("{INSTALL_HINT}");
}

#[cfg(unix)]
fn is_executable_file(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(p: &Path) -> bool {
    p.is_file()
}

/// Scan a directory or file with trufflehog. With `no_verification=false`
/// (default) candidates are round-tripped against the provider API, so the
/// `verified` bucket is high-precision (requires network). `true` is offline
/// pattern-only matching — everything lands as `unverified`.
pub fn scan_path(path: &Path, no_verification: bool) -> Result<ScanResult> {
    let bin = find_trufflehog()?;
    let mut cmd = Command::new(&bin);
    cmd.arg("filesystem")
        .arg(path)
        .args(["--json", "--no-color", "--results=verified,unverified"]);
    if no_verification {
        cmd.arg("--no-verification");
    }
    let out = cmd
        .output()
        .with_context(|| format!("running trufflehog at {}", bin.display()))?;

    let mut result = ScanResult::default();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if rec.get("DetectorName").is_none() {
            continue;
        }
        let hit = ScanHit {
            detector: rec
                .get("DetectorName")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            file: rec
                .pointer("/SourceMetadata/Data/Filesystem/file")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            verified: rec.get("Verified").and_then(Value::as_bool).unwrap_or(false),
            raw: rec
                .get("Raw")
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(80)
                .collect(),
        };
        if hit.verified {
            result.verified.push(hit);
        } else {
            result.unverified.push(hit);
        }
    }

    // stderr summary: `... finished scanning {"chunks":..., "bytes":...}`.
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        if !line.contains("finished scanning") {
            continue;
        }
        if let Some(brace) = line.find('{') {
            if let Ok(stats) = serde_json::from_str::<Value>(&line[brace..]) {
                result.bytes_scanned = stats.get("bytes").and_then(Value::as_i64).unwrap_or(0);
                result.chunks_scanned = stats.get("chunks").and_then(Value::as_i64).unwrap_or(0);
                break;
            }
        }
    }
    Ok(result)
}

/// Return a persisted scan if it covers the requested mode. A pattern-only cache
/// (`no_verification=true`) can't satisfy a verify request.
pub fn load_cached_scan(run_dir: &Path, no_verification: bool) -> Option<ScanResult> {
    let cache_path = run_dir.join(SCAN_CACHE_NAME);
    let text = std::fs::read_to_string(&cache_path).ok()?;
    let d: Value = serde_json::from_str(&text).ok()?;
    let cached_no_verify = d.get("no_verification").and_then(Value::as_bool).unwrap_or(true);
    if cached_no_verify && !no_verification {
        return None;
    }
    serde_json::from_value(d).ok()
}

/// Scan a run dir (captures/ traces/ sessions/), persisting `scan.json`. Returns
/// `(result, was_cached)`.
pub fn scan_run_dir(run_dir: &Path, no_verification: bool, rescan: bool) -> Result<(ScanResult, bool)> {
    if !rescan {
        if let Some(cached) = load_cached_scan(run_dir, no_verification) {
            return Ok((cached, true));
        }
    }

    let mut merged = ScanResult::default();
    for name in SCAN_SUBDIRS {
        let sub = run_dir.join(name);
        if !sub.is_dir() {
            continue;
        }
        let part = scan_path(&sub, no_verification)?;
        merged.bytes_scanned += part.bytes_scanned;
        merged.chunks_scanned += part.chunks_scanned;
        merged.verified.extend(part.verified);
        merged.unverified.extend(part.unverified);
    }

    // Cache write isn't load-bearing — let the result through on failure.
    let cache = serde_json::json!({
        "scanned_at": now_secs(),
        "no_verification": no_verification,
        "bytes_scanned": merged.bytes_scanned,
        "chunks_scanned": merged.chunks_scanned,
        "verified": merged.verified,
        "unverified": merged.unverified,
    });
    if let Ok(text) = serde_json::to_string_pretty(&cache) {
        let _ = std::fs::write(run_dir.join(SCAN_CACHE_NAME), text);
    }
    Ok((merged, false))
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_in_path_then_home_then_missing() {
        // PATH hit wins.
        let got = find_in(Some("/aa:/bb".into()), None, |p| p.starts_with("/bb")).unwrap();
        assert_eq!(got, PathBuf::from("/bb/trufflehog"));
        // No PATH hit → ~/.local/bin fallback.
        let got = find_in(Some("/aa".into()), Some("/home/u".into()), |p| {
            p == Path::new("/home/u/.local/bin/trufflehog")
        })
        .unwrap();
        assert_eq!(got, PathBuf::from("/home/u/.local/bin/trufflehog"));
        // Nothing → install hint.
        let err = find_in(Some("/aa".into()), Some("/home/u".into()), |_| false).unwrap_err();
        assert!(err.to_string().contains("install.sh"));
    }

    #[test]
    fn cached_scan_missing_and_invalid() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_cached_scan(tmp.path(), false).is_none());
        std::fs::write(tmp.path().join(SCAN_CACHE_NAME), "not json").unwrap();
        assert!(load_cached_scan(tmp.path(), false).is_none());
    }

    #[test]
    fn pattern_only_cache_rejected_for_verify_request() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = serde_json::json!({
            "scanned_at": 0, "no_verification": true,
            "bytes_scanned": 0, "chunks_scanned": 0, "verified": [], "unverified": []
        });
        std::fs::write(tmp.path().join(SCAN_CACHE_NAME), cache.to_string()).unwrap();
        assert!(load_cached_scan(tmp.path(), false).is_none());
        assert!(load_cached_scan(tmp.path(), true).is_some());
    }
}
