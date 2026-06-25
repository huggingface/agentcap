//! Resolve a captured request by id and hand back the body.
//!
//! No normalization of the JSON — captures persist the request as parsed JSON,
//! so the original byte sequence isn't recoverable, but the object is.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::Value;

use crate::parquet_io;

/// Return the raw captured request body for `request_id`. `source` resolves a
/// local capture dir, a local `.parquet`, or `hf://datasets/<owner>/<name>`
/// (or the bare `<owner>/<name>`). Errors if the id is not found.
pub fn load_request(source: &str, request_id: &str) -> Result<Value> {
    let mut bodies = load_requests(source, &[request_id.to_string()])?;
    bodies
        .remove(request_id)
        .ok_or_else(|| anyhow::anyhow!("request_id {request_id:?} not found in {source:?}"))
}

/// Batch form: returns `{id: body}`. Errors listing any ids not found.
pub fn load_requests(source: &str, request_ids: &[String]) -> Result<HashMap<String, Value>> {
    let wanted: HashSet<String> = request_ids.iter().cloned().collect();
    if wanted.is_empty() {
        return Ok(HashMap::new());
    }

    // Resolve local paths first — an existing dir/file wins over the HF
    // heuristic, so `runs/abc/captures` isn't misclassified as a repo.
    let p = expanduser(source);
    let bodies = if p.is_dir() {
        load_from_capture_dir(&p, &wanted)?
    } else if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("parquet") {
        parquet_io::read_request_bodies(&p, &wanted)?
    } else if looks_like_hf_source(source) {
        load_from_hf_dataset(source, &wanted)?
    } else {
        bail!("source must be a capture dir, a .parquet file, or an hf://datasets/... URI — got {source:?}");
    };

    let missing: Vec<String> = wanted.iter().filter(|id| !bodies.contains_key(*id)).cloned().collect();
    if !missing.is_empty() {
        let mut m = missing;
        m.sort();
        bail!("request_id(s) not found in {source:?}: {m:?}");
    }
    Ok(bodies)
}

fn looks_like_hf_source(source: &str) -> bool {
    if source.starts_with("hf://") {
        return true;
    }
    if source.starts_with('.') || source.starts_with('/') || source.starts_with('~') {
        return false;
    }
    source.matches('/').count() == 1
}

fn load_from_capture_dir(capture_dir: &Path, wanted: &HashSet<String>) -> Result<HashMap<String, Value>> {
    let mut out = HashMap::new();
    for rid in wanted {
        let path = capture_dir.join(format!("{rid}.request.json"));
        if !path.is_file() {
            continue;
        }
        let rec: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        if let Some(body) = rec.get("body") {
            if body.is_object() {
                out.insert(rid.clone(), body.clone());
            }
        }
    }
    Ok(out)
}

/// Parse `<owner>/<name>` out of an `hf://datasets/...` URI or bare shorthand.
pub fn parse_hf_repo_id(source: &str) -> Result<String> {
    let s = source
        .strip_prefix("hf://datasets/")
        .unwrap_or(source)
        .trim_matches('/');
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        bail!("hf source must be <owner>/<name>, got {source:?}");
    }
    Ok(format!("{}/{}", parts[0], parts[1]))
}

fn load_from_hf_dataset(source: &str, wanted: &HashSet<String>) -> Result<HashMap<String, Value>> {
    let repo_id = parse_hf_repo_id(source)?;
    crate::hub::load_request_bodies_from_dataset(&repo_id, wanted)
}

fn expanduser(source: &str) -> PathBuf {
    if let Some(rest) = source.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(source)
}

/// Find the capture dir + full rid for a (possibly truncated) request id under a
/// workspace. `Ok(Some(...))` for a unique match, `Ok(None)` if absent, and an
/// error when a short prefix matches more than one rid (git-style).
pub fn resolve_workspace_rid(workspace_root: &Path, request_id: &str) -> Result<Option<(PathBuf, String)>> {
    if !workspace_root.is_dir() {
        return Ok(None);
    }
    let mut matches: Vec<(PathBuf, String)> = Vec::new();
    for entry in std::fs::read_dir(workspace_root)? {
        let run_dir = entry?.path();
        let captures = run_dir.join("captures");
        if !captures.is_dir() {
            continue;
        }
        // Exact match shortcut — also makes full-length rids O(1).
        let exact = captures.join(format!("{request_id}.request.json"));
        if exact.is_file() {
            return Ok(Some((captures, request_id.to_string())));
        }
        for e in std::fs::read_dir(&captures)? {
            let name = e?.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.starts_with(request_id) {
                if let Some(stripped) = name.strip_suffix(".request.json") {
                    matches.push((captures.clone(), stripped.to_string()));
                }
            }
        }
    }
    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches.into_iter().next().unwrap())),
        _ => {
            let mut rids: Vec<String> = matches.iter().map(|(_, r)| r.clone()).collect();
            rids.sort();
            bail!(
                "rid prefix {request_id:?} is ambiguous ({} matches): {}",
                rids.len(),
                rids.join(", ")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_capture(dir: &Path, rid: &str, body: &Value) {
        let rec = json!({"request_id": rid, "captured_at": 1, "upstream_url": "http://x", "body": body});
        std::fs::write(dir.join(format!("{rid}.request.json")), rec.to_string()).unwrap();
    }

    #[test]
    fn load_request_from_capture_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        write_capture(&cap, "abc", &body);
        assert_eq!(load_request(cap.to_str().unwrap(), "abc").unwrap(), body);
    }

    #[test]
    fn load_requests_batch() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        write_capture(&cap, "a", &json!({"model": "m", "messages": []}));
        write_capture(&cap, "b", &json!({"model": "m", "messages": [{"role": "user"}]}));
        let out = load_requests(cap.to_str().unwrap(), &["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out["a"]["messages"], json!([]));
    }

    #[test]
    fn missing_id_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        write_capture(&cap, "a", &json!({"model": "m"}));
        assert!(load_request(cap.to_str().unwrap(), "ghost").is_err());
    }

    #[test]
    fn from_parquet_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        let body = json!({"model": "m", "messages": [{"role": "user", "content": "hello"}], "tools": []});
        write_capture(&cap, "rid", &body);
        std::fs::write(
            cap.join("rid.response.json"),
            json!({"request_id": "rid", "captured_at_resp": 2, "status_code": 200, "body": {"choices": []}})
                .to_string(),
        )
        .unwrap();
        let parquet = tmp.path().join("out.parquet");
        let n = parquet_io::write_parquet(&cap, &parquet, None, Some("m"), &[]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(load_request(parquet.to_str().unwrap(), "rid").unwrap(), body);
    }

    #[test]
    fn bad_source_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("nope.txt");
        std::fs::write(&f, "x").unwrap();
        assert!(load_requests(f.to_str().unwrap(), &["a".to_string()]).is_err());
    }

    #[test]
    fn resolve_finds_run() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join(".agentcap");
        let cap = ws.join("hermes-local-20260101-000000").join("captures");
        std::fs::create_dir_all(&cap).unwrap();
        write_capture(&cap, "rid-target", &json!({"model": "m"}));
        let found = resolve_workspace_rid(&ws, "rid-target").unwrap();
        assert_eq!(found, Some((cap, "rid-target".to_string())));
    }

    #[test]
    fn resolve_accepts_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join(".agentcap");
        let cap = ws.join("hermes-local-20260101-000000").join("captures");
        std::fs::create_dir_all(&cap).unwrap();
        write_capture(&cap, "abc12345deadbeef", &json!({"model": "m"}));
        let found = resolve_workspace_rid(&ws, "abc12345").unwrap();
        assert_eq!(found, Some((cap, "abc12345deadbeef".to_string())));
    }

    #[test]
    fn resolve_ambiguous_prefix_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join(".agentcap");
        let cap = ws.join("hermes-local-20260101-000000").join("captures");
        std::fs::create_dir_all(&cap).unwrap();
        write_capture(&cap, "abc12345_a", &json!({"model": "m"}));
        write_capture(&cap, "abc12345_b", &json!({"model": "m"}));
        let err = resolve_workspace_rid(&ws, "abc12345").unwrap_err();
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn resolve_returns_none_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join(".agentcap");
        std::fs::create_dir(&ws).unwrap();
        assert_eq!(resolve_workspace_rid(&ws, "ghost").unwrap(), None);
    }
}
