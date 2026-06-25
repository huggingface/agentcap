//! `export`: render capture dirs to parquet and push to the Hub.
//!
//! Three
//! artifacts per push: `<owner>/<base>-captures` (parquet), one
//! `<owner>/<base>-<agent>-traces` per agent (raw native traces), and a
//! Collection titled `<base>` grouping them. The trufflehog gate (verified hits
//! abort) runs first unless `--no-scan`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use hf_hub::{HFClientSync, RepoTypeDataset};
use serde_json::Value;

use crate::hub::{self, CommitOperation};
use crate::parquet_io;
use crate::provider::{hostname_fallback, refine_for_sub_provider};
use crate::scan;

const WORKSPACE_DIR: &str = ".agentcap";

/// Workspace root: `$AGENTCAP_WORKSPACE/.agentcap` else `./.agentcap`.
pub fn workspace_root() -> PathBuf {
    let base = std::env::var("AGENTCAP_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());
    base.join(WORKSPACE_DIR)
}

fn workspace_source() -> String {
    match std::env::var("AGENTCAP_WORKSPACE") {
        Ok(v) => format!("AGENTCAP_WORKSPACE={v:?}"),
        Err(_) => "cwd (AGENTCAP_WORKSPACE unset)".to_string(),
    }
}

/// Split `<owner>/<base>` (optionally `hf://datasets/`-prefixed).
pub fn parse_collection_base(uri: &str) -> Result<(String, String)> {
    let s = uri.strip_prefix("hf://datasets/").unwrap_or(uri).trim_matches('/');
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        bail!("--push must be <owner>/<base>, got {uri:?}");
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

pub fn captures_repo_id(owner: &str, base: &str) -> String {
    format!("{owner}/{base}-captures")
}

pub fn traces_repo_id_for(owner: &str, base: &str, agent: &str) -> String {
    format!("{owner}/{base}-{agent}-traces")
}

const FILENAME_SAFE: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.";

/// Filename-safe slug; strips an `org/` prefix from HF model ids.
fn slug(s: &str) -> String {
    let last = s.rsplit('/').next().unwrap_or(s);
    let mut out: String = last
        .chars()
        .map(|c| if FILENAME_SAFE.contains(c) { c } else { '-' })
        .collect();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let trimmed = out.trim_matches(|c| c == '-' || c == '_' || c == '.');
    if trimmed.is_empty() {
        "x".to_string()
    } else {
        trimmed.to_string()
    }
}

/// `train-[<agent>-<model>-<provider>-]<utc>-<hex6>.parquet`.
fn default_filename(agent: Option<&str>, model: Option<&str>, provider: Option<&str>) -> String {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();
    let mut parts = vec!["train".to_string()];
    if let Some(a) = agent {
        parts.push(slug(a));
    }
    if let Some(m) = model {
        parts.push(slug(m));
    }
    if let Some(p) = provider {
        // Preserve hf-router/fireworks-ai → hf-router-fireworks-ai (slug would
        // otherwise strip everything before the last `/`).
        parts.push(slug(&p.replace('/', "-")));
    }
    parts.push(ts);
    parts.push(rand_hex6());
    format!("{}.parquet", parts.join("-"))
}

fn rand_hex6() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h = DefaultHasher::new();
    (nanos, COUNTER.fetch_add(1, Ordering::Relaxed)).hash(&mut h);
    format!("{:06x}", h.finish() & 0xff_ffff)
}

/// Unique bare `body.model` across all captured requests, or `None`. Errors on
/// mixed models (datasets never mix models). `@revision` suffixes are stripped.
pub fn detect_model(capture_dir: &Path) -> Result<Option<String>> {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for req_path in sorted_request_files(capture_dir) {
        let Ok(rec) = read_json(&req_path) else { continue };
        if let Some(m) = rec.get("body").and_then(|b| b.get("model")).and_then(Value::as_str) {
            if !m.is_empty() {
                seen.insert(m.split('@').next().unwrap_or(m).to_string());
            }
        }
    }
    if seen.len() > 1 {
        let models: Vec<&String> = seen.iter().collect();
        bail!(
            "capture dir contains requests for multiple models: {models:?}. Datasets never mix models — \
             split into separate capture dirs and export each one independently."
        );
    }
    Ok(seen.into_iter().next())
}

/// Ordered `[(provider, ..), (upstream_url, ..)]` from the per-request stamp, or
/// empty for legacy capture dirs missing it.
fn detect_provider_columns(capture_dir: &Path) -> Vec<(String, String)> {
    for req_path in sorted_request_files(capture_dir) {
        let Ok(rec) = read_json(&req_path) else { continue };
        let Some(upstream) = rec
            .get("upstream_url")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let model = rec.get("body").and_then(|b| b.get("model")).and_then(Value::as_str);
        let provider = refine_for_sub_provider(&hostname_fallback(upstream), model);
        return vec![
            ("provider".to_string(), provider),
            ("upstream_url".to_string(), upstream.to_string()),
        ];
    }
    Vec::new()
}

fn sorted_request_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".request.json"))
        })
        .collect();
    files.sort();
    files
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

struct CapItem {
    capture_dir: PathBuf,
    model: String,
    agent: Option<String>,
    run_id: String,
}

struct TraceItem {
    traces_dir: PathBuf,
    run_id: String,
}

/// Entry point for the `export` subcommand.
pub fn run(targets: Vec<String>, all_runs: bool, push: String, no_scan: bool) -> Result<()> {
    let (owner, base) = parse_collection_base(&push)?;
    if all_runs && !targets.is_empty() {
        bail!("pass --all OR positional run-ids, not both");
    }
    if !all_runs && targets.is_empty() {
        bail!("specify one or more run-ids/paths, or pass --all");
    }

    let workspace = workspace_root();
    let targets = if all_runs {
        if !workspace.is_dir() {
            bail!(
                "no workspace at {:?} (from {}). Run `agentcap run` first, or set AGENTCAP_WORKSPACE.",
                workspace,
                workspace_source()
            );
        }
        let mut runs: Vec<String> = std::fs::read_dir(&workspace)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() && p.join("run.json").is_file())
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_string))
            .collect();
        runs.sort();
        if runs.is_empty() {
            bail!("no runs in {}", workspace.display());
        }
        runs
    } else {
        targets
    };

    let mut cap_items = Vec::new();
    let mut trace_items = Vec::new();
    for t in &targets {
        let (cap_dir, agent, run_id) = resolve_target(&workspace, t)?;
        let model = match detect_model(&cap_dir)? {
            Some(m) => m,
            None => {
                if all_runs {
                    eprintln!("  [{t}] skipped (no captures)");
                    continue;
                }
                bail!("{} has no captured requests with a model field", cap_dir.display());
            }
        };
        let traces_dir = cap_dir.parent().unwrap_or(Path::new(".")).join("traces");
        let n_traces = count_files(&traces_dir);
        eprintln!(
            "  [{t}] (agent={}, model={model}, traces={n_traces})",
            agent.as_deref().unwrap_or("?")
        );
        cap_items.push(CapItem {
            capture_dir: cap_dir,
            model,
            agent,
            run_id: run_id.clone(),
        });
        trace_items.push(TraceItem { traces_dir, run_id });
    }
    if cap_items.is_empty() {
        bail!("no runs with captures to export");
    }

    if !no_scan {
        let run_dirs: Vec<PathBuf> = cap_items
            .iter()
            .map(|c| c.capture_dir.parent().unwrap_or(Path::new(".")).to_path_buf())
            .collect();
        let n_verified = scan_run_dirs(&run_dirs, false, false)?;
        if n_verified > 0 {
            bail!(
                "export aborted: trufflehog found {n_verified} verified secret(s) — see output above. \
                 Inspect, redact, or pass --no-scan to override."
            );
        }
    }

    let client = hub::client_sync()?;
    let (cap_repo, n_rows) = push_captures_dataset(&client, &cap_items, &owner, &base)?;
    eprintln!(
        "agentcap export: pushed {} rows across {} run(s) -> {cap_repo}",
        n_rows.iter().sum::<usize>(),
        cap_items.len()
    );

    // One traces dataset per agent (homogeneous schema for the Hub viewer).
    let mut by_agent: BTreeMap<String, Vec<&TraceItem>> = BTreeMap::new();
    for (cap, tr) in cap_items.iter().zip(trace_items.iter()) {
        if count_files(&tr.traces_dir) == 0 {
            continue;
        }
        let agent = cap.agent.clone().unwrap_or_else(|| "unknown".to_string());
        by_agent.entry(agent).or_default().push(tr);
    }

    let mut traces_repos = Vec::new();
    for (agent, items) in &by_agent {
        let (repo, n_files) = push_agent_traces_dataset(&client, items, &owner, &base, agent)?;
        eprintln!(
            "agentcap export: pushed {n_files} trace file(s) for {agent} across {} run(s) -> {repo}",
            items.len()
        );
        traces_repos.push(repo);
    }

    let mut repos = vec![captures_repo_id(&owner, &base)];
    repos.extend(traces_repos);
    match hub::collection::ensure_collection(&owner, &base, &repos) {
        Ok(slug) => eprintln!("agentcap export: collection -> https://huggingface.co/collections/{slug}"),
        Err(e) => eprintln!("agentcap export: collection step skipped ({e})"),
    }
    Ok(())
}

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_file()).count())
        .unwrap_or(0)
}

/// Resolve a target to `(capture_dir, agent_from_run_json, run_id)`.
fn resolve_target(workspace: &Path, t: &str) -> Result<(PathBuf, Option<String>, String)> {
    // 1. run-id in the workspace.
    let candidate = workspace.join(t);
    if candidate.join("captures").is_dir() {
        let agent = agent_from_run_json(&candidate);
        return Ok((candidate.join("captures"), agent, t.to_string()));
    }
    // 2. an arbitrary workdir path with a captures/ subdir.
    let p = PathBuf::from(t);
    if p.join("captures").is_dir() {
        let agent = agent_from_run_json(&p);
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or(t).to_string();
        return Ok((p.join("captures"), agent, name));
    }
    // 3. a path that *is* a capture dir.
    if p.is_dir() && !sorted_request_files(&p).is_empty() {
        let parent = p
            .parent()
            .and_then(|d| d.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        return Ok((p, None, parent));
    }
    bail!("can't resolve {t:?} to a capture dir");
}

fn agent_from_run_json(run_dir: &Path) -> Option<String> {
    let rec = read_json(&run_dir.join("run.json")).ok()?;
    rec.get("agent").and_then(Value::as_str).map(str::to_string)
}

fn push_captures_dataset(
    client: &HFClientSync,
    items: &[CapItem],
    owner: &str,
    base: &str,
) -> Result<(String, Vec<usize>)> {
    let repo_id = captures_repo_id(owner, base);
    client
        .create_repository()
        .repo_id(&repo_id)
        .repo_type(RepoTypeDataset)
        .private(true)
        .exist_ok(true)
        .send()
        .with_context(|| format!("creating {repo_id}"))?;
    let repo = client.dataset(owner, format!("{base}-captures"));
    let include_readme = !hub::list_files(&repo)
        .unwrap_or_default()
        .iter()
        .any(|f| f == "README.md");

    let tmp = tempfile::tempdir()?;
    let mut ops: Vec<CommitOperation> = Vec::new();
    if include_readme {
        ops.push(CommitOperation::add_bytes(
            "README.md",
            captures_readme(&repo_id, owner, base).into_bytes(),
        ));
    }
    let mut n_rows = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let mut extra = detect_provider_columns(&item.capture_dir);
        let provider = extra.iter().find(|(k, _)| k == "provider").map(|(_, v)| v.clone());
        extra.push(("run_id".to_string(), item.run_id.clone()));
        let filename = default_filename(item.agent.as_deref(), Some(&item.model), provider.as_deref());
        let local = tmp.path().join(format!("{i}-{filename}"));
        let n = parquet_io::write_parquet(
            &item.capture_dir,
            &local,
            item.agent.as_deref(),
            Some(&item.model),
            &extra,
        )?;
        n_rows.push(n);
        ops.push(CommitOperation::add_file(format!("data/{filename}"), local));
    }

    let msg = format!("agentcap export: add {} parquet(s)", ops.len());
    repo.create_commit()
        .operations(ops)
        .commit_message(msg)
        .revision("main")
        .send()
        .with_context(|| format!("committing to {repo_id}"))?;
    Ok((repo_id, n_rows))
}

fn push_agent_traces_dataset(
    client: &HFClientSync,
    items: &[&TraceItem],
    owner: &str,
    base: &str,
    agent: &str,
) -> Result<(String, usize)> {
    let repo_id = traces_repo_id_for(owner, base, agent);
    let captures_repo = captures_repo_id(owner, base);
    client
        .create_repository()
        .repo_id(&repo_id)
        .repo_type(RepoTypeDataset)
        .private(true)
        .exist_ok(true)
        .send()
        .with_context(|| format!("creating {repo_id}"))?;
    let repo = client.dataset(owner, format!("{base}-{agent}-traces"));
    let include_readme = !hub::list_files(&repo)
        .unwrap_or_default()
        .iter()
        .any(|f| f == "README.md");

    let mut ops: Vec<CommitOperation> = Vec::new();
    if include_readme {
        ops.push(CommitOperation::add_bytes(
            "README.md",
            traces_readme(&repo_id, &captures_repo, owner, base, agent).into_bytes(),
        ));
    }
    let mut n_files = 0;
    for item in items {
        if !item.traces_dir.is_dir() {
            continue;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&item.traces_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for f in files {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("file");
            ops.push(CommitOperation::add_file(
                format!("data/{}/{name}", item.run_id),
                f.clone(),
            ));
            n_files += 1;
        }
    }

    if ops.is_empty() {
        return Ok((repo_id, n_files));
    }
    let msg = format!(
        "agentcap export: add {agent} traces ({n_files} file(s) across {} run(s))",
        items.len()
    );
    repo.create_commit()
        .operations(ops)
        .commit_message(msg)
        .revision("main")
        .send()
        .with_context(|| format!("committing to {repo_id}"))?;
    Ok((repo_id, n_files))
}

/// Scan each run dir, printing a per-run summary; return the total verified-hit
/// count across all runs (the caller aborts if > 0).
fn scan_run_dirs(run_dirs: &[PathBuf], no_verification: bool, rescan: bool) -> Result<usize> {
    let mut total_verified = 0;
    for run_dir in run_dirs {
        let (result, was_cached) = scan::scan_run_dir(run_dir, no_verification, rescan)?;
        total_verified += result.verified.len();
        let cache_tag = if was_cached { " (cached)" } else { "" };
        let name = run_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        eprintln!(
            "  [scan] {name}{cache_tag}: {} chunks / {} bytes; verified={} unverified={}",
            result.chunks_scanned,
            result.bytes_scanned,
            result.verified.len(),
            result.unverified.len()
        );
        for hit in &result.verified {
            eprintln!("    VERIFIED  {}  {}", hit.detector, hit.file);
        }
        if !result.unverified.is_empty() {
            let mut by_det: BTreeMap<&str, usize> = BTreeMap::new();
            for h in &result.unverified {
                *by_det.entry(h.detector.as_str()).or_default() += 1;
            }
            let tail = by_det
                .iter()
                .map(|(d, n)| format!("{d}={n}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("    unverified by detector: {tail}");
        }
    }
    Ok(total_verified)
}

fn captures_readme(repo_id: &str, owner: &str, base: &str) -> String {
    CAPTURES_README
        .replace("{repo_id}", repo_id)
        .replace("{owner}", owner)
        .replace("{base}", base)
        .replace("{collection_title}", base)
}

fn traces_readme(repo_id: &str, captures_repo: &str, owner: &str, base: &str, agent: &str) -> String {
    TRACES_README
        .replace("{repo_id}", repo_id)
        .replace("{captures_repo}", captures_repo)
        .replace("{owner}", owner)
        .replace("{collection_title}", base)
        .replace("{agent}", agent)
}

const CAPTURES_README: &str = r#"---
license: apache-2.0
tags:
- agentcap
- agentcap-captures
---

# {repo_id}

HTTP captures of agent ↔ model interactions — one parquet row per
`/v1/chat/completions` call. Produced by
[agentcap](https://github.com/huggingface/agentcap).

Native session traces for the same runs live in companion datasets
named `{base}-<agent>-traces`. They're all grouped under the
[{collection_title} Collection](https://huggingface.co/{owner})
alongside this dataset. Join on `run_id`.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
```

## Schema

| column | description |
|---|---|
| `run_id` | agentcap run id; matches the per-run folder in the traces dataset |
| `request_id` | UUID minted by the capture proxy |
| `model` | Model id from the captured request body |
| `captured_at` | Epoch seconds when the request was captured |
| `request` | Raw OpenAI request body, JSON-stringified |
| `response` | Raw OpenAI response body, JSON-stringified (or `{"stream": true, "raw": ...}` for SSE) |
| `served_by` | Per-response `X-Served-By` header (HF Router sub-provider routing) |
| `served_build_info` | Per-response `X-Build-Info` header |
| `served_model` | Per-response body-echoed `model` |
| `provider` | Derived from the proxy upstream URL (constant per file) |
| `upstream_url` | Proxy upstream URL at capture time (constant per file) |

`request` and `response` are JSON strings; consumers `json.loads(...)`
them. To recover per-message token ranges, render `request.messages`
through the model's chat template yourself —
`transformers.AutoTokenizer.apply_chat_template`.
"#;

const TRACES_README: &str = r#"---
license: apache-2.0
tags:
- agent-traces
- agentcap
- agentcap-traces
- agentcap-traces-{agent}
source_datasets:
- {captures_repo}
---

# {repo_id}

{agent} coding-agent session traces produced by
[agentcap](https://github.com/huggingface/agentcap) runs. Each run
contributes one folder under `data/<run_id>/`; inside, one file per
session in `{agent}`'s native export format.

The on-the-wire HTTP captures for these same runs live in
[{captures_repo}](https://huggingface.co/datasets/{captures_repo}).
Both belong to the
[{collection_title} Collection](https://huggingface.co/{owner})
— join on `run_id` to align captures with traces.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_collection_base_ok_and_err() {
        assert_eq!(parse_collection_base("acme/kb").unwrap(), ("acme".into(), "kb".into()));
        assert_eq!(
            parse_collection_base("hf://datasets/acme/kb").unwrap(),
            ("acme".into(), "kb".into())
        );
        assert!(parse_collection_base("acme").is_err());
        assert!(parse_collection_base("a/b/c").is_err());
    }

    #[test]
    fn slug_rules() {
        assert_eq!(slug("org/gemma-4-E4B-it"), "gemma-4-E4B-it");
        assert_eq!(slug("a//b"), "b");
        assert_eq!(slug("--weird--"), "weird");
        assert_eq!(slug("///"), "x");
        assert_eq!(slug("hf-router-fireworks-ai"), "hf-router-fireworks-ai");
    }

    #[test]
    fn default_filename_shape() {
        let f = default_filename(Some("hermes"), Some("org/m"), Some("hf-router/fireworks-ai"));
        assert!(f.starts_with("train-hermes-m-hf-router-fireworks-ai-"));
        assert!(f.ends_with(".parquet"));
    }

    #[test]
    fn detect_model_unique_and_mixed() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        std::fs::write(
            cap.join("a.request.json"),
            serde_json::json!({"body": {"model": "m@main"}}).to_string(),
        )
        .unwrap();
        std::fs::write(
            cap.join("b.request.json"),
            serde_json::json!({"body": {"model": "m"}}).to_string(),
        )
        .unwrap();
        // m@main and m collapse to the same bare id.
        assert_eq!(detect_model(&cap).unwrap().as_deref(), Some("m"));

        std::fs::write(
            cap.join("c.request.json"),
            serde_json::json!({"body": {"model": "other"}}).to_string(),
        )
        .unwrap();
        assert!(detect_model(&cap).is_err());
    }
}
