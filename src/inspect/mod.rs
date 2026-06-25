//! `inspect`: classify the TARGET and launch the picker (or dump a request body
//! for a bare hex rid).

mod app;
mod render;
mod sources;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{bail, Result};

use crate::captures;
use crate::hub;

const WORKSPACE_DIR: &str = ".agentcap";
const FOOTER_WORKERS: usize = 4;

enum Target {
    Workspace(PathBuf),
    WorkspaceRun(String),
    Rid(String),
    Parquet(PathBuf),
    Hf(String),
}

/// Entry point for the `inspect` subcommand.
pub fn run(target: Option<String>, rid_flag: bool) -> Result<()> {
    match classify_target(target.as_deref())? {
        Target::Rid(rid) => dump_rid(&rid, rid_flag),
        Target::Workspace(ws) => {
            if !ws.is_dir() {
                eprintln!("no workspace at {ws:?}. Run `agentcap run` first.");
                return Ok(());
            }
            let rows = sources::enumerate_workspace_runs(&ws);
            if rows.is_empty() {
                eprintln!("no runs with captures in {}.", ws.display());
                return Ok(());
            }
            app::run(vec![app::level_run(rows)])
        }
        Target::WorkspaceRun(run_id) => {
            let run_dir = std::env::current_dir()?.join(WORKSPACE_DIR).join(&run_id);
            app::run(vec![app::level_request_workspace(&run_dir)])
        }
        Target::Parquet(path) => app::run(vec![app::level_request_parquet(path)?]),
        Target::Hf(repo_id) => run_hf(&repo_id),
    }
}

fn run_hf(repo_id: &str) -> Result<()> {
    let repo = hub::dataset(repo_id)?;
    let files = hub::list_parquet_files(&repo)?;
    if files.is_empty() {
        eprintln!("no parquet files under data/ in {repo_id}.");
        return Ok(());
    }
    let rows: Vec<app::HfRow> = files
        .iter()
        .map(|(path, _size)| app::HfRow {
            path: path.clone(),
            agent: None,
            model: None,
            num_rows: None,
            tasks: None,
        })
        .collect();

    // Background footer reads: round-robin the files across a few workers, each
    // with its own client (own runtime), streaming results to the picker.
    let (tx, rx) = mpsc::channel();
    let n_workers = FOOTER_WORKERS.min(files.len());
    for w in 0..n_workers {
        let assignments: Vec<(usize, String, u64)> = files
            .iter()
            .enumerate()
            .skip(w)
            .step_by(n_workers)
            .map(|(i, (p, s))| (i, p.clone(), *s))
            .collect();
        let tx = tx.clone();
        let repo_id = repo_id.to_string();
        std::thread::spawn(move || {
            let Ok(repo) = hub::dataset(&repo_id) else { return };
            for (idx, path, size) in assignments {
                if let Ok(meta) = hub::footer::fetch_parquet_meta(&repo, &path, size) {
                    if tx.send((idx, meta)).is_err() {
                        return; // picker closed
                    }
                }
            }
        });
    }
    drop(tx); // so rx disconnects once all workers finish

    app::run(vec![app::level_hf(repo, rows, rx)])
}

fn dump_rid(rid: &str, rid_flag: bool) -> Result<()> {
    let cwd_ws = std::env::current_dir()?.join(WORKSPACE_DIR);
    let Some((cap_dir, full_rid)) = captures::resolve_workspace_rid(&cwd_ws, rid)? else {
        bail!("request id {rid:?} not found in {}", cwd_ws.display());
    };
    if rid_flag {
        println!("{full_rid}");
        return Ok(());
    }
    let body = captures::load_request(cap_dir.to_string_lossy().as_ref(), &full_rid)?;
    println!("{}", serde_json::to_string_pretty(&body)?);
    Ok(())
}

/// Classify the TARGET positional.
fn classify_target(target: Option<&str>) -> Result<Target> {
    let Some(target) = target else {
        return Ok(Target::Workspace(std::env::current_dir()?.join(WORKSPACE_DIR)));
    };

    if target.ends_with(".parquet") {
        let p = PathBuf::from(target);
        if !p.is_file() {
            bail!("parquet not found: {target}");
        }
        return Ok(Target::Parquet(p));
    }

    if let Some(rest) = target.strip_prefix("hf://") {
        let s = rest.strip_prefix("datasets/").unwrap_or(rest).trim_matches('/');
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() == 2 && parts.iter().all(|p| !p.is_empty()) {
            return Ok(Target::Hf(s.to_string()));
        }
        bail!("invalid hf URI: {target:?}");
    }

    // Local directory → workspace (accept either parent or the .agentcap dir).
    let tp = Path::new(target);
    if tp.is_dir() {
        let p = std::path::absolute(tp).unwrap_or_else(|_| tp.to_path_buf());
        let ws = if p.file_name().and_then(|n| n.to_str()) == Some(WORKSPACE_DIR) {
            p
        } else {
            p.join(WORKSPACE_DIR)
        };
        return Ok(Target::Workspace(ws));
    }

    // Run-id under cwd's .agentcap (run dirs carry a timestamp → a dash).
    let cwd_ws = std::env::current_dir()?.join(WORKSPACE_DIR);
    if target.contains('-') && cwd_ws.join(target).join("run.json").is_file() {
        return Ok(Target::WorkspaceRun(target.to_string()));
    }

    // <owner>/<name> HF shorthand — only when it's not a local path.
    let parts: Vec<&str> = target.split('/').collect();
    if parts.len() == 2 && parts.iter().all(|p| !p.is_empty()) {
        return Ok(Target::Hf(target.to_string()));
    }

    // All-hex (≥6) → request id, looked up in cwd's workspace.
    if target.len() >= 6 && target.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Ok(Target::Rid(target.to_string()));
    }

    bail!(
        "can't classify TARGET {target:?}: expected a directory, a .parquet file, an hf:// URI, \
         an <owner>/<name> shorthand, a run-id (under ./.agentcap/), or a request-id (hex)."
    );
}
