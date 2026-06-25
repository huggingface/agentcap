//! `ls`: list runs under a local workspace.
//!
//! Unlike `export`, `ls` does NOT consult `$AGENTCAP_WORKSPACE` — what you point
//! it at is what you get. Accepts either the parent dir or the `.agentcap/` dir.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::Value;

const WORKSPACE_DIR: &str = ".agentcap";

struct RunRow {
    run_id: String,
    agent: String,
    model: String,
    provider: String,
    upstream: String,
    n_tasks: usize,
    n_ok: usize,
    n_caps: usize,
}

pub fn run(workspace: Option<String>, long: bool) -> Result<()> {
    let root = match workspace {
        None => std::env::current_dir()?.join(WORKSPACE_DIR),
        Some(w) => {
            let p = std::path::absolute(Path::new(&w)).unwrap_or_else(|_| PathBuf::from(&w));
            if p.file_name().and_then(|n| n.to_str()) == Some(WORKSPACE_DIR) {
                p
            } else {
                p.join(WORKSPACE_DIR)
            }
        }
    };
    if !root.is_dir() {
        eprintln!(
            "no workspace at {root:?}. Run `agentcap run` first, or pass a directory that contains a `.agentcap/` subdir."
        );
        return Ok(());
    }

    let mut run_dirs: Vec<PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    run_dirs.sort();

    let mut rows = Vec::new();
    for run_dir in run_dirs {
        let meta_path = run_dir.join("run.json");
        if !run_dir.is_dir() || !meta_path.is_file() {
            continue;
        }
        let Ok(meta) = serde_json::from_slice::<Value>(&std::fs::read(&meta_path).unwrap_or_default()) else {
            continue;
        };
        let captures = run_dir.join("captures");
        let n_caps = std::fs::read_dir(&captures)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".request.json")))
                    .count()
            })
            .unwrap_or(0);
        let tasks = meta.get("tasks").and_then(Value::as_array).cloned().unwrap_or_default();
        let turns = meta.get("turns_per_task").and_then(Value::as_i64).unwrap_or(1);
        let n_ok = tasks
            .iter()
            .filter(|t| t.get("completed_turns").and_then(Value::as_i64) == Some(turns))
            .count();
        rows.push(RunRow {
            run_id: run_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string(),
            agent: str_or(&meta, "agent"),
            model: str_or(&meta, "model").rsplit('/').next().unwrap_or("?").to_string(),
            provider: str_or(&meta, "provider"),
            upstream: str_or(&meta, "upstream"),
            n_tasks: tasks.len(),
            n_ok,
            n_caps,
        });
    }

    if rows.is_empty() {
        eprintln!("no runs in {}.", root.display());
        return Ok(());
    }

    if long {
        let cols = ["RUN_ID", "AGENT", "MODEL", "PROVIDER", "TASKS", "CAPTURES", "UPSTREAM"];
        let widths = [
            width(cols[0], rows.iter().map(|r| r.run_id.len())),
            width(cols[1], rows.iter().map(|r| r.agent.len())),
            width(cols[2], rows.iter().map(|r| r.model.len())),
            width(cols[3], rows.iter().map(|r| r.provider.len())),
            cols[4].len(),
            cols[5].len(),
            width(cols[6], rows.iter().map(|r| r.upstream.len())),
        ];
        println!("{}", fmt(&cols, &widths));
        for r in &rows {
            let cells = [
                r.run_id.as_str(),
                r.agent.as_str(),
                r.model.as_str(),
                r.provider.as_str(),
                &format!("{}/{}", r.n_ok, r.n_tasks),
                &r.n_caps.to_string(),
                r.upstream.as_str(),
            ];
            println!("{}", fmt(&cells, &widths));
        }
    } else {
        let cols = ["RUN_ID", "AGENT", "MODEL", "TASKS", "CAPTURES"];
        let widths = [
            width(cols[0], rows.iter().map(|r| r.run_id.len())),
            width(cols[1], rows.iter().map(|r| r.agent.len())),
            width(cols[2], rows.iter().map(|r| r.model.len())),
            cols[3].len(),
            cols[4].len(),
        ];
        println!("{}", fmt(&cols, &widths));
        for r in &rows {
            let cells = [
                r.run_id.as_str(),
                r.agent.as_str(),
                r.model.as_str(),
                &format!("{}/{}", r.n_ok, r.n_tasks),
                &r.n_caps.to_string(),
            ];
            println!("{}", fmt(&cells, &widths));
        }
    }
    Ok(())
}

fn str_or(meta: &Value, key: &str) -> String {
    meta.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("?")
        .to_string()
}

fn width(header: &str, lens: impl Iterator<Item = usize>) -> usize {
    header.len().max(lens.max().unwrap_or(0))
}

fn fmt(cells: &[&str], widths: &[usize]) -> String {
    cells
        .iter()
        .zip(widths)
        .map(|(c, w)| format!("{c:<width$}", width = w))
        .collect::<Vec<_>>()
        .join("  ")
}
