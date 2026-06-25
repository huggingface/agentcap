//! Live end-to-end tests: drive the real `agentcap run` binary through a real
//! OpenAI-compatible server for each agent, asserting the wire path (the agent
//! reaches the model through the proxy and the turn completes) — not task
//! quality. Ports `test_cli_live.py` + `test_drivers_live.py`.
//!
//! `#[ignore]` by default so `cargo test` stays hermetic. The `Test - Live`
//! workflow provisions a llama.cpp server + builds the per-agent images, then
//! runs `cargo test --test live -- --ignored`. Run locally with a server up:
//!   AGENTCAP_TEST_LLM_URL=http://127.0.0.1:8000 cargo test --test live -- --ignored
//! Each test skips (passes) if no server is reachable.

use std::process::Command;
use std::time::Duration;

use serde_json::{json, Value};

/// Resolve a reachable llama upstream: `$AGENTCAP_TEST_LLM_URL`, else a server
/// already on :8000/:8080. `None` → no server, skip.
fn upstream() -> Option<String> {
    if let Ok(u) = std::env::var("AGENTCAP_TEST_LLM_URL") {
        if !u.trim().is_empty() {
            return Some(u.trim().trim_end_matches('/').to_string());
        }
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .ok()?;
    for port in [8000, 8080] {
        let url = format!("http://127.0.0.1:{port}");
        if client
            .get(format!("{url}/v1/models"))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return Some(url);
        }
    }
    None
}

/// Last `n` chars of `s`, for failure dumps.
fn tail(s: &str, n: usize) -> String {
    let start = s.char_indices().rev().take(n).last().map(|(i, _)| i).unwrap_or(0);
    s[start..].to_string()
}

/// Dump everything useful when a turn doesn't complete: run.json, the agentcap
/// binary's stderr (orchestrator `[turn_done] rc=…` / `[task_aborted] reason=…`),
/// and each per-turn session log (the agent's own stdout/stderr).
fn diagnostics(run_dir: &std::path::Path, summary: &Value, bin_stderr: &[u8]) -> String {
    let mut out = format!("--- run.json ---\n{summary:#}\n");
    out.push_str(&format!(
        "--- agentcap stderr (tail) ---\n{}\n",
        tail(&String::from_utf8_lossy(bin_stderr), 4000)
    ));
    if let Ok(rd) = std::fs::read_dir(run_dir.join("sessions")) {
        for e in rd.flatten() {
            let p = e.path();
            let body = std::fs::read_to_string(&p).unwrap_or_default();
            out.push_str(&format!(
                "--- sessions/{} (tail) ---\n{}\n",
                p.file_name().unwrap().to_string_lossy(),
                tail(&body, 2000)
            ));
        }
    }
    out
}

/// `agentcap run --agent <agent>` against the live server; assert the run dir,
/// run.json shape, captures, and (for pi) the streamed JSONL trace.
fn run_agent(agent: &str, expect_jsonl_traces: bool) {
    let Some(upstream) = upstream() else {
        eprintln!("skip live[{agent}]: no llama server (set AGENTCAP_TEST_LLM_URL or run one on :8000/:8080)");
        return;
    };
    let model = std::env::var("AGENTCAP_TEST_MODEL").unwrap_or_else(|_| "Qwen3-1.7B".to_string());

    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir(&ws).unwrap();
    let tasks = tmp.path().join("tasks.txt");
    std::fs::write(&tasks, "Say hello in one short sentence, then stop.\n").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_agentcap"))
        .args([
            "run",
            "--agent",
            agent,
            "--model",
            &model,
            "--upstream",
            &upstream,
            "--tasks",
            tasks.to_str().unwrap(),
            "--turns",
            "1",
            "--timeout",
            "600",
        ])
        .env("AGENTCAP_WORKSPACE", &ws)
        .output()
        .expect("spawn agentcap");
    assert!(
        out.status.success(),
        "agentcap run --agent {agent} failed (exit {:?})\n--- stderr ---\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    // Exactly one run dir for this agent.
    let run_dir = std::fs::read_dir(ws.join(".agentcap"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("{agent}-")))
        })
        .expect("a run dir under .agentcap");

    let summary: Value = serde_json::from_slice(&std::fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(summary["agent"], json!(agent));
    assert_eq!(summary["model"], json!(model));
    assert_eq!(summary["upstream"], json!(upstream));
    assert_eq!(summary["turns_per_task"], json!(1));
    let task = &summary["tasks"][0];
    assert_eq!(
        task["completed_turns"],
        json!(1),
        "wire path: agent didn't complete the turn\n{}",
        diagnostics(&run_dir, &summary, &out.stderr)
    );
    assert!(
        task["session_id"].as_str().is_some_and(|s| !s.is_empty()),
        "{agent} should mint a session id; run.json: {summary}"
    );

    // Captures landed via the in-process proxy.
    let n_caps = std::fs::read_dir(run_dir.join("captures"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".request.json")))
        .count();
    assert!(n_caps > 0, "proxy should have captured at least one request");

    if expect_jsonl_traces {
        let has_jsonl = std::fs::read_dir(run_dir.join("traces"))
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "jsonl"));
        assert!(has_jsonl, "{agent} should have streamed at least one .jsonl trace");
    }
}

#[test]
#[ignore = "live: needs a model server + podman"]
fn live_pi() {
    // pi streams native session JSONL through the in-container symlink.
    run_agent("pi", true);
}

#[test]
#[ignore = "live: needs a model server + podman"]
fn live_goose() {
    run_agent("goose", false);
}

// hermes and opencode are intentionally omitted — neither runs via `agentcap run`
// on the tiny CI model:
//   - hermes: its base system prompt (~3.9k tokens) exceeds the budget on
//     Qwen3-1.7B, so it bails before any model call. hermes stdout parsing is
//     covered by unit tests.
//   - opencode: 1.15.x doesn't pick up the baked `agent.minimal` from the image.
// pi (symlink/JSONL traces) + goose (dump-traces/SQLite) cover the full stack
// across both trace-surfacing mechanisms.
