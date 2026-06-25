//! Data layer for inspect: enumerate runs / requests / messages from a
//! workspace or a parquet, and load request/response bodies for the preview and
//! message levels. Ports `_enumerate_workspace_requests`,
//! `_enumerate_parquet_requests`, `_request_messages_for_view`, and the
//! body/response loaders from `__main__.py`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};

use crate::diff::{delta_label, divergence, message_summary, message_text};
use crate::parquet_io::{self, CaptureParquetRow};
use crate::sse::decode_response;

/// A run row in the workspace run picker.
pub struct RunRow {
    pub run_dir: PathBuf,
    pub run_id: String,
    pub agent: String,
    pub model: String,
    pub n_tasks: usize,
    pub n_caps: usize,
}

/// One captured request, shared shape for the workspace and parquet pickers.
#[derive(Clone)]
pub struct ReqRow {
    pub run_id: String,
    pub rid: String,
    pub captured_at: i64,
    pub status: String,
    pub task_id: Option<String>,
    pub turn: Option<i64>,
    pub req_index: usize,
    pub prev_rid: Option<String>,
    pub preview: String,
    pub searchable: String,
}

/// One flattened message row (tool calls expanded), for the message picker.
#[derive(Clone)]
pub struct MsgRecord {
    pub msg_idx: Option<i64>,
    pub role: String,
    pub summary: String,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub finish_reason: Option<String>,
}

/// Where a request picker loads bodies from. The parquet variant holds the full
/// file's rows in memory (parsed once at level creation) so preview/message
/// loads are O(1) lookups, not a re-read of the whole parquet per redraw.
#[derive(Clone)]
pub enum ReqSource {
    Workspace {
        cap_dir: PathBuf,
    },
    Parquet {
        rows: Arc<HashMap<String, CaptureParquetRow>>,
    },
}

impl ReqSource {
    /// The request body for `rid`, or an empty object.
    pub fn load_body(&self, rid: &str) -> Value {
        match self {
            ReqSource::Workspace { cap_dir } => read_capture(cap_dir, rid)
                .and_then(|r| r.get("body").cloned())
                .unwrap_or_else(|| json!({})),
            ReqSource::Parquet { rows } => rows
                .get(rid)
                .map(|r| serde_json::from_str(&r.request).unwrap_or_else(|_| json!({})))
                .unwrap_or_else(|| json!({})),
        }
    }

    /// The response record for `rid` in `decode_response` shape (has
    /// `stream`/`raw`/`body`), or `None`.
    pub fn load_resp(&self, rid: &str) -> Option<Value> {
        match self {
            ReqSource::Workspace { cap_dir } => read_capture_resp(cap_dir, rid),
            ReqSource::Parquet { rows } => rows.get(rid).map(|r| {
                let raw: Value = serde_json::from_str(&r.response).unwrap_or_else(|_| json!({}));
                if raw.get("stream").and_then(Value::as_bool).unwrap_or(false) {
                    raw
                } else {
                    json!({"stream": false, "body": raw})
                }
            }),
        }
    }
}

fn read_capture(cap_dir: &Path, rid: &str) -> Option<Value> {
    let bytes = std::fs::read(cap_dir.join(format!("{rid}.request.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_capture_resp(cap_dir: &Path, rid: &str) -> Option<Value> {
    let bytes = std::fs::read(cap_dir.join(format!("{rid}.response.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Runs (with captures) under a workspace root, for the run picker.
pub fn enumerate_workspace_runs(ws_root: &Path) -> Vec<RunRow> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(ws_root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let mut rows = Vec::new();
    for run_dir in dirs {
        let captures = run_dir.join("captures");
        if !captures.is_dir() {
            continue;
        }
        let n_caps = count_request_files(&captures);
        if n_caps == 0 {
            continue;
        }
        let meta: Value = std::fs::read(run_dir.join("run.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(|| json!({}));
        rows.push(RunRow {
            run_id: run_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string(),
            agent: meta.get("agent").and_then(Value::as_str).unwrap_or("?").to_string(),
            model: meta
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .rsplit('/')
                .next()
                .unwrap_or("?")
                .to_string(),
            n_tasks: meta
                .get("tasks")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0),
            n_caps,
            run_dir,
        });
    }
    rows
}

fn count_request_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".request.json")))
                .count()
        })
        .unwrap_or(0)
}

/// Requests for one run dir, grouped per task with diff/prev_rid chains.
pub fn enumerate_workspace_requests(run_dir: &Path) -> Vec<ReqRow> {
    let captures = run_dir.join("captures");
    let run_id = run_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
    let mut recs: Vec<(String, Value)> = Vec::new();
    for e in std::fs::read_dir(&captures).into_iter().flatten().flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.ends_with(".request.json") {
            continue;
        }
        let rid = name.split('.').next().unwrap_or("").to_string();
        if let Ok(v) = serde_json::from_slice::<Value>(&std::fs::read(e.path()).unwrap_or_default()) {
            recs.push((rid, v));
        }
    }
    // Sort within (task, time) so per-task prev_rid is the immediate predecessor.
    recs.sort_by(|a, b| {
        let ka = (
            task_key_str(&a.1),
            a.1.get("captured_at").and_then(Value::as_i64).unwrap_or(0),
        );
        let kb = (
            task_key_str(&b.1),
            b.1.get("captured_at").and_then(Value::as_i64).unwrap_or(0),
        );
        ka.cmp(&kb)
    });

    let mut rows = Vec::new();
    let mut chains: Chains = Chains::default();
    for (rid, req) in &recs {
        let status = read_capture_resp(&captures, rid)
            .and_then(|r| r.get("status_code").map(value_to_string))
            .unwrap_or_else(|| "?".to_string());
        let row = build_req_row(&run_id, rid, req, status, &mut chains);
        rows.push(row);
    }
    rows.sort_by_key(|a| (a.run_id.clone(), a.captured_at));
    rows
}

/// Read a `-captures` parquet once, returning both the request rows and a
/// `rid → row` map for O(1) body/response lookups in the preview/message levels.
pub fn parquet_request_level(parquet_path: &Path) -> Result<(Vec<ReqRow>, HashMap<String, CaptureParquetRow>)> {
    let mut prows = parquet_io::read_capture_rows(parquet_path)?;
    let map: HashMap<String, CaptureParquetRow> = prows.iter().map(|r| (r.request_id.clone(), r.clone())).collect();
    // Order by (run_id, captured_at), grouping per (run, task) for the diff chain.
    prows.sort_by(|a, b| {
        (a.run_id.clone().unwrap_or_default(), a.captured_at)
            .cmp(&(b.run_id.clone().unwrap_or_default(), b.captured_at))
    });

    let mut rows = Vec::new();
    let mut chains: Chains = Chains::default();
    for pr in &prows {
        // Drop rows whose request_id isn't the proxy's 32-hex format.
        if pr.request_id.len() != 32
            || !pr
                .request_id
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            continue;
        }
        let body: Value = serde_json::from_str(&pr.request).unwrap_or_else(|_| json!({}));
        let status = serde_json::from_str::<Value>(&pr.response)
            .ok()
            .and_then(|r| r.get("status_code").map(value_to_string))
            .unwrap_or_else(|| "?".to_string());
        let run_id = pr.run_id.clone().unwrap_or_else(|| "?".to_string());
        // Synthesize a request record so build_req_row sees task_id/turn/captured_at.
        let req = json!({"body": body, "task_id": pr.task_id, "turn": pr.turn, "captured_at": pr.captured_at});
        rows.push(build_req_row(&run_id, &pr.request_id, &req, status, &mut chains));
    }
    Ok((rows, map))
}

/// Per-(run,task) chain state for the diff / prev_rid / req_index columns.
#[derive(Default)]
struct Chains {
    prev_msgs: std::collections::HashMap<String, Vec<Value>>,
    prev_rid: std::collections::HashMap<String, String>,
    idx: std::collections::HashMap<String, usize>,
}

fn task_key_str(req: &Value) -> String {
    req.get("task_id").and_then(Value::as_str).unwrap_or("").to_string()
}

fn build_req_row(run_id: &str, rid: &str, req: &Value, status: String, chains: &mut Chains) -> ReqRow {
    let messages: Vec<Value> = req
        .get("body")
        .and_then(|b| b.get("messages"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let task_id = req.get("task_id").and_then(Value::as_str).map(str::to_string);
    let turn = req.get("turn").and_then(Value::as_i64);
    let captured_at = req.get("captured_at").and_then(Value::as_i64).unwrap_or(0);
    // Key per (run, task); fall back to (run, rid) when task_id is missing so
    // unrelated orphan captures don't chain together.
    let key = format!("{run_id}\u{0}{}", task_id.clone().unwrap_or_else(|| rid.to_string()));

    let (label, new_msgs): (String, Vec<Value>) = match chains.prev_msgs.get(&key) {
        None => (format!("(init {})", messages.len()), messages.clone()),
        Some(prev) => {
            let i = divergence(prev, &messages);
            let removed = prev.len() - i;
            let new: Vec<Value> = messages[i..].to_vec();
            (format!("({})", delta_label(removed, new.len())), new)
        }
    };
    let summary = new_msgs.last().map(message_summary).unwrap_or_default();
    let preview = format!("{label} {summary}").replace('\n', " ").trim().to_string();
    let searchable = new_msgs
        .iter()
        .map(message_text)
        .collect::<Vec<_>>()
        .join(" ")
        .replace(['\n', '\t'], " ");

    let prev_rid = chains.prev_rid.get(&key).cloned();
    let req_index = *chains.idx.entry(key.clone()).or_insert(0) + 1;
    chains.idx.insert(key.clone(), req_index);
    chains.prev_msgs.insert(key.clone(), messages);
    chains.prev_rid.insert(key, rid.to_string());

    ReqRow {
        run_id: run_id.to_string(),
        rid: rid.to_string(),
        captured_at,
        status,
        task_id,
        turn,
        req_index,
        prev_rid,
        preview,
        searchable,
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Flatten request `messages[]` + the decoded response into one record per
/// picker row. Ports `_request_messages_for_view`.
pub fn request_messages_for_view(body: &Value, resp: Option<&Value>) -> Vec<MsgRecord> {
    let mut records = Vec::new();
    let msgs = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
        let i = i as i64;
        if role == "assistant" {
            for tc in m
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                let (fname, args) = fn_call(&tc);
                records.push(MsgRecord {
                    msg_idx: Some(i),
                    role: format!("assistant→{fname}"),
                    summary: args.clone(),
                    content: args,
                    tool_call_id: tc.get("id").and_then(Value::as_str).map(str::to_string),
                    finish_reason: None,
                });
            }
            let content = message_text(m);
            if !content.is_empty() {
                records.push(MsgRecord {
                    msg_idx: Some(i),
                    role: "assistant".to_string(),
                    summary: content.clone(),
                    content,
                    tool_call_id: None,
                    finish_reason: None,
                });
            }
            continue;
        }
        if role == "tool" {
            let content = message_text(m);
            records.push(MsgRecord {
                msg_idx: Some(i),
                role: "tool".to_string(),
                summary: content.clone(),
                content,
                tool_call_id: m.get("tool_call_id").and_then(Value::as_str).map(str::to_string),
                finish_reason: None,
            });
            continue;
        }
        let content = message_text(m);
        records.push(MsgRecord {
            msg_idx: Some(i),
            role: role.to_string(),
            summary: content.clone(),
            content,
            tool_call_id: None,
            finish_reason: None,
        });
    }
    if let Some(resp) = resp {
        let decoded = decode_response(resp);
        for tc in &decoded.tool_calls {
            let (fname, args) = fn_call(tc);
            records.push(MsgRecord {
                msg_idx: None,
                role: format!("response→{fname}"),
                summary: args.clone(),
                content: args,
                tool_call_id: tc.get("id").and_then(Value::as_str).map(str::to_string),
                finish_reason: None,
            });
        }
        if !decoded.content.is_empty() {
            records.push(MsgRecord {
                msg_idx: None,
                role: "response".to_string(),
                summary: decoded.content.clone(),
                content: decoded.content,
                tool_call_id: None,
                finish_reason: decoded.finish_reason,
            });
        }
    }
    records
}

fn fn_call(tc: &Value) -> (String, String) {
    let func = tc.get("function");
    let name = func
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("?")
        .to_string();
    let args = func
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    (name, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_req(cap: &Path, rid: &str, at: i64, messages: Value) {
        let rec = json!({
            "request_id": rid, "captured_at": at, "task_id": "task_00", "turn": at,
            "upstream_url": "http://x", "body": {"model": "m", "messages": messages}
        });
        std::fs::write(cap.join(format!("{rid}.request.json")), rec.to_string()).unwrap();
        std::fs::write(
            cap.join(format!("{rid}.response.json")),
            json!({"request_id": rid, "status_code": 200, "body": {"choices": []}}).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn workspace_requests_chain_diff_and_index() {
        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("hermes-local-20260101-000000");
        let cap = run.join("captures");
        std::fs::create_dir_all(&cap).unwrap();
        let a = "a".repeat(32);
        let b = "b".repeat(32);
        write_req(&cap, &a, 1, json!([{"role": "user", "content": "hi"}]));
        write_req(
            &cap,
            &b,
            2,
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]),
        );

        let rows = enumerate_workspace_requests(&run);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rid, a);
        assert!(rows[0].preview.starts_with("(init 1)"));
        assert_eq!(rows[0].req_index, 1);
        assert_eq!(rows[0].prev_rid, None);
        assert_eq!(rows[1].rid, b);
        assert!(rows[1].preview.starts_with("(+1) assistant: ok"));
        assert_eq!(rows[1].req_index, 2);
        assert_eq!(rows[1].prev_rid.as_deref(), Some(a.as_str()));
        assert_eq!(rows[1].status, "200");
    }

    #[test]
    fn parquet_requests_round_trip_via_export() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        let a = "a".repeat(32);
        write_req(
            &cap,
            &a,
            1,
            json!([{"role": "user", "content": "find the hf-cli skill"}]),
        );
        let parquet = tmp.path().join("o.parquet");
        let extra = vec![("run_id".to_string(), "run-x".to_string())];
        parquet_io::write_parquet(&cap, &parquet, Some("hermes"), Some("m"), &extra).unwrap();

        let (rows, _map) = parquet_request_level(&parquet).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rid, a);
        assert_eq!(rows[0].run_id, "run-x");
        assert_eq!(rows[0].task_id.as_deref(), Some("task_00"));
        // Deep content is searchable even though it's not in the visible preview.
        assert!(rows[0].searchable.contains("hf-cli"));
    }

    #[test]
    fn messages_for_view_flattens_tool_calls_and_response() {
        let body = json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]});
        let resp =
            json!({"stream": false, "body": {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}});
        let recs = request_messages_for_view(&body, Some(&resp));
        let roles: Vec<&str> = recs.iter().map(|r| r.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant→read", "tool", "response"]);
        assert_eq!(recs.last().unwrap().content, "done");
        assert_eq!(recs.last().unwrap().finish_reason.as_deref(), Some("stop"));
        assert_eq!(recs[1].tool_call_id.as_deref(), Some("c1"));
    }
}
