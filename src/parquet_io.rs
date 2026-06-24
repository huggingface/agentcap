//! Capture dir → parquet (export) and parquet → request bodies (read).
//!
//! Ports `export.py`'s `export_local` / `_iter_pairs` / `_row` and the parquet
//! readers in `captures.py`. The `request` / `response` columns are
//! JSON-stringified bodies (Arrow can't infer a schema over heterogeneous
//! tool-schema fields); `agent` / `model` / `tasks` are stamped into the
//! parquet key-value metadata so the inspect picker can label files cheaply.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::format::KeyValue;
use serde_json::{json, Value};

const BATCH_SIZE: usize = 32;
const PROMPT_CAP: usize = 200;

/// One exported parquet row, with bodies already JSON-stringified.
struct Row {
    request_id: String,
    model: String,
    captured_at: i64,
    task_id: Option<String>,
    turn: Option<i64>,
    request: String,
    response: String,
    served_by: Option<String>,
    served_build_info: Option<String>,
    served_model: Option<String>,
}

#[derive(Default)]
struct TaskAcc {
    turns: i64,
    prompt: Option<String>,
}

/// Stream a capture dir into a single parquet at `output`. Returns the row
/// count. `agent` / `model` land in the schema metadata and the parquet KV;
/// `tasks` (one entry per `task_id`, with max turn + first user prompt) is
/// stamped into the parquet KV after the full pass. `extra` columns (e.g.
/// `provider`, `upstream_url`, `run_id`) are appended in the given order with a
/// constant value per file.
pub fn write_parquet(
    capture_dir: &Path,
    output: &Path,
    agent: Option<&str>,
    model: Option<&str>,
    extra: &[(String, String)],
) -> Result<usize> {
    let (rows, tasks) = read_capture_dir(capture_dir)?;
    if rows.is_empty() {
        bail!("no captured requests in {}", capture_dir.display());
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }

    let schema = build_schema(extra, agent, model);
    let file = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    for chunk in rows.chunks(BATCH_SIZE) {
        writer.write(&build_batch(&schema, chunk, extra)?)?;
    }

    // The streaming writer freezes the arrow schema at open, so `tasks` (known
    // only after the full pass) goes into the parquet KV. agent/model too, so a
    // reader that only reads parquet KV (our footer picker) gets all three.
    let tasks_list: Vec<Value> = tasks
        .iter()
        .map(|(id, acc)| json!({"id": id, "turns": acc.turns, "prompt": acc.prompt}))
        .collect();
    writer.append_key_value_metadata(kv("tasks", &Value::Array(tasks_list).to_string()));
    if let Some(a) = agent {
        writer.append_key_value_metadata(kv("agent", a));
    }
    if let Some(m) = model {
        writer.append_key_value_metadata(kv("model", m));
    }
    writer.close()?;
    Ok(rows.len())
}

fn kv(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: Some(value.to_string()),
    }
}

fn build_schema(extra: &[(String, String)], agent: Option<&str>, model: Option<&str>) -> SchemaRef {
    let mut fields = vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("model", DataType::Utf8, false),
        Field::new("captured_at", DataType::Int64, false),
        Field::new("task_id", DataType::Utf8, true),
        Field::new("turn", DataType::Int64, true),
        Field::new("request", DataType::Utf8, false),
        Field::new("response", DataType::Utf8, false),
        Field::new("served_by", DataType::Utf8, true),
        Field::new("served_build_info", DataType::Utf8, true),
        Field::new("served_model", DataType::Utf8, true),
    ];
    for (k, _) in extra {
        fields.push(Field::new(k, DataType::Utf8, false));
    }
    let mut md = HashMap::new();
    if let Some(a) = agent {
        md.insert("agent".to_string(), a.to_string());
    }
    if let Some(m) = model {
        md.insert("model".to_string(), m.to_string());
    }
    Arc::new(Schema::new_with_metadata(fields, md))
}

fn build_batch(schema: &SchemaRef, rows: &[Row], extra: &[(String, String)]) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.request_id.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.model.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.captured_at).collect::<Vec<_>>())),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.task_id.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.turn).collect::<Vec<_>>())),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.request.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.response.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.served_by.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.served_build_info.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.served_model.clone()).collect::<Vec<_>>(),
        )),
    ];
    for (_, v) in extra {
        columns.push(Arc::new(StringArray::from(vec![v.clone(); rows.len()])));
    }
    RecordBatch::try_new(schema.clone(), columns).context("building record batch")
}

/// Read every `<rid>.request.json` (paired with its `.response.json`) in
/// filename order, producing rows + the accumulated per-task metadata.
fn read_capture_dir(capture_dir: &Path) -> Result<(Vec<Row>, BTreeMap<String, TaskAcc>)> {
    let mut req_files: Vec<std::path::PathBuf> = std::fs::read_dir(capture_dir)
        .with_context(|| format!("reading {}", capture_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".request.json"))
        })
        .collect();
    req_files.sort();

    let mut rows = Vec::with_capacity(req_files.len());
    let mut tasks: BTreeMap<String, TaskAcc> = BTreeMap::new();
    for req_path in &req_files {
        let rec: Value = serde_json::from_slice(&std::fs::read(req_path)?)
            .with_context(|| format!("parsing {}", req_path.display()))?;
        let rid = rec
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                req_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.split('.').next())
                    .unwrap_or("")
                    .to_string()
            });
        let captured_at = rec.get("captured_at").and_then(Value::as_i64).unwrap_or(0);
        let body = rec.get("body").cloned().unwrap_or_else(|| json!({}));
        let task_id = rec.get("task_id").and_then(Value::as_str).map(str::to_string);
        let turn = rec.get("turn").and_then(Value::as_i64);

        let resp_path = capture_dir.join(format!("{rid}.response.json"));
        let mut response = json!({});
        let mut fp = json!({});
        if resp_path.is_file() {
            let resp_rec: Value = serde_json::from_slice(&std::fs::read(&resp_path)?)
                .with_context(|| format!("parsing {}", resp_path.display()))?;
            fp = resp_rec
                .get("upstream_fingerprint")
                .cloned()
                .unwrap_or_else(|| json!({}));
            response = if resp_rec.get("stream").and_then(Value::as_bool).unwrap_or(false) {
                json!({"stream": true, "raw": resp_rec.get("raw").and_then(Value::as_str).unwrap_or("")})
            } else {
                resp_rec.get("body").cloned().unwrap_or_else(|| json!({}))
            };
        }

        accumulate_task(&mut tasks, task_id.as_deref(), turn, &body);

        rows.push(Row {
            request_id: rid,
            model: body.get("model").and_then(Value::as_str).unwrap_or("").to_string(),
            captured_at,
            task_id,
            turn,
            request: serde_json::to_string(&body)?,
            response: serde_json::to_string(&response)?,
            served_by: fp.get("x_served_by").and_then(Value::as_str).map(str::to_string),
            served_build_info: fp.get("build_info").and_then(Value::as_str).map(str::to_string),
            served_model: fp.get("served_model").and_then(Value::as_str).map(str::to_string),
        });
    }
    Ok((rows, tasks))
}

fn accumulate_task(tasks: &mut BTreeMap<String, TaskAcc>, task_id: Option<&str>, turn: Option<i64>, body: &Value) {
    let Some(tid) = task_id else { return };
    let acc = tasks.entry(tid.to_string()).or_default();
    if let Some(t) = turn {
        if t > acc.turns {
            acc.turns = t;
        }
    }
    if acc.prompt.is_none() {
        if let Some(msgs) = body.get("messages").and_then(Value::as_array) {
            for m in msgs {
                if m.get("role").and_then(Value::as_str) == Some("user") {
                    let content = match m.get("content") {
                        Some(Value::Array(parts)) => parts
                            .iter()
                            .map(|p| p.get("text").and_then(Value::as_str).unwrap_or(""))
                            .collect::<Vec<_>>()
                            .join(" "),
                        Some(Value::String(s)) => s.clone(),
                        _ => String::new(),
                    };
                    let prompt: String = content.replace('\n', " ").trim().chars().take(PROMPT_CAP).collect();
                    acc.prompt = Some(prompt);
                    break;
                }
            }
        }
    }
}

/// Return `{request_id: body}` for the wanted ids found in a local parquet,
/// parsing the `request` column JSON. Mirrors `captures._load_from_parquet`.
pub fn read_request_bodies(parquet_path: &Path, wanted: &HashSet<String>) -> Result<HashMap<String, Value>> {
    let file = File::open(parquet_path).with_context(|| format!("opening {}", parquet_path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = HashMap::new();
    for batch in reader {
        let batch = batch?;
        let rids = str_col(&batch, "request_id")?;
        let reqs = str_col(&batch, "request")?;
        for i in 0..batch.num_rows() {
            if !rids.is_valid(i) {
                continue;
            }
            let rid = rids.value(i);
            if wanted.contains(rid) && reqs.is_valid(i) {
                if let Ok(v) = serde_json::from_str::<Value>(reqs.value(i)) {
                    out.insert(rid.to_string(), v);
                }
            }
        }
    }
    Ok(out)
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .with_context(|| format!("parquet missing/typed-wrong `{name}` column"))
}

/// One row of a `-captures` parquet, with `request`/`response` left as JSON
/// strings. `run_id`/`task_id`/`turn` are absent on pre-schema-upgrade parquets.
#[derive(Debug, Clone)]
pub struct CaptureParquetRow {
    pub request_id: String,
    pub captured_at: i64,
    pub request: String,
    pub response: String,
    pub run_id: Option<String>,
    pub task_id: Option<String>,
    pub turn: Option<i64>,
}

/// Read every row of a `-captures` parquet (used by inspect's parquet enumeration).
pub fn read_capture_rows(parquet_path: &Path) -> Result<Vec<CaptureParquetRow>> {
    let file = File::open(parquet_path).with_context(|| format!("opening {}", parquet_path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let rids = str_col(&batch, "request_id")?;
        let reqs = str_col(&batch, "request")?;
        let resps = str_col(&batch, "response")?;
        let times = batch
            .column_by_name("captured_at")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let runs = batch
            .column_by_name("run_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let tids = batch
            .column_by_name("task_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let turns = batch
            .column_by_name("turn")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        for i in 0..batch.num_rows() {
            if !rids.is_valid(i) {
                continue;
            }
            out.push(CaptureParquetRow {
                request_id: rids.value(i).to_string(),
                captured_at: times.filter(|c| c.is_valid(i)).map(|c| c.value(i)).unwrap_or(0),
                request: if reqs.is_valid(i) {
                    reqs.value(i).to_string()
                } else {
                    "{}".to_string()
                },
                response: if resps.is_valid(i) {
                    resps.value(i).to_string()
                } else {
                    "{}".to_string()
                },
                run_id: runs.filter(|c| c.is_valid(i)).map(|c| c.value(i).to_string()),
                task_id: tids.filter(|c| c.is_valid(i)).map(|c| c.value(i).to_string()),
                turn: turns.filter(|c| c.is_valid(i)).map(|c| c.value(i)),
            });
        }
    }
    Ok(out)
}

/// Read the parquet file's key-value metadata (`agent`, `model`, `tasks`, …).
pub fn read_kv_metadata(parquet_path: &Path) -> Result<HashMap<String, String>> {
    let file = File::open(parquet_path).with_context(|| format!("opening {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut out = HashMap::new();
    if let Some(kvs) = builder.metadata().file_metadata().key_value_metadata() {
        for kv in kvs {
            if let Some(v) = &kv.value {
                out.insert(kv.key.clone(), v.clone());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_capture(dir: &Path, rid: &str, body: &Value, task_id: Option<&str>, turn: Option<i64>) {
        let mut rec = json!({"request_id": rid, "captured_at": 1, "upstream_url": "http://x", "body": body});
        if let Some(t) = task_id {
            rec["task_id"] = json!(t);
        }
        if let Some(t) = turn {
            rec["turn"] = json!(t);
        }
        std::fs::write(dir.join(format!("{rid}.request.json")), rec.to_string()).unwrap();
    }

    #[test]
    fn roundtrip_request_body_and_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        let body = json!({"model": "m", "messages": [{"role": "user", "content": "hello"}], "tools": []});
        write_capture(&cap, "rid", &body, Some("task_00"), Some(3));
        std::fs::write(
            cap.join("rid.response.json"),
            json!({"request_id": "rid", "captured_at_resp": 2, "status_code": 200, "body": {"choices": []}})
                .to_string(),
        )
        .unwrap();

        let out = tmp.path().join("out.parquet");
        let n = write_parquet(&cap, &out, Some("hermes"), Some("m"), &[]).unwrap();
        assert_eq!(n, 1);

        let wanted: HashSet<String> = ["rid".to_string()].into_iter().collect();
        let bodies = read_request_bodies(&out, &wanted).unwrap();
        assert_eq!(bodies["rid"], body);

        let kv = read_kv_metadata(&out).unwrap();
        assert_eq!(kv.get("agent").map(String::as_str), Some("hermes"));
        assert_eq!(kv.get("model").map(String::as_str), Some("m"));
        let tasks: Value = serde_json::from_str(kv.get("tasks").unwrap()).unwrap();
        assert_eq!(tasks, json!([{"id": "task_00", "turns": 3, "prompt": "hello"}]));
    }

    #[test]
    fn empty_capture_dir_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        let err = write_parquet(&cap, &tmp.path().join("o.parquet"), None, None, &[]).unwrap_err();
        assert!(err.to_string().contains("no captured requests"));
    }

    #[test]
    fn streamed_response_wrapped_as_raw() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = tmp.path().join("captures");
        std::fs::create_dir(&cap).unwrap();
        write_capture(&cap, "r2", &json!({"model": "m", "messages": []}), None, None);
        std::fs::write(
            cap.join("r2.response.json"),
            json!({"request_id": "r2", "stream": true, "raw": "data: x\n"}).to_string(),
        )
        .unwrap();
        let out = tmp.path().join("o.parquet");
        write_parquet(&cap, &out, None, None, &[]).unwrap();
        // The response column should carry the SSE wrapper, not an empty object.
        let file = File::open(&out).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let resp = str_col(&batch, "response").unwrap().value(0);
        let v: Value = serde_json::from_str(resp).unwrap();
        assert_eq!(v, json!({"stream": true, "raw": "data: x\n"}));
    }
}
