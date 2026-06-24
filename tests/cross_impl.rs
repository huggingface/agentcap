//! Opt-in cross-implementation check: write a parquet with the Rust exporter to
//! `$AGENTCAP_PARQUET_OUT`, so a Python/pyarrow reader can confirm the schema,
//! KV metadata, and row JSON load cleanly. Ignored by default (needs the env
//! var + leaves the file in place):
//!
//!   AGENTCAP_PARQUET_OUT=/tmp/x.parquet cargo test --test cross_impl -- --ignored

use std::path::Path;

use serde_json::json;

#[test]
#[ignore]
fn write_parquet_for_pyarrow() {
    let out = std::env::var("AGENTCAP_PARQUET_OUT").expect("set AGENTCAP_PARQUET_OUT");
    let tmp = tempfile::tempdir().unwrap();
    let cap = tmp.path().join("captures");
    std::fs::create_dir(&cap).unwrap();

    // A non-stream request/response pair…
    let r1 = "1".repeat(32);
    std::fs::write(
        cap.join(format!("{r1}.request.json")),
        json!({"request_id": r1, "captured_at": 1718039521, "task_id": "task_00", "turn": 1,
               "upstream_url": "https://router.huggingface.co",
               "body": {"model": "google/gemma", "messages": [{"role": "user", "content": "héllo ☃"}]}})
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        cap.join(format!("{r1}.response.json")),
        json!({"request_id": r1, "status_code": 200, "stream": false,
               "body": {"choices": [{"message": {"content": "hi"}}]},
               "upstream_fingerprint": {"x_served_by": "router-1", "served_model": "google/gemma"}})
        .to_string(),
    )
    .unwrap();

    // …and a streamed one, to exercise the SSE-wrapped response column.
    let r2 = "2".repeat(32);
    std::fs::write(
        cap.join(format!("{r2}.request.json")),
        json!({"request_id": r2, "captured_at": 1718039525, "task_id": "task_00", "turn": 2,
               "upstream_url": "https://router.huggingface.co",
               "body": {"model": "google/gemma", "messages": [{"role": "user", "content": "again"}]}})
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        cap.join(format!("{r2}.response.json")),
        json!({"request_id": r2, "status_code": 200, "stream": true,
               "raw": "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\ndata: [DONE]\n"})
        .to_string(),
    )
    .unwrap();

    let extra = vec![
        ("provider".to_string(), "hf-router".to_string()),
        ("upstream_url".to_string(), "https://router.huggingface.co".to_string()),
        ("run_id".to_string(), "hermes-hf-router-20260101-000000".to_string()),
    ];
    let n = agentcap::parquet_io::write_parquet(&cap, Path::new(&out), Some("hermes"), Some("google/gemma"), &extra)
        .unwrap();
    assert_eq!(n, 2);
}
