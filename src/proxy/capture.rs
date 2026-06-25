//! Capture record shapes + persistence. Writes `<rid>.request.json` /
//! `<rid>.response.json` in the exact shape the data/UI half reads (see
//! `parquet_io` / `captures`).

use std::io;
use std::path::Path;

use reqwest::header::HeaderMap;
use serde_json::{json, Value};

/// Parse JSON; on failure keep the capture well-formed with a placeholder.
pub fn safe_json_loads(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap_or_else(|_| json!({ "_unparsed_raw": String::from_utf8_lossy(raw) }))
}

pub fn write_request(
    capture_dir: &Path,
    rid: &str,
    body_bytes: &[u8],
    upstream: &str,
    task_id: Option<&str>,
    turn: Option<i64>,
    captured_at: i64,
) -> io::Result<()> {
    let rec = json!({
        "request_id": rid,
        "captured_at": captured_at,
        "upstream_url": upstream,
        "task_id": task_id,
        "turn": turn,
        "body": safe_json_loads(body_bytes),
    });
    write_pretty(&capture_dir.join(format!("{rid}.request.json")), &rec)
}

pub fn write_response_nonstream(
    capture_dir: &Path,
    rid: &str,
    status_code: u16,
    body_bytes: &[u8],
    captured_at: i64,
    upstream_headers: &HeaderMap,
) -> io::Result<()> {
    let body = safe_json_loads(body_bytes);
    let rec = json!({
        "request_id": rid,
        "captured_at_resp": captured_at,
        "stream": false,
        "status_code": status_code,
        "body": body,
        "upstream_fingerprint": fingerprint(upstream_headers, &body),
    });
    write_pretty(&capture_dir.join(format!("{rid}.response.json")), &rec)
}

pub fn write_response_stream(
    capture_dir: &Path,
    rid: &str,
    status_code: u16,
    raw_bytes: &[u8],
    captured_at: i64,
    upstream_headers: &HeaderMap,
) -> io::Result<()> {
    let synthetic = extract_model_from_sse(raw_bytes).map(|m| json!({ "model": m }));
    let rec = json!({
        "request_id": rid,
        "captured_at_resp": captured_at,
        "stream": true,
        "status_code": status_code,
        "raw": String::from_utf8_lossy(raw_bytes),
        "upstream_fingerprint": fingerprint(upstream_headers, synthetic.as_ref().unwrap_or(&Value::Null)),
    });
    write_pretty(&capture_dir.join(format!("{rid}.response.json")), &rec)
}

fn write_pretty(path: &Path, rec: &Value) -> io::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(rec)?)
}

/// `server` / `x-served-by` / `via` / `x-build-info` headers + body-echoed model.
fn fingerprint(headers: &HeaderMap, body: &Value) -> Value {
    let h = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
    };
    let served_model = body.get("model").and_then(Value::as_str).filter(|s| !s.is_empty());
    json!({
        "server": h("server"),
        "x_served_by": h("x-served-by"),
        "via": h("via"),
        "build_info": h("x-build-info"),
        "served_model": served_model,
    })
}

/// Find a `"model"` in the first parseable SSE `data:` line.
pub fn extract_model_from_sse(raw: &[u8]) -> Option<String> {
    for line in raw.split(|&b| b == b'\n') {
        let Some(line) = line.strip_prefix(b"data:") else {
            continue;
        };
        let payload = line.trim_ascii();
        if payload.is_empty() || payload == b"[DONE]" {
            continue;
        }
        if let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(payload) {
            if let Some(m) = obj.get("model").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                return Some(m.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_model_from_first_data_line() {
        let raw = b": ping\ndata: {\"id\":\"x\",\"model\":\"google/gemma\"}\ndata: [DONE]\n";
        assert_eq!(extract_model_from_sse(raw).as_deref(), Some("google/gemma"));
        assert_eq!(extract_model_from_sse(b"data: [DONE]\n"), None);
    }

    #[test]
    fn fingerprint_pulls_headers_and_model() {
        let mut h = HeaderMap::new();
        h.insert("x-served-by", "router-1".parse().unwrap());
        h.insert("x-build-info", "b-9".parse().unwrap());
        let fp = fingerprint(&h, &json!({"model": "m"}));
        assert_eq!(fp["x_served_by"], json!("router-1"));
        assert_eq!(fp["build_info"], json!("b-9"));
        assert_eq!(fp["served_model"], json!("m"));
        assert_eq!(fp["via"], Value::Null);
    }

    #[test]
    fn safe_json_loads_placeholder_on_garbage() {
        assert_eq!(safe_json_loads(b"{not json"), json!({"_unparsed_raw": "{not json"}));
        assert_eq!(safe_json_loads(b"{\"a\":1}"), json!({"a": 1}));
    }
}
