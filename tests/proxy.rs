//! Proxy integration test (no podman): a mock upstream `tiny_http` server over
//! loopback, driven through `CaptureProxy` with `reqwest::blocking`. Asserts the
//! capture files, `(task_id, turn)` stamping, SSE reassembly, and fingerprint.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::Value;
use tiny_http::{Header, Response, Server, StatusCode};

/// Start a mock OpenAI-compatible upstream; returns (base_url, shutdown).
fn mock_upstream() -> (String, impl FnOnce()) {
    let server = Arc::new(Server::http("127.0.0.1:0").unwrap());
    let port = server.server_addr().to_ip().unwrap().port();
    let running = Arc::new(AtomicBool::new(true));
    let (s, r) = (server.clone(), running.clone());
    let handle = std::thread::spawn(move || {
        while r.load(Ordering::Relaxed) {
            let Ok(mut req) = s.recv() else { break };
            let served_by = Header::from_bytes(&b"x-served-by"[..], &b"mock"[..]).unwrap();
            if req.url().starts_with("/v1/chat/completions") {
                let mut body = Vec::new();
                req.as_reader().read_to_end(&mut body).unwrap();
                let stream = serde_json::from_slice::<Value>(&body)
                    .ok()
                    .and_then(|v| v.get("stream").and_then(Value::as_bool))
                    .unwrap_or(false);
                if stream {
                    let ct = Header::from_bytes(&b"content-type"[..], &b"text/event-stream"[..]).unwrap();
                    let raw = "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\ndata: [DONE]\n";
                    let _ = req.respond(Response::from_string(raw).with_header(ct).with_header(served_by));
                } else {
                    let ct = Header::from_bytes(&b"content-type"[..], &b"application/json"[..]).unwrap();
                    let json = r#"{"id":"c1","model":"m","choices":[{"message":{"content":"hi"}}]}"#;
                    let _ = req.respond(Response::from_string(json).with_header(ct).with_header(served_by));
                }
            } else {
                // passthrough target
                let _ =
                    req.respond(Response::from_string(r#"{"data":[{"id":"m"}]}"#).with_status_code(StatusCode(200)));
            }
        }
    });
    let shutdown = move || {
        running.store(false, Ordering::Relaxed);
        server.unblock();
        let _ = handle.join();
    };
    (format!("http://127.0.0.1:{port}"), shutdown)
}

fn read_json(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn requests(dir: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.file_name().unwrap().to_str().unwrap().ends_with(".request.json") {
            out.push(read_json(&p));
        }
    }
    out
}

#[test]
fn captures_nonstream_stream_and_passes_through() {
    let (upstream, shutdown) = mock_upstream();
    let tmp = tempfile::tempdir().unwrap();
    let cap = tmp.path().join("captures");

    let proxy = agentcap::proxy::serve_in_thread(&upstream, &cap, "127.0.0.1").unwrap();
    let base = format!("http://127.0.0.1:{}", proxy.port);
    let client = reqwest::blocking::Client::new();

    // Non-stream call, stamped task_00/turn 1.
    proxy.set_context(Some("task_00"), Some(1));
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": [], "stream": false}))
        .send()
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.text().unwrap(),
        r#"{"id":"c1","model":"m","choices":[{"message":{"content":"hi"}}]}"#
    );

    // Streamed call, stamped task_00/turn 2.
    proxy.set_context(Some("task_00"), Some(2));
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": [], "stream": true}))
        .send()
        .unwrap();
    assert!(r.text().unwrap().contains("data: [DONE]"));

    // Passthrough — not captured.
    let r = client.get(format!("{base}/v1/models")).send().unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.text().unwrap().contains("\"data\""));

    proxy.shutdown();
    shutdown();

    // Exactly two captures (the passthrough is not recorded).
    let reqs = requests(&cap);
    assert_eq!(reqs.len(), 2, "expected 2 request captures, got {}", reqs.len());

    // Match each capture to its response and assert shape + stamping.
    for req in &reqs {
        let rid = req["request_id"].as_str().unwrap();
        assert_eq!(req["upstream_url"], serde_json::json!(upstream));
        assert_eq!(req["task_id"], serde_json::json!("task_00"));
        let resp = read_json(&cap.join(format!("{rid}.response.json")));
        assert_eq!(resp["status_code"], serde_json::json!(200));
        assert_eq!(resp["upstream_fingerprint"]["x_served_by"], serde_json::json!("mock"));
        let turn = req["turn"].as_i64().unwrap();
        if turn == 1 {
            assert_eq!(resp["stream"], serde_json::json!(false));
            assert_eq!(
                resp["body"]["choices"][0]["message"]["content"],
                serde_json::json!("hi")
            );
            assert_eq!(resp["upstream_fingerprint"]["served_model"], serde_json::json!("m"));
        } else {
            assert_eq!(resp["stream"], serde_json::json!(true));
            assert!(resp["raw"].as_str().unwrap().contains("\"content\":\"ok\""));
            // model recovered from the SSE stream
            assert_eq!(resp["upstream_fingerprint"]["served_model"], serde_json::json!("m"));
        }
    }
}
