//! Decode OpenAI-compatible responses into a single synthesized assistant
//! message.

use crate::model::DecodedResponse;
use serde_json::{json, Value};

/// Decode an OpenAI-compatible SSE stream into `{content, tool_calls,
/// finish_reason}`. Concatenates `delta.content`; merges `delta.tool_calls` by
/// their `index` (first chunk for an index carries id + function.name, later
/// chunks accumulate `function.arguments` fragments). Malformed and non-`data:`
/// lines are skipped so one bad chunk never aborts the stream.
pub fn decode_sse_response(raw: &str) -> DecodedResponse {
    let mut content_parts = String::new();
    // (index, slot) kept in insertion-independent form; emitted sorted by index.
    let mut tool_calls_by_idx: Vec<(i64, Value)> = Vec::new();
    let mut finish_reason: Option<String> = None;

    for line in raw.lines() {
        if !line.starts_with("data:") {
            continue;
        }
        let payload = line["data:".len()..].trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let obj: Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let choices = obj.get("choices").and_then(Value::as_array);
        for ch in choices.into_iter().flatten() {
            let delta = ch.get("delta").cloned().unwrap_or_else(|| json!({}));
            if let Some(c) = delta.get("content").and_then(Value::as_str) {
                if !c.is_empty() {
                    content_parts.push_str(c);
                }
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc_delta in tcs {
                    let idx = tc_delta.get("index").and_then(Value::as_i64).unwrap_or(0);
                    let slot = match tool_calls_by_idx.iter_mut().find(|(i, _)| *i == idx) {
                        Some((_, slot)) => slot,
                        None => {
                            tool_calls_by_idx.push((
                                idx,
                                json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}),
                            ));
                            &mut tool_calls_by_idx.last_mut().unwrap().1
                        }
                    };
                    if let Some(id) = tc_delta.get("id").and_then(Value::as_str) {
                        if !id.is_empty() {
                            slot["id"] = json!(id);
                        }
                    }
                    if let Some(t) = tc_delta.get("type").and_then(Value::as_str) {
                        if !t.is_empty() {
                            slot["type"] = json!(t);
                        }
                    }
                    if let Some(fnobj) = tc_delta.get("function") {
                        if let Some(name) = fnobj.get("name").and_then(Value::as_str) {
                            if !name.is_empty() {
                                slot["function"]["name"] = json!(name);
                            }
                        }
                        if let Some(args) = fnobj.get("arguments").and_then(Value::as_str) {
                            if !args.is_empty() {
                                let prev = slot["function"]["arguments"].as_str().unwrap_or("").to_string();
                                slot["function"]["arguments"] = json!(prev + args);
                            }
                        }
                    }
                }
            }
            if let Some(fr) = ch.get("finish_reason").and_then(Value::as_str) {
                if !fr.is_empty() {
                    finish_reason = Some(fr.to_string());
                }
            }
        }
    }

    tool_calls_by_idx.sort_by_key(|(i, _)| *i);
    DecodedResponse {
        content: content_parts,
        tool_calls: tool_calls_by_idx.into_iter().map(|(_, v)| v).collect(),
        finish_reason,
    }
}

/// Synthesize an assistant message from a response record — handles both
/// non-stream (`body.choices[0].message`) and stream (raw SSE in `raw`).
pub fn decode_response(resp_rec: &Value) -> DecodedResponse {
    if resp_rec.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return decode_sse_response(resp_rec.get("raw").and_then(Value::as_str).unwrap_or(""));
    }
    let body = resp_rec.get("body").cloned().unwrap_or_else(|| json!({}));
    let ch = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let msg = ch.get("message").cloned().unwrap_or_else(|| json!({}));
    DecodedResponse {
        content: msg.get("content").and_then(Value::as_str).unwrap_or("").to_string(),
        tool_calls: msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        finish_reason: ch.get("finish_reason").and_then(Value::as_str).map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Assemble an SSE blob: one `data: <json>` line per object + trailing
    /// `[DONE]`.
    fn sse(objs: &[Value]) -> String {
        let mut s: String = objs
            .iter()
            .map(|o| format!("data: {}", serde_json::to_string(o).unwrap()))
            .collect::<Vec<_>>()
            .join("\n");
        s.push_str("\ndata: [DONE]\n");
        s
    }

    #[test]
    fn empty_returns_empty_message() {
        let out = decode_sse_response("");
        assert_eq!(out.content, "");
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.finish_reason, None);
    }

    #[test]
    fn concatenates_content_chunks() {
        let raw = sse(&[
            json!({"choices": [{"delta": {"content": "Hello"}}]}),
            json!({"choices": [{"delta": {"content": ", "}}]}),
            json!({"choices": [{"delta": {"content": "world!"}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]);
        let out = decode_sse_response(&raw);
        assert_eq!(out.content, "Hello, world!");
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn merges_tool_call_argument_fragments() {
        let raw = sse(&[
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "read", "arguments": ""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "{\"path\""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": ": \"a.py\"}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ]);
        let out = decode_sse_response(&raw);
        assert_eq!(out.content, "");
        assert_eq!(
            out.tool_calls,
            vec![json!({
                "id": "call_1", "type": "function",
                "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}})]
        );
        assert_eq!(out.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn keeps_multiple_tool_calls_in_index_order() {
        let raw = sse(&[
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "c0", "function": {"name": "first", "arguments": "{"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 1, "id": "c1", "function": {"name": "second", "arguments": "{}"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "}"}}]}}]}),
        ]);
        let out = decode_sse_response(&raw);
        let names: Vec<&str> = out
            .tool_calls
            .iter()
            .map(|tc| tc["function"]["name"].as_str().unwrap())
            .collect();
        let ids: Vec<&str> = out.tool_calls.iter().map(|tc| tc["id"].as_str().unwrap()).collect();
        let args: Vec<&str> = out
            .tool_calls
            .iter()
            .map(|tc| tc["function"]["arguments"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["first", "second"]);
        assert_eq!(ids, vec!["c0", "c1"]);
        assert_eq!(args, vec!["{}", "{}"]);
    }

    #[test]
    fn skips_malformed_json_lines() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: {not json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(decode_sse_response(raw).content, "ok!");
    }

    #[test]
    fn ignores_non_data_and_blank_lines() {
        let raw = ": keepalive\n\
                   \n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\
                   \n\
                   event: end\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let out = decode_sse_response(raw);
        assert_eq!(out.content, "x");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
    }
}
