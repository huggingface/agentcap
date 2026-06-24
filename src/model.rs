//! Shared types and small JSON helpers used across export and inspect.
//!
//! Bodies (request/response) are heterogeneous OpenAI payloads, so we keep them
//! as [`serde_json::Value`] and only reach for typed fields where we branch on
//! them — the same reason export stringifies them rather than imposing a schema.

use serde_json::Value;

/// An assistant message synthesized from a response record — the unified shape
/// returned by both the non-stream and streamed (SSE) decoders.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedResponse {
    pub content: String,
    /// Tool-call objects in OpenAI shape: `{id, type, function: {name, arguments}}`.
    pub tool_calls: Vec<Value>,
    pub finish_reason: Option<String>,
}

/// Canonical JSON string with object keys sorted recursively — the Rust analog
/// of Python's `json.dumps(obj, sort_keys=True)`. Used to make heterogeneous
/// sub-objects (message content arrays, tool_calls) comparable for diffing.
pub fn canonical_json(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // serde_json::to_string on a string yields a properly escaped JSON string.
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                write_canonical(&map[*k], out);
            }
            out.push('}');
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        let a = json!({"b": 1, "a": {"d": 2, "c": 3}});
        let b = json!({"a": {"c": 3, "d": 2}, "b": 1});
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(canonical_json(&a), r#"{"a":{"c":3,"d":2},"b":1}"#);
    }
}
