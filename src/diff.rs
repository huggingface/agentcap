//! Message diffing + one-line summaries for the inspect picker.

use crate::model::canonical_json;
use serde_json::Value;

pub const PICKER_SUMMARY_CAP: usize = 160;
pub const PREVIEW_MSG_CAP: usize = 400;

/// Canonical comparison key for a `messages[]` entry. Compares only the
/// load-bearing fields (role / content / tool_call_id / tool_calls) and ignores
/// optional metadata (e.g. the tool `name` field hermes adds on some turns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageKey {
    role: Option<String>,
    content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<String>,
}

pub fn message_key(m: &Value) -> MessageKey {
    let content = match m.get("content") {
        Some(Value::Array(_)) => Some(canonical_json(&m["content"])),
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    let tool_calls = match m.get("tool_calls") {
        Some(tc) if is_truthy(tc) => Some(canonical_json(tc)),
        _ => None,
    };
    MessageKey {
        role: m.get("role").and_then(Value::as_str).map(str::to_string),
        content,
        tool_call_id: m.get("tool_call_id").and_then(Value::as_str).map(str::to_string),
        tool_calls,
    }
}

/// Truthiness for the values we test: non-empty arrays/strings/objects,
/// non-zero numbers, `true`. `null`/`false`/empty are falsy.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Index of the first position where `prev` and `curr` diverge (element-by-element
/// over the common prefix). `prev[i..]` is the removed suffix, `curr[i..]` the
/// added one — mirrors `_diff_messages`'s return, but as the split point.
pub fn divergence(prev: &[Value], curr: &[Value]) -> usize {
    let n = prev.len().min(curr.len());
    for j in 0..n {
        if message_key(&prev[j]) != message_key(&curr[j]) {
            return j;
        }
    }
    n
}

/// Compact `messages[]` delta marker. Hides the removed count when zero (the
/// common pure-append case); surfaces it for swaps (e.g. `-1 +1`).
pub fn delta_label(removed: usize, added: usize) -> String {
    if removed > 0 {
        format!("-{removed} +{added}")
    } else {
        format!("+{added}")
    }
}

/// Flatten `message.content` to a string. Tool / multimodal messages carry
/// list-typed content; join the text parts.
pub fn message_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| p.get("text").and_then(Value::as_str).unwrap_or(""))
            .collect::<Vec<_>>()
            .join(" "),
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Single-line, char-length-capped text (collapses internal whitespace).
pub fn flatten(s: &str, cap: usize) -> String {
    let collapsed = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= cap {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(cap).collect();
        out.push('…');
        out
    }
}

/// One-line role-aware summary of one `messages[]` entry for the MESSAGES column.
pub fn message_summary(m: &Value) -> String {
    let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
    let s = if role == "assistant" {
        let tcs = m.get("tool_calls").and_then(Value::as_array);
        match tcs.filter(|a| !a.is_empty()) {
            Some(tcs) => {
                let tc = &tcs[0];
                let func = tc.get("function");
                let fname = func
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("?");
                let args = func
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let extra = if tcs.len() > 1 {
                    format!(" +{}", tcs.len() - 1)
                } else {
                    String::new()
                };
                format!("assistant→{fname}{extra} {args}")
            }
            None => format!("assistant: {}", message_text(m)),
        }
    } else if role == "tool" {
        format!("tool: {}", message_text(m))
    } else {
        format!("{role}: {}", message_text(m))
    };
    flatten(&s, PICKER_SUMMARY_CAP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn divergence_pure_append() {
        let prev = vec![json!({"role": "user", "content": "a"})];
        let curr = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
        ];
        let i = divergence(&prev, &curr);
        assert_eq!(i, 1);
        assert_eq!(prev.len() - i, 0); // removed
        assert_eq!(curr.len() - i, 1); // added
    }

    #[test]
    fn divergence_swap_at_tail() {
        // Same length, last element differs (turn-boundary followup swap).
        let prev = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "user", "content": "old"}),
        ];
        let curr = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "user", "content": "new"}),
        ];
        assert_eq!(divergence(&prev, &curr), 1);
    }

    #[test]
    fn message_key_ignores_tool_name_metadata() {
        // hermes adds `name` on the tool message some turns; the key must match.
        let a = json!({"role": "tool", "content": "x", "tool_call_id": "c1", "name": "read"});
        let b = json!({"role": "tool", "content": "x", "tool_call_id": "c1"});
        assert_eq!(message_key(&a), message_key(&b));
    }

    #[test]
    fn message_key_content_array_order_independent_inner_keys() {
        let a = json!({"role": "user", "content": [{"type": "text", "text": "hi"}]});
        let b = json!({"role": "user", "content": [{"text": "hi", "type": "text"}]});
        assert_eq!(message_key(&a), message_key(&b));
    }

    #[test]
    fn delta_label_formats() {
        assert_eq!(delta_label(0, 3), "+3");
        assert_eq!(delta_label(1, 1), "-1 +1");
        assert_eq!(delta_label(2, 5), "-2 +5");
    }

    #[test]
    fn message_text_joins_array_parts() {
        let m = json!({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]});
        assert_eq!(message_text(&m), "a b");
    }

    #[test]
    fn flatten_collapses_and_caps() {
        assert_eq!(flatten("a\n  b\tc", 100), "a b c");
        assert_eq!(flatten("abcdef", 3), "abc…");
    }

    #[test]
    fn summary_tool_call() {
        let m = json!({"role": "assistant", "tool_calls": [
            {"function": {"name": "read", "arguments": "{\"p\":1}"}},
            {"function": {"name": "write", "arguments": "{}"}}]});
        assert_eq!(message_summary(&m), "assistant→read +1 {\"p\":1}");
    }
}
