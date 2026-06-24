//! Preview-pane rendering for inspect: build `ratatui::Text` for a run, a
//! request (header + prompt + message diff), a flattened message, and an HF
//! parquet entry. Query terms are highlighted (bold red), replacing the Python
//! `_highlight` ANSI pipeline.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use serde_json::Value;

use crate::diff::{divergence, flatten, message_text, PREVIEW_MSG_CAP};

use super::sources::{MsgRecord, ReqRow, RunRow};

const ARGS_CAP: usize = 240;

/// Run metadata preview (run picker). Ports `_run_preview_cmd`.
pub fn run_preview(run: &RunRow) -> Text<'static> {
    let meta: Value = std::fs::read(run.run_dir.join("run.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let g = |k: &str| meta.get(k).map(value_str).unwrap_or_else(|| "?".to_string());
    let mut lines = vec![
        plain(format!("run:        {}", run.run_id)),
        plain(format!("agent:      {}", g("agent"))),
        plain(format!("model:      {}", g("model"))),
        plain(format!("upstream:   {}", g("upstream"))),
        plain(format!("followup:   {}", g("followup"))),
        plain(format!("turns/task: {}", g("turns_per_task"))),
        plain(format!("captures:   {}", run.n_caps)),
        plain(String::new()),
        plain("─── TASKS ───".to_string()),
    ];
    for t in meta.get("tasks").and_then(Value::as_array).cloned().unwrap_or_default() {
        let id = t.get("task_id").and_then(Value::as_str).unwrap_or("?");
        let completed = t
            .get("completed_turns")
            .map(value_str)
            .unwrap_or_else(|| "?".to_string());
        let prompt = t.get("prompt").and_then(Value::as_str).unwrap_or("").replace('\n', " ");
        lines.push(plain(format!("  {id}: ({completed} turns) {prompt}")));
    }
    Text::from(lines)
}

/// HF parquet preview (parquet picker).
pub fn hf_parquet_preview(
    path: &str,
    agent: &Option<String>,
    model: &Option<String>,
    num_rows: Option<i64>,
    tasks: &Option<Vec<(String, i64, Option<String>)>>,
) -> Text<'static> {
    let q = |o: &Option<String>| o.clone().unwrap_or_else(|| "?".to_string());
    let mut lines = vec![
        plain(format!("path:    {path}")),
        plain(format!("agent:   {}", q(agent))),
        plain(format!("model:   {}", q(model))),
        plain(format!(
            "rows:    {}",
            num_rows.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string())
        )),
        plain(String::new()),
        plain("─── TASKS ───".to_string()),
    ];
    match tasks {
        None => lines.push(plain("loading…".to_string())),
        Some(ts) => {
            for (id, turns, prompt) in ts {
                let prompt = prompt.clone().unwrap_or_default().replace('\n', " ");
                lines.push(plain(format!("  {id}: ({turns} turns) {prompt}")));
            }
        }
    }
    Text::from(lines)
}

/// Request preview: header + initial prompt + message diff. Ports `_preview_cmd`.
pub fn request_preview(row: &ReqRow, body: &Value, prev_body: Option<&Value>, terms: &[String]) -> Text<'static> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let initial_prompt = messages
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .map(message_text)
        .unwrap_or_default();
    let serialized = serde_json::to_string(body).unwrap_or_default();
    let size_b = serialized.len();
    let ts = if row.captured_at != 0 {
        chrono::DateTime::from_timestamp(row.captured_at, 0)
            .map(|dt| dt.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "?".to_string())
    } else {
        "?".to_string()
    };

    let mut lines = Vec::new();
    lines.push(plain(format!("rid:    {}", row.rid)));
    if row.task_id.is_some() || row.turn.is_some() {
        let tid = row.task_id.clone().unwrap_or_else(|| "?".to_string());
        let turn = row.turn.map(|t| t.to_string()).unwrap_or_else(|| "?".to_string());
        lines.push(plain(format!("task:   {tid}  turn={turn}")));
    }
    lines.push(plain(format!("time:   {ts}")));
    lines.push(plain(format!("status: {}", row.status)));
    lines.push(plain(format!(
        "model:  {}",
        body.get("model").and_then(Value::as_str).unwrap_or("?")
    )));
    lines.push(plain(format!(
        "size:   {} bytes (~{} tokens)",
        commafy(size_b),
        commafy(size_b / 4)
    )));
    lines.push(plain(String::new()));
    lines.push(divider("PROMPT"));
    for l in split_lines(if initial_prompt.is_empty() {
        "(no user message)"
    } else {
        &initial_prompt
    }) {
        lines.push(highlight_line(&l, terms, Style::default()));
    }
    lines.push(plain(String::new()));

    let prev_msgs: Vec<Value> = prev_body
        .and_then(|p| p.get("messages"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let has_previous = prev_body.is_some();
    let i = divergence(&prev_msgs, &messages);
    let removed = prev_msgs.len().saturating_sub(i);
    let new_msgs: Vec<Value> = messages.get(i..).map(|s| s.to_vec()).unwrap_or_default();
    let suffix = if has_previous {
        format!(
            "{} since previous call",
            crate::diff::delta_label(removed, new_msgs.len())
        )
    } else {
        let n = new_msgs.len();
        format!("initial: {n} msg{}", if n == 1 { "" } else { "s" })
    };
    lines.push(divider(&format!("MESSAGES ({suffix})")));
    if has_previous {
        lines.push(plain("  ...".to_string()));
    }
    if new_msgs.is_empty() && removed == 0 {
        lines.push(plain("(no diff vs previous call)".to_string()));
    }
    for m in &new_msgs {
        render_message(m, terms, &mut lines);
    }
    Text::from(lines)
}

/// One flattened message detail (message picker). Ports `_render_msg_preview`.
pub fn message_preview(rec: &MsgRecord, terms: &[String]) -> Text<'static> {
    let mut lines = vec![plain(format!("role:         {}", rec.role))];
    match rec.msg_idx {
        Some(i) => lines.push(plain(format!("msg_idx:      {i}"))),
        None => lines.push(plain("msg_idx:      (response)".to_string())),
    }
    if let Some(tcid) = &rec.tool_call_id {
        lines.push(plain(format!("tool_call_id: {tcid}")));
    }
    if let Some(fr) = &rec.finish_reason {
        lines.push(plain(format!("finish_reason: {fr}")));
    }
    lines.push(plain(String::new()));
    let content = if rec.content.is_empty() {
        "(no content)"
    } else {
        &rec.content
    };
    for l in split_lines(content) {
        lines.push(highlight_line(&l, terms, Style::default()));
    }
    Text::from(lines)
}

fn render_message(m: &Value, terms: &[String], out: &mut Vec<Line<'static>>) {
    let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
    if role == "assistant" {
        for tc in m
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
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
            out.push(tagged(
                &format!("assistant tool_call → {fname}"),
                &format!("  args={}", flatten(args, ARGS_CAP)),
                terms,
            ));
        }
        let content = message_text(m);
        if !content.is_empty() {
            out.push(tagged(
                "assistant content",
                &format!(" {}", flatten(&content, PREVIEW_MSG_CAP)),
                terms,
            ));
        }
        return;
    }
    if role == "tool" {
        let tcid: String = m
            .get("tool_call_id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .chars()
            .take(8)
            .collect();
        out.push(tagged(&format!("tool result, tool_call_id={tcid}"), "", terms));
        out.push(highlight_line(
            &format!("  {}", flatten(&message_text(m), PREVIEW_MSG_CAP)),
            terms,
            Style::default(),
        ));
        return;
    }
    out.push(tagged(
        role,
        &format!(" {}", flatten(&message_text(m), PREVIEW_MSG_CAP)),
        terms,
    ));
}

/// A line with a reverse-video `[tag]` marker followed by highlighted content.
fn tagged(tag: &str, content: &str, terms: &[String]) -> Line<'static> {
    let mut spans = vec![
        Span::raw("  "),
        Span::styled(format!("[{tag}]"), Style::default().add_modifier(Modifier::REVERSED)),
    ];
    spans.extend(highlight_spans(content, terms, Style::default()));
    Line::from(spans)
}

fn plain(s: String) -> Line<'static> {
    Line::from(Span::raw(s))
}

fn divider(label: &str) -> Line<'static> {
    let bar = "─".repeat(40usize.saturating_sub(label.len()));
    plain(format!("─── {label} {bar}"))
}

fn highlight_line(s: &str, terms: &[String], base: Style) -> Line<'static> {
    Line::from(highlight_spans(s, terms, base))
}

/// Split `s` into spans, wrapping case-insensitive occurrences of any term in
/// bold red. Longest terms first so a shorter prefix can't shadow a longer one.
fn highlight_spans(s: &str, terms: &[String], base: Style) -> Vec<Span<'static>> {
    let mut sorted: Vec<&String> = terms.iter().filter(|t| !t.is_empty()).collect();
    sorted.sort_by_key(|t| std::cmp::Reverse(t.len()));
    if sorted.is_empty() {
        return vec![Span::styled(s.to_string(), base)];
    }
    let hl = Style::default().fg(Color::Red).add_modifier(Modifier::BOLD);
    let lower = s.to_lowercase();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut hit: Option<usize> = None;
        for t in &sorted {
            if lower[i..].starts_with(&t.to_lowercase()) {
                hit = Some(t.len());
                break;
            }
        }
        match hit {
            Some(len) => {
                spans.push(Span::styled(s[i..i + len].to_string(), hl));
                i += len;
            }
            None => {
                // Advance to the next char boundary, accumulating plain text.
                let start = i;
                let step = s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                i += step;
                // Coalesce consecutive plain chars until the next potential match.
                while i < s.len() && !sorted.iter().any(|t| lower[i..].starts_with(&t.to_lowercase())) {
                    i += s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                }
                spans.push(Span::styled(s[start..i].to_string(), base));
            }
        }
    }
    spans
}

fn split_lines(s: &str) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    s.lines().map(str::to_string).collect()
}

fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn commafy(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    for (idx, b) in bytes.iter().enumerate() {
        if idx > 0 && (bytes.len() - idx).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}
