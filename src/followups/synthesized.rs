//! Synthesized follow-up: sends `(original_task, last_response)` to a model and
//! uses the reply as the next user message.
//!
//! By design this call **bypasses the capture proxy** — it talks to the model
//! server directly so the captured corpus stays a clean record of agent↔model
//! interaction. Any failure falls back to `"continue"` (logged).

use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::{json, Value};

use super::FollowUp;

const PROMPT_TEMPLATE: &str = "You are a developer interacting with a coding agent. Given the agent's\n\
last response, produce ONE short follow-up question or instruction\n\
(<=30 words) that pushes the conversation forward. Don't ask the\n\
agent to summarise; ask it to do or show something.\n\n\
Original task:\n<<<{task}>>>\n\n\
Agent's last response:\n<<<{response}>>>\n\n\
Follow-up:\n";

/// Build the chat-completions URL: append `/chat/completions` if the upstream
/// already ends in `/v1`, else `/v1/chat/completions`.
fn chat_url(upstream: &str) -> String {
    let base = upstream.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

pub struct SynthesizedFollowUp {
    upstream: String,
    model: String,
    api_key: Option<String>,
    fallback: String,
    client: Client,
}

impl SynthesizedFollowUp {
    pub fn new(upstream: &str, model: &str, api_key: Option<&str>) -> Self {
        SynthesizedFollowUp {
            upstream: upstream.to_string(),
            model: model.to_string(),
            api_key: api_key.map(str::to_string),
            fallback: "continue".to_string(),
            // Generous cap (blocking's 30s default is too short for a slow synth);
            // falls back to "continue" if it still times out.
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap_or_default(),
        }
    }

    fn call(&self, prompt: &str) -> anyhow::Result<String> {
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
        });
        let mut req = self.client.post(chat_url(&self.upstream)).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let data: Value = req.send()?.error_for_status()?.json()?;
        data.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("synthesizer response missing choices[0].message.content"))
    }
}

impl FollowUp for SynthesizedFollowUp {
    fn next(&self, original_task: &str, last_response: &str, turn: i64) -> String {
        let prompt = PROMPT_TEMPLATE
            .replace("{task}", original_task)
            .replace("{response}", last_response);
        match self.call(&prompt) {
            Ok(text) if !text.is_empty() => text,
            Ok(_) => self.fallback.clone(),
            Err(e) => {
                eprintln!(
                    "  [followups] synthesized turn={turn} fell back to {:?}: {e}",
                    self.fallback
                );
                self.fallback.clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_url_rules() {
        assert_eq!(chat_url("http://h:8000"), "http://h:8000/v1/chat/completions");
        assert_eq!(chat_url("http://h:8000/v1"), "http://h:8000/v1/chat/completions");
        assert_eq!(chat_url("http://h:8000/v1/"), "http://h:8000/v1/chat/completions");
    }

    #[test]
    fn falls_back_on_unreachable_upstream() {
        // Port 1 is unbound → connect error → fallback.
        let fu = SynthesizedFollowUp::new("http://127.0.0.1:1", "m", None);
        assert_eq!(fu.next("task", "resp", 2), "continue");
    }
}
