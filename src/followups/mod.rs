//! Follow-up strategies for multi-turn runs.
//!
//! `turn` is the 1-indexed number of the upcoming turn (first follow-up is
//! `turn=2`). `continue` is cheapest; `templates` rotates a small pool;
//! `synthesized` calls a model directly (bypassing the capture proxy) to produce
//! a realistic next user message.

mod continue_;
mod synthesized;
mod templates;

use anyhow::{Context, Result};

pub trait FollowUp {
    fn next(&self, original_task: &str, last_response: &str, turn: i64) -> String;
}

pub fn get_followup(
    name: &str,
    upstream: Option<&str>,
    model: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn FollowUp>> {
    Ok(match name {
        "continue" => Box::new(continue_::ContinueFollowUp::default()),
        "templates" => Box::new(templates::TemplatesFollowUp::default()),
        "synthesized" => {
            let upstream = upstream.context("synthesized follow-up requires --upstream")?;
            let model = model.context("synthesized follow-up requires --model")?;
            Box::new(synthesized::SynthesizedFollowUp::new(upstream, model, api_key))
        }
        _ => anyhow::bail!("unknown follow-up strategy: {name:?}"),
    })
}
