//! Literal-`continue` follow-up.

use super::FollowUp;

pub struct ContinueFollowUp {
    text: String,
}

impl Default for ContinueFollowUp {
    fn default() -> Self {
        ContinueFollowUp {
            text: "continue".to_string(),
        }
    }
}

impl FollowUp for ContinueFollowUp {
    fn next(&self, _original_task: &str, _last_response: &str, _turn: i64) -> String {
        self.text.clone()
    }
}
