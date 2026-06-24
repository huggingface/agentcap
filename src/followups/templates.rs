//! Rotating-template follow-up: cycles a small fixed pool.

use super::FollowUp;

pub struct TemplatesFollowUp {
    pool: Vec<String>,
}

impl Default for TemplatesFollowUp {
    fn default() -> Self {
        TemplatesFollowUp {
            pool: ["continue", "go on", "what else?", "keep going"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl FollowUp for TemplatesFollowUp {
    fn next(&self, _original_task: &str, _last_response: &str, turn: i64) -> String {
        // turn=2 (first follow-up) → pool[0]; turn=3 → pool[1]; …
        let idx = (turn - 2).rem_euclid(self.pool.len() as i64) as usize;
        self.pool[idx].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_from_turn_two() {
        let t = TemplatesFollowUp::default();
        assert_eq!(t.next("", "", 2), "continue");
        assert_eq!(t.next("", "", 3), "go on");
        assert_eq!(t.next("", "", 5), "keep going");
        assert_eq!(t.next("", "", 6), "continue"); // wraps
    }
}
