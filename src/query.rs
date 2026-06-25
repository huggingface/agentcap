//! fzf-style query parsing → the literal substrings to highlight in the preview
//! pane. nucleo handles the actual matching with the
//! same operator atoms; this only extracts what to colour.

/// Split a query into the literal text of each non-negated term, with the
/// operator prefix (`'`, `^`) and trailing anchor (`$`) stripped. Negated terms
/// (`!word`) and bare `|` OR separators are dropped — nothing to highlight.
pub fn parse_fzf_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for raw in query.split_whitespace() {
        if raw == "|" {
            continue;
        }
        if raw.starts_with('!') {
            continue;
        }
        let mut t = raw;
        if t.starts_with('\'') || t.starts_with('^') {
            t = &t[1..];
        }
        if t.ends_with('$') {
            t = &t[..t.len() - 1];
        }
        if !t.is_empty() {
            terms.push(t.to_string());
        }
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_returns_empty_list() {
        assert_eq!(parse_fzf_terms(""), Vec::<String>::new());
        assert_eq!(parse_fzf_terms("   "), Vec::<String>::new());
    }

    #[test]
    fn plain_words() {
        assert_eq!(parse_fzf_terms("alpha beta"), vec!["alpha", "beta"]);
    }

    #[test]
    fn strips_exact_match_quote() {
        assert_eq!(parse_fzf_terms("'hf-cli"), vec!["hf-cli"]);
    }

    #[test]
    fn strips_anchors() {
        assert_eq!(parse_fzf_terms("^foo"), vec!["foo"]);
        assert_eq!(parse_fzf_terms("bar$"), vec!["bar"]);
        assert_eq!(parse_fzf_terms("^baz$"), vec!["baz"]);
    }

    #[test]
    fn drops_negated_terms() {
        assert_eq!(parse_fzf_terms("keep !drop also"), vec!["keep", "also"]);
    }

    #[test]
    fn drops_bare_or_separator() {
        assert_eq!(parse_fzf_terms("a | b"), vec!["a", "b"]);
    }

    #[test]
    fn handles_mixed_operators() {
        assert_eq!(
            parse_fzf_terms("'exact ^prefix suffix$ !nope plain"),
            vec!["exact", "prefix", "suffix", "plain"]
        );
    }
}
