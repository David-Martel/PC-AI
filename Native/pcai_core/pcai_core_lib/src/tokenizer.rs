use regex::Regex;
use std::sync::OnceLock;

static TOKEN_RE: OnceLock<Regex> = OnceLock::new();

/// Estimates the number of tokens in a string for Gemma-like models.
/// Uses a heuristic approach based on word and punctuation patterns.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let re = TOKEN_RE.get_or_init(|| {
        Regex::new(r"(?x)
            [\p{L}\p{N}]+ |  # Words or numbers
            \s+           |  # Whitespace
            [^\p{L}\p{N}\s]  # Punctuation/Other
        ").unwrap_or_else(|_| Regex::new(".").expect("TODO: Verify unwrap")) // Fallback to safe catch-all
    });

    let mut count = 0;
    for mat in re.find_iter(text) {
        let piece = mat.as_str();

        // Rough heuristic for SentencePiece:
        // Long words are often split, average of 4 chars per token for typical text.
        let len = piece.len();
        if len > 8 {
            count += len.div_ceil(4);
        } else {
            count += 1;
        }
    }

    // Add 1 for the BOS token if not empty
    count + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("Hello world") >= 2);
        assert!(estimate_tokens("This is a test.") >= 5);
    }
}
