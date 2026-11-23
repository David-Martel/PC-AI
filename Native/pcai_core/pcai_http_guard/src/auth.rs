//! Bearer-token handling.

use subtle::ConstantTimeEq;

/// A configured bearer token. Its `Debug` output never contains the secret.
#[derive(Clone)]
pub(crate) struct AuthToken(String);

impl AuthToken {
    pub(crate) fn new(token: String) -> Self {
        Self(token)
    }

    /// Compare a presented token without branching on where the first difference is.
    ///
    /// Only the length can leak (the comparison of unequal-length slices returns early), which
    /// reveals nothing an attacker cannot already read from the configured format.
    pub(crate) fn matches(&self, presented: &str) -> bool {
        self.0.as_bytes().ct_eq(presented.as_bytes()).into()
    }
}

impl std::fmt::Debug for AuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AuthToken(<redacted>)")
    }
}

/// Choose the effective token from an environment value and a configuration value.
///
/// The environment wins, because that is where a secret should live and it lets an operator
/// override a file without editing it. Blank values (empty or whitespace) count as unset, so
/// `PCAI_INFERENCE_TOKEN=""` can never switch authentication off or unlock a network bind.
///
/// # Examples
///
/// ```
/// use pcai_http_guard::pick_token;
///
/// assert_eq!(pick_token(Some("from-env".into()), Some("from-file".into())).as_deref(), Some("from-env"));
/// assert_eq!(pick_token(Some("  ".into()), Some("from-file".into())).as_deref(), Some("from-file"));
/// assert_eq!(pick_token(None, Some("".into())), None);
/// ```
#[must_use]
pub fn pick_token(from_env: Option<String>, from_config: Option<String>) -> Option<String> {
    [from_env, from_config]
        .into_iter()
        .flatten()
        .map(|token| token.trim().to_string())
        .find(|token| !token.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protects: the comparison itself. Detects: a comparison that accepts a prefix, a longer
    // value or an empty string.
    #[test]
    fn matches_only_the_exact_token() {
        let token = AuthToken::new("s3cret".to_string());
        assert!(token.matches("s3cret"));
        for wrong in ["", "s3cre", "s3crett", "S3CRET", "s3cret ", " s3cret"] {
            assert!(!token.matches(wrong), "{wrong:?}");
        }
    }

    // Protects: log and panic output. Detects: deriving Debug on the token type.
    #[test]
    fn debug_output_hides_the_secret() {
        let shown = format!("{:?}", AuthToken::new("s3cret".to_string()));
        assert!(!shown.contains("s3cret"), "{shown}");
    }

    // Protects: precedence and the blank-is-unset rule. Detects: an empty environment variable
    // overriding a file token, or a blank token counting as configured.
    #[test]
    fn pick_token_precedence_and_blanks() {
        let some = |text: &str| Some(text.to_string());
        assert_eq!(pick_token(some("env"), some("file")), some("env"));
        assert_eq!(pick_token(None, some("file")), some("file"));
        assert_eq!(pick_token(some(""), some("file")), some("file"));
        assert_eq!(pick_token(some("  "), None), None);
        assert_eq!(pick_token(some(" env "), None), some("env"));
        assert_eq!(pick_token(None, None), None);
    }
}
