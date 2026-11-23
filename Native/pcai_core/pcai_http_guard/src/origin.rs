//! Browser `Origin` allow-list and the CORS layer built from it.
//!
//! The list is empty by default. Empty means no `CorsLayer` at all (no CORS headers are ever
//! sent) and any request that carries an `Origin` header is refused: such a request comes from a
//! web page, and no page is meant to call these servers unless the operator listed its origin.

use std::collections::HashSet;
use std::time::Duration;

use axum::http::{header, HeaderValue, Method};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::host::parse_authority_host;
use crate::SecurityError;

/// Normalised, validated origins.
#[derive(Debug, Clone, Default)]
pub(crate) struct OriginAllowList {
    /// Lowercase `scheme://host[:port]` strings, the form browsers send.
    origins: HashSet<String>,
    values: Vec<HeaderValue>,
}

impl OriginAllowList {
    pub(crate) fn new(configured: &[String]) -> Result<Self, SecurityError> {
        let mut origins = HashSet::new();
        let mut values = Vec::new();
        for entry in configured {
            let normalised = normalise_origin(entry).map_err(|reason| SecurityError::InvalidCorsOrigin {
                origin: entry.clone(),
                reason,
            })?;
            if origins.insert(normalised.clone()) {
                let value = HeaderValue::from_str(&normalised).map_err(|_| SecurityError::InvalidCorsOrigin {
                    origin: entry.clone(),
                    reason: "it is not a valid header value",
                })?;
                values.push(value);
            }
        }
        Ok(Self { origins, values })
    }

    /// Whether a request's `Origin` header value is listed.
    pub(crate) fn allows(&self, origin_header: &str) -> bool {
        self.origins.contains(&origin_header.to_ascii_lowercase())
    }

    /// The CORS layer for the listed origins, or `None` when the list is empty.
    pub(crate) fn cors_layer(&self) -> Option<CorsLayer> {
        if self.values.is_empty() {
            return None;
        }
        Some(
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(self.values.clone()))
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                .max_age(Duration::from_secs(600)),
        )
    }
}

/// Validate `scheme://host[:port]` and return it in the form a browser sends: lowercase, no
/// trailing slash, default port omitted.
fn normalise_origin(entry: &str) -> Result<String, &'static str> {
    let lowered = entry.trim().to_ascii_lowercase();
    let origin = lowered.strip_suffix('/').unwrap_or(&lowered);
    let (scheme, authority) = origin
        .split_once("://")
        .ok_or("it must look like https://host[:port]")?;
    if scheme != "http" && scheme != "https" {
        return Err("the scheme must be http or https");
    }
    if authority.is_empty() || authority.contains(['/', '?', '#', '@', ' ']) {
        return Err("it must be an origin only: no path, query, fragment or user info");
    }
    if parse_authority_host(authority).is_none() {
        return Err("the host or port is malformed");
    }
    let default_port = if scheme == "http" { ":80" } else { ":443" };
    let authority = authority.strip_suffix(default_port).unwrap_or(authority);
    Ok(format!("{scheme}://{authority}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(entries: &[&str]) -> Result<OriginAllowList, SecurityError> {
        let entries: Vec<String> = entries.iter().map(|entry| (*entry).to_string()).collect();
        OriginAllowList::new(&entries)
    }

    // Protects: matching exactly what browsers send. Detects: case-sensitive matching, a trailing
    // slash or an explicit default port in the config breaking a legitimate client.
    #[test]
    fn valid_origins_are_normalised() {
        let origins = list(&[
            "https://Tools.Example:3000/",
            "http://localhost:5173",
            "https://app.example:443",
            "http://[::1]:8000",
        ])
        .expect("valid origins");
        assert!(origins.allows("https://tools.example:3000"));
        assert!(origins.allows("HTTPS://TOOLS.EXAMPLE:3000"));
        assert!(origins.allows("http://localhost:5173"));
        assert!(origins.allows("https://app.example"));
        assert!(origins.allows("http://[::1]:8000"));
        assert!(!origins.allows("https://tools.example"));
        assert!(!origins.allows("https://tools.example:3001"));
        assert!(!origins.allows("null"));
        assert!(!origins.allows(""));
    }

    // Protects: rejection of config that cannot work or is too broad. Detects: accepting `*`,
    // `null`, paths or other schemes, any of which would silently widen or break the allow-list.
    #[test]
    fn invalid_origins_are_rejected() {
        for bad in [
            "*",
            "null",
            "",
            "evil.example",
            "https://",
            "https://a.example/path",
            "https://a.example?x=1",
            "https://user@a.example",
            "ftp://a.example",
            "https://a b.example",
            "https://a.example:99999",
        ] {
            assert!(
                matches!(list(&[bad]), Err(SecurityError::InvalidCorsOrigin { .. })),
                "{bad:?} must be rejected"
            );
        }
    }

    // Protects: the "empty means nothing" rule. Detects: building a CorsLayer with no origins,
    // which would still add headers or answer preflights.
    #[test]
    fn empty_list_builds_no_cors_layer() {
        assert!(list(&[]).expect("empty list").cors_layer().is_none());
        assert!(list(&["https://a.example"]).expect("one origin").cors_layer().is_some());
    }
}
