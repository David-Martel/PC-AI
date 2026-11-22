//! The security policy and the middleware that enforces it.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, Method, StatusCode, Version},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    Json, Router,
};

use crate::auth::AuthToken;
use crate::bind::is_loopback;
use crate::host::{parse_authority_host, HostAllowList};
use crate::origin::OriginAllowList;

/// Raw, unvalidated settings, as read from a configuration file or command line.
#[derive(Clone, Default)]
pub struct SecurityOptions {
    /// Extra `Host` values to accept besides loopback names and the bind address.
    pub allowed_hosts: Vec<String>,
    /// Browser origins allowed to call the API. Empty means no CORS and no browser callers.
    pub cors_allowed_origins: Vec<String>,
    /// Bearer token. Blank counts as unset. See [`crate::pick_token`] for the environment merge.
    pub auth_token: Option<String>,
}

impl std::fmt::Debug for SecurityOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityOptions")
            .field("allowed_hosts", &self.allowed_hosts)
            .field("cors_allowed_origins", &self.cors_allowed_origins)
            .field("auth_token", &self.auth_token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// Why a set of [`SecurityOptions`] cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    /// A network-reachable bind without a credential.
    #[error(
        "refusing to bind {bind}: it is not a loopback address and no authentication token is configured; set an \
         auth token (the `auth_token` setting or the token environment variable) or bind 127.0.0.1 or ::1"
    )]
    TokenRequired {
        /// The requested bind address.
        bind: SocketAddr,
    },

    /// A wildcard bind has no address clients could put in `Host`.
    #[error(
        "binding the wildcard address {bind} requires `allowed_hosts` naming the hosts or addresses clients use, \
         because the wildcard itself is never accepted as a Host header"
    )]
    AllowedHostsRequired {
        /// The requested bind address.
        bind: SocketAddr,
    },

    /// An `allowed_hosts` entry that can never match or would reopen a rebinding hole.
    #[error(
        "invalid allowed host {host:?}: use a host name or IP address, optionally with a port; wildcards and \
         0.0.0.0/:: are not accepted"
    )]
    InvalidAllowedHost {
        /// The entry as configured.
        host: String,
    },

    /// A `cors_allowed_origins` entry that is not a plain origin.
    #[error("invalid CORS origin {origin:?}: {reason}")]
    InvalidCorsOrigin {
        /// The entry as configured.
        origin: String,
        /// What is wrong with it.
        reason: &'static str,
    },
}

/// Validated security settings for one server, ready to wrap a router.
///
/// Cloning is cheap; the state is shared.
#[derive(Clone)]
pub struct SecurityPolicy {
    inner: Arc<Inner>,
}

struct Inner {
    hosts: HostAllowList,
    origins: OriginAllowList,
    token: Option<AuthToken>,
}

impl std::fmt::Debug for SecurityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityPolicy")
            .field("hosts", &self.inner.hosts)
            .field("origins", &self.inner.origins)
            .field("token", &self.inner.token)
            .finish()
    }
}

impl SecurityPolicy {
    /// Validate `options` for a server that will listen on `bind`.
    ///
    /// # Errors
    ///
    /// - [`SecurityError::TokenRequired`] when `bind` is not a loopback address and no non-blank
    ///   token is configured;
    /// - [`SecurityError::AllowedHostsRequired`] when `bind` is a wildcard and `allowed_hosts` is
    ///   empty;
    /// - [`SecurityError::InvalidAllowedHost`] / [`SecurityError::InvalidCorsOrigin`] for a bad
    ///   list entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use pcai_http_guard::{SecurityError, SecurityOptions, SecurityPolicy};
    ///
    /// let loopback = "127.0.0.1:8080".parse()?;
    /// assert!(SecurityPolicy::new(loopback, &SecurityOptions::default()).is_ok());
    ///
    /// let lan = "192.168.1.20:8080".parse()?;
    /// let refused = SecurityPolicy::new(lan, &SecurityOptions::default());
    /// assert!(matches!(refused, Err(SecurityError::TokenRequired { .. })));
    /// # Ok::<(), std::net::AddrParseError>(())
    /// ```
    pub fn new(bind: SocketAddr, options: &SecurityOptions) -> Result<Self, SecurityError> {
        let token = options
            .auth_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| AuthToken::new(token.to_string()));

        if token.is_none() && !is_loopback(bind.ip()) {
            return Err(SecurityError::TokenRequired { bind });
        }
        if bind.ip().is_unspecified() && options.allowed_hosts.is_empty() {
            return Err(SecurityError::AllowedHostsRequired { bind });
        }

        Ok(Self {
            inner: Arc::new(Inner {
                hosts: HostAllowList::new(bind.ip(), &options.allowed_hosts)?,
                origins: OriginAllowList::new(&options.cors_allowed_origins)?,
                token,
            }),
        })
    }

    /// Whether a bearer token is required on protected routes.
    #[must_use]
    pub fn auth_required(&self) -> bool {
        self.inner.token.is_some()
    }

    /// Wrap `router` with the guards.
    ///
    /// Outermost first, a request meets: the `Host` check, the `Origin` check, CORS (only when
    /// origins are configured), and the bearer check. CORS sits outside the bearer check so a
    /// browser preflight, which carries no credentials, is answered rather than refused. Add
    /// request logging outside this call.
    pub fn apply<S>(&self, router: Router<S>) -> Router<S>
    where
        S: Clone + Send + Sync + 'static,
    {
        let mut router = router.layer(from_fn_with_state(self.inner.clone(), bearer_guard));
        if let Some(cors) = self.inner.origins.cors_layer() {
            router = router.layer(cors);
        }
        router
            .layer(from_fn_with_state(self.inner.clone(), origin_guard))
            .layer(from_fn_with_state(self.inner.clone(), host_guard))
    }
}

/// A short JSON error in the shape OpenAI-compatible clients already parse.
fn reject(status: StatusCode, kind: &'static str, code: &'static str, message: &'static str) -> Response {
    let body = serde_json::json!({ "error": { "message": message, "type": kind, "code": code } });
    (status, Json(body)).into_response()
}

/// Escape and truncate an attacker-controlled value before it reaches a log line.
fn loggable(value: &str) -> String {
    let shown: String = value.chars().take(96).collect();
    format!("{shown:?}")
}

/// Refuse any request whose `Host` is not on the allow-list (DNS rebinding defence).
///
/// - `421 Misdirected Request` (`host_not_allowed`): a well-formed Host that is not allowed.
/// - `400 Bad Request` (`invalid_host`): missing, duplicated or malformed Host. HTTP/1.1
///   requires the header, so its absence is a malformed request. HTTP/1.0 may omit it, and a
///   browser never speaks HTTP/1.0, so a Host-less HTTP/1.0 request is accepted for old clients.
/// - Both the header and the URI authority (HTTP/2, absolute-form) are checked when present.
async fn host_guard(State(inner): State<Arc<Inner>>, request: Request, next: Next) -> Response {
    match check_host(&inner.hosts, &request) {
        Ok(()) => next.run(request).await,
        Err(response) => response,
    }
}

fn check_host(allowed: &HostAllowList, request: &Request) -> Result<(), Response> {
    let malformed = || {
        reject(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_host",
            "The Host header is missing, duplicated or malformed.",
        )
    };

    let mut keys = Vec::with_capacity(2);
    let mut host_headers = request.headers().get_all(header::HOST).iter();
    if let Some(value) = host_headers.next() {
        if host_headers.next().is_some() {
            return Err(malformed());
        }
        let text = value.to_str().map_err(|_| malformed())?;
        keys.push((text, parse_authority_host(text).ok_or_else(malformed)?));
    }
    if let Some(authority) = request.uri().authority() {
        let text = authority.as_str();
        keys.push((text, parse_authority_host(text).ok_or_else(malformed)?));
    }

    if keys.is_empty() {
        return if request.version() == Version::HTTP_10 {
            Ok(())
        } else {
            Err(malformed())
        };
    }
    if let Some((text, _)) = keys.iter().find(|(_, key)| !allowed.allows(key)) {
        tracing::warn!(host = %loggable(text), "rejected request: Host header is not allowed");
        return Err(reject(
            StatusCode::MISDIRECTED_REQUEST,
            "invalid_request_error",
            "host_not_allowed",
            "This server does not answer to that Host.",
        ));
    }
    Ok(())
}

/// Refuse any request carrying an `Origin` that is not in `cors_allowed_origins`.
///
/// A request with an `Origin` header comes from a web page. Without this check a cross-site
/// "simple" request (a form post, or `fetch` with a safelisted content type) is delivered even
/// though the page cannot read the answer.
async fn origin_guard(State(inner): State<Arc<Inner>>, request: Request, next: Next) -> Response {
    let mut origins = request.headers().get_all(header::ORIGIN).iter();
    if let Some(first) = origins.next() {
        let listed = origins.next().is_none() && first.to_str().is_ok_and(|text| inner.origins.allows(text));
        if !listed {
            tracing::warn!(
                origin = %loggable(first.to_str().unwrap_or("<non-ascii>")),
                "rejected request: Origin is not allowed"
            );
            return reject(
                StatusCode::FORBIDDEN,
                "invalid_request_error",
                "origin_not_allowed",
                "Cross-origin requests are not enabled for this origin.",
            );
        }
    }
    next.run(request).await
}

/// Require `Authorization: Bearer <token>` on every route except `GET /health`.
async fn bearer_guard(State(inner): State<Arc<Inner>>, request: Request, next: Next) -> Response {
    let Some(token) = inner.token.as_ref() else {
        return next.run(request).await;
    };

    let is_health = matches!(*request.method(), Method::GET | Method::HEAD) && request.uri().path() == "/health";
    if is_health || presented_token(&request).is_some_and(|presented| token.matches(presented)) {
        return next.run(request).await;
    }

    let mut response = reject(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "invalid_token",
        "A valid bearer token is required.",
    );
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"pcai\""),
    );
    response
}

fn presented_token(request: &Request) -> Option<&str> {
    let value = request.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind(text: &str) -> SocketAddr {
        text.parse().expect("socket address literal")
    }

    fn options(token: Option<&str>) -> SecurityOptions {
        SecurityOptions {
            auth_token: token.map(str::to_string),
            ..SecurityOptions::default()
        }
    }

    // Protects: the startup rule. Detects: allowing a network bind without a credential, or
    // counting a blank token as a credential. This is the unit-level kill for the mutant that
    // drops the loopback check.
    #[test]
    fn network_bind_requires_a_non_blank_token() {
        for address in [
            "0.0.0.0:80",
            "[::]:80",
            "192.168.1.20:80",
            "[2001:db8::1]:80",
            "[fe80::1%3]:80",
        ] {
            let hosts = SecurityOptions {
                allowed_hosts: vec!["inference.lan".to_string()],
                ..SecurityOptions::default()
            };
            for token in [None, Some(""), Some("   ")] {
                let candidate = SecurityOptions {
                    auth_token: token.map(str::to_string),
                    ..hosts.clone()
                };
                let outcome = SecurityPolicy::new(bind(address), &candidate);
                assert!(
                    matches!(outcome, Err(SecurityError::TokenRequired { .. })),
                    "{address} with token {token:?} must be refused, got {outcome:?}"
                );
            }
            let with_token = SecurityOptions {
                auth_token: Some("s3cret".to_string()),
                ..hosts
            };
            assert!(SecurityPolicy::new(bind(address), &with_token).is_ok(), "{address}");
        }
    }

    // Protects: loopback binds in every spelling start without a token. Detects: a validator that
    // is too strict, which would break the default setup.
    #[test]
    fn loopback_binds_need_no_token() {
        for address in ["127.0.0.1:80", "127.7.7.7:80", "[::1]:80", "[::ffff:127.0.0.1]:80"] {
            let policy =
                SecurityPolicy::new(bind(address), &options(None)).unwrap_or_else(|error| panic!("{address}: {error}"));
            assert!(!policy.auth_required());
        }
        assert!(SecurityPolicy::new(bind("127.0.0.1:80"), &options(Some("x")))
            .expect("token")
            .auth_required());
    }

    // Protects: wildcard binds, whose own address is never a valid Host. Detects: starting such a
    // server without telling clients what Host to use, and the wrong error order.
    #[test]
    fn wildcard_bind_requires_allowed_hosts() {
        let outcome = SecurityPolicy::new(bind("0.0.0.0:80"), &options(Some("s3cret")));
        assert!(
            matches!(outcome, Err(SecurityError::AllowedHostsRequired { .. })),
            "{outcome:?}"
        );
        let message = outcome.expect_err("refused").to_string();
        assert!(message.contains("allowed_hosts"), "{message}");
    }

    // Protects: the token never reaching a log line through Debug. Detects: deriving Debug on
    // the options or the policy.
    #[test]
    fn debug_output_never_contains_the_token() {
        let candidate = options(Some("s3cret-value"));
        assert!(!format!("{candidate:?}").contains("s3cret-value"));
        let policy = SecurityPolicy::new(bind("127.0.0.1:80"), &candidate).expect("policy");
        assert!(!format!("{policy:?}").contains("s3cret-value"));
    }

    // Protects: the error text an operator reads at startup. Detects: a message that does not say
    // what to do, or that echoes the token.
    #[test]
    fn token_required_message_names_the_remedy() {
        let message = SecurityPolicy::new(bind("192.168.1.20:80"), &options(None))
            .expect_err("refused")
            .to_string();
        assert!(
            message.contains("192.168.1.20:80") && message.contains("token"),
            "{message}"
        );
    }

    // Protects: the log-injection guard. Detects: logging a Host or Origin value raw, which would
    // let a request forge log lines with embedded newlines.
    #[test]
    fn loggable_escapes_and_truncates() {
        assert_eq!(loggable("a\nb"), "\"a\\nb\"");
        assert!(loggable(&"x".repeat(500)).len() < 110);
    }
}
