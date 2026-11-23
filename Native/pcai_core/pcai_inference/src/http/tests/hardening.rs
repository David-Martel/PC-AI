//! Browser-facing hardening of the pcai-inference HTTP server.
//!
//! The server listens on loopback, so the attackers that matter are web pages the user visits:
//! a page can send requests to `127.0.0.1` (CSRF), and an attacker domain can be re-pointed at
//! `127.0.0.1` after the page loaded (DNS rebinding), which makes the API same-origin. These
//! tests drive the real router and assert the three defences: a `Host` allow-list, a CORS
//! allow-list that is empty by default, and an optional bearer token.
//!
//! Each test names what it protects, what breaks it and what it needs, in the style of the
//! other reviewed test files in this repository.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode, Version},
    response::Response,
    Router,
};
use tower::ServiceExt;

use crate::http::{build_router, run_server, AppState, NotReadyReason};
use crate::{
    backends::{FinishReason, GenerateRequest, GenerateResponse, InferenceBackend},
    config::{RouterConfig, ServerConfig},
    Result,
};

/// Backend that answers instantly so the tests need no model, GPU or socket.
#[derive(Debug)]
struct EchoBackend;

#[async_trait::async_trait]
impl InferenceBackend for EchoBackend {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        Ok(())
    }

    async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
        Ok(GenerateResponse {
            text: "ok".to_string(),
            tokens_generated: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn unload_model(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "echo"
    }
}

const TOKEN: &str = "s3cret-token-for-tests";

fn state() -> Arc<AppState> {
    Arc::new(AppState::new(Box::new(EchoBackend)))
}

fn config(edit: impl FnOnce(&mut ServerConfig)) -> ServerConfig {
    let mut config = ServerConfig {
        port: 0,
        ..ServerConfig::default()
    };
    edit(&mut config);
    config
}

fn router(edit: impl FnOnce(&mut ServerConfig)) -> Router {
    build_router(state(), &config(edit)).expect("test: a loopback configuration must build a router")
}

/// GET `/v1/models` with the given `Host` header (`None` omits it).
fn models_request(host: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(Method::GET).uri("/v1/models");
    if let Some(host) = host {
        builder = builder.header(header::HOST, host);
    }
    builder.body(Body::empty()).expect("test: request must build")
}

fn with_header(host: &str, name: &str, value: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::HOST, host)
        .header(name, value)
        .body(Body::empty())
        .expect("test: request must build")
}

async fn send(app: &Router, request: Request<Body>) -> Response {
    app.clone()
        .oneshot(request)
        .await
        .expect("test: the router is infallible")
}

async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test: body must be readable");
    String::from_utf8_lossy(&bytes).into_owned()
}

fn allow_origin(response: &Response) -> Option<String> {
    response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .map(|value| value.to_str().expect("test: header is ASCII").to_string())
}

// ---------------------------------------------------------------------------
// Host header allow-list (DNS rebinding)
// ---------------------------------------------------------------------------

// Protects: a rebound attacker domain cannot reach the API. Detects: removing the Host check
// (the rebinding page sends `Host: evil.example`). Needs: the real router, no socket.
// Breadcrumb: pcai_http_guard::host_guard, http::build_router.
#[tokio::test]
async fn forged_host_header_is_refused_with_421() {
    let app = router(|_| {});
    let response = send(&app, models_request(Some("evil.example"))).await;
    assert_eq!(response.status(), StatusCode::MISDIRECTED_REQUEST);
    let body = body_text(response).await;
    assert!(body.contains("host_not_allowed"), "body: {body}");
}

// Protects: clients that use the documented loopback names, including the HVSocket proxy that
// forwards 18080 to the server port. Detects: a Host check that pins the port or drops IPv6.
// Needs: the real router.
#[tokio::test]
async fn loopback_hosts_are_accepted_on_any_port() {
    let app = router(|_| {});
    for host in [
        "127.0.0.1",
        "127.0.0.1:18080",
        "localhost:1",
        "LocalHost:8080",
        "localhost.:8080",
        "[::1]:9",
        "[::1]",
        "::1",
        "[0:0:0:0:0:0:0:1]:80",
    ] {
        let response = send(&app, models_request(Some(host))).await;
        assert_eq!(response.status(), StatusCode::OK, "Host {host:?} must be accepted");
    }
}

// Protects: look-alike hosts. Detects: suffix or prefix matching and lenient number parsing.
// Needs: the real router.
#[tokio::test]
async fn look_alike_hosts_are_refused() {
    let app = router(|_| {});
    for host in [
        "127.0.0.1.evil.example",
        "evil.example:80",
        "localhost.evil.example",
        "evil-localhost",
        "0.0.0.0",
        "0x7f.0.0.1",
        "2130706433",
        "user@127.0.0.1",
        "[::1]x",
        "127.0.0.1:",
        "127.0.0.1:99999",
        "",
    ] {
        let response = send(&app, models_request(Some(host))).await;
        assert!(
            response.status() == StatusCode::MISDIRECTED_REQUEST || response.status() == StatusCode::BAD_REQUEST,
            "Host {host:?} must be refused, got {}",
            response.status()
        );
    }
}

// Protects: the documented missing-Host policy. HTTP/1.1 requires Host, so its absence is a
// malformed request (400). HTTP/1.0 may omit it and browsers never speak it, so it is allowed.
// Detects: a guard that lets a Host-less request through, or one that breaks `curl --http1.0`.
#[tokio::test]
async fn missing_host_is_refused_unless_http_1_0() {
    let app = router(|_| {});

    let response = send(&app, models_request(None)).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let mut request = models_request(None);
    *request.version_mut() = Version::HTTP_10;
    let response = send(&app, request).await;
    assert_eq!(response.status(), StatusCode::OK);
}

// Protects: Host smuggling through a second header. Detects: checking only the first value.
#[tokio::test]
async fn duplicate_host_headers_are_refused() {
    let app = router(|_| {});
    let request = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::HOST, "127.0.0.1")
        .header(header::HOST, "evil.example")
        .body(Body::empty())
        .expect("test: request must build");
    let response = send(&app, request).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// Protects: HTTP/2 and absolute-form requests, which carry the authority in the URI and no Host
// header. Detects: trusting a clean Host header while the URI names another authority.
#[tokio::test]
async fn uri_authority_is_checked_as_well_as_the_host_header() {
    let app = router(|_| {});
    let request = Request::builder()
        .method(Method::GET)
        .uri("http://evil.example/v1/models")
        .header(header::HOST, "127.0.0.1")
        .body(Body::empty())
        .expect("test: request must build");
    assert_eq!(send(&app, request).await.status(), StatusCode::MISDIRECTED_REQUEST);

    let request = Request::builder()
        .method(Method::GET)
        .uri("http://127.0.0.1:8080/v1/models")
        .body(Body::empty())
        .expect("test: request must build");
    assert_eq!(send(&app, request).await.status(), StatusCode::OK);
}

// Protects: the configured extension point and the bind address. Detects: an allow-list that
// ignores `allowed_hosts` or does not add the real bind address.
#[tokio::test]
async fn configured_hosts_and_the_bind_address_are_accepted() {
    let app = router(|c| {
        c.host = "127.0.0.2".to_string();
        c.allowed_hosts = vec!["Inference.LAN".to_string()];
    });
    assert_eq!(
        send(&app, models_request(Some("127.0.0.2:5"))).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, models_request(Some("inference.lan:80"))).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, models_request(Some("127.0.0.1"))).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, models_request(Some("other.lan"))).await.status(),
        StatusCode::MISDIRECTED_REQUEST
    );
}

// Protects: wildcard binds. `0.0.0.0` is not a Host any client should use, and browsers have
// treated it as loopback, so it must never be added automatically. Detects: allowing the bind
// IP when it is the unspecified address.
#[tokio::test]
async fn wildcard_bind_does_not_allow_the_unspecified_host() {
    let app = router(|c| {
        c.host = "0.0.0.0".to_string();
        c.auth_token = Some(TOKEN.to_string());
        c.allowed_hosts = vec!["inference.lan".to_string()];
    });
    let auth = format!("Bearer {TOKEN}");
    let request = |host: &str| with_header(host, "authorization", &auth);
    assert_eq!(
        send(&app, request("0.0.0.0:8080")).await.status(),
        StatusCode::MISDIRECTED_REQUEST
    );
    assert_eq!(send(&app, request("inference.lan")).await.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// CORS and Origin
// ---------------------------------------------------------------------------

// Protects: CSRF from any web page. Detects: restoring `CorsLayer::permissive`, which answers
// every origin with `Access-Control-Allow-Origin: *` and lets the page read the response.
// Also pins that the legacy `"cors": true` key no longer switches the wildcard back on.
#[tokio::test]
async fn default_config_sends_no_cors_headers_for_a_foreign_origin() {
    for legacy in [None, Some(true)] {
        let app = router(|c| c.legacy_cors = legacy);
        let response = send(&app, with_header("127.0.0.1", "origin", "https://evil.example")).await;
        assert_eq!(allow_origin(&response), None, "legacy cors {legacy:?}");
        assert_eq!(response.status(), StatusCode::FORBIDDEN, "legacy cors {legacy:?}");
    }
}

// Protects: the "empty means nothing" rule from the other side. A permissive CORS layer adds
// `Access-Control-Allow-Origin: *` even to requests that carry no Origin, so this catches it
// where the Origin check cannot. Detects: restoring `CorsLayer::permissive`, or building a CORS
// layer from an empty list.
#[tokio::test]
async fn ordinary_responses_carry_no_cors_headers_by_default() {
    let app = router(|_| {});
    for host in ["127.0.0.1:8080", "localhost"] {
        let response = send(&app, models_request(Some(host))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let cors: Vec<&str> = response
            .headers()
            .keys()
            .map(|name| name.as_str())
            .filter(|name| name.starts_with("access-control-"))
            .collect();
        assert!(cors.is_empty(), "unexpected CORS headers: {cors:?}");
    }
}

// Protects: the preflight a browser sends before a JSON POST. Detects: a permissive preflight
// answer, which is what made the cross-origin POST possible.
#[tokio::test]
async fn preflight_is_refused_without_configuration() {
    let app = router(|_| {});
    let request = Request::builder()
        .method(Method::OPTIONS)
        .uri("/v1/chat/completions")
        .header(header::HOST, "127.0.0.1:8080")
        .header(header::ORIGIN, "https://evil.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type")
        .body(Body::empty())
        .expect("test: request must build");
    let response = send(&app, request).await;
    assert_eq!(allow_origin(&response), None);
    assert!(!response.status().is_success(), "status {}", response.status());
}

// Protects: the opt-in for a real browser client. Detects: an allow-list that does not echo the
// configured origin, echoes any origin, or lets a non-listed origin through the Origin check.
#[tokio::test]
async fn configured_origin_is_echoed_and_others_are_not() {
    let app = router(|c| c.cors_allowed_origins = vec!["https://tools.example:3000".to_string()]);

    let response = send(&app, with_header("127.0.0.1", "origin", "https://tools.example:3000")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(allow_origin(&response).as_deref(), Some("https://tools.example:3000"));

    let response = send(&app, with_header("127.0.0.1", "origin", "https://evil.example")).await;
    assert_eq!(allow_origin(&response), None);
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

// Protects: preflight for the configured origin, which must pass even when a token is set,
// because browsers send no credentials on a preflight. Detects: auth layered outside CORS (the
// preflight would get 401) and headers missing from the allow list.
#[tokio::test]
async fn preflight_for_a_configured_origin_succeeds_without_a_token() {
    let app = router(|c| {
        c.cors_allowed_origins = vec!["https://tools.example:3000".to_string()];
        c.auth_token = Some(TOKEN.to_string());
    });
    let preflight = |origin: &str| {
        Request::builder()
            .method(Method::OPTIONS)
            .uri("/v1/chat/completions")
            .header(header::HOST, "127.0.0.1:8080")
            .header(header::ORIGIN, origin)
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type,authorization")
            .body(Body::empty())
            .expect("test: request must build")
    };

    let response = send(&app, preflight("https://tools.example:3000")).await;
    assert!(response.status().is_success(), "status {}", response.status());
    assert_eq!(allow_origin(&response).as_deref(), Some("https://tools.example:3000"));
    let allowed_headers = response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_HEADERS)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    assert!(allowed_headers.contains("authorization"), "headers: {allowed_headers}");
    assert!(allowed_headers.contains("content-type"), "headers: {allowed_headers}");

    let response = send(&app, preflight("https://evil.example")).await;
    assert_eq!(allow_origin(&response), None);
    assert!(!response.status().is_success(), "status {}", response.status());
}

// Protects: the cross-site "simple request" path. A form post or `text/plain` fetch is sent
// without a preflight, so the only things stopping it are the Origin check and the JSON
// content-type requirement. This pins the second one (it already holds on main, so it is a pin
// and not a red/green test): the JSON extractor must not start accepting other media types.
#[tokio::test]
async fn text_plain_post_is_unsupported_media_type() {
    let app = router(|_| {});
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::HOST, "127.0.0.1:8080")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#))
        .expect("test: request must build");
    assert_eq!(send(&app, request).await.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

// ---------------------------------------------------------------------------
// Bearer token
// ---------------------------------------------------------------------------

fn authed_request(path: &str, authorization: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::GET)
        .uri(path)
        .header(header::HOST, "127.0.0.1:8080");
    if let Some(value) = authorization {
        builder = builder.header(header::AUTHORIZATION, value);
    }
    builder.body(Body::empty()).expect("test: request must build")
}

// Protects: every route but `/health` when a token is configured. Detects: skipping the token
// comparison, comparing only a prefix, accepting another scheme, or leaving a route open.
#[tokio::test]
async fn token_is_required_and_checked_on_protected_routes() {
    let app = router(|c| c.auth_token = Some(TOKEN.to_string()));
    let good = format!("Bearer {TOKEN}");
    let wrong = "Bearer not-the-token".to_string();
    let prefix = format!("Bearer {}", &TOKEN[..TOKEN.len() - 1]);
    let longer = format!("Bearer {TOKEN}x");
    let basic = format!("Basic {TOKEN}");

    for path in ["/v1/models", "/v1/completions", "/v1/chat/completions", "/nope"] {
        let missing = send(&app, authed_request(path, None)).await;
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED, "{path} without a token");
        assert!(missing.headers().contains_key(header::WWW_AUTHENTICATE));
        for bad in [&wrong, &prefix, &longer, &basic] {
            let response = send(&app, authed_request(path, Some(bad))).await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path} with {bad:?}");
        }
    }

    let response = send(&app, authed_request("/v1/models", Some(&good))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let lower = format!("bearer {TOKEN}");
    assert_eq!(
        send(&app, authed_request("/v1/models", Some(&lower))).await.status(),
        StatusCode::OK
    );
}

// Protects: readiness probes, which run before a client has a token. Detects: guarding
// `GET /health`, or opening other routes along with it.
#[tokio::test]
async fn health_stays_open_when_a_token_is_set() {
    let app = router(|c| c.auth_token = Some(TOKEN.to_string()));
    assert_eq!(
        send(&app, authed_request("/health", None)).await.status(),
        StatusCode::OK
    );
    let post = Request::builder()
        .method(Method::POST)
        .uri("/health")
        .header(header::HOST, "127.0.0.1:8080")
        .body(Body::empty())
        .expect("test: request must build");
    assert_eq!(send(&app, post).await.status(), StatusCode::UNAUTHORIZED);
}

// Protects: the token guard while the server is not ready. Detects: the readiness gate
// answering an unauthenticated caller with the failure text, or `/health` needing a token.
#[tokio::test]
async fn not_ready_server_still_demands_a_token_but_keeps_health_open() {
    let state = state();
    state.mark_not_ready(NotReadyReason::GenerationProbeFailed);
    let app = build_router(Arc::clone(&state), &config(|c| c.auth_token = Some(TOKEN.to_string())))
        .expect("test: a loopback configuration must build a router");
    let completion = |token: Option<&str>| {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header(header::HOST, "127.0.0.1:8080")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(token) = token {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        builder
            .body(Body::from(r#"{"prompt":"hi"}"#))
            .expect("test: request must build")
    };
    let anonymous = send(&app, completion(None)).await;
    assert_eq!(anonymous.status(), StatusCode::UNAUTHORIZED);
    assert!(!body_text(anonymous).await.contains("generation_probe_failed"));

    let authed = send(&app, completion(Some(TOKEN))).await;
    assert_eq!(authed.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(body_text(authed).await.contains("generation_probe_failed"));

    let health = send(&app, authed_request("/health", None)).await;
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_text(health).await;
    assert!(
        body.contains("not_ready") && body.contains("generation_probe_failed"),
        "{body}"
    );
}

// Protects: the default, which must not break the in-repo clients (they send no token).
// Detects: auth switched on by default.
#[tokio::test]
async fn no_token_configured_means_no_authentication() {
    let app = router(|_| {});
    assert_eq!(
        send(&app, authed_request("/v1/models", None)).await.status(),
        StatusCode::OK
    );
}

// ---------------------------------------------------------------------------
// Startup validation
// ---------------------------------------------------------------------------

// Protects: the rule that nothing is reachable from the network without a credential. Detects:
// allowing a non-loopback bind without a token, or treating a blank token as set.
#[test]
fn non_loopback_bind_without_a_token_is_refused() {
    for host in ["0.0.0.0", "::", "192.168.1.20", "[2001:db8::1]", "fe80::1%3"] {
        for token in [None, Some(String::new()), Some("   ".to_string())] {
            let outcome = build_router(
                state(),
                &config(|c| {
                    c.host = host.to_string();
                    c.auth_token = token.clone();
                    c.allowed_hosts = vec!["inference.lan".to_string()];
                }),
            );
            let error = outcome.expect_err(&format!("{host} with token {token:?} must refuse to start"));
            let message = error.to_string();
            assert!(
                message.contains("token"),
                "message must name the missing token: {message}"
            );
            assert!(!message.contains(TOKEN));
        }
    }
}

// Protects: the permitted combinations. Detects: a validator that is too strict (loopback in
// any spelling must start without a token; a token unlocks a wider bind).
#[test]
fn loopback_binds_and_token_protected_binds_start() {
    for host in [
        "127.0.0.1",
        "127.0.0.9",
        "localhost",
        "::1",
        "[::1]",
        "::ffff:127.0.0.1",
    ] {
        let _router = build_router(state(), &config(|c| c.host = host.to_string()))
            .unwrap_or_else(|error| panic!("{host} must start without a token: {error}"));
    }
    let _router = build_router(
        state(),
        &config(|c| {
            c.host = "0.0.0.0".to_string();
            c.auth_token = Some(TOKEN.to_string());
            c.allowed_hosts = vec!["inference.lan".to_string()];
        }),
    )
    .expect("test: a token must unlock a non-loopback bind");
}

// Protects: configuration that cannot be honoured. Detects: silently ignoring a bad origin,
// a wildcard origin or a wildcard host.
#[test]
fn invalid_cors_origins_and_hosts_are_rejected() {
    for origin in [
        "*",
        "evil.example",
        "https://",
        "https://a.example/path",
        "ftp://a.example",
        "null",
        "",
    ] {
        let outcome = build_router(state(), &config(|c| c.cors_allowed_origins = vec![origin.to_string()]));
        assert!(outcome.is_err(), "origin {origin:?} must be rejected");
    }
    for host in ["*", "", "has space", "a/b", "0.0.0.0", "::"] {
        let outcome = build_router(state(), &config(|c| c.allowed_hosts = vec![host.to_string()]));
        assert!(outcome.is_err(), "allowed host {host:?} must be rejected");
    }
}

// Protects: the token never leaves the process through logs or saved files. Detects: deriving
// Debug on the config, or serializing the token in `to_file`.
#[test]
fn token_is_redacted_in_debug_and_never_serialized() {
    let config = config(|c| c.auth_token = Some(TOKEN.to_string()));
    assert!(!format!("{config:?}").contains(TOKEN));
    assert!(!serde_json::to_string(&config)
        .expect("test: config must serialize")
        .contains(TOKEN));
    let parsed: ServerConfig = serde_json::from_str(&format!(r#"{{"auth_token":"{TOKEN}"}}"#)).expect("test: parse");
    assert_eq!(parsed.auth_token.as_deref(), Some(TOKEN));
}

// Protects: legacy configuration files (`"cors": true` is in three generators and a checked-in
// config). Detects: a schema change that stops them loading.
#[test]
fn legacy_cors_key_still_loads() {
    let parsed: ServerConfig = serde_json::from_str(r#"{"host":"127.0.0.1","port":8085,"cors":true}"#)
        .expect("test: the legacy cors key must still parse");
    assert_eq!(parsed.port, 8085);
    assert!(parsed.cors_allowed_origins.is_empty());
}

// ---------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------

// Protects: loud bind failures (FunctionGemma got the same fix in #408). Detects: a fallback to
// another port, or an error that does not name the address that failed.
#[tokio::test]
async fn occupied_address_fails_and_names_the_address() {
    let occupied = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("test: bind loopback");
    let addr = occupied.local_addr().expect("test: local addr");
    let outcome = run_server(
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: addr.port(),
            ..ServerConfig::default()
        },
        RouterConfig::default(),
        Box::new(EchoBackend),
    )
    .await;
    let message = outcome.expect_err("test: an occupied port must fail").to_string();
    assert!(
        message.contains(&addr.to_string()),
        "message must name {addr}: {message}"
    );
    drop(occupied);
}

// Protects: a bind host that is not an IP literal. Detects: handing it to the resolver.
#[tokio::test]
async fn unparsable_bind_host_fails_and_names_it() {
    let outcome = run_server(
        ServerConfig {
            host: "not a host".to_string(),
            port: 0,
            ..ServerConfig::default()
        },
        RouterConfig::default(),
        Box::new(EchoBackend),
    )
    .await;
    let message = outcome.expect_err("test: a bad host must fail").to_string();
    assert!(message.contains("not a host"), "message: {message}");
}

// ---------------------------------------------------------------------------
// End to end over a socket
// ---------------------------------------------------------------------------

async fn raw_exchange(addr: SocketAddr, request: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("test: connect");
    stream.write_all(request.as_bytes()).await.expect("test: write");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.expect("test: read");
    String::from_utf8_lossy(&response).into_owned()
}

// Protects: the whole stack as a browser-driven attacker sees it: a real listener on an
// ephemeral loopback port, hyper's own request parsing, forged Host and Origin headers, and a
// hand-written request with no Host at all. Detects: any layer that works only under `oneshot`.
// Needs: a loopback socket on 127.0.0.1:0, closed when the test ends.
#[tokio::test]
async fn real_listener_enforces_host_origin_and_token() {
    let app = router(|c| c.auth_token = Some(TOKEN.to_string()));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("test: bind");
    let addr = listener.local_addr().expect("test: local addr");
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let get = |host: &str, extra: &str| {
        format!("GET /v1/models HTTP/1.1\r\nHost: {host}\r\n{extra}Connection: close\r\n\r\n")
    };
    let auth = format!("Authorization: Bearer {TOKEN}\r\n");

    let forged = raw_exchange(addr, &get("evil.example", &auth)).await;
    assert!(forged.starts_with("HTTP/1.1 421"), "forged Host: {forged}");

    let rebinding_page = raw_exchange(
        addr,
        &get(
            &format!("127.0.0.1:{}", addr.port()),
            &format!("Origin: https://evil.example\r\n{auth}"),
        ),
    )
    .await;
    assert!(
        rebinding_page.starts_with("HTTP/1.1 403"),
        "foreign Origin: {rebinding_page}"
    );
    assert!(!rebinding_page
        .to_ascii_lowercase()
        .contains("access-control-allow-origin"));

    let no_token = raw_exchange(addr, &get(&format!("127.0.0.1:{}", addr.port()), "")).await;
    assert!(no_token.starts_with("HTTP/1.1 401"), "no token: {no_token}");

    let good = raw_exchange(addr, &get(&format!("127.0.0.1:{}", addr.port()), &auth)).await;
    assert!(good.starts_with("HTTP/1.1 200"), "good request: {good}");

    let no_host = raw_exchange(addr, "GET /v1/models HTTP/1.1\r\nConnection: close\r\n\r\n").await;
    assert!(!no_host.starts_with("HTTP/1.1 200"), "missing Host: {no_host}");

    let http10 = raw_exchange(addr, &format!("GET /v1/models HTTP/1.0\r\n{auth}\r\n")).await;
    assert!(
        http10.starts_with("HTTP/1.0 200") || http10.starts_with("HTTP/1.1 200"),
        "HTTP/1.0: {http10}"
    );

    server.abort();
}
