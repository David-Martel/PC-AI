//! Browser-facing hardening of the pcai-media HTTP server.
//!
//! Same threat model and same defences as `pcai-inference` (see its `http::hardening`):
//! a `Host` allow-list against DNS rebinding, an empty-by-default CORS allow-list plus an Origin
//! check against cross-site requests, and an optional bearer token. The router is built with no
//! model loaded, so nothing here needs a GPU or weights.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
    response::Response,
    Router,
};
use pcai_media::config::PipelineConfig;
use tokio::sync::RwLock;
use tower::ServiceExt;

use crate::{build_router, AppState, ServeSecurity};

const TOKEN: &str = "s3cret-token-for-tests";

fn security(edit: impl FnOnce(&mut ServeSecurity)) -> ServeSecurity {
    let mut security = ServeSecurity {
        host: "127.0.0.1".to_string(),
        port: 0,
        allowed_hosts: Vec::new(),
        cors_allowed_origins: Vec::new(),
        auth_token: None,
    };
    edit(&mut security);
    security
}

fn try_router(edit: impl FnOnce(&mut ServeSecurity)) -> anyhow::Result<Router> {
    let state = Arc::new(RwLock::new(AppState {
        pipeline: None,
        config: PipelineConfig::default(),
    }));
    build_router(state, &security(edit))
}

fn router(edit: impl FnOnce(&mut ServeSecurity)) -> Router {
    try_router(edit).expect("test: a loopback configuration must build a router")
}

fn get(path: &str, host: Option<&str>, extra: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder().method(Method::GET).uri(path);
    if let Some(host) = host {
        builder = builder.header(header::HOST, host);
    }
    for (name, value) in extra {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::empty()).expect("test: request must build")
}

async fn send(app: &Router, request: Request<Body>) -> Response {
    app.clone()
        .oneshot(request)
        .await
        .expect("test: the router is infallible")
}

fn allow_origin(response: &Response) -> Option<String> {
    response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .map(|value| value.to_str().expect("test: header is ASCII").to_string())
}

// Protects: a rebound attacker domain cannot reach the media API. Detects: removing the Host
// check. Needs: the real router with no model loaded.
#[tokio::test]
async fn forged_host_header_is_refused_with_421() {
    let app = router(|_| {});
    let response = send(&app, get("/v1/models", Some("evil.example"), &[])).await;
    assert_eq!(response.status(), StatusCode::MISDIRECTED_REQUEST);
}

// Protects: the documented loopback spellings and the missing-Host rule (400, because HTTP/1.1
// requires the header). Detects: a Host check that pins the port or drops IPv6.
#[tokio::test]
async fn loopback_hosts_are_accepted_and_missing_host_is_refused() {
    let app = router(|_| {});
    for host in ["127.0.0.1:8090", "localhost:1", "[::1]:9", "::1"] {
        let response = send(&app, get("/v1/models", Some(host), &[])).await;
        assert_eq!(response.status(), StatusCode::OK, "Host {host:?}");
    }
    assert_eq!(
        send(&app, get("/v1/models", None, &[])).await.status(),
        StatusCode::BAD_REQUEST
    );
}

// Protects: the configured extension point. Detects: ignoring `--allowed-host`.
#[tokio::test]
async fn configured_host_is_accepted() {
    let app = router(|s| s.allowed_hosts = vec!["media.lan".to_string()]);
    assert_eq!(
        send(&app, get("/v1/models", Some("media.lan:80"), &[])).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, get("/v1/models", Some("other.lan"), &[])).await.status(),
        StatusCode::MISDIRECTED_REQUEST
    );
}

// Protects: CSRF from any web page. Detects: restoring `CorsLayer::permissive`.
#[tokio::test]
async fn default_config_sends_no_cors_headers_for_a_foreign_origin() {
    let app = router(|_| {});
    let response = send(
        &app,
        get("/v1/models", Some("127.0.0.1"), &[("origin", "https://evil.example")]),
    )
    .await;
    assert_eq!(allow_origin(&response), None);
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    let preflight = Request::builder()
        .method(Method::OPTIONS)
        .uri("/v1/images/generate")
        .header(header::HOST, "127.0.0.1:8090")
        .header(header::ORIGIN, "https://evil.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .body(Body::empty())
        .expect("test: request must build");
    let response = send(&app, preflight).await;
    assert_eq!(allow_origin(&response), None);
    assert!(!response.status().is_success());
}

// Protects: the "empty means nothing" rule. A permissive CORS layer adds
// `Access-Control-Allow-Origin: *` even to requests that carry no Origin. Detects: restoring
// `CorsLayer::permissive`, or building a CORS layer from an empty list.
#[tokio::test]
async fn ordinary_responses_carry_no_cors_headers_by_default() {
    let app = router(|_| {});
    let response = send(&app, get("/v1/models", Some("127.0.0.1:8090"), &[])).await;
    assert_eq!(response.status(), StatusCode::OK);
    let cors: Vec<&str> = response
        .headers()
        .keys()
        .map(|name| name.as_str())
        .filter(|name| name.starts_with("access-control-"))
        .collect();
    assert!(cors.is_empty(), "unexpected CORS headers: {cors:?}");
}

// Protects: the opt-in for a real browser client, including a preflight that carries no token.
// Detects: an allow-list that does not echo the configured origin, echoes others, or puts auth
// outside CORS.
#[tokio::test]
async fn configured_origin_is_echoed_including_preflight() {
    let app = router(|s| {
        s.cors_allowed_origins = vec!["https://tools.example:3000".to_string()];
        s.auth_token = Some(TOKEN.to_string());
    });
    let preflight = Request::builder()
        .method(Method::OPTIONS)
        .uri("/v1/images/generate")
        .header(header::HOST, "127.0.0.1:8090")
        .header(header::ORIGIN, "https://tools.example:3000")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type,authorization")
        .body(Body::empty())
        .expect("test: request must build");
    let response = send(&app, preflight).await;
    assert!(response.status().is_success(), "status {}", response.status());
    assert_eq!(allow_origin(&response).as_deref(), Some("https://tools.example:3000"));

    let response = send(
        &app,
        get("/v1/models", Some("127.0.0.1"), &[("origin", "https://evil.example")]),
    )
    .await;
    assert_eq!(allow_origin(&response), None);
}

// Protects: every route but `/health` when a token is set. Detects: skipping the comparison,
// guarding `/health`, or leaving the generation routes open.
#[tokio::test]
async fn token_is_required_and_checked_but_health_stays_open() {
    let app = router(|s| s.auth_token = Some(TOKEN.to_string()));
    let bearer = format!("Bearer {TOKEN}");
    let wrong = "Bearer not-the-token";

    assert_eq!(
        send(&app, get("/health", Some("127.0.0.1"), &[])).await.status(),
        StatusCode::OK
    );
    for path in ["/v1/models", "/v1/images/generate", "/v1/images/understand"] {
        let missing = send(&app, get(path, Some("127.0.0.1"), &[])).await;
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED, "{path}");
        let bad = send(&app, get(path, Some("127.0.0.1"), &[("authorization", wrong)])).await;
        assert_eq!(bad.status(), StatusCode::UNAUTHORIZED, "{path}");
    }
    let good = send(
        &app,
        get("/v1/models", Some("127.0.0.1"), &[("authorization", &bearer)]),
    )
    .await;
    assert_eq!(good.status(), StatusCode::OK);
}

// Protects: the default, so existing clients (which send no token) keep working. Detects: auth
// switched on by default.
#[tokio::test]
async fn no_token_configured_means_no_authentication() {
    let app = router(|_| {});
    assert_eq!(
        send(&app, get("/v1/models", Some("127.0.0.1"), &[])).await.status(),
        StatusCode::OK
    );
}

// Protects: the rule that nothing is reachable from the network without a credential. Detects:
// allowing a non-loopback bind without a token, or treating a blank token as set.
#[test]
fn non_loopback_bind_without_a_token_is_refused() {
    for host in ["0.0.0.0", "::", "192.168.1.20", "fe80::1%3"] {
        for token in [None, Some(String::new()), Some("  ".to_string())] {
            let outcome = try_router(|s| {
                s.host = host.to_string();
                s.auth_token = token.clone();
                s.allowed_hosts = vec!["media.lan".to_string()];
            });
            let error = outcome
                .err()
                .unwrap_or_else(|| panic!("{host} with {token:?} must refuse to start"));
            assert!(error.to_string().contains("token"), "message: {error}");
        }
    }
    let _router = try_router(|s| {
        s.host = "0.0.0.0".to_string();
        s.auth_token = Some(TOKEN.to_string());
        s.allowed_hosts = vec!["media.lan".to_string()];
    })
    .expect("test: a token must unlock a non-loopback bind");
}

// Protects: the whole stack as an attacker sees it: a real listener on an ephemeral loopback
// port, hyper's request parsing, forged Host and Origin. Detects: a layer that only works under
// `oneshot`. Needs: a loopback socket on 127.0.0.1:0, closed when the test ends.
#[tokio::test]
async fn real_listener_enforces_host_origin_and_token() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let app = router(|s| s.auth_token = Some(TOKEN.to_string()));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("test: bind");
    let addr = listener.local_addr().expect("test: local addr");
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let exchange = |request: String| async move {
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("test: connect");
        stream.write_all(request.as_bytes()).await.expect("test: write");
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.expect("test: read");
        String::from_utf8_lossy(&response).into_owned()
    };
    let auth = format!("Authorization: Bearer {TOKEN}\r\n");
    let request = |host: &str, extra: &str| {
        format!("GET /v1/models HTTP/1.1\r\nHost: {host}\r\n{extra}Connection: close\r\n\r\n")
    };

    let forged = exchange(request("evil.example", &auth)).await;
    assert!(forged.starts_with("HTTP/1.1 421"), "forged Host: {forged}");
    let foreign = exchange(request(
        &addr.to_string(),
        &format!("Origin: https://evil.example\r\n{auth}"),
    ))
    .await;
    assert!(foreign.starts_with("HTTP/1.1 403"), "foreign Origin: {foreign}");
    let anonymous = exchange(request(&addr.to_string(), "")).await;
    assert!(anonymous.starts_with("HTTP/1.1 401"), "no token: {anonymous}");
    let good = exchange(request(&addr.to_string(), &auth)).await;
    assert!(good.starts_with("HTTP/1.1 200"), "good request: {good}");
    let no_host = exchange("GET /v1/models HTTP/1.1\r\nConnection: close\r\n\r\n".to_string()).await;
    assert!(!no_host.starts_with("HTTP/1.1 200"), "missing Host: {no_host}");

    server.abort();
}
