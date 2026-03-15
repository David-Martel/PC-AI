use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;

use crate::config::api_key;

/// Bearer-token authentication middleware.
///
/// The middleware is a no-op when no API key is configured (see [`api_key`]).
/// When an API key is present, every request whose path is not `/health` must
/// supply a matching `Authorization: Bearer <token>` header.  Any mismatch
/// returns `401 Unauthorized` with a JSON body:
///
/// ```json
/// {"error": "unauthorized"}
/// ```
///
/// # Examples
///
/// Register in an axum router:
///
/// ```rust,no_run
/// use axum::{middleware, Router, routing::get};
/// use rust_functiongemma_runtime::auth::bearer_auth;
///
/// let app = Router::new()
///     .route("/v1/models", get(|| async { "ok" }))
///     .layer(middleware::from_fn(bearer_auth));
/// ```
pub(crate) async fn bearer_auth(req: Request<Body>, next: Next) -> Response {
    // Fast path: no key configured -> pass through unconditionally.
    let expected = match api_key() {
        Some(k) => k,
        None => return next.run(req).await,
    };

    // The /health route is always public regardless of auth configuration.
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    // Extract and validate the Authorization header.
    let authorized = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|token| token == expected.as_str())
        .unwrap_or(false);

    if authorized {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response()
    }
}
