//! Browser-facing hardening shared by the local PC_AI HTTP servers.
//!
//! `pcai-inference` and `pcai-media-server` listen on loopback with no login, which leaves them
//! open to two attacks from any web page the user visits:
//!
//! - **CSRF**: a page can send requests to `http://127.0.0.1:<port>`. A permissive CORS layer
//!   (`Access-Control-Allow-Origin: *`) even lets it read the answers.
//! - **DNS rebinding**: an attacker's domain is re-pointed at `127.0.0.1` after the page loaded,
//!   which makes the API same-origin and bypasses CORS altogether.
//!
//! This crate holds the defences, so both servers enforce one set of rules:
//!
//! | Defence | Where | Default |
//! |---|---|---|
//! | `Host` allow-list (421 / 400) | [`SecurityPolicy::apply`] | loopback names plus the bind address |
//! | `Origin` allow-list and CORS | [`SecurityOptions::cors_allowed_origins`] | empty: no CORS, browser callers refused |
//! | Bearer token, constant-time | [`SecurityOptions::auth_token`] | off on loopback, **required** otherwise |
//! | Bind-address parsing, loud failure | [`parse_bind_addr`], [`bind_listener`] | n/a |
//!
//! # Examples
//!
//! ```
//! use axum::{routing::get, Router};
//! use pcai_http_guard::{parse_bind_addr, SecurityOptions, SecurityPolicy};
//!
//! let bind = parse_bind_addr("127.0.0.1", 8080)?;
//! let policy = SecurityPolicy::new(bind, &SecurityOptions::default())?;
//! let app: Router = policy.apply(Router::new().route("/health", get(|| async { "ok" })));
//! # let _ = app;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod auth;
mod bind;
mod host;
mod origin;
mod policy;

pub use auth::pick_token;
pub use bind::{bind_listener, is_loopback, parse_bind_addr, BindError};
pub use policy::{SecurityError, SecurityOptions, SecurityPolicy};
