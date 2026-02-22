use axum::{
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
    Router,
};
use std::fs;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

mod auth;
mod config;
mod error;
#[cfg(feature = "model")]
mod gpu;
mod handlers;
#[cfg(feature = "model")]
mod inference;
mod metrics;
#[cfg(feature = "model")]
mod model_support;
mod routing;
mod types;

pub use config::{init_runtime_config, runtime_addr};

use config::{build_log_filter, runtime_config};
use handlers::{chat, health, list_models, MAX_BODY_SIZE};
use tokio::sync::Semaphore;
use types::AppState;

pub fn build_router() -> Router {
    let cfg = runtime_config();
    let state = Arc::new(AppState {
        semaphore: Arc::new(Semaphore::new(cfg.router_queue_depth)),
        request_timeout: std::time::Duration::from_secs(cfg.router_request_timeout_secs),
    });
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .layer(middleware::from_fn(auth::bearer_auth))
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
        .with_state(state)
}

fn write_bound_addr(addr: SocketAddr) {
    let reports_dir = PathBuf::from("Reports");
    if let Err(err) = fs::create_dir_all(&reports_dir) {
        tracing::warn!("failed to create Reports directory: {err}");
        return;
    }
    let port_path = reports_dir.join("functiongemma-runtime.port");
    let addr_path = reports_dir.join("functiongemma-runtime.addr");
    let _ = fs::write(&port_path, addr.port().to_string());
    let _ = fs::write(&addr_path, addr.to_string());
}

async fn bind_listener(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => Ok(listener),
        Err(err) if err.kind() == io::ErrorKind::AddrInUse => {
            let fallback = std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0));
            tracing::warn!(
                "router_addr {} already in use; falling back to {}",
                addr,
                fallback
            );
            Ok(tokio::net::TcpListener::bind(fallback).await?)
        }
        Err(err) => Err(err.into()),
    }
}

pub async fn serve(addr: SocketAddr) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(build_log_filter())
        .init();

    let app = build_router();
    let listener = bind_listener(addr).await?;
    let bound_addr = listener.local_addr()?;
    write_bound_addr(bound_addr);
    tracing::info!("FunctionGemma runtime listening on {}", bound_addr);
    axum::serve(listener, app).await?;
    Ok(())
}
