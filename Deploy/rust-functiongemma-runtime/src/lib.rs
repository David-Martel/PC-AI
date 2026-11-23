use anyhow::Context;
use axum::{
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
    Router,
};
use std::fs;
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
    tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind FunctionGemma router to {addr}"))
}

pub async fn serve(addr: SocketAddr) -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(build_log_filter()).init();

    let app = build_router();
    let listener = bind_listener(addr).await?;
    let bound_addr = listener.local_addr()?;
    write_bound_addr(bound_addr);
    tracing::info!("FunctionGemma runtime listening on {}", bound_addr);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::bind_listener;
    use std::io;
    use std::net::{Ipv4Addr, SocketAddr};
    use tokio::net::TcpListener;

    // Protects: a fixed router binding. Detects: silently using a different port
    // after a bind conflict. Needs: a loopback socket held by this test.
    // Breadcrumb: bind_listener and serve.
    #[tokio::test]
    async fn occupied_router_address_returns_bind_error() -> anyhow::Result<()> {
        let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let addr = occupied.local_addr()?;
        let Err(error) = bind_listener(addr).await else {
            anyhow::bail!("router unexpectedly bound while {addr} was occupied");
        };

        assert!(error.to_string().contains(&addr.to_string()));
        assert_eq!(
            error.downcast_ref::<io::Error>().map(io::Error::kind),
            Some(io::ErrorKind::AddrInUse)
        );
        drop(occupied);
        Ok(())
    }

    // Protects: an explicit ephemeral router binding. Detects: rejecting port
    // zero or binding outside loopback. Needs: an OS-assigned loopback port.
    // Breadcrumb: bind_listener and config::runtime_addr.
    #[tokio::test]
    async fn explicit_ephemeral_router_address_binds_loopback() -> anyhow::Result<()> {
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let listener = bind_listener(addr).await?;
        let bound = listener.local_addr()?;

        assert_eq!(bound.ip(), addr.ip());
        assert_ne!(bound.port(), 0);
        Ok(())
    }
}
