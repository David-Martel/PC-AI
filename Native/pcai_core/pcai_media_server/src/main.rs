//! HTTP server binary for Janus-Pro media inference.
//!
//! Exposes a small REST API for text-to-image generation backed by the
//! `pcai-media` pipeline crate.
//!
//! # Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `GET`  | `/health` | Server and model load status |
//! | `POST` | `/v1/images/generate` | Generate an image from a text prompt |
//!
//! # Usage
//!
//! ```text
//! pcai-media --model deepseek-ai/Janus-Pro-7B --port 8090 --device cuda:0
//! ```

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use clap::Parser;
use image::{codecs::png::PngEncoder, ImageEncoder};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use pcai_media::config::PipelineConfig;
use pcai_media::generate::GenerationPipeline;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// HTTP server for Janus-Pro media inference.
#[derive(Debug, Parser)]
#[command(name = "pcai-media", about = "Janus-Pro media inference HTTP server")]
struct Args {
    /// Model path or HuggingFace repo ID.
    #[arg(short = 'm', long = "model", default_value = "deepseek-ai/Janus-Pro-7B")]
    model: String,

    /// Server bind address.
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,

    /// Server port.
    #[arg(short = 'p', long = "port", default_value_t = 8090)]
    port: u16,

    /// Target device string, e.g. `cpu`, `cuda`, `cuda:0`.
    #[arg(short = 'd', long = "device", default_value = "cuda:0")]
    device: String,
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Shared state stored behind an `Arc<RwLock<…>>`.
struct AppState {
    /// Loaded pipeline, present only after successful model load.
    pipeline: Option<GenerationPipeline>,
    /// Configuration used to load (or to attempt loading) the pipeline.
    config: PipelineConfig,
}

type SharedState = Arc<RwLock<AppState>>;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/images/generate`.
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    /// Text prompt describing the image to generate.
    prompt: String,

    /// Classifier-Free Guidance scale (overrides pipeline default when set).
    #[serde(default)]
    cfg_scale: Option<f64>,

    /// Sampling temperature (overrides pipeline default when set).
    #[serde(default)]
    temperature: Option<f64>,
}

/// Response body for `POST /v1/images/generate`.
#[derive(Debug, Serialize)]
struct GenerateResponse {
    /// PNG-encoded image as a Base64 string.
    image_base64: String,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
}

/// Response body for `GET /health`.
#[derive(Debug, Serialize)]
struct HealthResponse {
    /// `"ready"` when the model is loaded, otherwise `"not_ready"`.
    status: &'static str,
    /// Model identifier from the configuration.
    model: String,
    /// Whether the generation pipeline has been successfully loaded.
    model_loaded: bool,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /health` — reports server and model load status.
async fn health(State(state): State<SharedState>) -> impl IntoResponse {
    let guard = state.read().await;
    let loaded = guard.pipeline.is_some();
    Json(HealthResponse {
        status: if loaded { "ready" } else { "not_ready" },
        model: guard.config.model.clone(),
        model_loaded: loaded,
    })
}

/// `POST /v1/images/generate` — run the Janus-Pro text-to-image pipeline.
///
/// Returns a PNG-encoded image as a Base64 string together with its dimensions.
async fn generate_image(
    State(state): State<SharedState>,
    Json(req): Json<GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Build an ephemeral config that merges per-request overrides.
    let base_config = {
        let guard = state.read().await;
        guard.config.clone()
    };

    let effective_config = PipelineConfig {
        guidance_scale: req.cfg_scale.unwrap_or(base_config.guidance_scale),
        temperature: req.temperature.unwrap_or(base_config.temperature),
        ..base_config
    };

    // Check that the model is loaded before taking a write lock.
    {
        let guard = state.read().await;
        if guard.pipeline.is_none() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "model not loaded".to_string(),
            ));
        }
        // Verify the effective config values look reasonable (non-negative).
        if effective_config.guidance_scale < 0.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "cfg_scale must be non-negative".to_string(),
            ));
        }
        if effective_config.temperature <= 0.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "temperature must be positive".to_string(),
            ));
        }
    }

    // Run generation.  Janus-Pro inference is CPU/GPU-bound, so we offload to
    // a blocking thread pool to avoid stalling the Tokio runtime.
    let image = {
        // Clone the prompt into the closure.
        let prompt = req.prompt.clone();

        // We need to move the pipeline out of the RwLock-protected state to
        // call `generate`, but the pipeline must live in shared state for
        // concurrent health checks.  We call `generate` while holding a read
        // guard so other readers remain unblocked; writers (there are none
        // at runtime) would simply wait for the generation to finish.
        let guard = state.read().await;
        let pipeline = guard
            .pipeline
            .as_ref()
            .expect("pipeline presence checked above");

        tokio::task::block_in_place(|| pipeline.generate(&prompt))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };

    // Encode the ImageBuffer to PNG bytes.
    let width = image.width();
    let height = image.height();
    let raw_pixels = image.into_raw();

    let png_bytes = encode_png(raw_pixels, width, height)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Base64-encode the PNG bytes.
    let image_base64 =
        base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    Ok(Json(GenerateResponse {
        image_base64,
        width,
        height,
    }))
}

// ---------------------------------------------------------------------------
// PNG encoding helper
// ---------------------------------------------------------------------------

/// Encode raw RGB u8 pixels into PNG bytes.
///
/// # Errors
///
/// Returns an error if the encoder fails to write the image data.
fn encode_png(raw: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    let encoder = PngEncoder::new(&mut buf);
    encoder
        .write_image(&raw, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise tracing subscriber, respecting `RUST_LOG` env var.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    tracing::info!(
        model = %args.model,
        host = %args.host,
        port = args.port,
        device = %args.device,
        "starting pcai-media server"
    );

    // Build pipeline configuration from CLI args.
    let config = PipelineConfig {
        model: args.model.clone(),
        device: args.device.clone(),
        ..PipelineConfig::default()
    };

    // Load the model eagerly at startup.
    let pipeline = match GenerationPipeline::load(config.clone()) {
        Ok(p) => {
            tracing::info!(model = %args.model, "model loaded successfully");
            Some(p)
        }
        Err(err) => {
            tracing::error!(
                error = %err,
                model = %args.model,
                "model load failed — server will start but /health will report not_ready"
            );
            None
        }
    };

    // Initialise shared application state.
    let state: SharedState = Arc::new(RwLock::new(AppState { pipeline, config }));

    // Build the Axum router.
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/images/generate", post(generate_image))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "server listening");

    axum::serve(listener, app).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_png` must produce a non-empty byte buffer for a minimal 1x1 image.
    #[test]
    fn test_encode_png_minimal() {
        // 1×1 RGB image — 3 bytes of raw pixel data.
        let raw = vec![255u8, 0, 128];
        let result = encode_png(raw, 1, 1);
        assert!(result.is_ok(), "encode_png failed: {:?}", result.err());
        let bytes = result.unwrap();
        // PNG magic bytes: 0x89 0x50 0x4E 0x47
        assert!(bytes.len() > 8, "PNG output too short");
        assert_eq!(&bytes[1..4], b"PNG", "missing PNG signature");
    }

    /// `encode_png` on a 2×2 RGB image must produce valid PNG output.
    #[test]
    fn test_encode_png_2x2() {
        // 2×2 RGB image — 12 bytes.
        let raw: Vec<u8> = (0..12).collect();
        let result = encode_png(raw, 2, 2);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(bytes.len() > 8);
    }

    /// `HealthResponse` must serialise with the expected fields.
    #[test]
    fn test_health_response_serde() {
        let resp = HealthResponse {
            status: "ready",
            model: "deepseek-ai/Janus-Pro-7B".to_string(),
            model_loaded: true,
        };
        let json = serde_json::to_string(&resp).expect("serialise failed");
        assert!(json.contains("\"status\":\"ready\""));
        assert!(json.contains("\"model_loaded\":true"));
    }

    /// `GenerateRequest` must deserialise with all optional fields absent.
    #[test]
    fn test_generate_request_defaults() {
        let json = r#"{"prompt":"a cat"}"#;
        let req: GenerateRequest = serde_json::from_str(json).expect("deserialise failed");
        assert_eq!(req.prompt, "a cat");
        assert!(req.cfg_scale.is_none());
        assert!(req.temperature.is_none());
    }

    /// `GenerateRequest` must deserialise with all optional fields present.
    #[test]
    fn test_generate_request_full() {
        let json = r#"{"prompt":"a dog","cfg_scale":7.5,"temperature":0.9}"#;
        let req: GenerateRequest = serde_json::from_str(json).expect("deserialise failed");
        assert_eq!(req.prompt, "a dog");
        assert_eq!(req.cfg_scale, Some(7.5));
        assert_eq!(req.temperature, Some(0.9));
    }

    /// `GenerateResponse` must round-trip through JSON.
    #[test]
    fn test_generate_response_roundtrip() {
        let resp = GenerateResponse {
            image_base64: "abc123".to_string(),
            width: 384,
            height: 384,
        };
        let json = serde_json::to_string(&resp).expect("serialise");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded["image_base64"], "abc123");
        assert_eq!(decoded["width"], 384);
        assert_eq!(decoded["height"], 384);
    }
}
