//! HTTP server binary for Janus-Pro media inference.
//!
//! Exposes a small REST API for text-to-image generation and image
//! understanding backed by the `pcai-media` pipeline crate.
//!
//! # Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `GET`  | `/health` | Server and model load status |
//! | `GET`  | `/v1/models` | List loaded models |
//! | `POST` | `/v1/images/generate` | Generate an image from a text prompt |
//! | `POST` | `/v1/images/understand` | Describe / answer questions about an image |
//!
//! # Usage
//!
//! ```text
//! pcai-media --model deepseek-ai/Janus-Pro-1B --port 8090 --device cuda:0
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
use mimalloc::MiMalloc;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use pcai_media::config::PipelineConfig;
use pcai_media::generate::GenerationPipeline;
use pcai_media::understand::UnderstandingPipeline;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// HTTP server for Janus-Pro media inference.
#[derive(Debug, Parser)]
#[command(name = "pcai-media", about = "Janus-Pro media inference HTTP server")]
struct Args {
    /// Model path or HuggingFace repo ID.
    #[arg(short = 'm', long = "model", default_value = "deepseek-ai/Janus-Pro-1B")]
    model: String,

    /// Server bind address.
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,

    /// Server port.
    #[arg(short = 'p', long = "port", default_value_t = 8090)]
    port: u16,

    /// Target device string, e.g. `cpu`, `cuda`, `cuda:auto`, `cuda:0`.
    #[arg(short = 'd', long = "device", default_value = "cuda:auto")]
    device: String,

    /// Enable self-speculative decoding (draft 8/24 layers + verify K tokens).
    #[arg(long = "speculative", default_value_t = false)]
    speculative: bool,

    /// Number of draft tokens per speculative step (K).
    #[arg(long = "spec-lookahead", default_value_t = 4)]
    spec_lookahead: usize,

    /// Path to GGUF quantized weights for the LLM backbone.
    /// When set, loads quantized weights from this file instead of full-precision safetensors.
    #[arg(long = "gguf")]
    gguf: Option<String>,
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
    /// One active inference job bounds per-request model/cache allocations.
    inference_admission: Arc<Semaphore>,
}

type SharedState = Arc<RwLock<AppState>>;

async fn acquire_inference(state: &SharedState) -> Result<OwnedSemaphorePermit, (StatusCode, String)> {
    let admission = state.read().await.inference_admission.clone();
    admission.acquire_owned().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference admission closed".to_string(),
        )
    })
}

fn generate_request_image(
    request: &GenerateRequest,
    generate: impl FnOnce(&str, Option<f64>, Option<f64>) -> Result<image::RgbImage>,
) -> Result<image::RgbImage> {
    generate(&request.prompt, request.cfg_scale, request.temperature)
}

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
    /// Whether both trained native vision and understanding aligner are loaded.
    native_understanding_available: bool,
}

/// Request body for `POST /v1/images/understand`.
#[derive(Debug, Deserialize)]
struct UnderstandRequest {
    /// Base64-encoded image bytes (any format supported by the `image` crate).
    image_base64: String,

    /// User question or instruction about the image.
    prompt: String,

    /// Maximum number of new tokens to generate (default: 512).
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,

    /// Sampling temperature (default: 0.7).  Values ≤ 0.01 use greedy argmax.
    #[serde(default = "default_understand_temperature")]
    temperature: f64,
}

fn default_max_tokens() -> u32 {
    512
}

fn default_understand_temperature() -> f64 {
    0.7
}

/// Response body for `POST /v1/images/understand`.
#[derive(Debug, Serialize)]
struct UnderstandResponse {
    /// Generated text description or answer.
    text: String,
}

/// A single model entry returned by `GET /v1/models`.
#[derive(Debug, Serialize)]
struct ModelInfo {
    /// Model identifier (HuggingFace repo ID or local path).
    id: String,
    /// Whether the pipeline has been successfully loaded and is ready.
    loaded: bool,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Serialize)]
struct ModelsResponse {
    /// List of models known to this server instance.
    models: Vec<ModelInfo>,
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
        native_understanding_available: guard
            .pipeline
            .as_ref()
            .is_some_and(|pipeline| pipeline.vision_tower().is_some()),
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

    // Validate parameters and model availability before inference admission.
    {
        let guard = state.read().await;
        // Verify the effective config values look reasonable (non-negative).
        if !effective_config.guidance_scale.is_finite() || effective_config.guidance_scale < 0.0 {
            return Err((StatusCode::BAD_REQUEST, "cfg_scale must be non-negative".to_string()));
        }
        if !effective_config.temperature.is_finite() || effective_config.temperature <= 0.0 {
            return Err((StatusCode::BAD_REQUEST, "temperature must be positive".to_string()));
        }
        if guard.pipeline.is_none() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "model not loaded".to_string()));
        }
    }

    let inference_permit = acquire_inference(&state).await?;

    // Run generation.  Janus-Pro inference is CPU/GPU-bound, so we offload to
    // a blocking thread pool to avoid stalling the Tokio runtime.
    let image = {
        // A read guard preserves responsive health checks while the separate
        // admission permit serializes generation and understanding.
        let guard = state.read().await;
        let pipeline = guard.pipeline.as_ref().expect("pipeline presence checked above");

        tokio::task::block_in_place(|| {
            generate_request_image(&req, |prompt, cfg_scale, temperature| {
                pipeline.generate_with_overrides(prompt, cfg_scale, temperature)
            })
        })
        .map_err(|e| generation_error_response(&e))?
    };
    drop(inference_permit);

    // Encode the ImageBuffer to PNG bytes.
    let width = image.width();
    let height = image.height();
    let raw_pixels = image.into_raw();

    let png_bytes =
        encode_png(raw_pixels, width, height).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Base64-encode the PNG bytes.
    let image_base64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    Ok(Json(GenerateResponse {
        image_base64,
        width,
        height,
    }))
}

/// Return the HTTP status and complete generation diagnostic.
fn generation_error_response(error: &anyhow::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
}

/// `POST /v1/images/understand` — run the Janus-Pro image-understanding pipeline.
///
/// Decodes the Base64-encoded image, preprocesses it, and generates a text
/// response conditioned on the user's prompt.  Returns the generated text.
async fn understand_image(
    State(state): State<SharedState>,
    Json(req): Json<UnderstandRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Validate inputs before acquiring any lock.
    if !req.temperature.is_finite() || req.temperature <= 0.0 || req.temperature > f64::from(f32::MAX) {
        return Err((StatusCode::BAD_REQUEST, "temperature must be positive".to_string()));
    }
    if req.max_tokens == 0 {
        return Err((StatusCode::BAD_REQUEST, "max_tokens must be at least 1".to_string()));
    }

    // Decode Base64 → raw bytes.
    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(&req.image_base64)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("base64 decode failed: {e}")))?;

    // Check pipeline is loaded.
    {
        let guard = state.read().await;
        if guard.pipeline.is_none() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "model not loaded".to_string()));
        }
    }

    let inference_permit = acquire_inference(&state).await?;

    // Decode only after admission so queued requests cannot each allocate a
    // decompressed image alongside the active model/cache allocations.
    let dynamic_image = image::load_from_memory(&image_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("image decode failed: {e}")))?;

    // Run understanding.  Borrow model/tokenizer/device/dtype from the pipeline
    // while holding a read lock — other readers remain unblocked.
    let text = {
        let prompt = req.prompt.clone();
        let max_tokens = req.max_tokens;
        let temperature = req.temperature as f32;

        let guard = state.read().await;
        let pipeline = guard.pipeline.as_ref().expect("pipeline presence checked above");

        tokio::task::block_in_place(|| {
            UnderstandingPipeline::understand_with_fallback(
                pipeline,
                &dynamic_image,
                None,
                &prompt,
                max_tokens,
                temperature,
            )
        })
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    drop(inference_permit);

    Ok(Json(UnderstandResponse { text }))
}

/// `GET /v1/models` — list models known to this server instance.
///
/// Returns the configured model name and whether the pipeline has been
/// successfully loaded.
async fn list_models(State(state): State<SharedState>) -> impl IntoResponse {
    let guard = state.read().await;
    let loaded = guard.pipeline.is_some();
    Json(ModelsResponse {
        models: vec![ModelInfo {
            id: guard.config.model.clone(),
            loaded,
        }],
    })
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
        use_prealloc_kv_cache: true,
        use_speculative_decoding: args.speculative,
        speculative_lookahead: args.spec_lookahead,
        gguf_path: args.gguf.clone(),
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
    let state: SharedState = Arc::new(RwLock::new(AppState {
        pipeline,
        config,
        inference_admission: Arc::new(Semaphore::new(1)),
    }));

    // Build the Axum router.
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/images/generate", post(generate_image))
        .route("/v1/images/understand", post(understand_image))
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

    fn unloaded_state() -> SharedState {
        Arc::new(RwLock::new(AppState {
            pipeline: None,
            config: PipelineConfig::default(),
            inference_admission: Arc::new(Semaphore::new(1)),
        }))
    }

    #[test]
    fn test_generate_boundary_forwards_request_overrides() {
        for (cfg_scale, temperature) in [(Some(7.5), Some(0.9)), (None, None)] {
            let request = GenerateRequest {
                prompt: "a dog".to_string(),
                cfg_scale,
                temperature,
            };
            let image = generate_request_image(&request, |prompt, actual_cfg, actual_temperature| {
                assert_eq!(prompt, "a dog");
                assert_eq!(actual_cfg, cfg_scale);
                assert_eq!(actual_temperature, temperature);
                Ok(image::RgbImage::new(1, 1))
            })
            .unwrap();
            assert_eq!(image.dimensions(), (1, 1));
        }
    }

    #[tokio::test]
    async fn test_shared_inference_admission_preserves_health_and_releases() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};
        let state = unloaded_state();
        let generation = acquire_inference(&state).await.unwrap();
        let mut understanding = std::pin::pin!(acquire_inference(&state));
        assert!(matches!(
            understanding.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), health(State(state.clone())))
            .await
            .expect("health must remain responsive during inference and queued admission")
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let health: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(health["native_understanding_available"], false);
        drop(generation);
        let understanding = understanding.await.unwrap();
        assert_eq!(state.read().await.inference_admission.available_permits(), 0);
        drop(understanding);
        assert_eq!(state.read().await.inference_admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_inference_admission_released_on_generation_error() {
        async fn failing_inference(state: &SharedState) -> Result<(), (StatusCode, String)> {
            let _permit = acquire_inference(state).await?;
            let request = GenerateRequest {
                prompt: "a dog".to_string(),
                cfg_scale: None,
                temperature: None,
            };
            generate_request_image(&request, |_, _, _| Err(anyhow::anyhow!("injected inference failure")))
                .map_err(|error| generation_error_response(&error))?;
            Ok(())
        }
        let state = unloaded_state();
        let (status, message) = failing_inference(&state).await.unwrap_err();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(message.contains("injected inference failure"));
        assert!(state
            .read()
            .await
            .inference_admission
            .clone()
            .try_acquire_owned()
            .is_ok());
    }

    #[tokio::test]
    async fn test_invalid_requests_fail_before_inference_admission() {
        let state = unloaded_state();
        let _active_inference = acquire_inference(&state).await.unwrap();
        let request = GenerateRequest {
            prompt: "a dog".to_string(),
            cfg_scale: Some(-1.0),
            temperature: None,
        };
        let generation = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            generate_image(State(state.clone()), Json(request)),
        )
        .await
        .expect("invalid generation must not wait for admission");
        assert!(matches!(generation, Err((StatusCode::BAD_REQUEST, _))));
        let request = UnderstandRequest {
            image_base64: String::new(),
            prompt: "describe".to_string(),
            max_tokens: 0,
            temperature: 0.7,
        };
        let understanding = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            understand_image(State(state.clone()), Json(request)),
        )
        .await
        .expect("invalid understanding must not wait for admission");
        assert!(matches!(understanding, Err((StatusCode::BAD_REQUEST, _))));
    }

    #[tokio::test]
    async fn test_generation_error_response_preserves_complete_chain() {
        let error = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid image token",
        ))
        .context("step 7: projection failed")
        .context("generate_inner failed");
        let (status, message) = generation_error_response(&error);
        let expected = "generate_inner failed: step 7: projection failed: invalid image token";
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(message, expected);
        let response = (status, message).into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("error response body should be readable");
        assert_eq!(body.as_ref(), expected.as_bytes());
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .expect("typed cause must be retained")
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }

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
            native_understanding_available: false,
        };
        let json = serde_json::to_string(&resp).expect("serialise failed");
        assert!(json.contains("\"status\":\"ready\""));
        assert!(json.contains("\"model_loaded\":true"));
        assert!(json.contains("\"native_understanding_available\":false"));
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

    /// `UnderstandRequest` must deserialise with optional fields absent, using defaults.
    #[test]
    fn test_understand_request_defaults() {
        let json = r#"{"image_base64":"aGVsbG8=","prompt":"what is this?"}"#;
        let req: UnderstandRequest = serde_json::from_str(json).expect("deserialise failed");
        assert_eq!(req.image_base64, "aGVsbG8=");
        assert_eq!(req.prompt, "what is this?");
        assert_eq!(req.max_tokens, 512);
        assert!((req.temperature - 0.7).abs() < f64::EPSILON);
    }

    /// `UnderstandRequest` must deserialise with all optional fields present.
    #[test]
    fn test_understand_request_full() {
        let json = r#"{
            "image_base64": "aGVsbG8=",
            "prompt": "describe in detail",
            "max_tokens": 256,
            "temperature": 0.3
        }"#;
        let req: UnderstandRequest = serde_json::from_str(json).expect("deserialise failed");
        assert_eq!(req.prompt, "describe in detail");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 0.3).abs() < f64::EPSILON);
    }

    /// `UnderstandResponse` must serialise with the expected `text` field.
    #[test]
    fn test_understand_response_serde() {
        let resp = UnderstandResponse {
            text: "a fluffy cat sitting on a mat".to_string(),
        };
        let json = serde_json::to_string(&resp).expect("serialise failed");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise failed");
        assert_eq!(decoded["text"], "a fluffy cat sitting on a mat");
    }

    /// `UnderstandResponse` must round-trip through JSON without data loss.
    #[test]
    fn test_understand_response_roundtrip() {
        let original = UnderstandResponse {
            text: "the image shows a sunset over the ocean".to_string(),
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded["text"], original.text);
    }

    /// `ModelsResponse` must serialise with a `models` array containing a
    /// single entry with the expected fields.
    #[test]
    fn test_models_response_serde() {
        let resp = ModelsResponse {
            models: vec![ModelInfo {
                id: "deepseek-ai/Janus-Pro-1B".to_string(),
                loaded: true,
            }],
        };
        let json = serde_json::to_string(&resp).expect("serialise failed");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise failed");
        assert!(decoded["models"].is_array());
        assert_eq!(decoded["models"].as_array().unwrap().len(), 1);
        assert_eq!(decoded["models"][0]["id"], "deepseek-ai/Janus-Pro-1B");
        assert_eq!(decoded["models"][0]["loaded"], true);
    }

    /// `ModelsResponse` with `loaded: false` must serialise correctly.
    #[test]
    fn test_models_response_not_loaded() {
        let resp = ModelsResponse {
            models: vec![ModelInfo {
                id: "deepseek-ai/Janus-Pro-1B".to_string(),
                loaded: false,
            }],
        };
        let json = serde_json::to_string(&resp).expect("serialise failed");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise failed");
        assert_eq!(decoded["models"][0]["loaded"], false);
        assert_eq!(decoded["models"][0]["id"], "deepseek-ai/Janus-Pro-1B");
    }

    /// `ModelInfo` must round-trip through JSON preserving both fields.
    #[test]
    fn test_model_info_roundtrip() {
        let info = ModelInfo {
            id: "local/janus-1b".to_string(),
            loaded: true,
        };
        let json = serde_json::to_string(&info).expect("serialise");
        let decoded: serde_json::Value = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded["id"], "local/janus-1b");
        assert_eq!(decoded["loaded"], true);
    }
}
