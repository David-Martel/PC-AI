//! Janus-Pro media pipeline orchestration layer.
//!
//! This crate wraps [`pcai_media_model`] with a high-level pipeline API
//! covering model download, weight loading, text-to-image generation, and
//! (eventually) image-to-text understanding.
//!
//! # Feature flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `ffi` (default) | C ABI exports for P/Invoke integration |
//! | `server` | Optional axum HTTP server |
//! | `cuda` | GPU acceleration via CUDA |
//! | `flash-attn` | Flash-attention for the LLM backbone |
//!
//! # Quick example
//!
//! ```rust,no_run
//! use pcai_media::config::PipelineConfig;
//! use pcai_media::generate::GenerationPipeline;
//!
//! let config = PipelineConfig::default();
//! let pipeline = GenerationPipeline::load(config).expect("failed to load pipeline");
//! let image = pipeline.generate("a glowing circuit board").expect("generation failed");
//! image.save("output.png").expect("failed to save");
//! ```

pub mod config;
pub mod generate;
pub mod hub;
pub mod understand;

#[cfg(feature = "upscale")]
pub mod upscale;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use pcai_media_model as model;
