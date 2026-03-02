//! Janus-Pro model architecture for the PC_AI media agent.
//!
//! This crate provides:
//! - [`config`]: [`JanusConfig`] with 1B and 7B presets, and conversion to
//!   `candle_transformers` Llama config.
//! - [`vq_vae`]: VQ-VAE decoder (work in progress).
//! - [`generation_head`]: Generation head that projects LLM hidden states to
//!   the image vocabulary (work in progress).
//! - [`tensor_utils`]: Image tensor pre/post-processing utilities.

pub mod config;
pub mod generation_head;
pub mod tensor_utils;
pub mod vq_vae;
