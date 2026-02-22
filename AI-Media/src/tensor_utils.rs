2. Utility Code (src/tensor_utils.rs)
This file handles the necessary pre/post processing of image tensors, specifically managing the normalization constants used by DeepSeek (which are often standard ImageNet or simple -1 to 1).

Rust
// src/tensor_utils.rs

use candle_core::{Tensor, Result, DType};

/// Normalizes a tensor from [0, 255] to [-1, 1]
pub fn normalize(image: &Tensor) -> Result<Tensor> {
    let image = image.to_dtype(DType::F32)?;
    let image = (image / 127.5)? - 1.0;
    Ok(image?)
}

/// Denormalizes a tensor from [-1, 1] to [0, 255] uint8
pub fn denormalize(tensor: &Tensor) -> Result<Tensor> {
    // Input: [Batch, 3, H, W] in range [-1, 1]

    // 1. Scale to [0, 2] -> [0, 1] -> [0, 255]
    let tensor = (tensor + 1.0)? / 2.0;
    let tensor = (tensor? * 255.0)?.clamp(0.0, 255.0)?;

    // 2. Cast to u8
    let tensor = tensor.to_dtype(DType::U8)?;

    Ok(tensor)
}

/// Helper to create the causal mask for the LLM if manually stepping
pub fn create_causal_mask(size: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_vec(mask, (size, size), device)
}
