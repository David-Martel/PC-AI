//! Image tensor pre- and post-processing utilities for Janus-Pro.
//!
//! Migrated from `AI-Media/src/tensor_utils.rs`.
//!
//! # Normalisation convention
//!
//! Janus-Pro (following DeepSeek) uses a simple linear mapping between
//! `u8` pixel values and the float range expected by the model:
//!
//! | Direction | Formula                        |
//! |-----------|--------------------------------|
//! | forward   | `x_f = (x_u8 / 127.5) − 1.0` |
//! | inverse   | `x_u8 = clamp((x_f + 1) × 127.5, 0, 255)` |

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Normalises an image tensor from `[0, 255]` to `[-1, 1]`.
///
/// The input is first cast to `f32`, then scaled with `x / 127.5 - 1.0`.
///
/// # Arguments
///
/// * `image` — tensor of any shape with pixel values in `[0, 255]`.
///
/// # Errors
///
/// Propagates any candle error (dtype conversion, arithmetic).
///
/// # Example
///
/// ```rust,no_run
/// use candle_core::{Tensor, Device, DType};
/// use pcai_media_model::tensor_utils::normalize;
///
/// let dev = Device::Cpu;
/// let img = Tensor::from_vec(vec![0_u8, 127, 255], (3,), &dev).unwrap();
/// let norm = normalize(&img).unwrap();
/// ```
pub fn normalize(image: &Tensor) -> Result<Tensor> {
    let image = image.to_dtype(DType::F32)?;
    // Divide first, then subtract to stay within candle's op set.
    let scaled = (image / 127.5)?;
    (scaled - 1.0)?.contiguous()
}

/// Denormalises a tensor from `[-1, 1]` back to `[0, 255]` `u8`.
///
/// Expected input shape: `[Batch, 3, H, W]` or any compatible shape.
///
/// The inverse formula is `clamp((x + 1) / 2 × 255, 0, 255)`.
///
/// # Errors
///
/// Propagates any candle error (dtype conversion, arithmetic, clamp).
///
/// # Example
///
/// ```rust,no_run
/// use candle_core::{Tensor, Device, DType};
/// use pcai_media_model::tensor_utils::{normalize, denormalize};
///
/// let dev = Device::Cpu;
/// let img = Tensor::from_vec(vec![255_u8, 0, 128], (3,), &dev).unwrap();
/// let norm   = normalize(&img).unwrap();
/// let recon  = denormalize(&norm).unwrap();
/// ```
pub fn denormalize(tensor: &Tensor) -> Result<Tensor> {
    // [-1, 1] → [0, 2] → [0, 1] → [0, 255]
    let shifted = (tensor + 1.0)?;
    let scaled = (shifted / 2.0)?;
    let pixel = (scaled * 255.0)?.clamp(0.0_f64, 255.0_f64)?;
    pixel.to_dtype(DType::U8)
}

/// Creates a square upper-triangular causal mask of shape `[size, size]`.
///
/// An entry is `1` where the key index `j > i` (i.e., positions that should
/// be masked out), and `0` otherwise.  This matches the convention used by
/// `candle_transformers`' manual attention loops.
///
/// # Arguments
///
/// * `size`   — sequence length.
/// * `device` — target device for the output tensor.
///
/// # Errors
///
/// Propagates any candle error from tensor creation.
///
/// # Example
///
/// ```rust,no_run
/// use candle_core::Device;
/// use pcai_media_model::tensor_utils::create_causal_mask;
///
/// let mask = create_causal_mask(4, &Device::Cpu).unwrap();
/// // mask[i][j] == 1 iff j > i
/// ```
pub fn create_causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_vec(mask, (size, size), device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// After `normalize`, all values must be within `[-1.0, 1.0]`.
    #[test]
    fn test_normalize_range() {
        let dev = Device::Cpu;
        // Full range of u8 pixel values.
        let pixels: Vec<f32> = (0..=255_u8).map(f32::from).collect();
        let img = Tensor::from_vec(pixels, (256,), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let norm = normalize(&img).unwrap();
        let data: Vec<f32> = norm.to_vec1().unwrap();

        for v in &data {
            assert!(*v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5, "value {v} out of [-1, 1]");
        }
        // 0 → -1.0
        assert!((data[0] - (-1.0_f32)).abs() < 1e-5, "0 maps to -1.0");
        // 255 → +1.0
        assert!(
            (data[255] - 1.0_f32).abs() < 1e-3,
            "255 maps to ~1.0 (got {})",
            data[255]
        );
    }

    /// After `denormalize`, all values must be within `[0, 255]` as `u8`.
    #[test]
    fn test_denormalize_range() {
        let dev = Device::Cpu;
        // Sample of float values spanning beyond [-1, 1] to test clamping.
        let floats: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let tensor = Tensor::from_vec(floats, (7,), &dev).unwrap();

        let out = denormalize(&tensor).unwrap();
        assert_eq!(out.dtype(), DType::U8);

        let data: Vec<u8> = out.to_vec1().unwrap();
        for v in &data {
            // The u8 type itself guarantees [0, 255].
            let _ = v; // just assert it compiled and cast correctly
        }
        // -1.0 → 0
        assert_eq!(data[1], 0, "-1.0 should map to 0");
        // 1.0 → 255
        assert_eq!(data[5], 255, "1.0 should map to 255");
        // 0.0 → 127 or 128 (floating-point rounding)
        assert!(data[3] == 127 || data[3] == 128, "0.0 maps to midpoint");
        // clamped values stay within range
        assert_eq!(data[0], 0, "-2.0 clamped to 0");
        assert_eq!(data[6], 255, "2.0 clamped to 255");
    }

    /// `normalize` followed by `denormalize` should recover the original
    /// pixel values (within rounding error of ±1).
    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let dev = Device::Cpu;
        // Use a representative sample; avoid 127/128 ambiguity at midpoint.
        let original: Vec<f32> = vec![0.0, 64.0, 128.0, 192.0, 255.0];
        let img = Tensor::from_vec(original.clone(), (5,), &dev).unwrap();

        let norm = normalize(&img).unwrap();
        let recon = denormalize(&norm).unwrap();
        let data: Vec<u8> = recon.to_vec1().unwrap();

        for (i, (&orig, &rec)) in original.iter().zip(data.iter()).enumerate() {
            let diff = (orig - rec as f32).abs();
            assert!(
                diff <= 1.0,
                "index {i}: original {orig} reconstructed as {rec} (diff {diff})"
            );
        }
    }

    /// The causal mask must have 0 on and below the diagonal, 1 above it.
    #[test]
    fn test_causal_mask() {
        let dev = Device::Cpu;
        let size = 4;
        let mask = create_causal_mask(size, &dev).unwrap();

        assert_eq!(mask.dims(), &[size, size]);
        let data: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..size {
            for j in 0..size {
                let expected: u8 = u8::from(j > i);
                let got = data[i * size + j];
                assert_eq!(got, expected, "mask[{i}][{j}] expected {expected} got {got}");
            }
        }
        // Spot-check: diagonal must be 0.
        assert_eq!(data[0 * size + 0], 0);
        assert_eq!(data[1 * size + 1], 0);
        // Above diagonal must be 1.
        assert_eq!(data[0 * size + 1], 1);
        assert_eq!(data[0 * size + 3], 1);
        // Below diagonal must be 0.
        assert_eq!(data[2 * size + 0], 0);
        assert_eq!(data[3 * size + 1], 0);
    }
}
