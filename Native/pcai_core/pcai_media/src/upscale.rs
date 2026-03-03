//! RealESRGAN 4x image upscaling via ONNX Runtime.
//!
//! Wraps a RealESRGAN ONNX model for 4x super-resolution upscaling.
//! Requires the `upscale` feature flag.
//!
//! # Example
//!
//! ```rust,no_run
//! use pcai_media::upscale::UpscalePipeline;
//!
//! let pipeline = UpscalePipeline::load("Models/RealESRGAN/RealESRGAN_x4.onnx")
//!     .expect("failed to load upscale model");
//! let img = image::open("input.png").unwrap();
//! let upscaled = pipeline.upscale(&img).expect("upscale failed");
//! upscaled.save("output_4x.png").expect("save failed");
//! ```

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

/// Scale factor for RealESRGAN x4plus.
const SCALE: usize = 4;

/// Holds a loaded ONNX Runtime session for RealESRGAN inference.
pub struct UpscalePipeline {
    session: ort::session::Session,
    input_name: String,
}

impl UpscalePipeline {
    /// Load a RealESRGAN ONNX model from disk.
    pub fn load(model_path: &str) -> Result<Self> {
        let session = ort::session::Session::builder()
            .context("failed to create ORT session builder")?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .context("failed to set optimization level")?
            .with_intra_threads(4)
            .context("failed to set intra threads")?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ONNX model: {model_path}"))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());

        tracing::info!(
            model = model_path,
            input = %input_name,
            "RealESRGAN model loaded"
        );

        Ok(Self {
            session,
            input_name,
        })
    }

    /// Upscale an image by 4x using RealESRGAN.
    ///
    /// Input image is converted to RGB, normalized to [0, 1], passed through
    /// the ONNX model, and the output is denormalized back to an `RgbImage`.
    pub fn upscale(&mut self, img: &DynamicImage) -> Result<RgbImage> {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);

        let tensor = Self::image_to_tensor(&rgb, h, w)?;

        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => tensor])
            .context("RealESRGAN inference failed")?;

        let out_h = h * SCALE;
        let out_w = w * SCALE;

        Self::tensor_to_image(&outputs[0], out_h, out_w)
    }

    /// Returns the scale factor (4).
    pub fn scale_factor(&self) -> usize {
        SCALE
    }

    /// Convert an RGB image to a [1, 3, H, W] f32 tensor normalized to [0, 1].
    fn image_to_tensor(
        rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        height: usize,
        width: usize,
    ) -> Result<ort::value::Tensor<f32>> {
        let mut data = vec![0.0f32; 3 * height * width];

        for y in 0..height {
            for x in 0..width {
                let px = rgb.get_pixel(x as u32, y as u32);
                let idx = y * width + x;
                data[idx] = px[0] as f32 / 255.0; // R plane
                data[height * width + idx] = px[1] as f32 / 255.0; // G plane
                data[2 * height * width + idx] = px[2] as f32 / 255.0; // B plane
            }
        }

        ort::value::Tensor::from_array(([1usize, 3, height, width], data))
            .context("failed to create input tensor")
    }

    /// Convert a [1, 3, H*4, W*4] f32 output tensor to an RgbImage.
    fn tensor_to_image(
        output: &ort::value::DynValue,
        out_h: usize,
        out_w: usize,
    ) -> Result<RgbImage> {
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .context("failed to extract output tensor")?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        anyhow::ensure!(
            dims.len() == 4 && dims[0] == 1 && dims[1] == 3,
            "unexpected output shape: {dims:?}, expected [1, 3, {out_h}, {out_w}]"
        );

        let plane_size = out_h * out_w;
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(out_w as u32, out_h as u32);

        for y in 0..out_h {
            for x in 0..out_w {
                let idx = y * out_w + x;
                let r = (data[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (data[plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (data[2 * plane_size + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_factor() {
        assert_eq!(SCALE, 4);
    }

    #[test]
    fn test_image_to_tensor_shape() {
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(8, 6);
        let tensor = UpscalePipeline::image_to_tensor(&img, 6, 8).unwrap();
        // The tensor shape should be [1, 3, 6, 8]
        let (shape, data) = tensor.try_extract_tensor::<f32>().unwrap();
        let dims: Vec<i64> = shape.iter().copied().collect();
        assert_eq!(dims, vec![1, 3, 6, 8]);
        assert_eq!(data.len(), 1 * 3 * 6 * 8);
    }

    #[test]
    fn test_image_to_tensor_values() {
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(2, 1);
        img.put_pixel(0, 0, Rgb([255, 0, 128]));
        img.put_pixel(1, 0, Rgb([0, 255, 64]));

        let tensor = UpscalePipeline::image_to_tensor(&img, 1, 2).unwrap();
        let (_shape, data) = tensor.try_extract_tensor::<f32>().unwrap();

        // R plane: [255/255, 0/255] = [1.0, 0.0]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        // G plane: [0/255, 255/255] = [0.0, 1.0]
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
        // B plane: [128/255, 64/255]
        assert!((data[4] - 128.0 / 255.0).abs() < 1e-4);
        assert!((data[5] - 64.0 / 255.0).abs() < 1e-4);
    }
}
