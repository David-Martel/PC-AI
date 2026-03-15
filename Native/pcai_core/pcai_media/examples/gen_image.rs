//! Quick end-to-end image generation test.
//!
//! Usage:
//!   cargo run --example gen_image --release --no-default-features -- \
//!     --model C:\codedev\PC_AI\Models\Janus-Pro-1B \
//!     --prompt "a beautiful sunset over mountains" \
//!     --output test_output.png

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

fn main() -> Result<()> {
    // Simple arg parsing
    let args: Vec<String> = std::env::args().collect();
    let model_path =
        get_arg(&args, "--model").unwrap_or_else(|| "C:\\codedev\\PC_AI\\Models\\Janus-Pro-1B".to_string());
    let prompt =
        get_arg(&args, "--prompt").unwrap_or_else(|| "a glowing circuit board on a dark background".to_string());
    let output = get_arg(&args, "--output").unwrap_or_else(|| "gen_test_output.png".to_string());
    let device_str = get_arg(&args, "--device").unwrap_or_else(|| "cpu".to_string());

    eprintln!("=== pcai-media image generation test ===");
    eprintln!("Model:  {model_path}");
    eprintln!("Prompt: {prompt}");
    eprintln!("Device: {device_str}");
    eprintln!("Output: {output}");
    eprintln!();

    // Build config
    // CPU cannot do bf16 matmul — use f32 for CPU, bf16 for CUDA
    let dtype_str = if device_str == "cpu" { "f32" } else { "bf16" };
    let cfg = pcai_media::config::PipelineConfig {
        model: model_path,
        device: device_str,
        dtype: dtype_str.to_string(),
        guidance_scale: 5.0,
        temperature: 1.0,
        parallel_size: 1,
        gpu_layers: 0,
    };

    // Load model
    eprintln!("[1/3] Loading model...");
    let t0 = Instant::now();
    let pipeline = pcai_media::generate::GenerationPipeline::load(cfg)?;
    eprintln!("       Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Generate image
    eprintln!("[2/3] Generating image (576 tokens)...");
    let t1 = Instant::now();
    let image = pipeline.generate(&prompt)?;
    eprintln!("       Generated in {:.1}s", t1.elapsed().as_secs_f64());

    // Save
    eprintln!("[3/3] Saving to {output}...");
    let output_path = PathBuf::from(&output);
    image.save(&output_path)?;
    let file_size = std::fs::metadata(&output_path)?.len();
    eprintln!("       Saved ({} bytes)", file_size);

    eprintln!();
    eprintln!("=== Done! Total: {:.1}s ===", t0.elapsed().as_secs_f64());
    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
