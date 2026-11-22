#![cfg(feature = "upscale")]

use image::{DynamicImage, Rgb, RgbImage};
use pcai_media::upscale::UpscalePipeline;

// Protects: CPU model loading, inference and the full RGB conversion path.
// Detects: runtime/session failures, wrong dimensions or changed channel values.
// Needs: linked ONNX Runtime and a writable temporary directory. The local graph
// uses ONNX IR 8 and opset 13. Its only node computes Tile(rgb, [1, 1, 4, 4])
// from a [1, 3, 2, 3] input to a [1, 3, 8, 12] output.
// Breadcrumb: ../src/upscale.rs (UpscalePipeline::load and UpscalePipeline::upscale).
#[test]
fn cpu_model_tiles_non_square_rgb_input() {
    let directory = tempfile::tempdir().expect("temporary directory should be available");
    let model_path = directory.path().join("tile-4x.onnx");
    std::fs::write(&model_path, include_bytes!("fixtures/tile-4x.onnx.pb"))
        .expect("synthetic ONNX graph should be written");
    let mut pipeline = UpscalePipeline::load(model_path.to_str().expect("temporary path should be UTF-8"))
        .expect("CPU session should load the graph");
    eprintln!("{}", ort::info());

    let pixels = [
        [Rgb([255, 0, 128]), Rgb([0, 90, 10]), Rgb([17, 55, 240])],
        [Rgb([23, 0, 250]), Rgb([222, 121, 0]), Rgb([75, 255, 33])],
    ];
    let input = DynamicImage::ImageRgb8(RgbImage::from_fn(3, 2, |x, y| pixels[y as usize][x as usize]));
    let output = pipeline.upscale(&input).expect("CPU inference should run the graph");

    assert_eq!(output.dimensions(), (12, 8));
    let input = input.as_rgb8().expect("input should still be RGB");
    for (x, y, pixel) in output.enumerate_pixels() {
        assert_eq!(
            pixel,
            input.get_pixel(x % input.width(), y % input.height()),
            "pixel ({x}, {y})"
        );
    }
}
