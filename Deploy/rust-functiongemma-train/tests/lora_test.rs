use candle_core::{Device, Module, Tensor};
use rust_functiongemma_train::lora::{LoraConfig, LoraLinear};

#[test]
fn test_lora_linear_forward() {
    let device = Device::Cpu;
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec!["q_proj".to_string()],
    };
    let lora = LoraLinear::new(768, 768, &config, &device).expect("TODO: Verify unwrap");
    let input = Tensor::randn(0f32, 1f32, (2, 10, 768), &device).expect("TODO: Verify unwrap");
    let output = lora.forward(&input).expect("TODO: Verify unwrap");
    assert_eq!(output.dims(), &[2, 10, 768]);
}
