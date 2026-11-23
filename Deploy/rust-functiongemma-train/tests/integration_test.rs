use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rust_functiongemma_train::{Config, LoraSettings, Model};

// Protects: CPU FunctionGemma forward through the shared trainer model.
// Detects: a bad attention-mask dtype or a future-token leak.
// Needs: in-memory weights and token IDs. Breadcrumb: model::Model.
#[test]
fn cpu_model_forward_preserves_shape_and_masks_future_tokens() -> Result<()> {
    let device = Device::Cpu;
    let config = Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 16,
        vocab_size: 1000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        sliding_window: None,
        layer_types: None,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&config, LoraSettings::new(0, 1.0, 0.0, false), vb, true)?;

    // Fix the weight values independently of Candle's unseedable CPU RNG.
    {
        let variables = varmap
            .data()
            .lock()
            .map_err(|_| anyhow!("fixture weights unavailable"))?;
        for var in variables.values() {
            let values: Vec<f32> = (0..var.shape().elem_count())
                .map(|i| ((i * 31 % 101) as f32 - 50.0) / 100.0)
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &device)?)?;
        }
    }

    let first = Tensor::new(&[[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]], &device)?;
    let changed = Tensor::new(&[[0u32, 1, 2, 3, 4, 5, 10, 11, 12, 13]], &device)?;
    let output = model.forward(&first)?;
    let changed_output = model.forward(&changed)?;

    assert_eq!(output.dims(), &[1, 10, 1000]);
    assert_eq!(output.dims(), changed_output.dims());
    let prefix_delta = (&output.narrow(1, 0, 6)? - &changed_output.narrow(1, 0, 6)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let suffix_delta = (&output.narrow(1, 6, 4)? - &changed_output.narrow(1, 6, 4)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        prefix_delta < 1e-5,
        "future tokens changed the prefix logits by {prefix_delta}"
    );
    assert!(suffix_delta > 1e-4, "fixture must produce distinct suffix logits");
    Ok(())
}
