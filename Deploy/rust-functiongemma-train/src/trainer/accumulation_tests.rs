use super::*;
use crate::data_gen::{Message, TrainingItem};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use rand::{RngExt, SeedableRng};
use rust_functiongemma_core::model::LoraSettings;

fn output_weights(varmap: &VarMap) -> Result<Vec<f32>> {
    let variables = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("fixture weights unavailable"))?;
    let projection = variables
        .get("lm_head.weight")
        .ok_or_else(|| anyhow::anyhow!("fixture output projection unavailable"))?;
    Ok(projection.as_tensor().flatten_all()?.to_vec1::<f32>()?)
}

fn fixture_config() -> Config {
    Config {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        sliding_window: None,
        layer_types: None,
    }
}

fn train_windows(windows: &[(&[&str], usize)]) -> Result<(Vec<f32>, Vec<f32>)> {
    let tokenizer = Tokenizer::from_bytes(
        br#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{
                "[UNK]":0,"question":1,"a":2,"b":3,"c":4,"model":5,"user":6,
                "start_of_turn":7,"end_of_turn":8
            }}
        }"#,
    )
    .map_err(anyhow::Error::msg)?;
    let cfg = fixture_config();
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = Model::new(
        &cfg,
        LoraSettings::new(0, 1.0, 0.0, false),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
        false,
    )?;
    {
        let weights = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("fixture weights unavailable"))?;
        for (name, var) in weights.iter() {
            // Seed each tensor by name so HashMap iteration order has no effect.
            let seed = name.bytes().fold(73u64, |value, byte| {
                value.wrapping_mul(31).wrapping_add(u64::from(byte))
            });
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let values: Vec<f32> = (0..var.shape().elem_count())
                .map(|_| rng.random_range(-0.5..0.5))
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &device)?)?;
        }
    }
    let initial = output_weights(&vars)?;
    let mut trainer = Trainer::new(
        model,
        &cfg,
        TrainerConfig {
            lr: 0.1,
            epochs: 1,
            batch_size: 1,
            use_lora: false,
            pack_sequences: false,
            warmup_steps: 0,
            scheduler_type: "constant".to_owned(),
            optimizer: "sgd".to_owned(),
            shuffle: false,
            max_grad_norm: None,
            progress_interval: 0,
            ..TrainerConfig::default()
        },
        device,
        None,
        vars.clone(),
    );
    let checkpoint_dir = tempfile::tempdir()?;
    trainer.checkpoint_config.output_dir = checkpoint_dir.path().to_path_buf();
    for &(answers, cap) in windows {
        let dataset = Dataset {
            items: answers
                .iter()
                .map(|answer| TrainingItem {
                    messages: vec![
                        Message {
                            role: "user".to_owned(),
                            content: Some("question".to_owned()),
                            tool_calls: None,
                        },
                        Message {
                            role: "assistant".to_owned(),
                            content: Some((*answer).to_owned()),
                            tool_calls: None,
                        },
                    ],
                    tools: serde_json::json!([]),
                })
                .collect(),
            token_cache: None,
            chat_template: None,
        };
        trainer.trainer_cfg.grad_accum = cap;
        let result = trainer.train(&dataset, Some(&tokenizer), None)?;
        assert_eq!(result.total_steps, answers.len().div_ceil(cap));
    }
    Ok((initial, output_weights(&vars)?))
}

fn max_difference(first: &[f32], second: &[f32]) -> f32 {
    assert!(!first.is_empty());
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(left, right)| {
            assert!(left.is_finite() && right.is_finite());
            (left - right).abs()
        })
        .fold(0.0, f32::max)
}

// Protects: the update magnitude when an epoch fits in one partial window.
// Detects: dividing its one microbatch by the configured cap.
// Needs: a tiny CPU model with an untied output projection, SGD and a temporary checkpoint path.
// Breadcrumb: trainer::Trainer::train, loss scaling before backward().
#[test]
fn single_microbatch_has_the_same_update_at_caps_one_and_four() -> Result<()> {
    let (initial, reference) = train_windows(&[(&["a"], 1)])?;
    let (other_initial, actual) = train_windows(&[(&["a"], 4)])?;
    assert_eq!(initial, other_initial);
    assert!(
        max_difference(&initial, &reference) > 1e-5,
        "reference should update weights"
    );
    let difference = max_difference(&reference, &actual);
    assert!(difference < 1e-6, "partial window diluted the update by {difference}");
    Ok(())
}

// Protects: the final update after a full accumulation window.
// Detects: using the preceding full-window divisor for the last microbatch.
// Needs: three tiny CPU fixture items, SGD and temporary checkpoint paths.
// Breadcrumb: trainer::Trainer::train, accumulation window sizing and optimizer step.
#[test]
fn tail_after_a_full_window_matches_explicitly_partitioned_updates() -> Result<()> {
    let (_, before_tail) = train_windows(&[(&["a", "b"], 2)])?;
    let (_, reference) = train_windows(&[(&["a", "b"], 2), (&["c"], 1)])?;
    let (_, actual) = train_windows(&[(&["a", "b", "c"], 2)])?;
    assert!(
        max_difference(&before_tail, &reference) > 1e-5,
        "tail should update weights"
    );
    let difference = max_difference(&reference, &actual);
    assert!(
        difference < 1e-6,
        "tail update differed from the reference by {difference}"
    );
    Ok(())
}

// Protects: global gradient clipping handles BF16 gradients without reducing in BF16 or
// changing either parameter values or any gradient's stored dtype.
// Detects: a BF16 scalar extraction error, BF16 rounding before the norm, or wrong
// mixed-dtype scale or output dtype.
// Needs: Candle's CPU backend and a synthetic model and gradient store. No model files or CUDA.
// Breadcrumb: trainer::Trainer::clip_gradients, F32 global norm and dtype-preserving scaling.
#[test]
fn clips_mixed_bf16_and_f32_gradients_in_their_original_dtypes() -> Result<()> {
    let cfg = fixture_config();
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let model = Model::new(
        &cfg,
        LoraSettings::new(0, 1.0, 0.0, false),
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
        false,
    )?;
    let max_norm = 939.0f64.sqrt() / 2.0;
    let trainer = Trainer::new(
        model,
        &cfg,
        TrainerConfig {
            max_grad_norm: Some(max_norm),
            ..TrainerConfig::default()
        },
        device.clone(),
        None,
        varmap,
    );
    let bf16 = Var::from_tensor(&Tensor::new(&[1.0f32, -2.0], &device)?.to_dtype(DType::BF16)?)?;
    let fp32 = Var::from_tensor(&Tensor::new(&[3.0f32], &device)?)?;
    let seed = Var::from_tensor(&Tensor::new(&[0.0f32], &device)?)?;
    let mut gradients = seed.as_tensor().backward()?;
    gradients.remove(seed.as_tensor());
    // 19² + 23² + 7² = 939. The BF16 squares would round before the reduction.
    gradients.insert(
        bf16.as_tensor(),
        Tensor::new(&[19.0f32, -23.0], &device)?.to_dtype(DType::BF16)?,
    );
    gradients.insert(fp32.as_tensor(), Tensor::new(&[7.0f32], &device)?);

    trainer.clip_gradients(&mut gradients, &[bf16.clone(), fp32.clone()])?;

    let bf16_gradient = gradients.get(bf16.as_tensor()).expect("BF16 gradient");
    let fp32_gradient = gradients.get(fp32.as_tensor()).expect("F32 gradient");
    assert_eq!(bf16_gradient.dtype(), DType::BF16);
    assert_eq!(fp32_gradient.dtype(), DType::F32);
    let bf16_values = bf16_gradient.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let fp32_values = fp32_gradient.to_vec1::<f32>()?;
    assert_eq!(bf16_values, [9.5, -11.5]);
    assert_eq!(fp32_values, [3.5]);
    let clipped_norm = bf16_values
        .into_iter()
        .chain(fp32_values)
        .map(|value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((clipped_norm - max_norm).abs() < 1e-6);
    assert_eq!(bf16.as_tensor().dtype(), DType::BF16);
    assert_eq!(bf16.as_tensor().to_dtype(DType::F32)?.to_vec1::<f32>()?, [1.0, -2.0]);
    assert_eq!(fp32.as_tensor().dtype(), DType::F32);
    assert_eq!(fp32.as_tensor().to_vec1::<f32>()?, [3.0]);
    Ok(())
}
