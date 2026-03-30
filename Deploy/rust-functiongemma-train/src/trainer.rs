use crate::checkpoint::{Checkpoint, CheckpointConfig};
use crate::dataset::Dataset;
use crate::early_stopping::{EarlyStopping, EarlyStoppingConfig};
use crate::scheduler::{LRScheduler, SchedulerConfig, SchedulerType};
use anyhow::Result;
use candle_core::{DType, Device, Var};
use candle_nn::{Optimizer, VarMap};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rust_functiongemma_core::gpu::cuda_mem_snapshot;
use rust_functiongemma_core::model::{Config, Model};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Per-step performance metrics for training analytics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingStepMetrics {
    pub step: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub forward_ms: f64,
    pub backward_ms: f64,
    pub optimizer_ms: f64,
    pub total_step_ms: f64,
    pub tokens_per_second: f64,
    pub gpu_mem_used_mb: Option<u64>,
    pub gpu_mem_free_mb: Option<u64>,
}

/// Aggregated training run metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingRunMetrics {
    pub total_steps: usize,
    pub total_tokens: usize,
    pub total_time_sec: f64,
    pub avg_tokens_per_second: f64,
    pub avg_step_ms: f64,
    pub final_loss: f64,
    pub best_loss: f64,
    /// Loss reduction per 100 steps (negative means loss is decreasing, which is good).
    pub loss_convergence_rate: f64,
    pub steps: Vec<TrainingStepMetrics>,
}

pub struct TrainerConfig {
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub lora_r: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub pack_sequences: bool,
    pub max_seq_len: Option<usize>,
    pub eos_token_id: u32,
    pub use_lora: bool,
    pub warmup_steps: usize,
    pub scheduler_type: String,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub use_4bit: bool,
    pub eval_max_batches: Option<usize>,
    pub max_grad_norm: Option<f64>,
    pub shuffle: bool,
    pub seed: Option<u64>,
    pub progress_interval: usize,
    pub progress_json: bool,
    pub optimizer: String,
    pub report_gpu_memory: bool,
    pub gpu_mem_snapshot: bool,
    pub loss_in_f32: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            epochs: 3,
            batch_size: 4,
            grad_accum: 4,
            lora_r: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.0,
            pack_sequences: true,
            max_seq_len: Some(512),
            eos_token_id: 2,
            use_lora: true,
            warmup_steps: 100,
            scheduler_type: "cosine".to_string(),
            early_stopping: None,
            use_4bit: false,
            eval_max_batches: None,
            max_grad_norm: Some(1.0),
            shuffle: true,
            seed: None,
            progress_interval: 1,
            progress_json: false,
            optimizer: "adamw".to_string(),
            report_gpu_memory: false,
            gpu_mem_snapshot: false,
            loss_in_f32: true,
        }
    }
}

/// Compute the slope of the ordinary least-squares regression line for the
/// given `(x, y)` points.
///
/// Formula: slope = sum((xi - x_mean) * (yi - y_mean)) / sum((xi - x_mean)^2)
///
/// Returns `0.0` when there are fewer than 2 data points or the denominator is
/// effectively zero (all x values are identical).
fn linear_regression_slope(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let n = points.len() as f64;
    let x_mean = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_mean = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let numerator: f64 = points.iter().map(|(x, y)| (x - x_mean) * (y - y_mean)).sum();
    let denominator: f64 = points.iter().map(|(x, _)| (x - x_mean).powi(2)).sum();
    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

pub struct Trainer<'a> {
    pub model: Model,
    pub config: &'a Config,
    pub trainer_cfg: TrainerConfig,
    pub device: Device,
    pub cuda_index: Option<usize>,
    pub varmap: VarMap,
    pub scheduler: LRScheduler,
    pub checkpoint_config: CheckpointConfig,
    pub global_step: usize,
}

enum OptimizerWrapper {
    AdamW(candle_nn::AdamW),
    Sgd(candle_nn::SGD),
}

impl OptimizerWrapper {
    fn set_lr(&mut self, lr: f64) {
        match self {
            OptimizerWrapper::AdamW(opt) => opt.set_learning_rate(lr),
            OptimizerWrapper::Sgd(opt) => opt.set_learning_rate(lr),
        }
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        match self {
            OptimizerWrapper::AdamW(opt) => opt.step(grads)?,
            OptimizerWrapper::Sgd(opt) => opt.step(grads)?,
        }
        Ok(())
    }
}

impl<'a> Trainer<'a> {
    pub fn new(
        model: Model,
        config: &'a Config,
        trainer_cfg: TrainerConfig,
        device: Device,
        cuda_index: Option<usize>,
        varmap: VarMap,
    ) -> Self {
        let scheduler_type = match trainer_cfg.scheduler_type.as_str() {
            "linear" => SchedulerType::Linear,
            "constant" => SchedulerType::Constant,
            _ => SchedulerType::Cosine,
        };

        // Initialize scheduler with placeholder total_steps (will be updated in train())
        let scheduler = LRScheduler::new(SchedulerConfig {
            scheduler_type,
            warmup_steps: trainer_cfg.warmup_steps,
            total_steps: trainer_cfg.epochs * 1000, // Placeholder, updated in train()
            min_lr: trainer_cfg.lr / 10.0,
            max_lr: trainer_cfg.lr,
        });

        let checkpoint_config = CheckpointConfig {
            output_dir: PathBuf::from("./checkpoints"),
            save_every_n_steps: 500,
            max_checkpoints: 3,
        };

        Self {
            model,
            config,
            trainer_cfg,
            device,
            cuda_index,
            varmap,
            scheduler,
            checkpoint_config,
            global_step: 0,
        }
    }

    fn query_gpu_memory_mb(&self) -> Option<u64> {
        if !self.trainer_cfg.report_gpu_memory {
            return None;
        }
        let idx = self.cuda_index?;
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                &idx.to_string(),
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse::<u64>().ok()
    }

    fn trainable_vars(&self) -> Result<Vec<Var>> {
        if !self.trainer_cfg.use_lora {
            return Ok(self.varmap.all_vars());
        }
        let data = self
            .varmap
            .data()
            .lock()
            .map_err(|e| anyhow::anyhow!("VarMap lock poisoned in trainable_vars: {}", e))?;
        let mut vars = Vec::new();
        for (name, var) in data.iter() {
            if name.contains("lora_a") || name.contains("lora_b") {
                vars.push(var.clone());
            }
        }
        if vars.is_empty() {
            return Err(anyhow::anyhow!(
                "No LoRA tensors found in VarMap while use_lora=true. Refusing to train full base weights."
            ));
        }
        Ok(vars)
    }

    pub fn train(
        &mut self,
        dataset: &Dataset,
        tokenizer: Option<&Tokenizer>,
        eval_dataset: Option<&Dataset>,
    ) -> Result<TrainingRunMetrics> {
        if self.trainer_cfg.use_4bit {
            println!("QLoRA 4-bit enabled.");
        }
        if dataset.len() == 0 {
            return Err(anyhow::anyhow!("Empty training dataset"));
        }
        let grad_accum = self.trainer_cfg.grad_accum.max(1);
        let num_batches = (dataset.len() + self.trainer_cfg.batch_size - 1) / self.trainer_cfg.batch_size;
        let num_update_steps = (num_batches + grad_accum - 1) / grad_accum;
        let total_steps = self.trainer_cfg.epochs * num_update_steps;

        // Update scheduler with correct total_steps
        let scheduler_type = match self.trainer_cfg.scheduler_type.as_str() {
            "linear" => SchedulerType::Linear,
            "constant" => SchedulerType::Constant,
            _ => SchedulerType::Cosine,
        };

        let mut warmup_steps = self.trainer_cfg.warmup_steps;
        if warmup_steps > total_steps {
            if self.trainer_cfg.progress_json {
                println!(
                    "{{\"event\":\"train_warning\",\"message\":\"warmup_steps exceeds total_steps; clamping\",\"warmup_steps\":{},\"total_steps\":{}}}",
                    warmup_steps,
                    total_steps
                );
            } else {
                println!(
                    "Warning: warmup_steps ({}) exceeds total_steps ({}). Clamping.",
                    warmup_steps, total_steps
                );
            }
            warmup_steps = total_steps;
        }
        self.scheduler = LRScheduler::new(SchedulerConfig {
            scheduler_type,
            warmup_steps,
            total_steps,
            min_lr: self.trainer_cfg.lr / 10.0,
            max_lr: self.trainer_cfg.lr,
        });

        let trainable_vars = self.trainable_vars()?;
        let mut optimizer = match self.trainer_cfg.optimizer.to_lowercase().as_str() {
            "sgd" => OptimizerWrapper::Sgd(candle_nn::SGD::new(trainable_vars.clone(), self.trainer_cfg.lr)?),
            _ => OptimizerWrapper::AdamW(candle_nn::AdamW::new_lr(trainable_vars.clone(), self.trainer_cfg.lr)?),
        };
        let mut best_loss = f64::MAX;
        let mut early_stopper = self.trainer_cfg.early_stopping.clone().map(EarlyStopping::new);
        let mut last_epoch = 0usize;

        // Performance metrics collection
        let run_start = Instant::now();
        let mut all_step_metrics: Vec<TrainingStepMetrics> = Vec::new();
        let mut total_tokens_processed: usize = 0;

        // Enable training mode so LoRA dropout is applied during forward passes
        self.model.set_training(true);

        for epoch in 0..self.trainer_cfg.epochs {
            if self.trainer_cfg.progress_json {
                println!(
                    "{{\"event\":\"train_epoch_start\",\"epoch\":{},\"total_epochs\":{}}}",
                    epoch + 1,
                    self.trainer_cfg.epochs
                );
            } else {
                println!("Epoch {}/{}", epoch + 1, self.trainer_cfg.epochs);
            }
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            let mut update_step = 0usize;
            let mut indices: Vec<usize> = (0..dataset.len()).collect();
            if self.trainer_cfg.shuffle {
                if let Some(seed) = self.trainer_cfg.seed {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(epoch as u64));
                    indices.shuffle(&mut rng);
                } else {
                    let mut rng = rand::thread_rng();
                    indices.shuffle(&mut rng);
                }
            }

            let mut grad_accum_store: Option<candle_core::backprop::GradStore> = None;
            let mut current_lr = self.scheduler.get_lr(self.global_step);

            // Per-update-step timing accumulators (reset at each accumulation boundary)
            let mut accum_fwd_ms: f64 = 0.0;
            let mut accum_bwd_ms: f64 = 0.0;
            let mut accum_step_start: Option<Instant> = None;
            let mut accum_batch_tokens: usize = 0;

            for i in 0..num_batches {
                let accum_index = i % grad_accum;
                if accum_index == 0 {
                    current_lr = self.scheduler.get_lr(self.global_step);
                    optimizer.set_lr(current_lr);
                    // Reset per-update-step accumulators
                    accum_fwd_ms = 0.0;
                    accum_bwd_ms = 0.0;
                    accum_step_start = Some(Instant::now());
                    accum_batch_tokens = 0;
                }

                let start_idx = i * self.trainer_cfg.batch_size;
                let end_idx = (start_idx + self.trainer_cfg.batch_size).min(indices.len());
                if start_idx >= end_idx {
                    continue;
                }
                let batch_indices = &indices[start_idx..end_idx];
                let loss_val = {
                    let (inputs, targets, loss_mask) = dataset.get_batch_by_indices(
                        batch_indices,
                        tokenizer,
                        &self.device,
                        self.trainer_cfg.pack_sequences,
                        self.trainer_cfg.max_seq_len,
                        self.trainer_cfg.eos_token_id,
                    )?;

                    // Count tokens in this micro-batch for throughput calculation
                    let micro_batch_tokens = inputs.dims().iter().product::<usize>();
                    accum_batch_tokens += micro_batch_tokens;

                    // Forward pass timing (model forward + loss computation)
                    let fwd_start = Instant::now();
                    let logits = self.model.forward(&inputs)?;
                    let (loss, loss_val, mask_sum) = self.compute_loss(&logits, &targets, &loss_mask)?;
                    let fwd_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
                    accum_fwd_ms += fwd_ms;

                    if mask_sum == 0.0 {
                        return Err(anyhow::anyhow!(
                            "Loss mask sum is zero; check dataset/tokenization alignment."
                        ));
                    }

                    if !loss_val.is_finite() {
                        return Err(anyhow::anyhow!(
                            "Training diverged: loss is {} at step {}. \
                             Check learning rate and data.",
                            loss_val,
                            update_step,
                        ));
                    }

                    epoch_loss += loss_val;
                    batch_count += 1;

                    // Backward pass timing (gradient computation)
                    let bwd_start = Instant::now();
                    let scaled_loss = loss.affine(1.0 / (grad_accum as f64), 0.0)?;
                    let grads = scaled_loss.backward()?;
                    let bwd_ms = bwd_start.elapsed().as_secs_f64() * 1000.0;
                    accum_bwd_ms += bwd_ms;

                    // Accumulate only trainable (LoRA) parameter gradients.
                    // backward() returns gradients for every tensor in the
                    // computation graph; storing all of them in the accumulator
                    // wastes VRAM on frozen base-model weights.  By always
                    // extracting only trainable_vars we keep memory proportional
                    // to the LoRA adapter size, not the full model.
                    if grad_accum_store.is_none() {
                        // First micro-batch: create an empty GradStore.
                        // GradStore::new() is private in candle, so we obtain
                        // one via backward() on a detached zero scalar (cheap).
                        let seed = candle_core::Var::from_tensor(
                            &candle_core::Tensor::new(&[0.0f32], &self.device)?,
                        )?;
                        let mut empty = seed.as_tensor().backward()?;
                        empty.remove(seed.as_tensor());
                        grad_accum_store = Some(empty);
                    }
                    let accum = grad_accum_store.as_mut().expect("initialized above");
                    for var in trainable_vars.iter() {
                        let tensor = var.as_tensor();
                        if let Some(g) = grads.get(&tensor) {
                            if let Some(existing) = accum.get(&tensor) {
                                let merged = existing.add(g)?;
                                accum.insert(&tensor, merged);
                            } else {
                                accum.insert(&tensor, g.clone());
                            }
                        }
                    }
                    drop(grads);

                    // Drop intermediates to avoid graph creep and reduce peak memory.
                    drop(logits);
                    drop(loss);
                    drop(scaled_loss);
                    drop(inputs);
                    drop(targets);
                    drop(loss_mask);
                    loss_val
                };

                let is_accum_end = accum_index == grad_accum - 1 || i + 1 == num_batches;
                if is_accum_end {
                    // Optimizer step timing
                    let opt_start = Instant::now();
                    if let Some(mut accum) = grad_accum_store.take() {
                        self.clip_gradients(&mut accum, &trainable_vars)?;
                        optimizer.step(&accum)?;
                    }
                    let opt_ms = opt_start.elapsed().as_secs_f64() * 1000.0;

                    let total_step_ms = accum_step_start
                        .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    let tokens_per_sec = if total_step_ms > 0.0 {
                        accum_batch_tokens as f64 / (total_step_ms / 1000.0)
                    } else {
                        0.0
                    };
                    total_tokens_processed += accum_batch_tokens;

                    self.global_step += 1;
                    update_step += 1;

                    // Collect GPU memory snapshot
                    let gpu_snapshot = if self.trainer_cfg.gpu_mem_snapshot {
                        cuda_mem_snapshot(self.cuda_index)
                    } else {
                        None
                    };

                    // Record step metrics
                    let step_metrics = TrainingStepMetrics {
                        step: self.global_step,
                        loss: loss_val,
                        learning_rate: current_lr,
                        forward_ms: accum_fwd_ms,
                        backward_ms: accum_bwd_ms,
                        optimizer_ms: opt_ms,
                        total_step_ms,
                        tokens_per_second: tokens_per_sec,
                        gpu_mem_used_mb: gpu_snapshot.map(|s| s.used_mb()),
                        gpu_mem_free_mb: gpu_snapshot.map(|s| s.free_mb()),
                    };
                    all_step_metrics.push(step_metrics);

                    // Emit structured performance log line (JSON mode)
                    if self.trainer_cfg.progress_json {
                        println!(
                            "{{\"event\":\"train_step\",\"step\":{},\"loss\":{:.6},\"lr\":{:.6e},\"fwd_ms\":{:.2},\"bwd_ms\":{:.2},\"opt_ms\":{:.2},\"total_ms\":{:.2},\"tok_s\":{:.1},\"gpu_used_mb\":{},\"gpu_free_mb\":{}}}",
                            self.global_step,
                            loss_val,
                            current_lr,
                            accum_fwd_ms,
                            accum_bwd_ms,
                            opt_ms,
                            total_step_ms,
                            tokens_per_sec,
                            gpu_snapshot.map(|s| s.used_mb().to_string()).unwrap_or_else(|| "null".to_string()),
                            gpu_snapshot.map(|s| s.free_mb().to_string()).unwrap_or_else(|| "null".to_string()),
                        );
                    }

                    // Save checkpoint periodically
                    if self.global_step % self.checkpoint_config.save_every_n_steps == 0 {
                        self.save_checkpoint(epoch, best_loss)?;
                        Checkpoint::cleanup_old(&self.checkpoint_config)?;
                    }

                    self.report_progress(epoch, update_step, num_update_steps, loss_val, current_lr)?;
                }
            }

            if batch_count == 0 {
                return Err(anyhow::anyhow!("No valid batches processed (mask sum may be zero)."));
            }
            let avg_epoch_loss = epoch_loss / batch_count as f64;
            if !self.trainer_cfg.progress_json {
                println!("Epoch {} completed. Avg Loss: {:.4}", epoch + 1, avg_epoch_loss);
            } else {
                println!(
                    "{{\"event\":\"train_epoch_end\",\"epoch\":{},\"avg_loss\":{:.6}}}",
                    epoch + 1,
                    avg_epoch_loss
                );
            }

            let mut metric_loss = avg_epoch_loss;
            if let Some(eval_ds) = eval_dataset {
                // Disable training mode (turns off LoRA dropout) for evaluation
                self.model.set_training(false);
                match self.evaluate_loss(eval_ds, tokenizer) {
                    Ok(eval_loss) => {
                        metric_loss = eval_loss;
                        println!("Eval Loss: {:.4}", eval_loss);
                    }
                    Err(err) => {
                        println!("Eval failed: {:?}", err);
                    }
                }
                // Re-enable training mode for subsequent epochs
                self.model.set_training(true);
            }

            // Update best loss
            if metric_loss < best_loss {
                best_loss = metric_loss;
                println!("New best loss: {:.4}", best_loss);
            }
            last_epoch = epoch;

            if let Some(stopper) = early_stopper.as_mut() {
                if stopper.should_stop(metric_loss) {
                    println!(
                        "Early stopping triggered at epoch {} (best loss {:.4}).",
                        epoch + 1,
                        stopper.best_loss()
                    );
                    break;
                }
            }
        }

        // Disable training mode now that training is complete
        self.model.set_training(false);

        // Save final checkpoint
        self.save_checkpoint(last_epoch, best_loss)?;

        // Compute aggregated training run metrics
        let total_time_sec = run_start.elapsed().as_secs_f64();
        let total_steps = all_step_metrics.len();
        let final_loss = all_step_metrics.last().map(|m| m.loss).unwrap_or(0.0);
        let avg_step_ms = if total_steps > 0 {
            all_step_metrics.iter().map(|m| m.total_step_ms).sum::<f64>() / total_steps as f64
        } else {
            0.0
        };
        let avg_tokens_per_second = if total_time_sec > 0.0 {
            total_tokens_processed as f64 / total_time_sec
        } else {
            0.0
        };

        // Loss convergence rate: loss reduction per 100 steps, computed via
        // ordinary least-squares linear regression over all step losses.
        // A negative value means the loss is decreasing (good).
        let loss_convergence_rate = if total_steps >= 2 {
            let slope = linear_regression_slope(
                &all_step_metrics
                    .iter()
                    .map(|m| (m.step as f64, m.loss))
                    .collect::<Vec<_>>(),
            );
            slope * 100.0
        } else {
            0.0
        };

        let run_metrics = TrainingRunMetrics {
            total_steps,
            total_tokens: total_tokens_processed,
            total_time_sec,
            avg_tokens_per_second,
            avg_step_ms,
            final_loss,
            best_loss,
            loss_convergence_rate,
            steps: all_step_metrics,
        };

        if self.trainer_cfg.progress_json {
            // Emit summary without the per-step array (too large for log line)
            println!(
                "{{\"event\":\"train_complete\",\"best_loss\":{:.6},\"final_loss\":{:.6},\"total_steps\":{},\"total_tokens\":{},\"total_time_sec\":{:.2},\"avg_tok_s\":{:.1},\"avg_step_ms\":{:.2},\"convergence_rate_per_100\":{:.6}}}",
                run_metrics.best_loss,
                run_metrics.final_loss,
                run_metrics.total_steps,
                run_metrics.total_tokens,
                run_metrics.total_time_sec,
                run_metrics.avg_tokens_per_second,
                run_metrics.avg_step_ms,
                run_metrics.loss_convergence_rate,
            );
        } else {
            println!("Training completed. Best loss: {:.4}", best_loss);
            println!(
                "  Steps: {} | Tokens: {} | Time: {:.1}s | Avg tok/s: {:.1} | Avg step: {:.1}ms | Convergence: {:.4}/100 steps",
                run_metrics.total_steps,
                run_metrics.total_tokens,
                run_metrics.total_time_sec,
                run_metrics.avg_tokens_per_second,
                run_metrics.avg_step_ms,
                run_metrics.loss_convergence_rate,
            );
        }

        Ok(run_metrics)
    }

    fn compute_loss(
        &self,
        logits: &candle_core::Tensor,
        targets: &candle_core::Tensor,
        loss_mask: &candle_core::Tensor,
    ) -> Result<(candle_core::Tensor, f64, f64)> {
        let (b, s, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * s, v))?;
        let logits_for_loss = if self.trainer_cfg.loss_in_f32 && logits_flat.dtype() != candle_core::DType::F32 {
            logits_flat.to_dtype(candle_core::DType::F32)?
        } else {
            logits_flat
        };
        let targets_flat = targets.reshape((b * s,))?;
        let log_probs = candle_nn::ops::log_softmax(&logits_for_loss, 1)?;
        let target_log_probs = log_probs.gather(&targets_flat.unsqueeze(1)?, 1)?.squeeze(1)?;
        let mask_flat = loss_mask.reshape((b * s,))?;
        let mask_flat = if mask_flat.dtype() != target_log_probs.dtype() {
            mask_flat.to_dtype(target_log_probs.dtype())?
        } else {
            mask_flat
        };
        let masked_log_probs = (&target_log_probs * &mask_flat)?;
        let mask_sum = mask_flat.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64;
        let denom = if mask_sum > 0.0 { mask_sum } else { 1.0 };
        let loss = masked_log_probs.sum_all()?.affine(-1.0 / denom, 0.0)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        Ok((loss, loss_val, mask_sum))
    }

    fn clip_gradients(&self, grads: &mut candle_core::backprop::GradStore, vars: &[Var]) -> Result<()> {
        let max_norm = match self.trainer_cfg.max_grad_norm {
            Some(v) if v > 0.0 => v,
            _ => return Ok(()),
        };
        let mut sum_sq = 0.0f64;
        for var in vars {
            let tensor = var.as_tensor();
            if let Some(g) = grads.get(&tensor) {
                let val = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                sum_sq += val;
            }
        }
        if sum_sq <= 0.0 {
            return Ok(());
        }
        let norm = sum_sq.sqrt();
        if norm <= max_norm {
            return Ok(());
        }
        let scale = max_norm / norm;
        for var in vars {
            let tensor = var.as_tensor();
            if let Some(g) = grads.get(&tensor) {
                let scaled = g.affine(scale, 0.0)?;
                grads.insert(&tensor, scaled);
            }
        }
        Ok(())
    }

    fn report_progress(
        &self,
        epoch: usize,
        update_step: usize,
        total_updates: usize,
        loss_val: f64,
        lr: f64,
    ) -> Result<()> {
        if self.trainer_cfg.progress_interval == 0 {
            return Ok(());
        }
        if update_step % self.trainer_cfg.progress_interval != 0 && update_step < total_updates {
            return Ok(());
        }
        if self.trainer_cfg.progress_json {
            let gpu_mem_mb = self.query_gpu_memory_mb();
            let cuda_snapshot = if self.trainer_cfg.gpu_mem_snapshot {
                cuda_mem_snapshot(self.cuda_index)
            } else {
                None
            };
            println!(
                "{{\"event\":\"train_progress\",\"epoch\":{},\"step\":{},\"total_steps\":{},\"loss\":{:.6},\"lr\":{:.6e},\"gpu_mem_mb\":{},\"gpu_mem_free_mb\":{},\"gpu_mem_used_mb\":{},\"gpu_mem_total_mb\":{}}}",
                epoch + 1,
                update_step,
                total_updates,
                loss_val,
                lr,
                gpu_mem_mb.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
                cuda_snapshot.map(|v| v.free_mb().to_string()).unwrap_or_else(|| "null".to_string()),
                cuda_snapshot.map(|v| v.used_mb().to_string()).unwrap_or_else(|| "null".to_string()),
                cuda_snapshot.map(|v| v.total_mb().to_string()).unwrap_or_else(|| "null".to_string())
            );
            return Ok(());
        }

        let total = if total_updates == 0 { 1 } else { total_updates };
        let width = 24usize;
        let filled = ((update_step as f64 / total as f64) * width as f64).round() as usize;
        let filled = filled.min(width);
        let empty = width - filled;
        let bar = format!("{}{}", "=".repeat(filled), " ".repeat(empty));
        let percent = ((update_step as f64 / total as f64) * 100.0).round() as i32;
        let gpu_mem = self.query_gpu_memory_mb();
        let cuda_snapshot = if self.trainer_cfg.gpu_mem_snapshot {
            cuda_mem_snapshot(self.cuda_index)
        } else {
            None
        };
        if let Some(mem_mb) = gpu_mem {
            print!(
                "\rEpoch {}/{} [{}] {}/{} ({}%) loss {:.4} lr {:.2e} gpu {} MB",
                epoch + 1,
                self.trainer_cfg.epochs,
                bar,
                update_step,
                total_updates,
                percent,
                loss_val,
                lr,
                mem_mb
            );
        } else {
            print!(
                "\rEpoch {}/{} [{}] {}/{} ({}%) loss {:.4} lr {:.2e}",
                epoch + 1,
                self.trainer_cfg.epochs,
                bar,
                update_step,
                total_updates,
                percent,
                loss_val,
                lr
            );
        }
        if let Some(snapshot) = cuda_snapshot {
            print!(
                " (cuda used {} MB / free {} MB)",
                snapshot.used_mb(),
                snapshot.free_mb()
            );
        }
        io::stdout().flush()?;
        if update_step >= total_updates {
            println!();
        }
        Ok(())
    }

    fn evaluate_loss(&self, dataset: &Dataset, tokenizer: Option<&Tokenizer>) -> Result<f64> {
        if dataset.len() == 0 {
            return Err(anyhow::anyhow!("Empty eval dataset"));
        }
        let num_batches = (dataset.len() + self.trainer_cfg.batch_size - 1) / self.trainer_cfg.batch_size;
        let max_batches = self.trainer_cfg.eval_max_batches.unwrap_or(num_batches);
        let mut total_loss = 0.0;
        let mut batch_count = 0usize;

        for i in 0..num_batches {
            if batch_count >= max_batches {
                break;
            }
            let start_idx = i * self.trainer_cfg.batch_size;
            let (inputs, targets, loss_mask) = dataset.get_batch(
                start_idx,
                self.trainer_cfg.batch_size,
                tokenizer,
                &self.device,
                self.trainer_cfg.pack_sequences,
                self.trainer_cfg.max_seq_len,
                self.trainer_cfg.eos_token_id,
            )?;

            let logits = self.model.forward(&inputs)?;
            let (_, loss_val, mask_sum) = self.compute_loss(&logits, &targets, &loss_mask)?;
            if mask_sum == 0.0 {
                return Err(anyhow::anyhow!(
                    "Eval batch has zero loss mask sum; check dataset/tokenization alignment."
                ));
            }
            drop(logits);
            drop(inputs);
            drop(targets);
            drop(loss_mask);
            total_loss += loss_val;
            batch_count += 1;
        }

        if batch_count == 0 {
            return Err(anyhow::anyhow!("Eval produced no batches"));
        }

        Ok(total_loss / batch_count as f64)
    }

    pub fn save_adapters(&self, path: &std::path::Path) -> Result<()> {
        // Collect only lora tensors
        let mut lora_vars = std::collections::HashMap::new();
        for (name, var) in self
            .varmap
            .data()
            .lock()
            .map_err(|e| anyhow::anyhow!("VarMap lock poisoned in save_adapters: {}", e))?
            .iter()
        {
            if name.contains("lora_a") || name.contains("lora_b") {
                lora_vars.insert(name.clone(), var.as_tensor().clone());
            }
        }

        let out_path = if path.extension().is_some() {
            path.to_path_buf()
        } else {
            std::fs::create_dir_all(path)?;
            path.join("adapter_model.safetensors")
        };

        candle_core::safetensors::save(&lora_vars, &out_path)?;
        println!("Adapters saved to {:?}", out_path);
        Ok(())
    }

    pub fn save_peft_adapter(&self, output_path: &std::path::Path, base_model_path: Option<&str>) -> Result<()> {
        use std::fs;

        fs::create_dir_all(output_path)?;

        // Collect LoRA tensors
        let mut lora_vars = std::collections::HashMap::new();
        let mut target_modules = std::collections::BTreeSet::new();
        for (name, var) in self
            .varmap
            .data()
            .lock()
            .map_err(|e| anyhow::anyhow!("VarMap lock poisoned in save_peft_adapter: {}", e))?
            .iter()
        {
            if name.contains("lora_a") || name.contains("lora_b") {
                lora_vars.insert(name.clone(), var.as_tensor().clone());
                if let Some(module) = Self::lora_target_from_name(name) {
                    target_modules.insert(module);
                }
            }
        }

        // Save adapter weights as safetensors
        let weights_path = output_path.join("adapter_model.safetensors");
        candle_core::safetensors::save(&lora_vars, &weights_path)?;

        // Save adapter config (PEFT-compatible JSON)
        let target_modules: Vec<String> = if target_modules.is_empty() {
            vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ]
        } else {
            target_modules.into_iter().collect()
        };
        let base_model_name = base_model_path.unwrap_or("unknown");
        let adapter_config = serde_json::json!({
            "peft_type": "LORA",
            "base_model_name_or_path": base_model_name,
            "r": self.trainer_cfg.lora_r,
            "lora_alpha": self.trainer_cfg.lora_alpha,
            "lora_dropout": self.trainer_cfg.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        });

        let config_path = output_path.join("adapter_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;

        println!("PEFT adapter saved to {:?}", output_path);
        Ok(())
    }

    fn lora_target_from_name(name: &str) -> Option<String> {
        let parts: Vec<&str> = name.split('.').collect();
        for (idx, part) in parts.iter().enumerate() {
            if *part == "lora_a" || *part == "lora_b" {
                if idx >= 1 {
                    return Some(parts[idx - 1].to_string());
                }
            }
        }
        None
    }

    /// Save checkpoint to disk
    fn save_checkpoint(&self, epoch: usize, best_loss: f64) -> Result<()> {
        use std::fs;

        let checkpoint_dir = self
            .checkpoint_config
            .output_dir
            .join(format!("checkpoint-{}", self.global_step));
        fs::create_dir_all(&checkpoint_dir)?;

        // Save model weights (LoRA adapters)
        let weights_path = checkpoint_dir.join("adapter_model.safetensors");
        let mut lora_vars = std::collections::HashMap::new();
        for (name, var) in self
            .varmap
            .data()
            .lock()
            .map_err(|e| anyhow::anyhow!("VarMap lock poisoned in save_checkpoint: {}", e))?
            .iter()
        {
            if name.contains("lora_a") || name.contains("lora_b") {
                lora_vars.insert(name.clone(), var.as_tensor().clone());
            }
        }
        candle_core::safetensors::save(&lora_vars, &weights_path)?;

        // Create checkpoint metadata
        // NOTE: Optimizer state (Adam moments) and RNG state are not yet serialized.
        // On resume, the optimizer reinitializes from scratch and shuffling order
        // may differ. This can cause a brief loss spike after resuming.
        eprintln!(
            "Warning: Checkpoint does not include optimizer state or RNG state. \
             On resume, optimizer moments will reinitialize and training shuffle order may differ."
        );
        let checkpoint = Checkpoint {
            epoch,
            global_step: self.global_step,
            best_loss,
            optimizer_state: vec![],
            rng_state: None,
        };

        checkpoint.save(&checkpoint_dir)?;
        println!("Checkpoint saved to {:?}", checkpoint_dir);

        Ok(())
    }

    /// Resume training from a checkpoint
    pub fn resume_from_checkpoint(&mut self, checkpoint_path: &std::path::Path) -> Result<()> {
        // Load checkpoint metadata
        let checkpoint = Checkpoint::load(checkpoint_path)?;
        self.global_step = checkpoint.global_step;

        println!("Resuming from checkpoint at step {}", self.global_step);
        println!("Previous best loss: {:.4}", checkpoint.best_loss);

        if checkpoint.optimizer_state.is_empty() {
            eprintln!(
                "Warning: Checkpoint has no saved optimizer state. \
                 Optimizer moments (Adam beta1/beta2) will reinitialize from scratch, \
                 which may cause a brief loss spike."
            );
        }
        if checkpoint.rng_state.is_none() {
            eprintln!(
                "Warning: Checkpoint has no saved RNG state. \
                 Data shuffling order will differ from the original run."
            );
        }

        // Load model weights
        let weights_path = checkpoint_path.join("adapter_model.safetensors");
        if weights_path.exists() {
            let tensors = candle_core::safetensors::load(&weights_path, &self.device)?;

            // Update varmap with loaded tensors
            for (name, tensor) in tensors {
                if let Some(var) = self
                    .varmap
                    .data()
                    .lock()
                    .map_err(|e| anyhow::anyhow!("VarMap lock poisoned in resume_from_checkpoint: {}", e))?
                    .get_mut(&name)
                {
                    var.set(&tensor)?;
                } else {
                    println!("Warning: Checkpoint contains tensor '{}' not found in model", name);
                }
            }

            println!("Loaded adapter weights from {:?}", weights_path);
        } else {
            println!("Warning: No adapter weights found at {:?}", weights_path);
        }

        Ok(())
    }
}
