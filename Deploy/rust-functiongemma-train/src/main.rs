use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

use rust_functiongemma_core::chat_template::{load_chat_template, render_chat_template};
use rust_functiongemma_core::gpu::{
    configure_and_log_cuda_mem_pool, log_cuda_snapshot, query_nvidia_smi, CudaMemPoolConfig,
};
use rust_functiongemma_core::lora_utils::read_lora_config;
use rust_functiongemma_core::{
    collect_model_safetensors, custom_load_verbose, default_dtype, detect_tie_embeddings, is_degenerate_output,
    open_mmaped_safetensors, parse_ggml_dtype, resolve_device_with_index, trim_input_ids, DeviceSelectionParams,
    KvCacheQuant, PcaiConfig,
};
use rust_functiongemma_train::data_gen::DataGenerator;
use rust_functiongemma_train::dataset::Dataset;
use rust_functiongemma_train::early_stopping::EarlyStoppingConfig;
use rust_functiongemma_train::eval::{evaluate_sample, EvaluationMetrics};
use rust_functiongemma_train::router_dataset::{
    build_router_dataset, build_tool_test_vectors, write_jsonl, write_jsonl_streaming, write_test_vectors,
    RouterDatasetConfig,
};
use rust_functiongemma_train::trainer::{Trainer, TrainerConfig};
use rust_functiongemma_train::{Config, LoraSettings, Model};

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct TrainConfigFile {
    force_cpu: bool,
    device: String,
    gpu: Option<usize>,
    cuda_visible_devices: Vec<usize>,
    min_vram_mb: Option<u64>,
    verbose: bool,
    max_seq_len: Option<usize>,
    batch_size: Option<usize>,
    grad_accum: Option<usize>,
    pack_sequences: Option<bool>,
    qlora_block_size: Option<usize>,
    qlora_double_quant: Option<bool>,
    qlora_cache_dequantized: Option<bool>,
    qlora_target: Option<String>,
    use_4bit: bool,
    flash_attn: Option<bool>,
    candle_qmatmul: Option<bool>,
    candle_qmatmul_dtype: Option<String>,
    kv_cache_quant: Option<String>,
    kv_cache_max_len: Option<usize>,
    kv_cache_store: Option<String>,
    kv_cache_streaming: Option<bool>,
    kv_cache_block_len: Option<usize>,
    optimizer: Option<String>,
    cuda_launch_blocking: Option<bool>,
    report_gpu_memory: Option<bool>,
    cuda_mem_pool: Option<bool>,
    cuda_mem_pool_release_threshold_mb: Option<u64>,
    cuda_mem_pool_trim_mb: Option<u64>,
    cuda_mem_snapshot: Option<bool>,
    loss_dtype: Option<String>,
}

impl Default for TrainConfigFile {
    fn default() -> Self {
        Self {
            force_cpu: false,
            device: "auto".to_string(),
            gpu: None,
            cuda_visible_devices: Vec::new(),
            min_vram_mb: None,
            verbose: false,
            max_seq_len: None,
            batch_size: None,
            grad_accum: None,
            pack_sequences: None,
            qlora_block_size: None,
            qlora_double_quant: None,
            qlora_cache_dequantized: None,
            qlora_target: None,
            use_4bit: false,
            flash_attn: None,
            candle_qmatmul: None,
            candle_qmatmul_dtype: None,
            kv_cache_quant: None,
            kv_cache_max_len: None,
            kv_cache_store: None,
            kv_cache_streaming: None,
            kv_cache_block_len: None,
            optimizer: None,
            cuda_launch_blocking: None,
            report_gpu_memory: None,
            cuda_mem_pool: None,
            cuda_mem_pool_release_threshold_mb: None,
            cuda_mem_pool_trim_mb: None,
            cuda_mem_snapshot: None,
            loss_dtype: None,
        }
    }
}

static TRAIN_CONFIG: OnceLock<TrainConfigFile> = OnceLock::new();

fn load_train_config(path: &Path) -> TrainConfigFile {
    match PcaiConfig::load_from(path) {
        Ok(cfg) => serde_json::from_value(cfg.train).unwrap_or_default(),
        Err(_) => TrainConfigFile::default(),
    }
}

fn train_config() -> &'static TrainConfigFile {
    TRAIN_CONFIG.get_or_init(|| load_train_config(&PcaiConfig::config_path()))
}

fn init_train_config<P: AsRef<Path>>(path: P) -> TrainConfigFile {
    if let Some(existing) = TRAIN_CONFIG.get() {
        return existing.clone();
    }
    let loaded = load_train_config(path.as_ref());
    let _ = TRAIN_CONFIG.set(loaded.clone());
    loaded
}

fn resolve_loss_in_f32(device: &Device) -> bool {
    let raw = train_config()
        .loss_dtype
        .as_deref()
        .map(|v| v.trim().to_ascii_lowercase());
    match raw.as_deref() {
        Some("f32") | Some("fp32") => true,
        Some("bf16") => false,
        Some("auto") | None => !device.is_cuda(),
        Some(other) => {
            println!("Warning: Unknown loss_dtype '{}', defaulting to auto.", other);
            !device.is_cuda()
        }
    }
}

fn build_lora_settings(
    r: usize,
    alpha: f64,
    dropout: f64,
    use_4bit: bool,
    allow_candle_qmatmul: bool,
    allow_flash_attn: bool,
) -> LoraSettings {
    let mut settings = LoraSettings::new(r, alpha, dropout, use_4bit);
    let cfg = train_config();
    if let Some(block_size) = cfg.qlora_block_size {
        settings.qlora_block_size = block_size;
    }
    if let Some(double_quant) = cfg.qlora_double_quant {
        settings.qlora_double_quant = double_quant;
    }
    if let Some(cache) = cfg.qlora_cache_dequantized {
        settings.qlora_cache_dequantized = cache;
    }
    if let Some(target) = &cfg.qlora_target {
        if target.eq_ignore_ascii_case("qv") || target.eq_ignore_ascii_case("qv_only") {
            settings.qlora_qv_only = true;
        }
    }
    if allow_candle_qmatmul && cfg.candle_qmatmul.unwrap_or(false) {
        let dtype = parse_ggml_dtype(cfg.candle_qmatmul_dtype.as_deref()).unwrap_or(GgmlDType::Q4_0);
        settings.enable_candle_qmatmul(dtype);
    }
    if allow_flash_attn && cfg.flash_attn.unwrap_or(false) {
        settings.enable_flash_attn();
    }
    settings
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, global = true, default_value = "Config/pcai-functiongemma.json")]
    config: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare training data from tool schema and scenarios
    Prepare {
        #[arg(long)]
        tools: String,
        #[arg(long)]
        output: String,
        #[arg(long)]
        scenarios: Option<String>,
        #[arg(long, default_value = "24")]
        max_cases: usize,
        #[arg(long)]
        system_prompt: Option<String>,
    },
    /// Prepare router-only dataset (tool_calls or NO_TOOL)
    PrepareRouter {
        #[arg(long)]
        tools: String,
        #[arg(long)]
        output: String,
        #[arg(long)]
        diagnose_prompt: String,
        #[arg(long)]
        chat_prompt: String,
        #[arg(long)]
        scenarios: Option<String>,
        #[arg(long, default_value = "24")]
        max_cases: usize,
        #[arg(long)]
        no_tool_coverage: bool,
        #[arg(long)]
        test_vectors: Option<String>,
        #[arg(long)]
        stream: bool,
    },
    /// Train the model using LoRA
    Train {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        train_data: String,
        #[arg(long)]
        eval_data: Option<String>,
        #[arg(long)]
        eval_split: Option<f64>,
        #[arg(long)]
        token_cache: Option<String>,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "1")]
        epochs: usize,
        #[arg(long, default_value = "1e-5")]
        lr: f64,
        #[arg(long, default_value = "16")]
        lora_r: usize,
        #[arg(long, default_value = "32.0")]
        lora_alpha: f64,
        #[arg(long, default_value = "0.0")]
        lora_dropout: f64,
        #[arg(long, default_value = "1")]
        batch_size: usize,
        #[arg(long, default_value = "4")]
        grad_accum: usize,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        no_shuffle: bool,
        #[arg(long)]
        pack_sequences: bool,
        #[arg(long)]
        max_seq_len: Option<usize>,
        #[arg(long, default_value = "1")]
        eos_token_id: u32,
        #[arg(long)]
        use_lora: bool,
        #[arg(long, default_value = "100")]
        warmup_steps: usize,
        #[arg(long, default_value = "cosine")]
        scheduler_type: String,
        #[arg(long, default_value = "0")]
        early_stopping_patience: usize,
        #[arg(long, default_value = "0.001")]
        early_stopping_min_delta: f64,
        #[arg(long)]
        use_4bit: bool,
        #[arg(long)]
        eval_max_batches: Option<usize>,
        #[arg(long, default_value = "1.0")]
        max_grad_norm: f64,
        #[arg(long, default_value = "1")]
        progress_interval: usize,
        #[arg(long)]
        progress_json: bool,
        #[arg(long)]
        optimizer: Option<String>,
    },
    /// Evaluate a trained model or adapter
    Eval {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        test_data: String,
        #[arg(long, default_value = "16")]
        lora_r: usize,
        #[arg(long)]
        adapters: Option<String>,
        #[arg(long)]
        fast_eval: bool,
        #[arg(long)]
        metrics_output: Option<String>,
        #[arg(long, default_value = "64")]
        max_new_tokens: usize,
        #[arg(long, default_value_t = true)]
        schema_validate: bool,
        #[arg(long)]
        samples_output: Option<String>,
        #[arg(long)]
        max_samples: Option<usize>,
        #[arg(long)]
        kv_cache_quant: Option<String>,
        #[arg(long)]
        kv_cache_max_len: Option<usize>,
        #[arg(long)]
        kv_cache_store: Option<String>,
        #[arg(long, default_value_t = false)]
        kv_cache_streaming: bool,
        #[arg(long)]
        kv_cache_block_len: Option<usize>,
        #[arg(long, default_value_t = false)]
        quiet: bool,
        #[arg(long)]
        verbose: bool,
    },
    /// Merge LoRA adapters into the base model
    Merge {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        adapters: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "16")]
        lora_r: usize,
    },
    /// Build a token cache for faster training
    PrepareCache {
        #[arg(long)]
        input: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long)]
        output_dir: String,
        #[arg(long)]
        chat_template: Option<String>,
    },
}

fn select_device_with_index() -> (Device, Option<usize>) {
    let cfg = train_config();
    let params = DeviceSelectionParams {
        device_label: &cfg.device,
        gpu_index: cfg.gpu,
        force_cpu: cfg.force_cpu,
        min_vram_mb: cfg.min_vram_mb,
        cuda_visible_devices: &cfg.cuda_visible_devices,
    };
    let (device, cuda_index) = resolve_device_with_index(&params);
    if let Some(idx) = cuda_index {
        // Log which GPU was selected for visibility during training.
        let gpus = query_nvidia_smi(cfg.min_vram_mb, &cfg.cuda_visible_devices);
        if let Some(info) = gpus.iter().find(|g| g.runtime_index == idx) {
            println!(
                "Selected CUDA device {} (physical {}): {} ({} MB)",
                info.runtime_index, info.physical_index, info.name, info.memory_mb
            );
        }
    }
    (device, cuda_index)
}

fn apply_cuda_mem_pool(cuda_index: Option<usize>) {
    let idx = match cuda_index {
        Some(v) => v,
        None => return,
    };
    if !train_config().cuda_mem_pool.unwrap_or(false) {
        return;
    }
    let cfg = CudaMemPoolConfig {
        enable: true,
        release_threshold_mb: train_config().cuda_mem_pool_release_threshold_mb,
        reuse_follow_event_dependencies: true,
        reuse_allow_opportunistic: true,
        reuse_allow_internal_dependencies: true,
        trim_to_mb: train_config().cuda_mem_pool_trim_mb,
    };
    configure_and_log_cuda_mem_pool(idx, cfg);
}

fn maybe_log_cuda_snapshot(tag: &str, cuda_index: Option<usize>) {
    if !train_config().cuda_mem_snapshot.unwrap_or(false) {
        return;
    }
    log_cuda_snapshot(tag, cuda_index);
}

fn adapter_config_path(adapters: &str) -> PathBuf {
    let path = PathBuf::from(adapters);
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    dir.join("adapter_config.json")
}

fn require_model_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let files = collect_model_safetensors(model_dir);
    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "model safetensors not found under {}",
            model_dir.display()
        ));
    }
    Ok(files)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let _ = init_train_config(&cli.config);

    match cli.command {
        Commands::Prepare {
            tools,
            output,
            scenarios,
            max_cases,
            system_prompt,
        } => {
            println!("Preparing dataset...");
            let generator = DataGenerator::new(
                &PathBuf::from(tools),
                system_prompt.as_ref().map(PathBuf::from).as_deref(),
            )?;

            let mut items = generator.generate_from_schema(max_cases)?;
            if let Some(s_path) = scenarios {
                let s_items = generator.generate_from_scenarios(&PathBuf::from(s_path))?;
                items.extend(s_items);
            }

            let mut out_text = String::new();
            for item in items {
                out_text.push_str(&serde_json::to_string(&item)?);
                out_text.push('\n');
            }
            fs::write(output, out_text)?;
            println!("Dataset prepared successfully.");
        }
        Commands::PrepareRouter {
            tools,
            output,
            diagnose_prompt,
            chat_prompt,
            scenarios,
            max_cases,
            no_tool_coverage,
            test_vectors,
            stream,
        } => {
            println!("Preparing router dataset...");
            let cfg = RouterDatasetConfig {
                output: PathBuf::from(&output),
                tools_path: PathBuf::from(&tools),
                diagnose_prompt: PathBuf::from(&diagnose_prompt),
                chat_prompt: PathBuf::from(&chat_prompt),
                scenarios_path: scenarios.map(PathBuf::from),
                include_tool_coverage: !no_tool_coverage,
                max_cases,
            };

            if stream {
                let count = write_jsonl_streaming(&cfg)?;
                println!("Wrote {} examples to {}", count, cfg.output.display());
            } else {
                let items = build_router_dataset(&cfg)?;
                write_jsonl(&cfg.output, &items)?;
                println!("Wrote {} examples to {}", items.len(), cfg.output.display());
            }

            if let Some(test_vectors_path) = test_vectors {
                let vectors = build_tool_test_vectors(&cfg.tools_path, cfg.max_cases)?;
                write_test_vectors(&PathBuf::from(&test_vectors_path), &vectors)?;
                println!("Wrote {} tool test vectors to {}", vectors.len(), test_vectors_path);
            }
        }
        Commands::Train {
            model_path,
            train_data,
            eval_data,
            eval_split,
            token_cache,
            output,
            epochs,
            lr,
            lora_r,
            lora_alpha,
            lora_dropout,
            batch_size,
            grad_accum,
            seed,
            no_shuffle,
            pack_sequences,
            max_seq_len,
            eos_token_id,
            use_lora,
            warmup_steps,
            scheduler_type,
            early_stopping_patience,
            early_stopping_min_delta,
            use_4bit,
            eval_max_batches,
            max_grad_norm,
            progress_interval,
            progress_json,
            optimizer,
        } => {
            if train_config().cuda_launch_blocking.unwrap_or(false) {
                std::env::set_var("CUDA_LAUNCH_BLOCKING", "1");
            }

            let use_4bit = use_4bit || train_config().use_4bit;
            let batch_size = if batch_size != 1 {
                batch_size
            } else {
                train_config().batch_size.unwrap_or(1)
            };
            let grad_accum = if grad_accum != 4 {
                grad_accum
            } else {
                train_config().grad_accum.unwrap_or(4)
            };
            let pack_sequences = if pack_sequences {
                true
            } else {
                train_config().pack_sequences.unwrap_or(false)
            };
            let max_seq_len = max_seq_len.or(train_config().max_seq_len);
            let (device, cuda_index) = select_device_with_index();
            if matches!(device, Device::Cpu) && !train_config().force_cpu {
                return Err(anyhow::anyhow!(
                    "GPU required for training (force_cpu=false). Update config or provide a CUDA device."
                ));
            }
            println!("Training on device: {:?}", device);
            apply_cuda_mem_pool(cuda_index);
            maybe_log_cuda_snapshot("before_model_load", cuda_index);

            let model_dir = PathBuf::from(&model_path);
            let config_raw = fs::read_to_string(model_dir.join("config.json"))?;
            let config: Config = serde_json::from_str(&config_raw)?;
            let chat_template = load_chat_template(&model_dir).ok();

            let model_files = require_model_safetensors(&model_dir)?;
            let st = open_mmaped_safetensors(&model_files)?;
            let tie_embeddings = detect_tie_embeddings(&st);

            let varmap = VarMap::new();
            // QLoRA requires F32 VarBuilder: LoRA adapter weights must match
            // the F32 dequantized NF4 base weights inside QuantizedLinear.
            // Using BF16 here causes dtype mismatch in matmul (F32 vs BF16).
            let vb_dtype = if use_4bit { DType::F32 } else { default_dtype(&device) };
            let vb = VarBuilder::from_varmap(&varmap, vb_dtype, &device);
            let lora_settings = build_lora_settings(lora_r, lora_alpha, lora_dropout, use_4bit, false, true);

            let model = if use_4bit {
                println!("Loading base weights (QLoRA)...");
                Model::new_qlora(
                    &config,
                    lora_settings,
                    vb,
                    tie_embeddings,
                    &st,
                    &device,
                    default_dtype(&device),
                )?
            } else {
                let model = Model::new(&config, lora_settings, vb, tie_embeddings)?;
                println!("Loading base weights...");
                let _ = custom_load_verbose(&varmap, &model_files, train_config().verbose)?;
                model
            };
            maybe_log_cuda_snapshot("after_model_load", cuda_index);

            let mut eval_dataset = if let Some(eval_path) = &eval_data {
                Some(Dataset::load(&PathBuf::from(eval_path))?.with_chat_template(chat_template.clone()))
            } else {
                None
            };

            let dataset = if let Some(split) = eval_split {
                if split <= 0.0 || split >= 0.5 {
                    println!("Warning: eval_split should be between 0.0 and 0.5; ignoring.");
                    if let Some(cache_dir) = &token_cache {
                        Dataset::load_cached(&PathBuf::from(cache_dir))?.with_chat_template(chat_template.clone())
                    } else {
                        Dataset::load(&PathBuf::from(train_data))?.with_chat_template(chat_template.clone())
                    }
                } else {
                    if token_cache.is_some() {
                        println!("Warning: eval_split requested; ignoring token cache for split.");
                    }
                    let full = Dataset::load(&PathBuf::from(train_data))?.with_chat_template(chat_template.clone());
                    let split_idx = ((1.0 - split) * full.items.len() as f64).round() as usize;
                    let split_idx = split_idx.clamp(1, full.items.len().saturating_sub(1));
                    let eval_items = full.items[split_idx..].to_vec();
                    let train_items = full.items[..split_idx].to_vec();
                    eval_dataset = Some(Dataset {
                        items: eval_items,
                        token_cache: None,
                        chat_template: chat_template.clone(),
                    });
                    Dataset {
                        items: train_items,
                        token_cache: None,
                        chat_template: chat_template.clone(),
                    }
                }
            } else if let Some(cache_dir) = &token_cache {
                Dataset::load_cached(&PathBuf::from(cache_dir))?.with_chat_template(chat_template.clone())
            } else {
                Dataset::load(&PathBuf::from(train_data))?.with_chat_template(chat_template.clone())
            };
            let tokenizer = if token_cache.is_some() {
                None
            } else {
                Some(Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?)
            };

            let early_stopping = if early_stopping_patience > 0 && eval_dataset.is_some() {
                Some(EarlyStoppingConfig {
                    patience: early_stopping_patience,
                    min_delta: early_stopping_min_delta,
                })
            } else {
                None
            };
            if early_stopping_patience > 0 && eval_dataset.is_none() {
                println!("Warning: early stopping configured but no eval dataset provided.");
            }

            let max_grad_norm = if max_grad_norm > 0.0 { Some(max_grad_norm) } else { None };
            let optimizer = optimizer
                .or_else(|| train_config().optimizer.clone())
                .unwrap_or_else(|| "adamw".to_string());
            let report_gpu_memory = train_config().report_gpu_memory.unwrap_or(false);
            let gpu_mem_snapshot = train_config().cuda_mem_snapshot.unwrap_or(false);
            let loss_in_f32 = resolve_loss_in_f32(&device);
            let t_cfg = TrainerConfig {
                lr,
                epochs,
                batch_size,
                grad_accum,
                lora_r,
                lora_alpha,
                lora_dropout,
                pack_sequences,
                max_seq_len,
                eos_token_id,
                use_lora,
                warmup_steps,
                scheduler_type: scheduler_type.clone(),
                early_stopping,
                use_4bit,
                eval_max_batches,
                max_grad_norm,
                shuffle: !no_shuffle,
                seed,
                progress_interval,
                progress_json,
                optimizer,
                report_gpu_memory,
                gpu_mem_snapshot,
                loss_in_f32,
            };
            let mut trainer = Trainer::new(model, &config, t_cfg, device, cuda_index, varmap);

            trainer.train(&dataset, tokenizer.as_ref(), eval_dataset.as_ref())?;

            let output_path = PathBuf::from(&output);
            trainer.save_adapters(&output_path)?;

            let output_dir = if output_path.extension().is_some() {
                output_path
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."))
            } else {
                output_path.clone()
            };

            trainer.save_peft_adapter(&output_dir, Some(&model_path))?;
            write_tokenizer_metadata(&model_dir, &output_dir)?;
        }
        Commands::Eval {
            model_path,
            test_data,
            lora_r,
            adapters,
            fast_eval,
            metrics_output,
            max_new_tokens,
            schema_validate,
            samples_output,
            max_samples,
            kv_cache_quant,
            kv_cache_max_len,
            kv_cache_store,
            kv_cache_streaming,
            kv_cache_block_len,
            quiet,
            verbose,
        } => {
            let verbose = verbose || train_config().verbose;
            let quiet = if quiet { true } else { !verbose };
            let (device, cuda_index) = select_device_with_index();
            apply_cuda_mem_pool(cuda_index);
            maybe_log_cuda_snapshot("eval_before_model_load", cuda_index);
            let model_dir = PathBuf::from(&model_path);
            let config_raw = fs::read_to_string(model_dir.join("config.json"))?;
            let config: Config = serde_json::from_str(&config_raw)?;

            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, default_dtype(&device), &device);
            let mut resolved_r = lora_r;
            let mut resolved_alpha = 32.0;
            let mut resolved_dropout = 0.0;
            if let Some(ref a_path) = adapters {
                if let Some((r, alpha, dropout)) = read_lora_config(&adapter_config_path(a_path)) {
                    resolved_r = r;
                    resolved_alpha = alpha;
                    resolved_dropout = dropout;
                }
            }
            let lora_settings = build_lora_settings(resolved_r, resolved_alpha, resolved_dropout, false, true, true);
            let model = Model::new(&config, lora_settings, vb, true)?;

            let model_files = require_model_safetensors(&model_dir)?;
            let base_loaded = custom_load_verbose(&varmap, &model_files, verbose)?;
            let adapter_loaded = if let Some(a_path) = adapters.as_ref() {
                custom_load_verbose(&varmap, &[PathBuf::from(a_path)], verbose)?
            } else {
                0
            };
            maybe_log_cuda_snapshot("eval_after_load", cuda_index);
            if let Some(samples_path) = samples_output.as_ref() {
                let meta_path = PathBuf::from(samples_path).with_extension("meta.json");
                let total_vars = varmap
                    .data()
                    .lock()
                    .map_err(|e| anyhow::anyhow!("VarMap lock poisoned during eval metadata: {}", e))?
                    .len();
                let meta = json!({
                    "base_loaded": base_loaded,
                    "adapter_loaded": adapter_loaded,
                    "total_vars": total_vars
                });
                fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;
            }

            let dataset = Dataset::load(&PathBuf::from(test_data))?;
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
            let chat_template = load_chat_template(&model_dir).ok();
            let kv_quant = KvCacheQuant::from_str(
                kv_cache_quant
                    .as_deref()
                    .or_else(|| train_config().kv_cache_quant.as_deref()),
            );
            let kv_max_len = kv_cache_max_len.or(train_config().kv_cache_max_len);
            let kv_store = kv_cache_store.clone().or_else(|| train_config().kv_cache_store.clone());
            let kv_store_on_cpu = kv_store
                .as_deref()
                .map(|v| v.eq_ignore_ascii_case("cpu"))
                .unwrap_or(false);
            let kv_streaming = if kv_cache_streaming {
                true
            } else {
                train_config().kv_cache_streaming.unwrap_or(false)
            };
            let kv_block_len = kv_cache_block_len.or(train_config().kv_cache_block_len).unwrap_or(0);

            let mut correct = 0;
            let mut metrics = EvaluationMetrics::default();
            let mut samples_writer = match samples_output.as_ref() {
                Some(path) => Some(BufWriter::new(fs::File::create(path)?)),
                None => None,
            };
            let total = max_samples.map(|v| v.min(dataset.len())).unwrap_or(dataset.len());
            for i in 0..total {
                let item = &dataset.items[i];
                // For eval, we want to prompt with all messages EXCEPT the last model response
                let mut eval_item = item.clone();
                if let Some(last) = eval_item.messages.last_mut() {
                    if last.role == "assistant" || last.role == "model" {
                        eval_item.messages.pop();
                    }
                }

                let messages_value = serde_json::to_value(&eval_item.messages)?;
                let prompt = if let Some(template) = chat_template.as_deref() {
                    render_chat_template(template, &messages_value, &eval_item.tools, true)?
                } else {
                    eval_item.to_prompt() + "<start_of_turn>model\n"
                };
                let encoding = tokenizer.encode(prompt.as_str(), true).map_err(anyhow::Error::msg)?;
                let mut input_ids = encoding.get_ids().to_vec();
                if let Some(max_len) = train_config().max_seq_len {
                    let original_len = input_ids.len();
                    input_ids = trim_input_ids(input_ids, max_len);
                    if !quiet && input_ids.len() < original_len {
                        println!("Truncated prompt from {} to {} tokens for eval", original_len, max_len);
                    }
                }
                let input_len = input_ids.len();
                let input_tensor = Tensor::new(input_ids, &device)?.unsqueeze(0)?;

                if !quiet {
                    println!("\nTest Case {}:", i + 1);
                    let user_text = item.messages.get(0).and_then(|m| m.content.as_deref()).unwrap_or("");
                    println!("User: {}", user_text);
                }

                let output_ids = model.generate_with_cache(
                    &input_tensor,
                    max_new_tokens,
                    &device,
                    kv_quant,
                    kv_max_len,
                    kv_store_on_cpu,
                    kv_streaming,
                    kv_block_len,
                )?;
                let output_text = tokenizer.decode(&output_ids, true).map_err(anyhow::Error::msg)?;
                if is_degenerate_output(&output_text) {
                    return Err(anyhow::anyhow!(
                        "Degenerate model output detected during eval; aborting for debugging."
                    ));
                }

                let expected_text = item.messages.last().and_then(|m| m.content.as_deref()).unwrap_or("");
                if !quiet {
                    println!("Model Output: {}", output_text);
                    println!("Expected: {}", expected_text);
                }
                if let Some(writer) = samples_writer.as_mut() {
                    let expected_tool_calls = item
                        .messages
                        .iter()
                        .find(|m| m.role == "assistant")
                        .and_then(|m| m.tool_calls.clone());
                    let output_raw = tokenizer.decode(&output_ids, false).unwrap_or_default();
                    let output_len = output_ids.len();
                    let record = json!({
                        "index": i,
                        "user": item.messages.get(0).and_then(|m| m.content.as_deref()).unwrap_or(""),
                        "expected": expected_text,
                        "output": output_text,
                        "output_raw": output_raw,
                        "input_len": input_len,
                        "output_len": output_len,
                        "output_ids": output_ids,
                        "expected_tool_calls": expected_tool_calls,
                    });
                    writeln!(writer, "{}", record)?;
                }

                if output_text.contains(expected_text) || expected_text.contains(&output_text) {
                    correct += 1;
                }

                let sample = evaluate_sample(&output_text, item, fast_eval, schema_validate)?;
                metrics.total += 1;
                let expected_tool = item
                    .messages
                    .iter()
                    .find(|m| m.role == "assistant")
                    .and_then(|m| m.tool_calls.as_ref());
                if expected_tool.is_some() {
                    if sample.tool_match {
                        metrics.tool_name_correct += 1;
                    }
                    if sample.arg_match {
                        metrics.arg_exact_match += 1;
                    }
                } else if sample.no_tool_match {
                    metrics.no_tool_correct += 1;
                }
                if schema_validate && !sample.schema_valid {
                    metrics.schema_failures += 1;
                }
            }
            println!("\nEvaluation complete: {}/{} correct", correct, total);
            println!(
                "Tool accuracy: {:.2}%, Arg accuracy: {:.2}%, NO_TOOL correct: {}",
                metrics.tool_accuracy() * 100.0,
                metrics.arg_accuracy() * 100.0,
                metrics.no_tool_correct
            );

            if let Some(out_path) = metrics_output {
                fs::write(out_path, serde_json::to_string_pretty(&metrics)?)?;
            }
        }
        Commands::Merge {
            model_path,
            adapters,
            output,
            lora_r,
        } => {
            let device = Device::Cpu; // Use CPU for merging to avoid OOM
            let model_dir = PathBuf::from(&model_path);
            let config_raw = fs::read_to_string(model_dir.join("config.json"))?;
            let config: Config = serde_json::from_str(&config_raw)?;

            println!("Loading model for merging...");
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
            let mut resolved_r = lora_r;
            let mut resolved_alpha = 32.0;
            let mut resolved_dropout = 0.0;
            if let Some((r, alpha, dropout)) = read_lora_config(&adapter_config_path(&adapters)) {
                resolved_r = r;
                resolved_alpha = alpha;
                resolved_dropout = dropout;
            }
            let lora_settings = build_lora_settings(resolved_r, resolved_alpha, resolved_dropout, false, false, false);
            let mut model = Model::new(&config, lora_settings, vb, true)?;

            let model_files = require_model_safetensors(&model_dir)?;
            let _ = custom_load_verbose(&varmap, &model_files, train_config().verbose)?;
            let _ = custom_load_verbose(&varmap, &[PathBuf::from(adapters)], train_config().verbose)?;

            println!("Merging adapters...");
            model.merge_adapters()?;

            println!("Saving merged model to {}...", output);
            varmap.save(output)?;
            println!("Merged model saved successfully.");
        }
        Commands::PrepareCache {
            input,
            tokenizer,
            output_dir,
            chat_template,
        } => {
            println!("Building token cache...");
            let tokenizer_path = PathBuf::from(&tokenizer);
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
            let template = chat_template
                .as_deref()
                .and_then(|path| fs::read_to_string(path).ok())
                .or_else(|| {
                    tokenizer_path
                        .parent()
                        .map(|dir| dir.join("chat_template.jinja"))
                        .filter(|p| p.exists())
                        .and_then(|p| fs::read_to_string(p).ok())
                });
            let meta = Dataset::build_token_cache(
                &PathBuf::from(&input),
                &tokenizer,
                &tokenizer_path,
                &PathBuf::from(&output_dir),
                template.as_deref(),
            )?;
            println!("Cache built: {} items", meta.item_count);
        }
    }

    Ok(())
}

fn write_tokenizer_metadata(model_dir: &PathBuf, output_dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let tokenizer_src = model_dir.join("tokenizer.json");
    if tokenizer_src.exists() {
        let tokenizer_dst = output_dir.join("tokenizer.json");
        fs::copy(&tokenizer_src, &tokenizer_dst)?;
    }

    let config_src = model_dir.join("tokenizer_config.json");
    if config_src.exists() {
        let config_dst = output_dir.join("tokenizer_config.json");
        fs::copy(&config_src, &config_dst)?;
    }

    let meta = serde_json::json!({
        "tokenizer_source": tokenizer_src.display().to_string(),
        "config_source": if config_src.exists() { config_src.display().to_string() } else { "".to_string() },
        "output_dir": output_dir.display().to_string()
    });
    fs::write(
        output_dir.join("tokenizer_metadata.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}
