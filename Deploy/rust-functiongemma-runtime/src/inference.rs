#![cfg(feature = "model")]

use crate::config::{default_model, router_lora_path_override, router_model_path_override, runtime_config};
use crate::gpu::{
    default_dtype, maybe_configure_cuda_mem_pool, maybe_log_cuda_snapshot, resolve_cuda_index_for_config,
    resolve_device,
};
use crate::routing::build_tool_calls;
use crate::types::{InferenceResult, LoraMetadata, Message};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rust_functiongemma_core::chat_template::render_chat_prompt;
use rust_functiongemma_core::lora_utils::resolve_lora_from_path;
use rust_functiongemma_core::prompt::parse_function_call;
use rust_functiongemma_core::{
    collect_model_safetensors, custom_load, detect_tie_embeddings, is_degenerate_output, open_mmaped_safetensors,
    parse_ggml_dtype, trim_input_ids, Config, KvCacheQuant, LoraInfo, LoraSettings, Model, PreAllocKvCache,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::OnceLock;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::metrics::{TOTAL_INFER_MS, TOTAL_REQUESTS, TOTAL_TOKENS};

// ---------------------------------------------------------------------------
// Model cache
// ---------------------------------------------------------------------------

/// All per-process model state loaded once at first inference and reused for
/// every subsequent request.
struct ModelCache {
    model: Model,
    tokenizer: Tokenizer,
    config: Config,
    tie_embeddings: bool,
    lora_loaded: bool,
}

// SAFETY: `Model` is composed entirely of `candle_core::Tensor` (which uses
// `Arc` storage and is `Send + Sync`) and standard `Arc<Mutex<_>>` wrappers
// (`RotaryEmbedding`).  `Tokenizer` and `Config` are also `Send + Sync`.
// The compiler cannot derive the impls automatically because some transitive
// types are opaque, so we assert them here.
unsafe impl Send for ModelCache {}
unsafe impl Sync for ModelCache {}

/// Stores either the successfully initialised cache or the error message that
/// prevented initialisation.  Using `String` for the error keeps the type
/// `Send + Sync`, which is required for a `static`.
static MODEL_CACHE: OnceLock<Result<ModelCache, String>> = OnceLock::new();

/// Returns a reference to the lazily initialised [`ModelCache`], loading the
/// model on first call and returning a cached reference on every subsequent
/// call.
///
/// # Errors
///
/// Returns an error if the model files cannot be found or loaded.  Once an
/// error has been recorded the same error message is returned on every future
/// call without re-attempting to load.
fn get_or_init_model() -> anyhow::Result<&'static ModelCache> {
    let result = MODEL_CACHE.get_or_init(|| load_model_cache().map_err(|e| e.to_string()));
    result
        .as_ref()
        .map_err(|e| anyhow::anyhow!("model initialisation failed: {}", e))
}

/// Performs the one-time expensive model loading and returns a [`ModelCache`].
fn load_model_cache() -> anyhow::Result<ModelCache> {
    let model_id = router_model_path_override().unwrap_or_else(default_model);
    let model_path = crate::model_support::resolve_model_path(&model_id)?;
    let lora_info = resolve_lora_adapter(&model_path);
    let lora_r = lora_info.as_ref().map(|l| l.r).unwrap_or(0);
    let lora_alpha = lora_info.as_ref().map(|l| l.alpha).unwrap_or(32.0);
    let lora_dropout = lora_info.as_ref().map(|l| l.dropout).unwrap_or(0.0);
    let lora_loaded = lora_info.is_some();

    let config_raw = std::fs::read_to_string(model_path.join("config.json"))?;
    let config: Config = serde_json::from_str(&config_raw)?;

    let device = resolve_device();
    let device_index = if device.is_cuda() {
        resolve_cuda_index_for_config()
    } else {
        None
    };
    maybe_configure_cuda_mem_pool(device_index);
    maybe_log_cuda_snapshot("before_model_load", device_index);

    // Preflight VRAM check: verify sufficient free memory before loading weights.
    // This turns opaque CUDA OOM crashes into actionable error messages.
    if let Some(idx) = device_index {
        if let Some(snapshot) = rust_functiongemma_core::gpu::cuda_mem_snapshot(Some(idx)) {
            let free_mb = snapshot.free_mb();
            let min_mb = runtime_config().min_vram_mb.unwrap_or(0);
            tracing::info!(
                gpu = idx,
                free_mb = free_mb,
                total_mb = snapshot.total_mb(),
                used_mb = snapshot.used_mb(),
                min_vram_mb = min_mb,
                "preflight VRAM check"
            );
            if min_mb > 0 && free_mb < min_mb {
                return Err(anyhow::anyhow!(
                    "Preflight VRAM check failed: GPU{} has {}MB free but {}MB required. \
                     Unload other models or reduce min_vram_mb in config.",
                    idx,
                    free_mb,
                    min_mb
                ));
            }
        }
    }

    let model_files = collect_model_safetensors(&model_path);
    if model_files.is_empty() {
        return Err(anyhow::anyhow!(
            "model safetensors not found under {}",
            model_path.display()
        ));
    }
    let st = open_mmaped_safetensors(&model_files)?;
    let tie_embeddings = detect_tie_embeddings(&st);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, default_dtype(&device), &device);
    let mut lora_settings = LoraSettings::new(lora_r, lora_alpha, lora_dropout, false);
    if runtime_config().router_flash_attn {
        lora_settings.enable_flash_attn();
    }
    if runtime_config().router_candle_qmatmul {
        let dtype = parse_ggml_dtype(runtime_config().router_candle_qmatmul_dtype.as_deref())
            .unwrap_or(candle_core::quantized::GgmlDType::Q4_0);
        lora_settings.enable_candle_qmatmul(dtype);
    }
    let model = Model::new(&config, lora_settings, vb, tie_embeddings)?;
    maybe_log_cuda_snapshot("after_model_init", device_index);
    custom_load(&varmap, &model_files)?;
    if let Some(lora) = lora_info {
        custom_load(&varmap, &[lora.path])?;
    }
    maybe_log_cuda_snapshot("after_weights_load", device_index);

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;

    Ok(ModelCache {
        model,
        tokenizer,
        config,
        tie_embeddings,
        lora_loaded,
    })
}

pub(crate) fn resolve_lora_adapter(model_path: &Path) -> Option<LoraInfo> {
    if let Some(path) = router_lora_path_override() {
        if let Some(info) = resolve_lora_from_path(path) {
            return Some(info);
        }
    }

    let candidate = model_path.join("adapter_model.safetensors");
    if candidate.exists() {
        return resolve_lora_from_path(candidate);
    }

    None
}

pub(crate) fn lora_metadata() -> Option<crate::types::LoraMetadata> {
    let from_config = router_lora_path_override();
    let from_model = router_model_path_override().map(|p| PathBuf::from(p).join("adapter_model.safetensors"));
    let candidate = from_config.or(from_model)?;
    let info = resolve_lora_from_path(candidate)?;
    Some(LoraMetadata {
        path: info.path.to_string_lossy().to_string(),
        r: info.r,
    })
}

pub(crate) fn render_prompt(messages: &[Message], tools: &Value) -> anyhow::Result<String> {
    let model_id = router_model_path_override().unwrap_or_else(default_model);
    let model_path = crate::model_support::resolve_model_path(&model_id)?;
    let messages_value = serde_json::to_value(messages)?;
    render_chat_prompt(&model_path, &messages_value, tools, true)
}

pub(crate) fn use_kv_cache() -> bool {
    runtime_config().router_kv_cache
}

pub(crate) fn infer_with_model(req: &crate::types::ChatCompletionRequest) -> anyhow::Result<InferenceResult> {
    let cache = get_or_init_model()?;

    let prompt = match req.tools.as_ref() {
        Some(tools) => render_prompt(&req.messages, tools)?,
        None => render_prompt(&req.messages, &json!([]))?,
    };

    let device = resolve_device();
    let device_index = if device.is_cuda() {
        resolve_cuda_index_for_config()
    } else {
        None
    };

    // Resolve the effective seed: per-request seed takes priority, then the
    // config-level default.  With greedy (argmax) decoding the seed does not
    // affect output, but we log it for traceability.
    let effective_seed: Option<u64> = req
        .seed
        .map(|s| s as u64)
        .or(runtime_config().router_seed);
    let effective_temp = req
        .temperature
        .unwrap_or(runtime_config().router_default_temperature);
    tracing::debug!(
        seed = ?effective_seed,
        temperature = effective_temp,
        "deterministic generation settings"
    );

    let encoding = cache.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut input_ids = encoding.get_ids().to_vec();
    if let Some(max_len) = runtime_config().router_max_seq_len {
        input_ids = trim_input_ids(input_ids, max_len);
    }
    let input_tensor = Tensor::new(input_ids, &device)?.unsqueeze(0)?;

    let max_tokens = req.max_tokens.unwrap_or(runtime_config().router_default_max_tokens) as usize;
    let start = Instant::now();
    let kv_quant = KvCacheQuant::from_str(runtime_config().router_kv_cache_quant.as_deref());
    let kv_max_len = runtime_config().router_kv_cache_max_len;
    let kv_store_on_cpu = runtime_config().router_kv_cache_store.eq_ignore_ascii_case("cpu");
    maybe_log_cuda_snapshot("before_generate", device_index);
    let output_ids = if use_kv_cache() {
        // Use the pre-allocated ring-buffer KV cache when int8 quantisation
        // is not requested (int8 requires per-step re-quantisation with a
        // varying scale factor, which is incompatible with the fixed buffer).
        let use_prealloc = !matches!(kv_quant, KvCacheQuant::Int8) && !kv_store_on_cpu;
        if use_prealloc {
            let prealloc_max = kv_max_len.unwrap_or(cache.config.num_hidden_layers.max(4096));
            cache.model.generate_with_prealloc_cache(
                &input_tensor,
                max_tokens,
                &device,
                &cache.config,
                prealloc_max,
            )?
        } else {
            cache.model.generate_with_cache(
                &input_tensor,
                max_tokens,
                &device,
                kv_quant,
                kv_max_len,
                kv_store_on_cpu,
            )?
        }
    } else {
        cache.model.generate(&input_tensor, max_tokens, &device)?
    };
    drop(input_tensor);
    maybe_log_cuda_snapshot("after_generate", device_index);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    TOTAL_TOKENS.fetch_add(output_ids.len() as u64, Ordering::Relaxed);
    TOTAL_INFER_MS.fetch_add(elapsed_ms, Ordering::Relaxed);
    let output_text = cache.tokenizer.decode(&output_ids, true).map_err(anyhow::Error::msg)?;
    drop(output_ids);

    if is_degenerate_output(&output_text) {
        return Err(anyhow::anyhow!("model output was empty/degenerate"));
    }

    if let Some((name, args)) = parse_function_call(&output_text) {
        let calls = build_tool_calls(&name, args, "call_model");
        return Ok(InferenceResult {
            content: None,
            tool_calls: Some(calls),
        });
    }

    if output_text.contains("NO_TOOL") {
        Ok(InferenceResult {
            content: Some("NO_TOOL".to_string()),
            tool_calls: None,
        })
    } else {
        Ok(InferenceResult {
            content: Some(output_text),
            tool_calls: None,
        })
    }
}
