# PC_AI FunctionGemma Consolidation Context - 2026-02-21

## Project State

**Branch:** main @ 4428af0
**Phase:** Cross-crate code consolidation complete (Tasks 1-42)
**Status:** All three FunctionGemma Rust crates clean, ~700+ lines of duplication eliminated

## Architecture After Consolidation

```
Deploy/rust-functiongemma-core/     # Shared library - authoritative implementations
  src/config.rs                     # PcaiConfig, RuntimeConfig (unified, 45+ fields)
  src/gpu.rs                        # DeviceSelectionParams, resolve_device_with_index, nvidia-smi, CUDA helpers
  src/model.rs                      # Model struct, generate, KV cache, LoRA
  src/prompt.rs                     # Function call format/parse, is_degenerate_output, trim_input_ids
  src/safetensors_utils.rs          # collect/open_mmaped, custom_load(_verbose), detect_prefix/tie_embeddings
  src/lora_utils.rs                 # LoraInfo, resolve_lora_from_path, read_lora_config
  src/chat_template.rs              # Jinja chat template rendering
  src/error.rs                      # PcaiError

Deploy/rust-functiongemma-runtime/  # HTTP router runtime - thin adapters over core
  src/config.rs                     # ~99 lines, uses PcaiConfig + RuntimeConfig from core
  src/auth.rs                       # Bearer token middleware
  src/handlers.rs                   # HTTP handlers (health, models, chat)
  src/inference.rs                  # ModelCache with OnceLock, infer_with_model
  src/gpu.rs                        # Thin wrappers + OnceLock caches calling core GPU
  src/routing.rs                    # Heuristic routing, tool selection
  src/model_support.rs              # resolve_model_path only (dead code removed)
  src/types.rs                      # Request/response types
  src/metrics.rs                    # Atomics, system metrics
  src/error.rs                      # ApiError

Deploy/rust-functiongemma-train/    # Training pipeline
  src/main.rs                       # TrainConfigFile (train-specific), CLI, train/eval/merge commands
  src/data_gen.rs                   # DataGenerator, Message, Scenario, TrainingItem
  src/dataset.rs                    # Dataset, tokenization
  src/router_dataset.rs             # Router-specific dataset generation
  src/eval.rs                       # Evaluation metrics
  src/trainer.rs                    # Trainer, TrainerConfig
```

## Consolidation Summary (42 Tasks)

### What was extracted to core:
- RuntimeConfig (unified with 3 added fields: api_key, queue_depth, timeout)
- PcaiConfig::config_path() replaces local default_config_path in both crates
- DeviceSelectionParams + resolve_device_with_index (parameterized device selection)
- open_mmaped_safetensors + detect_tie_embeddings
- custom_load + custom_load_verbose
- collect_model_safetensors + detect_safetensors_prefix
- parse_ggml_dtype, default_dtype, normalize_device_label, parse_cuda_index
- auto_cuda_index, query_nvidia_smi (parameterized for config independence)
- is_degenerate_output, trim_input_ids
- configure_and_log_cuda_mem_pool, log_cuda_snapshot
- LoraInfo, resolve_lora_from_path, read_lora_config

### What was eliminated:
- RuntimeConfigFile (36 fields + Default) from runtime - now uses core's RuntimeConfig
- AppConfigFile wrapper from both runtime and training
- AdapterConfigFile + resolve_adapter_config from training
- Dead ModelAssets struct/impl + load_safetensors_summary from runtime
- Unused detect_safetensors_prefix import from training
- parse_ggml_dtype relay through runtime/config.rs
- 3x duplicated safetensors loading block in training (extracted to require_model_safetensors)
- map_role deduplication + Scenario struct unification within training crate

### Remaining opportunities (lower priority):
- Message struct duplicated between runtime/types.rs and training/data_gen.rs
- train/lora.rs: 208-line standalone LoRA impl (dead production code, used in tests only)
- TrainConfigFile could be typed in core like RuntimeConfig (high effort)

## Build Environment
- CUDA v13.1 has fatal nvcc error preventing cargo check/build/test
- All verification done via `rustfmt --check --edition 2021`
- Build system: `.\Build.ps1 -Component inference -EnableCuda`

## Key Patterns
- OnceLock for process-lifetime caches (config, nvidia-smi, model, tools)
- unsafe Send+Sync for ModelCache (candle Model has opaque types)
- Feature-gated: `#[cfg(feature = "model")]` for GPU/inference code
- Auth: config `api_key` + `PCAI_API_KEY` env var fallback
- Config loading: PcaiConfig::load_from() with env var overrides

## Agent Work Registry

| Agent | Tasks | Key Contributions |
|-------|-------|-------------------|
| rust-pro (sonnet) | 31-35 | Parallel file modifications across crates |
| Explore (sonnet) | Scan 1, Scan 2 | Comprehensive duplication analysis |
| Main (opus) | 36-42 | Direct edits, evaluation, cleanup |

## Files Changed (this consolidation arc)

### Core crate (+364 lines net):
- src/config.rs: +14 (api_key, queue_depth, timeout fields)
- src/gpu.rs: +299 (DeviceSelectionParams, resolve_device_with_index, nvidia-smi, CUDA helpers)
- src/prompt.rs: +36 (trim_input_ids)
- src/lib.rs: +14 (re-exports)
- src/safetensors_utils.rs: new (open_mmaped, detect_tie_embeddings)
- src/lora_utils.rs: new (LoraInfo, read_lora_config, resolve_lora_from_path)

### Runtime crate (-255 lines net):
- src/lib.rs: -834 lines (decomposed into modules)
- src/config.rs: -91 lines (RuntimeConfigFile eliminated)
- src/model_support.rs: -35 lines (dead code removed)

### Training crate (+746 lines net, mostly new features):
- src/main.rs: major refactoring (config, device, adapter, safetensors consolidated)
- src/data_gen.rs: +382 (new features)
- src/dataset.rs: +451 (new features)
- src/eval.rs: +293 (new features)
