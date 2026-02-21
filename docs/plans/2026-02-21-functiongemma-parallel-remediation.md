# FunctionGemma Parallel Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate duplicated FunctionGemma code, implement QLoRA merge, expand training data, and add production runtime features (queuing, streaming, async FFI).

**Architecture:** Three parallel workstreams with consolidation gating training model changes. The training model.rs (921 LOC) is an older copy of core/model.rs (1113 LOC) missing flash attention and smarter rotary caching. Training pipeline already has prompt masking wired correctly. QLoRA merge CLI exists but errors on quantized weights. Runtime needs concurrency, streaming, and better error handling.

**Tech Stack:** Rust (candle 0.9.2, axum 0.7, tokio 1.41), PowerShell, CUDA via cudarc

---

## Workstream 1: Codebase Consolidation

### Task 1: Delete training model.rs and import from core

**Files:**
- Delete: `Deploy/rust-functiongemma-train/src/model.rs` (921 lines)
- Modify: `Deploy/rust-functiongemma-train/src/lib.rs`
- Modify: `Deploy/rust-functiongemma-train/src/trainer.rs`
- Modify: `Deploy/rust-functiongemma-train/src/main.rs`
- Modify: `Deploy/rust-functiongemma-train/Cargo.toml`
- Test: `Deploy/rust-functiongemma-train/` (cargo test)

**Context:** The training model.rs is 921 LOC that duplicates core/model.rs (1113 LOC) but with:
- Simpler RotaryEmbedding cache (HashMap vs Option tuple) at lines 263-304
- No flash attention support
- No flash_attn function stubs
- Simpler KvCache without quantization at lines 734-746
- Different forward_with_cache signatures

The core version is strictly superior: smarter cache (reuses if >= seq_len), flash attention support, quantized KV cache.

**Step 1: Verify training already depends on core**

Check `Deploy/rust-functiongemma-train/Cargo.toml` confirms:
```toml
rust-functiongemma-core = { path = "../rust-functiongemma-core", features = ["flash-attn"] }
```
Expected: dependency exists.

**Step 2: Audit all imports of training model.rs**

Run: `grep -rn "use crate::model\|mod model\|model::" Deploy/rust-functiongemma-train/src/ --include="*.rs"`

Map every usage of the local model module to the equivalent in core. Key mappings:
- `crate::model::Config` -> `rust_functiongemma_core::model::Config`
- `crate::model::Model` -> `rust_functiongemma_core::model::Model`
- `crate::model::LoraSettings` -> `rust_functiongemma_core::model::LoraSettings`
- `crate::model::LoraLinear` -> `rust_functiongemma_core::model::LoraLinear`
- `crate::model::KvCache` -> `rust_functiongemma_core::model::KvCache`

**Step 3: Update all imports in training crate**

Replace `crate::model::*` with `rust_functiongemma_core::*` (or a re-export alias). In `lib.rs`, remove `pub mod model;` and add:
```rust
pub use rust_functiongemma_core::model;
```

**Step 4: Handle API differences**

The core `Config` has `use_flash_attn: bool` field (line 346 core vs absent in train). Update all `Config` construction in train to include this field. In trainer.rs, set `config.enable_flash_attn()` when the training config enables flash attention.

The core `Attention` struct has `use_flash_attn: bool` (line 330 core). This is set by `Config` so no training code changes needed.

**Step 5: Delete training model.rs**

```bash
git rm Deploy/rust-functiongemma-train/src/model.rs
```

**Step 6: Build and test**

Run: `cargo build -p rust-functiongemma-train`
Run: `cargo test -p rust-functiongemma-train`
Expected: All compile and tests pass.

**Step 7: Commit**

```bash
git add Deploy/rust-functiongemma-train/
git commit -m "refactor: remove duplicate model.rs from training, import from core"
```

---

### Task 2: Extract shared error and config types into core

**Files:**
- Create: `Deploy/rust-functiongemma-core/src/error.rs`
- Create: `Deploy/rust-functiongemma-core/src/config.rs`
- Modify: `Deploy/rust-functiongemma-core/src/lib.rs`
- Modify: `Deploy/rust-functiongemma-core/Cargo.toml`
- Test: `Deploy/rust-functiongemma-core/` (cargo test)

**Step 1: Write failing test for config loading with env override**

Create test in `Deploy/rust-functiongemma-core/src/config.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config_from_path() {
        let config = PcaiConfig::load("../../Config/pcai-functiongemma.json");
        assert!(config.is_ok());
    }

    #[test]
    fn test_env_override_router_addr() {
        std::env::set_var("PCAI_ROUTER_ADDR", "0.0.0.0:9000");
        let config = PcaiConfig::load("../../Config/pcai-functiongemma.json").unwrap();
        assert_eq!(config.router.addr, "0.0.0.0:9000");
        std::env::remove_var("PCAI_ROUTER_ADDR");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rust-functiongemma-core test_load_config`
Expected: FAIL (module doesn't exist yet)

**Step 3: Implement PcaiConfig**

```rust
// Deploy/rust-functiongemma-core/src/config.rs
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct PcaiConfig {
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub train: serde_json::Value,
    #[serde(default)]
    pub inference: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RouterConfig {
    #[serde(default = "default_router_addr")]
    pub addr: String,
    pub model_path: Option<String>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_queue_depth")]
    pub request_queue_depth: usize,
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
}

fn default_router_addr() -> String { "127.0.0.1:8000".to_string() }
fn default_temperature() -> f64 { 0.1 }
fn default_max_tokens() -> u32 { 64 }
fn default_queue_depth() -> usize { 8 }
fn default_timeout() -> u64 { 30 }

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            addr: default_router_addr(),
            model_path: None,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            request_queue_depth: default_queue_depth(),
            request_timeout_secs: default_timeout(),
        }
    }
}

impl PcaiConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = std::env::var("PCAI_CONFIG_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| path.as_ref().to_path_buf());

        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        let mut config: Self = serde_json::from_str(&raw)?;

        // Apply env overrides
        if let Ok(addr) = std::env::var("PCAI_ROUTER_ADDR") {
            config.router.addr = addr;
        }
        if let Ok(model) = std::env::var("PCAI_ROUTER_MODEL") {
            config.router.model_path = Some(model);
        }

        Ok(config)
    }
}
```

**Step 4: Implement PcaiError**

```rust
// Deploy/rust-functiongemma-core/src/error.rs
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum PcaiError {
    #[error("Config error: {0}")]
    Config(#[from] anyhow::Error),
    #[error("Model error: {source}")]
    Model { source: candle_core::Error },
    #[error("Template not found: {path}")]
    TemplateNotFound { path: PathBuf, source: std::io::Error },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl From<candle_core::Error> for PcaiError {
    fn from(e: candle_core::Error) -> Self {
        PcaiError::Model { source: e }
    }
}
```

**Step 5: Add thiserror dependency and update lib.rs**

In `Cargo.toml`, add: `thiserror = "2"`

In `lib.rs`, add:
```rust
pub mod config;
pub mod error;
```

**Step 6: Run tests**

Run: `cargo test -p rust-functiongemma-core`
Expected: PASS

**Step 7: Commit**

```bash
git add Deploy/rust-functiongemma-core/
git commit -m "feat(core): add shared PcaiConfig with env overrides and PcaiError types"
```

---

### Task 3: Add token cache version validation

**Files:**
- Modify: `Deploy/rust-functiongemma-train/src/dataset.rs:69-114` (build_token_cache)
- Modify: `Deploy/rust-functiongemma-train/src/dataset.rs:418-440` (get_ids/get_mask loading)
- Test: `Deploy/rust-functiongemma-train/` (cargo test)

**Step 1: Write failing test for cache validation**

```rust
#[test]
fn test_token_cache_detects_stale_tokenizer() {
    // Build cache with tokenizer A
    // Try to load cache with tokenizer B (different vocab)
    // Should return Err with "tokenizer mismatch" message
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p rust-functiongemma-train test_token_cache_detects_stale`
Expected: FAIL

**Step 3: Add version header to token cache**

In `build_token_cache()` at `dataset.rs:69`, before writing tokens, write a 32-byte header:

```rust
const CACHE_MAGIC: &[u8; 4] = b"PCAI";
const CACHE_VERSION: u32 = 1;

fn compute_tokenizer_hash(tokenizer_path: &Path) -> [u8; 16] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let content = std::fs::read(tokenizer_path).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let h = hasher.finish();
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h.to_le_bytes());
    out
}
```

Write header at start of `tokens.bin` and `tokens.mask.bin`. On load, validate header matches.

**Step 4: Update loading to validate header**

In the `TokenCache::open()` or equivalent, read first 32 bytes, validate magic + version + tokenizer hash. If mismatch, return error suggesting rebuild.

**Step 5: Run tests**

Run: `cargo test -p rust-functiongemma-train`
Expected: PASS

**Step 6: Commit**

```bash
git add Deploy/rust-functiongemma-train/src/dataset.rs
git commit -m "feat(train): add token cache version validation with tokenizer hash"
```

---

## Workstream 2: Training Quality (can start Tasks 4-5 immediately; Task 6 after Task 1)

### Task 4: Expand training dataset with hard negatives and multi-tool examples

**Files:**
- Modify: `Deploy/rust-functiongemma-train/src/data_gen.rs:80-172`
- Modify: `Deploy/rust-functiongemma-train/examples/scenarios.json`
- Create: `Deploy/rust-functiongemma-train/examples/negative_scenarios.json`
- Create: `Deploy/rust-functiongemma-train/examples/multi_tool_scenarios.json`
- Test: `Deploy/rust-functiongemma-train/` (cargo test)

**Step 1: Write test for expanded dataset generation**

```rust
#[test]
fn test_generate_includes_hard_negatives() {
    let gen = DatasetGenerator::new(tools, system_msg);
    let items = gen.generate_from_schema(10).unwrap();
    let no_tool_count = items.iter().filter(|i| {
        i.messages.iter().any(|m| m.role == "assistant" && m.content.as_deref() == Some("NO_TOOL"))
    }).count();
    assert!(no_tool_count > 0, "Must include hard negative (NO_TOOL) examples");
}

#[test]
fn test_generate_includes_ambiguous_examples() {
    let gen = DatasetGenerator::new(tools, system_msg);
    let items = gen.generate_from_schema(10).unwrap();
    // At least some examples should have multiple candidate tools mentioned in the user prompt
    assert!(items.len() >= 5);
}
```

**Step 2: Create negative_scenarios.json**

Add 40 scenarios for queries that should NOT invoke any tool:
```json
{
  "scenarios": [
    { "mode": "no_tool", "user_content": "What is the weather like today?", "assistant_content": "NO_TOOL" },
    { "mode": "no_tool", "user_content": "Tell me a joke about computers", "assistant_content": "NO_TOOL" },
    { "mode": "no_tool", "user_content": "Thanks, that fixed my issue!", "assistant_content": "NO_TOOL" }
  ]
}
```

**Step 3: Create multi_tool_scenarios.json**

Add 30 scenarios needing 2+ tools:
```json
{
  "scenarios": [
    {
      "mode": "multi_tool",
      "user_content": "Check my disk health and also look for any USB errors",
      "tool_sequence": ["Get-DiskSmartStatus", "Get-UsbDeviceErrors"]
    }
  ]
}
```

**Step 4: Update data_gen.rs to load additional scenario files**

In `generate_from_schema()`, after existing negative cases (line ~135), add loading from the new scenario files. Accept an optional `--extra-scenarios` CLI arg.

**Step 5: Run dataset generation and verify counts**

Run: `cargo run -p rust-functiongemma-train -- prepare-router --tools Config/pcai-tools.json --output /tmp/test.jsonl --diagnose-prompt DIAGNOSE.md --chat-prompt CHAT.md --no-tool-coverage`
Expected: Output JSONL with >150 examples including NO_TOOL entries.

**Step 6: Commit**

```bash
git add Deploy/rust-functiongemma-train/
git commit -m "feat(train): expand dataset with 40 hard negatives and 30 multi-tool scenarios"
```

---

### Task 5: Improve evaluation metrics with confusion matrix

**Files:**
- Modify: `Deploy/rust-functiongemma-train/src/eval.rs:7-88`
- Test: `Deploy/rust-functiongemma-train/` (cargo test)

**Step 1: Write failing test for confusion matrix**

```rust
#[test]
fn test_eval_produces_confusion_matrix() {
    let mut metrics = EvaluationMetrics::default();
    metrics.record_prediction("Get-DiskHealth", "Get-DiskHealth"); // correct
    metrics.record_prediction("Get-UsbErrors", "Get-DiskHealth"); // wrong

    let cm = metrics.confusion_matrix();
    assert_eq!(cm[&("Get-DiskHealth".to_string(), "Get-DiskHealth".to_string())], 1);
    assert_eq!(cm[&("Get-UsbErrors".to_string(), "Get-DiskHealth".to_string())], 1);
}

#[test]
fn test_per_tool_f1() {
    let mut metrics = EvaluationMetrics::default();
    // ... populate with test data
    let f1 = metrics.per_tool_f1();
    assert!(f1.contains_key("Get-DiskHealth"));
    assert!(f1["Get-DiskHealth"] > 0.0);
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p rust-functiongemma-train test_eval_produces`
Expected: FAIL

**Step 3: Add confusion matrix and per-tool F1 to EvaluationMetrics**

Extend the struct at `eval.rs:7`:

```rust
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub total: usize,
    pub tool_name_correct: usize,
    pub arg_exact_match: usize,
    pub no_tool_correct: usize,
    pub schema_failures: usize,
    // NEW fields:
    #[serde(default)]
    pub predictions: Vec<(String, String)>, // (expected, predicted)
}

impl EvaluationMetrics {
    // existing methods...

    pub fn record_prediction(&mut self, expected: &str, predicted: &str) {
        self.predictions.push((expected.to_string(), predicted.to_string()));
    }

    pub fn confusion_matrix(&self) -> std::collections::HashMap<(String, String), u32> {
        let mut cm = std::collections::HashMap::new();
        for (exp, pred) in &self.predictions {
            *cm.entry((exp.clone(), pred.clone())).or_insert(0) += 1;
        }
        cm
    }

    pub fn per_tool_f1(&self) -> std::collections::HashMap<String, f64> {
        let mut tp: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut fp: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut fn_: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

        for (exp, pred) in &self.predictions {
            if exp == pred {
                *tp.entry(exp.clone()).or_insert(0) += 1;
            } else {
                *fp.entry(pred.clone()).or_insert(0) += 1;
                *fn_.entry(exp.clone()).or_insert(0) += 1;
            }
        }

        let all_tools: std::collections::HashSet<_> = tp.keys()
            .chain(fp.keys())
            .chain(fn_.keys())
            .cloned()
            .collect();

        all_tools.into_iter().map(|tool| {
            let t = *tp.get(&tool).unwrap_or(&0) as f64;
            let f = *fp.get(&tool).unwrap_or(&0) as f64;
            let n = *fn_.get(&tool).unwrap_or(&0) as f64;
            let precision = if t + f > 0.0 { t / (t + f) } else { 0.0 };
            let recall = if t + n > 0.0 { t / (t + n) } else { 0.0 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            (tool, f1)
        }).collect()
    }
}
```

**Step 4: Wire into evaluate_sample to call record_prediction**

In `evaluate_sample()` at eval.rs:33, add prediction recording to the metrics accumulator.

**Step 5: Run tests**

Run: `cargo test -p rust-functiongemma-train`
Expected: PASS

**Step 6: Commit**

```bash
git add Deploy/rust-functiongemma-train/src/eval.rs
git commit -m "feat(eval): add confusion matrix and per-tool F1 scores"
```

---

### Task 6: Implement QLoRA merge (after Task 1)

**Files:**
- Modify: `Deploy/rust-functiongemma-core/src/model.rs:136-146` (LoraLinear::merge)
- Test: `Deploy/rust-functiongemma-core/` (cargo test)

**Context:** The `merge()` method at core/model.rs:136 currently errors on QLoRA:
```rust
if self.qlora.is_some() {
    return Err(candle_core::Error::msg("QLoRA merge is not supported..."));
}
```

The Merge CLI subcommand exists at train/main.rs:1234-1270 and calls `model.merge_adapters()` which iterates all LoraLinear layers calling `merge()`. The fix is in the LoraLinear::merge() method to handle the QLoRA case.

**Step 1: Write failing test for QLoRA merge**

```rust
#[test]
fn test_qlora_linear_merge() {
    let device = Device::Cpu;
    let settings = LoraSettings { r: 4, alpha: 16.0, use_4bit: true, ..Default::default() };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut linear = LoraLinear::new(64, 32, settings, vb).unwrap();
    // Should not error on merge
    assert!(linear.merge().is_ok());
    // After merge, qlora should be None and base should be Some
    assert!(linear.qlora.is_none());
    assert!(linear.base.is_some());
}
```

**Step 2: Run test**

Run: `cargo test -p rust-functiongemma-core test_qlora_linear_merge`
Expected: FAIL with "QLoRA merge is not supported"

**Step 3: Implement QLoRA merge**

Replace the error path at model.rs:136-141:

```rust
pub fn merge(&mut self) -> Result<()> {
    if let Some(qlora) = self.qlora.take() {
        // Dequantize base weight to f32
        let base_f32 = qlora.dequantize()?;
        // Apply LoRA delta if adapters exist
        let merged = if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let delta = b.matmul(a)?.affine(self.scale as f64, 0.0)?;
            (base_f32 + delta)?
        } else {
            base_f32
        };
        self.base = Some(candle_nn::Linear::new(merged, None));
        self.lora_a = None;
        self.lora_b = None;
        return Ok(());
    }
    if self.qmatmul.is_some() {
        // Similar: dequantize QMatMul, add LoRA delta, store as Linear
        let qm = self.qmatmul.take().unwrap();
        let base_f32 = qm.dequantize()?; // QMatMul has dequantize method
        let merged = if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let delta = b.matmul(a)?.affine(self.scale as f64, 0.0)?;
            (base_f32 + delta)?
        } else {
            base_f32
        };
        self.base = Some(candle_nn::Linear::new(merged, None));
        self.lora_a = None;
        self.lora_b = None;
        return Ok(());
    }
    // Standard LoRA merge (existing code)
    if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
        let delta = b.matmul(a)?.affine(self.scale as f64, 0.0)?;
        if let Some(base) = &self.base {
            let w = base.weight();
            let merged = (w + &delta)?;
            self.base = Some(candle_nn::Linear::new(merged, None));
        }
        self.lora_a = None;
        self.lora_b = None;
    }
    Ok(())
}
```

**Note:** The exact `dequantize()` API depends on `qlora-rs` and candle's QMatMul. Check the actual API:
- `qlora-rs::QuantizedLinear` may expose `dequantize()` or require manual unpacking
- `candle_core::quantized::QMatMul` has `dequantize_f16()` method

**Step 4: Run test**

Run: `cargo test -p rust-functiongemma-core test_qlora_linear_merge`
Expected: PASS

**Step 5: Integration test with Merge CLI**

Run: `cargo run -p rust-functiongemma-train -- merge --model-path Models/functiongemma-270m-it --adapters checkpoints/checkpoint-46 --output /tmp/merged-test`
Expected: "Merged model saved successfully."

**Step 6: Commit**

```bash
git add Deploy/rust-functiongemma-core/src/model.rs
git commit -m "feat(core): implement QLoRA and QMatMul merge for adapter deployment"
```

---

## Workstream 3: Performance & Runtime (can start immediately)

### Task 7: Add request queuing with semaphore

**Files:**
- Modify: `Deploy/rust-functiongemma-runtime/src/lib.rs:1256-1261` (build_router)
- Modify: `Deploy/rust-functiongemma-runtime/src/lib.rs:1311-1380` (chat handler)
- Modify: `Deploy/rust-functiongemma-runtime/src/lib.rs:1410-1422` (serve)
- Modify: `Deploy/rust-functiongemma-runtime/Cargo.toml`
- Test: `Deploy/rust-functiongemma-runtime/tests/http_router.rs`

**Step 1: Write test for 429 response under load**

In `tests/http_router.rs`:
```rust
#[tokio::test]
async fn test_returns_429_when_queue_full() {
    // Start server with queue_depth=1
    // Send 3 concurrent requests
    // At least one should get 429 Too Many Requests
}
```

**Step 2: Add AppState with semaphore**

In `lib.rs`, add:
```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

struct AppState {
    semaphore: Arc<Semaphore>,
    request_timeout: std::time::Duration,
}
```

**Step 3: Update build_router to accept state**

```rust
pub fn build_router(queue_depth: usize, timeout_secs: u64) -> Router {
    let state = Arc::new(AppState {
        semaphore: Arc::new(Semaphore::new(queue_depth)),
        request_timeout: std::time::Duration::from_secs(timeout_secs),
    });
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .with_state(state)
}
```

**Step 4: Update chat handler with semaphore**

At `lib.rs:1311`:
```rust
async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    let permit = match tokio::time::timeout(
        state.request_timeout,
        state.semaphore.acquire()
    ).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => return Err(ErrorResponse::service_unavailable("Server shutting down")),
        Err(_) => return Err(ErrorResponse::too_many_requests("Queue full, try again later")),
    };
    // ... existing handler logic, permit auto-dropped
}
```

**Step 5: Add structured ErrorResponse type**

```rust
#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    r#type: String,
    message: String,
    code: String,
}

struct ErrorResponse(StatusCode, Json<ErrorBody>);

impl ErrorResponse {
    fn too_many_requests(msg: &str) -> Self { /* 429 */ }
    fn service_unavailable(msg: &str) -> Self { /* 503 */ }
    fn bad_request(msg: &str) -> Self { /* 400 */ }
    fn internal(msg: &str) -> Self { /* 500 */ }
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}
```

**Step 6: Run tests**

Run: `cargo test -p rust-functiongemma-runtime`
Expected: PASS

**Step 7: Commit**

```bash
git add Deploy/rust-functiongemma-runtime/
git commit -m "feat(runtime): add request queuing with semaphore and structured error responses"
```

---

### Task 8: Add SSE streaming support

**Files:**
- Modify: `Deploy/rust-functiongemma-runtime/src/lib.rs`
- Modify: `Deploy/rust-functiongemma-runtime/Cargo.toml`
- Test: `Deploy/rust-functiongemma-runtime/tests/http_router.rs`

**Step 1: Add dependencies**

In Cargo.toml:
```toml
tokio-stream = "0.1"
async-stream = "0.3"
futures = "0.3"
```

**Step 2: Write test for streaming response**

```rust
#[tokio::test]
async fn test_streaming_chat_completion() {
    // Send request with stream: true
    // Verify response is SSE (text/event-stream content type)
    // Verify [DONE] is last event
}
```

**Step 3: Add stream field to ChatCompletionRequest**

```rust
#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    // ... existing fields
    #[serde(default)]
    pub stream: bool,
}
```

**Step 4: Implement streaming handler**

```rust
async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.stream {
        return chat_stream(state, req).await.into_response();
    }
    chat_normal(state, req).await.into_response()
}

async fn chat_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> axum::response::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        // Generate tokens one at a time
        // For non-model mode (heuristic), yield single result
        let result = process_request(&req);
        yield Ok(axum::response::sse::Event::default()
            .data(serde_json::to_string(&ChatChunk::from(&result)).unwrap()));
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    axum::response::Sse::new(stream)
}
```

**Step 5: Run tests**

Run: `cargo test -p rust-functiongemma-runtime`
Expected: PASS

**Step 6: Commit**

```bash
git add Deploy/rust-functiongemma-runtime/
git commit -m "feat(runtime): add SSE streaming for chat completions"
```

---

### Task 9: Add async FFI exports

**Files:**
- Modify: `Native/pcai_core/pcai_inference/src/ffi/mod.rs:39-481`
- Modify: `Native/PcaiNative/InferenceModule.cs`
- Test: `Native/pcai_core/pcai_inference/tests/integration_test.rs`

**Step 1: Add request tracking to global state**

At `ffi/mod.rs:39`:
```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};

static NEXT_REQUEST_ID: AtomicI64 = AtomicI64::new(1);

enum RequestStatus {
    Pending,
    Running,
    Complete(String),
    Failed(String),
    Cancelled,
}

struct GlobalState {
    runtime: Runtime,
    backend: Option<Box<dyn InferenceBackend>>,
    requests: HashMap<i64, RequestStatus>,  // NEW
}
```

**Step 2: Add pcai_generate_async export**

```rust
#[no_mangle]
pub extern "C" fn pcai_generate_async(
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
) -> i64 {
    clear_last_error();
    let prompt_str = match unsafe_cstr_to_string(prompt) {
        Ok(s) => s,
        Err(_) => { set_last_error("Invalid prompt string"); return -1; }
    };
    let id = NEXT_REQUEST_ID.fetch_add(1, Ordering::SeqCst);
    let state = get_global_state();
    let mut guard = state.lock().unwrap();
    guard.requests.insert(id, RequestStatus::Pending);

    // Spawn async task
    let runtime_handle = guard.runtime.handle().clone();
    runtime_handle.spawn(async move {
        // ... generate and update status to Complete or Failed
    });

    id
}
```

**Step 3: Add pcai_poll_result export**

```rust
#[repr(C)]
pub struct PcaiAsyncResult {
    pub status: i32,  // 0=pending, 1=running, 2=complete, 3=failed, 4=cancelled
    pub text: *mut c_char,
}

#[no_mangle]
pub extern "C" fn pcai_poll_result(request_id: i64) -> PcaiAsyncResult {
    // Check request status, return result if complete
}

#[no_mangle]
pub extern "C" fn pcai_cancel(request_id: i64) -> i32 {
    // Set cancellation flag
}
```

**Step 4: Add pcai_get_last_error export**

```rust
#[no_mangle]
pub extern "C" fn pcai_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(msg) => msg.as_ptr() as *const c_char,
            None => std::ptr::null(),
        }
    })
}
```

**Step 5: Update C# wrapper**

In `InferenceModule.cs`, add P/Invoke signatures and async wrapper method.

**Step 6: Run tests**

Run: `cargo test -p pcai-inference -- --test-threads=1`
Expected: PASS

**Step 7: Commit**

```bash
git add Native/pcai_core/pcai_inference/src/ffi/mod.rs Native/PcaiNative/InferenceModule.cs
git commit -m "feat(ffi): add async generation API with polling and cancellation"
```

---

### Task 10: Add model path resolution with search order

**Files:**
- Modify: `Deploy/rust-functiongemma-runtime/src/lib.rs:1131-1254` (infer_with_model)
- Test: `Deploy/rust-functiongemma-runtime/tests/http_router.rs`

**Step 1: Write test for path resolution**

```rust
#[test]
fn test_model_path_resolution_absolute() {
    let path = resolve_model_path("/absolute/path/to/model");
    assert_eq!(path, PathBuf::from("/absolute/path/to/model"));
}

#[test]
fn test_model_path_resolution_env_var() {
    std::env::set_var("PCAI_MODELS_DIR", "/tmp/models");
    let path = resolve_model_path("functiongemma-270m-it");
    assert_eq!(path, PathBuf::from("/tmp/models/functiongemma-270m-it"));
    std::env::remove_var("PCAI_MODELS_DIR");
}
```

**Step 2: Implement resolve_model_path**

```rust
fn resolve_model_path(model_id: &str) -> PathBuf {
    let p = PathBuf::from(model_id);
    // 1. Absolute path
    if p.is_absolute() && p.exists() {
        tracing::info!("Model path (absolute): {}", p.display());
        return p;
    }
    // 2. Env variable
    if let Ok(dir) = std::env::var("PCAI_MODELS_DIR") {
        let candidate = PathBuf::from(dir).join(model_id);
        if candidate.exists() {
            tracing::info!("Model path ($PCAI_MODELS_DIR): {}", candidate.display());
            return candidate;
        }
    }
    // 3. Relative to CWD/Models/
    let models_dir = PathBuf::from("Models").join(model_id);
    if models_dir.exists() {
        tracing::info!("Model path (Models/): {}", models_dir.display());
        return models_dir;
    }
    // 4. Fallback to input as-is
    tracing::warn!("Model path not found, using as-is: {}", model_id);
    p
}
```

**Step 3: Run tests**

Run: `cargo test -p rust-functiongemma-runtime`
Expected: PASS

**Step 4: Commit**

```bash
git add Deploy/rust-functiongemma-runtime/src/lib.rs
git commit -m "feat(runtime): add model path resolution with env var and search order"
```

---

## Plugin Installation (run before starting implementation)

### Task 0: Install recommended plugins

**Step 1: Install wshobson/agents plugins**

```bash
claude plugin install systems-programming@upstream-agents
claude plugin install machine-learning-ops@upstream-agents
claude plugin install performance-testing-review@upstream-agents
```

**Step 2: Verify installation**

```bash
claude plugin list
```

Expected: All three new plugins appear in the list alongside existing 4 upstream plugins.

---

## Execution Order

```
Task 0:  Install plugins (prerequisite)
         │
         ├── Task 1:  Delete train/model.rs, import from core (GATE)
         │   ├── Task 6:  QLoRA merge (needs consolidated model)
         │   └── Task 3:  Token cache validation
         │
         ├── Task 4:  Expand dataset (independent)
         ├── Task 5:  Evaluation metrics (independent)
         │
         ├── Task 7:  Request queuing (independent)
         ├── Task 8:  SSE streaming (independent)
         ├── Task 9:  Async FFI (independent)
         └── Task 10: Model path resolution (independent)
```

**Parallel groups:**
- Group A (consolidation): Tasks 1 → 2 → 3 → 6
- Group B (training): Tasks 4, 5 (start immediately)
- Group C (performance): Tasks 7, 8, 9, 10 (start immediately)

All groups can run concurrently. Within Group A, tasks are sequential.
