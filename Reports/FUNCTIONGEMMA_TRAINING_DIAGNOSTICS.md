# FunctionGemma Training Pipeline - Diagnostic Report

**Agent:** fg-train-debug
**Date:** 2026-03-30
**Scope:** `Deploy/rust-functiongemma-train/` (12 source files) + `Deploy/rust-functiongemma-core/src/model.rs`

---

## Executive Summary

The FunctionGemma training pipeline has **1 build-breaking issue** that prevents compilation entirely, **2 critical runtime bugs** that produce incorrect training results, and **8 additional issues** ranging from high to low severity. The most impactful finding is a missing causal attention mask in the non-flash-attention training path, which means the model can attend to future tokens during training -- producing a model that appears to train but generates garbage at inference time.

---

## Issue 1: BUILD BLOCKER -- cudarc Version Conflict Prevents Compilation

**Severity:** CRITICAL (build failure)
**Type:** Compile-time
**File:** `Deploy/rust-functiongemma-train/Cargo.toml` / `Cargo.lock`

### Problem

The training crate cannot compile at all. `rust-functiongemma-core` requires `cudarc = "0.19.4"`, but the training crate's stale `Cargo.lock` pins `cudarc` to `0.19.0`. Since `candle-core 0.9.2` (a shared dependency) requires `cudarc ^0.19.0` and the lock file resolves it to `0.19.0`, cargo cannot satisfy core's `0.19.4` requirement simultaneously.

```
error: failed to select a version for `cudarc`.
    ... required by package `candle-core v0.9.2`
    previously selected package `cudarc v0.19.4`
    ... which satisfies dependency `cudarc = "^0.19.4"` of package `rust-functiongemma-core`
```

### Root Cause

The workspace configuration at `Deploy/rust-functiongemma/Cargo.toml` uses `../` relative paths for members, which is unsupported by Cargo:

```
error: workspace member `...\rust-functiongemma-runtime\Cargo.toml` is not hierarchically below
the workspace root `...\rust-functiongemma\Cargo.toml`
```

This means each crate resolves dependencies independently with its own `Cargo.lock`, and the lock files have drifted. The core crate upgraded `cudarc` to 0.19.4 but the train crate's lock was never regenerated.

### Fix

1. Delete `Deploy/rust-functiongemma-train/Cargo.lock` and regenerate it.
2. Fix the workspace layout: either move crates under the workspace root, or abandon the workspace and ensure each crate has a compatible lock file.
3. The train crate also needs the `[patch.crates-io]` section that core has for vendored `candle-kernels` and `candle-flash-attn`, since patches do NOT propagate through path dependencies when there is no shared workspace.

---

## Issue 2: CRITICAL -- Missing Causal Attention Mask in Training Forward Pass

**Severity:** CRITICAL (silent training corruption)
**Type:** Runtime
**File:** `Deploy/rust-functiongemma-core/src/model.rs:583-585`

### Problem

The `Attention::forward()` method (used during training via `Model::forward()`) has two code paths:

1. **Flash attention (CUDA):** Correctly passes `causal: true` to `flash_attn()` (line 580).
2. **Standard attention (CPU or non-flash):** Computes `softmax(Q * K^T * scale)` with NO causal mask (lines 583-585).

```rust
// Line 583-585 -- NO CAUSAL MASK!
let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
attn_weights.matmul(&v)?
```

For causal language modeling, every token must only attend to itself and previous tokens. Without a mask, each position attends to all positions including future ones. This means:

- **With flash_attn=true on CUDA:** Training works correctly.
- **With flash_attn=false OR on CPU:** Training produces a model that has seen future tokens during training, leading to degenerate outputs at inference time where the model has never learned actual next-token prediction.

The config file `pcai-functiongemma.json` sets `flash_attn: true`, so the default path is currently correct. However, any CPU fallback, debugging run, or CI test without flash attention will produce silently corrupt models.

### Fix

Add a lower-triangular causal mask before the softmax in the non-flash path:

```rust
let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
// Apply causal mask: set future positions to -inf
let causal_mask = Tensor::tril2(seq_len, x.dtype(), x.device())?;
let neg_inf = f32::NEG_INFINITY;
let attn_weights = attn_weights.broadcast_add(
    &causal_mask.unsqueeze(0)?.unsqueeze(0)?
        .where_cond(&Tensor::zeros_like(&attn_weights)?,
                     &Tensor::full(neg_inf, attn_weights.shape(), x.device())?)?
)?;
let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
```

---

## Issue 3: CRITICAL -- Checkpoint Resume Not Wired Into CLI

**Severity:** HIGH (data loss / wasted compute)
**Type:** Runtime (incomplete feature)
**File:** `Deploy/rust-functiongemma-train/src/trainer.rs:934-968` and `src/main.rs`

### Problem

`Trainer::resume_from_checkpoint()` is fully implemented in `trainer.rs` (lines 934-968) and correctly loads checkpoint metadata + adapter weights. However, there is no `--resume` CLI flag in `main.rs`, and `resume_from_checkpoint()` is never called anywhere.

Additionally, the checkpoint metadata saves `optimizer_state: vec![]` (line 923) and `rng_state: None` (line 924) with TODO comments, meaning even if resume were wired up:
- The optimizer state (Adam momentum/variance) would be lost, causing a training quality regression on resume.
- The RNG state would be lost, making training non-reproducible after resume.

### Fix

1. Add `--resume <checkpoint_path>` to the `Train` CLI subcommand.
2. Call `trainer.resume_from_checkpoint(path)` before `trainer.train()`.
3. Implement optimizer state serialization (serialize the `AdamW` internal state).
4. Save `rng_state` from the `StdRng` seed.

---

## Issue 4: HIGH -- No NaN/Inf Guard on Loss Values

**Severity:** HIGH
**Type:** Runtime
**File:** `Deploy/rust-functiongemma-train/src/trainer.rs` (compute_loss, train loop)

### Problem

The training loop never checks if `loss_val` is NaN or Inf. If the loss diverges (common with high learning rates, dtype underflow, or degenerate batches), the training continues silently, accumulating NaN gradients and corrupting all model weights. The loss values flow into:

- `epoch_loss` accumulator (becomes NaN, making `avg_epoch_loss` NaN)
- `best_loss` comparison (NaN < f64::MAX is false, so best_loss is never updated)
- Early stopping (NaN comparisons always return false, so early stopping never triggers)
- Checkpoint saving (saves a corrupted model)
- JSON progress output (prints "NaN" which breaks downstream JSON parsers)

### Fix

Add a NaN/Inf check after `compute_loss` returns:

```rust
if loss_val.is_nan() || loss_val.is_infinite() {
    return Err(anyhow::anyhow!(
        "Training diverged: loss is {} at step {}. Check learning rate and data.",
        loss_val, self.global_step
    ));
}
```

---

## Issue 5: HIGH -- Gradient Accumulation Drops Gradients from Last Micro-batch

**Severity:** HIGH
**Type:** Runtime (edge case)
**File:** `Deploy/rust-functiongemma-train/src/trainer.rs:378-393`

### Problem

The gradient accumulation logic initializes `grad_accum_store` as `None` and on the first micro-batch assigns the backward grads directly:

```rust
// Line 391-392
} else {
    grad_accum_store = Some(grads);
}
```

However, the accumulation loop on lines 378-390 iterates `trainable_vars` and merges gradients for each var. The issue is subtle: if the first micro-batch's `grads` GradStore contains gradients for variables NOT in `trainable_vars` (e.g., non-LoRA base model vars that still participate in the compute graph), those extra gradients are carried into the accumulator. Then on the optimizer step, the optimizer only operates on `trainable_vars`, so this is not a correctness bug per se, but it means the GradStore accumulates garbage entries that consume memory proportional to the number of non-trainable parameters times the number of accumulation steps.

The actual bug is: if `grad_accum` is set larger than the number of batches in an epoch, the `is_accum_end` check at line 405 fires on `i + 1 == num_batches`, which is correct. But if the dataset has exactly 0 valid batches (empty dataset after filtering), `grad_accum_store` is never populated and the `take()` at line 409 returns `None`, so the optimizer step is silently skipped. The `batch_count == 0` check at line 477 catches this, so it is guarded.

**Revised severity:** MEDIUM -- no data corruption, but unnecessary memory pressure from accumulated non-trainable gradients.

---

## Issue 6: HIGH -- LoRA Dropout Declared But Never Applied

**Severity:** MEDIUM
**Type:** Runtime (incomplete feature)
**File:** `Deploy/rust-functiongemma-core/src/model.rs:379` and `Deploy/rust-functiongemma-train/src/lora.rs:13`

### Problem

`LoraSettings.dropout` is carried through configuration and stored, but never applied in either:
- `LoraLinear::forward()` in `model.rs` (the production LoRA implementation used during training)
- `LoraLinear::forward()` in `lora.rs` (the standalone LoRA implementation, which has a comment "Dropout probability (not yet implemented)")

If a user sets `lora_dropout: 0.1` expecting regularization, they get no regularization at all. The QLoRA path passes `dropout` to `qlora-rs`, where it may or may not be applied depending on that library's implementation.

### Fix

Apply dropout after the LoRA matmul in `model.rs` `LoraLinear::forward()`:

```rust
let lora_out = if self.dropout > 0.0 && /* training mode flag */ {
    candle_nn::ops::dropout(&lora_out, self.dropout)?
} else {
    lora_out
};
```

Note: candle does not have a native training mode flag, so this needs a `training: bool` field or method parameter.

---

## Issue 7: MEDIUM -- Workspace Layout Broken (Non-hierarchical Members)

**Severity:** MEDIUM
**Type:** Build infrastructure
**File:** `Deploy/rust-functiongemma/Cargo.toml`

### Problem

The workspace `Cargo.toml` references members outside its directory:

```toml
members = [
    "../rust-functiongemma-runtime",
    "../rust-functiongemma-train",
    "../rust-functiongemma-core",
]
```

Cargo requires workspace members to be hierarchically below the workspace root. This causes:

```
error: workspace member ... is not hierarchically below the workspace root
```

This means the workspace cannot be used at all, and each crate must be built independently with its own lock file. The `[workspace.dependencies]` section (candle-core, candle-nn, etc.) is completely unused.

### Fix

Restructure the directory layout to place all crates under a single workspace root, or create the workspace Cargo.toml one level up at `Deploy/Cargo.toml`.

---

## Issue 8: MEDIUM -- `std::env::set_var` is Unsound in Multi-threaded Context

**Severity:** MEDIUM
**Type:** Runtime (safety)
**File:** `Deploy/rust-functiongemma-train/src/main.rs:523`

### Problem

```rust
std::env::set_var("CUDA_LAUNCH_BLOCKING", "1");
```

`set_var` is unsound in multi-threaded programs and deprecated as of Rust 2024 edition. While the training crate uses edition 2021, the call happens after `main()` starts where other threads (from tokio, rayon, or CUDA runtime) may already be reading environment variables.

### Fix

Set `CUDA_LAUNCH_BLOCKING=1` before process start (e.g., in the shell script or launcher), or use `unsafe { std::env::set_var(...) }` with explicit documentation about thread safety.

---

## Issue 9: MEDIUM -- Eval Path Mutates-Then-Pops Last Message (Logic Bug)

**Severity:** MEDIUM
**Type:** Runtime
**File:** `Deploy/rust-functiongemma-train/src/main.rs:806-809`

### Problem

```rust
let mut eval_item = item.clone();
if let Some(last) = eval_item.messages.last_mut() {
    if last.role == "assistant" || last.role == "model" {
        eval_item.messages.pop();
    }
}
```

This code gets a mutable reference to the last message, checks its role, then drops the reference and pops the last element. The `last_mut()` is wasteful (could use `last()`) but more importantly, if the last message's role is NOT "assistant"/"model", the message is kept, meaning the model is evaluated with the expected output still in the prompt. For well-formed training data this should never happen, but it is fragile.

### Fix

Use `last()` instead of `last_mut()` since no mutation is performed:

```rust
if eval_item.messages.last().map_or(false, |m| m.role == "assistant" || m.role == "model") {
    eval_item.messages.pop();
}
```

---

## Issue 10: MEDIUM -- Token Cache `[patch.crates-io]` Missing from Train Crate

**Severity:** MEDIUM
**Type:** Build
**File:** `Deploy/rust-functiongemma-train/Cargo.toml`

### Problem

The core crate has:

```toml
[patch.crates-io]
candle-kernels = { path = "../vendor/candle-kernels-0.9.2" }
candle-flash-attn = { path = "../vendor/candle-flash-attn-0.9.2" }
```

These patches fix CUDA 13+ CCCL preprocessor compatibility on MSVC. Since the workspace is broken (Issue 7), the train crate does NOT inherit these patches. When compiled independently, it will use the unpatched `candle-kernels`, causing CUDA build failures on this machine (CUDA 13.2 + MSVC).

### Fix

Add the same `[patch.crates-io]` section to `Deploy/rust-functiongemma-train/Cargo.toml`.

---

## Issue 11: LOW -- Standalone `lora.rs` Module is Dead Code

**Severity:** LOW
**Type:** Code quality
**File:** `Deploy/rust-functiongemma-train/src/lora.rs`

### Problem

The training crate has its own `lora.rs` module defining `LoraLinear` and `LoraConfig`. However, the actual training loop uses `rust_functiongemma_core::model::LoraLinear` and `LoraSettings` (re-exported via `lib.rs`). The standalone `lora.rs` is never imported or used in the training pipeline -- it appears to be an early prototype that was superseded by the core implementation.

The standalone version also has different behavior:
- Uses `DType::F32` hardcoded (core version respects VarBuilder dtype)
- Uses `Kaiming uniform` initialization (core version uses VarBuilder's get())
- Does not support QLoRA or QMatMul (core version does)

### Fix

Remove `lora.rs` from the training crate or mark it as a reference implementation with `#[cfg(test)]`.

---

## Issue 12: LOW -- `loss_convergence_rate` Uses Naive First-Last Slope

**Severity:** LOW
**Type:** Metrics quality
**File:** `Deploy/rust-functiongemma-train/src/trainer.rs:543-554`

### Problem

The convergence rate is calculated as:

```rust
((last_loss - first_loss) / step_span) * 100.0
```

This is a simple two-point slope, not a proper linear regression. If the loss has a noisy spike at the start or end, the reported convergence rate will be wildly inaccurate. The comment says "via linear regression slope" but the implementation is not linear regression.

### Fix

Implement proper least-squares linear regression over all step losses, or use a windowed average (first N steps vs last N steps) for robustness.

---

## Prioritized Fix Order

| Priority | Issue | Type | Effort | Impact |
|----------|-------|------|--------|--------|
| **P0** | #1 cudarc version conflict | Build | 30 min | Unblocks all other work |
| **P0** | #7 Workspace layout | Build | 1 hr | Prevents future drift |
| **P0** | #10 Missing patches | Build | 10 min | CUDA build on MSVC |
| **P1** | #2 Missing causal mask | Training correctness | 2 hr | Prevents silent corruption |
| **P1** | #4 NaN/Inf guard | Training stability | 30 min | Prevents silent divergence |
| **P2** | #3 Checkpoint resume | Feature completeness | 4 hr | Enables long training |
| **P2** | #6 LoRA dropout | Training quality | 1 hr | Enables regularization |
| **P2** | #8 set_var safety | Safety | 15 min | Thread safety |
| **P3** | #5 GradStore memory | Performance | 1 hr | Reduces VRAM pressure |
| **P3** | #9 Eval path logic | Code quality | 15 min | Cleaner eval |
| **P3** | #11 Dead lora.rs | Code quality | 10 min | Less confusion |
| **P3** | #12 Convergence metric | Metrics | 30 min | More accurate reporting |

---

## Summary of Key Files

| File | Lines | Role | Issues Found |
|------|-------|------|--------------|
| `trainer.rs` | 969 | Training loop, loss, optimizer | #3, #4, #5, #12 |
| `main.rs` | 1010 | CLI entry, model loading, eval | #8, #9 |
| `model.rs` (core) | ~1520 | Model forward pass, LoRA | #2, #6 |
| `Cargo.toml` (train) | 22 | Dependencies | #1, #10 |
| `Cargo.toml` (workspace) | 32 | Workspace config | #7 |
| `lora.rs` | 207 | Unused LoRA prototype | #11 |
| `dataset.rs` | 677 | Token cache, batching | (clean) |
| `checkpoint.rs` | 205 | Save/load checkpoints | #3 (incomplete) |
| `eval.rs` | 463 | Evaluation metrics | (clean) |
| `data_gen.rs` | 482 | Training data generation | (clean) |
| `router_dataset.rs` | 349 | Router dataset builder | (clean) |
| `schema_utils.rs` | 168 | Arg set generation | (clean) |
| `early_stopping.rs` | 124 | Early stopping monitor | (clean) |
| `scheduler.rs` | 179 | LR scheduler | (clean) |
