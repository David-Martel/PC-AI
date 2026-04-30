# LLM, Preflight, And FunctionGemma TODO

This ledger tracks active LLM/evaluation work. The original `pcai_preflight`
implementation is complete; keep this file focused on remaining model,
runtime, evaluation, and large-context gaps.

Last reconciled: 2026-04-30.

## Completed Baseline To Preserve

- GGUF header parsing, memory estimation, VRAM audit, readiness verdict logic,
  FFI export, CLI subcommand, PowerShell wrapper, Rust tests, and Pester tests
  for GPU preflight are complete.
- FunctionGemma now uses deterministic generation defaults and has NVML-backed
  preflight checks before model load.
- CUDA 13.2 build validation, SM 89/120 support, cudarc 0.19.4, and patched
  candle kernel build flow were validated in the prior LLM optimization pass.
- Performance analytics now include roofline modeling, bandwidth-efficiency
  metrics, regression detection, evaluation harness wiring, and training
  metrics output.
- All-backend operational hardening is complete for current scope:
  preflight checks in `PcaiInference.psm1` and `PcaiMedia.psm1`, enriched OOM
  errors, Ollama config optimization, benchmark config determinism, and
  `Invoke-LlmPerfBenchmark.ps1`.

## Active Work

### FunctionGemma Runtime And Training Parity

- [ ] Match FunctionGemma runtime chat-template behavior to the training and
  evaluation assumptions.
- [ ] Port Python dataset/schema unit tests into Rust so training data and tool
  schema handling are covered without relying on the Python prototype path.
- [ ] Keep deterministic routing settings covered by regression tests when
  routing prompts, tool schema, or model defaults change.

### Large-Context And Memory Behavior

- [ ] Implement or prototype KV-cache offload to CPU/disk for
  `pcai_inference` and FunctionGemma runtime scenarios.
- [ ] Add chunked softmax attention for large-context prefill and benchmark the
  memory/latency tradeoff against the current implementation.
- [ ] Evaluate CUDA memory pool behavior, including `candle-cuda-vmm` or a
  successor, before adopting it as a default.
- [ ] Evaluate GPUDirect Storage through `cudarc` only after a reproducible
  model-load or prefill benchmark shows storage transfer is the bottleneck.

### Evaluation And Regression Discipline

- [ ] Add a small compatibility suite that compares FunctionGemma routed output
  against the expected diagnose/chat tool-selection envelopes.
- [ ] Ensure prompt, router, and model-default changes update baseline snapshots
  or explicitly document why no baseline refresh is required.
- [ ] Add failure-mode fixtures for OOM, no GPU, insufficient VRAM, missing
  model, malformed GGUF header, and provider health-gate failures.

### Runtime Operations

- [ ] Surface LLM preflight verdicts in the same structured status view used by
  `Get-PcaiServiceHealth` and `Get-PcaiNativeStatus`.
- [ ] Add cancellation and progress reporting for long model-load, prefill, and
  evaluation runs.
- [ ] Align LLM runtime logs with the repo-wide structured logging goal:
  stable event names, JSON-serializable evidence, and clear nonzero exit codes.

## Validation Anchors

- `pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 -Backend llamacpp-bin -Dataset diagnostic`
- `pwsh .\Tests\Evaluation\Invoke-FunctionGemmaEval.ps1`
- `cargo test --manifest-path Native\pcai_core\Cargo.toml`
- `dotnet build .\Native\PcaiNative\PcaiNative.csproj --no-restore`
- `pwsh .\Tools\Invoke-LlmPerfBenchmark.ps1 -Suite quick`
