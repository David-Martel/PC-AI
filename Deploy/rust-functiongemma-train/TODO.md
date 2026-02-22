# TODO - Rust FunctionGemma (PC_AI)

This TODO captures the minimum work required to reach feature parity with
Deploy/functiongemma-finetune (Python) and to enable a Rust-only router runtime.

## Build environment
- [x] Use CargoTools wrapper for all builds/tests (Tools/Invoke-RustBuild.ps1).
- [x] Keep lld-link optional; default to link.exe unless explicitly enabled.
- [x] Ensure LLVM lld-link path is configured (C:\Program Files\LLVM\bin\lld-link.exe).

## P0 - I/O parity (required for drop-in replacement)
- [x] Implement OpenAI-compatible Chat Completions server (POST /v1/chat/completions).
- [x] Accept tools schema (Config/pcai-tools.json) and return message.tool_calls.
- [x] Support router prompt format: [MODE], [SYSTEM_PROMPT], [USER_REQUEST].
- [x] Emit NO_TOOL when no tool is needed (chat mode or non-tool cases).
- [x] Provide a tool-call parser that matches FunctionGemma expectations.

## P0 - Dataset + prompt parity
- [x] Router dataset generator (prepare_dataset.py parity) implemented via `prepare-router`:
  - Modes: diagnose/chat
  - System prompts: DIAGNOSE.md + CHAT.md
  - Scenario file: Deploy/rust-functiongemma-train/examples/scenarios.json
  - Tool coverage from pcai-tools.json
- [x] Ensure chat template rendering uses the tokenizer template with tools.
- [x] Add prompt masking so user/developer content does not contribute to loss.
- [x] Emit tool test vectors alongside tool-coverage datasets (parity with generate_training_data.py). Implemented via `prepare-router --test-vectors`.

## P0 - Training parity
- [ ] LoRA/QLoRA support with target modules (q/k/v/o/gate/up/down). (LoRA targets updated; QLoRA quantization pending)
  - [ ] Evaluate qlora-rs (NF4 + double quantization) as a Rust-first QLoRA path.
- [x] Warmup + LR scheduling (linear or cosine).
- [x] Resume from checkpoint.
- [x] Eval split and optional early stopping. (early stopping wired; eval split implemented)
- [x] Save PEFT-style adapter outputs + tokenizer metadata.

## P0 - Runtime inference parity
- [x] Load base model + LoRA adapters, or merged model.
- [ ] Match FunctionGemma chat template behavior.
- [ ] Provide deterministic generation settings for routing (low temp, short max tokens).
- [ ] Expose model + tools + version in /v1/models or /health endpoints.

## P1 - Tests and regressions
- [ ] Port Python unit tests for dataset and schema handling.
- [ ] Add router eval harness against a local runtime.
- [x] Validate tool call accuracy on scenarios.json and test vectors.

## P1 - PC_AI integration
- [x] PowerShell wrapper to replace Python tool_router.py.
- [x] Update Tools/run-functiongemma-tests.ps1 to prefer Rust pipeline.
- [x] Add config in Config/llm-config.json to point router base URL to Rust runtime.

## P2 - Performance + UX
- [x] Incremental dataset generation and streaming JSONL output.
- [x] Memory/throughput metrics in runtime server.
- [x] Optional GPU selection and memory limits in config.
- [x] Pre-tokenize datasets and cache token IDs on disk (memmap2) for faster training/eval.
- [x] Add prompt packing (multiple short samples per batch) to improve GPU utilization.
- [x] Add deterministic eval metrics (tool-name accuracy + argument exact match) with JSON output.
- [x] Add JSON schema validation for tool call outputs (reject invalid arguments early).
- [ ] Evaluate candle-cuda-vmm or CUDA memory pool options for more stable VRAM allocation.
- [ ] oLLM parity: add optional KV cache offload to CPU/disk (with streaming readback).
- [ ] oLLM parity: implement chunked/online softmax attention + chunked MLP for large-context prefill.
- [ ] oLLM parity: safe-tensors reader to avoid mmap-induced RAM spikes (optional path).
- [ ] oLLM parity: evaluate cuFILE/GPUDirect Storage via cudarc for direct GPU <-> disk transfer.
- [ ] Prototype RAG memory hooks (rag-redis at W:\dropbox-local\rag-redis; Redis 6379/6380) for router eval inputs.
- [ ] Scope reduction: consider narrowing FunctionGemma to an intermediate tool-execution layer between pcai_inference and PC-AI modules (drop chat/diagnose router role) to simplify training targets and reduce memory/compute. (See README.md, ARCHITECTURE.md)

## Crate candidates (easy wins)
- hf-hub: download and cache gated models.
- tokenizers: fast HF tokenizer and chat template support.
- safetensors: safe model weights IO.
- minijinja: render chat templates (Jinja-compatible).
- axum: lightweight HTTP server for OpenAI-compatible endpoints.
- tracing + tracing-subscriber: structured logs + log filtering.
- tower-http: runtime middleware (trace, timeouts, compression).
- schemars: generate JSON schema from Rust tool definitions.
- jsonschema: validate tool schemas and dataset payloads.

## Structure proposal
- [x] Convert to a Rust workspace:
  - [x] rust-functiongemma-runtime (server) - Deploy/rust-functiongemma-runtime
  - [x] rust-functiongemma-train (dataset + training) - Deploy/rust-functiongemma-train
  - [x] rust-functiongemma-core (shared model/prompt/util)
