# Rust FunctionGemma (PC_AI)

This folder is the Rust-first replacement for the legacy Python FunctionGemma pipeline.
The goal is drop-in I/O compatibility with the existing PC_AI router calls while
removing Python and Docker from the critical path.

## Scope and goals
- Provide a Rust training pipeline that can replace the Python/Unsloth scripts.
- Provide a Rust runtime service that speaks OpenAI-compatible Chat Completions.
- Preserve the router I/O contract used by PC_AI (tools in, tool_calls out).
- Keep Docker optional and disabled by default.

## Runtime + training split
Runtime and training are separate Rust crates:

1) Runtime
   - crate: rust-functiongemma-runtime
   - path: Deploy/rust-functiongemma-runtime
   - OpenAI-compatible HTTP server (Chat Completions)
   - FunctionGemma model inference with tool-call parsing (TODO)

2) Training
   - crate: rust-functiongemma-train
   - path: Deploy/rust-functiongemma-train
   - Dataset generation (tools + scenarios + prompts)
   - LoRA/QLoRA fine-tuning (TODO)
   - Eval harness + regression checks (TODO)

## I/O contract (PC_AI router compatibility)
PC_AI calls the router using Invoke-FunctionGemmaReAct and expects:
- OpenAI Chat Completions payloads (messages + tools + tool_choice)
- Tool calls in the assistant response under message.tool_calls
- The tools schema sourced from Config/pcai-tools.json

The router prompt format in PC_AI is:

[MODE]
<chat|diagnose>

[SYSTEM_PROMPT]
<content of CHAT.md or DIAGNOSE.md>

[USER_REQUEST]
<user request>

Your Rust runtime must accept that input (or equivalent) and return tool_calls
(or NO_TOOL) in a way that matches the OpenAI-compatible schema.

## Build environment (Primary Build.ps1)
Use the repository build orchestrator for normal build/test flows:

- Build FunctionGemma crates:
  .\Build.ps1 -Component functiongemma

- Build + test FunctionGemma crates:
  .\Build.ps1 -Component functiongemma -RunTests

- Generate router dataset + tool-call vectors:
  .\Build.ps1 -Component functiongemma-router-data

- Generate token cache:
  .\Build.ps1 -Component functiongemma-token-cache

- Run training:
  .\Build.ps1 -Component functiongemma-train

- Run evaluation harness:
  .\Build.ps1 -Component functiongemma-eval

For advanced crate-level workflows, wrappers are still available:

- Build/test via wrapper:
  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-train test

- Enable lld-link explicitly:
  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-train -UseLld build --release

Notes:
- lld-link is optional and disabled by default.
- The wrapper configures LLVM at `C:\Program Files\LLVM\bin\lld-link.exe`.
- Use `-NoLld` to force link.exe if lld-link fails on Windows.
- Tokenizers is configured with `fancy-regex` to avoid onig_sys when lld-link is enabled.

## Advanced train crate CLI (direct cargo)
For low-level debugging from `Deploy/rust-functiongemma-train`:

- Prepare dataset:
  cargo run --release -- prepare \
    --tools C:\Users\david\PC_AI\Config\pcai-tools.json \
    --output rust_train_data.jsonl \
    --scenarios C:\Users\david\PC_AI\Deploy\rust-functiongemma-train\examples\scenarios.json

- Prepare router dataset (tool_calls or NO_TOOL):
  cargo run --release -- prepare-router \
    --tools C:\Users\david\PC_AI\Config\pcai-tools.json \
    --output rust_router_data.jsonl \
    --diagnose-prompt C:\Users\david\PC_AI\DIAGNOSE.md \
    --chat-prompt C:\Users\david\PC_AI\CHAT.md \
    --scenarios C:\Users\david\PC_AI\Deploy\rust-functiongemma-train\examples\scenarios.json \
    --test-vectors C:\Users\david\PC_AI\Deploy\rust-functiongemma-train\data\test_vectors.json

- Prepare token cache (faster training):
  cargo run --release -- prepare-cache \
    --input rust_router_data.jsonl \
    --tokenizer C:\Users\david\PC_AI\Models\functiongemma-270m-it\tokenizer.json \
    --output-dir C:\Users\david\PC_AI\output\functiongemma-token-cache

## Wrapper scripts (Build.ps1-backed)
Use these for repeatable runs with Build.ps1 orchestration:

  .\Tools\prepare-functiongemma-router-data.ps1

## Token cache (PowerShell)
Build a token cache for faster training:

  .\Tools\prepare-functiongemma-token-cache.ps1

## Eval report (PowerShell)
Generate a metrics report from the eval harness:

  .\Tools\run-functiongemma-eval.ps1 -FastEval

## Training (PowerShell)
Run LoRA training via Build.ps1-backed wrapper:

  .\Tools\Invoke-FunctionGemmaTrain.ps1 -Epochs 1 -LoraR 16 -LoraAlpha 32

## C# / Rust DLL (CSharp_RustDLL format)

### RouterDatasetGeneration
* **Current State:** PowerShell can shell out to the Rust CLI; the native DLL now exposes `pcai_build_router_dataset_jsonl` for C#/PowerShell.
* **Rust Advantage:** Single-pass JSON parsing, deterministic output, and reduced PowerShell string handling.
* **Proposed Architecture:**
  * **Rust Signature:** `pub extern "C" fn pcai_build_router_dataset_jsonl(tools_json: *const c_char, scenarios_json: *const c_char, out_jsonl: *const c_char, out_vectors: *const c_char) -> i32`
  * **C# P/Invoke Definition:** `[DllImport("pcai_rust.dll")] static extern int pcai_build_router_dataset_jsonl(string toolsJson, string scenariosJson, string outJsonl, string outVectors);`
  * **PowerShell Strategy:** Add-Type wrapper calls the C# method; emits compact status JSON.
* **LLM Data Benefit:** Compact, stable JSONL/test vectors with consistent tool-call shapes.

- Train LoRA:
  cargo run --release -- train \
    --model-path C:\Users\david\PC_AI\Models\functiongemma-270m-it \
    --train-data rust_train_data.jsonl \
    --output output\functiongemma-lora \
    --pack-sequences \
    --max-seq-len 512

- Train LoRA (cached tokens):
  cargo run --release -- train \
    --model-path C:\Users\david\PC_AI\Models\functiongemma-270m-it \
    --train-data rust_train_data.jsonl \
    --token-cache C:\Users\david\PC_AI\output\functiongemma-token-cache \
    --output output\functiongemma-lora

- Eval:
  cargo run --release -- eval \
    --model-path C:\Users\david\PC_AI\Models\functiongemma-270m-it \
    --test-data rust_train_data.jsonl \
    --metrics-output C:\Users\david\PC_AI\Reports\functiongemma_eval_metrics.json \
    --fast-eval

- Merge adapters:
  cargo run --release -- merge \
    --model-path C:\Users\david\PC_AI\Models\functiongemma-270m-it \
    --adapters output\functiongemma-lora\adapter_model.safetensors \
    --output output\functiongemma-merged.safetensors

## Resource control
- Docker-based training is disabled by default.
- To run Docker training explicitly, set:
  PCAI_ENABLE_DOCKER_TRAINING=1

## Status
This is an early scaffold. It compiles core model pieces but lacks feature parity
with the Python pipeline and has several TODOs (see TODO.md).
