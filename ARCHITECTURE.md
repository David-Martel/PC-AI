# PC_AI Architecture

## Overview

PC_AI is a local-first diagnostics and optimization framework for Windows 10/11 with
WSL2, Docker, and GPU tooling. It combines PowerShell orchestration, Rust/C# native
acceleration, and local LLMs via **pcai-inference** (OpenAI-compatible HTTP + native FFI).

Key goals:

- Deterministic diagnostics with explicit tool execution
- Local LLM reasoning with clear safety constraints
- Optional tool-calling router for safe automation (FunctionGemma)

## Core Components

```
PC-AI.ps1 (CLI)
  └─ Modules (PowerShell)
     ├─ PC-AI.Hardware / Network / USB / Virtualization / Performance / Cleanup
     ├─ PC-AI.LLM (LLM orchestration + routing)
     ├─ PC-AI.Evaluation (LLM evaluation + benchmarking)
     ├─ PC-AI.Acceleration (Rust CLI + native DLLs)
     ├─ PC-AI.CLI / PC-AI.Common
     └─ Native/ (Rust + C# P/Invoke + HTTP APIs)
          ├─ pcai_core (Rust workspace: inference + libs)
          ├─ PcaiNative (.NET interop)
          ├─ PcaiChatTui (C# TUI)
          └─ PcaiServiceHost (C# service orchestration)
```

## LLM + Router Pipeline

```
User Request
   │
   ├─ (Optional) FunctionGemma Router (via Invoke-LLMChatRouted)
   │      ├─ Uses pcai-tools.json tool schema
   │      ├─ Selects/executes PowerShell tools
   │      └─ Returns tool outputs
   │
   └─ Primary LLM (pcai-inference)
          ├─ System prompt: DIAGNOSE.md + DIAGNOSE_LOGIC.md (diagnose)
          └─ System prompt: CHAT.md (chat)
```

## Native LLM Pipeline (FFI)

```
pcai_inference.dll (Rust FFI)
   ├─ llama.cpp backend (GGUF, CUDA)
   └─ mistral.rs backend (Candle, CUDA)
        │
        ├─ PowerShell (PcaiInference module)
        ├─ PcaiChatTui (C# native chat)
        └─ PcaiServiceHost (inference status/init/load)
```

Native inference is exposed through `pcai_inference.dll` and shared across:

- **PC-AI.ps1 / Modules** via `Modules/PcaiInference.psm1`
- **PcaiChatTui** via `PcaiNative.InferenceModule`
- **PcaiServiceHost** for status/init/load/generate tooling

## Diagnostic Flow (Diagnose Mode)

1. Collect system data via PC-AI modules (Hardware/Virtualization/Network/USB).
2. (Optional) Router selects additional tools based on report gaps.
3. Assemble diagnostic report and tool outputs.
4. Invoke LLM analysis with DIAGNOSE.md + DIAGNOSE_LOGIC.md.
5. Generate structured recommendations.

## Chat Flow (Chat Mode)

1. Use CHAT.md for system prompt.
2. (Optional) Router selects and executes tools if needed.
3. Main LLM produces final response.

## Configuration

- `Config/llm-config.json`: pcai-inference + router endpoints, defaults, tool schema.
- `Config/pcai-functiongemma.json`: FunctionGemma runtime + training defaults.
- `Config/pcai-functiongemma-eval.json`: CPU-safe FunctionGemma training/eval defaults.
- `Config/pcai-tools.json`: tool schema for FunctionGemma.
- `DIAGNOSE.md`, `DIAGNOSE_LOGIC.md`: diagnostic system prompts.
- `CHAT.md`: general chat system prompt.
- `Config/hvsock-proxy.conf`: optional HVSocket aliases for local routing (`hvsock://pcai-inference`, `hvsock://functiongemma`).
- `.pcai/`: local build + evaluation artifacts (gitignored, rg-accessible).

## Extending Tool Coverage

1. Add a tool definition in `Config/pcai-tools.json`.
2. Map it to a PowerShell cmdlet/module in the `pcai_mapping` section.
3. Add scenario examples in `Deploy/rust-functiongemma-train/examples/scenarios.json`.
4. Rebuild training data and fine-tune FunctionGemma.

FunctionGemma uses the native function-call tag format:

```
<start_function_call>call:tool_name{arg:<escape>value<escape>}<end_function_call>
```

Training/evaluation data is generated to match this format.

## FunctionGemma Training Memory

- Base model weights load via safetensors mmap (CPU RAM efficient).
- Token caches are memmap-backed to minimize RAM pressure during training.
- `Config/pcai-functiongemma.json` supports `train.max_seq_len` to cap context
  length and reduce VRAM use.
- `train.cuda_visible_devices` + `train.min_vram_mb` steer GPU selection toward
  higher-memory devices.

## Documentation Automation

- `Tools/Invoke-DocPipeline.ps1`: full documentation + training pipeline (Rust, PowerShell, C#).
- `Tools/generate-tools-catalog.ps1`: catalogs `Tools/*.ps1` scripts + help metadata.
- `Tools/generate-auto-docs.ps1`: lightweight auto-docs summary.
- Reports written to `Reports/` (e.g. `DOC_PIPELINE_REPORT.md`, `AUTO_DOCS_SUMMARY.md`).

## Evaluation Flow (LLM Backends)

1. Load evaluation suite + dataset in `Modules/PC-AI.Evaluation`.
2. Initialize backend (FFI or compiled server).
3. Execute test cases with streaming progress + JSONL events.
4. Persist outputs under `.pcai\evaluation\runs\<timestamp-label>\`.
5. Optional: compare baselines and regressions.
