# Rust FunctionGemma Runtime (PC_AI)

This is the Rust runtime server intended to replace the Python/vLLM FunctionGemma
router in PC_AI. It exposes OpenAI-compatible endpoints and returns tool_calls.

## Endpoints
- GET /health
- GET /v1/models
- POST /v1/chat/completions

## Build (Primary: Build.ps1)
From repo root:

  .\Build.ps1 -Component functiongemma

## Tests (Primary: Build.ps1)

  .\Build.ps1 -Component functiongemma -RunTests

## Run

  # Edit Config/pcai-functiongemma.json (runtime section) as needed
  .\Deploy\rust-functiongemma-runtime\target\debug\rust-functiongemma-runtime.exe --config Config\pcai-functiongemma.json

Notes:
- Default address is 127.0.0.1:8000 to match Invoke-FunctionGemmaReAct defaults.
- The default engine is heuristic: it emits tool_calls when the user request
  mentions a tool name, otherwise it returns NO_TOOL.

## Model inference (experimental)
Build with model features, then enable model engine:

  # Preferred repo-wide build path
  .\Build.ps1 -Component functiongemma

  # Advanced crate-level override (optional)
  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-runtime -CargoArgs @('build','--features','model')
  # Update Config/pcai-functiongemma.json:
  #   runtime.router_engine = "model"
  #   runtime.router_model_path = "C:\\Users\\david\\PC_AI\\Models\\functiongemma-270m-it"

This path loads the base model and attempts to parse FunctionGemma-style
tool calls from the generated output. It is functional but not optimized.

## KV cache
KV cache is enabled by default for model inference. Toggle in
Config/pcai-functiongemma.json (runtime.router_kv_cache).

## Optional model features
Enable extra dependencies only when needed:

  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-runtime -CargoArgs @('build','--features','model')

This enables minijinja, tokenizers, hf-hub, and safetensors for future
model loading + chat template rendering.

## Router behavior (current)
This runtime currently returns:
- tool_calls when the user request mentions a tool name, or
- NO_TOOL otherwise.

The tool selection logic is intentionally minimal and will be replaced
by real FunctionGemma inference.
