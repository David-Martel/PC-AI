# GEMINI.md

This repo supports local LLM routing and diagnostics. When interacting with the project, adhere to these core operating guidelines:

## Core Agent Guidelines

- **Always adhere to safety boundaries**: Act read-only by default. Destructive operations (e.g., hardware/disk modifications) require explicit user knowledge, consent, and warnings.
- **Reference existing documentation**: Consult `CLAUDE.md` for architectural blueprints and project expectations, and `AGENTS.md` for additional deterministic and native-tool constraints.
- Use `DIAGNOSE.md` + `DIAGNOSE_LOGIC.md` for diagnostic output generation (evidence-first reasoning, valid JSON).
- Use `CHAT.md` for general assistance.
- Prefer the FunctionGemma runtime for tool selection (`Invoke-LLMChatRouted`).
- **Use Native Acceleration**: Prefer Rust-accelerated modules (`PC-AI.Acceleration`) over standard PowerShell file tools when parsing or searching on the host.

## Tool schema and routing

- `Config/pcai-tools.json` defines tool names and PowerShell mappings.
- Training and evaluation: `Deploy/rust-functiongemma-train/`
- Router runtime: `Deploy/rust-functiongemma-runtime/`
- Shared core libraries: `Deploy/rust-functiongemma-core/`
- HVSocket endpoints: `hvsock://pcai-inference` and `hvsock://functiongemma` configured via `Config/hvsock-proxy.conf`.

## Unified Build System

```powershell
# Build all components (recommended)
.\Build.ps1

# Build with CUDA GPU acceleration
.\Build.ps1 -Component llamacpp -EnableCuda

# Build both backends with release packages
.\Build.ps1 -Component inference -EnableCuda -Package
```

**Output:** `.pcai/build/artifacts/` with `manifest.json` (version + SHA256 hashes)

## pcai-inference and Native Acceleration

The project utilizes a hybrid architecture consisting of native Rust code exposed via FFI to .NET/C# and consumed by PowerShell frameworks.

- **Root Workspace**: `Native/pcai_core/` for core Rust implementations (inference engines, shared libraries).
- **Interop layer**: `Native/PcaiNative/` and `Native/PcaiChatTui/` interface Rust APIs with user interfaces, surfacing via `PC-AI.Acceleration`.

Location: `Native/pcai_core/pcai_inference/`

**Direct build commands (advanced):**

```powershell
cd Native\pcai_core\pcai_inference
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -EnableCuda
.\Invoke-PcaiBuild.ps1 -Backend all -EnableCuda
```

**Backends:**

- `llamacpp`: llama.cpp via llama-cpp-2 (default, GGUF models)
- `mistralrs`: mistral.rs with flash attention (12GB+ VRAM)

**Version info:** `.\pcai-llamacpp.exe --version` or `GET /version`

**API endpoints:**

- `GET /health` - Health check
- `GET /v1/models` - List loaded models
- `POST /v1/completions` - Generate completion
- `GET /version` - Build info (git hash, timestamp, features)

**Performance config (`Config/llm-config.json`):**

```json
{
    "backend": { "type": "llama_cpp", "n_gpu_layers": 35, "n_ctx": 4096 },
    "model": { "path": "Models/model.gguf" }
}
```

GPU layer offload: 4GB→10-15 layers, 8GB→25-30, 12GB→35-40, 24GB→50+

## Evaluation Harness

Runner: `Tests/Evaluation/Invoke-InferenceEvaluation.ps1`

Outputs go to `.pcai/evaluation/runs/<timestamp-label>/`:
`events.jsonl`, `progress.log`, `summary.json`, `stop.signal`.

```powershell
pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin `
  -ModelPath "C:\Models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
  -Dataset diagnostic `
  -MaxTestCases 5 `
  -ProgressMode stream `
  -RunLabel local-smoke
```

## CI/CD Releases

Pre-compiled CUDA binaries: `.github/workflows/release-cuda.yml`

- Tag `v*` to trigger release builds
- Artifacts: `pcai-inference-{backend}-{cuda|cpu}-win64.zip`
- GPU targets: SM 75/80/86/89 (Turing through Ada)
