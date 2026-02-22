# AGENTS.md

Local-first diagnostics agent guidance for PC_AI (PowerShell + Rust/C# + local LLMs).

## High-level goals

- Deterministic diagnostics with explicit tool execution (no hallucinated system state).
- Safety-first: read-only by default; destructive actions require explicit consent.
- Local LLM reasoning with clear prompts and structured output.
- Performance via Rust native DLLs and Rust CLI fallbacks.

## Architecture quick map

- `PC-AI.ps1` is the unified CLI entry point.
- PowerShell modules live under `Modules/` (Hardware/Virtualization/USB/Network/Performance/Cleanup/LLM/Acceleration).
- Native acceleration: `Native/` (Rust DLLs → C# P/Invoke → PowerShell wrapper).
- LLM inference: `Native/pcai_core/pcai_inference/` (Rust, dual-backend: llama.cpp or mistral.rs)
- Router pipeline (optional):
    1. FunctionGemma runtime selects tools from `Config/pcai-tools.json`
    2. Tool outputs are gathered
    3. Primary LLM (pcai-inference) writes the response

## Prompt contracts

- Diagnose mode: `DIAGNOSE.md` + `DIAGNOSE_LOGIC.md`
    - Output must be valid JSON per `Config/DIAGNOSE_TEMPLATE.json`
    - Evidence-first: tie findings to exact report/log lines
- Chat mode: `CHAT.md` (concise, safe, actionable)

## LLM + Router integration

- Provider config: `Config/llm-config.json`
    - `pcai-inference` → `http://127.0.0.1:8080` (OpenAI-compatible)
    - `functiongemma` → `http://127.0.0.1:8000` (router runtime)
- HVSocket aliases: `Config/hvsock-proxy.conf` (optional)
- Router entry points: `Invoke-FunctionGemmaReAct`, `Invoke-LLMChatRouted`

## Unified Build System

The project uses a unified build orchestrator for all components:

```powershell
# Build all components (recommended)
.\Build.ps1

# Build specific backend with CUDA
.\Build.ps1 -Component llamacpp -EnableCuda

# Build both inference backends
.\Build.ps1 -Component inference -EnableCuda

# Clean build and create release packages
.\Build.ps1 -Clean -Package -EnableCuda
```

**Output structure:**
```
.pcai/build/
├── artifacts/           # Final distributable binaries
│   ├── pcai-llamacpp/   # llamacpp backend
│   ├── pcai-mistralrs/  # mistralrs backend
│   └── manifest.json    # Build manifest with version + SHA256 hashes
├── logs/                # Timestamped build logs
└── packages/            # Release ZIPs (with -Package flag)
```

## pcai-inference compilation

For direct backend builds (debugging/advanced):

```powershell
cd Native\pcai_core\pcai_inference

# CPU-only build (llamacpp backend)
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -Configuration Release

# CUDA GPU build
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -Configuration Release -EnableCuda

# Both backends with CUDA
.\Invoke-PcaiBuild.ps1 -Backend all -Configuration Release -EnableCuda
```

**Build prerequisites:**
- Visual Studio 2022 C++ Build Tools + Windows SDK
- CMake 3.x (auto-detected from VS)
- CUDA 12.x+ (for `-EnableCuda`)

**Performance optimizations (auto-enabled):**
- sccache: Compiler caching for faster rebuilds
- Ninja generator: Parallel CMake builds
- lld-link: Fast LLVM linker
- CRT alignment: Forces `/MD` to avoid CUDA linker errors

**Backend comparison:**
| Backend | Strength | When to use |
|---------|----------|-------------|
| `llamacpp` | Mature, GGUF support, lower VRAM | Default, most models |
| `mistralrs` | Flash attention, cuDNN, newer arch | Mistral/Llama3 with 12GB+ VRAM |

## Version System

Version info is embedded at compile time from git metadata:

```powershell
# Get version info
.\Tools\Get-BuildVersion.ps1

# Output as JSON
.\Tools\Get-BuildVersion.ps1 -Format Json

# Set environment variables for builds
.\Tools\Get-BuildVersion.ps1 -SetEnv
```

**Version format:** `{semver}.{commits}+{hash}[.dirty]`
- `0.2.0.15+abc1234` - 15 commits since v0.2.0
- `0.2.0+abc1234` - exactly at tag v0.2.0
- `0.2.0.3+abc1234.dirty` - uncommitted changes

**Runtime access:**
```powershell
# Check binary version
.\pcai-llamacpp.exe --version
.\pcai-llamacpp.exe --version-json

# HTTP endpoint
GET http://127.0.0.1:8080/version
```

## LLM runtime debugging (native-first)

Use these when diagnosing LLM stack failures or routing issues:

- `Invoke-PcaiDoctor` (summary + recommendations)
- `Get-PcaiServiceHealth` (HTTP + FFI health checks)
- `Get-PcaiNativeStatus` / `Get-PcaiCapabilities` (native DLL availability)
- LLM endpoints:
    - pcai-inference: `GET http://127.0.0.1:8080/health` or `/v1/models`
    - FunctionGemma router: `GET http://127.0.0.1:8000/health` or `/v1/models`

## LLM evaluation harness

Use the evaluation runner to benchmark inference backends and capture structured outputs:

- Runner: `Tests\Evaluation\Invoke-InferenceEvaluation.ps1`
- Default output root: `.pcai\evaluation\runs\`
- Per-run outputs: `events.jsonl`, `progress.log`, `summary.json`, `stop.signal`

Example:
```powershell
pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin `
  -ModelPath "C:\Models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
  -Dataset diagnostic `
  -MaxTestCases 5 `
  -ProgressMode stream `
  -RunLabel local-smoke
```

Stop a run:
```powershell
Stop-EvaluationRun
```

## Tooling update workflow

1. Add/update tool in `Config/pcai-tools.json` (with `pcai_mapping`).
2. Add/update scenarios in `Deploy/rust-functiongemma-train/examples/scenarios.json`.
3. Rebuild training data + fine-tune FunctionGemma.
4. If tool changes impact diagnostics, update `DIAGNOSE.md` / `DIAGNOSE_LOGIC.md`.

## Native acceleration guidelines (CSharp_RustDLL)

- Prioritize heavy loops, deep recursion, or regex-heavy operations.
- Emit compact, stable JSON for LLM ingestion.
- Use C ABI (`extern "C"`) + C# P/Invoke wrapper for PowerShell.

## CI/CD: Native Binary Releases

Pre-compiled CUDA binaries are published via GitHub Actions on version tags:

```bash
# Tag a release
git tag v1.0.0
git push origin v1.0.0
```

**Release artifacts (4 variants):**
| File | Backend | GPU |
|------|---------|-----|
| `pcai-inference-llamacpp-cuda-win64.zip` | llama.cpp | CUDA |
| `pcai-inference-llamacpp-cpu-win64.zip` | llama.cpp | CPU-only |
| `pcai-inference-mistralrs-cuda-win64.zip` | mistral.rs | CUDA |
| `pcai-inference-mistralrs-cpu-win64.zip` | mistral.rs | CPU-only |

**CUDA GPU targets:** SM 75 (Turing), SM 80/86 (Ampere), SM 89 (Ada)

**Workflow:** `.github/workflows/release-cuda.yml`

## Known gaps / TODOs

- Define a versioned C ABI contract for Rust DLL exports (error codes, ownership).
- Standardize JSON schemas for native outputs (schema folder + version pinning).
- Provide progress + streaming updates for long native operations.
- Finalize eval split + QLoRA quantization for rust-functiongemma-train.

## Documentation automation

- Full pipeline: `Tools/Invoke-DocPipeline.ps1 -Mode Full`
- Docs-only: `Tools/Invoke-DocPipeline.ps1 -Mode DocsOnly`
- Lightweight summaries: `Tools/generate-auto-docs.ps1 -BuildDocs`
- Tools catalog: `Tools/generate-tools-catalog.ps1`
