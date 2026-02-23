# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PC_AI** is a local LLM-powered PC diagnostics and optimization agent designed to:

- Diagnose hardware issues, device errors, and system problems
- Analyze event logs, SMART status, and device configurations
- Propose optimizations for disk, network, and system performance
- Clean up duplicates, PATH entries, and unnecessary system artifacts
- Route tool execution via FunctionGemma runtime before final LLM analysis

The agent operates on **Windows 10/11** with native-first inference via **pcai-inference**. WSL/Docker are optional and not required for the LLM stack.

## Architecture

```
PC_AI/
├── DIAGNOSE.md                        # LLM system prompt defining assistant behavior
├── DIAGNOSE_LOGIC.md                  # Branched reasoning decision tree for analysis
├── CHAT.md                            # General chat system prompt
├── Native/pcai_core/                  # Rust workspace (monorepo)
│   ├── pcai_inference/                # Rust LLM inference engine (HTTP + FFI)
│   └── pcai_core_lib/                 # Shared Rust library (telemetry, fs, search)
├── Native/PcaiNative/                 # C# P/Invoke wrapper for PowerShell
├── Native/PcaiChatTui/                # C# interactive chat TUI (async backends)
├── Native/PcaiServiceHost/            # C# Windows service host for inference
├── Native/NukeNul/                    # C# utility for null-file removal
├── Deploy/rust-functiongemma/         # Rust workspace root (includes core/runtime/train)
├── Deploy/rust-functiongemma-core/    # Shared Rust library (model, GPU, prompt, config)
├── Deploy/rust-functiongemma-runtime/ # Rust router runtime (axum HTTP server)
├── Deploy/rust-functiongemma-train/   # Rust router dataset + training pipeline
├── Modules/
│   ├── PC-AI.Acceleration/            # Native FFI wrappers (P/Invoke bridge)
│   ├── PC-AI.Cleanup/                 # Duplicate detection, PATH cleanup
│   ├── PC-AI.CLI/                     # Command map, help extraction, argument parsing
│   ├── PC-AI.Common/                  # Shared utilities (error formatting, path resolution)
│   ├── PC-AI.Evaluation/              # LLM evaluation framework (metrics, A/B testing)
│   ├── PC-AI.Hardware/                # Device, disk, USB, network diagnostics
│   ├── PC-AI.LLM/                     # Local LLM integration (Ollama, native, TUI)
│   ├── PC-AI.Network/                 # Network diagnostics and monitoring
│   ├── PC-AI.Performance/             # Native perf metrics (Rust FFI)
│   ├── PC-AI.USB/                     # USB device management
│   ├── PC-AI.Virtualization/          # WSL2, Hyper-V, HVSocket proxy
│   └── PcaiInference.psm1             # Inference module (load/generate/stream/async)
├── Tools/
│   ├── Invoke-DocPipeline.ps1         # Master doc generation orchestrator
│   ├── generate-auto-docs.ps1         # Unified auto-doc (ast-grep, PS, C#, Rust)
│   ├── validate-doc-accuracy.ps1      # Doc accuracy validation
│   ├── Build.ps1 / Invoke-RustBuild   # Build system scripts
│   └── ...                            # 30+ utility scripts
├── Config/
│   ├── llm-config.json                # Backend + model configuration
│   ├── pcai-tools.json                # FunctionGemma tool schema
│   └── pcai-functiongemma.json        # Router GPU + runtime config
├── .litho/litho.toml                  # Litho (deepwiki-rs) documentation config
└── CLAUDE.md                          # This file
```

### Design Pattern

1. **DIAGNOSE.md** - Defines the LLM assistant's role, safety constraints, and workflow
2. **DIAGNOSE_LOGIC.md** - Branched reasoning logic for analyzing diagnostic output
3. **Modules/PC-AI.Hardware/** - PowerShell module that collects system data via structured functions

The agent follows a **collect → parse → route → reason → recommend** workflow where diagnostics output is structured into categories, optional tool routing is executed via the FunctionGemma runtime, and the main LLM produces recommendations.

## Documentation Generation

### Automated Documentation Pipeline

```powershell
# Full documentation pipeline (docs + training data)
.\Tools\Invoke-DocPipeline.ps1 -Mode Full

# Documentation only (no training data)
.\Tools\Invoke-DocPipeline.ps1 -Mode DocsOnly

# Auto-docs with all generators (ast-grep, PS, C#, Rust)
.\Tools\generate-auto-docs.ps1

# Validate documentation accuracy
.\Tools\validate-doc-accuracy.ps1
```

### Litho (deepwiki-rs) Architecture Documentation

```powershell
# Generate comprehensive architecture docs via LLM analysis
# Requires Ollama running with qwen2.5-coder:7b
deepwiki-rs.exe -c .litho/litho.toml -p . -v

# Output: docs/auto/deepwiki_litho_docs/
```

Config: `.litho/litho.toml` — knowledge categories cover architecture, modules, native code, deployment, tests, reports, and build tools.

### Reports Directory

The doc pipeline generates structured reports under `Reports/`:

- `DOC_STATUS.md` — TODO/FIXME/DEPRECATED markers across codebase
- `PS_MODULE_INDEX.md` — PowerShell function index by module
- `API_SIGNATURE_REPORT.md` — Cross-layer API alignment
- `TOOLS_CATALOG.md` — All Tools/ scripts with descriptions
- `POWERSHELL_EXPORTS.json` — Machine-readable module exports
- `DOC_PIPELINE_REPORT.json` — Pipeline run summary

## Commands

### Unified Build System

```powershell
# Build all components (recommended)
.\Build.ps1

# Build specific component with CUDA
.\Build.ps1 -Component llamacpp -EnableCuda

# Build both inference backends
.\Build.ps1 -Component inference -EnableCuda

# Build C# Chat TUI
.\Build.ps1 -Component tui

# Clean build and create release packages
.\Build.ps1 -Clean -Package -EnableCuda

# Debug build
.\Build.ps1 -Component mistralrs -Configuration Debug
```

**Build Output Structure:**

```
.pcai/build/
├── artifacts/           # Final distributable binaries
│   ├── pcai-llamacpp/   # llamacpp backend (exe + dll)
│   ├── pcai-mistralrs/  # mistralrs backend (exe + dll)
│   ├── functiongemma/   # FunctionGemma router
│   └── manifest.json    # Build manifest with version + SHA256 hashes
├── logs/                # Timestamped build logs
└── packages/            # Release ZIPs (with -Package flag)
```

Override artifact location: `$env:PCAI_ARTIFACTS_ROOT = 'D:\build'`

### Version Information

```powershell
# Get version info from git metadata
.\Tools\Get-BuildVersion.ps1

# Set version environment variables for build
.\Tools\Get-BuildVersion.ps1 -SetEnv

# Output formats
.\Tools\Get-BuildVersion.ps1 -Format Json    # JSON output
.\Tools\Get-BuildVersion.ps1 -Format Env     # Shell export format
.\Tools\Get-BuildVersion.ps1 -Format Cargo   # Cargo rustc-env format
```

**Version Format:** `{semver}.{commits}+{hash}[.dirty]`

- Example: `0.2.0.15+abc1234` (15 commits since v0.2.0, hash abc1234)
- Example: `0.2.0+abc1234` (exactly at tag v0.2.0)
- Example: `0.2.0.3+abc1234.dirty` (uncommitted changes)

**Embedded in binaries:**

- `pcai-llamacpp.exe --version` shows full build info
- `/version` endpoint returns JSON with git hash, timestamp, features

### Direct Backend Build (Advanced)

```powershell
# Build with low-level script (for debugging build issues)
cd Native\pcai_core\pcai_inference
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -Configuration Release -EnableCuda

# Clean build (wipe target/ first)
.\Invoke-PcaiBuild.ps1 -Backend all -Clean
```

**Feature Flags:**
| Feature | Description |
|---------|-------------|
| `llamacpp` | llama.cpp backend (default, mature) |
| `mistralrs-backend` | mistral.rs backend (alternative) |
| `cuda-llamacpp` | CUDA for llama.cpp |
| `cuda-mistralrs` | CUDA for mistral.rs |
| `ffi` | C FFI exports for PowerShell (15 functions) |
| `server` | HTTP server with OpenAI-compatible API |

**Minimal Build Variants:**

```powershell
# FFI-only DLL (289KB, no backend — for P/Invoke testing)
cargo build --no-default-features --features ffi --release

# Server + FFI (no backend — for unit testing pure logic)
cargo test --no-default-features --features server,ffi --lib
```

### Evaluation Harness

Use the LLM evaluation runner to benchmark backends and capture structured run outputs:

```powershell
pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin `
  -ModelPath "C:\Models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
  -Dataset diagnostic `
  -MaxTestCases 5 `
  -ProgressMode stream `
  -RunLabel local-smoke
```

Outputs are written under `.pcai\evaluation\runs\<timestamp-label>\` with:
`events.jsonl`, `progress.log`, `summary.json`, and `stop.signal`.

**Performance Tips:**

- Enable sccache: `Tools\Initialize-CacheEnvironment.ps1`
- Use Ninja generator (auto-detected)
- CUDA builds require matching CRT: script auto-forces `/MD`

### Run Hardware Diagnostics

```powershell
# Requires Administrator
Import-Module PC-AI.Hardware

# Individual diagnostics
Get-DeviceErrors          # Device Manager errors
Get-DiskHealth            # SMART disk status
Get-UsbStatus             # USB controller/device status
Get-NetworkAdapters       # Physical network adapters
Get-SystemEvents -Days 3  # Recent critical system events

# Full combined report
New-DiagnosticReport      # Generates complete diagnostic report
# Creates: Desktop\Hardware-Diagnostics-Report.txt
```

### Output Sections

The diagnostic report contains:

1. **Device Manager Errors** - Devices with ConfigManagerErrorCode != 0
2. **Disk SMART Status** - Drive health via wmic
3. **System Event Errors** - Disk/USB errors from last 3 days
4. **USB Device Status** - USB controllers and device status
5. **Network Adapter Status** - Physical adapter configuration

### CI/CD: Releasing Native Binaries

The project uses GitHub Actions to build and release pre-compiled CUDA binaries.

**Trigger a release:**

```bash
# Tag a version to trigger the release workflow
git tag v1.0.0
git push origin v1.0.0
```

**Manual trigger (for testing):**

- Go to Actions > "Release Native Binaries" > Run workflow
- Enter a tag name (e.g., `v1.0.0-beta`)

**Release artifacts (4 variants):**
| File | Backend | GPU |
|------|---------|-----|
| `pcai-inference-llamacpp-cuda-win64.zip` | llama.cpp | CUDA |
| `pcai-inference-llamacpp-cpu-win64.zip` | llama.cpp | CPU-only |
| `pcai-inference-mistralrs-cuda-win64.zip` | mistral.rs | CUDA |
| `pcai-inference-mistralrs-cpu-win64.zip` | mistral.rs | CPU-only |

**CUDA builds target:**

- SM 75: Turing (RTX 20 series, GTX 16xx)
- SM 80/86: Ampere (RTX 30 series)
- SM 89: Ada Lovelace (RTX 40 series)

**Workflow file:** `.github/workflows/release-cuda.yml`

## Potential Enhancements from Home Directory Scripts

The following scripts from the home directory could potentially enhance PC-AI capabilities:

### Disk Optimization

- `Optimize-Disks.ps1` - Smart TRIM/defrag for SSD/HDD with scheduled task support (consider integrating into PC-AI.Performance)

### Cleanup

- `clean_machine_path.ps1` - Remove duplicate/stale PATH entries (extends PC-AI.Cleanup functionality)
- `cleanup-duplicates.ps1` - Duplicate file detection and removal (complements PC-AI.Cleanup module)

### Performance

- `wezterm-performance-profiler.ps1` - Terminal startup/memory/render benchmarking (candidate for PC-AI.Performance)

## Diagnostic Categories

### Priority Classification

- **Critical**: SMART failures, disk bad blocks, hardware virtualization disabled
- **High**: USB controller errors, device driver failures, service crashes
- **Medium**: Performance degradation, missing Defender exclusions, VMQ issues
- **Low**: Unused adapters, informational warnings

### ConfigManagerErrorCode Reference

| Code | Meaning                           |
| ---- | --------------------------------- |
| 1    | Device not configured correctly   |
| 10   | Device cannot start               |
| 12   | Cannot find enough free resources |
| 22   | Device is disabled                |
| 28   | Drivers not installed             |
| 31   | Device not working properly       |
| 43   | Device stopped responding         |

## Safety Constraints

- **Read-only by default** - Diagnostics collect data without modifications
- **No destructive commands** without explicit user consent and backup warnings
- **Disk repair** (chkdsk /r) requires backup confirmation first
- **BIOS/firmware updates** need context and warning
- **Professional escalation** for suspected hardware failure

## Integration Points

### Event Log Queries

```powershell
# Disk/USB errors
Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2,3; StartTime=(Get-Date).AddDays(-3)}

```

### WMI/CIM Queries

```powershell
# Device errors
Get-CimInstance Win32_PnPEntity | Where-Object { $_.ConfigManagerErrorCode -ne 0 }

# Physical network adapters
Get-CimInstance Win32_NetworkAdapter | Where-Object { $_.PhysicalAdapter -eq $true }

# Disk status
wmic diskdrive get model, status
```

### FunctionGemma Router

- Tool schema: `Config/pcai-tools.json`
- Router interface: `Invoke-FunctionGemmaReAct` / `Invoke-LLMChatRouted`
- Shared library: `Deploy/rust-functiongemma-core/` (model, GPU, prompt, config, LoRA, safetensors)
- Training pipeline: `Deploy/rust-functiongemma-train/`
- Runtime server: `Deploy/rust-functiongemma-runtime/` (axum, port 8000)
- HVSocket aliases: `Config/hvsock-proxy.conf` with `hvsock://functiongemma` / `hvsock://pcai-inference`

### pcai-inference Endpoints

- Health check: `GET http://127.0.0.1:8080/health`
- Models list: `GET http://127.0.0.1:8080/v1/models`
- Completion: `POST http://127.0.0.1:8080/v1/completions`

### Async Inference (FFI)

- `pcai_generate_async(prompt, max_tokens, temperature)` - Initiate async generation, returns request ID
- `pcai_poll_result(request_id)` - Poll result status without blocking, returns status + partial text
- `pcai_cancel(request_id)` - Cancel async request, returns success/failure

PowerShell wrappers:

- `Invoke-PcaiGenerateAsync` - Async generation with `-NoWait` for manual polling
- `Get-PcaiAsyncResult` - Poll or wait for async result completion
- `Stop-PcaiGeneration` - Cancel ongoing async request

## Expected Output Format

When reporting findings, use this structure:

```
## Summary
- [2-4 bullet points of key findings]

## Findings by Category
### Devices with Errors
### Disk Health
### USB Stability
### Network Adapters

## Priority Issues
- Critical: [list]
- High: [list]
- Medium: [list]

## Recommended Next Steps
1. [Numbered, safe actions]
2. [Warnings for risky operations]
```

## Development Notes

### Adding New Diagnostics

1. Add data collection function to `Modules/PC-AI.Hardware/Public/` or relevant module
2. Add parsing logic to `DIAGNOSE_LOGIC.md`
3. Update category handling in `DIAGNOSE.md`
4. Export function in module manifest (`.psd1` file)

### Testing

**Rust Unit Tests (61+ tests):**

```powershell
cd Native\pcai_core\pcai_inference
cargo test --no-default-features --features server,ffi --lib
```

| Module            | Tests | Coverage                                                 |
| ----------------- | ----- | -------------------------------------------------------- |
| `lib.rs`          | 5     | Error Display + From conversions                         |
| `config.rs`       | 8     | Serde roundtrip, file I/O, defaults                      |
| `backends/mod.rs` | 7     | Request/Response serde, FinishReason                     |
| `http/mod.rs`     | 21    | Chat prompt, tokens, stop sequences, chunks, StopTracker |
| `ffi/mod.rs`      | 17    | FFI edge cases, init/shutdown, error codes               |
| Version           | 3     | Build version detection and formatting                   |

**FFI Integration Tests (49 tests):**

```powershell
# Integration tests (28 tests — requires DLL at bin/pcai_inference.dll)
pwsh -Command "Invoke-Pester Tests/Integration/FFI.Inference.Tests.ps1"

# Stress tests (21 tests — concurrency, memory, error hammering)
pwsh -Command "Invoke-Pester Tests/Integration/FFI.Stress.Tests.ps1"
```

**PowerShell Diagnostics:**

```powershell
# Import and test diagnostics module
Import-Module PC-AI.Hardware
Get-DeviceErrors

# Verify report creation
Test-Path "$env:USERPROFILE\Desktop\Hardware-Diagnostics-Report.txt"
```

### PowerShell Requirements

- Requires Administrator for full diagnostics
- Uses Get-CimInstance (not deprecated Get-WmiObject)
- Handles missing features gracefully with try/catch

### pcai-inference Build Requirements

**Required:**

- Visual Studio 2022 with C++ Build Tools + Windows SDK
- CMake 3.x (included with VS or `winget install Kitware.CMake`)
- Rust toolchain (`rustup`)

**Optional (for GPU):**

- CUDA Toolkit 13.x (`CUDA_PATH` env var) — 13.1 tested and working
- cuDNN (for mistral.rs flash attention)
- sccache (for faster rebuilds)

**GPU Configuration:**
| GPU | VRAM | Compute | Role |
|-----|------|---------|------|
| RTX 2000 Ada | 8GB | SM 89 | Inference / runtime (GPU 0) |
| RTX 5060 Ti | 16GB | SM 120 | Training / QLoRA (GPU 1) |

Config files: `Config/pcai-functiongemma.json` (router_gpu, cuda_visible_devices)

**Common Build Issues:**
| Issue | Solution |
|-------|----------|
| "GNU compiler not supported" | Run from VS Developer PowerShell, not WSL/MinGW |
| "CMake not found" | `winget install Kitware.CMake`, restart terminal |
| "CUDA not found" | Install CUDA Toolkit, verify `$env:CUDA_PATH` |
| CRT mismatch linker errors | Script auto-forces `/MD`; run with `-Clean` if switching backends |
| CUDA env override needed | `Tools\Initialize-CudaEnvironment.ps1 -WorkaroundMsvc1944` or `.cargo/config.toml` env vars |

### Performance Configuration

```json
// Config/llm-config.json
{
  "backend": {
    "type": "llama_cpp",
    "n_gpu_layers": 35, // GPU offload (0 = CPU only)
    "n_ctx": 4096 // Context window
  },
  "model": {
    "path": "Models/model.gguf",
    "generation": {
      "max_tokens": 512,
      "temperature": 0.7
    }
  }
}
```

**GPU Layer Offload Guide:**
| VRAM | Recommended `n_gpu_layers` |
|------|---------------------------|
| 4GB | 10-15 |
| 8GB | 25-30 |
| 12GB | 35-40 |
| 24GB | 50+ (full offload) |
