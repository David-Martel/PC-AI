# PC-AI: Local LLM-Powered PC Diagnostics Framework

A comprehensive PowerShell 7+ framework for Windows PC diagnostics, optimization, and system analysis powered by local LLMs via **pcai-inference** (Rust).

## Features

- **Hardware Diagnostics**: Device errors, SMART status, USB controllers, network adapters
- **Virtualization Diagnostics (optional)**: Hyper-V/WSL status and remediation tooling
- **Performance Acceleration**: Rust tool integration (ripgrep, fd, procs) with PS7+ parallelism
- **LLM Analysis**: Local AI-powered diagnostic interpretation via **pcai-inference** (OpenAI-compatible HTTP + native FFI)
- **Tool-Calling Router**: **FunctionGemma** runtime selects and executes PC-AI tools before analysis
- **Unified CLI**: Single entry point for all diagnostic and optimization tasks

## Requirements

- **Windows 10/11** with PowerShell 7.0+
- **Optional**: pcai-inference HTTP server or pcai-inference DLL for LLM features
- **Optional**: FunctionGemma runtime (rust-functiongemma-runtime) for tool routing
- **No WSL/Docker required** for pcai-inference or FunctionGemma (native-first)
- **Optional**: Rust CLI tools for acceleration (fd, ripgrep, procs, bat, etc.)

## Quick Start

```powershell
# Import the main modules
Import-Module .\Modules\PC-AI.Hardware\PC-AI.Hardware.psd1
Import-Module .\Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1

# Run hardware diagnostics (requires Admin)
.\PC-AI.ps1 diagnose hardware

# Check available Rust tools for acceleration
Get-RustToolStatus

# Fast file search using fd
Find-FilesFast -Path "C:\Projects" -Pattern "*.ps1"

# Fast content search using ripgrep
Search-ContentFast -Path "C:\Scripts" -Pattern "function" -FilePattern "*.ps1"
```

## Modules

| Module                   | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| **PC-AI.Hardware**       | Device manager, disk health, USB, network diagnostics      |
| **PC-AI.Virtualization** | Optional Hyper-V/WSL diagnostics and optimization          |
| **PC-AI.USB**            | USB device management and WSL passthrough                  |
| **PC-AI.Network**        | Network diagnostics, VSock optimization                    |
| **PC-AI.Performance**    | Disk optimization, resource monitoring                     |
| **PC-AI.Cleanup**        | PATH cleanup, duplicate detection, temp cleanup            |
| **PC-AI.LLM**            | pcai-inference + FunctionGemma integration for AI analysis |
| **PC-AI.Acceleration**   | Rust tools integration with PS7+ parallelism               |

## Native Acceleration (Rust DLL + C# Hybrid)

PC-AI includes a high-performance native layer built with Rust and C#:

### Architecture

```
Rust DLLs (pcai_core_lib.dll, pcai_search.dll)
         ↓
C# P/Invoke Wrapper (PcaiNative.dll, .NET 8)
         ↓
PowerShell 7 Modules (PC-AI.Acceleration)
         ↓
pcai-inference LLM Analysis (local GGUF)
```

### Native Operations

| Operation           | Speedup | Technology                      |
| ------------------- | ------- | ------------------------------- |
| Duplicate Detection | 5-10x   | Parallel SHA-256 with rayon     |
| File Search         | 5-10x   | Parallel glob with ignore crate |
| Content Search      | 3-8x    | Parallel regex matching         |

### Quick Start (Native)

```powershell
# Build components with the unified build orchestrator (recommended)
.\Build.ps1

# Run all tests (requires PowerShell 7 + Pester 5)
pwsh .\Tests\Invoke-AllTests.ps1 -Suite All

# Use native duplicate detection
Import-Module .\Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1
Get-PcaiNativeStatus
Get-PcaiCapabilities -IncludeGpu
Invoke-PcaiNativeDuplicates -Path "D:\Downloads" -MinimumSize 1MB

# Smart diagnosis with LLM
Import-Module .\Modules\PC-AI.LLM\PC-AI.LLM.psd1
Invoke-SmartDiagnosis -Path "C:\Temp" -AnalysisType Quick
```

### Test Results

| Suite     | Passed  | Failed | Duration |
| --------- | ------- | ------ | -------- |
| Rust      | 40      | 0      | ~1s      |
| Pester    | 37      | 0      | ~7s      |
| Module    | 199     | 0      | ~88s     |
| **Total** | **276** | **0**  | **~98s** |

## Rust CLI Tools Integration

The `PC-AI.Acceleration` module also supports external Rust CLI tools with automatic fallback to PowerShell:

### Supported Rust Tools

| Tool           | Use Case        | Speedup              |
| -------------- | --------------- | -------------------- |
| `fd`           | File finding    | 5-10x                |
| `ripgrep` (rg) | Content search  | **40x+**             |
| `procs`        | Process listing | Better formatting    |
| `bat`          | File viewing    | Syntax highlighting  |
| `hyperfine`    | Benchmarking    | Statistical accuracy |

### Performance Pattern

All acceleration functions follow a consistent fallback:

1. **Native DLL** (fastest) - Rust via C# P/Invoke
2. **Rust CLI tool** - if native DLLs unavailable
3. **PS7+ parallel** - ForEach-Object -Parallel
4. **Sequential PS** - compatible fallback

### Example: Content Search Performance

```
Directory: C:\Users (1000+ files)
----------------------------------------
ripgrep (Rust):     391ms
Select-String (PS): 17,471ms
Speedup:            44.6x
```

## Unified CLI

```powershell
# Diagnostics
.\PC-AI.ps1 diagnose hardware
.\PC-AI.ps1 diagnose all

# Optimization
.\PC-AI.ps1 optimize disk

# USB Management
.\PC-AI.ps1 usb list
.\PC-AI.ps1 usb attach <busid>

# LLM Analysis
.\PC-AI.ps1 analyze
.\PC-AI.ps1 analyze --model mistral

# Repair
.\PC-AI.ps1 cleanup path

# Cleanup
.\PC-AI.ps1 cleanup path --dry-run
.\PC-AI.ps1 cleanup temp

# Doctor (runtime health checks)
.\PC-AI.ps1 doctor
```

## LLM Integration

PC-AI integrates with local LLM providers for intelligent diagnostic analysis and tool routing:

### pcai-inference (Default)

```powershell
# Build inference backends with the unified build flow (recommended)
.\Build.ps1 -Component inference -EnableCuda

# Direct backend build flow (advanced)
cd .\Native\pcai_core\pcai_inference
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -Configuration Release -EnableCuda

# Start the local service (creates config + launches)
Invoke-PcaiServiceHost -Backend rust -NativeBackend llamacpp -ModelPath "C:\Models\model.gguf" -GpuLayers 35

# Run analysis with default model (pcai-inference)
Invoke-PCDiagnosis -ReportPath ".\report.txt"
```

### FunctionGemma Router Training (Local)

Use the unified CLI to regenerate datasets, build token caches, train adapters, and evaluate routing quality:

```powershell
# Rebuild router dataset + test vectors
.\PC-AI.ps1 llm router-prepare --stream

# Build token cache for faster training
.\PC-AI.ps1 llm router-cache

# Fine-tune (LoRA) with default settings
.\PC-AI.ps1 llm router-train --epochs 1 --lora-r 16 --lora-alpha 32

# Evaluate routing accuracy (CPU-safe config)
.\PC-AI.ps1 llm router-eval --config .\Config\pcai-functiongemma-eval.json --max-samples 25 --fast-eval
```

Key files:

- `Config/pcai-functiongemma.json` (runtime + training defaults)
- `Config/pcai-functiongemma-eval.json` (CPU-safe training/eval defaults)
- `Deploy/rust-functiongemma-train/examples/scenarios.json` (routing scenarios)

### pcai-native (FFI)

```powershell
# Native FFI (no HTTP server)
Import-Module .\Modules\PcaiInference.psm1
Initialize-PcaiInference -Backend llamacpp
Import-PcaiModel -ModelPath "C:\Models\model.gguf" -GpuLayers 35
Invoke-PcaiInference -Prompt "Summarize this report..."
```

### PcaiServiceHost (FFI utilities)

```powershell
pcai-servicehost inference status
pcai-servicehost inference init --backend llamacpp
pcai-servicehost inference load --model-path C:\Models\model.gguf --gpu-layers 35
pcai-servicehost inference generate --prompt "Quick summary"
pcai-servicehost inference shutdown
```

### FunctionGemma (Tool Router)

FunctionGemma is used as a **tool-calling router** to choose and execute PC-AI tools,
then pcai-inference produces the final narrative response.

```powershell
# Run rust-functiongemma runtime (router)
.\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-runtime build
.\Deploy\rust-functiongemma-runtime\target\debug\rust-functiongemma-runtime.exe

# Route a request through FunctionGemma, then answer with the main LLM
Invoke-LLMChatRouted -Message "Check disk health and summarize issues." -Mode diagnose

# Or use routed chat (non-interactive)
Invoke-LLMChat -Message "Explain WSL vs Docker." -UseRouter -RouterMode chat
```

## Evaluation & Benchmarking

PC-AI ships an evaluation harness for LLM backends with structured progress logging and run outputs.

**Runner:** `Tests\Evaluation\Invoke-InferenceEvaluation.ps1`

**What it does:**

- Loads a dataset (built-in or custom JSON)
- Starts/initializes the requested backend
- Runs test cases and aggregates metrics
- Writes run outputs to `.pcai\evaluation\runs\<timestamp-label>\`

**Outputs per run:**

- `events.jsonl` (structured event stream for LLM agents)
- `progress.log` (progress snapshots)
- `summary.json` (aggregate results)

**Example (compiled llama.cpp backend):**

```powershell
pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin `
  -ModelPath "C:\Models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
  -Dataset diagnostic `
  -MaxTestCases 5 `
  -ProgressMode stream `
  -RunLabel local-smoke
```

**Stop a running evaluation:**
Create the `stop.signal` file in the run folder (shown in output), or call:

```powershell
Stop-EvaluationRun
```

### HVSocket / VSock Endpoints

`Config/llm-config.json` supports HVSocket aliases for local routing. Use the `hvsock://` scheme
to resolve endpoints through `Config/hvsock-proxy.conf` (if configured).
Primary aliases:

- `hvsock://pcai-inference` (8080)
- `hvsock://functiongemma` (8000)

### TUI Modes

`PcaiChatTui.exe` supports single-shot, multi-turn, streaming, and tool-routing modes:

```
PcaiChatTui.exe --provider pcai-inference --mode stream
PcaiChatTui.exe --provider pcai-inference --mode react --tools .\Config\pcai-tools.json
PcaiChatTui.exe --provider pcai-native --backend llamacpp --model-path C:\Models\model.gguf
```

### Recommended Models

| Model         | Size   | Best For                   |
| ------------- | ------ | -------------------------- |
| GGUF: Llama 3 | Varies | General analysis           |
| GGUF: Mistral | Varies | Fast general analysis      |
| GGUF: Phi     | Varies | Lightweight local analysis |
| GGUF: Gemma   | Varies | High-quality responses     |

### Router Inputs

- `DIAGNOSE.md` + `DIAGNOSE_LOGIC.md` define diagnostic routing behavior
- `CHAT.md` defines general chat behavior
- `Config/pcai-tools.json` defines tool schema and mappings

## Documentation Automation

PC-AI includes automated documentation generation tools:

```powershell
# Full documentation + training pipeline
.\Tools\Invoke-DocPipeline.ps1 -Mode Full

# Docs-only (PowerShell, Rust, C# summaries)
.\Tools\Invoke-DocPipeline.ps1 -Mode DocsOnly

# Lightweight auto-docs summary
.\Tools\generate-auto-docs.ps1 -BuildDocs
```

Outputs are written to `Reports/` (e.g. `AUTO_DOCS_SUMMARY.md`, `DOC_PIPELINE_REPORT.md`).

## Testing

```powershell
# Run all tests with coverage
.\tests\.pester.ps1 -Type All -Coverage

# Run unit tests only
.\tests\.pester.ps1 -Type Unit

# Run integration tests
.\tests\.pester.ps1 -Type Integration
```

## Project Structure

```
PC_AI/
├── Build.ps1                 # Unified build orchestrator
├── PC-AI.ps1                 # Unified CLI entry point
├── DIAGNOSE.md               # LLM system prompt
├── DIAGNOSE_LOGIC.md         # Decision tree for analysis
├── CHAT.md                   # Chat system prompt
├── Modules/
│   ├── PC-AI.Hardware/       # Hardware diagnostics
│   ├── PC-AI.Virtualization/ # WSL2, Hyper-V, Docker
│   ├── PC-AI.USB/            # USB management
│   ├── PC-AI.Network/        # Network diagnostics
│   ├── PC-AI.Performance/    # Performance optimization
│   ├── PC-AI.Cleanup/        # System cleanup
│   ├── PC-AI.LLM/            # LLM integration
│   └── PC-AI.Acceleration/   # Rust tools + parallelism
├── Native/
│   └── pcai_core/pcai_inference/ # Rust LLM inference engine (HTTP + FFI)
├── Deploy/
│   ├── rust-functiongemma-runtime/ # Rust router runtime (tool_calls)
│   ├── rust-functiongemma-train/   # Rust router dataset + training
│   ├── rust-functiongemma-core/    # Shared Rust library
│   └── rust-functiongemma/         # Top-level workspace
├── Tests/                    # Pester test suites
├── Config/                   # Configuration files
└── Reports/                  # Generated diagnostic reports
```

## Safety

- **Read-only by default**: Diagnostics collect data without modifications
- **Explicit consent**: Destructive operations require confirmation
- **Backup warnings**: BIOS/disk operations include backup prompts
- **Dry-run support**: Preview changes before execution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `.\tests\.pester.ps1 -Type All`
4. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Rust CLI tools](https://github.com/sharkdp/fd) for performance acceleration
- pcai-inference (Rust) for local LLM inference
- [Pester](https://pester.dev/) for PowerShell testing
