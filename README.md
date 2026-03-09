<p align="center">
  <h1 align="center">PC-AI</h1>
  <p align="center">
    Local LLM-powered PC diagnostics and optimization for Windows
  </p>
  <p align="center">
    <a href="https://github.com/David-Martel/PC-AI/actions/workflows/powershell-tests.yml"><img src="https://github.com/David-Martel/PC-AI/actions/workflows/powershell-tests.yml/badge.svg" alt="PowerShell Tests"></a>
    <a href="https://github.com/David-Martel/PC-AI/actions/workflows/rust-inference.yml"><img src="https://github.com/David-Martel/PC-AI/actions/workflows/rust-inference.yml/badge.svg" alt="Rust Build"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <img src="https://img.shields.io/badge/PowerShell-7.0%2B-blue?logo=powershell" alt="PowerShell 7+">
    <img src="https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6?logo=windows" alt="Windows">
  </p>
</p>

---

PC-AI is a modular PowerShell framework that diagnoses hardware issues, analyzes system health, and recommends optimizations -- all powered by local LLMs running on your own machine. No cloud APIs, no data leaves your PC.

## Key Features

- **Hardware Diagnostics** -- Device errors, SMART disk health, USB controllers, network adapters
- **Native Acceleration** -- Rust + C# hybrid engine delivering 5-40x speedups over pure PowerShell
- **Local LLM Analysis** -- AI-powered diagnostic interpretation via pcai-inference (llama.cpp / mistral.rs backends)
- **Tool-Calling Router** -- FunctionGemma selects and executes the right diagnostic tool before LLM analysis
- **Interactive TUI** -- Terminal chat interface with streaming, multi-turn, and ReAct tool-routing modes
- **Unified CLI** -- Single `PC-AI.ps1` entry point for all operations
- **Safety First** -- Read-only by default, explicit consent for any system modifications

## Architecture

```
                          PC-AI.ps1 (Unified CLI)
                                  |
              +-------------------+-------------------+
              |                   |                   |
      PowerShell Modules    FunctionGemma        pcai-inference
     (Hardware, USB, Net,   (Tool Router)        (LLM Engine)
      Perf, Cleanup, LLM)       |                    |
              |            Rust Runtime         llama.cpp / mistral.rs
              |            (axum, port 8000)    (HTTP + FFI, port 8080)
              |                   |                   |
              +-------------------+-------------------+
                                  |
                     Native Acceleration Layer
                   Rust (pcai_core_lib) + C# (PcaiNative)
                        via P/Invoke bridge
```

## Quick Start

### Install

```powershell
# Clone the repository
git clone https://github.com/David-Martel/PC-AI.git
cd PC-AI

# Run hardware diagnostics (requires Administrator)
.\PC-AI.ps1 diagnose hardware

# Check system health
.\PC-AI.ps1 doctor
```

### Build Native Components (Optional)

```powershell
# Build everything (Rust + C# + PowerShell validation)
.\Build.ps1

# Build with CUDA GPU acceleration
.\Build.ps1 -Component inference -EnableCuda

# Run lint and format checks
.\Build.ps1 -Component lint
```

### Use the CLI

```powershell
# Diagnostics
.\PC-AI.ps1 diagnose hardware    # Device errors, disk health, USB, network
.\PC-AI.ps1 diagnose all         # Full system scan

# Optimization
.\PC-AI.ps1 optimize disk        # TRIM/defrag recommendations

# Cleanup
.\PC-AI.ps1 cleanup path --dry-run   # Preview PATH cleanup
.\PC-AI.ps1 cleanup temp             # Clean temp files

# LLM Analysis (requires pcai-inference or Ollama)
.\PC-AI.ps1 analyze                   # AI-powered diagnostic interpretation
.\PC-AI.ps1 analyze --model mistral   # Use specific model
```

## Modules

| Module | Purpose |
|--------|---------|
| **PC-AI.Hardware** | Device manager errors, SMART status, USB, network adapters |
| **PC-AI.Acceleration** | Rust/C# native bridge + CLI tool integration (fd, ripgrep) |
| **PC-AI.Performance** | Resource monitoring, memory pressure, optimization planning |
| **PC-AI.Cleanup** | PATH deduplication, temp cleanup, duplicate file detection |
| **PC-AI.LLM** | pcai-inference + FunctionGemma integration for AI analysis |
| **PC-AI.Network** | Network diagnostics, adapter configuration |
| **PC-AI.USB** | USB device management and status |
| **PC-AI.Virtualization** | Hyper-V / WSL2 diagnostics (optional) |

## Native Acceleration

PC-AI includes a high-performance native layer for compute-intensive operations:

| Operation | Native (Rust+C#) | PowerShell | Speedup |
|-----------|-------------------|------------|---------|
| Directory manifest | 29 ms | 1,162 ms | **40x** |
| File search | 104 ms | 3,436 ms | **33x** |
| Content search (ripgrep) | 12 ms | 4,735 ms | **395x** |
| Token estimation | 2.9x faster | baseline | **2.9x** |
| Duplicate detection | parallel SHA-256 | sequential | **5-10x** |

The acceleration layer follows a tiered fallback strategy:
1. **Native DLL** (Rust via C# P/Invoke) -- fastest
2. **Rust CLI tools** (fd, ripgrep, procs) -- if DLLs unavailable
3. **PS7+ parallel** (ForEach-Object -Parallel) -- no external deps
4. **Sequential PowerShell** -- universal compatibility

## LLM Integration

PC-AI works with local LLM providers. No cloud APIs required.

### Supported Backends

| Backend | Type | GPU Support | Notes |
|---------|------|-------------|-------|
| **pcai-inference** | Native (Rust) | CUDA | Built-in, HTTP + FFI modes |
| **Ollama** | External | CUDA/ROCm | Drop-in, auto-detected |
| **LM Studio** | External | CUDA | OpenAI-compatible API |
| **vLLM** | External (Docker) | CUDA | High-throughput serving |

### FunctionGemma Router

FunctionGemma acts as a tool-calling router: it analyzes user requests, selects appropriate PC-AI diagnostic tools, executes them, then passes structured results to the main LLM for interpretation.

```powershell
# Routed diagnosis: FunctionGemma picks tools, LLM interprets results
Invoke-LLMChatRouted -Message "Check disk health and summarize issues." -Mode diagnose

# Direct LLM chat (no tool routing)
Invoke-LLMChat -Message "Explain WSL vs Docker." -Mode chat
```

### Interactive TUI

```powershell
# Streaming chat
PcaiChatTui.exe --provider pcai-inference --mode stream

# ReAct tool-routing mode
PcaiChatTui.exe --provider pcai-inference --mode react --tools Config\pcai-tools.json
```

## Evaluation and Benchmarking

```powershell
# Benchmark LLM backends
pwsh Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin -Dataset diagnostic -MaxTestCases 5

# Benchmark the native tooling stack
pwsh Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -Suite quick
```

## Build System

The unified `Build.ps1` orchestrator supports 20+ components:

```powershell
.\Build.ps1 -Component inference -EnableCuda   # LLM backends with GPU
.\Build.ps1 -Component functiongemma            # Tool router
.\Build.ps1 -Component native                   # All .NET components
.\Build.ps1 -Component lint -LintProfile all    # Multi-language linting
.\Build.ps1 -Component fix -AutoFix             # Auto-fix lint issues
.\Build.ps1 -Clean -Package -EnableCuda         # Release build
```

**CUDA targets**: SM 75 (Turing), SM 80/86 (Ampere), SM 89 (Ada Lovelace), SM 120 (Blackwell)

## Testing

```powershell
# Run the full test suite
pwsh Tests\Invoke-AllTests.ps1 -Suite All

# Rust unit tests
cargo test --manifest-path Native\pcai_core\Cargo.toml
```

| Suite | Tests | Status |
|-------|-------|--------|
| Rust (pcai_core + inference) | 197 | Passing |
| Pester (PowerShell) | 37 | Passing |
| Module validation | 199 | Passing |
| **Total** | **433** | **All passing** |

## Project Structure

```
PC-AI/
+-- PC-AI.ps1                    # Unified CLI entry point
+-- Build.ps1                    # Build orchestrator (20+ components)
+-- Modules/                     # PowerShell diagnostic modules
|   +-- PC-AI.Hardware/          #   Device, disk, USB, network
|   +-- PC-AI.Acceleration/      #   Native FFI + CLI tool wrappers
|   +-- PC-AI.Performance/       #   Resource monitoring + optimizer
|   +-- PC-AI.LLM/               #   LLM integration layer
|   +-- PC-AI.Cleanup/           #   PATH, temp, duplicate cleanup
|   +-- ...                      #   (8 modules total)
+-- Native/
|   +-- pcai_core/               # Rust workspace (6 crates)
|   |   +-- pcai_inference/      #   LLM engine (llama.cpp + mistral.rs)
|   |   +-- pcai_core_lib/       #   Shared lib (search, perf, telemetry)
|   |   +-- pcai_media/          #   Media processing (Janus-Pro)
|   |   +-- pcai_ollama_rs/      #   Ollama benchmarking
|   +-- PcaiNative/              # C# P/Invoke bridge (.NET 8)
|   +-- PcaiChatTui/             # Interactive chat TUI
+-- Deploy/
|   +-- rust-functiongemma*/     # FunctionGemma router (3 crates)
|   +-- rag-database/            # RAG pipeline (PostgreSQL + pgvector)
+-- Config/                      # Runtime configuration
+-- Tests/                       # Pester + Rust test suites
+-- Tools/                       # 33 utility scripts
```

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Windows | 10 / 11 | Primary platform |
| PowerShell | 7.0+ | Core runtime |
| Pester | 5.0+ | Testing (optional) |
| Rust | Latest stable | Native components (optional) |
| .NET SDK | 8.0 | C# interop (optional) |
| CUDA Toolkit | 12.x+ | GPU acceleration (optional) |

No WSL or Docker required for core functionality.

## Safety

- **Read-only by default** -- diagnostics collect data without modifications
- **Explicit consent** -- destructive operations require confirmation
- **Backup prompts** -- disk repair and BIOS operations warn first
- **Dry-run support** -- preview changes before execution (`--dry-run`)
- **Professional escalation** -- recommends expert help for suspected hardware failure

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

[MIT License](LICENSE) -- Copyright (c) 2025 David Martel
