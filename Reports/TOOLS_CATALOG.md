# Tools Catalog

Generated: 2026-03-01 23:33:13

| Script | Synopsis |
|--------|----------|
| Export-OllamaModels.ps1 | Export Ollama model blobs as hardlinked GGUF files into the repo. |
| generate-api-signature-report.ps1 | Generates API signature alignment reports for PowerShell, C#, and Rust. |
| generate-auto-docs.ps1 | Unified auto-documentation generator for PC_AI (PowerShell, C#, Rust, ast-grep). |
| generate-functiongemma-tool-docs.ps1 | Generate FunctionGemma tool documentation from pcai-tools.json. |
| Generate-HelpGapsPriority.ps1 | Generates a prioritized help documentation gaps report |
| generate-tools-catalog.ps1 | Generate a catalog of PowerShell helper scripts under Tools/. |
| Get-BuildVersion.ps1 | Generate build version information from git metadata. |
| Initialize-CacheEnvironment.ps1 | Configure build caches (sccache/ccache) for Rust and C/C++ builds. |
| Initialize-CmakeEnvironment.ps1 | Normalizes CMake environment variables for the current session. |
| Initialize-CudaEnvironment.ps1 | Initializes CUDA environment variables for the current PowerShell session. |
| Invoke-DocPipeline.ps1 | Unified documentation generation and FunctionGemma training data pipeline. |
| Invoke-FunctionGemmaTrain.ps1 | Run Rust FunctionGemma LoRA fine-tuning with sensible defaults. |
| Invoke-ModelDiscovery.ps1 | Discover local LLM model files and generate MODELS.md. |
| Invoke-RustBuild.ps1 | Rust build helper that routes through CargoTools. |
| Link-ModelInventory.ps1 | Create links/junctions under Models/ for discovered model files and folders. |
| llm-router.ps1 | Lightweight Ollama-compatible router with LM Studio fallback. |
| llm-validate.ps1 | Validates PC_AI LLM flows using DIAGNOSE.md + DIAGNOSE_LOGIC.md system prompts. |
| New-PcaiPowerShellRelease.ps1 |  |
| normalize-help-blocks.ps1 | Normalize comment-based help blocks for public functions. |
| prepare-functiongemma-router-data.ps1 | Build FunctionGemma router datasets using the Rust pipeline. |
| prepare-functiongemma-token-cache.ps1 | Build token cache for FunctionGemma training (Rust). |
| run-functiongemma-eval.ps1 | Runs a FunctionGemma evaluation pass via Build.ps1 and writes a metrics report. |
| run-functiongemma-tests.ps1 | Runs FunctionGemma fine-tuning test suite and tool coverage reports. |
| run-psscriptanalyzer.ps1 | Run PSScriptAnalyzer and export results to Reports\PSSCRIPTANALYZER.json/.md |
| Set-CudaBuildEnv.ps1 | Sets CUDA build environment for Rust/Candle compilation. |
| Set-CudaEnvironment.ps1 | Sets CUDA environment variables for Rust/Candle compilation. |
| Test-PcaiReleaseModule.ps1 |  |
| update-doc-status.ps1 | Generate documentation/status reports using ast-grep (sg) with rg fallback. |
| update-help-parameters.ps1 | Auto-fills missing .PARAMETER blocks in PowerShell help comments. |
| update-tool-coverage.ps1 | Analyze tool schema coverage against PC_AI tool implementations. |
| validate-doc-accuracy.ps1 |  |

## Details

### Export-OllamaModels.ps1
Path: `C:\codedev\pc-ai\Tools\Export-OllamaModels.ps1`
Synopsis: Export Ollama model blobs as hardlinked GGUF files into the repo.
Description: Reads Ollama manifests under %USERPROFILE%\.ollama\models\manifests and hardlinks the model blob (sha256-*) into a usable .gguf filename under the repo's Models\ollama directory.

### generate-api-signature-report.ps1
Path: `C:\codedev\pc-ai\Tools\generate-api-signature-report.ps1`
Synopsis: Generates API signature alignment reports for PowerShell, C#, and Rust.
Description: - Parses PowerShell public functions and compares parameters to help blocks - Compares C# DllImport declarations to Rust exported functions - Compares PowerShell wrapper calls to available C# methods Writes Reports\API_SIGNATURE_REPORT.json and Reports\API_SIGNATURE_REPORT.md

### generate-auto-docs.ps1
Path: `C:\codedev\pc-ai\Tools\generate-auto-docs.ps1`
Synopsis: Unified auto-documentation generator for PC_AI (PowerShell, C#, Rust, ast-grep).
Description: - Runs ast-grep-based doc status + tool coverage reports - Optionally runs global ast-grep rules from ~/.config/ast-grep - Builds PowerShell module command index - Optionally generates C# XML docs and Rust docs - Links outputs to PCAI_BUILD_VERSION (from Native\build.ps1)

### generate-functiongemma-tool-docs.ps1
Path: `C:\codedev\pc-ai\Tools\generate-functiongemma-tool-docs.ps1`
Synopsis: Generate FunctionGemma tool documentation from pcai-tools.json.
Description: Produces a markdown doc that lists tool names, descriptions, parameters, and the negative examples used for NO_TOOL routing.

### Generate-HelpGapsPriority.ps1
Path: `C:\codedev\pc-ai\Tools\Generate-HelpGapsPriority.ps1`
Synopsis: Generates a prioritized help documentation gaps report
Description: Analyzes API signature report to identify functions missing help documentation, organized by module with priority ordering

### generate-tools-catalog.ps1
Path: `C:\codedev\pc-ai\Tools\generate-tools-catalog.ps1`
Synopsis: Generate a catalog of PowerShell helper scripts under Tools/.
Description: Scans Tools/*.ps1 for comment-based help and emits a Markdown + JSON catalog with synopsis/description for LLM-friendly documentation.

### Get-BuildVersion.ps1
Path: `C:\codedev\pc-ai\Tools\Get-BuildVersion.ps1`
Synopsis: Generate build version information from git metadata.
Description: Extracts version information from git tags, commits, and timestamps for embedding into compiled binaries and build manifests. Version Format: {semver}-{commits}+{hash}.{timestamp} Example: 0.2.0-15+abc1234.20260201T143000Z

### Initialize-CacheEnvironment.ps1
Path: `C:\codedev\pc-ai\Tools\Initialize-CacheEnvironment.ps1`
Synopsis: Configure build caches (sccache/ccache) for Rust and C/C++ builds.

### Initialize-CmakeEnvironment.ps1
Path: `C:\codedev\pc-ai\Tools\Initialize-CmakeEnvironment.ps1`
Synopsis: Normalizes CMake environment variables for the current session.
Description: Ensures CMAKE_ROOT points to the installed CMake share directory that matches the active cmake.exe version. Also aligns CMAKE_PREFIX_PATH and CMAKE_PROGRAM when they are missing or stale. Intended for build/doc pipelines.

### Initialize-CudaEnvironment.ps1
Path: `C:\codedev\pc-ai\Tools\Initialize-CudaEnvironment.ps1`
Synopsis: Initializes CUDA environment variables for the current PowerShell session.
Description: Detects installed CUDA toolkits using a preferred version list and common environment variables, then sets CUDA_PATH/CUDA_HOME and updates PATH with CUDA bin and nvvm/bin. Intended for build and doc pipelines (non-destructive).

### Invoke-DocPipeline.ps1
Path: `C:\codedev\pc-ai\Tools\Invoke-DocPipeline.ps1`
Synopsis: Unified documentation generation and FunctionGemma training data pipeline.
Description: Master orchestrator that: 1. Generates documentation from code (Rust, PowerShell, C#) 2. Exports structured training data for FunctionGemma 3. Validates training data format 4. Updates Reports/ with current status

### Invoke-FunctionGemmaTrain.ps1
Path: `C:\codedev\pc-ai\Tools\Invoke-FunctionGemmaTrain.ps1`
Synopsis: Run Rust FunctionGemma LoRA fine-tuning with sensible defaults.
Description: Routes rust-functiongemma-train via Build.ps1 and prefers token caches and packed sequences for faster training.

### Invoke-ModelDiscovery.ps1
Path: `C:\codedev\pc-ai\Tools\Invoke-ModelDiscovery.ps1`
Synopsis: Discover local LLM model files and generate MODELS.md.
Description: Uses PC-AI search tooling (Find-FilesFast) to scan common locations and/or fixed drives for model files (GGUF, SafeTensors, etc). Produces a markdown inventory suitable for LLM agents.

### Invoke-RustBuild.ps1
Path: `C:\codedev\pc-ai\Tools\Invoke-RustBuild.ps1`
Synopsis: Rust build helper that routes through CargoTools.
Description: Standardizes Rust builds with CargoTools env setup, sccache, and optional lld-link configuration. Intended for repeatable, LLM-friendly builds.

### Link-ModelInventory.ps1
Path: `C:\codedev\pc-ai\Tools\Link-ModelInventory.ps1`
Synopsis: Create links/junctions under Models/ for discovered model files and folders.
Description: Reads MODELS.md (from Invoke-ModelDiscovery) and creates hardlinks or symlinks to model files under Models\linked. Also creates junctions for HF-style model directories (detected via config.json).

### llm-router.ps1
Path: `C:\codedev\pc-ai\Tools\llm-router.ps1`
Synopsis: Lightweight Ollama-compatible router with LM Studio fallback.
Description: Listens on a local port and forwards Ollama API requests to Ollama when available. If Ollama is down, it converts requests to LM Studio's OpenAI-compatible API.

### llm-validate.ps1
Path: `C:\codedev\pc-ai\Tools\llm-validate.ps1`
Synopsis: Validates PC_AI LLM flows using DIAGNOSE.md + DIAGNOSE_LOGIC.md system prompts.
Description: Runs Invoke-PCDiagnosis with a small synthetic report and Invoke-SmartDiagnosis against a target path. Fails fast if Ollama/Router is not reachable.

### New-PcaiPowerShellRelease.ps1
Path: `C:\codedev\pc-ai\Tools\New-PcaiPowerShellRelease.ps1`

### normalize-help-blocks.ps1
Path: `C:\codedev\pc-ai\Tools\normalize-help-blocks.ps1`
Synopsis: Normalize comment-based help blocks for public functions.
Description: Ensures each top-level public function has a single well-formed help block. Preserves existing non-auto-generated help and inserts missing .PARAMETER entries. Rebuilds malformed or auto-generated blocks.

### prepare-functiongemma-router-data.ps1
Path: `C:\codedev\pc-ai\Tools\prepare-functiongemma-router-data.ps1`
Synopsis: Build FunctionGemma router datasets using the Rust pipeline.
Description: Routes rust-functiongemma-train prepare-router through Build.ps1 so this workflow remains inside the unified build orchestration layer while matching the Python I/O contract (tool_calls or NO_TOOL). Optional: Use PcaiNative.dll to run the same dataset generation via native FFI.

### prepare-functiongemma-token-cache.ps1
Path: `C:\codedev\pc-ai\Tools\prepare-functiongemma-token-cache.ps1`
Synopsis: Build token cache for FunctionGemma training (Rust).
Description: Routes rust-functiongemma-train prepare-cache through Build.ps1 to pre-tokenize JSONL datasets under the unified build workflow.

### run-functiongemma-eval.ps1
Path: `C:\codedev\pc-ai\Tools\run-functiongemma-eval.ps1`
Synopsis: Runs a FunctionGemma evaluation pass via Build.ps1 and writes a metrics report.

### run-functiongemma-tests.ps1
Path: `C:\codedev\pc-ai\Tools\run-functiongemma-tests.ps1`
Synopsis: Runs FunctionGemma fine-tuning test suite and tool coverage reports.

### run-psscriptanalyzer.ps1
Path: `C:\codedev\pc-ai\Tools\run-psscriptanalyzer.ps1`
Synopsis: Run PSScriptAnalyzer and export results to Reports\PSSCRIPTANALYZER.json/.md

### Set-CudaBuildEnv.ps1
Path: `C:\codedev\pc-ai\Tools\Set-CudaBuildEnv.ps1`
Synopsis: Sets CUDA build environment for Rust/Candle compilation.
Description: Prefers CUDA 13.1 by default for current pc-ai CUDA builds, auto-detects the newest MSVC x64 toolchain, and updates process-scoped environment variables for immediate build use.

### Set-CudaEnvironment.ps1
Path: `C:\codedev\pc-ai\Tools\Set-CudaEnvironment.ps1`
Synopsis: Sets CUDA environment variables for Rust/Candle compilation.
Description: Configures CUDA_PATH plus PATH/include/lib entries. Defaults to process scope for safety; use -Scope Machine for persistent system updates.

### Test-PcaiReleaseModule.ps1
Path: `C:\codedev\pc-ai\Tools\Test-PcaiReleaseModule.ps1`

### update-doc-status.ps1
Path: `C:\codedev\pc-ai\Tools\update-doc-status.ps1`
Synopsis: Generate documentation/status reports using ast-grep (sg) with rg fallback.
Description: Scans the repo for TODO/FIXME/INCOMPLETE/@status/DEPRECATED markers and writes: - Reports\DOC_STATUS.json (raw sg json when available) - Reports\DOC_STATUS.md (human summary + matches)

### update-help-parameters.ps1
Path: `C:\codedev\pc-ai\Tools\update-help-parameters.ps1`
Synopsis: Auto-fills missing .PARAMETER blocks in PowerShell help comments.
Description: Scans Public functions in Modules, compares parameter lists, and inserts missing .PARAMETER sections into the nearest comment-based help block. If no help block exists, generates a minimal help block above the function.

### update-tool-coverage.ps1
Path: `C:\codedev\pc-ai\Tools\update-tool-coverage.ps1`
Synopsis: Analyze tool schema coverage against PC_AI tool implementations.
Description: Loads Config\pcai-tools.json and compares tool names with the tool-mapping in Invoke-FunctionGemmaReAct.ps1. Uses ast-grep (sg) if available, with rg fallback. Writes Reports\TOOL_SCHEMA_REPORT.json and Reports\TOOL_SCHEMA_REPORT.md.

### validate-doc-accuracy.ps1
Path: `C:\codedev\pc-ai\Tools\validate-doc-accuracy.ps1`

