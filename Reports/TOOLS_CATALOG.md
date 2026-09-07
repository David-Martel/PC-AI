# Tools Catalog

Generated: 2026-09-07 13:10:09

| Script | Synopsis |
|--------|----------|
| Apply-ProcessLassoUiSyncTuning.ps1 |  |
| bench-media-pipeline.ps1 |  |
| Bootstrap-ThunderboltPeerRemote.ps1 |  |
| Collect-BootDiagnostics.ps1 |  |
| Collect-DetailedProcessData.ps1 | Collects detailed process data focused on LLM agent terminals and memory hogs. |
| Collect-DrivePerformanceSyncRisk.ps1 | Collects drive-performance and cloud-sync risk evidence. |
| Collect-SystemPerformanceData.ps1 | Collects comprehensive system performance data for RAM optimization analysis. |
| Collect-UiGlitchDiagnostics.ps1 | Collects UI responsiveness, OneDrive, Process Lasso, and filter-driver evidence. |
| Ensure-ProcessLassoGovernor.ps1 | Ensures the Process Lasso governor is running. |
| Export-OllamaModels.ps1 |  |
| Find-ThunderboltPeer.ps1 |  |
| generate-api-signature-report.ps1 |  |
| generate-auto-docs.ps1 |  |
| generate-functiongemma-tool-docs.ps1 |  |
| Generate-HelpGapsPriority.ps1 | Generates a prioritized help documentation gaps report |
| generate-tools-catalog.ps1 |  |
| Get-BuildVersion.ps1 |  |
| Get-JanusProModel.ps1 |  |
| Get-JulesPRStatus.ps1 |  |
| Get-PcaiDataServicesStatus.ps1 |  |
| Get-PcaiModuleStatus.ps1 |  |
| Get-RustAnalyzerStatus.ps1 | Checks rust-analyzer and lspmux status, configuration, and all LSP clients. |
| Initialize-CacheEnvironment.ps1 |  |
| Initialize-CmakeEnvironment.ps1 |  |
| Initialize-CudaEnvironment.ps1 |  |
| Initialize-ThunderboltLink.ps1 |  |
| Install-InfDriver.ps1 |  |
| Install-LlvmFromSource.ps1 |  |
| Install-PcaiDevModules.ps1 |  |
| Install-PcaiRedisCache.ps1 |  |
| Invoke-AstGrepAutoFix.ps1 |  |
| Invoke-DocPipeline.ps1 |  |
| Invoke-FunctionGemmaEval.ps1 | Router eval harness for FunctionGemma against a running runtime instance. |
| Invoke-FunctionGemmaTrain.ps1 |  |
| Invoke-JanusGpuSmoke.ps1 |  |
| Invoke-JulesBatchReview.ps1 |  |
| Invoke-JulesOrchestrator.ps1 |  |
| Invoke-JulesSession.ps1 |  |
| Invoke-LlmPerfBenchmark.ps1 |  |
| Invoke-LocalLLMReview.ps1 |  |
| Invoke-ModelDiscovery.ps1 |  |
| Invoke-NetworkDiscovery.ps1 |  |
| Invoke-PcaiMappedTool.ps1 |  |
| Invoke-ProcessLassoAnalysis.ps1 |  |
| Invoke-RustBuild.ps1 |  |
| Invoke-RustProfile.ps1 |  |
| Invoke-ThunderboltNetworking.ps1 |  |
| Link-ModelInventory.ps1 |  |
| llm-router.ps1 | Lightweight Ollama-compatible router with LM Studio fallback. |
| llm-validate.ps1 | Validates PC_AI LLM flows using DIAGNOSE.md + DIAGNOSE_LOGIC.md system prompts. |
| Migrate-SystemScriptsIntoRepo.ps1 | Moves workstation system scripts into this repo and repoints scheduled tasks. |
| Mount-PersistentVHDX.ps1 |  |
| New-PcaiPowerShellRelease.ps1 |  |
| normalize-help-blocks.ps1 |  |
| Optimize-InferenceConfig.ps1 |  |
| PcaiModuleBootstrap.ps1 |  |
| prepare-functiongemma-router-data.ps1 |  |
| prepare-functiongemma-token-cache.ps1 |  |
| Register-PersistentVHDXTasks.ps1 |  |
| Register-ProcessLassoGovernorWatchdog.ps1 | Registers a Process Lasso governor watchdog scheduled task. |
| Repair-OneDriveSync.ps1 |  |
| run-functiongemma-eval.ps1 |  |
| run-functiongemma-tests.ps1 |  |
| Show-PerfSummary.ps1 | Displays summary of collected performance data. |
| Sync-NvidiaDriverVersion.ps1 |  |
| Sync-PowerShellModuleRelease.ps1 |  |
| Test-BootMountHealth.ps1 |  |
| test-janus-quick.ps1 |  |
| test-media-dll.ps1 |  |
| Test-Optimizer.ps1 | Tests the optimizer cmdlets (PowerShell fallback path). |
| Test-PcaiReleaseModule.ps1 |  |
| Test-ProcessLassoBootSafety.ps1 |  |
| Test-SyncProviderHealth.ps1 |  |
| update-doc-status.ps1 | Generate documentation/status reports using ast-grep (sg) with rg fallback. |
| Update-Drivers.ps1 |  |
| update-help-parameters.ps1 |  |
| Update-NvidiaSoftware.ps1 |  |
| update-tool-coverage.ps1 |  |
| Update-UsbDrivers.ps1 |  |
| validate-doc-accuracy.ps1 |  |

## Details

### Apply-ProcessLassoUiSyncTuning.ps1
Path: `C:\codedev\PC_AI\Tools\Apply-ProcessLassoUiSyncTuning.ps1`

### bench-media-pipeline.ps1
Path: `C:\codedev\PC_AI\Tools\bench-media-pipeline.ps1`

### Bootstrap-ThunderboltPeerRemote.ps1
Path: `C:\codedev\PC_AI\Tools\Bootstrap-ThunderboltPeerRemote.ps1`

### Collect-BootDiagnostics.ps1
Path: `C:\codedev\PC_AI\Tools\Collect-BootDiagnostics.ps1`

### Collect-DetailedProcessData.ps1
Path: `C:\codedev\PC_AI\Tools\Collect-DetailedProcessData.ps1`
Synopsis: Collects detailed process data focused on LLM agent terminals and memory hogs.

### Collect-DrivePerformanceSyncRisk.ps1
Path: `C:\codedev\PC_AI\Tools\Collect-DrivePerformanceSyncRisk.ps1`
Synopsis: Collects drive-performance and cloud-sync risk evidence.
Description: Captures read-only evidence for registry tuning, sync providers, Task
Scheduler, Process Lasso, Defender, Windows Search, filter drivers, disk state,
and recent OneDrive/storage events. Use this before and after registry or script
changes to validate the effect on OneDrive and UI responsiveness.

### Collect-SystemPerformanceData.ps1
Path: `C:\codedev\PC_AI\Tools\Collect-SystemPerformanceData.ps1`
Synopsis: Collects comprehensive system performance data for RAM optimization analysis.
Description: Gathers memory usage, process details, GPU info, page file config, performance
counters, and terminal/LLM process analysis. Outputs structured JSON report.

### Collect-UiGlitchDiagnostics.ps1
Path: `C:\codedev\PC_AI\Tools\Collect-UiGlitchDiagnostics.ps1`
Synopsis: Collects UI responsiveness, OneDrive, Process Lasso, and filter-driver evidence.
Description: Writes a timestamped diagnostic bundle under Reports\ui-glitch-diagnostics by
capturing process state, Process Lasso logs, OneDrive sync diagnostics, filter
state, and recent event-log evidence.

### Ensure-ProcessLassoGovernor.ps1
Path: `C:\codedev\PC_AI\Tools\Ensure-ProcessLassoGovernor.ps1`
Synopsis: Ensures the Process Lasso governor is running.
Description: Checks for ProcessGovernor.exe and starts it when missing. The script writes a
structured result object, optional JSON report, and Windows Application events
when a restart or failure occurs. Dry-run mode performs all checks without
starting processes or writing event-log/report output.

### Export-OllamaModels.ps1
Path: `C:\codedev\PC_AI\Tools\Export-OllamaModels.ps1`

### Find-ThunderboltPeer.ps1
Path: `C:\codedev\PC_AI\Tools\Find-ThunderboltPeer.ps1`

### generate-api-signature-report.ps1
Path: `C:\codedev\PC_AI\Tools\generate-api-signature-report.ps1`

### generate-auto-docs.ps1
Path: `C:\codedev\PC_AI\Tools\generate-auto-docs.ps1`

### generate-functiongemma-tool-docs.ps1
Path: `C:\codedev\PC_AI\Tools\generate-functiongemma-tool-docs.ps1`

### Generate-HelpGapsPriority.ps1
Path: `C:\codedev\PC_AI\Tools\Generate-HelpGapsPriority.ps1`
Synopsis: Generates a prioritized help documentation gaps report
Description: Analyzes API signature report to identify functions missing help documentation,
organized by module with priority ordering

### generate-tools-catalog.ps1
Path: `C:\codedev\PC_AI\Tools\generate-tools-catalog.ps1`

### Get-BuildVersion.ps1
Path: `C:\codedev\PC_AI\Tools\Get-BuildVersion.ps1`

### Get-JanusProModel.ps1
Path: `C:\codedev\PC_AI\Tools\Get-JanusProModel.ps1`

### Get-JulesPRStatus.ps1
Path: `C:\codedev\PC_AI\Tools\Get-JulesPRStatus.ps1`

### Get-PcaiDataServicesStatus.ps1
Path: `C:\codedev\PC_AI\Tools\Get-PcaiDataServicesStatus.ps1`

### Get-PcaiModuleStatus.ps1
Path: `C:\codedev\PC_AI\Tools\Get-PcaiModuleStatus.ps1`

### Get-RustAnalyzerStatus.ps1
Path: `C:\codedev\PC_AI\Tools\Get-RustAnalyzerStatus.ps1`
Synopsis: Checks rust-analyzer and lspmux status, configuration, and all LSP clients.

### Initialize-CacheEnvironment.ps1
Path: `C:\codedev\PC_AI\Tools\Initialize-CacheEnvironment.ps1`

### Initialize-CmakeEnvironment.ps1
Path: `C:\codedev\PC_AI\Tools\Initialize-CmakeEnvironment.ps1`

### Initialize-CudaEnvironment.ps1
Path: `C:\codedev\PC_AI\Tools\Initialize-CudaEnvironment.ps1`

### Initialize-ThunderboltLink.ps1
Path: `C:\codedev\PC_AI\Tools\Initialize-ThunderboltLink.ps1`

### Install-InfDriver.ps1
Path: `C:\codedev\PC_AI\Tools\Install-InfDriver.ps1`

### Install-LlvmFromSource.ps1
Path: `C:\codedev\PC_AI\Tools\Install-LlvmFromSource.ps1`

### Install-PcaiDevModules.ps1
Path: `C:\codedev\PC_AI\Tools\Install-PcaiDevModules.ps1`

### Install-PcaiRedisCache.ps1
Path: `C:\codedev\PC_AI\Tools\Install-PcaiRedisCache.ps1`

### Invoke-AstGrepAutoFix.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-AstGrepAutoFix.ps1`

### Invoke-DocPipeline.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-DocPipeline.ps1`

### Invoke-FunctionGemmaEval.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-FunctionGemmaEval.ps1`
Synopsis: Router eval harness for FunctionGemma against a running runtime instance.
Description: Sends test scenarios from scenarios.json to the FunctionGemma runtime's
OpenAI-compatible /v1/chat/completions endpoint and evaluates response
accuracy across three dimensions:

- Tool call accuracy: does the response contain the correct tool name?
- Argument accuracy: do the returned arguments match expected values?
- NO_TOOL accuracy: does the model correctly return NO_TOOL for chat inputs?

Reports per-category accuracy, overall accuracy, and latency statistics.

This script complements Tools/run-functiongemma-eval.ps1, which runs the
Rust-native eval binary. This harness instead exercises the live HTTP API,
making it suitable for integration testing and runtime regression checks.

### Invoke-FunctionGemmaTrain.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-FunctionGemmaTrain.ps1`

### Invoke-JanusGpuSmoke.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-JanusGpuSmoke.ps1`

### Invoke-JulesBatchReview.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-JulesBatchReview.ps1`

### Invoke-JulesOrchestrator.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-JulesOrchestrator.ps1`

### Invoke-JulesSession.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-JulesSession.ps1`

### Invoke-LlmPerfBenchmark.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-LlmPerfBenchmark.ps1`

### Invoke-LocalLLMReview.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-LocalLLMReview.ps1`

### Invoke-ModelDiscovery.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-ModelDiscovery.ps1`

### Invoke-NetworkDiscovery.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-NetworkDiscovery.ps1`

### Invoke-PcaiMappedTool.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-PcaiMappedTool.ps1`

### Invoke-ProcessLassoAnalysis.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-ProcessLassoAnalysis.ps1`

### Invoke-RustBuild.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-RustBuild.ps1`

### Invoke-RustProfile.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-RustProfile.ps1`

### Invoke-ThunderboltNetworking.ps1
Path: `C:\codedev\PC_AI\Tools\Invoke-ThunderboltNetworking.ps1`

### Link-ModelInventory.ps1
Path: `C:\codedev\PC_AI\Tools\Link-ModelInventory.ps1`

### llm-router.ps1
Path: `C:\codedev\PC_AI\Tools\llm-router.ps1`
Synopsis: Lightweight Ollama-compatible router with LM Studio fallback.
Description: Listens on a local port and forwards Ollama API requests to Ollama when available.
If Ollama is down, it converts requests to LM Studio's OpenAI-compatible API.

### llm-validate.ps1
Path: `C:\codedev\PC_AI\Tools\llm-validate.ps1`
Synopsis: Validates PC_AI LLM flows using DIAGNOSE.md + DIAGNOSE_LOGIC.md system prompts.
Description: Runs Invoke-PCDiagnosis with a small synthetic report and Invoke-SmartDiagnosis
against a target path. Fails fast if Ollama/Router is not reachable.

### Migrate-SystemScriptsIntoRepo.ps1
Path: `C:\codedev\PC_AI\Tools\Migrate-SystemScriptsIntoRepo.ps1`
Synopsis: Moves workstation system scripts into this repo and repoints scheduled tasks.
Description: Centralizes PowerShell and command scripts that are used by Task Scheduler or
that can modify workstation startup, network, sync, WSL/Docker, RAG Redis, or
developer-tool state. The script preserves source provenance under
Tools\SystemScripts and can run in dry-run mode before any move.

### Mount-PersistentVHDX.ps1
Path: `C:\codedev\PC_AI\Tools\Mount-PersistentVHDX.ps1`

### New-PcaiPowerShellRelease.ps1
Path: `C:\codedev\PC_AI\Tools\New-PcaiPowerShellRelease.ps1`

### normalize-help-blocks.ps1
Path: `C:\codedev\PC_AI\Tools\normalize-help-blocks.ps1`

### Optimize-InferenceConfig.ps1
Path: `C:\codedev\PC_AI\Tools\Optimize-InferenceConfig.ps1`

### PcaiModuleBootstrap.ps1
Path: `C:\codedev\PC_AI\Tools\PcaiModuleBootstrap.ps1`

### prepare-functiongemma-router-data.ps1
Path: `C:\codedev\PC_AI\Tools\prepare-functiongemma-router-data.ps1`

### prepare-functiongemma-token-cache.ps1
Path: `C:\codedev\PC_AI\Tools\prepare-functiongemma-token-cache.ps1`

### Register-PersistentVHDXTasks.ps1
Path: `C:\codedev\PC_AI\Tools\Register-PersistentVHDXTasks.ps1`

### Register-ProcessLassoGovernorWatchdog.ps1
Path: `C:\codedev\PC_AI\Tools\Register-ProcessLassoGovernorWatchdog.ps1`
Synopsis: Registers a Process Lasso governor watchdog scheduled task.
Description: Creates or updates a delayed logon scheduled task that runs
Ensure-ProcessLassoGovernor.ps1. The watchdog checks whether
ProcessGovernor.exe is running, starts it when missing, and writes loud
Application event-log entries for remediation/failure cases.

### Repair-OneDriveSync.ps1
Path: `C:\codedev\PC_AI\Tools\Repair-OneDriveSync.ps1`

### run-functiongemma-eval.ps1
Path: `C:\codedev\PC_AI\Tools\run-functiongemma-eval.ps1`

### run-functiongemma-tests.ps1
Path: `C:\codedev\PC_AI\Tools\run-functiongemma-tests.ps1`

### Show-PerfSummary.ps1
Path: `C:\codedev\PC_AI\Tools\Show-PerfSummary.ps1`
Synopsis: Displays summary of collected performance data.

### Sync-NvidiaDriverVersion.ps1
Path: `C:\codedev\PC_AI\Tools\Sync-NvidiaDriverVersion.ps1`

### Sync-PowerShellModuleRelease.ps1
Path: `C:\codedev\PC_AI\Tools\Sync-PowerShellModuleRelease.ps1`

### Test-BootMountHealth.ps1
Path: `C:\codedev\PC_AI\Tools\Test-BootMountHealth.ps1`

### test-janus-quick.ps1
Path: `C:\codedev\PC_AI\Tools\test-janus-quick.ps1`

### test-media-dll.ps1
Path: `C:\codedev\PC_AI\Tools\test-media-dll.ps1`

### Test-Optimizer.ps1
Path: `C:\codedev\PC_AI\Tools\Test-Optimizer.ps1`
Synopsis: Tests the optimizer cmdlets (PowerShell fallback path).

### Test-PcaiReleaseModule.ps1
Path: `C:\codedev\PC_AI\Tools\Test-PcaiReleaseModule.ps1`

### Test-ProcessLassoBootSafety.ps1
Path: `C:\codedev\PC_AI\Tools\Test-ProcessLassoBootSafety.ps1`

### Test-SyncProviderHealth.ps1
Path: `C:\codedev\PC_AI\Tools\Test-SyncProviderHealth.ps1`

### update-doc-status.ps1
Path: `C:\codedev\PC_AI\Tools\update-doc-status.ps1`
Synopsis: Generate documentation/status reports using ast-grep (sg) with rg fallback.
Description: Scans the repo for TODO/FIXME/INCOMPLETE/@status/DEPRECATED markers and writes:
- Reports\DOC_STATUS.json (raw sg json when available)
- Reports\DOC_STATUS.md (human summary + matches)

### Update-Drivers.ps1
Path: `C:\codedev\PC_AI\Tools\Update-Drivers.ps1`

### update-help-parameters.ps1
Path: `C:\codedev\PC_AI\Tools\update-help-parameters.ps1`

### Update-NvidiaSoftware.ps1
Path: `C:\codedev\PC_AI\Tools\Update-NvidiaSoftware.ps1`

### update-tool-coverage.ps1
Path: `C:\codedev\PC_AI\Tools\update-tool-coverage.ps1`

### Update-UsbDrivers.ps1
Path: `C:\codedev\PC_AI\Tools\Update-UsbDrivers.ps1`

### validate-doc-accuracy.ps1
Path: `C:\codedev\PC_AI\Tools\validate-doc-accuracy.ps1`

