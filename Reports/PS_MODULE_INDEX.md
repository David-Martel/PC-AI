# PS_MODULE_INDEX

Generated: 2026-09-07 13:39:33

## PC-AI.Acceleration.psm1
- Compare-ToolPerformance: Compares performance between Rust tools and PowerShell equivalents
- Find-DuplicatesFast: Fast duplicate file detection using parallel hashing and fd
- Find-FilesFast: Fast file finding using fd
- Get-DiskUsageFast: Fast disk usage analysis using dust or parallel enumeration
- Get-FileHashParallel: Computes file hashes in parallel using a compiled .NET helper
- Get-PcaiCapabilities: Returns a capability registry for native and LLM components.
- Get-ProcessesFast: Fast process listing using procs
- Get-ProcessLassoSnapshot
- Get-RustToolStatus: Reports the status of available Rust acceleration tools
- Get-UnifiedHardwareReportJson: Returns a unified hardware report combining WMI and native diagnostics
- Measure-CommandPerformance: Benchmarks command performance using hyperfine or native measurement
- New-ProcessLassoOverlay
- Search-ContentFast: Fast content search using ripgrep with parallel fallback
- Search-LogsFast: Fast log file searching using ripgrep

## PC-AI.Cleanup.psm1
- Clear-TempFiles: Safely cleans temporary files from common system locations.
- Format-DuplicateReport: Finds duplicate files by content hash in specified directory.
- Get-PathDuplicates: Analyzes PATH environment variable for duplicates and non-existent entries.
- Invoke-NukeNulCleanup: Runs the NukeNul reserved filename cleanup tool and returns JSON results.
- Optimize-PathCompression: Compresses and optimizes the PATH environment variable below the Windows GUI limit.
- Repair-MachinePath: Repairs PATH environment variable by removing duplicates and invalid entries.

## PC-AI.CLI.psm1
- Get-PCCommandList
- Get-PCCommandMap
- Get-PCCommandModules
- Get-PCCommandSummary
- Get-PCModuleHelpEntry
- Get-PCModuleHelpIndex
- ConvertTo-PCArgumentMap
- Resolve-PCArguments

## PC-AI.Drivers.psm1
- Compare-DriverVersion: Compares installed driver versions against registry target versions and classifies
    each device.
- Connect-ThunderboltPeer: Connects to a Windows peer over a Thunderbolt / USB4 networking link.
- Find-ThunderboltPeer
- resolves: Loads the PC-AI driver registry JSON file and returns it as a structured object.
- Get-DriverReport: Generates a consolidated driver status report for all matched PnP devices.
- Get-NetworkDiscoverySnapshot
- Get-PnpDeviceInventory: Discovers all PnP devices and extracts structured driver metadata.
- focuses: Discovers Thunderbolt / USB4 networking adapters and reports link state, IP data,
    peer candidates, and tuning hints.
- Install-DriverUpdate: Downloads and installs a driver update for a registry-tracked device.
- is: Builds or applies a conservative optimization plan for a Thunderbolt / USB4 peer link.
- Update-DriverRegistry: Adds or updates a device entry in the driver registry JSON.

## PC-AI.Evaluation.psm1
- Add-ABTestResult: Adds a result to an A/B test
- Compare-ResponsePair: Compares two responses using LLM-as-judge pairwise comparison
- Compare-ResponseSimilarity: Compares semantic similarity between response and expected output
- Measure-DiagnosticQuality: Evaluates diagnostic output quality specific to PC-AI use case
- Export-EvaluationDataset
- Get-ABTestAnalysis: Performs statistical analysis on A/B test results
- Get-EvaluationDataset: Gets a built-in or custom evaluation dataset
- Get-EvaluationResults
- Get-EvaluationRunState
- Get-PcaiArtifactsRoot
- Get-PcaiCompiledBinaryPath
- Get-PcaiProjectRoot
- Get-RegressionReport
- Import-EvaluationDataset
- Initialize-EvaluationPaths
- Invoke-EvaluationSuite: Runs the evaluation suite against specified backend
- Invoke-LLMJudge: Uses an LLM to judge response quality
- Measure-Coherence
- Measure-Groundedness
- Measure-InferenceLatency: Measures inference latency over multiple runs
- Measure-MemoryUsage
- Measure-TokenThroughput: Measures token generation throughput
- Measure-Toxicity
- New-ABTest: Creates a new A/B test for comparing inference variants
- New-BaselineSnapshot: Creates a baseline snapshot of current model performance
- New-EvaluationSuite: Creates a new evaluation suite for testing LLM inference
- New-EvaluationTestCase
- New-PcaiEvaluationRunContext
- New-PcaiServerConfigFile
- Start-PcaiCompiledServer
- Stop-EvaluationRun
- Test-ForRegression: Tests current performance against baseline for regressions

## PC-AI.Gpu.psm1
- emits: Builds a per-GPU compatibility matrix showing whether each installed NVIDIA
    software component meets the minimum requirements for every GPU in the system.
- Resolve-PcaiCoreLibDll: Enumerates all installed NVIDIA GPUs and returns structured hardware metadata.
- is: Returns a real-time utilization snapshot for each installed NVIDIA GPU.
- resolves: Loads the PC-AI NVIDIA software registry JSON and returns it as a structured
    object, with optional filtering by component ID or category.
- Get-NvidiaSoftwareStatus: Compares installed NVIDIA software component versions against the curated
    software registry and returns a per-component status table.
- are: Configures NVIDIA-related environment variables for the current session or
    persistently for the current user or machine.
- Install-NvidiaSoftware: Downloads and silently installs an NVIDIA software component from the
    curated registry.
- writes: Runs a GPU preflight readiness check for LLM inference workloads.
- ran: Updates the local nvidia-software-registry.json with patched component
    entries or with versions auto-detected from the running system.

## PC-AI.Hardware.psm1
- Get-DeviceErrors: Gets devices with errors from Device Manager
- Get-DiskHealth: Gets disk health status including SMART information
- Get-NetworkAdapters: Gets physical network adapter status
- Get-SystemEvents: Gets system events related to disk and USB devices
- Get-UsbStatus: Gets USB device and controller status
- New-DiagnosticReport: Generates a comprehensive hardware diagnostic report

## PC-AI.LLM.psm1
- Get-LLMStatus: Checks the status of pcai-inference LLM service and available models
- Get-SystemInfoTool: Active system interrogation tool for the AI agent.
- Invoke-DocSearch: Search Microsoft and manufacturer documentation for technical details.
- Invoke-FunctionGemmaDataset: Build router training datasets for FunctionGemma.
- Invoke-FunctionGemmaEval: Run FunctionGemma evaluation metrics.
- Invoke-FunctionGemmaChat: Uses FunctionGemma (OpenAI-compatible API) to plan tool calls and optionally executes them.
- Invoke-FunctionGemmaTests: Run FunctionGemma dataset + evaluation tests.
- Invoke-FunctionGemmaTokenCache: Build a token cache for FunctionGemma training.
- Invoke-FunctionGemmaTrain: Run FunctionGemma LoRA fine-tuning.
- Invoke-LLMChat: Interactive chat interface with LLM providers (pcai-inference, FunctionGemma, OpenAI-compatible)
- Invoke-LLMChatRouted: Routes a user request through FunctionGemma tool-calling, then returns a final LLM response.
- Invoke-LLMChatTui
- Invoke-LogSearch: Search log files for a regex pattern (native-first).
- Invoke-NativeSearch: Invokes native high-performance search operations for LLM analysis
- script: Analyzes PC diagnostic reports using LLM
- Invoke-SmartDiagnosis
- Send-OllamaRequest: Sends a request to pcai-inference for text generation
- Set-LLMConfig: Configures LLM module settings
- Set-LLMProviderOrder

## PC-AI.Network.psm1
- Get-NetworkDiagnostics: Performs comprehensive network stack analysis
- Optimize-VSock: Optimizes VSock and TCP settings for WSL2 performance (Requires Administrator)
- Test-WSLConnectivity: Tests connectivity between Windows and WSL
- Watch-VSockPerformance: Real-time VSock and network interface performance monitoring

## PC-AI.Performance.psm1
- Get-DiskSpace: Analyzes disk space usage on all or specified drives.
- Get-PcaiDiskUsage: Gets disk usage statistics for a directory.
- Get-PcaiMemoryPressure: Analyzes system memory pressure for LLM agent optimization.
- Get-PcaiMemoryStat
- Get-PcaiOptimizationPlan: Generates prioritized memory/performance optimization recommendations.
- Get-PcaiProcessCategories: Classifies running processes into LLM-workload-relevant categories.
- Get-PcaiTopProcess: Gets top resource-consuming processes.
- Get-ProcessPerformance: Gets top processes sorted by CPU or memory usage.
- Optimize-Disks: Optimizes disks using TRIM for SSDs or defragmentation for HDDs.
- Test-PcaiNative
- Watch-SystemResources: Real-time monitoring of CPU, memory, and disk I/O.

## PC-AI.USB.psm1
- Dismount-UsbFromWSL: Detaches a USB device from WSL
- Get-PcaiNativeUsbDiagnostics
- Get-UsbDeviceList
- Get-UsbWSLStatus: Gets complete USB/WSL status
- Invoke-UsbBind: Binds a USB device for WSL sharing
- Mount-UsbToWSL: Attaches a USB device to WSL

## PC-AI.Virtualization.psm1
- Backup-WSLConfig: Creates a backup of .wslconfig
- Enable-WSLSystemd: Enables systemd in a WSL distribution by updating /etc/wsl.conf.
- Get-DockerStatus: Gets Docker Desktop status and configuration
- Get-HVSockProxyStatus
- Get-HyperVStatus: Gets Hyper-V status and configuration
- Get-PcaiServiceHealth: Checks the health of PC_AI inference services.
- Get-WSLEnvironmentHealth: Comprehensive health check for WSL, Docker, and VSock bridges
- Get-WSLStatus: Gets comprehensive WSL status information
- Get-WSLVsockBridgeStatus
- Install-HVSockProxy
- Install-WSLVsockBridge
- Invoke-PcaiDoctor: Runs a one-command doctor check for common runtime failures.
- Invoke-PcaiServiceHost: Starts the PC_AI inference service (Rust or legacy C# backend).
- Invoke-WSLDockerHealthCheck
- Invoke-WSLNetworkToolkit: Wrapper for the external WSL network toolkit script.
- Optimize-ModelHost: Optimizes the model host (WSL) for vLLM performance and resource safety.
- Optimize-WSLConfig: Optimizes .wslconfig for performance
- Register-HVSockServices: Registers Hyper-V socket (HVSOCK) services for WSL guests.
- Repair-WSLNetworking: Repairs WSL networking issues
- Set-PCaiServiceState
- Set-WSLDefenderExclusion: Adds Windows Defender exclusions for WSL
- Start-HVSockProxy
- Stop-HVSockProxy


