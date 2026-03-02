#Requires -Version 5.1

<#
.SYNOPSIS
    Checks the health of PC_AI inference services.

.DESCRIPTION
    Validates the status of pcai-inference Rust backend, FunctionGemma router,
    WSL2, Docker, and optional legacy providers (Ollama, vLLM).

.PARAMETER Distribution
    WSL distribution to check. Default: Ubuntu

.PARAMETER PcaiInferenceUrl
    URL of the pcai-inference HTTP server. Default: http://127.0.0.1:8080

.PARAMETER FunctionGemmaUrl
    URL of the FunctionGemma router. Default: http://127.0.0.1:8000

.PARAMETER CheckLegacyProviders
    Also check legacy Ollama/vLLM providers for backward compatibility.

.PARAMETER NativeDllSearchPaths
    Optional list of pcai_inference.dll search paths to override defaults.

.EXAMPLE
    Get-PcaiServiceHealth

.EXAMPLE
    Get-PcaiServiceHealth -CheckLegacyProviders
#>
function Get-PcaiServiceHealth {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Distribution = "Ubuntu",

        [Parameter()]
        [string]$PcaiInferenceUrl = "http://127.0.0.1:8080",

        [Parameter()]
        [string]$FunctionGemmaUrl = "http://127.0.0.1:8000",

        [Parameter()]
        [switch]$CheckLegacyProviders,

        [Parameter()]
        [string[]]$NativeDllSearchPaths,

        # Legacy parameters (mapped to new backends)
        [Parameter()]
        [string]$OllamaBaseUrl = "http://localhost:11434",

        [Parameter()]
        [string]$vLLMBaseUrl = "http://localhost:8000"
    )

    $results = [PSCustomObject]@{
        Timestamp       = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        OverallStatus   = 'Unknown'
        PcaiInference   = @{ Status = 'Unknown'; Responding = $false; Backend = $null; ModelLoaded = $false }
        FunctionGemma   = @{ Status = 'Unknown'; Responding = $false }
        WSL             = @{ Status = 'Unknown'; Running = $false }
        Docker          = @{ Status = 'Unknown'; Running = $false }
        NativeFFI       = @{ Status = 'Unknown'; DllExists = $false }
        Bridges         = @{ Status = 'Unknown'; Count = 0 }
        Gpu             = @{ Status = 'Unknown'; Devices = @(); NvidiaSmi = $false; NvidiaRuntime = $false }
    }

    # Add legacy fields if checking
    if ($CheckLegacyProviders) {
        $results | Add-Member -NotePropertyName 'Ollama' -NotePropertyValue @{ Status = 'Unknown'; Responding = $false; Version = $null }
        $results | Add-Member -NotePropertyName 'vLLM' -NotePropertyValue @{ Status = 'Unknown'; Responding = $false }
    }

    # 1. Check pcai-inference HTTP server (primary backend)
    try {
        $response = Invoke-RestMethod -Uri "$PcaiInferenceUrl/health" -Method Get -TimeoutSec 3 -ErrorAction Stop
        $results.PcaiInference.Status = 'OK'
        $results.PcaiInference.Responding = $true
        if ($response.backend) {
            $results.PcaiInference.Backend = $response.backend
        }
        if ($response.model_loaded) {
            $results.PcaiInference.ModelLoaded = $response.model_loaded
        }
    } catch {
        # Try OpenAI-compatible models endpoint as fallback
        try {
            $modelsResponse = Invoke-RestMethod -Uri "$PcaiInferenceUrl/v1/models" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $results.PcaiInference.Status = 'OK'
            $results.PcaiInference.Responding = $true
            if ($modelsResponse.data -and $modelsResponse.data.Count -gt 0) {
                $results.PcaiInference.Backend = $modelsResponse.data[0].id
                $results.PcaiInference.ModelLoaded = $true
            }
        } catch {
            $results.PcaiInference.Status = 'NotRunning'
        }
    }

    # 2. Check FunctionGemma router
    try {
        $fgResponse = Invoke-RestMethod -Uri "$FunctionGemmaUrl/v1/models" -Method Get -TimeoutSec 2 -ErrorAction Stop
        $results.FunctionGemma.Status = 'OK'
        $results.FunctionGemma.Responding = $true
    } catch {
        try {
            # Try health endpoint
            $null = Invoke-RestMethod -Uri "$FunctionGemmaUrl/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $results.FunctionGemma.Status = 'OK'
            $results.FunctionGemma.Responding = $true
        } catch {
            $results.FunctionGemma.Status = 'NotRunning'
        }
    }

    # 3. Check Native FFI DLL
    $dllPaths = if ($NativeDllSearchPaths -and $NativeDllSearchPaths.Count -gt 0) {
        $NativeDllSearchPaths
    } else {
        @(
            "$env:USERPROFILE\.local\bin\pcai_inference.dll",
            'T:\RustCache\cargo-target\release\pcai_inference.dll',
            "$PSScriptRoot\..\..\..\Native\pcai_core\pcai_inference\target\release\pcai_inference.dll",
            "$env:USERPROFILE\PC_AI\Native\pcai_core\pcai_inference\target\release\pcai_inference.dll",
            "$PSScriptRoot\..\..\..\.pcai\build\artifacts\pcai-llamacpp\pcai_inference.dll",
            "$PSScriptRoot\..\..\..\.pcai\build\artifacts\pcai-mistralrs\pcai_inference.dll"
        )
    }
    foreach ($dllPath in $dllPaths) {
        if (Test-Path $dllPath) {
            $results.NativeFFI.DllExists = $true
            $results.NativeFFI.Status = 'Available'
            $results.NativeFFI | Add-Member -NotePropertyName 'Path' -NotePropertyValue $dllPath -Force
            break
        }
    }
    if (-not $results.NativeFFI.DllExists) {
        $results.NativeFFI.Status = 'NotBuilt'
    }

    # 4. Check WSL
    try {
        $wslStatus = wsl -l -v 2>$null | Select-String "$Distribution"
        if ($wslStatus -match "Running") {
            $results.WSL.Status = 'OK'
            $results.WSL.Running = $true
        } else {
            $results.WSL.Status = 'Stopped'
        }
    } catch {
        $results.WSL.Status = 'NotInstalled'
    }

    # 5. Check Docker (optional, not required for native inference)
    try {
        $dockerProc = Get-Process -Name "Docker Desktop", "com.docker.backend" -ErrorAction SilentlyContinue
        if ($dockerProc) {
            $results.Docker.Running = $true
            $info = docker info --format '{{.ID}}' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $results.Docker.Status = 'OK'
            } else {
                $results.Docker.Status = 'DaemonNotResponding'
            }
        } else {
            $results.Docker.Status = 'NotRunning'
        }
    } catch {
        $results.Docker.Status = 'NotInstalled'
    }

    # 5b. Check GPU status (host + optional Docker runtime)
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop
        $results.Gpu.Devices = @(
            $gpus | ForEach-Object {
                [PSCustomObject]@{
                    Name          = $_.Name
                    DriverVersion = $_.DriverVersion
                    Status        = $_.Status
                    PnpDeviceId   = $_.PNPDeviceID
                }
            }
        )
        $results.Gpu.Status = if ($results.Gpu.Devices.Count -gt 0) { 'OK' } else { 'NotFound' }

        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $results.Gpu.NvidiaSmi = $true
        }

        if ($results.Docker.Running) {
            try {
                $runtimes = docker info --format '{{json .Runtimes}}' 2>$null | ConvertFrom-Json
                if ($runtimes -and ($runtimes.PSObject.Properties.Name -contains 'nvidia')) {
                    $results.Gpu.NvidiaRuntime = $true
                }
            } catch { }
        }
    } catch {
        $results.Gpu.Status = 'Unknown'
    }

    # 6. Check legacy providers if requested
    if ($CheckLegacyProviders) {
        # Ollama
        try {
            $ollamaResponse = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $results.Ollama.Status = 'OK'
            $results.Ollama.Responding = $true
            try {
                $ver = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/version" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
                $results.Ollama.Version = $ver.version
            } catch { }
        } catch {
            $results.Ollama.Status = 'NotRunning'
        }

        # vLLM
        try {
            $vResponse = Invoke-RestMethod -Uri "$vLLMBaseUrl/v1/models" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $results.vLLM.Status = 'OK'
            $results.vLLM.Responding = $true
        } catch {
            $results.vLLM.Status = 'NotRunning'
        }
    }

    # 7. Check Bridges (socat processes in WSL) - optional
    if ($results.WSL.Running) {
        try {
            $bridgeCount = wsl -d $Distribution -- pgrep -c socat 2>$null
            $results.Bridges.Count = [int]($bridgeCount.Trim() -as [int])
            $results.Bridges.Status = if ($results.Bridges.Count -gt 0) { 'OK' } else { 'None' }
        } catch {
            $results.Bridges.Status = 'NotChecked'
        }
    }

    # Final Overall Status - based on pcai-inference (primary backend)
    if ($results.PcaiInference.Status -eq 'OK') {
        $results.OverallStatus = 'Healthy'
    } elseif ($results.NativeFFI.DllExists) {
        $results.OverallStatus = 'FFIAvailable'
    } else {
        $results.OverallStatus = 'Degraded'
    }

    return $results
}
