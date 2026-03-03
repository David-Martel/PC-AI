#Requires -Version 5.1

<#
.SYNOPSIS
    Runs a one-command doctor check for common runtime failures.

.DESCRIPTION
    Executes a focused health check for pcai-inference, FunctionGemma router,
    native inference DLL availability, and GPU readiness. Returns the health
    report plus recommended next steps.

.PARAMETER PcaiInferenceUrl
    URL of the pcai-inference HTTP server. Defaults to providers.pcai-inference.baseUrl from Config/llm-config.json.

.PARAMETER FunctionGemmaUrl
    URL of the FunctionGemma router. Defaults to providers.functiongemma.baseUrl from Config/llm-config.json.

.PARAMETER NativeDllSearchPaths
    Optional list of pcai_inference.dll search paths to override defaults.

.PARAMETER CheckLegacyProviders
    Also check legacy Ollama/vLLM providers for backward compatibility.

.EXAMPLE
    Invoke-PcaiDoctor

.EXAMPLE
    Invoke-PcaiDoctor -CheckLegacyProviders
#>
function Invoke-PcaiDoctor {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$PcaiInferenceUrl,

        [Parameter()]
        [string]$FunctionGemmaUrl,

        [Parameter()]
        [string]$ConfigPath,

        [Parameter()]
        [string[]]$NativeDllSearchPaths,

        [Parameter()]
        [switch]$CheckLegacyProviders
    )

    $healthParams = @{
        ConfigPath = $ConfigPath
    }
    if ($PSBoundParameters.ContainsKey('PcaiInferenceUrl') -and -not [string]::IsNullOrWhiteSpace($PcaiInferenceUrl)) { $healthParams.PcaiInferenceUrl = $PcaiInferenceUrl }
    if ($PSBoundParameters.ContainsKey('FunctionGemmaUrl') -and -not [string]::IsNullOrWhiteSpace($FunctionGemmaUrl)) { $healthParams.FunctionGemmaUrl = $FunctionGemmaUrl }

    if ($NativeDllSearchPaths -and $NativeDllSearchPaths.Count -gt 0) {
        $healthParams.NativeDllSearchPaths = $NativeDllSearchPaths
    }

    if ($CheckLegacyProviders) {
        $healthParams.CheckLegacyProviders = $true
    }

    $health = Get-PcaiServiceHealth @healthParams

    $recommendations = @()
    $pcaiTarget = if (-not [string]::IsNullOrWhiteSpace($PcaiInferenceUrl)) { $PcaiInferenceUrl } else { 'providers.pcai-inference.baseUrl' }
    $routerTarget = if (-not [string]::IsNullOrWhiteSpace($FunctionGemmaUrl)) { $FunctionGemmaUrl } else { 'providers.functiongemma.baseUrl' }

    if ($health.PcaiInference.Status -ne 'OK') {
        $recommendations += "pcai-inference not responding at $pcaiTarget. Start it with Invoke-PcaiServiceHost."
    } elseif (-not $health.PcaiInference.ModelLoaded) {
        $recommendations += 'pcai-inference is responding but no model is loaded. Load a model or configure the server.'
    }

    if ($health.FunctionGemma.Status -ne 'OK') {
        $recommendations += "FunctionGemma router not responding at $routerTarget. Start rust-functiongemma-runtime for providers.functiongemma.baseUrl."
    }

    if (-not $health.NativeFFI.DllExists) {
        $recommendations += 'pcai_inference.dll not found. Build with: .\Build.ps1 -Component inference.'
    }

    if ($health.Gpu.Status -ne 'OK') {
        $recommendations += 'No GPU detected. If you expect GPU acceleration, verify driver installation.'
    } else {
        $hasNvidia = $false
        if ($health.Gpu.Devices) {
            foreach ($gpu in $health.Gpu.Devices) {
                if ($gpu.Name -match 'NVIDIA') {
                    $hasNvidia = $true
                    break
                }
            }
        }
        if ($hasNvidia -and -not $health.Gpu.NvidiaSmi) {
            $recommendations += 'NVIDIA GPU detected but nvidia-smi is missing. Install or update NVIDIA drivers.'
        }
        if ($health.Docker.Running -and -not $health.Gpu.NvidiaRuntime) {
            $recommendations += 'Docker running but NVIDIA runtime not detected. GPU containers may not work.'
        }
    }

    return [PSCustomObject]@{
        Health = $health
        Recommendations = $recommendations
    }
}
