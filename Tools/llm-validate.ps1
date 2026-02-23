#Requires -Version 5.1

<#
.SYNOPSIS
  Validates PC_AI LLM flows using DIAGNOSE.md + DIAGNOSE_LOGIC.md system prompts.

.DESCRIPTION
  Runs Invoke-PCDiagnosis with a small synthetic report and Invoke-SmartDiagnosis
  against a target path. Fails fast if Ollama/Router is not reachable.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\llm-validate.ps1
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$Model,

    [Parameter()]
    [string]$Path = $env:TEMP
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = if ($env:PCAI_ROOT) {
    $env:PCAI_ROOT
} else {
    Split-Path -Parent $PSScriptRoot
}

$llmConfigPath = Join-Path $repoRoot 'Config\llm-config.json'
if (-not $Model) {
    if (Test-Path $llmConfigPath) {
        try {
            $llmConfig = Get-Content -Path $llmConfigPath -Raw | ConvertFrom-Json
            $Model = $llmConfig.providers.'pcai-inference'.defaultModel
        } catch {
            Write-Warning "Failed to parse llm-config.json: $_"
        }
    }
    if (-not $Model) {
        $Model = 'pcai-inference'
    }
}

Import-Module (Join-Path $repoRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1') -Force

Write-Host "[LLM Validation] Checking router/Ollama connectivity..." -ForegroundColor Cyan
$status = Get-LLMStatus -TestConnection -IncludeLMStudio
if (-not $status.Ollama.ApiConnected) {
    throw "Ollama/router not reachable at $($status.Ollama.ApiUrl). Start the router or Ollama first."
}

$sampleReport = @"
SYSTEM SUMMARY
OS: Windows 11 Pro 23H2
CPU: 16-core
RAM: 32 GB
GPU: NVIDIA RTX 5060 Ti

STORAGE
Drive C: 5% free (9 GB of 180 GB)
Drive T: 62% free (310 GB of 500 GB)

EVENT LOG EXCERPT
- Disk warning: \"The device, \Device\Harddisk1\DR2, has a bad block.\"
- App crash: \"AppX service failed to start\"

THERMALS
CPU: 91C sustained under load
GPU: 82C sustained under load
"@

Write-Host "[LLM Validation] Running Invoke-PCDiagnosis..." -ForegroundColor Cyan
$pcResult = Invoke-PCDiagnosis -ReportText $sampleReport -Model $Model -Temperature 0.2
if (-not $pcResult.Analysis -or $pcResult.Analysis.Length -lt 200) {
    throw "Invoke-PCDiagnosis returned an unexpectedly short response."
}

Write-Host "[LLM Validation] Running Invoke-SmartDiagnosis..." -ForegroundColor Cyan
$smartResult = Invoke-SmartDiagnosis -Path $Path -AnalysisType Quick -Model $Model
if (-not $smartResult.LLMAnalysis -or $smartResult.LLMAnalysis.Length -lt 200) {
    throw "Invoke-SmartDiagnosis returned an unexpectedly short response."
}

Write-Host "[LLM Validation] OK - both flows returned substantial responses." -ForegroundColor Green
