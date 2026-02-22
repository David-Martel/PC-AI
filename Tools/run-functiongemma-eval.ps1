#Requires -Version 5.1

<#+
.SYNOPSIS
  Runs a Rust FunctionGemma evaluation pass and writes a metrics report.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ModelPath,
    [string]$TestData,
    [string]$Adapters,
    [string]$Output,
    [string]$ConfigPath,
    [int]$MaxNewTokens = 64,
    [int]$LoraR = 16,
    [int]$MaxSamples = 0,
    [string]$KvCacheQuant,
    [int]$KvCacheMaxLen,
    [string]$KvCacheStore,
    [switch]$KvCacheStreaming,
    [int]$KvCacheBlockLen,
    [switch]$Quiet,
    [switch]$VerboseOutput,
    [switch]$FastEval,
    [switch]$NoSchemaValidate,
    [string]$SamplesOutput,
    [switch]$UseLld,
    [switch]$LlmDebug
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$VerboseOutput = $VerboseOutput -or ($VerbosePreference -ne 'SilentlyContinue')

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainRoot = Join-Path $repoRoot 'Deploy\rust-functiongemma-train'

if (-not $ModelPath) { $ModelPath = Join-Path $repoRoot 'Models\functiongemma-270m-it' }
if (-not $TestData) { $TestData = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl' }
if (-not $Output) { $Output = Join-Path $repoRoot 'Reports\functiongemma_eval_metrics.json' }
if (-not $ConfigPath) { $ConfigPath = Join-Path $repoRoot 'Config\pcai-functiongemma.json' }
if (-not $SamplesOutput) { $SamplesOutput = $null }

function Resolve-FullPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    if (Test-Path $Path) {
        return (Resolve-Path $Path).Path
    }
    return [System.IO.Path]::GetFullPath($Path)
}

$ModelPath = Resolve-FullPath $ModelPath
$TestData = Resolve-FullPath $TestData
$Output = Resolve-FullPath $Output
$ConfigPath = Resolve-FullPath $ConfigPath
if ($Adapters) {
    $Adapters = Resolve-FullPath $Adapters
    if (Test-Path $Adapters -PathType Container) {
        $adapterFile = Join-Path $Adapters 'adapter_model.safetensors'
        if (Test-Path $adapterFile) {
            $Adapters = $adapterFile
        } else {
            Write-Warning "Adapters path is a directory without adapter_model.safetensors: $Adapters"
        }
    }
}
if ($SamplesOutput) { $SamplesOutput = Resolve-FullPath $SamplesOutput }

if (-not (Test-Path $ModelPath)) {
    Write-Warning "Model path not found. Skipping eval: $ModelPath"
    return
}

if (-not (Test-Path $TestData)) {
    throw "Test data not found: $TestData"
}

$parentDir = Split-Path -Parent $Output
if ($parentDir -and -not (Test-Path $parentDir)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

$cargoArgs = @(
    'run','--','--config', $ConfigPath, 'eval',
    '--model-path', $ModelPath,
    '--test-data', $TestData,
    '--lora-r', $LoraR,
    '--max-new-tokens', $MaxNewTokens,
    '--metrics-output', $Output
)

if ($Adapters) { $cargoArgs += @('--adapters', $Adapters) }
if ($FastEval) { $cargoArgs += '--fast-eval' }
if ($NoSchemaValidate) { $cargoArgs += '--schema-validate=false' }
if ($MaxSamples -gt 0) { $cargoArgs += @('--max-samples', $MaxSamples) }
if ($SamplesOutput) { $cargoArgs += @('--samples-output', $SamplesOutput) }
if ($KvCacheQuant) { $cargoArgs += @('--kv-cache-quant', $KvCacheQuant) }
if ($KvCacheMaxLen -gt 0) { $cargoArgs += @('--kv-cache-max-len', $KvCacheMaxLen) }
if ($KvCacheStore) { $cargoArgs += @('--kv-cache-store', $KvCacheStore) }
if ($KvCacheStreaming) { $cargoArgs += '--kv-cache-streaming' }
if ($KvCacheBlockLen -gt 0) { $cargoArgs += @('--kv-cache-block-len', $KvCacheBlockLen) }
if ($VerboseOutput -or $VerbosePreference -ne 'SilentlyContinue' -or $SamplesOutput) {
    $cargoArgs += '--verbose'
} elseif ($Quiet -or $FastEval -or $MaxSamples -gt 0 -or -not $PSBoundParameters.ContainsKey('Quiet')) {
    $cargoArgs += '--quiet'
}

Write-Host "Running FunctionGemma eval..." -ForegroundColor Cyan
Write-Host "Model: $ModelPath"
Write-Host "Test Data: $TestData"
Write-Host "Metrics Output: $Output"

& (Join-Path $repoRoot 'Tools\Invoke-RustBuild.ps1') `
    -Path $trainRoot `
    -CargoArgs $cargoArgs `
    -UseLld:$UseLld `
    -LlmDebug:$LlmDebug

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path $Output)) {
    throw "Eval report missing: $Output"
}

$size = (Get-Item $Output).Length
if ($size -le 0) {
    throw "Eval report is empty: $Output"
}

Write-Host "Eval report written." -ForegroundColor Green
