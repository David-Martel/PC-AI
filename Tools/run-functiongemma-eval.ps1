#Requires -Version 5.1

<#+
.SYNOPSIS
  Runs a FunctionGemma evaluation pass via Build.ps1 and writes a metrics report.
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

$functionGemmaArgs = @(
    '--config', $ConfigPath, 'eval',
    '--model-path', $ModelPath,
    '--test-data', $TestData,
    '--lora-r', $LoraR,
    '--max-new-tokens', $MaxNewTokens,
    '--metrics-output', $Output
)

if ($Adapters) { $functionGemmaArgs += @('--adapters', $Adapters) }
if ($FastEval) { $functionGemmaArgs += '--fast-eval' }
if ($NoSchemaValidate) { $functionGemmaArgs += '--schema-validate=false' }
if ($MaxSamples -gt 0) { $functionGemmaArgs += @('--max-samples', $MaxSamples) }
if ($SamplesOutput) { $functionGemmaArgs += @('--samples-output', $SamplesOutput) }
if ($KvCacheQuant) { $functionGemmaArgs += @('--kv-cache-quant', $KvCacheQuant) }
if ($KvCacheMaxLen -gt 0) { $functionGemmaArgs += @('--kv-cache-max-len', $KvCacheMaxLen) }
if ($KvCacheStore) { $functionGemmaArgs += @('--kv-cache-store', $KvCacheStore) }
if ($KvCacheStreaming) { $functionGemmaArgs += '--kv-cache-streaming' }
if ($KvCacheBlockLen -gt 0) { $functionGemmaArgs += @('--kv-cache-block-len', $KvCacheBlockLen) }
if ($VerboseOutput -or $VerbosePreference -ne 'SilentlyContinue' -or $SamplesOutput) {
    $functionGemmaArgs += '--verbose'
} elseif ($Quiet -or $FastEval -or $MaxSamples -gt 0 -or -not $PSBoundParameters.ContainsKey('Quiet')) {
    $functionGemmaArgs += '--quiet'
}

Write-Host "Running FunctionGemma eval..." -ForegroundColor Cyan
Write-Host "Model: $ModelPath"
Write-Host "Test Data: $TestData"
Write-Host "Metrics Output: $Output"

if ($LlmDebug) {
    Write-Warning '-LlmDebug is not currently consumed by Build.ps1; continuing without extra debug toggles.'
}

$buildScript = Join-Path $repoRoot 'Build.ps1'
$prevUseLld = $env:CARGO_USE_LLD
if ($UseLld) { $env:CARGO_USE_LLD = '1' }

try {
    & $buildScript -Component functiongemma-eval -Configuration Release -FunctionGemmaArgs $functionGemmaArgs
} finally {
    if ($UseLld) {
        if ([string]::IsNullOrEmpty($prevUseLld)) {
            Remove-Item Env:CARGO_USE_LLD -ErrorAction SilentlyContinue
        } else {
            $env:CARGO_USE_LLD = $prevUseLld
        }
    }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path $Output)) {
    throw "Eval report missing: $Output"
}

$size = (Get-Item $Output).Length
if ($size -le 0) {
    throw "Eval report is empty: $Output"
}

Write-Host "Eval report written." -ForegroundColor Green
