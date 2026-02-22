#Requires -Version 5.1
<#+
.SYNOPSIS
  Run Rust FunctionGemma LoRA fine-tuning with sensible defaults.

.DESCRIPTION
  Uses rust-functiongemma-train via Invoke-RustBuild.ps1 and prefers token caches
  and packed sequences for faster training.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ModelPath,
    [string]$TrainData,
    [string]$EvalData,
    [double]$EvalSplit,
    [string]$TokenCache,
    [string]$Output,
    [int]$Epochs = 1,
    [double]$Lr = 1e-5,
    [int]$LoraR = 16,
    [double]$LoraAlpha = 32.0,
    [double]$LoraDropout = 0.0,
    [int]$BatchSize = 1,
    [int]$GradAccum = 4,
    [switch]$PackSequences,
    [int]$MaxSeqLen,
    [int]$EosTokenId = 1,
    [switch]$DisableLora,
    [int]$WarmupSteps = 100,
    [string]$SchedulerType = 'cosine',
    [int]$EarlyStoppingPatience = 0,
    [double]$EarlyStoppingMinDelta = 0.001,
    [switch]$Use4Bit,
    [int]$EvalMaxBatches,
    [double]$MaxGradNorm = 1.0,
    [Nullable[UInt64]]$Seed,
    [switch]$NoShuffle,
    [int]$ProgressInterval = 1,
    [switch]$ProgressJson,
    [string]$ConfigPath,
    [string]$TargetDir,
    [switch]$UseLld,
    [switch]$LlmDebug
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainRoot = Join-Path $repoRoot 'Deploy\rust-functiongemma-train'

if (-not $ModelPath) { $ModelPath = Join-Path $repoRoot 'Models\functiongemma-270m-it' }
if (-not $TrainData) { $TrainData = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl' }
if (-not $Output) { $Output = Join-Path $repoRoot 'output\functiongemma-lora' }
if (-not $ConfigPath) { $ConfigPath = Join-Path $repoRoot 'Config\pcai-functiongemma.json' }

function Resolve-FullPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    if (Test-Path $Path) {
        return (Resolve-Path $Path).Path
    }
    return [System.IO.Path]::GetFullPath($Path)
}

$ModelPath = Resolve-FullPath $ModelPath
$TrainData = Resolve-FullPath $TrainData
$Output = Resolve-FullPath $Output
$ConfigPath = Resolve-FullPath $ConfigPath
if ($EvalData) { $EvalData = Resolve-FullPath $EvalData }
if ($TokenCache) { $TokenCache = Resolve-FullPath $TokenCache }
if ($TargetDir) { $TargetDir = Resolve-FullPath $TargetDir }

# Isolate cargo target dir to avoid workspace lock contention during training.
if (-not $TargetDir) {
    $preferredCacheRoot = 'T:\RustCache'
    if (Test-Path $preferredCacheRoot) {
        $TargetDir = Join-Path $preferredCacheRoot 'cargo-target-functiongemma'
    } else {
        $TargetDir = Join-Path $repoRoot 'output\cargo-target-functiongemma'
    }
}
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}
$env:CARGO_TARGET_DIR = $TargetDir
# Disable rust-analyzer preflight by default for training (RA diagnostics currently error).
$env:CARGO_RA_PREFLIGHT = '0'
$env:CARGO_RAP_PREFLIGHT = '0'

if (-not $PSBoundParameters.ContainsKey('MaxSeqLen') -and (Test-Path $ConfigPath)) {
    try {
        $cfg = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
        if ($null -ne $cfg.train.max_seq_len) {
            $MaxSeqLen = [int]$cfg.train.max_seq_len
        }
    } catch {
        Write-Host "Warning: Failed to parse $ConfigPath for max_seq_len; using CLI defaults." -ForegroundColor Yellow
    }
}

# Prefer existing token cache if present and none provided.
if (-not $TokenCache) {
    $candidate = Join-Path $repoRoot 'output\functiongemma-token-cache'
    if (Test-Path (Join-Path $candidate 'tokens.idx.json')) {
        $TokenCache = $candidate
    }
}

$pack = if ($PSBoundParameters.ContainsKey('PackSequences')) { $PackSequences.IsPresent } else { $true }

$cargoArgs = @(
    'run','--',
    '--config', $ConfigPath,
    'train',
    '--model-path', $ModelPath,
    '--train-data', $TrainData,
    '--output', $Output,
    '--epochs', $Epochs,
    '--lr', $Lr,
    '--lora-r', $LoraR,
    '--lora-alpha', $LoraAlpha,
    '--lora-dropout', $LoraDropout,
    '--batch-size', $BatchSize,
    '--grad-accum', $GradAccum,
    '--max-grad-norm', $MaxGradNorm,
    '--progress-interval', $ProgressInterval,
    '--eos-token-id', $EosTokenId,
    '--warmup-steps', $WarmupSteps,
    '--scheduler-type', $SchedulerType
)

if ($EvalData) { $cargoArgs += @('--eval-data', $EvalData) }
if ($EvalSplit) { $cargoArgs += @('--eval-split', $EvalSplit) }
if ($TokenCache) { $cargoArgs += @('--token-cache', $TokenCache) }
if ($pack) { $cargoArgs += '--pack-sequences' }
if ($null -ne $MaxSeqLen -and $MaxSeqLen -gt 0) { $cargoArgs += @('--max-seq-len', $MaxSeqLen) }
if (-not $DisableLora) { $cargoArgs += '--use-lora' }
if ($EarlyStoppingPatience -gt 0) { $cargoArgs += @('--early-stopping-patience', $EarlyStoppingPatience) }
if ($EarlyStoppingMinDelta -gt 0) { $cargoArgs += @('--early-stopping-min-delta', $EarlyStoppingMinDelta) }
if ($Use4Bit) { $cargoArgs += '--use-4bit' }
if ($PSBoundParameters.ContainsKey('EvalMaxBatches')) { $cargoArgs += @('--eval-max-batches', $EvalMaxBatches) }
if ($Seed) { $cargoArgs += @('--seed', $Seed) }
if ($NoShuffle) { $cargoArgs += '--no-shuffle' }
if ($ProgressJson) { $cargoArgs += '--progress-json' }

Write-Host "Running FunctionGemma training..." -ForegroundColor Cyan
Write-Host "Model: $ModelPath"
Write-Host "Train Data: $TrainData"
Write-Host "Output: $Output"
if ($TokenCache) { Write-Host "Token Cache: $TokenCache" }

& (Join-Path $repoRoot 'Tools\Invoke-RustBuild.ps1') `
    -Path $trainRoot `
    -CargoArgs $cargoArgs `
    -UseLld:$UseLld `
    -LlmDebug:$LlmDebug

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training complete." -ForegroundColor Green
