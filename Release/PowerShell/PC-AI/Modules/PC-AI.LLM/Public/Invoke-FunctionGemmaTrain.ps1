function Invoke-FunctionGemmaTrain {
    <#
    .SYNOPSIS
        Run FunctionGemma LoRA fine-tuning.
    .DESCRIPTION
        Wraps Tools\Invoke-FunctionGemmaTrain.ps1 with optimized defaults.
    #>
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [string]$ModelPath,
        [string]$TrainData,
        [string]$EvalData,
        [double]$EvalSplit,
        [string]$TokenCache,
        [string]$Output,
        [int]$Epochs,
        [double]$Lr,
        [int]$LoraR,
        [double]$LoraAlpha,
        [double]$LoraDropout,
        [int]$BatchSize,
        [int]$GradAccum,
        [switch]$PackSequences,
        [int]$MaxSeqLen,
        [int]$EosTokenId,
        [switch]$DisableLora,
        [int]$WarmupSteps,
        [string]$SchedulerType,
        [int]$EarlyStoppingPatience,
        [double]$EarlyStoppingMinDelta,
        [switch]$Use4Bit,
        [int]$EvalMaxBatches,
        [double]$MaxGradNorm,
        [Nullable[UInt64]]$Seed,
        [switch]$NoShuffle,
        [int]$ProgressInterval,
        [switch]$ProgressJson,
        [string]$ConfigPath,
        [string]$TargetDir,
        [switch]$UseLld,
        [switch]$LlmDebug
    )

    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $scriptPath = Join-Path $repoRoot 'Tools\Invoke-FunctionGemmaTrain.ps1'
    if (-not (Test-Path $scriptPath)) {
        throw "Invoke-FunctionGemmaTrain.ps1 not found at $scriptPath"
    }

    & $scriptPath @PSBoundParameters
}
