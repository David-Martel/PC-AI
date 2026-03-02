function Invoke-FunctionGemmaEval {
    <#
    .SYNOPSIS
        Run FunctionGemma evaluation metrics.
    .DESCRIPTION
        Wraps Tools\run-functiongemma-eval.ps1.
    #>
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [string]$ModelPath,
        [string]$TestData,
        [string]$Adapters,
        [string]$Output,
        [string]$ConfigPath,
        [int]$MaxNewTokens,
        [int]$LoraR,
        [int]$MaxSamples,
        [switch]$Quiet,
        [switch]$VerboseOutput,
        [switch]$FastEval,
        [switch]$NoSchemaValidate,
        [switch]$UseLld,
        [switch]$LlmDebug
    )

    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $scriptPath = Join-Path $repoRoot 'Tools\run-functiongemma-eval.ps1'
    if (-not (Test-Path $scriptPath)) {
        throw "run-functiongemma-eval.ps1 not found at $scriptPath"
    }

    & $scriptPath @PSBoundParameters
}
