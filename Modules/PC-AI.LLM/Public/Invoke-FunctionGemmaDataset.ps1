function Invoke-FunctionGemmaDataset {
    <#
    .SYNOPSIS
        Build router training datasets for FunctionGemma.
    .DESCRIPTION
        Wraps Tools\prepare-functiongemma-router-data.ps1 to generate JSONL
        training data and tool test vectors.
    #>
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [string]$ToolsPath,
        [string]$DiagnosePrompt,
        [string]$ChatPrompt,
        [string]$ScenariosPath,
        [string]$Output,
        [string]$TestVectors,
        [int]$MaxCases = 24,
        [switch]$NoToolCoverage,
        [switch]$Stream,
        [switch]$UseLld,
        [switch]$LlmDebug,
        [switch]$UseNative,
        [switch]$NativeOnly
    )

    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $scriptPath = Join-Path $repoRoot 'Tools\prepare-functiongemma-router-data.ps1'
    if (-not (Test-Path $scriptPath)) {
        throw "prepare-functiongemma-router-data.ps1 not found at $scriptPath"
    }

    & $scriptPath @PSBoundParameters
}
