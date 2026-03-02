function Invoke-FunctionGemmaTokenCache {
    <#
    .SYNOPSIS
        Build a token cache for FunctionGemma training.
    .DESCRIPTION
        Wraps Tools\prepare-functiongemma-token-cache.ps1.
    #>
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [string]$Input,
        [string]$TokenizerPath,
        [string]$OutputDir,
        [switch]$UseLld,
        [switch]$LlmDebug
    )

    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $scriptPath = Join-Path $repoRoot 'Tools\prepare-functiongemma-token-cache.ps1'
    if (-not (Test-Path $scriptPath)) {
        throw "prepare-functiongemma-token-cache.ps1 not found at $scriptPath"
    }

    & $scriptPath @PSBoundParameters
}
