function Invoke-FunctionGemmaTests {
    <#
    .SYNOPSIS
        Run FunctionGemma dataset + evaluation tests.
    .DESCRIPTION
        Wraps Tools\run-functiongemma-tests.ps1.
    #>
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [switch]$UseLld,
        [switch]$LlmDebug
    )

    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $scriptPath = Join-Path $repoRoot 'Tools\run-functiongemma-tests.ps1'
    if (-not (Test-Path $scriptPath)) {
        throw "run-functiongemma-tests.ps1 not found at $scriptPath"
    }

    & $scriptPath @PSBoundParameters
}
