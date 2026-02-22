#Requires -Version 5.1
<#
.SYNOPSIS
  Compatibility wrapper for FunctionGemma tests.

.DESCRIPTION
  Uses the root Build.ps1 orchestrator with -RunTests to execute the
  FunctionGemma runtime/train Rust test suites.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Fast,
    [switch]$EvalReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$buildScript = Join-Path $repoRoot 'Build.ps1'
if (-not (Test-Path $buildScript)) {
    throw "Build.ps1 not found at expected path: $buildScript"
}

if ($Fast) {
    Write-Host 'Fast mode requested: running Build.ps1 functiongemma test path only (no extra eval).' -ForegroundColor Yellow
}

& $buildScript -Component functiongemma -RunTests
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}

if ($EvalReport) {
    & (Join-Path $repoRoot 'Tools\run-functiongemma-eval.ps1') -FastEval:$Fast
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

exit 0
