#Requires -Version 5.1
<#
.SYNOPSIS
  Compatibility wrapper for FunctionGemma builds.

.DESCRIPTION
  Forwards build requests to the root Build.ps1 orchestrator so FunctionGemma
  builds remain compliant with the unified build framework.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Release,
    [switch]$EnableCuda,
    [switch]$Clean,
    [switch]$Test,
    [switch]$Package,
    [switch]$Deploy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$buildScript = Join-Path $repoRoot 'Build.ps1'
if (-not (Test-Path $buildScript)) {
    throw "Build.ps1 not found at expected path: $buildScript"
}

$args = @{
    Component     = 'functiongemma'
    Configuration = if ($Release) { 'Release' } else { 'Debug' }
}
if ($EnableCuda) { $args.EnableCuda = $true }
if ($Clean) { $args.Clean = $true }
if ($Test) { $args.RunTests = $true }
if ($Package) { $args.Package = $true }
if ($Deploy) { $args.Deploy = $true }

& $buildScript @args
exit $LASTEXITCODE
