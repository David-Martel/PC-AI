#Requires -Version 5.1
<#
.SYNOPSIS
    Compatibility wrapper that routes NukeNul builds through the repo root Build.ps1.

.DESCRIPTION
    Build.ps1 is the canonical build orchestrator for this repository. This script is
    retained for backwards compatibility and forwards supported options to Build.ps1.
#>

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$Publish,
    [switch]$Clean,
    [switch]$SkipRust,
    [switch]$SkipCSharp
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$buildScript = Join-Path $repoRoot 'Build.ps1'

if (-not (Test-Path $buildScript)) {
    throw "Root Build.ps1 not found: $buildScript"
}

if ($Publish) {
    Write-Warning '-Publish is not currently exposed via Build.ps1 nukenul component; proceeding with standard orchestrated build.'
}
if ($SkipRust -or $SkipCSharp) {
    Write-Warning '-SkipRust/-SkipCSharp are not supported in unified mode; Build.ps1 executes the full NukeNul pipeline.'
}

& $buildScript -Component nukenul -Configuration $Configuration -Clean:$Clean
exit $LASTEXITCODE
