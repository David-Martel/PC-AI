#Requires -Version 5.1
<#
.SYNOPSIS
    Compatibility wrapper for the unified repository build orchestrator.

.DESCRIPTION
    This script is retained for backward compatibility. It forwards legacy
    Native build workflows to the root Build.ps1 so all build/test/deploy
    operations are centralized in one primary entrypoint.
#>

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$Clean,
    [switch]$SkipRust,
    [switch]$SkipCSharp,
    [switch]$Test,
    [switch]$Coverage,
    [switch]$PreFlight,
    [switch]$Docs,
    [switch]$DocsBuild,
    [switch]$EnableCuda,
    [switch]$Package,
    [switch]$Deploy
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildScript = Join-Path $repoRoot 'Build.ps1'
if (-not (Test-Path $buildScript)) {
    throw "Build.ps1 not found at expected path: $buildScript"
}

if ($Coverage -or $PreFlight -or $Docs -or $DocsBuild) {
    Write-Warning 'Coverage/PreFlight/Docs switches are no longer handled by Native/build.ps1. Use root tooling under Build.ps1 and Tools/*.ps1.'
}

$components = @()
if (-not $SkipRust) { $components += 'inference' }
if (-not $SkipCSharp) { $components += 'native' }

if ($components.Count -eq 0) {
    Write-Warning 'Nothing to build (both -SkipRust and -SkipCSharp were specified).'
    exit 0
}

$componentCount = $components.Count
$componentIndex = 0
foreach ($component in $components) {
    $componentIndex++
    $isLast = ($componentIndex -eq $componentCount)

    $args = @{
        Component     = $component
        Configuration = $Configuration
    }

    if ($EnableCuda -and $component -eq 'inference') { $args.EnableCuda = $true }
    if ($Clean -and $componentIndex -eq 1) { $args.Clean = $true }
    if ($Test) { $args.RunTests = $true }
    if ($SkipCSharp -and $component -eq 'inference') { $args.SkipTests = $true }
    if ($Package -and $isLast) { $args.Package = $true }
    if ($Deploy -and $isLast) { $args.Deploy = $true }

    Write-Host "Forwarding Native build to Build.ps1 (Component=$component)..." -ForegroundColor Cyan
    & $buildScript @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

exit 0
