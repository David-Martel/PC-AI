#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [switch]$Child,
    [string]$OverrideName,
    [switch]$AlreadyLoaded
)
$ErrorActionPreference = 'Stop'
if ($Child) {
    foreach ($name in @('PCAI_CORE_LIB_PATH', 'PCAI_TEST_CORE_DLL', 'PCAI_NATIVE_DIR')) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $missing = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString('N'))
    [Environment]::SetEnvironmentVariable($OverrideName, $missing, 'Process')
    . (Join-Path $RepoRoot 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')
    # A fallback search is a failure even when there are no native artifacts installed.
    function Get-PcaiNativeCandidatePaths { throw 'Unexpected fallback discovery' }
    $script:PcaiNativeLoaded = [bool]$AlreadyLoaded
    $script:PcaiNativeVersion = 'stale-version'
    $script:PcaiNativeArtifactHash = 'stale-hash'
    $beforePath = $env:PATH
    $result = Initialize-PcaiNative -WarningAction SilentlyContinue
    if ($result -or $script:PcaiNativeLoaded) { throw 'Invalid override admitted' }
    if ([Environment]::GetEnvironmentVariable($OverrideName) -cne $missing) { throw 'Override overwritten' }
    if ($env:PATH -cne $beforePath) { throw 'PATH changed on rejection' }
    if ($script:PcaiNativeVersion -or $script:PcaiNativeArtifactHash) { throw 'Stale provenance retained' }
    $script:PcaiNativeLoaded = $true
    if (Test-PcaiNativeAvailable -WarningAction SilentlyContinue) { throw 'Availability wrapper bypassed admission' }
    return
}
$pwsh = (Get-Process -Id $PID).Path
# Selectors: core-path/test-path/native-directory x unloaded/stale-loaded.
# Protects: explicit operator intent over fallback or cached availability.
# Detects: discovery, environment mutation and stale provenance after rejection.
# Needs: PowerShell 7; no native binaries or hardware required.
# Breadcrumb: Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1.
$count = 0
foreach ($name in @('PCAI_CORE_LIB_PATH', 'PCAI_TEST_CORE_DLL', 'PCAI_NATIVE_DIR')) {
    foreach ($loaded in @($false, $true)) {
        $arguments = @('-NoLogo', '-NoProfile', '-File', $PSCommandPath,
            '-RepoRoot', $RepoRoot, '-Child', '-OverrideName', $name)
        if ($loaded) { $arguments += '-AlreadyLoaded' }
        & $pwsh @arguments
        if ($LASTEXITCODE -ne 0) { throw "Override admission failed: $name / loaded=$loaded" }
        $count++
    }
}
Write-Output "PASS: $count isolated initializer override controls"
