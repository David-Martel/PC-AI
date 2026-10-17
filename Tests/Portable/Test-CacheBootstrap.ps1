#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1'))
$ErrorActionPreference = 'Stop'
. $SourcePath
$names = @('USERPROFILE','LOCALAPPDATA','PCAI_REDIS_CLI_PATH','PCAI_CACHE_PROVIDER')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$fixture = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
try {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    # Protects memory-only startup; detects null-root Join-Path failure.
    # Needs profile-free pwsh on either OS; breadcrumb: Get-PcaiExternalCacheConfig.
    $config = Get-PcaiExternalCacheConfig
    if ($config.Provider -ne 'memory' -or $config.Enabled) { throw 'Default cache must be memory-only' }
    Write-Output 'PASS: absent Windows roots preserve memory bootstrap'
    # Protects partial/minimal profiles; detects whitespace being used as a root.
    # Needs no Redis service; breadcrumb: Redis CLI candidate construction.
    $env:USERPROFILE = ' '; $env:LOCALAPPDATA = ' '
    $config = Get-PcaiExternalCacheConfig
    if ($config.Provider -ne 'memory') { throw 'Whitespace roots altered provider' }
    Write-Output 'PASS: whitespace Windows roots do not break bootstrap'
    # Protects explicit CLI selection; detects candidate precedence regressions.
    # Needs only a temporary file, never executed; breadcrumb: CLI candidate loop.
    [IO.File]::WriteAllText($fixture, 'not executable; path-selection fixture only')
    $env:PCAI_REDIS_CLI_PATH = $fixture
    $config = Get-PcaiExternalCacheConfig
    if ($config.RedisCliPath -ne (Resolve-Path -LiteralPath $fixture).Path) { throw 'Explicit CLI selection lost' }
    Write-Output 'PASS: explicit CLI path remains first'
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture }
}
