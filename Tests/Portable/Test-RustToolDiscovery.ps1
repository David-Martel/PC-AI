#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1'))
$ErrorActionPreference = 'Stop'
. $SourcePath
$savedProfile = $env:USERPROFILE; $savedRoot = $env:PCAI_ROOT; $savedPath = $env:PATH
$fixture = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
try {
    $env:USERPROFILE = $null; $env:PCAI_ROOT = $null
    $script:RustToolCache = @{}; $script:ToolPaths = @{}; $script:SearchPaths = @()
    New-Item -ItemType Directory -Path $fixture | Out-Null
    $name = 'pcai-discovery-fixture'
    $leaf = if ($IsWindows) { "$name.exe" } else { $name }
    $expected = Join-Path $fixture $leaf
    [IO.File]::WriteAllText($expected, 'discovery fixture; never execute')
    $env:PATH = $fixture + [IO.Path]::PathSeparator + $savedPath
    # Protects native PATH and suffix selection; detects null roots, Unix semicolon
    # splitting and forced .exe. Needs pwsh, no external tool; breadcrumb: Find-RustTool.
    $actual = Find-RustTool -ToolName $name
    if ($actual -ne $expected) { throw "Expected exact PATH fixture, got '$actual'" }
    Write-Output 'PASS: host PATH discovery works without Windows roots'
    # Protects absent-tool reporting; detects invented availability.
    # Needs isolated improbable name; breadcrumb: Find-RustTool fallback.
    if ($null -ne (Find-RustTool -ToolName 'pcai-deliberately-missing-tool-fixture')) { throw 'Missing tool reported present' }
    Write-Output 'PASS: absent tool remains unavailable'
} finally {
    $env:USERPROFILE=$savedProfile; $env:PCAI_ROOT=$savedRoot; $env:PATH=$savedPath
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse }
}
