<#
.SYNOPSIS
    Universal Test Runner for PC-AI
    Automates environment setup (Path, Modules) before executing tests.
#>

[CmdletBinding()]
param(
    [Parameter()]
    [switch]$IntegrationOnly,
    [Parameter()]
    [switch]$UnitOnly
)

$PesterRoot = $PSScriptRoot
$PcaiRoot = Split-Path $PesterRoot -Parent
$BinPath = Join-Path $PcaiRoot "bin"
$ModulePath = Join-Path $PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1"

Write-Host "--- PC-AI Test Orchestration ---" -ForegroundColor Cyan
Write-Host "Root: $PcaiRoot"

# 1. Configure Process Path
if ($env:PATH -notlike "*$BinPath*") {
    Write-Host "Updating Process PATH with: $BinPath" -ForegroundColor Gray
    $env:PATH = "$BinPath;$env:PATH"
}

# 2. Load Core Acceleration Module
if (Test-Path $ModulePath) {
    Write-Host "Loading PC-AI.Acceleration module..." -ForegroundColor Gray
    Import-Module $ModulePath -Force
} else {
    Write-Warning "PC-AI.Acceleration module not found at $ModulePath"
}

# 3. Determine Test Path
$TestPath = $PesterRoot
if ($IntegrationOnly) { $TestPath = Join-Path $PesterRoot "Integration" }
if ($UnitOnly) { $TestPath = Join-Path $PesterRoot "Unit" }

# 4. Execute Pester
Write-Host "Executing Pester tests in: $TestPath`n" -ForegroundColor Cyan
Invoke-Pester -Path $TestPath -Output Detailed

Write-Host "`nTest Execution Complete." -ForegroundColor Green
