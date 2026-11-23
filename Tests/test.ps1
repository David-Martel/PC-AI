<#
.SYNOPSIS
    Universal Test Runner for PC-AI
    Automates environment setup (Path, Modules) before executing tests.

.DESCRIPTION
    Sets the process PATH and loads PC-AI.Acceleration, then runs Pester and exits 1 unless the run is clean.
    Failed tests, failed containers (a test file that does not parse) and a run that executed nothing all fail,
    see Tests/Helpers/Get-PesterRunVerdict.ps1. Before this, the script printed "Test Execution Complete" in green
    and never set an exit code, whatever Pester reported.
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

. (Join-Path $PesterRoot 'Helpers/Get-PesterRunVerdict.ps1')

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
$result = Invoke-Pester -Path $TestPath -Output Detailed -PassThru

# 5. Judge the run
$verdict = Get-PesterRunVerdict -Result $result
if ($verdict.Passed) {
    Write-Host "`nTest Execution Complete." -ForegroundColor Green
    exit 0
}
foreach ($reason in $verdict.Violations) { Write-Host "  $reason" -ForegroundColor Red }
Write-Host "`nTests FAILED." -ForegroundColor Red
exit 1
