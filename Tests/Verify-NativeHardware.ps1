. (Join-Path $PSScriptRoot 'Helpers\Resolve-TestRepoRoot.ps1')

# Requires -Version 5.1

$ErrorActionPreference = 'Stop'

# Path to the main entry point
$repoRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$pcai = Join-Path $repoRoot 'PC-AI.ps1'

Write-Host '--- PC-AI Native Hardware Diagnostics Verification ---' -ForegroundColor Cyan

# 1. Verify Disk Health
Write-Host "`n1. Benchmarking Disk Health (Native via PC-AI)..." -ForegroundColor Yellow
$sw = [Diagnostics.Stopwatch]::StartNew()
& $pcai diagnose hardware | Out-Null
$sw.Stop()
$nativeTime = $sw.Elapsed.TotalMilliseconds
Write-Host "PC-AI diagnose hardware took $nativeTime ms"

# 2. Verify System Events
Write-Host "`n2. Benchmarking System Events (Native via PC-AI)..." -ForegroundColor Yellow
$sw.Restart()
& $pcai diagnose events -days 3 | Out-Null
$sw.Stop()
$nativeTime = $sw.Elapsed.TotalMilliseconds
Write-Host "PC-AI diagnose events took $nativeTime ms"

# 3. Verify Full Hardware Report
Write-Host "`n3. Benchmarking Full Hardware Report..." -ForegroundColor Yellow
$sw.Restart()
& $pcai diagnose all | Out-Null
$sw.Stop()
$nativeTime = $sw.Elapsed.TotalMilliseconds
Write-Host "PC-AI diagnose all took $nativeTime ms"

Write-Host "`nVerification Complete." -ForegroundColor Green
