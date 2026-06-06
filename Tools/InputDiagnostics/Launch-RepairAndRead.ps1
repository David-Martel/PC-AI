#Requires -Version 7.0
<#
.SYNOPSIS
    Non-elevated launcher: starts Repair-WorkstationInputReliability.ps1 elevated (via
    Invoke-RepairElevatedLogged.ps1 + UAC), waits for completion, prints the elevated log.
.PARAMETER Mode
    WhatIf (default) | Apply | Revert
.PARAMETER WaitSeconds
    How long to wait for UAC approval + completion. Default 120.
#>
[CmdletBinding()]
param([ValidateSet('WhatIf','Apply','Revert')][string]$Mode='WhatIf',[int]$WaitSeconds=120)

$wrapper = 'C:\codedev\PC_AI\Tools\InputDiagnostics\Invoke-RepairElevatedLogged.ps1'
$marker  = 'C:\codedev\PC_AI\Logs\elevated\last-run.txt'
New-Item -ItemType Directory -Path (Split-Path $marker) -Force | Out-Null
Remove-Item $marker -ErrorAction SilentlyContinue

try {
    Start-Process pwsh -Verb RunAs -ArgumentList '-NoProfile','-File',$wrapper,'-Mode',$Mode -ErrorAction Stop
} catch {
    Write-Host "Could not launch elevated process: $_" -ForegroundColor Red
    Write-Host "Run manually from an elevated PowerShell:`n  pwsh -NoProfile -File `"$wrapper`" -Mode $Mode" -ForegroundColor Yellow
    return
}

Write-Host ">>> APPROVE THE UAC PROMPT on your screen (Mode=$Mode). Waiting up to $WaitSeconds s..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline -and -not (Test-Path $marker)) { Start-Sleep -Seconds 2 }

if (Test-Path $marker) {
    $log = (Get-Content $marker -Raw).Trim()
    Write-Host "`n================ ELEVATED LOG: $log ================" -ForegroundColor Green
    Get-Content $log
} else {
    Write-Host "No completion detected yet (UAC not approved or still running)." -ForegroundColor Yellow
    Write-Host "After approving, view the log with: Get-ChildItem C:\codedev\PC_AI\Logs\elevated\ | sort LastWriteTime | select -last 1 | Get-Content"
}
