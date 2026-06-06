#Requires -Version 7.0
<#
.SYNOPSIS
    Captures thermal / power / latency evidence during a heavy ML load session to
    diagnose the acute "hard freeze" tier (5/29 cascade: Kernel-Power 41 x6 + WHEA +
    nvlddmkm, no crash dump). Uses native, no-download Windows tooling, and will also
    drive HWiNFO64 sensor-logging and LatencyMon if they are installed.

.DESCRIPTION
    Native captures (always available, MS-documented powercfg):
      - powercfg /energy   : USB selective-suspend + power-efficiency issues report (HTML)
      - powercfg /sleepstudy (Modern Standby) report (HTML)
      - powercfg /batteryreport / /systempowerreport (power-delivery / battery wear)
      - powercfg /requests : what is currently blocking idle (drivers/processes)
    Optional (if installed; this box had NO internet at authoring time):
      - HWiNFO64.exe  : launched in sensor-only mode; configure CSV logging to
                        C:\codedev\PC_AI\Logs\hwinfo\ then re-run Invoke-InputStackDiagnostics
      - LatencyMon    : recommended manual run during load; confirms DPC/ISR spikes from
                        pro-audio (RME/Focusrite/Topping/miniDSP) or nvlddmkm that freeze
                        keyboard+trackpad TOGETHER.

    Install the optional tools when online + elevated:
      winget install REALiX.HWiNFO -e
      winget install Resplendence.LatencyMon -e   # or download from resplendence.com

.PARAMETER OutputPath
    Report directory. Default: C:\codedev\PC_AI\Logs\loadcapture

.PARAMETER EnergyDurationSec
    Seconds for 'powercfg /energy' observation (run while system is under load). Default 60.

.PARAMETER LaunchHwinfo
    If HWiNFO64 is installed, launch it (sensors-only) so you can enable CSV logging.

.EXAMPLE
    # Run elevated, then start your CUDA/ML workload for the duration:
    pwsh -File .\Start-LoadCapture.ps1 -EnergyDurationSec 90 -LaunchHwinfo

.NOTES
    Author: input-stack investigation (Claude Code) - 2026-05-30.
    powercfg /energy and the *report variants need elevation.
#>
[CmdletBinding()]
param(
    [string]$OutputPath = 'C:\codedev\PC_AI\Logs\loadcapture',
    [int]$EnergyDurationSec = 60,
    [switch]$LaunchHwinfo
)

$ErrorActionPreference = 'SilentlyContinue'
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
            ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $elevated) {
    Write-Warning "Not elevated - powercfg /energy and report variants need admin. Re-run elevated:"
    Write-Warning "  Start-Process pwsh -Verb RunAs -ArgumentList '-File','$PSCommandPath'"
}

Write-Host "Capturing power requests (what blocks idle right now)..." -ForegroundColor Yellow
powercfg /requests | Set-Content (Join-Path $OutputPath "requests-$stamp.txt") -Encoding utf8

Write-Host "Generating sleepstudy (Modern Standby) report..." -ForegroundColor Yellow
powercfg /sleepstudy /output (Join-Path $OutputPath "sleepstudy-$stamp.html") | Out-Null

Write-Host "Generating battery / power-delivery report..." -ForegroundColor Yellow
powercfg /batteryreport /output (Join-Path $OutputPath "batteryreport-$stamp.html") | Out-Null
powercfg /systempowerreport /output (Join-Path $OutputPath "systempowerreport-$stamp.html") | Out-Null

# Optional HWiNFO
$hwinfo = 'C:\Program Files\HWiNFO64\HWiNFO64.EXE'
if ($LaunchHwinfo -and (Test-Path $hwinfo)) {
    New-Item -ItemType Directory -Path 'C:\codedev\PC_AI\Logs\hwinfo' -Force | Out-Null
    Write-Host "Launching HWiNFO64 (sensors-only). In Sensors, enable 'Log to CSV' -> C:\codedev\PC_AI\Logs\hwinfo\" -ForegroundColor Cyan
    Start-Process $hwinfo -ArgumentList '-sensors-only'
} elseif ($LaunchHwinfo) {
    Write-Warning "HWiNFO64 not found. Install with: winget install REALiX.HWiNFO -e"
}

Write-Host "Now START YOUR HEAVY ML WORKLOAD, then leave it running..." -ForegroundColor Magenta
Write-Host "Running powercfg /energy for $EnergyDurationSec s (observe under load)..." -ForegroundColor Yellow
$energyOut = Join-Path $OutputPath "energy-$stamp.html"
powercfg /energy /output $energyOut /duration $EnergyDurationSec | Out-Null

Write-Host ""
Write-Host "==== CAPTURE COMPLETE ====" -ForegroundColor Green
Write-Host "Reports in: $OutputPath"
Write-Host "  energy-$stamp.html            (USB selective-suspend + power-efficiency errors/warnings)"
Write-Host "  sleepstudy-$stamp.html        (Modern Standby drains / wake reasons)"
Write-Host "  batteryreport-$stamp.html     (battery wear / capacity history)"
Write-Host "  systempowerreport-$stamp.html (power-delivery, sleep transitions)"
Write-Host "  requests-$stamp.txt           (active power requests blocking idle)"
Write-Host ""
Write-Host "NEXT: For the keyboard+trackpad co-freeze, run LatencyMon during load (~10 min)."
Write-Host "If Highest DPC/ISR is an audio driver (ksthunk/RME/Focusrite .sys), nvlddmkm, or"
Write-Host "Wdf01000, that driver is the co-freeze cause -> update/rollback it specifically."
