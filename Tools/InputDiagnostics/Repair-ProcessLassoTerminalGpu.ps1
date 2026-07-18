#Requires -Version 7.0
<#
.SYNOPSIS
    Optimizes Process Lasso for the eGPU + Terminal + governor conflicts found on
    DTM-P1GEN7:
      1. [ProcessAllowances] EfficiencyMode: remove windowsterminal.exe + pwsh.exe
         (stop pinning the interactive terminal to EcoQoS / E-cores).
      2. [ProcessDefaults] DefaultGPUAdapterPreferences: set ALL entries to auto (0)
         -> removes the external-GPU (Razer Core X V2) forcing on Windows Terminal
         and everything else, so app rendering no longer competes with eGPU compute
         over Thunderbolt. (Auto = Windows renders UI apps on the iGPU and only uses
         the dGPU/eGPU when an app explicitly requests it.)
    Full prolasso.ini backup + governor restart. -WhatIf previews; -Revert restores.

.DESCRIPTION
    Requires elevation (writes C:\ProgramData\ProcessLasso, restarts the governor).
    .ini is UTF-16LE; edits are surgical (EfficiencyMode regex; GPU prefs via
    split/zero/join of the one CSV line). Self-transcripts to Logs\elevated\.

.PARAMETER Revert    Restore prolasso.ini from newest backup + restart governor.
.PARAMETER BackupDir Backup dir. Default: this script's \backups.
.EXAMPLE pwsh -File .\Repair-ProcessLassoTerminalGpu.ps1 -WhatIf
.EXAMPLE pwsh -File .\Repair-ProcessLassoTerminalGpu.ps1
.EXAMPLE pwsh -File .\Repair-ProcessLassoTerminalGpu.ps1 -Revert
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Revert,
    [string]$BackupDir = (Join-Path $PSScriptRoot 'backups')
)
$ErrorActionPreference = 'Stop'
$cfg = 'C:\ProgramData\ProcessLasso\config\prolasso.ini'
$logDir = 'C:\codedev\PC_AI\Logs\elevated'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$mode = if ($Revert) { 'Revert' } elseif ($WhatIfPreference) { 'WhatIf' } else { 'Apply' }
$log = Join-Path $logDir ("plgpu-$mode-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -Path $log -Force | Out-Null

function Test-IsElevated {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}
function Restart-Governor {
    [CmdletBinding(SupportsShouldProcess)]
    param()
    $svc = Get-Service -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'ProcessGovernor|Lasso' } | Select-Object -First 1
    if ($svc) {
        if ($PSCmdlet.ShouldProcess($svc.Name, 'Restart Process Governor')) {
            Restart-Service -Name $svc.Name -Force -ErrorAction SilentlyContinue
            Write-Host "  Restarted service: $($svc.Name)"
        }
    } else { Write-Warning "  Process Governor service not found; restart Process Lasso to reload." }
}

try {
    if (-not (Test-IsElevated)) {
        Write-Host "ERROR: must run elevated. Relaunch:" -ForegroundColor Red
        Write-Host "  Start-Process pwsh -Verb RunAs -ArgumentList '-NoProfile','-File','$PSCommandPath'" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $cfg)) { throw "prolasso.ini not found at $cfg" }
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

    if ($Revert) {
        $bk = Get-ChildItem $BackupDir -Filter 'prolasso.ini.bak-*' -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $bk) { throw "No prolasso.ini backup in $BackupDir" }
        if ($PSCmdlet.ShouldProcess($cfg, "Restore from $($bk.Name)")) {
            Copy-Item $bk.FullName $cfg -Force
            Write-Host "Restored prolasso.ini from $($bk.FullName)" -ForegroundColor Green
            Restart-Governor
        }
        return
    }

    $enc  = [System.Text.Encoding]::Unicode
    $text = [System.IO.File]::ReadAllText($cfg, $enc)
    $orig = $text

    # --- 1) EfficiencyMode: drop windowsterminal.exe + pwsh.exe ---
    $hadTermEco = [bool]($text -match 'windowsterminal\.exe,0,') -or [bool]($text -match 'pwsh\.exe,0,')
    $text = $text -replace 'windowsterminal\.exe,0,', ''
    $text = $text -replace 'pwsh\.exe,0,', ''

    # --- 2) DefaultGPUAdapterPreferences: every pref value -> 0 (auto) ---
    $gpuBefore = ''
    $lines = $text -split "`r`n"
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].StartsWith('DefaultGPUAdapterPreferences=')) {
            $val = $lines[$i].Substring('DefaultGPUAdapterPreferences='.Length)
            $gpuBefore = $val
            if ($val.Trim()) {
                $parts = $val -split ','
                for ($j = 1; $j -lt $parts.Count; $j += 2) { $parts[$j] = '0' }   # pref tokens at odd indices
                $lines[$i] = 'DefaultGPUAdapterPreferences=' + ($parts -join ',')
            }
        }
    }
    $text = $lines -join "`r`n"

    Write-Host "=== Process Lasso Terminal/eGPU optimization ===" -ForegroundColor Cyan
    Write-Host "  EfficiencyMode had terminal/pwsh pinned : $hadTermEco -> removed"
    Write-Host "  GPU prefs (before): $gpuBefore"
    Write-Host "  GPU prefs (after) : all values set to 0 (auto)"
    if ($text -eq $orig) { Write-Host "  No changes needed (already optimized)." -ForegroundColor Green; return }

    if ($PSCmdlet.ShouldProcess($cfg, "Backup + apply EfficiencyMode/GPU optimization")) {
        $bkPath = Join-Path $BackupDir ("prolasso.ini.bak-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Copy-Item $cfg $bkPath -Force
        Write-Host "  Backup: $bkPath"
        [System.IO.File]::WriteAllText($cfg, $text, $enc)
        Write-Host "  prolasso.ini patched." -ForegroundColor Green
        Restart-Governor
        Write-Host "Done. Verify in Process Lasso GUI; -Revert restores the backup." -ForegroundColor Green
    }
}
finally {
    Stop-Transcript | Out-Null
    Set-Content -Path (Join-Path $logDir 'last-run.txt') -Value $log -Encoding utf8
}
