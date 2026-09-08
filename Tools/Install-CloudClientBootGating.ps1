#Requires -Version 5.1
<#
.SYNOPSIS
Installs logon-time gating so cloud sync clients start only after F: is mounted.

.DESCRIPTION
Registers a logon-triggered scheduled task that runs
Start-CloudClientsAfterVHDX.ps1, then removes the clients' own autostart Run
keys so they cannot race the VHDX mount.

Ordering is deliberate and safe:
  1. Register the task.
  2. Run it once and require success.
  3. Only then remove the Run keys (backing up their values first).
If step 2 fails the Run keys are left untouched, so the clients still autostart
as they do today. The install can never leave the machine with no start path.

Use -Rollback to restore the Run keys and remove the task.

.PARAMETER Rollback
Undo: restore Run keys from the backup and unregister the task.

.EXAMPLE
pwsh -File .\Tools\Install-CloudClientBootGating.ps1 -WhatIf
.EXAMPLE
pwsh -File .\Tools\Install-CloudClientBootGating.ps1 -Rollback
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = 'CloudClients-AfterVHDX',
    [string]$LauncherPath = 'C:\codedev\PC_AI\Tools\Start-CloudClientsAfterVHDX.ps1',
    [string]$BackupPath = 'C:\codedev\PC_AI\Logs\CloudClientStart\runkey-backup.json',
    [switch]$Rollback
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$pwsh = 'C:\Program Files\PowerShell\7\pwsh.exe'
if (-not (Test-Path $pwsh)) { $pwsh = (Get-Command powershell.exe).Source }

# Autostart entries that must not race the mount.
# NOTE: Dropbox is a 32-bit app, so its machine Run key lives under WOW6432Node.
# A 64-bit PowerShell does NOT see it at the plain HKLM Run path.
$runKeys = @(
    @{ Hive = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run'; Name = 'GoogleDriveFS' },
    @{ Hive = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Run'; Name = 'Dropbox' },
    @{ Hive = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run'; Name = 'Dropbox' }
)

function Test-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ---------------------------------------------------------------- ROLLBACK ---
if ($Rollback) {
    Write-Host '=== ROLLBACK ===' -ForegroundColor Yellow
    if (Test-Path $BackupPath) {
        $backup = Get-Content $BackupPath -Raw | ConvertFrom-Json
        foreach ($e in $backup) {
            if ($PSCmdlet.ShouldProcess("$($e.Hive)\$($e.Name)", 'Restore Run key')) {
                if (-not (Test-Path $e.Hive)) { New-Item -Path $e.Hive -Force | Out-Null }
                New-ItemProperty -Path $e.Hive -Name $e.Name -Value $e.Value -PropertyType String -Force | Out-Null
                Write-Host "  restored $($e.Name) = $($e.Value)"
            }
        }
    } else { Write-Warning "No backup at $BackupPath - Run keys not restored." }

    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($TaskName, 'Unregister task')) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "  unregistered task $TaskName"
        }
    }
    Write-Host 'Rollback complete.' -ForegroundColor Green
    return
}

# ----------------------------------------------------------------- INSTALL ---
if (-not (Test-Path $LauncherPath)) { throw "Launcher not found: $LauncherPath" }

Write-Host '=== STEP 1: register logon task ===' -ForegroundColor Cyan
$action = New-ScheduledTaskAction -Execute $pwsh `
    -Argument ('-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -RequireMountLog' -f $LauncherPath)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

if ($PSCmdlet.ShouldProcess($TaskName, 'Register scheduled task')) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings `
        -Description 'Starts Dropbox and Google Drive only after the F: cloud-cache-disk VHDX is mounted.' `
        -Force | Out-Null
    Write-Host "  registered $TaskName" -ForegroundColor Green
}

Write-Host '=== STEP 2: prove the task runs before touching Run keys ===' -ForegroundColor Cyan
$proved = $false
if ($PSCmdlet.ShouldProcess($TaskName, 'Start task once and verify')) {
    Start-ScheduledTask -TaskName $TaskName
    $deadline = (Get-Date).AddSeconds(240)
    do {
        Start-Sleep -Seconds 5
        $info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
        $state = (Get-ScheduledTask -TaskName $TaskName).State
    } while ($state -eq 'Running' -and (Get-Date) -lt $deadline)

    Write-Host ("  state={0} lastResult={1}" -f $state, $info.LastTaskResult)
    if ($info.LastTaskResult -eq 0) { $proved = $true; Write-Host '  task verified OK' -ForegroundColor Green }
    else { Write-Warning "  task returned $($info.LastTaskResult) - Run keys will NOT be removed." }
} else { $proved = $true }  # WhatIf: show intended step 3

Write-Host '=== STEP 3: remove racing Run keys (backed up first) ===' -ForegroundColor Cyan
if (-not $proved) {
    Write-Warning 'Skipped - task did not verify. Clients keep their existing autostart.'
    return
}

$backup = @()
foreach ($rk in $runKeys) {
    $existing = Get-ItemProperty -Path $rk.Hive -Name $rk.Name -ErrorAction SilentlyContinue
    if (-not $existing) { Write-Host "  $($rk.Name): not present - nothing to do"; continue }
    $val = $existing.$($rk.Name)
    $backup += [pscustomobject]@{ Hive = $rk.Hive; Name = $rk.Name; Value = $val }

    if ($rk.Hive -like 'HKLM*' -and -not (Test-Elevated)) {
        Write-Warning "  $($rk.Name): HKLM needs elevation - left in place. Re-run elevated."
        continue
    }
    if ($PSCmdlet.ShouldProcess("$($rk.Hive)\$($rk.Name)", 'Remove Run key')) {
        Remove-ItemProperty -Path $rk.Hive -Name $rk.Name -Force
        Write-Host "  removed $($rk.Name)" -ForegroundColor Green
    }
}

if ($backup.Count -gt 0 -and $PSCmdlet.ShouldProcess($BackupPath, 'Write Run key backup')) {
    $dir = Split-Path $BackupPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $backup | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $BackupPath -Encoding UTF8
    Write-Host "  backup written: $BackupPath"
}

Write-Host ''
Write-Host 'Install complete.' -ForegroundColor Green
Write-Host "  Clients now start via task '$TaskName' at logon, gated on the cloud-cache-disk volume."
Write-Host "  Undo with: pwsh -File `"$PSCommandPath`" -Rollback"
Write-Host '  NOTE: Dropbox/Google updates may re-add their Run keys; the launcher is idempotent'
Write-Host '        (it skips already-running clients), so a returning key is harmless but'
Write-Host '        re-removing it restores full gating.'
