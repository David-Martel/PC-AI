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

.PARAMETER DryRun
Non-mutating preview: report the task registration and Run-key changes without
making them. The long CLI form `--DryRun` is also accepted. Equivalent to
-WhatIf, exposed under the name the repo's session-script contract requires.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Install-CloudClientBootGating.ps1 -WhatIf
.EXAMPLE
pwsh -File .\Tools\Install-CloudClientBootGating.ps1 -DryRun
.EXAMPLE
pwsh -File .\Tools\Install-CloudClientBootGating.ps1 -Rollback
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = 'CloudClients-AfterVHDX',
    [string]$LauncherPath = 'C:\codedev\PC_AI\Tools\Start-CloudClientsAfterVHDX.ps1',
    [string]$BackupPath = 'C:\codedev\PC_AI\Logs\CloudClientStart\runkey-backup.json',
    [switch]$Rollback,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# CLI contract (AGENTS.md). Help is handled before anything else resolves paths
# or touches Task Scheduler, so `-h` works on a machine where this script could
# not actually run.
$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
# Task registration, task execution and every Run-key write are ShouldProcess-
# guarded, so forcing $WhatIfPreference makes -DryRun genuinely non-mutating --
# it cannot register a task, start one, or delete a Run key.
if ($DryRun) { $WhatIfPreference = $true }

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

    $drivePolicy = 'HKLM:\SOFTWARE\Policies\Google\DriveFS'
    if ((Test-Path $drivePolicy) -and (Test-Elevated)) {
        if ($PSCmdlet.ShouldProcess($drivePolicy, 'Remove AutoStartOnLogin policy')) {
            Remove-ItemProperty -Path $drivePolicy -Name 'AutoStartOnLogin' -Force -ErrorAction SilentlyContinue
            Write-Host '  removed AutoStartOnLogin policy (Drive resumes its own autostart)'
        }
    }

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

# Google Drive re-creates its own HKCU Run key every time it starts, so simply
# deleting the value does not stick. The supported way to stop that is the
# AutoStartOnLogin policy. Set under Policies\ so it is an enforced override.
# Ref: https://support.google.com/a/answer/7644837
$drivePolicy = 'HKLM:\SOFTWARE\Policies\Google\DriveFS'
if (Test-Elevated) {
    if ($PSCmdlet.ShouldProcess($drivePolicy, 'Set AutoStartOnLogin=0')) {
        if (-not (Test-Path $drivePolicy)) { New-Item -Path $drivePolicy -Force | Out-Null }
        New-ItemProperty -Path $drivePolicy -Name 'AutoStartOnLogin' -Value 0 -PropertyType DWord -Force | Out-Null
        Write-Host '  policy set: AutoStartOnLogin=0 (Drive will stop re-adding its Run key)' -ForegroundColor Green
    }
} else {
    Write-Warning '  AutoStartOnLogin policy needs elevation - Google Drive will keep re-adding its Run key.'
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

# MERGE with any existing backup. Re-running the installer finds already-removed
# keys "not present", so a plain overwrite would silently drop entries captured by
# an earlier run and leave -Rollback unable to restore them.
if ($PSCmdlet.ShouldProcess($BackupPath, 'Write/merge Run key backup')) {
    $dir = Split-Path $BackupPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $merged = @{}
    if (Test-Path $BackupPath) {
        $prior = Get-Content $BackupPath -Raw | ConvertFrom-Json
        foreach ($e in @($prior)) { if ($e) { $merged["$($e.Hive)|$($e.Name)"] = $e } }
    }
    foreach ($e in $backup) { $merged["$($e.Hive)|$($e.Name)"] = $e }

    if ($merged.Count -gt 0) {
        @($merged.Values) | ConvertTo-Json -Depth 4 -AsArray |
            Set-Content -LiteralPath $BackupPath -Encoding UTF8
        Write-Host "  backup written: $BackupPath ($($merged.Count) entr$(if($merged.Count -eq 1){'y'}else{'ies'}))"
    }
}

Write-Host ''
Write-Host 'Install complete.' -ForegroundColor Green
Write-Host "  Clients now start via task '$TaskName' at logon, gated on the cloud-cache-disk volume."
Write-Host "  Undo with: pwsh -File `"$PSCommandPath`" -Rollback"
Write-Host '  NOTE: Dropbox/Google updates may re-add their Run keys; the launcher is idempotent'
Write-Host '        (it skips already-running clients), so a returning key is harmless but'
Write-Host '        re-removing it restores full gating.'
