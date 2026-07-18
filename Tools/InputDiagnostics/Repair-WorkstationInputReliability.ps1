#Requires -Version 7.0
<#
.SYNOPSIS
    Remediates chronic input-stack glitches on a Lenovo ThinkPad P1 Gen 7 (Win 11, 64 GB RAM).

.DESCRIPTION
    Applies admin-gated, idempotent fixes for Shift-key / trackpad freeze and fingerprint
    reader wake failures.  Before every change, the prior state is written to a timestamped
    JSON backup under .\backups\.  Rerun freely; each action is guarded and idempotent.

    Actions (all individually skippable):
      1  USB Selective Suspend OFF (AC + DC) via powercfg               (-SkipUsbSuspend)
      2  Crash dump = Automatic kernel dump                              (-SkipCrashDump)
      3  Per-device USB power management OFF for fingerprint/hubs/BT    (-SkipDevicePower)
      4  Disable boot-churn services PC_AI-HVSockProxy and vtss         (-SkipServices)
      5  (Opt-in) Switch to High Performance power plan                 (-SetHighPerformancePowerPlan)

    Use -Revert to restore all values from the newest backup file
    (or a specific file with -BackupFile).

.PARAMETER Revert
    Restore all backed-up values from the newest backup JSON (or -BackupFile).

.PARAMETER BackupFile
    Full path to a specific backup JSON to revert from.  Overrides newest-backup selection.

.PARAMETER SkipUsbSuspend
    Skip Action 1: USB Selective Suspend power policy changes.

.PARAMETER SkipCrashDump
    Skip Action 2: CrashControl registry changes.

.PARAMETER SkipDevicePower
    Skip Action 3: Per-device USB Enhanced Power Management registry changes.

.PARAMETER SkipServices
    Skip Action 4: Disabling boot-churn services.

.PARAMETER SetHighPerformancePowerPlan
    (OFF by default) Switch the active power plan to High Performance.
    WARNING: Process Lasso may revert this at next boot if StartWithPowerPlan=Balanced
    is configured in C:\ProgramData\ProcessLasso\config\prolasso.ini.

.EXAMPLE
    # Preview every change without applying anything
    pwsh -NoLogo -File "C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1" -WhatIf

.EXAMPLE
    # Apply all default actions
    pwsh -NoLogo -File "C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1"

.EXAMPLE
    # Apply with High Performance power plan
    pwsh -NoLogo -File "C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1" -SetHighPerformancePowerPlan

.EXAMPLE
    # Revert from the newest backup
    pwsh -NoLogo -File "C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1" -Revert

.EXAMPLE
    # Revert from a specific backup
    pwsh -NoLogo -File "C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1" -Revert -BackupFile "C:\codedev\PC_AI\Tools\InputDiagnostics\backups\backup_20260530T143000.json"

.NOTES
    Requires PowerShell 7.0+ and elevation (Administrator).
    Tested against Windows 11 Build 26200, Lenovo ThinkPad P1 Gen 7.
    Sign out / sign in after running to allow Bluetooth and HID driver re-init.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
param(
    [switch]$Revert,
    [string]$BackupFile,
    [switch]$SkipUsbSuspend,
    [switch]$SkipCrashDump,
    [switch]$SkipDevicePower,
    [switch]$SkipServices,
    [switch]$SetHighPerformancePowerPlan
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 0. Elevation guard
# ---------------------------------------------------------------------------
$currentPrincipal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    $scriptPath = $PSCommandPath
    Write-Host "ERROR: This script must run as Administrator." -ForegroundColor Red
    Write-Host ""
    Write-Host "Run the following command from an elevated PowerShell 7 prompt:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  pwsh -NoLogo -File `"$scriptPath`"" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To add optional parameters, append them, e.g.:" -ForegroundColor Yellow
    Write-Host "  pwsh -NoLogo -File `"$scriptPath`" -WhatIf" -ForegroundColor Cyan
    Write-Host "  pwsh -NoLogo -File `"$scriptPath`" -Revert" -ForegroundColor Cyan
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Paths and backup infrastructure
# ---------------------------------------------------------------------------
$scriptDir  = Split-Path -Parent $PSCommandPath
$backupDir  = Join-Path $scriptDir 'backups'
$timestamp  = Get-Date -Format 'yyyyMMddTHHmmss'
$backupPath = Join-Path $backupDir "backup_$timestamp.json"

if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    Write-Verbose "Created backup directory: $backupDir"
}

# Central backup object — written atomically at the end of the forward pass
$backup = [ordered]@{
    Timestamp = $timestamp
    ScriptVersion = '1.0.0'
    Actions   = [ordered]@{}
}

# Summary collection
$summary = [System.Collections.Generic.List[object]]::new()

function Add-Summary {
    param([string]$Action, [string]$Status, [string]$Detail)
    $summary.Add([PSCustomObject]@{
        Action = $Action
        Status = $Status
        Detail = $Detail
    })
}

# ---------------------------------------------------------------------------
# 2. Helper: run external command with ShouldProcess guard
# ---------------------------------------------------------------------------
function Invoke-ExternalIfShouldProcess {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$Target,
        [string]$Operation,
        [scriptblock]$Command
    )
    if ($PSCmdlet.ShouldProcess($Target, $Operation)) {
        & $Command
        return $true
    }
    return $false
}

# ---------------------------------------------------------------------------
# 3. REVERT MODE
# ---------------------------------------------------------------------------
if ($Revert) {
    Write-Host "`n=== REVERT MODE ===" -ForegroundColor Cyan

    # Resolve backup file
    if ($BackupFile) {
        $resolvedBackup = $BackupFile
    } else {
        $candidates = Get-ChildItem -Path $backupDir -Filter 'backup_*.json' -File |
                      Sort-Object LastWriteTime -Descending
        if (-not $candidates) {
            Write-Error "No backup files found in $backupDir. Cannot revert."
            exit 1
        }
        $resolvedBackup = $candidates[0].FullName
    }

    if (-not (Test-Path $resolvedBackup)) {
        Write-Error "Backup file not found: $resolvedBackup"
        exit 1
    }

    Write-Host "Reverting from: $resolvedBackup" -ForegroundColor Yellow
    $bk = Get-Content $resolvedBackup -Raw | ConvertFrom-Json

    $revertErrors = $false

    # --- Revert Action 1: USB Selective Suspend ---
    if ($bk.Actions.UsbSuspend) {
        $u = $bk.Actions.UsbSuspend
        Write-Host "`n[Revert] USB Selective Suspend power indexes..." -ForegroundColor Yellow
        try {
            $sub  = '2a737441-1930-4402-8d77-b2bebba308a3'
            $set  = '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'
            $acOld = $u.AcIndexHex -replace '^0x','' | ForEach-Object { [Convert]::ToInt32($_, 16) }
            $dcOld = $u.DcIndexHex -replace '^0x','' | ForEach-Object { [Convert]::ToInt32($_, 16) }
            if ($PSCmdlet.ShouldProcess('powercfg AC USB suspend index', "Restore to $acOld")) {
                powercfg /setacvalueindex SCHEME_CURRENT $sub $set $acOld
                powercfg /setdcvalueindex SCHEME_CURRENT $sub $set $dcOld
                powercfg /setactive SCHEME_CURRENT
            }
            Write-Host "  Restored AC=$acOld DC=$dcOld" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to revert UsbSuspend: $_"
            $revertErrors = $true
        }
    }

    # --- Revert Action 2: CrashControl ---
    if ($bk.Actions.CrashDump) {
        Write-Host "`n[Revert] CrashControl registry values..." -ForegroundColor Yellow
        $crashKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'
        foreach ($entry in $bk.Actions.CrashDump) {
            try {
                if ($entry.ExistedBefore) {
                    if ($PSCmdlet.ShouldProcess("$crashKey\$($entry.Name)", "Restore to $($entry.PriorValue)")) {
                        Set-ItemProperty -Path $crashKey -Name $entry.Name -Value $entry.PriorValue -Type DWord
                    }
                    Write-Host "  Restored $($entry.Name) = $($entry.PriorValue)" -ForegroundColor Green
                } else {
                    if ($PSCmdlet.ShouldProcess("$crashKey\$($entry.Name)", 'Remove (was created by this script)')) {
                        Remove-ItemProperty -Path $crashKey -Name $entry.Name -ErrorAction SilentlyContinue
                    }
                    Write-Host "  Removed $($entry.Name) (did not exist before)" -ForegroundColor Green
                }
            } catch {
                Write-Warning "Failed to revert CrashDump.$($entry.Name): $_"
                $revertErrors = $true
            }
        }
    }

    # --- Revert Action 3: Per-device power ---
    if ($bk.Actions.DevicePower) {
        Write-Host "`n[Revert] Per-device power management registry values..." -ForegroundColor Yellow
        foreach ($deviceEntry in $bk.Actions.DevicePower) {
            $devPath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($deviceEntry.InstanceId)\Device Parameters"
            Write-Host "  Device: $($deviceEntry.FriendlyName)" -ForegroundColor Yellow
            foreach ($valEntry in $deviceEntry.Values) {
                try {
                    if ($valEntry.ExistedBefore) {
                        if ($PSCmdlet.ShouldProcess("$devPath\$($valEntry.Name)", "Restore to $($valEntry.PriorValue)")) {
                            Set-ItemProperty -Path $devPath -Name $valEntry.Name -Value $valEntry.PriorValue -Type DWord
                        }
                        Write-Host "    Restored $($valEntry.Name) = $($valEntry.PriorValue)" -ForegroundColor Green
                    } else {
                        if ($PSCmdlet.ShouldProcess("$devPath\$($valEntry.Name)", 'Remove (was created by this script)')) {
                            Remove-ItemProperty -Path $devPath -Name $valEntry.Name -ErrorAction SilentlyContinue
                        }
                        Write-Host "    Removed $($valEntry.Name) (did not exist before)" -ForegroundColor Green
                    }
                } catch {
                    Write-Warning "    Failed to revert $($valEntry.Name) on $($deviceEntry.InstanceId): $_"
                    $revertErrors = $true
                }
            }
        }
    }

    # --- Revert Action 4: Services ---
    if ($bk.Actions.Services) {
        Write-Host "`n[Revert] Service startup types..." -ForegroundColor Yellow
        foreach ($svcEntry in $bk.Actions.Services) {
            try {
                $svc = Get-Service -Name $svcEntry.Name -ErrorAction SilentlyContinue
                if ($svc) {
                    $startType = $svcEntry.PriorStartType
                    if ($PSCmdlet.ShouldProcess($svcEntry.Name, "Restore StartupType to $startType")) {
                        try {
                            Set-Service -Name $svcEntry.Name -StartupType $startType
                        } catch {
                            sc.exe config $svcEntry.Name start= $startType.ToLower() | Out-Null
                        }
                    }
                    Write-Host "  Restored $($svcEntry.Name) StartupType = $startType" -ForegroundColor Green
                } else {
                    Write-Host "  Service '$($svcEntry.Name)' not present; skip." -ForegroundColor DarkGray
                }
            } catch {
                Write-Warning "Failed to revert service $($svcEntry.Name): $_"
                $revertErrors = $true
            }
        }
    }

    Write-Host "`n=== Revert complete. Sign out/in to allow driver re-init. ===" -ForegroundColor Cyan
    exit $(if ($revertErrors) { 1 } else { 0 })
}

# ---------------------------------------------------------------------------
# 4. FORWARD PASS — Action implementations
# ---------------------------------------------------------------------------
Write-Host "`n=== Repair-WorkstationInputReliability (forward pass) ===" -ForegroundColor Cyan
Write-Host "  Backup will be written to: $backupPath" -ForegroundColor DarkGray

$anyFailure = $false

# ============================================================
# Action 1: USB Selective Suspend OFF
# Ref: https://learn.microsoft.com/troubleshoot/microsoftteams/teams-rooms-and-devices/usb-selective-suspend-status-unhealthy
# Power subgroup : 2a737441-1930-4402-8d77-b2bebba308a3  (USB settings)
# Power setting  : 48e6b7a6-50f5-4782-a5d4-53bb8f07e226  (USB selective suspend)
# Value 0 = Disabled, 1 = Enabled
# ============================================================
if (-not $SkipUsbSuspend) {
    Write-Host "`n[Action 1] USB Selective Suspend" -ForegroundColor Magenta
    try {
        $sub = '2a737441-1930-4402-8d77-b2bebba308a3'
        $set = '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'

        # Read current AC and DC values via powercfg /q
        $qOutput = powercfg /q SCHEME_CURRENT $sub $set 2>&1
        $acLine  = $qOutput | Select-String 'Current AC Power Setting Index'
        $dcLine  = $qOutput | Select-String 'Current DC Power Setting Index'
        $acHex   = if ($acLine) { ($acLine -split ':')[-1].Trim() } else { '0x00000001' }
        $dcHex   = if ($dcLine) { ($dcLine -split ':')[-1].Trim() } else { '0x00000001' }

        Write-Host "  Before: AC index = $acHex  DC index = $dcHex"

        $backup.Actions['UsbSuspend'] = [ordered]@{
            AcIndexHex = $acHex
            DcIndexHex = $dcHex
        }

        if ($PSCmdlet.ShouldProcess('SCHEME_CURRENT USB Selective Suspend (AC)', 'Set to 0 (Disabled)')) {
            powercfg /setacvalueindex SCHEME_CURRENT $sub $set 0
        }
        if ($PSCmdlet.ShouldProcess('SCHEME_CURRENT USB Selective Suspend (DC)', 'Set to 0 (Disabled)')) {
            powercfg /setdcvalueindex SCHEME_CURRENT $sub $set 0
        }
        if ($PSCmdlet.ShouldProcess('SCHEME_CURRENT', 'Set as active plan (commit changes)')) {
            powercfg /setactive SCHEME_CURRENT
        }

        Write-Host "  After:  AC index = 0x00000000  DC index = 0x00000000" -ForegroundColor Green
        Add-Summary 'UsbSuspend' 'OK' "AC: $acHex -> 0x00000000 | DC: $dcHex -> 0x00000000"
    } catch {
        Write-Warning "Action 1 (UsbSuspend) failed: $_"
        Add-Summary 'UsbSuspend' 'FAILED' $_.ToString()
        $anyFailure = $true
    }
} else {
    Write-Host "`n[Action 1] USB Selective Suspend — SKIPPED (-SkipUsbSuspend)" -ForegroundColor DarkGray
    Add-Summary 'UsbSuspend' 'SKIPPED' 'User requested skip'
}

# ============================================================
# Action 2: Crash dump = Automatic kernel dump
# Ref: https://learn.microsoft.com/troubleshoot/windows-server/performance/troubleshoot-stop-errors-best-practices-dump-configuration-recommendations
# CrashDumpEnabled: 7 = Automatic memory dump
# AutoReboot     : 1 = On
# ============================================================
if (-not $SkipCrashDump) {
    Write-Host "`n[Action 2] Crash Dump Configuration" -ForegroundColor Magenta
    try {
        $crashKey    = 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'
        $targetValues = @{
            CrashDumpEnabled = 7
            AutoReboot       = 1
        }

        $crashBackup = [System.Collections.Generic.List[object]]::new()

        foreach ($name in $targetValues.Keys) {
            $target = $targetValues[$name]
            try {
                $existing = Get-ItemProperty -Path $crashKey -Name $name -ErrorAction SilentlyContinue
                if ($null -ne $existing -and $null -ne $existing.$name) {
                    $priorVal   = $existing.$name
                    $didExist   = $true
                } else {
                    $priorVal = $null
                    $didExist = $false
                }
            } catch {
                $priorVal = $null
                $didExist = $false
            }

            $crashBackup.Add([ordered]@{
                Name         = $name
                PriorValue   = $priorVal
                ExistedBefore = $didExist
            })

            Write-Host "  $name  Before: $(if ($didExist) { $priorVal } else { '(not set)' })  ->  After: $target"

            if ($PSCmdlet.ShouldProcess("$crashKey\$name", "Set DWord = $target")) {
                Set-ItemProperty -Path $crashKey -Name $name -Value $target -Type DWord
            }
        }

        $backup.Actions['CrashDump'] = $crashBackup
        Write-Host "  CrashControl updated." -ForegroundColor Green
        Add-Summary 'CrashDump' 'OK' 'CrashDumpEnabled=7 AutoReboot=1'
    } catch {
        Write-Warning "Action 2 (CrashDump) failed: $_"
        Add-Summary 'CrashDump' 'FAILED' $_.ToString()
        $anyFailure = $true
    }
} else {
    Write-Host "`n[Action 2] Crash Dump — SKIPPED (-SkipCrashDump)" -ForegroundColor DarkGray
    Add-Summary 'CrashDump' 'SKIPPED' 'User requested skip'
}

# ============================================================
# Action 3: Per-device USB power management
# Targets:
#   - Synaptics fingerprint reader : USB\VID_06CB&PID_00F9*
#   - USB Root Hubs                : USB\ROOT_HUB* / USB\VID_*ROOT*
#   - Bluetooth radio              : Get-PnpDevice -Class Bluetooth (FriendlyName ~ radio)
# Registry path: HKLM:\SYSTEM\CurrentControlSet\Enum\<InstanceId>\Device Parameters
# EnhancedPowerManagementEnabled = 0  (always write/create)
# SelectiveSuspendEnabled        = 0  (write ONLY if value already exists)
# AllowIdleIrpInD3               = 0  (write ONLY if value already exists)
# ============================================================
if (-not $SkipDevicePower) {
    Write-Host "`n[Action 3] Per-device USB Power Management" -ForegroundColor Magenta
    try {
        # Collect target devices
        $targetDevices = [System.Collections.Generic.List[object]]::new()

        # Fingerprint reader
        try {
            $fps = Get-PnpDevice -InstanceId 'USB\VID_06CB&PID_00F9*' -ErrorAction SilentlyContinue |
                   Where-Object { $_.Present }
            foreach ($d in $fps) { $targetDevices.Add($d) }
        } catch { Write-Verbose "Fingerprint query failed: $_" }

        # USB Root Hubs
        try {
            $hubs = Get-PnpDevice -ErrorAction SilentlyContinue |
                    Where-Object {
                        $_.Present -and (
                            $_.InstanceId -match '^USB\\ROOT_HUB' -or
                            $_.InstanceId -match '^USB\\VID_.+ROOT'
                        )
                    }
            foreach ($d in $hubs) { $targetDevices.Add($d) }
        } catch { Write-Verbose "USB Root Hub query failed: $_" }

        # Bluetooth radio
        try {
            $btDevices = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue |
                         Where-Object { $_.Present -and $_.FriendlyName -match 'radio' }
            foreach ($d in $btDevices) { $targetDevices.Add($d) }
        } catch { Write-Verbose "Bluetooth radio query failed: $_" }

        if ($targetDevices.Count -eq 0) {
            Write-Host "  No matching present devices found; skipping." -ForegroundColor DarkYellow
            Add-Summary 'DevicePower' 'NODEV' 'No matching present devices found'
        } else {
            $devPowerBackup = [System.Collections.Generic.List[object]]::new()

            foreach ($dev in $targetDevices) {
                $instId  = $dev.InstanceId
                $devPath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$instId\Device Parameters"

                Write-Host "  Device: $($dev.FriendlyName)  [$instId]" -ForegroundColor Yellow

                if (-not (Test-Path $devPath)) {
                    Write-Host "    'Device Parameters' key not found; skipping." -ForegroundColor DarkGray
                    continue
                }

                $deviceValuesBackup = [System.Collections.Generic.List[object]]::new()

                # EnhancedPowerManagementEnabled: always write (create if absent)
                $valName = 'EnhancedPowerManagementEnabled'
                try {
                    $existingProp = Get-ItemProperty -Path $devPath -Name $valName -ErrorAction SilentlyContinue
                    $existed = ($null -ne $existingProp -and $null -ne $existingProp.$valName)
                    $priorVal = if ($existed) { $existingProp.$valName } else { $null }

                    Write-Host "    $valName  Before: $(if ($existed) { $priorVal } else { '(not set)' })  ->  After: 0"

                    if ($PSCmdlet.ShouldProcess("$devPath\$valName", 'Set DWord = 0')) {
                        Set-ItemProperty -Path $devPath -Name $valName -Value 0 -Type DWord
                    }
                    $deviceValuesBackup.Add([ordered]@{
                        Name          = $valName
                        PriorValue    = $priorVal
                        ExistedBefore = $existed
                    })
                } catch {
                    Write-Warning "    Failed to set $valName on $instId : $_"
                }

                # SelectiveSuspendEnabled and AllowIdleIrpInD3: write ONLY if value already exists
                foreach ($conditionalVal in @('SelectiveSuspendEnabled', 'AllowIdleIrpInD3')) {
                    try {
                        $existingProp = Get-ItemProperty -Path $devPath -Name $conditionalVal -ErrorAction SilentlyContinue
                        $existed = ($null -ne $existingProp -and $null -ne $existingProp.$conditionalVal)

                        if ($existed) {
                            $priorVal = $existingProp.$conditionalVal
                            Write-Host "    $conditionalVal  Before: $priorVal  ->  After: 0"

                            if ($PSCmdlet.ShouldProcess("$devPath\$conditionalVal", 'Set DWord = 0')) {
                                Set-ItemProperty -Path $devPath -Name $conditionalVal -Value 0 -Type DWord
                            }
                            $deviceValuesBackup.Add([ordered]@{
                                Name          = $conditionalVal
                                PriorValue    = $priorVal
                                ExistedBefore = $true
                            })
                        } else {
                            Write-Host "    $conditionalVal : not present; skip (per spec)" -ForegroundColor DarkGray
                        }
                    } catch {
                        Write-Warning "    Failed to process $conditionalVal on $instId : $_"
                    }
                }

                $devPowerBackup.Add([ordered]@{
                    InstanceId   = $instId
                    FriendlyName = $dev.FriendlyName
                    Values       = $deviceValuesBackup
                })
            }

            $backup.Actions['DevicePower'] = $devPowerBackup
            Write-Host "  Per-device power management applied." -ForegroundColor Green
            Add-Summary 'DevicePower' 'OK' "$($targetDevices.Count) device(s) processed"
        }
    } catch {
        Write-Warning "Action 3 (DevicePower) failed: $_"
        Add-Summary 'DevicePower' 'FAILED' $_.ToString()
        $anyFailure = $true
    }
} else {
    Write-Host "`n[Action 3] Per-device Power — SKIPPED (-SkipDevicePower)" -ForegroundColor DarkGray
    Add-Summary 'DevicePower' 'SKIPPED' 'User requested skip'
}

# ============================================================
# Action 4: Disable boot-churn services
# Targets: PC_AI-HVSockProxy (NSSM-wrapped, broken app path, auto-start, fail loop)
#          vtss              (Intel VTune sampler, fails to start)
# SCM event IDs confirming failures: 7000, 7024, 7034
# ============================================================
if (-not $SkipServices) {
    Write-Host "`n[Action 4] Disable Boot-Churn Services" -ForegroundColor Magenta
    try {
        $targetServices = @('PC_AI-HVSockProxy', 'vtss')
        $svcBackup = [System.Collections.Generic.List[object]]::new()

        foreach ($svcName in $targetServices) {
            try {
                $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
                if (-not $svc) {
                    Write-Host "  Service '$svcName' not found; skip." -ForegroundColor DarkGray
                    continue
                }

                $prior = $svc.StartType.ToString()
                Write-Host "  $svcName  Before: StartType = $prior  ->  After: Disabled"

                $svcBackup.Add([ordered]@{
                    Name           = $svcName
                    PriorStartType = $prior
                })

                if ($PSCmdlet.ShouldProcess($svcName, 'Set StartupType = Disabled')) {
                    try {
                        Set-Service -Name $svcName -StartupType Disabled
                    } catch {
                        # Fallback to sc.exe for services that resist Set-Service
                        Write-Verbose "  Set-Service failed, trying sc.exe: $_"
                        $scResult = sc.exe config $svcName start= disabled
                        if ($LASTEXITCODE -ne 0) {
                            throw "sc.exe failed (exit $LASTEXITCODE): $scResult"
                        }
                    }
                }

                Write-Host "  $svcName disabled." -ForegroundColor Green
            } catch {
                Write-Warning "  Failed to disable service '$svcName': $_"
                Add-Summary "Service:$svcName" 'FAILED' $_.ToString()
                $anyFailure = $true
            }
        }

        $backup.Actions['Services'] = $svcBackup
        Add-Summary 'Services' 'OK' (($targetServices -join ', ') + ' -> Disabled')
    } catch {
        Write-Warning "Action 4 (Services) failed: $_"
        Add-Summary 'Services' 'FAILED' $_.ToString()
        $anyFailure = $true
    }
} else {
    Write-Host "`n[Action 4] Services — SKIPPED (-SkipServices)" -ForegroundColor DarkGray
    Add-Summary 'Services' 'SKIPPED' 'User requested skip'
}

# ============================================================
# Action 5 (opt-in): High Performance power plan
# GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
# ============================================================
if ($SetHighPerformancePowerPlan) {
    Write-Host "`n[Action 5] Set High Performance Power Plan" -ForegroundColor Magenta
    try {
        $hpGuid = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
        $current = (powercfg /getactivescheme 2>&1) -join ''
        Write-Host "  Before: $current"

        if ($PSCmdlet.ShouldProcess('Active power scheme', "Set to High Performance ($hpGuid)")) {
            powercfg /setactive $hpGuid
        }

        Write-Host "  After:  High Performance ($hpGuid) active." -ForegroundColor Green
        Write-Warning ("RECONCILE REQUIRED: Process Lasso may revert this at next boot " +
                       "if StartWithPowerPlan=Balanced is configured in " +
                       "C:\ProgramData\ProcessLasso\config\prolasso.ini. " +
                       "Reconcile via Tools\Apply-ProcessLassoUiSyncTuning.ps1 if that exists. " +
                       "This script does NOT modify prolasso.ini.")

        Add-Summary 'HighPerfPlan' 'OK' "GUID $hpGuid set active (verify Process Lasso)"
    } catch {
        Write-Warning "Action 5 (HighPerfPlan) failed: $_"
        Add-Summary 'HighPerfPlan' 'FAILED' $_.ToString()
        $anyFailure = $true
    }
} else {
    Write-Host "`n[Action 5] High Performance Plan — SKIPPED (not requested; use -SetHighPerformancePowerPlan to enable)" -ForegroundColor DarkGray
    Add-Summary 'HighPerfPlan' 'SKIPPED' 'Opt-in switch not provided'
}

# ---------------------------------------------------------------------------
# 5. Write backup JSON (forward pass)
# ---------------------------------------------------------------------------
if (-not $WhatIfPreference) {
    try {
        $backup | ConvertTo-Json -Depth 10 | Set-Content -Path $backupPath -Encoding UTF8
        Write-Host "`nBackup written: $backupPath" -ForegroundColor DarkGray
    } catch {
        Write-Warning "Could not write backup file: $_"
    }
} else {
    Write-Host "`n[WhatIf] Backup would be written to: $backupPath" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$summary | Format-Table -AutoSize -Property Action, Status, Detail

Write-Host "REMINDER: Sign out and sign back in (or reboot) to allow HID / Bluetooth" -ForegroundColor Yellow
Write-Host "          driver re-initialization after power-management registry changes." -ForegroundColor Yellow

if ($anyFailure) {
    Write-Host "`nOne or more actions failed. Review warnings above." -ForegroundColor Red
    exit 1
}

Write-Host "`nAll actions completed successfully." -ForegroundColor Green
exit 0
