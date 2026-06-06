#Requires -Version 7.0
<#
.SYNOPSIS
    Applies (or reverts) the safe, reversible "quick win" remediations for the
    Shift-key / trackpad / fingerprint input-stack glitches investigated on
    DTM-P1GEN7 (Lenovo ThinkPad P1 Gen 7, Windows 11).

.DESCRIPTION
    Three remediations, each grounded in official Microsoft documentation:

      1. Accessibility activation hotkeys (HKCU - no elevation required)
         Clears FKF/SKF_HOTKEYACTIVE (0x04) on StickyKeys/FilterKeys/ToggleKeys so
         resting on Right-Shift (8s -> FilterKeys/SlowKeys, wait=1000ms) or pressing
         Shift x5 (StickyKeys) can no longer silently engage an input filter that
         mimics a "frozen" Shift key. Features stay AVAILABLE via Settings.
         Ref: https://learn.microsoft.com/windows/win32/api/winuser/ns-winuser-filterkeys
              https://learn.microsoft.com/windows/win32/dxtecharts/disabling-shortcut-keys-in-games

      2. USB selective suspend OFF (AC + DC) (HKLM/power - requires elevation)
         Stops Windows powering down idle USB devices, which causes resume latency
         on the USB Synaptics fingerprint reader (VID_06CB) and USB HID input.
         Ref: https://learn.microsoft.com/troubleshoot/microsoftteams/teams-rooms-and-devices/usb-selective-suspend-status-unhealthy
              https://learn.microsoft.com/windows-hardware/drivers/hid/selective-suspend-for-hid-over-usb-devices

      3. Crash dump = Automatic (0x7) (HKLM - requires elevation)
         For >32 GB RAM machines MS recommends Automatic Memory Dump so the NEXT
         hard freeze/bugcheck is actually captured (current value 0x3 = small, and
         the 5/29 hard hangs produced no dump at all).
         Ref: https://learn.microsoft.com/troubleshoot/windows-server/performance/troubleshoot-stop-errors-best-practices-dump-configuration-recommendations

    Every change is backed up to a timestamped JSON before being applied, and -Revert
    restores from the most recent backup (or a specified -BackupFile).

.PARAMETER Revert
    Restore the previous state from the most recent backup file (or -BackupFile).

.PARAMETER BackupFile
    Explicit backup JSON to read (with -Revert) or write (default: timestamped file
    under -BackupDir).

.PARAMETER BackupDir
    Directory for backup JSON files. Default: this script's folder \ backups.

.PARAMETER SkipElevatedChanges
    Apply only the HKCU accessibility hotkey change (works without admin); skip the
    USB selective-suspend and crash-dump changes that need elevation.

.EXAMPLE
    # Apply everything (run from an ELEVATED PowerShell 7 prompt):
    pwsh -File .\Repair-InputStackQuickWins.ps1

.EXAMPLE
    # Apply only the no-admin accessibility fix:
    pwsh -File .\Repair-InputStackQuickWins.ps1 -SkipElevatedChanges

.EXAMPLE
    # Undo the last run:
    pwsh -File .\Repair-InputStackQuickWins.ps1 -Revert

.NOTES
    Author : input-stack investigation (Claude Code) - 2026-05-30
    Target : DTM-P1GEN7 / Windows 11 26200 / 64 GB RAM
    Safe   : All changes reversible. Per-device "Allow the computer to turn off this
             device" for the fingerprint reader and Bluetooth radio is intentionally
             left as a manual Device Manager step (device-specific, see README.md).
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Revert,
    [string]$BackupFile,
    [string]$BackupDir = (Join-Path $PSScriptRoot 'backups'),
    [switch]$SkipElevatedChanges
)

$ErrorActionPreference = 'Stop'

# --- USB selective suspend GUIDs (MS-documented) ---
$UsbSubGroup = '2a737441-1930-4402-8d77-b2bebba308a3'   # USB settings
$UsbSetting  = '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'   # USB selective suspend setting
$AccessRoot  = 'HKCU:\Control Panel\Accessibility'
$CrashCtl    = 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'

function Test-IsElevated {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}

function Get-AccessFlags($leaf) {
    (Get-ItemProperty -Path (Join-Path $AccessRoot $leaf) -Name Flags -ErrorAction Stop).Flags
}

function Get-UsbSuspendIndex {
    # Returns @{AC=..;DC=..} parsed from powercfg, or $null on failure.
    try {
        $out = powercfg /q SCHEME_CURRENT $UsbSubGroup $UsbSetting 2>$null
        $ac = ($out | Select-String 'Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)').Matches.Groups[1].Value
        $dc = ($out | Select-String 'Current DC Power Setting Index:\s*(0x[0-9a-fA-F]+)').Matches.Groups[1].Value
        return @{ AC = $ac; DC = $dc }
    } catch { return $null }
}

# =====================================================================
# REVERT
# =====================================================================
if ($Revert) {
    if (-not $BackupFile) {
        $BackupFile = Get-ChildItem -Path $BackupDir -Filter 'input-quickwins-*.json' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $BackupFile -or -not (Test-Path $BackupFile)) {
        throw "No backup file found in '$BackupDir'. Specify -BackupFile."
    }
    Write-Host "Reverting from backup: $BackupFile" -ForegroundColor Cyan
    $b = Get-Content $BackupFile -Raw | ConvertFrom-Json

    foreach ($k in 'StickyKeys','Keyboard Response','ToggleKeys') {
        if ($b.Accessibility.$k) {
            Set-ItemProperty -Path (Join-Path $AccessRoot $k) -Name Flags -Value ([string]$b.Accessibility.$k) -Type String
            Write-Host "  [HKCU] $k Flags -> $($b.Accessibility.$k)"
        }
    }
    if (Test-IsElevated) {
        if ($b.UsbSelectiveSuspend.AC) {
            $acVal = [int]$b.UsbSelectiveSuspend.AC
            $dcVal = [int]$b.UsbSelectiveSuspend.DC
            powercfg /setacvalueindex SCHEME_CURRENT $UsbSubGroup $UsbSetting $acVal | Out-Null
            powercfg /setdcvalueindex SCHEME_CURRENT $UsbSubGroup $UsbSetting $dcVal | Out-Null
            powercfg /setactive SCHEME_CURRENT | Out-Null
            Write-Host "  [power] USB selective suspend -> AC=$acVal DC=$dcVal"
        }
        if ($null -ne $b.CrashDumpEnabled) {
            Set-ItemProperty -Path $CrashCtl -Name CrashDumpEnabled -Value ([int]$b.CrashDumpEnabled) -Type DWord
            Write-Host "  [HKLM] CrashDumpEnabled -> $($b.CrashDumpEnabled)"
        }
    } else {
        Write-Warning "Not elevated: USB suspend / crash-dump values NOT reverted (HKCU accessibility reverted)."
    }
    Write-Host "Revert complete. Sign out/in for accessibility flags to fully re-read." -ForegroundColor Green
    return
}

# =====================================================================
# BACKUP CURRENT STATE
# =====================================================================
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
if (-not $BackupFile) {
    $stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $BackupFile = Join-Path $BackupDir "input-quickwins-$stamp.json"
}

$usb = Get-UsbSuspendIndex
$backup = [ordered]@{
    Timestamp           = (Get-Date).ToString('o')
    Machine             = $env:COMPUTERNAME
    Elevated            = (Test-IsElevated)
    Accessibility       = [ordered]@{
        'StickyKeys'        = (Get-AccessFlags 'StickyKeys')
        'Keyboard Response' = (Get-AccessFlags 'Keyboard Response')
        'ToggleKeys'        = (Get-AccessFlags 'ToggleKeys')
    }
    UsbSelectiveSuspend = if ($usb) { @{ AC = $usb.AC; DC = $usb.DC } } else { $null }
    CrashDumpEnabled    = (Get-ItemProperty -Path $CrashCtl -Name CrashDumpEnabled -ErrorAction SilentlyContinue).CrashDumpEnabled
}
$backup | ConvertTo-Json -Depth 5 | Set-Content -Path $BackupFile -Encoding utf8
Write-Host "Backup written: $BackupFile" -ForegroundColor Cyan
Write-Host ""

# =====================================================================
# 1. ACCESSIBILITY HOTKEYS (HKCU - no elevation needed)
#    Clear 0x04 HOTKEYACTIVE: StickyKeys 510->506, FilterKeys 126->122, ToggleKeys 62->58
#    (computed generically: AND-out 0x04 so it works from any starting Flags value)
# =====================================================================
Write-Host "[1/3] Disabling accidental accessibility activation hotkeys (HKCU)..." -ForegroundColor Yellow
foreach ($leaf in 'StickyKeys','Keyboard Response','ToggleKeys') {
    $cur = [int](Get-AccessFlags $leaf)
    $new = $cur -band (-bnot 0x04)            # clear HOTKEYACTIVE
    $new = $new -band (-bnot 0x01)            # ensure feature itself is OFF
    if ($PSCmdlet.ShouldProcess("$leaf Flags", "set $cur -> $new")) {
        Set-ItemProperty -Path (Join-Path $AccessRoot $leaf) -Name Flags -Value ([string]$new) -Type String
        Write-Host ("    {0,-18} {1} -> {2}" -f $leaf, $cur, $new)
    }
}

# =====================================================================
# 2 & 3 require elevation
# =====================================================================
if ($SkipElevatedChanges) {
    Write-Warning "Skipping elevated changes (-SkipElevatedChanges). USB suspend + crash dump unchanged."
} elseif (-not (Test-IsElevated)) {
    Write-Warning "NOT ELEVATED - skipped USB selective-suspend and crash-dump changes."
    Write-Warning "Re-run from an elevated PowerShell to apply them, e.g.:"
    Write-Warning "    Start-Process pwsh -Verb RunAs -ArgumentList '-File','$PSCommandPath'"
} else {
    Write-Host "[2/3] Disabling USB selective suspend (AC + DC)..." -ForegroundColor Yellow
    if ($PSCmdlet.ShouldProcess('USB selective suspend', 'disable AC+DC')) {
        powercfg /setacvalueindex SCHEME_CURRENT $UsbSubGroup $UsbSetting 0 | Out-Null
        powercfg /setdcvalueindex SCHEME_CURRENT $UsbSubGroup $UsbSetting 0 | Out-Null
        powercfg /setactive SCHEME_CURRENT | Out-Null
        $after = Get-UsbSuspendIndex
        Write-Host "    USB selective suspend now AC=$($after.AC) DC=$($after.DC) (0x0 = disabled)"
    }

    Write-Host "[3/3] Setting crash dump to Automatic (0x7) for next-freeze capture..." -ForegroundColor Yellow
    if ($PSCmdlet.ShouldProcess('CrashDumpEnabled', 'set 0x7 Automatic')) {
        Set-ItemProperty -Path $CrashCtl -Name CrashDumpEnabled -Value 7 -Type DWord
        Set-ItemProperty -Path $CrashCtl -Name AutoReboot       -Value 1 -Type DWord
        $cde = (Get-ItemProperty -Path $CrashCtl -Name CrashDumpEnabled).CrashDumpEnabled
        Write-Host "    CrashDumpEnabled = $cde (7 = Automatic Memory Dump). Page file should be System-managed."
    }
}

Write-Host ""
Write-Host "Done. Sign out/in (or reboot) so accessibility flags + crash-dump settings fully apply." -ForegroundColor Green
Write-Host "Revert anytime with:  pwsh -File `"$PSCommandPath`" -Revert" -ForegroundColor Green
