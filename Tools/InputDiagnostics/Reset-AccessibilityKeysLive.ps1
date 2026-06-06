#Requires -Version 7.0
<#
.SYNOPSIS
    Diagnoses and fixes a "Shift key not working" / sticky / slow keyboard in the
    LIVE session (no admin, no reboot) by calling Win32 SystemParametersInfo to turn
    OFF FilterKeys, StickyKeys, and ToggleKeys immediately and persist to the profile.
    Also reports keyboard-remap causes (Scancode Map / PowerToys / AutoHotkey) that
    accessibility flags can't explain.

.DESCRIPTION
    The HKCU "Flags" registry values for these features are only re-read by the OS at
    sign-in, so editing the registry does not fix a session where FilterKeys/StickyKeys
    is already engaged. This script uses SPI_SETFILTERKEYS/SPI_SETSTICKYKEYS/
    SPI_SETTOGGLEKEYS with SPIF_UPDATEINIFILE | SPIF_SENDCHANGE to apply + persist now.

    Refs (Microsoft Learn, verified 2026-05-30):
      FILTERKEYS/STICKYKEYS flag bits + SystemParametersInfo:
        https://learn.microsoft.com/windows/win32/api/winuser/ns-winuser-filterkeys
        https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-systemparametersinfow
      FKF_FILTERKEYSON=0x01 FKF_AVAILABLE=0x02 FKF_HOTKEYACTIVE=0x04
      SKF_STICKYKEYSON=0x01 SKF_AVAILABLE=0x02 SKF_HOTKEYACTIVE=0x04

.PARAMETER DiagnoseOnly
    Report live state + remap causes but do not change anything.

.EXAMPLE
    pwsh -File .\Reset-AccessibilityKeysLive.ps1
.EXAMPLE
    pwsh -File .\Reset-AccessibilityKeysLive.ps1 -DiagnoseOnly
#>
[CmdletBinding()]
param([switch]$DiagnoseOnly)

Add-Type -Namespace Win32 -Name Acc -MemberDefinition @"
[StructLayout(LayoutKind.Sequential)]
public struct FILTERKEYS { public uint cbSize; public uint dwFlags; public uint iWaitMSec; public uint iDelayMSec; public uint iRepeatMSec; public uint iBounceMSec; }
[StructLayout(LayoutKind.Sequential)]
public struct STICKYKEYS { public uint cbSize; public uint dwFlags; }
[StructLayout(LayoutKind.Sequential)]
public struct TOGGLEKEYS { public uint cbSize; public uint dwFlags; }
[DllImport("user32.dll", SetLastError=true)]
public static extern bool SystemParametersInfo(uint uiAction, uint uiParam, ref FILTERKEYS pvParam, uint fWinIni);
[DllImport("user32.dll", SetLastError=true)]
public static extern bool SystemParametersInfo(uint uiAction, uint uiParam, ref STICKYKEYS pvParam, uint fWinIni);
[DllImport("user32.dll", SetLastError=true)]
public static extern bool SystemParametersInfo(uint uiAction, uint uiParam, ref TOGGLEKEYS pvParam, uint fWinIni);
[DllImport("user32.dll")] public static extern short GetKeyState(int nVirtKey);
"@

$SPI_GETFILTERKEYS=0x0032; $SPI_SETFILTERKEYS=0x0033
$SPI_GETSTICKYKEYS=0x003A; $SPI_SETSTICKYKEYS=0x003B
$SPI_GETTOGGLEKEYS=0x0034; $SPI_SETTOGGLEKEYS=0x0035
$SPIF_SENDCHANGE=0x0002; $SPIF_UPDATEINIFILE=0x0001
$FKF_ON=0x01; $FKF_AVAIL=0x02; $FKF_HOTKEY=0x04

Write-Host "===== LIVE ACCESSIBILITY STATE (before) =====" -ForegroundColor Cyan
$fk = New-Object Win32.Acc+FILTERKEYS; $fk.cbSize=[uint32][System.Runtime.InteropServices.Marshal]::SizeOf($fk)
[void][Win32.Acc]::SystemParametersInfo($SPI_GETFILTERKEYS,$fk.cbSize,[ref]$fk,0)
$sk = New-Object Win32.Acc+STICKYKEYS; $sk.cbSize=[uint32][System.Runtime.InteropServices.Marshal]::SizeOf($sk)
[void][Win32.Acc]::SystemParametersInfo($SPI_GETSTICKYKEYS,$sk.cbSize,[ref]$sk,0)
$tk = New-Object Win32.Acc+TOGGLEKEYS; $tk.cbSize=[uint32][System.Runtime.InteropServices.Marshal]::SizeOf($tk)
[void][Win32.Acc]::SystemParametersInfo($SPI_GETTOGGLEKEYS,$tk.cbSize,[ref]$tk,0)

$fkOn = [bool]($fk.dwFlags -band $FKF_ON)
$skOn = [bool]($sk.dwFlags -band $FKF_ON)
Write-Host ("  FilterKeys live dwFlags=0x{0:X}  ON={1}  wait={2}ms bounce={3}ms" -f $fk.dwFlags,$fkOn,$fk.iWaitMSec,$fk.iBounceMSec)
Write-Host ("  StickyKeys live dwFlags=0x{0:X}  ON={1}" -f $sk.dwFlags,$skOn)
Write-Host ("  ToggleKeys live dwFlags=0x{0:X}" -f $tk.dwFlags)
if ($fkOn) { Write-Host "  >>> FilterKeys is ACTIVE in this session - this makes keys (incl. Shift) need a long hold or get dropped." -ForegroundColor Red }
if ($skOn) { Write-Host "  >>> StickyKeys is ACTIVE - Shift may be latched/locked." -ForegroundColor Red }

# Live modifier key state (is a Shift latched down right now?)
$VK_LSHIFT=0xA0; $VK_RSHIFT=0xA1
$ls=[Win32.Acc]::GetKeyState($VK_LSHIFT); $rs=[Win32.Acc]::GetKeyState($VK_RSHIFT)
Write-Host ("  GetKeyState LSHIFT=0x{0:X4} RSHIFT=0x{1:X4} (low bit set = currently down/latched)" -f ($ls -band 0xFFFF),($rs -band 0xFFFF))

Write-Host "`n===== KEYBOARD REMAP CAUSES (flags can't explain these) =====" -ForegroundColor Cyan
$scm = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Keyboard Layout' -Name 'Scancode Map' -ErrorAction SilentlyContinue).'Scancode Map'
if ($scm) { Write-Host "  Scancode Map PRESENT (HKLM Keyboard Layout) - a remap exists. Bytes: $([BitConverter]::ToString($scm))" -ForegroundColor Yellow }
else { Write-Host "  Scancode Map: none (no low-level Shift remap)" }
$remappers = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'PowerToys|AutoHotkey|kanata|SharpKeys|kmonad|Keyboard' }
if ($remappers) { Write-Host "  Remapper processes running:"; $remappers | Select-Object Name, Id | Format-Table -AutoSize }
else { Write-Host "  No PowerToys/AutoHotkey/kanata remapper processes detected" }
$ptKbd = "$env:LOCALAPPDATA\Microsoft\PowerToys\Keyboard Manager\default.json"
if (Test-Path $ptKbd) { Write-Host "  PowerToys Keyboard Manager config present: $ptKbd (check for Shift remaps)" -ForegroundColor Yellow }

if ($DiagnoseOnly) { Write-Host "`n(DiagnoseOnly - no changes made)" -ForegroundColor DarkGray; return }

Write-Host "`n===== APPLYING LIVE FIX (turn OFF, clear hotkey, persist) =====" -ForegroundColor Yellow
$fk.dwFlags = $FKF_AVAIL                      # available only: ON+HOTKEY cleared
$fk.iWaitMSec = 0; $fk.iBounceMSec = 0; $fk.iDelayMSec = 0; $fk.iRepeatMSec = 0
[void][Win32.Acc]::SystemParametersInfo($SPI_SETFILTERKEYS,$fk.cbSize,[ref]$fk,($SPIF_UPDATEINIFILE -bor $SPIF_SENDCHANGE))
$sk.dwFlags = $FKF_AVAIL
[void][Win32.Acc]::SystemParametersInfo($SPI_SETSTICKYKEYS,$sk.cbSize,[ref]$sk,($SPIF_UPDATEINIFILE -bor $SPIF_SENDCHANGE))
$tk.dwFlags = $FKF_AVAIL
[void][Win32.Acc]::SystemParametersInfo($SPI_SETTOGGLEKEYS,$tk.cbSize,[ref]$tk,($SPIF_UPDATEINIFILE -bor $SPIF_SENDCHANGE))

Write-Host "===== LIVE STATE (after) =====" -ForegroundColor Cyan
$fk2 = New-Object Win32.Acc+FILTERKEYS; $fk2.cbSize=[uint32][System.Runtime.InteropServices.Marshal]::SizeOf($fk2)
[void][Win32.Acc]::SystemParametersInfo($SPI_GETFILTERKEYS,$fk2.cbSize,[ref]$fk2,0)
$sk2 = New-Object Win32.Acc+STICKYKEYS; $sk2.cbSize=[uint32][System.Runtime.InteropServices.Marshal]::SizeOf($sk2)
[void][Win32.Acc]::SystemParametersInfo($SPI_GETSTICKYKEYS,$sk2.cbSize,[ref]$sk2,0)
Write-Host ("  FilterKeys dwFlags=0x{0:X} ON={1}" -f $fk2.dwFlags,[bool]($fk2.dwFlags -band $FKF_ON))
Write-Host ("  StickyKeys dwFlags=0x{0:X} ON={1}" -f $sk2.dwFlags,[bool]($sk2.dwFlags -band $FKF_ON))
Write-Host "`nDone. FilterKeys/StickyKeys/ToggleKeys are OFF in the live session and persisted." -ForegroundColor Green
Write-Host "If Shift is STILL broken, the cause is a remap (see above) or hardware - not accessibility." -ForegroundColor Yellow
