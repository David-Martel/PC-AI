#Requires -Version 7.0
<#
.SYNOPSIS
    Raw low-level keyboard monitor. Installs a WH_KEYBOARD_LL hook and logs every
    key event (vkCode, scanCode, up/down) for -Seconds, with special highlighting of
    LSHIFT/RSHIFT/SHIFT. Definitively answers: "does pressing Shift reach the OS?"

.DESCRIPTION
    Run this in an INTERACTIVE terminal, then press keys (especially Left Shift and
    Right Shift, alone and with letters) during the capture window.

      - If Shift keydown/keyup events APPEAR here  -> the keyboard/EC hardware and OS
        input queue are fine; the failure is higher up (focused app, IME, or a hook
        ABOVE this one swallowing it before apps).
      - If Shift events DO NOT appear while other keys do -> the scancodes are not
        reaching Windows: keyboard hardware / ThinkPad EC / keyboard firmware (BIOS).
        That points to a Lenovo BIOS/EC or keyboard issue, not software.

    Read-only: installs a passive hook, never blocks or remaps keys.

.PARAMETER Seconds
    Capture duration. Default 20.

.EXAMPLE
    pwsh -File .\Test-KeyInput.ps1 -Seconds 20
    # then press Left Shift, Right Shift, Shift+A, Ctrl+Shift+A
.NOTES
    Author: input-stack investigation (Claude Code) - 2026-05-30. Passive monitor only.
#>
[CmdletBinding()]
param([int]$Seconds = 20)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace KbMon -Name Hook -UsingNamespace System.Runtime.InteropServices -MemberDefinition @"
public delegate System.IntPtr HookProc(int nCode, System.IntPtr wParam, System.IntPtr lParam);
[DllImport("user32.dll", SetLastError=true)] public static extern System.IntPtr SetWindowsHookExW(int idHook, HookProc lpfn, System.IntPtr hMod, uint dwThreadId);
[DllImport("user32.dll", SetLastError=true)] public static extern bool UnhookWindowsHookEx(System.IntPtr hhk);
[DllImport("user32.dll")] public static extern System.IntPtr CallNextHookEx(System.IntPtr hhk, int nCode, System.IntPtr wParam, System.IntPtr lParam);
[DllImport("kernel32.dll")] public static extern System.IntPtr GetModuleHandleW(string lpModuleName);
"@

$WH_KEYBOARD_LL = 13
$events = [System.Collections.Generic.List[string]]::new()
$script:hookId = [IntPtr]::Zero

# vkCode is the first DWORD of the KBDLLHOOKSTRUCT pointed to by lParam
$proc = [KbMon.Hook+HookProc]{
    param($nCode, $wParam, $lParam)
    if ($nCode -ge 0) {
        $vk = [System.Runtime.InteropServices.Marshal]::ReadInt32($lParam, 0)
        $sc = [System.Runtime.InteropServices.Marshal]::ReadInt32($lParam, 4)
        $msg = [int]$wParam   # 256=WM_KEYDOWN 257=WM_KEYUP 260=WM_SYSKEYDOWN 261=WM_SYSKEYUP
        $dir = if ($msg -eq 256 -or $msg -eq 260) { 'DOWN' } else { 'UP  ' }
        $name = switch ($vk) { 16 {'SHIFT'} 160 {'LSHIFT'} 161 {'RSHIFT'} 162 {'LCTRL'} 163 {'RCTRL'} default { "VK=$vk" } }
        $tag = if ($vk -in 16,160,161) { '  <<< SHIFT' } else { '' }
        $events.Add(("{0} {1,-7} vk={2,-3} sc={3}{4}" -f $dir, $name, $vk, $sc, $tag))
    }
    return [KbMon.Hook]::CallNextHookEx($script:hookId, $nCode, $wParam, $lParam)
}

$hMod = [KbMon.Hook]::GetModuleHandleW($null)
$script:hookId = [KbMon.Hook]::SetWindowsHookExW($WH_KEYBOARD_LL, $proc, $hMod, 0)
if ($script:hookId -eq [IntPtr]::Zero) { Write-Error "Failed to install hook (LastError=$([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"; return }

Write-Host "Monitoring keyboard for $Seconds seconds. PRESS: Left Shift, Right Shift, Shift+A, Ctrl+Shift+A ..." -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
    [System.Windows.Forms.Application]::DoEvents()
    Start-Sleep -Milliseconds 8
}
[void][KbMon.Hook]::UnhookWindowsHookEx($script:hookId)

Write-Host "`n===== CAPTURED KEY EVENTS ($($events.Count)) =====" -ForegroundColor Cyan
$events | ForEach-Object { Write-Host $_ }
$shiftSeen = ($events | Where-Object { $_ -match 'SHIFT' } | Measure-Object).Count
Write-Host "`nShift events seen: $shiftSeen" -ForegroundColor Yellow
if ($shiftSeen -gt 0) {
    Write-Host "=> Shift IS reaching the OS. Hardware/EC fine. Cause is higher (focused app / a hook above this one)." -ForegroundColor Green
} else {
    Write-Host "=> NO Shift events while you pressed Shift => scancodes not reaching Windows => keyboard hardware / ThinkPad EC / BIOS firmware. Update Lenovo BIOS/EC, test an external keyboard." -ForegroundColor Red
}
