#Requires -Version 7.0
<#
.SYNOPSIS
    Monitors pointer movement and button transitions during haptic touchpad repros.

.DESCRIPTION
    Installs a passive WH_MOUSE_LL hook for a bounded capture window and records
    pointer movement, left/right button down/up transitions, wheel events, and
    suspicious long button-held intervals.  This helps distinguish a lost
    button-up event from pointer motion stalls or application-level focus issues.

    Low-level mouse hooks do not identify the physical source device.  For best
    diagnostic value, avoid using external mice during the capture and pair this
    with Start-HapticTouchpadTrace.ps1.

.PARAMETER Seconds
    Capture duration. Default: 60.

.PARAMETER OutDir
    Output directory. Default: Reports\haptic-touchpad\pointer-<timestamp>.

.PARAMETER Note
    Free-text scenario note written to the JSON output.

.PARAMETER StuckThresholdMs
    Button-held duration that should be flagged as suspicious. Default: 1200 ms.

.PARAMETER AsJson
    Emit the JSON report to stdout after capture.

.PARAMETER SelfTest
    Write a synthetic empty capture report without installing the desktop hook.
    Intended for agent/CI validation in non-interactive hosts.

.EXAMPLE
    pwsh -File .\Watch-HapticTouchpadInput.ps1 -Seconds 60 -Note "press stickiness repro"

.NOTES
    Read-only passive monitor. It does not block, remap, or synthesize input.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 600)] [int]$Seconds = 60,
    [string]$OutDir,
    [string]$Note = '',
    [ValidateRange(100, 10000)] [int]$StuckThresholdMs = 1200,
    [switch]$AsJson,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

if (-not $OutDir) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $OutDir = Join-Path $repo ("Reports\haptic-touchpad\pointer-{0}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

if ($SelfTest) {
    $summary = [ordered]@{
        Timestamp = (Get-Date).ToString('o')
        Machine = $env:COMPUTERNAME
        Note = $Note
        Seconds = 0
        StuckThresholdMs = $StuckThresholdMs
        EventCount = 0
        MoveCount = 0
        LeftDownCount = 0
        LeftUpCount = 0
        RightDownCount = 0
        RightUpCount = 0
        WarningCount = 0
        Warnings = @()
        Events = @()
        SelfTest = $true
    }
    $jsonPath = Join-Path $OutDir 'pointer-input.json'
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8
    Write-Host "Pointer input self-test report written: $jsonPath" -ForegroundColor Green
    if ($AsJson) { $summary | ConvertTo-Json -Depth 8 }
    return
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace PcaiInput -Name MouseHook -MemberDefinition @"
public delegate System.IntPtr HookProc(int nCode, System.IntPtr wParam, System.IntPtr lParam);
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError=true)] public static extern System.IntPtr SetWindowsHookExW(int idHook, HookProc lpfn, System.IntPtr hMod, uint dwThreadId);
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError=true)] public static extern bool UnhookWindowsHookEx(System.IntPtr hhk);
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern System.IntPtr CallNextHookEx(System.IntPtr hhk, int nCode, System.IntPtr wParam, System.IntPtr lParam);
[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern System.IntPtr GetModuleHandleW(string lpModuleName);
"@

$WH_MOUSE_LL = 14
$WM_MOUSEMOVE = 0x0200
$WM_LBUTTONDOWN = 0x0201
$WM_LBUTTONUP = 0x0202
$WM_RBUTTONDOWN = 0x0204
$WM_RBUTTONUP = 0x0205
$WM_MOUSEWHEEL = 0x020A
$WM_MOUSEHWHEEL = 0x020E

$events = [System.Collections.Generic.List[object]]::new()
$warnings = [System.Collections.Generic.List[object]]::new()
$script:hookId = [IntPtr]::Zero
$script:leftDownAt = $null
$script:rightDownAt = $null
$script:lastPoint = $null

function Get-MessageName {
    param([int]$Message)
    switch ($Message) {
        $WM_MOUSEMOVE { 'MOVE' }
        $WM_LBUTTONDOWN { 'LDOWN' }
        $WM_LBUTTONUP { 'LUP' }
        $WM_RBUTTONDOWN { 'RDOWN' }
        $WM_RBUTTONUP { 'RUP' }
        $WM_MOUSEWHEEL { 'WHEEL' }
        $WM_MOUSEHWHEEL { 'HWHEEL' }
        default { "MSG_$Message" }
    }
}

$start = Get-Date
$proc = [PcaiInput.MouseHook+HookProc]{
    param($nCode, $wParam, $lParam)
    if ($nCode -ge 0) {
        $now = Get-Date
        $msg = [int]$wParam
        $x = [System.Runtime.InteropServices.Marshal]::ReadInt32($lParam, 0)
        $y = [System.Runtime.InteropServices.Marshal]::ReadInt32($lParam, 4)
        $name = Get-MessageName -Message $msg
        $dx = $null
        $dy = $null
        if ($script:lastPoint) {
            $dx = $x - $script:lastPoint.X
            $dy = $y - $script:lastPoint.Y
        }
        $script:lastPoint = [pscustomobject]@{ X = $x; Y = $y }

        if ($msg -eq $WM_LBUTTONDOWN) { $script:leftDownAt = $now }
        if ($msg -eq $WM_RBUTTONDOWN) { $script:rightDownAt = $now }

        $heldMs = $null
        if ($msg -eq $WM_LBUTTONUP -and $script:leftDownAt) {
            $heldMs = [math]::Round(($now - $script:leftDownAt).TotalMilliseconds, 1)
            if ($heldMs -ge $StuckThresholdMs) {
                $warnings.Add([pscustomobject]@{ Time = $now.ToString('o'); Type = 'LongLeftButtonHold'; HeldMs = $heldMs })
            }
            $script:leftDownAt = $null
        }
        if ($msg -eq $WM_RBUTTONUP -and $script:rightDownAt) {
            $heldMs = [math]::Round(($now - $script:rightDownAt).TotalMilliseconds, 1)
            if ($heldMs -ge $StuckThresholdMs) {
                $warnings.Add([pscustomobject]@{ Time = $now.ToString('o'); Type = 'LongRightButtonHold'; HeldMs = $heldMs })
            }
            $script:rightDownAt = $null
        }

        $events.Add([pscustomobject]@{
            Time = $now.ToString('o')
            MsFromStart = [math]::Round(($now - $start).TotalMilliseconds, 1)
            Event = $name
            X = $x
            Y = $y
            DeltaX = $dx
            DeltaY = $dy
            HeldMs = $heldMs
        })
    }
    return [PcaiInput.MouseHook]::CallNextHookEx($script:hookId, $nCode, $wParam, $lParam)
}

$hMod = [PcaiInput.MouseHook]::GetModuleHandleW($null)
$script:hookId = [PcaiInput.MouseHook]::SetWindowsHookExW($WH_MOUSE_LL, $proc, $hMod, 0)
if ($script:hookId -eq [IntPtr]::Zero) {
    throw "Failed to install mouse hook (LastError=$([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
}

try {
    Write-Host "Monitoring pointer/button state for $Seconds seconds. Reproduce the touchpad press/pointer issue now." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 8
    }
} finally {
    if ($script:hookId -ne [IntPtr]::Zero) {
        [void][PcaiInput.MouseHook]::UnhookWindowsHookEx($script:hookId)
    }
}

if ($script:leftDownAt) {
    $warnings.Add([pscustomobject]@{ Time = (Get-Date).ToString('o'); Type = 'LeftButtonStillDownAtEnd'; HeldMs = [math]::Round(((Get-Date) - $script:leftDownAt).TotalMilliseconds, 1) })
}
if ($script:rightDownAt) {
    $warnings.Add([pscustomobject]@{ Time = (Get-Date).ToString('o'); Type = 'RightButtonStillDownAtEnd'; HeldMs = [math]::Round(((Get-Date) - $script:rightDownAt).TotalMilliseconds, 1) })
}

$summary = [ordered]@{
    Timestamp = (Get-Date).ToString('o')
    Machine = $env:COMPUTERNAME
    Note = $Note
    Seconds = $Seconds
    StuckThresholdMs = $StuckThresholdMs
    EventCount = $events.Count
    MoveCount = @($events | Where-Object Event -eq 'MOVE').Count
    LeftDownCount = @($events | Where-Object Event -eq 'LDOWN').Count
    LeftUpCount = @($events | Where-Object Event -eq 'LUP').Count
    RightDownCount = @($events | Where-Object Event -eq 'RDOWN').Count
    RightUpCount = @($events | Where-Object Event -eq 'RUP').Count
    WarningCount = $warnings.Count
    Warnings = $warnings
    Events = $events
}

$jsonPath = Join-Path $OutDir 'pointer-input.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8
Write-Host "Pointer input report written: $jsonPath" -ForegroundColor Green
Write-Host "Events=$($summary.EventCount) Warnings=$($summary.WarningCount)" -ForegroundColor Yellow
if ($AsJson) {
    $summary | ConvertTo-Json -Depth 8
}
