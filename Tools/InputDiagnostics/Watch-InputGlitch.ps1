#Requires -Version 7.0
<#
.SYNOPSIS
    Gate-C glitch-capture + symptom-frequency harness for the intermittent Shift-key /
    touchpad-lockup problem. READ-ONLY: it never changes system state.

.DESCRIPTION
    Intermittent input bugs can't be fixed-and-forgotten — you must MEASURE them before and
    after each change. This tool provides the measurement that prior investigations lacked:

      Watch    Interactive session. When a glitch happens, press a key to tag it; the tool
               captures the discriminating system state at that instant (the "Gate C" snapshot
               that was never collected) and appends it to a running symptom ledger.
      Snapshot Capture one state snapshot now (for scripting / scheduled sampling).
      Report   Aggregate the ledger into glitches-per-day, optionally split before/after a
               fix date, so you can judge whether a remediation worked.

    Captured per snapshot (all read-only):
      - Accessibility flags (FilterKeys/StickyKeys/ToggleKeys + DelayBeforeAcceptance)
      - Touchpad (SNSL002D) + parent Intel I2C controller (7E78) status & power-down policy
      - Recent Modern-Standby transitions (Kernel-Power 506/507)
      - dwm.exe / explorer.exe health (the 05-26 dwmcore.dll crash burst froze the whole UI)
      - Top CPU / IO processes; recent input & power events

.PARAMETER Mode      Watch (default) | Snapshot | Report
.PARAMETER OutDir    Ledger + snapshot folder. Default: <repo>\Reports\input-glitch-watch
.PARAMETER Symptom   For Snapshot mode tagging: shift | touchpad | both | none
.PARAMETER Note      Free-text note attached to the tagged glitch
.PARAMETER SinceFix  For Report mode: ISO date (yyyy-MM-dd) splitting baseline vs post-fix

.EXAMPLE
    .\Watch-InputGlitch.ps1                       # baseline watch session; tag glitches as they happen
.EXAMPLE
    .\Watch-InputGlitch.ps1 -Mode Report -SinceFix 2026-06-10   # did the fix reduce glitches/day?
.NOTES
    Companion to FINDINGS.md in Reports\input-stack-investigation-20260606\. No elevation required
    for capture (some event queries return more detail when elevated). Never writes to the registry,
    services, devices, or power settings.
#>
[CmdletBinding()]
param(
    [ValidateSet('Watch', 'Snapshot', 'Report')] [string] $Mode = 'Watch',
    [string] $OutDir,
    [ValidateSet('shift', 'touchpad', 'both', 'none')] [string] $Symptom = 'none',
    [string] $Note = '',
    [string] $SinceFix
)

$ErrorActionPreference = 'Stop'
if (-not $OutDir) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $OutDir = Join-Path $repo 'Reports\input-glitch-watch'
}
$snapDir = Join-Path $OutDir 'snapshots'
$ledger = Join-Path $OutDir 'symptom-log.jsonl'
New-Item -ItemType Directory -Path $snapDir -Force | Out-Null

function Write-Status { param($M, $C = 'Gray') Write-Host $M -ForegroundColor $C }

function Get-AccessibilityState {
    $out = @{}
    foreach ($k in 'Keyboard Response', 'StickyKeys', 'ToggleKeys') {
        $p = Get-ItemProperty "HKCU:\Control Panel\Accessibility\$k" -ErrorAction SilentlyContinue
        if ($p) {
            $flags = [int]$p.Flags
            $out[$k] = [ordered]@{
                Flags         = $flags
                On            = [bool]($flags -band 0x01)
                HotkeyArmed   = [bool]($flags -band 0x04)
                DelayMs       = $p.DelayBeforeAcceptance
            }
        }
    }
    return $out
}

function Get-InputDevicePower {
    # Read-only: device status + "allow computer to turn off this device" policy.
    $res = [ordered]@{}
    try {
        $touch = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -like '*SNSL002D*' } | Select-Object -First 1
        if ($touch) { $res.Touchpad = @{ Id = $touch.InstanceId; Status = $touch.Status; Problem = $touch.Problem } }
    } catch {}
    try {
        $kbd = Get-PnpDevice -PresentOnly -Class Keyboard -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -like '*LEN0071*' -or $_.FriendlyName -like '*PS/2*' } | Select-Object -First 1
        if ($kbd) { $res.Keyboard = @{ Id = $kbd.InstanceId; Status = $kbd.Status; Problem = $kbd.Problem } }
    } catch {}
    try {
        # MSPower_DeviceEnable: $true = device IS allowed to power down (the touchpad-lockup risk)
        $pw = Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable -ErrorAction SilentlyContinue
        $res.PowerDownEnabled = @($pw | Where-Object { $_.InstanceName -match 'SNSL002D|7E78' } |
            ForEach-Object { @{ Instance = $_.InstanceName; CanPowerDown = $_.Enable } })
    } catch {}
    return $res
}

function Get-RecentEvents {
    param([int]$Minutes = 30)
    $since = (Get-Date).AddMinutes(-$Minutes)
    $out = [ordered]@{}
    try {
        $modern = Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-Power'; Id = 506, 507; StartTime = $since } -ErrorAction SilentlyContinue
        $out.ModernStandby = @($modern | Select-Object -First 6 | ForEach-Object { "$($_.TimeCreated.ToString('HH:mm:ss')) id=$($_.Id)" })
    } catch {}
    try {
        $inp = Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'i8042prt', 'kbdhid', 'hidi2c', 'mouhid'; StartTime = $since } -ErrorAction SilentlyContinue
        $out.InputErrors = @($inp | Select-Object -First 6 | ForEach-Object { "$($_.TimeCreated.ToString('HH:mm:ss')) $($_.ProviderName) id=$($_.Id)" })
    } catch {}
    return $out
}

function Get-UiHealth {
    $out = [ordered]@{}
    foreach ($n in 'dwm', 'explorer', 'ctfmon', 'OneDrive') {
        $p = Get-Process -Name $n -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($p) {
            $responding = $true
            try { if ($p.MainWindowHandle -ne 0) { $responding = $p.Responding } } catch {}
            $out[$n] = @{ Cpu_s = [math]::Round($p.CPU, 1); Responding = $responding; WS_MB = [math]::Round($p.WorkingSet64 / 1MB) }
        }
    }
    return $out
}

function New-Snapshot {
    param([string]$Tag = 'none', [string]$NoteText = '')
    $ts = Get-Date
    $snap = [ordered]@{
        timestamp     = $ts.ToString('o')
        symptom       = $Tag
        note          = $NoteText
        accessibility = Get-AccessibilityState
        devicePower   = Get-InputDevicePower
        recentEvents  = Get-RecentEvents -Minutes 30
        uiHealth      = Get-UiHealth
        topCpu        = @(Get-Process | Sort-Object CPU -Descending | Select-Object -First 5 |
                          ForEach-Object { @{ name = $_.ProcessName; cpu_s = [math]::Round($_.CPU, 1) } })
        powerScheme   = (powercfg /getactivescheme) 2>$null
    }
    $file = Join-Path $snapDir ("snap-{0}.json" -f $ts.ToString('yyyyMMdd-HHmmss'))
    $snap | ConvertTo-Json -Depth 6 | Set-Content -Path $file -Encoding UTF8
    # Append compact ledger line
    $line = @{ timestamp = $snap.timestamp; symptom = $Tag; note = $NoteText; snapshot = (Split-Path $file -Leaf) } | ConvertTo-Json -Compress
    Add-Content -Path $ledger -Value $line -Encoding UTF8
    return $file
}

switch ($Mode) {
    'Snapshot' {
        $f = New-Snapshot -Tag $Symptom -NoteText $Note
        Write-Status "Snapshot saved: $f" Green
        if ($Symptom -ne 'none') { Write-Status "Tagged symptom '$Symptom' in $ledger" Cyan }
    }

    'Watch' {
        Write-Status "Input-glitch watch — READ-ONLY. Tag a glitch the instant it happens." Cyan
        Write-Status "  [s] Shift not recognized   [t] Touchpad lockup   [b] Both   [m] mark/note   [q] quit" Gray
        Write-Status "  Ledger: $ledger`n" DarkGray
        while ($true) {
            $key = [Console]::ReadKey($true)
            switch ($key.KeyChar) {
                's' { $f = New-Snapshot -Tag 'shift';    Write-Status "  [shift] captured -> $(Split-Path $f -Leaf)" Yellow }
                't' { $f = New-Snapshot -Tag 'touchpad'; Write-Status "  [touchpad] captured -> $(Split-Path $f -Leaf)" Yellow }
                'b' { $f = New-Snapshot -Tag 'both';     Write-Status "  [both] captured -> $(Split-Path $f -Leaf)" Yellow }
                'm' {
                    Write-Host "  note> " -NoNewline -ForegroundColor Gray
                    $n = Read-Host
                    $f = New-Snapshot -Tag 'none' -NoteText $n; Write-Status "  [note] captured" DarkYellow
                }
                'q' { Write-Status "Done. Run -Mode Report to summarize." Green; return }
                default { }
            }
        }
    }

    'Report' {
        if (-not (Test-Path $ledger)) { Write-Status "No ledger yet at $ledger" Red; return }
        $rows = Get-Content $ledger | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json }
        $tagged = $rows | Where-Object { $_.symptom -ne 'none' }
        if (-not $tagged) { Write-Status "No tagged glitches recorded yet." Yellow; return }
        $span = ($tagged | Sort-Object timestamp)
        $first = [datetime]$span[0].timestamp; $last = [datetime]$span[-1].timestamp
        $days = [math]::Max(1, ($last - $first).TotalDays)
        Write-Status "Glitch ledger: $($tagged.Count) tagged events over $([math]::Round($days,1)) days" Cyan
        $tagged | Group-Object symptom | ForEach-Object {
            "  {0,-9} {1,3}  ({2}/day)" -f $_.Name, $_.Count, [math]::Round($_.Count / $days, 2)
        }
        if ($SinceFix) {
            $cut = [datetime]::ParseExact($SinceFix, 'yyyy-MM-dd', $null)
            $before = $tagged | Where-Object { [datetime]$_.timestamp -lt $cut }
            $after = $tagged | Where-Object { [datetime]$_.timestamp -ge $cut }
            $bd = [math]::Max(1, ($cut - $first).TotalDays)
            $ad = [math]::Max(1, ($last - $cut).TotalDays)
            Write-Status "`nFix evaluation (split at $SinceFix):" Cyan
            "  baseline : {0} events / {1:N1}d = {2}/day" -f $before.Count, $bd, [math]::Round($before.Count / $bd, 2) | Write-Host
            "  post-fix : {0} events / {1:N1}d = {2}/day" -f $after.Count, $ad, [math]::Round($after.Count / $ad, 2) | Write-Host
        }
    }
}
