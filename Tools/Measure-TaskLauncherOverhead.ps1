#Requires -Version 7.0
<#
.SYNOPSIS
Benchmarks the launch overhead of the scheduled-task wrappers and checks that none shows a window.

.DESCRIPTION
Two independent measurements, both read-only with respect to real tasks:

  Overhead   Wall time from Process.Start to exit for a trivial child, run directly, behind
             `conhost.exe --headless`, and behind each supplied dtm-task-run.exe build. Variants
             are interleaved round-robin so machine noise lands on all of them equally; warm-up
             runs are discarded. Two children are used: `pwsh -NoProfile -Command exit 0` (the
             real task shape, dominated by pwsh start-up) and `cmd.exe /c exit 0` (isolates the
             wrapper's own cost).

  Popup      Registers a throwaway Task Scheduler task per variant (same InteractiveToken principal
             as the real tasks), starts it, and polls EnumWindows every 10 ms for NEW visible
             top-level windows owned by console-ish processes. The direct variant is the positive
             control: it must be seen, or the detector proves nothing. The throwaway tasks are
             always deleted.

Writes machine-readable results to -OutputJson and prints a markdown table. Check CPU load first:
a busy machine inflates every number (see repo CLAUDE.md, "Tooling Benchmarks").

.PARAMETER LauncherExe
One or more `label=path` pairs for dtm-task-run.exe builds, e.g. `fd=C:\x\dtm-task-run.exe`.

.PARAMETER Runs
Measured runs per variant (default 20).

.PARAMETER WarmUp
Discarded runs per variant (default 2).

.PARAMETER PopupRuns
Task Scheduler launches per variant for the popup test (default 4). 0 skips the popup test.

.PARAMETER OutputJson
Where to write the JSON results.

.EXAMPLE
./Measure-TaskLauncherOverhead.ps1 -LauncherExe fd=C:\tmp\fd\dtm-task-run.exe,sc=C:\tmp\sc\dtm-task-run.exe -OutputJson docs\scheduled-tasks\benchmark-results.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string[]]$LauncherExe,
    [ValidateRange(3, 200)][int]$Runs = 20,
    [ValidateRange(0, 10)][int]$WarmUp = 2,
    [ValidateRange(0, 20)][int]$PopupRuns = 4,
    [string]$OutputJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$pwsh = (Get-Command pwsh.exe -ErrorAction Stop).Source
$conhost = Join-Path $env:SystemRoot 'System32\conhost.exe'
$cmd = Join-Path $env:SystemRoot 'System32\cmd.exe'
$launchers = [ordered]@{}
# `pwsh -File` delivers an array parameter as one comma-joined string; split it back into label=path pairs.
foreach ($pair in @($LauncherExe | ForEach-Object { $_ -split ',(?=[A-Za-z0-9_-]+=)' })) {
    $label, $path = $pair -split '=', 2
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Launcher not found: $path" }
    $launchers[$label] = (Resolve-Path -LiteralPath $path).Path
}

$benchDir = Join-Path ([IO.Path]::GetTempPath()) ("dtm-launcher-bench-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $benchDir | Out-Null
$env:DTM_TASKS_DIR = $benchDir

function New-Variant {
    param([string]$Name, [string]$Exe, [string[]]$Arguments)
    [pscustomobject]@{ Name = $Name; Exe = $Exe; Arguments = $Arguments }
}

function Get-VariantSet {
    param([string]$ChildExe, [string[]]$ChildArgs)
    $set = @(
        New-Variant 'direct' $ChildExe $ChildArgs
        New-Variant 'conhost --headless' $conhost (@('--headless', $ChildExe) + $ChildArgs)
    )
    foreach ($label in $launchers.Keys) {
        $set += New-Variant "dtm-task-run ($label)" $launchers[$label] (@('--name', 'bench', '--', $ChildExe) + $ChildArgs)
    }
    $set
}

function Measure-OneRun {
    param($Variant)
    $psi = [System.Diagnostics.ProcessStartInfo]::new($Variant.Exe)
    foreach ($a in $Variant.Arguments) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.WaitForExit()
    $sw.Stop()
    $p.Dispose()
    $sw.Elapsed.TotalMilliseconds
}

function Get-Stats {
    param([double[]]$Samples)
    $s = @($Samples | Sort-Object)
    $n = $s.Count
    $median = if ($n % 2) { $s[($n - 1) / 2] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    $p95 = $s[[math]::Min($n - 1, [math]::Ceiling(0.95 * $n) - 1)]   # nearest-rank
    [pscustomobject]@{
        Runs = $n; MedianMs = [math]::Round($median, 1); P95Ms = [math]::Round($p95, 1)
        MinMs = [math]::Round($s[0], 1); MaxMs = [math]::Round($s[-1], 1)
    }
}

function Invoke-OverheadBenchmark {
    param([string]$ChildLabel, [string]$ChildExe, [string[]]$ChildArgs)
    $variants = Get-VariantSet $ChildExe $ChildArgs
    $samples = @{}
    foreach ($v in $variants) { $samples[$v.Name] = [System.Collections.Generic.List[double]]::new() }
    for ($i = -$WarmUp; $i -lt $Runs; $i++) {
        foreach ($v in $variants) {
            $ms = Measure-OneRun $v
            if ($i -ge 0) { $samples[$v.Name].Add($ms) }
        }
    }
    $direct = $null
    foreach ($v in $variants) {
        $stat = Get-Stats $samples[$v.Name]
        if ($v.Name -eq 'direct') { $direct = $stat.MedianMs }
        [pscustomobject]@{
            Child = $ChildLabel; Variant = $v.Name; Runs = $stat.Runs; MedianMs = $stat.MedianMs; P95Ms = $stat.P95Ms
            MinMs = $stat.MinMs; MaxMs = $stat.MaxMs; OverheadVsDirectMs = [math]::Round($stat.MedianMs - $direct, 1)
        }
    }
}

# --- Popup detector --------------------------------------------------------------------------
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public static class WinProbe {
    public delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc p, IntPtr l);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassNameW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out RECT r);
    [StructLayout(LayoutKind.Sequential)] struct RECT { public int L, T, R, B; }
    public static List<string[]> Snapshot() {
        var list = new List<string[]>();
        EnumWindows((h, l) => {
            if (!IsWindowVisible(h)) return true;
            RECT r; GetWindowRect(h, out r);
            if (r.R - r.L <= 0 || r.B - r.T <= 0) return true;
            uint pid; GetWindowThreadProcessId(h, out pid);
            var t = new StringBuilder(128); GetWindowTextW(h, t, 128);
            var c = new StringBuilder(128); GetClassNameW(h, c, 128);
            list.Add(new[] { h.ToInt64().ToString(), pid.ToString(), c.ToString(), t.ToString() });
            return true;
        }, IntPtr.Zero);
        return list;
    }
}
'@

$ConsoleLike = @('pwsh', 'powershell', 'conhost', 'cmd', 'dtm-task-run', 'OpenConsole', 'WindowsTerminal', 'sleep', 'ping')

function Get-NewWindows {
    param($Baseline, [datetime]$Since)
    foreach ($w in [WinProbe]::Snapshot()) {
        if ($Baseline.ContainsKey($w[0])) { continue }
        $proc = Get-Process -Id ([int]$w[1]) -ErrorAction SilentlyContinue
        if ($null -eq $proc) { continue }
        if ($ConsoleLike -notcontains $proc.ProcessName) { continue }
        if ($proc.StartTime -lt $Since) { continue }
        [pscustomobject]@{ Process = $proc.ProcessName; Class = $w[2]; Title = $w[3] }
    }
}

function Invoke-PopupTest {
    $stamp = Get-Date
    $results = @()
    $child = @('-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Milliseconds 1500')
    $variants = @(
        New-Variant 'direct' $pwsh $child
        New-Variant 'conhost --headless' $conhost (@('--headless', $pwsh) + $child)
    )
    foreach ($label in $launchers.Keys) {
        $variants += New-Variant "dtm-task-run ($label)" $launchers[$label] (@('--name', 'popup', '--', $pwsh) + $child)
    }
    foreach ($v in $variants) {
        $taskName = 'dtm-launcher-popup-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        $argString = ($v.Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
        $action = New-ScheduledTaskAction -Execute $v.Exe -Argument $argString
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
        $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
        Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
        $seen = [System.Collections.Generic.List[object]]::new()
        try {
            for ($r = 0; $r -lt $PopupRuns; $r++) {
                $baseline = @{}
                foreach ($w in [WinProbe]::Snapshot()) { $baseline[$w[0]] = $true }
                $since = Get-Date
                Start-ScheduledTask -TaskName $taskName
                $deadline = (Get-Date).AddSeconds(8)
                $started = $false
                do {
                    foreach ($w in @(Get-NewWindows $baseline $since)) { $seen.Add($w) }
                    $state = (Get-ScheduledTask -TaskName $taskName).State
                    if ($state -eq 'Running') { $started = $true }
                    Start-Sleep -Milliseconds 10
                } while ((Get-Date) -lt $deadline -and (-not $started -or $state -eq 'Running'))
                for ($k = 0; $k -lt 30; $k++) {   # tail: a window can outlive the task by a moment
                    foreach ($w in @(Get-NewWindows $baseline $since)) { $seen.Add($w) }
                    Start-Sleep -Milliseconds 10
                }
            }
            $info = Get-ScheduledTaskInfo -TaskName $taskName
            $distinct = @($seen | Group-Object Process, Class | ForEach-Object { '{0}/{1}' -f $_.Group[0].Process, $_.Group[0].Class })
            $results += [pscustomobject]@{
                Variant = $v.Name; Launches = $PopupRuns; WindowSightings = $seen.Count
                WindowKinds = ($distinct -join '; '); LastTaskResult = ('0x{0:X}' -f [uint32]$info.LastTaskResult)
            }
        }
        finally {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        }
    }
    $results
}

# --- Run ---------------------------------------------------------------------------------------
$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
Write-Host ("CPU load before run: {0}%  (high load inflates every number)" -f $load)

$overhead = @()
$overhead += Invoke-OverheadBenchmark 'pwsh -NoProfile -Command exit 0' $pwsh @('-NoProfile', '-Command', 'exit 0')
$overhead += Invoke-OverheadBenchmark 'cmd /c exit 0' $cmd @('/c', 'exit 0')
$popup = @()
if ($PopupRuns -gt 0) { $popup = @(Invoke-PopupTest) }

$sizes = [ordered]@{}
foreach ($k in $launchers.Keys) { $sizes[$k] = [int]((Get-Item $launchers[$k]).Length / 1KB) }
$report = [pscustomobject]@{
    Host = $env:COMPUTERNAME
    CpuLoadPercentBefore = $load
    CpuLoadPercentAfter = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    Runs = $Runs
    WarmUp = $WarmUp
    Percentile = 'nearest-rank p95'
    Launchers = $launchers
    LauncherSizesKB = $sizes
    Overhead = $overhead
    Popup = $popup
}

Write-Host ''
$overhead | Format-Table Child, Variant, MedianMs, P95Ms, MinMs, MaxMs, OverheadVsDirectMs -AutoSize
if ($popup.Count) { $popup | Format-Table -AutoSize }

if ($OutputJson) {
    $dir = Split-Path -Parent $OutputJson
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputJson -Encoding utf8
    Write-Host "Results written to $OutputJson"
}
Remove-Item -LiteralPath $benchDir -Recurse -Force -ErrorAction SilentlyContinue
