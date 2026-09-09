#Requires -Version 5.1
<#
.SYNOPSIS
Self-healing watchdog for the F: cloud-cache-disk volume.

.DESCRIPTION
Runs periodically. If the cloud-cache-disk volume is missing it re-runs the mount
task and, only if it actually repaired something, restarts the cloud clients
through the gated launcher.

Why this exists: the boot mount task is single-shot. Task Scheduler's
restart-on-failure does NOT re-fire on a nonzero exit code - measured 2026-09-09,
where AutoMount_VHDX_cloud-cache-disk failed with ExitCode 51 and LastRunTime never
advanced despite RestartCount=3. One missed mount therefore cost the whole session,
and Google Drive's content cache (ContentCachePath=F:\Google) stayed broken until a
human noticed.

Design rules this obeys:
  * Identify volumes by LABEL, never by drive letter - a removable can take F:.
  * Only ever mount from INTERNAL storage. If the label turns up backed by an
    external disk, that is reported and NOT treated as healthy, because the whole
    point is that externals are absent by default.
  * Do nothing when healthy. It must not fight the user: the cloud clients are
    only (re)started when this script actually repaired the mount, so a
    deliberately-quit Dropbox stays quit.

.PARAMETER DryRun
Report what would be done, change nothing. The long form `--DryRun` also works.

.PARAMETER Help
Print help and exit. `-h` / `--help` also work.

.EXAMPLE
pwsh -File .\Tools\Repair-CloudCacheMount.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Label = 'cloud-cache-disk',
    [string]$ExpectedVhd = 'T:\vm\cloud-cache-disk.vhdx',
    [string]$MountTask = 'AutoMount_VHDX_cloud-cache-disk',
    [string]$LauncherTask = 'CloudClients-AfterVHDX',
    [int]$MountWaitSeconds = 180,
    [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudCacheRepair',
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') { $Help = $true }
if (@($CliArgs) -contains '--DryRun') { $DryRun = $true }
if ($Help) {
    $m = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($m.Success) { $m.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if ($DryRun) { $WhatIfPreference = $true }

$logFile = $null
if (-not $DryRun) {
    if (-not (Test-Path $LogRoot)) { New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null }
    $logFile = Join-Path $LogRoot ('{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
}
function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line
    if ($logFile) { Add-Content -LiteralPath $logFile -Value $line }
}

function Get-DiskDriveLetter {
    # Drive letters on a disk, defensively. Under Set-StrictMode -Version 2.0 a
    # property access on a null Get-Partition result throws, so every hop is
    # guarded - this helper must never be the thing that kills the watchdog.
    param([int]$DiskNumber)
    $out = @()
    try {
        foreach ($p in @(Get-Partition -DiskNumber $DiskNumber -ErrorAction SilentlyContinue)) {
            if ($null -ne $p -and $p.PSObject.Properties.Name -contains 'DriveLetter' -and $p.DriveLetter) {
                $out += [string]$p.DriveLetter
            }
        }
    } catch { }
    return $out
}

function Get-CacheVolumeState {
    # Returns: Healthy | Missing | WrongBacking, plus detail.
    $v = Get-Volume -FileSystemLabel $Label -ErrorAction SilentlyContinue
    if (-not $v) { return @{ State = 'Missing'; Volume = $null; Backing = $null } }
    $letter = [string]$v.DriveLetter

    $expected = @(Get-Disk -ErrorAction SilentlyContinue | Where-Object { $_.Location -eq $ExpectedVhd })
    if ($expected.Count -gt 0) {
        if ((Get-DiskDriveLetter -DiskNumber $expected[0].Number) -contains $letter) {
            return @{ State = 'Healthy'; Volume = $v; Backing = $ExpectedVhd }
        }
    }

    # The label exists but is NOT on the expected internal VHDX. This is the branch
    # that fires when the external enclosure comes back and its older copy grabs the
    # label - the one case where doing nothing is much safer than "repairing".
    $backing = '(unknown)'
    try {
        foreach ($d in @(Get-Disk -ErrorAction SilentlyContinue)) {
            if ((Get-DiskDriveLetter -DiskNumber $d.Number) -contains $letter) {
                $backing = if ($d.Location) { [string]$d.Location } else { [string]$d.FriendlyName }
                break
            }
        }
    } catch { }
    return @{ State = 'WrongBacking'; Volume = $v; Backing = $backing }
}

function Test-InteractiveSession {
    # The launcher task runs as 'david / Interactive' and Start-Process'es GUI cloud
    # clients. With nobody logged on that either fails or risks starting them in
    # session 0, where Google Drive is known to misbehave (G:/J: never mount - see
    # the reverted AutoStartOnLogin experiment in Install-CloudClientBootGating.ps1).
    # The MOUNT half is safe headless and still runs; only client-start is gated.
    try {
        $u = (Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).UserName
        if ($u) { return $true }
    } catch { }
    return [bool](Get-Process -Name explorer -ErrorAction SilentlyContinue)
}

$repaired = $false
$s = Get-CacheVolumeState

switch ($s.State) {
    'Healthy' {
        Write-Log ("healthy: {0}: '{1}' backed by {2}" -f $s.Volume.DriveLetter, $Label, $s.Backing)
    }
    'WrongBacking' {
        Write-Log ("'{0}' is mounted at {1}: but backed by '{2}', not {3}. Not repairing - resolve manually." -f `
                $Label, $s.Volume.DriveLetter, $s.Backing, $ExpectedVhd) 'ERROR'
        exit 3
    }
    'Missing' {
        Write-Log ("'{0}' volume is MISSING - attempting repair" -f $Label) 'WARN'
        if (-not (Test-Path -LiteralPath $ExpectedVhd)) {
            Write-Log ("backing file absent: {0} - cannot repair (is it still on the external enclosure?)" -f $ExpectedVhd) 'ERROR'
            exit 2
        }
        if ($PSCmdlet.ShouldProcess($MountTask, 'Start-ScheduledTask')) {
            Start-ScheduledTask -TaskName $MountTask
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            while ($sw.Elapsed.TotalSeconds -lt $MountWaitSeconds) {
                Start-Sleep -Seconds 5
                if ((Get-ScheduledTask -TaskName $MountTask).State -ne 'Running' -and
                    (Get-Volume -FileSystemLabel $Label -ErrorAction SilentlyContinue)) { break }
            }
            $after = Get-CacheVolumeState
            if ($after.State -ne 'Healthy') {
                Write-Log ("repair FAILED - state is still '{0}'" -f $after.State) 'ERROR'
                exit 1
            }
            Write-Log ("repaired: {0}: '{1}' now mounted from {2}" -f $after.Volume.DriveLetter, $Label, $after.Backing)
            $repaired = $true
        }
    }
}

# Only touch the clients when we actually fixed something. A healthy run must be a
# no-op so this never restarts a client the user deliberately closed.
if ($repaired) {
    if (-not (Test-InteractiveSession)) {
        Write-Log ("mount repaired, but no interactive session - NOT starting {0}. The clients will " -f $LauncherTask +
            'come up on the next logon, when its logon trigger fires.') 'WARN'
    }
    elseif ($PSCmdlet.ShouldProcess($LauncherTask, 'Start-ScheduledTask')) {
        Start-ScheduledTask -TaskName $LauncherTask
        Write-Log ("started {0} to bring the cloud clients back onto the restored cache" -f $LauncherTask)
    }
}
elseif ($DryRun) {
    Write-Log 'dry run: nothing to do'
}

exit 0
