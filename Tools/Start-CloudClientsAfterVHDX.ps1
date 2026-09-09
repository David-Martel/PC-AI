#Requires -Version 5.1
<#
.SYNOPSIS
Starts cloud sync clients only after the F: cloud-cache-disk VHDX is mounted.

.DESCRIPTION
Run at logon in the user's session. Waits for the cloud-cache VHDX volume to
appear and validate, then launches the configured cloud clients.

If the volume never appears the clients are deliberately NOT started and the
script exits nonzero. Starting a sync client whose cache/root is missing is the
failure that silently relocates a sync root onto C: - which is exactly what this
gating exists to prevent.

Validation is by volume LABEL, never by drive letter alone: a USB stick can take
F:. Optionally the AutoMount task's own result JSON is checked for ExitCode 0 so
a Degraded mount does not count as success.

.PARAMETER ExpectedLabel
Volume label that identifies the cloud cache disk.

.PARAMETER TimeoutSeconds
How long to wait for the volume before giving up.

.PARAMETER RequireMountLog
Also require the newest AutoMount result JSON to report ExitCode 0.

.PARAMETER WhatIf
Validate and report without launching anything.

.PARAMETER DryRun
Non-mutating preview: validate the volume and report which clients would be
launched, without launching any and without creating the log directory or
writing a log file. The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Start-CloudClientsAfterVHDX.ps1 -WhatIf
.EXAMPLE
pwsh -File .\Tools\Start-CloudClientsAfterVHDX.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ExpectedLabel = 'cloud-cache-disk',
    [string]$ExpectedDriveLetter = 'F',
    [int]$TimeoutSeconds = 180,
    [int]$PollSeconds = 3,
    [switch]$RequireMountLog,
    [switch]$AllowElevated,
    [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
    [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudClientStart',
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# CLI contract (AGENTS.md). Handled before the log directory is created, so
# `-h` and `-DryRun` do not leave a directory behind on a machine that was only
# ever asked for help.
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
if ($DryRun) { $WhatIfPreference = $true }

# The log file is the one side effect NOT behind ShouldProcess, so -DryRun has
# to suppress it explicitly. AGENTS.md requires dry run to suppress file writes,
# and a "preview" that creates a directory and a log file is not a preview.
$logFile = $null
if (-not $DryRun) {
    if (-not (Test-Path $LogRoot)) { New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null }
    $logFile = Join-Path $LogRoot ('{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message
    Write-Host $line
    if ($logFile) { Add-Content -LiteralPath $logFile -Value $line }
}

# Clients to launch once the volume is ready. Order matters only cosmetically.
$clients = @(
    @{ Name    = 'GoogleDriveFS'
       Process = 'GoogleDriveFS'
       Path    = 'C:\Program Files\Google\Drive File Stream\130.0.2.0\GoogleDriveFS.exe'
       Args    = '--startup_mode'
       Glob    = 'C:\Program Files\Google\Drive File Stream\*\GoogleDriveFS.exe' },
    @{ Name    = 'Dropbox'
       Process = 'Dropbox'
       Path    = 'C:\Program Files (x86)\Dropbox\Client\Dropbox.exe'
       Args    = '/systemstartup'
       Glob    = 'C:\Program Files (x86)\Dropbox\Client\Dropbox.exe' }
)

# Cloud sync clients refuse to run elevated (Google Drive exits immediately, and
# an elevated Dropbox would own the sync root as Administrator). The scheduled
# task runs Limited; this guard catches a manual run from an elevated shell.
$isElevated = (New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isElevated -and -not $AllowElevated) {
    Write-Log 'Running ELEVATED. Cloud clients must start unelevated - Google Drive exits on launch.' 'ERROR'
    Write-Log 'Use the CloudClients-AfterVHDX scheduled task, or pass -AllowElevated to override.' 'ERROR'
    exit 4
}

Write-Log "Waiting for volume '$ExpectedLabel' (max ${TimeoutSeconds}s)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$vol = $null
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    $v = Get-Volume -ErrorAction SilentlyContinue |
         Where-Object { $_.FileSystemLabel -eq $ExpectedLabel -and $_.DriveLetter }
    if ($v) {
        # Prove it is really writable/browsable, not just enumerated.
        $root = '{0}:\' -f $v.DriveLetter
        if (Test-Path $root) { $vol = $v; break }
    }
    Start-Sleep -Seconds $PollSeconds
}
$sw.Stop()

if (-not $vol) {
    Write-Log "Volume '$ExpectedLabel' did NOT appear within ${TimeoutSeconds}s." 'ERROR'
    Write-Log 'Cloud clients NOT started (prevents sync-root relocation onto C:).' 'ERROR'
    exit 2
}

$letter = [string]$vol.DriveLetter
Write-Log ("Volume ready: {0}: label='{1}' free={2:N0} GB after {3:N1}s" -f `
    $letter, $vol.FileSystemLabel, ($vol.SizeRemaining / 1GB), $sw.Elapsed.TotalSeconds)

if ($letter -ne $ExpectedDriveLetter) {
    Write-Log "Volume mounted at ${letter}: but expected ${ExpectedDriveLetter}: - continuing, paths may be wrong." 'WARN'
}

if ($RequireMountLog) {
    # The result JSON must be from THIS boot. Picking "newest by LastWriteTime"
    # alone is not enough: the mount task and this task both fire at logon, so on
    # a fast auto-logon this boot's JSON may not exist yet and the newest file on
    # disk is the PREVIOUS boot's. If that one succeeded, a stale ExitCode 0 would
    # satisfy the gate on a boot where the mount actually failed - the exact
    # inversion this gate exists to prevent. Observed 2026-09-09, when this
    # directory held an 08:00:22 success alongside two later failures.
    $bootTime = (Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue).LastBootUpTime
    $newest = Get-ChildItem $MountLogRoot -Filter '*.result.json' -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $newest) {
        # Absence of evidence is not evidence of a good mount.
        Write-Log 'RequireMountLog set but no mount result JSON found - clients NOT started.' 'ERROR'
        exit 3
    }
    $res = Get-Content $newest.FullName -Raw | ConvertFrom-Json

    $mountStartedAt = $null
    if ($res.PSObject.Properties.Name -contains 'StartedAt') {
        try { $mountStartedAt = [datetime]$res.StartedAt } catch { $mountStartedAt = $null }
    }
    if (-not $bootTime) {
        Write-Log 'Could not read LastBootUpTime - cannot prove mount log freshness.' 'WARN'
    } elseif (-not $mountStartedAt) {
        Write-Log "Mount log $($newest.Name) has no usable StartedAt - clients NOT started." 'ERROR'
        exit 3
    } elseif ($mountStartedAt -lt $bootTime) {
        Write-Log ("Mount log {0} is STALE (StartedAt {1:yyyy-MM-dd HH:mm:ss} predates boot {2:yyyy-MM-dd HH:mm:ss}) - clients NOT started." -f `
            $newest.Name, $mountStartedAt, $bootTime) 'ERROR'
        exit 3
    }

    if ($res.ExitCode -ne 0) {
        Write-Log "Mount reported ExitCode $($res.ExitCode) ($($newest.Name)) - clients NOT started." 'ERROR'
        exit 3
    }
    Write-Log "Mount log OK (ExitCode 0, this boot, $($newest.Name))"
}

$started = 0; $skipped = 0; $failed = 0
foreach ($c in $clients) {
    if (Get-Process -Name $c.Process -ErrorAction SilentlyContinue) {
        Write-Log "$($c.Name): already running - skip"
        $skipped++
        continue
    }
    $exe = $c.Path
    if (-not (Test-Path $exe)) {
        # version-numbered install dirs move on update; fall back to newest match
        $cand = Get-ChildItem $c.Glob -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($cand) { $exe = $cand.FullName; Write-Log "$($c.Name): resolved to $exe" }
        else { Write-Log "$($c.Name): executable not found - skip" 'WARN'; $failed++; continue }
    }
    if ($PSCmdlet.ShouldProcess($c.Name, "Start $exe $($c.Args)")) {
        try {
            Start-Process -FilePath $exe -ArgumentList $c.Args -ErrorAction Stop
            Write-Log "$($c.Name): started"
            $started++
        } catch {
            Write-Log "$($c.Name): FAILED to start - $($_.Exception.Message)" 'ERROR'
            $failed++
        }
    }
}

Write-Log "Done. started=$started skipped=$skipped failed=$failed"
if ($failed -gt 0) { exit 1 }
exit 0
