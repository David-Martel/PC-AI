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

.EXAMPLE
pwsh -File .\Tools\Start-CloudClientsAfterVHDX.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ExpectedLabel = 'cloud-cache-disk',
    [string]$ExpectedDriveLetter = 'F',
    [int]$TimeoutSeconds = 180,
    [int]$PollSeconds = 3,
    [switch]$RequireMountLog,
    [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
    [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudClientStart'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $LogRoot)) { New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null }
$logFile = Join-Path $LogRoot ('{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message
    Write-Host $line
    Add-Content -LiteralPath $logFile -Value $line
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
    $newest = Get-ChildItem $MountLogRoot -Filter '*.result.json' -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $newest) {
        Write-Log 'RequireMountLog set but no mount result JSON found.' 'WARN'
    } else {
        $res = Get-Content $newest.FullName -Raw | ConvertFrom-Json
        if ($res.ExitCode -ne 0) {
            Write-Log "Mount reported ExitCode $($res.ExitCode) ($($newest.Name)) - clients NOT started." 'ERROR'
            exit 3
        }
        Write-Log "Mount log OK (ExitCode 0, $($newest.Name))"
    }
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
