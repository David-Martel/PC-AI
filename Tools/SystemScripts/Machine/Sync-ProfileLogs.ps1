#Requires -Version 5.1
<#
.SYNOPSIS
    Syncs local profile logs to OneDrive with conflict handling.

.DESCRIPTION
    This script synchronizes local JSONL profile logs to OneDrive while handling:
    - Multiple machines writing to the same log files
    - Concurrent PowerShell instances
    - OneDrive sync conflicts
    - Network interruptions

    Uses .NET file operations for optimized performance.

.PARAMETER LocalPath
    Local log directory. Default: $env:USERPROFILE\.machine\logs

.PARAMETER OneDrivePath
    OneDrive sync destination. Default: $env:USERPROFILE\OneDrive\Documents\PowerShell\ProfileLogs

.PARAMETER Force
    Force sync even if files appear unchanged.

.PARAMETER Verbose
    Show detailed progress.

.EXAMPLE
    .\Sync-ProfileLogs.ps1
    Syncs logs using default paths.

.EXAMPLE
    .\Sync-ProfileLogs.ps1 -Force -Verbose
    Force sync with detailed output.

.NOTES
    Version: 1.0.0
    Created: 2026-01-05
    Designed for scheduled task execution every 15 minutes.
#>
[CmdletBinding()]
param(
    [string]$LocalPath = (Join-Path $env:USERPROFILE '.machine\logs'),
    [string]$OneDrivePath = (Join-Path $env:USERPROFILE 'OneDrive\Documents\PowerShell\ProfileLogs'),
    [switch]$Force
)

#region Configuration
$script:MachineId = $null
$script:SyncLockTimeout = 5000  # 5 seconds max wait for lock
# A lock older than this is treated as abandoned and broken. Well beyond any
# legitimate profile-log sync, which completes in seconds.
$script:SyncLockStaleMinutes = 30
$script:MaxRetries = 3
#endregion

#region Helper Functions
function Get-MachineId {
    if ($script:MachineId) { return $script:MachineId }

    try {
        $regPath = 'HKLM:\SOFTWARE\Microsoft\Cryptography'
        $guid = (Get-ItemProperty -Path $regPath -Name MachineGuid -ErrorAction Stop).MachineGuid
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($guid)
        $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
        $script:MachineId = [BitConverter]::ToString($hash[0..2]).Replace('-', '').ToLower()
    } catch {
        $script:MachineId = $env:COMPUTERNAME.ToLower().Substring(0, [Math]::Min(6, $env:COMPUTERNAME.Length))
    }

    return $script:MachineId
}

function Get-ProfileLogManifestPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    return "$Path.manifest.json"
}

function Get-ProfileLogEntryIdFromLine {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return $null
    }

    try {
        $entry = $Line | ConvertFrom-Json
        if ($null -ne $entry.ts -and $null -ne $entry.mid -and $null -ne $entry.seq) {
            return "$($entry.ts)_$($entry.mid)_$($entry.seq)"
        }
    } catch {
    }

    return $null
}

function Get-ProfileLogManifestData {
    param(
        [string]$Path,
        [switch]$ForceRefresh
    )

    $manifestPath = Get-ProfileLogManifestPath -Path $Path
    if (-not [System.IO.File]::Exists($Path)) {
        return [pscustomobject]@{
            Exists = $false
            ManifestPath = $manifestPath
            SizeBytes = 0L
            LastEntryId = $null
            LastWriteTimeUtc = $null
        }
    }

    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if (-not $ForceRefresh -and [System.IO.File]::Exists($manifestPath)) {
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            if (
                $null -ne $manifest -and
                [int64]$manifest.sizeBytes -eq $item.Length -and
                [DateTime]$manifest.lastWriteTimeUtc -eq $item.LastWriteTimeUtc
            ) {
                return [pscustomobject]@{
                    Exists = $true
                    ManifestPath = $manifestPath
                    SizeBytes = [int64]$manifest.sizeBytes
                    LastEntryId = [string]$manifest.lastEntryId
                    LastWriteTimeUtc = [DateTime]$manifest.lastWriteTimeUtc
                }
            }
        } catch {
        }
    }

    $tailLine = Get-Content -LiteralPath $Path -Tail 1 -ErrorAction SilentlyContinue | Select-Object -Last 1
    return [pscustomobject]@{
        Exists = $true
        ManifestPath = $manifestPath
        SizeBytes = [int64]$item.Length
        LastEntryId = Get-ProfileLogEntryIdFromLine -Line $tailLine
        LastWriteTimeUtc = $item.LastWriteTimeUtc
    }
}

function Write-ProfileLogManifestData {
    param([string]$Path)

    $manifestPath = Get-ProfileLogManifestPath -Path $Path
    if (-not [System.IO.File]::Exists($Path)) {
        if ($manifestPath -and [System.IO.File]::Exists($manifestPath)) {
            [System.IO.File]::Delete($manifestPath)
        }
        return
    }

    $info = Get-ProfileLogManifestData -Path $Path -ForceRefresh
    $manifest = [ordered]@{
        version = 1
        logFileName = [System.IO.Path]::GetFileName($Path)
        sizeBytes = $info.SizeBytes
        lastEntryId = $info.LastEntryId
        lastWriteTimeUtc = $info.LastWriteTimeUtc.ToString('o')
        updatedUtc = [DateTime]::UtcNow.ToString('o')
        machineId = Get-MachineId
    } | ConvertTo-Json -Depth 4

    $tempPath = "$manifestPath.sync.tmp"
    [System.IO.File]::WriteAllText($tempPath, $manifest, [System.Text.UTF8Encoding]::new($false))
    if ([System.IO.File]::Exists($manifestPath)) {
        [System.IO.File]::Delete($manifestPath)
    }
    [System.IO.File]::Move($tempPath, $manifestPath)
}

function Test-ProfileLogManifestMatch {
    param(
        [Parameter(Mandatory)]
        $SourceManifest,

        [Parameter(Mandatory)]
        $DestinationManifest
    )

    return (
        $SourceManifest.Exists -and
        $DestinationManifest.Exists -and
        $SourceManifest.SizeBytes -eq $DestinationManifest.SizeBytes -and
        $SourceManifest.LastEntryId -eq $DestinationManifest.LastEntryId
    )
}

function Merge-JsonlFiles {
    <#
    .SYNOPSIS
        Merges two JSONL files, deduplicating by timestamp+machine+sequence.
    .DESCRIPTION
        Uses .NET for fast file I/O. Handles conflicts by keeping all unique entries.
    #>
    param(
        [string]$SourcePath,
        [string]$DestPath
    )

    # Read source entries
    $sourceLines = @()
    if ([System.IO.File]::Exists($SourcePath)) {
        $sourceLines = [System.IO.File]::ReadAllLines($SourcePath, [System.Text.Encoding]::UTF8)
    }

    # Read destination entries
    $destLines = @()
    if ([System.IO.File]::Exists($DestPath)) {
        $destLines = [System.IO.File]::ReadAllLines($DestPath, [System.Text.Encoding]::UTF8)
    }

    # Build hash set of unique entry keys (ts+mid+seq)
    $seenEntries = [System.Collections.Generic.HashSet[string]]::new()
    $mergedLines = [System.Collections.Generic.List[string]]::new()

    # Process destination first (it has priority for existing entries)
    foreach ($line in $destLines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }

        try {
            $entry = $line | ConvertFrom-Json
            $key = "$($entry.ts)_$($entry.mid)_$($entry.seq)"

            if ($seenEntries.Add($key)) {
                $mergedLines.Add($line)
            }
        } catch {
            # Keep malformed lines
            $mergedLines.Add($line)
        }
    }

    # Add source entries that don't exist in destination
    $newEntries = 0
    foreach ($line in $sourceLines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }

        try {
            $entry = $line | ConvertFrom-Json
            $key = "$($entry.ts)_$($entry.mid)_$($entry.seq)"

            if ($seenEntries.Add($key)) {
                $mergedLines.Add($line)
                $newEntries++
            }
        } catch {
            # Keep malformed lines if they're unique
            if ($seenEntries.Add($line)) {
                $mergedLines.Add($line)
                $newEntries++
            }
        }
    }

    return @{
        Lines = $mergedLines
        NewEntries = $newEntries
        TotalEntries = $mergedLines.Count
    }
}

function Write-SyncLog {
    <#
    .SYNOPSIS
        Logs sync operations to a dedicated sync log.
    #>
    param(
        [string]$Message,
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level = 'Info'
    )

    $timestamp = [DateTime]::UtcNow.ToString('o')
    $logEntry = "[$timestamp][$Level][Sync-$(Get-MachineId)] $Message"

    $syncLogPath = Join-Path $LocalPath 'sync.log'

    try {
        [System.IO.File]::AppendAllText($syncLogPath, "$logEntry`n", [System.Text.Encoding]::UTF8)
    } catch {
        # Silent failure
    }

    if ($VerbosePreference -eq 'Continue') {
        $color = switch ($Level) {
            'Warning' { 'Yellow' }
            'Error' { 'Red' }
            default { 'Gray' }
        }
        Write-Host $logEntry -ForegroundColor $color
    }
}

function Acquire-SyncLock {
    <#
    .SYNOPSIS
        Acquires a lock file for sync operation.
    #>
    param([string]$LockPath)

    $lockFile = Join-Path $LockPath '.sync.lock'
    $waited = 0
    $interval = 100
    $breakAttempted = $false

    while ($waited -lt $script:SyncLockTimeout) {
        try {
            # Try to create exclusive lock file
            $stream = [System.IO.File]::Open(
                $lockFile,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )

            # Write lock info
            $lockInfo = @{
                Machine = Get-MachineId
                Pid = $PID
                Time = [DateTime]::UtcNow.ToString('o')
            } | ConvertTo-Json -Compress

            $bytes = [System.Text.Encoding]::UTF8.GetBytes($lockInfo)
            $stream.Write($bytes, 0, $bytes.Length)
            # Flush so the lock is readable by anyone diagnosing a stuck sync.
            # Without this the bytes sit in the stream buffer and a killed
            # process leaves a zero-byte lock nobody can attribute -- which is
            # exactly what was found on this machine.
            $stream.Flush()

            return $stream
        } catch [System.IO.IOException] {
            # The lock exists. Before waiting on it, decide whether anyone still
            # holds it. CreateNew fails forever against an abandoned lock file,
            # so without this check one killed run poisons every future run
            # permanently. Measured on this machine: a lock left at 2026-02-21
            # failed every hourly sync for over six months, entirely silently.
            if (-not $breakAttempted) {
                $breakAttempted = $true
                if (Test-StaleSyncLock -LockFile $lockFile) {
                    try {
                        [System.IO.File]::Delete($lockFile)
                        Write-Verbose "Broke a stale sync lock at $lockFile"
                        continue   # retry immediately against the cleared path
                    } catch {
                        # Someone else won the race and recreated it; fall
                        # through and wait like a normal contender.
                        Write-Verbose "Could not break stale lock: $($_.Exception.Message)"
                    }
                }
            }
            Start-Sleep -Milliseconds $interval
            $waited += $interval
        }
    }

    return $null
}

function Test-StaleSyncLock {
    <#
    .SYNOPSIS
        Decide whether an existing lock file has been abandoned.
    .DESCRIPTION
        Age is the primary signal and the only universally valid one: this lock
        lives in OneDrive and is shared between machines, so a PID from another
        host means nothing here. A lock older than SyncLockStaleMinutes is
        treated as abandoned regardless of content.

        Content is used only to strengthen the decision, never to keep a lock
        alive -- an unreadable or empty lock is exactly what a killed process
        leaves behind, so it must not be allowed to block forever.
    #>
    param([string]$LockFile)

    try {
        $item = Get-Item -LiteralPath $LockFile -Force -ErrorAction Stop
    } catch {
        return $false   # vanished between the failure and here; just retry
    }

    $ageMinutes = ([DateTime]::UtcNow - $item.LastWriteTimeUtc).TotalMinutes
    if ($ageMinutes -lt $script:SyncLockStaleMinutes) { return $false }

    # Old enough to be abandoned. If it happens to name a live process on THIS
    # machine, respect it anyway -- a genuinely long sync should not be broken.
    try {
        $raw = Get-Content -LiteralPath $LockFile -Raw -ErrorAction Stop
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            $info = $raw | ConvertFrom-Json -ErrorAction Stop
            $sameMachine = ($info.PSObject.Properties.Name -contains 'Machine') -and
                           ($info.Machine -eq (Get-MachineId))
            $hasPid = $info.PSObject.Properties.Name -contains 'Pid'
            if ($sameMachine -and $hasPid) {
                if (Get-Process -Id ([int]$info.Pid) -ErrorAction SilentlyContinue) {
                    return $false
                }
            }
        }
    } catch {
        # Empty or malformed lock. Age already says abandoned.
    }

    return $true
}

function Release-SyncLock {
    param(
        [System.IO.FileStream]$LockStream,
        [string]$LockPath
    )

    if ($LockStream) {
        $LockStream.Close()
        $LockStream.Dispose()
    }

    $lockFile = Join-Path $LockPath '.sync.lock'
    if ([System.IO.File]::Exists($lockFile)) {
        try {
            [System.IO.File]::Delete($lockFile)
        } catch {
            # Ignore - may be held by another process
        }
    }
}
#endregion

#region Main Sync Logic
function Start-ProfileLogSync {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$ForceSync
    )

    Write-SyncLog "Starting sync: $Source -> $Destination"

    # Verify source exists
    if (-not [System.IO.Directory]::Exists($Source)) {
        Write-SyncLog "Source directory does not exist: $Source" -Level Warning
        return @{ Success = $false; Message = "Source not found" }
    }

    # Create destination if needed
    if (-not [System.IO.Directory]::Exists($Destination)) {
        try {
            [System.IO.Directory]::CreateDirectory($Destination) | Out-Null
            Write-SyncLog "Created destination directory: $Destination"
        } catch {
            Write-SyncLog "Failed to create destination: $_" -Level Error
            return @{ Success = $false; Message = "Cannot create destination" }
        }
    }

    # Acquire sync lock
    $lockStream = Acquire-SyncLock -LockPath $Destination
    if (-not $lockStream) {
        Write-SyncLog "Could not acquire sync lock - another sync in progress" -Level Warning
        return @{ Success = $false; Message = "Sync lock held" }
    }

    try {
        $stats = @{
            FilesSynced = 0
            NewEntries = 0
            Conflicts = 0
            Errors = 0
        }

        # Sync only canonical merged profile logs. Spool files stay local.
        $sourceFiles = [System.IO.Directory]::GetFiles($Source, 'profile_*.jsonl')

        foreach ($sourceFile in $sourceFiles) {
            $fileName = [System.IO.Path]::GetFileName($sourceFile)

            # Skip temp files
            if ($fileName -match '\.tmp$') { continue }

            $destFile = Join-Path $Destination $fileName

            $sourceManifest = Get-ProfileLogManifestData -Path $sourceFile
            $destManifest = Get-ProfileLogManifestData -Path $destFile

            if (-not $ForceSync -and (Test-ProfileLogManifestMatch -SourceManifest $sourceManifest -DestinationManifest $destManifest)) {
                if (-not [System.IO.File]::Exists($destManifest.ManifestPath)) {
                    Write-ProfileLogManifestData -Path $destFile
                }
                Write-SyncLog "Skipping $fileName - unchanged"
                continue
            }

            # Merge files
            try {
                $mergeResult = Merge-JsonlFiles -SourcePath $sourceFile -DestPath $destFile

                if ($mergeResult.NewEntries -gt 0 -or $ForceSync) {
                    # Write merged content atomically
                    $tempDest = "$destFile.sync.tmp"
                    [System.IO.File]::WriteAllLines(
                        $tempDest,
                        $mergeResult.Lines,
                        [System.Text.Encoding]::UTF8
                    )

                    # Atomic replace
                    if ([System.IO.File]::Exists($destFile)) {
                        [System.IO.File]::Delete($destFile)
                    }
                    [System.IO.File]::Move($tempDest, $destFile)
                    Write-ProfileLogManifestData -Path $destFile

                    $stats.FilesSynced++
                    $stats.NewEntries += $mergeResult.NewEntries

                    Write-SyncLog "Synced $fileName (+$($mergeResult.NewEntries) entries, $($mergeResult.TotalEntries) total)"
                }
            } catch {
                $stats.Errors++
                Write-SyncLog "Failed to sync $fileName : $_" -Level Error
            }
        }

        # Handle OneDrive conflict files (files with machine name in them)
        $conflictFiles = [System.IO.Directory]::GetFiles($Destination, '*-*.jsonl') |
            Where-Object { $_ -match 'profile_\d{4}-\d{2}-[^.]+\.jsonl$' }

        foreach ($conflictFile in $conflictFiles) {
            try {
                # Extract base filename
                $conflictName = [System.IO.Path]::GetFileName($conflictFile)
                if ($conflictName -match '^(profile_\d{4}-\d{2})-') {
                    $baseName = $Matches[1] + '.jsonl'
                    $baseFile = Join-Path $Destination $baseName

                    # Merge conflict file into base
                    $mergeResult = Merge-JsonlFiles -SourcePath $conflictFile -DestPath $baseFile

                    $tempBase = "$baseFile.merge.tmp"
                    [System.IO.File]::WriteAllLines($tempBase, $mergeResult.Lines, [System.Text.Encoding]::UTF8)

                    if ([System.IO.File]::Exists($baseFile)) {
                        [System.IO.File]::Delete($baseFile)
                    }
                    [System.IO.File]::Move($tempBase, $baseFile)
                    Write-ProfileLogManifestData -Path $baseFile

                    # Move conflict file to resolved folder
                    $resolvedDir = Join-Path $Destination 'resolved-conflicts'
                    if (-not [System.IO.Directory]::Exists($resolvedDir)) {
                        [System.IO.Directory]::CreateDirectory($resolvedDir) | Out-Null
                    }
                    $resolvedPath = Join-Path $resolvedDir $conflictName
                    [System.IO.File]::Move($conflictFile, $resolvedPath)

                    $stats.Conflicts++
                    Write-SyncLog "Resolved conflict: $conflictName (+$($mergeResult.NewEntries) entries)"
                }
            } catch {
                Write-SyncLog "Failed to resolve conflict $conflictFile : $_" -Level Warning
            }
        }

        Write-SyncLog "Sync complete: $($stats.FilesSynced) files, +$($stats.NewEntries) entries, $($stats.Conflicts) conflicts resolved"

        return @{
            Success = $true
            Stats = $stats
        }
    } finally {
        Release-SyncLock -LockStream $lockStream -LockPath $Destination
    }
}
#endregion

#region Entry Point
$result = Start-ProfileLogSync -Source $LocalPath -Destination $OneDrivePath -ForceSync:$Force

if ($result.Success) {
    Write-Host "Sync completed successfully." -ForegroundColor Green
    if ($result.Stats) {
        Write-Host "  Files synced: $($result.Stats.FilesSynced)"
        Write-Host "  New entries: $($result.Stats.NewEntries)"
        Write-Host "  Conflicts resolved: $($result.Stats.Conflicts)"
    }
} else {
    Write-Host "Sync failed: $($result.Message)" -ForegroundColor Red
    exit 1
}
#endregion
