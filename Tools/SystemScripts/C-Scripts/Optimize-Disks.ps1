
#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Smart TRIM + Optimize for all fixed NTFS/ReFS volumes, with safe cache flush.

.DESCRIPTION
  - Finds fixed NTFS/ReFS volumes (incl. Dev Drive and mounted VHDX).
  - Detects NVMe/SSD/HDD and chooses TRIM vs default optimize.
  - Flushes write cache via Write-VolumeCache; falls back to Sysinternals Sync or Win32 FlushFileBuffers.
  - Optional pre-analysis and weekly Scheduled Task creation.

.PARAMETER AnalyzeFirst
  Analyze volumes before optimizing.

.PARAMETER ShowOnly
  Discovery mode: list actions, do not change anything.

.PARAMETER SkipFlushBefore
  Skip flushing the write cache before optimizing.

.PARAMETER SkipFlushAfter
  Skip flushing the write cache after optimizing.

.PARAMETER RegisterWeeklyTask
  Create/refresh a weekly scheduled task to run this script.

.PARAMETER TaskTime
  Time (24h HH:mm) for the scheduled task.

.NOTES
  Optimize-Volume behavior: SSD -> -ReTrim; HDD/Fixed VHD -> defrag; thin-provisioned -> slab + retrim. (docs)
#>

[CmdletBinding()]
param(
    [switch]$AnalyzeFirst,
    [switch]$ShowOnly,
    [switch]$RegisterWeeklyTask,
    [string]$TaskTime = '03:30',
    [switch]$SkipFlushBefore,
    [switch]$SkipFlushAfter
)

function Test-Admin {
    $isAdmin = ([Security.Principal.WindowsPrincipal]
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
    if (-not $isAdmin) { throw "This script must be run as Administrator." }
}

function Get-TrimStatus {
    try {
        $out = & fsutil behavior query DisableDeleteNotify 2>$null
        $out -split "`r?`n" | Where-Object { $_ -match 'DisableDeleteNotify' }
    } catch {
        @("Unable to query TRIM (fsutil).")
    }
}

function Get-VolumeInfo {
    $vols = Get-Volume | Where-Object {
        $_.DriveType -eq 'Fixed' -and $_.FileSystem -in @('NTFS','ReFS')
    }

    foreach ($v in $vols) {
        $diskNumber = $null; $busType = $null; $mediaType = $null
        try {
            if ($v.DriveLetter) {
                $part = Get-Partition -DriveLetter $v.DriveLetter -ErrorAction Stop
                $disk = Get-Disk -Number $part.DiskNumber -ErrorAction Stop
                $diskNumber = $disk.Number
                $busType    = $disk.BusType      # NVMe, SATA, Virtual, iSCSI, etc.
                $mediaType  = $disk.MediaType    # SSD, HDD, Unspecified
            }
        } catch {}

        $isSsdLike = ($mediaType -eq 'SSD' -or $busType -eq 'NVMe' -or $busType -eq 'Virtual')

        [pscustomobject]@{
            DriveLetter   = $v.DriveLetter
            Path          = $v.Path                # e.g. C:\ or \\?\Volume{GUID}\
            FileSystem    = $v.FileSystem
            Label         = $v.FileSystemLabel
            SizeGB        = [math]::Round($v.Size/1GB,2)
            FreeGB        = [math]::Round($v.SizeRemaining/1GB,2)
            HealthStatus  = $v.HealthStatus
            DiskNumber    = $diskNumber
            BusType       = $busType
            MediaType     = $mediaType
            IsSsdLike     = $isSsdLike
        }
    } | Sort-Object DriveLetter, Path
}

function Optimize-OneVolume {
    param(
        [Parameter(Mandatory)][pscustomobject]$Vol,
        [switch]$AnalyzeFirst,
        [switch]$ShowOnly
    )

    $target = if ($Vol.DriveLetter) { "$($Vol.DriveLetter):" } else { $Vol.Path }

    if (-not $target) {
        Write-Warning "Skipping volume with no drive letter/path (Label: '$($Vol.Label)')"
        return
    }

    if ($AnalyzeFirst) {
        Write-Host "Analyzing volume $target ..." -ForegroundColor Cyan
        if (-not $ShowOnly) {
            try {
                if ($Vol.DriveLetter) { Optimize-Volume -DriveLetter $Vol.DriveLetter -Analyze -Verbose -ErrorAction Stop }
                else                  { Optimize-Volume -Path $Vol.Path -Analyze -Verbose -ErrorAction Stop }
            } catch {
                Write-Warning "Analyze failed on $target: $($_.Exception.Message)"
            }
        }
    }

    if ($Vol.IsSsdLike) {
        Write-Host "TRIM (ReTrim) on $target [FS=$($Vol.FileSystem), Bus=$($Vol.BusType), Media=$($Vol.MediaType)]" -ForegroundColor Green
        if (-not $ShowOnly) {
            try {
                if ($Vol.DriveLetter) { Optimize-Volume -DriveLetter $Vol.DriveLetter -ReTrim -Verbose -ErrorAction Stop }
                else                  { Optimize-Volume -Path $Vol.Path -ReTrim -Verbose -ErrorAction Stop }
            } catch {
                Write-Warning "ReTrim failed on $target. Falling back to default Optimize. $_"
                try {
                    if ($Vol.DriveLetter) { Optimize-Volume -DriveLetter $Vol.DriveLetter -Verbose -ErrorAction Stop }
                    else                  { Optimize-Volume -Path $Vol.Path -Verbose -ErrorAction Stop }
                } catch {
                    Write-Error "Optimize failed on $target: $($_.Exception.Message)"
                }
            }
        }
    } else {
        Write-Host "Default Optimize on $target [likely HDD/unspecified]" -ForegroundColor Yellow
        if (-not $ShowOnly) {
            try {
                if ($Vol.DriveLetter) { Optimize-Volume -DriveLetter $Vol.DriveLetter -Verbose -ErrorAction Stop }
                else                  { Optimize-Volume -Path $Vol.Path -Verbose -ErrorAction Stop }
            } catch {
                Write-Error "Optimize failed on $target: $($_.Exception.Message)"
            }
        }
    }
}

function Flush-WriteCache {
    param(
        [Parameter(Mandatory)][pscustomobject[]]$Volumes,
        [switch]$ShowOnly
    )

    $withLetter = $Volumes | Where-Object { $_.DriveLetter }
    $noLetter   = $Volumes | Where-Object { -not $_.DriveLetter -and $_.Path }

    Write-Host "Flushing file-system write cache..." -ForegroundColor Cyan

    # Preferred: Write-VolumeCache (Storage module)  — flushes FS cache to disk
    $wvc = Get-Command -Name Write-VolumeCache -ErrorAction SilentlyContinue
    if ($wvc) {
        if ($ShowOnly) {
            if ($withLetter) { Write-Host "Would run: Write-VolumeCache -DriveLetter $($withLetter.DriveLetter -join ',')" -ForegroundColor DarkCyan }
            foreach ($v in $noLetter) { Write-Host "Would run: Write-VolumeCache -Path $($v.Path)" -ForegroundColor DarkCyan }
            return
        }
        try {
            if ($withLetter) { Write-VolumeCache -DriveLetter ($withLetter.DriveLetter) -ErrorAction Continue }
            foreach ($v in $noLetter) { Write-VolumeCache -Path $v.Path -ErrorAction Continue }
            return
        } catch {
            Write-Warning "Write-VolumeCache encountered an error. Trying fallbacks..."
        }
    }

    # Fallback 1: Sysinternals Sync (flushes cache for specified drive letters)
    $sync = Get-Command -Name sync.exe -ErrorAction SilentlyContinue
    if ($sync -and $withLetter) {
        $letters = ($withLetter.DriveLetter | ForEach-Object { $_.ToString().ToLower() })
        if ($ShowOnly) {
            Write-Host "Would run: $($sync.Source) $($letters -join ' ')" -ForegroundColor DarkCyan
        } else {
            try { & $sync.Source @($letters) | Out-Null } catch { Write-Warning "Sync.exe fallback failed: $($_.Exception.Message)" }
        }
        # sync.exe can't target mount paths; keep going for no-letter volumes.
        if (-not $noLetter) { return }
    }

    # Fallback 2: Win32 FlushFileBuffers on \\.\X: or \\?\Volume{GUID}
    if ($ShowOnly) {
        foreach ($v in $withLetter) { Write-Host "Would P/Invoke FlushFileBuffers on \\.\$($v.DriveLetter):" -ForegroundColor DarkCyan }
        foreach ($v in $noLetter)   { Write-Host "Would P/Invoke FlushFileBuffers on $($v.Path.TrimEnd('\'))" -ForegroundColor DarkCyan }
        return
    }

    $cs = @"
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class NativeFlush {
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  public static extern SafeFileHandle CreateFile(
      string lpFileName,
      uint dwDesiredAccess,
      uint dwShareMode,
      IntPtr lpSecurityAttributes,
      uint dwCreationDisposition,
      uint dwFlagsAndAttributes,
      IntPtr hTemplateFile);

  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool FlushFileBuffers(SafeFileHandle hFile);
}
"@
    try { Add-Type -TypeDefinition $cs -Language CSharp -ErrorAction Stop | Out-Null } catch {}

    $GENERIC_WRITE    = 0x40000000
    $SHARE_READWRITE_DELETE = 0x00000007
    $OPEN_EXISTING    = 3

    # Drive-letter volumes
    foreach ($v in $withLetter) {
        $volHandlePath = "\\.\$($v.DriveLetter):"
        $h = [NativeFlush]::CreateFile($volHandlePath, $GENERIC_WRITE, $SHARE_READWRITE_DELETE, [IntPtr]::Zero, $OPEN_EXISTING, 0, [IntPtr]::Zero)
        if ($h -and -not $h.IsInvalid) {
            try { [void][NativeFlush]::FlushFileBuffers($h) } catch {}
            $h.Dispose()
        } else {
            Write-Warning "Could not open volume handle for $volHandlePath (Flush skipped)."
        }
    }

    # Mount-path / volume-GUID volumes
    foreach ($v in $noLetter) {
        # Convert "\\?\Volume{GUID}\" to "\\?\Volume{GUID}" (no trailing '\')
        $guidPath = $v.Path.TrimEnd('\')
        $h2 = [NativeFlush]::CreateFile($guidPath, $GENERIC_WRITE, $SHARE_READWRITE_DELETE, [IntPtr]::Zero, $OPEN_EXISTING, 0, [IntPtr]::Zero)
        if ($h2 -and -not $h2.IsInvalid) {
            try { [void][NativeFlush]::FlushFileBuffers($h2) } catch {}
            $h2.Dispose()
        } else {
            Write-Warning "Could not open volume handle for $guidPath (Flush skipped)."
        }
    }
}

function Register-WeeklyTask {
    param([string]$TaskTime = '03:30')

    $scriptPath = $PSCommandPath
    if (-not (Test-Path $scriptPath)) {
        throw "Cannot determine script path for scheduling. Save the script and rerun with -RegisterWeeklyTask."
    }

    $taskName = 'Weekly SSD TRIM and Disk Optimize'
    Write-Host "Creating/Updating Scheduled Task: $taskName at $TaskTime weekly" -ForegroundColor Cyan

    $action    = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
    $trigger   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At ([datetime]::ParseExact($TaskTime, 'HH:mm', $null))
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $task      = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal

    try {
        Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
        Write-Host "Scheduled Task registered." -ForegroundColor Green
    } catch {
        Write-Error "Failed to register Scheduled Task: $($_.Exception.Message)"
    }
}

# ---------------- MAIN ----------------
try { Test-Admin } catch { Write-Error $_.Exception.Message; return }

Write-Host "TRIM status (NTFS/ReFS):" -ForegroundColor Cyan
Get-TrimStatus | ForEach-Object { Write-Host "  $_" }

$volInfo = Get-VolumeInfo

Write-Host "`nDiscovered fixed NTFS/ReFS volumes:" -ForegroundColor Cyan
$volInfo | Format-Table DriveLetter, Label, FileSystem, BusType, MediaType, SizeGB, FreeGB, HealthStatus -AutoSize

if ($ShowOnly) { Write-Host "`nSHOW-ONLY mode: No changes will be made." -ForegroundColor Yellow }

# Flush BEFORE optimize unless skipped
if (-not $SkipFlushBefore) {
    Flush-WriteCache -Volumes $volInfo -ShowOnly:$ShowOnly
}

foreach ($v in $volInfo) {
    Optimize-OneVolume -Vol $v -AnalyzeFirst:$AnalyzeFirst -ShowOnly:$ShowOnly
}

# Flush AFTER optimize unless skipped
if (-not $SkipFlushAfter) {
    Flush-WriteCache -Volumes $volInfo -ShowOnly:$ShowOnly
}

if ($RegisterWeeklyTask) { Register-WeeklyTask -TaskTime $TaskTime }

Write-Host "`nDone." -ForegroundColor Cyan
``
