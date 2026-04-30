#Requires -Version 7.0
<#
.SYNOPSIS
Checks current VHD mount and Filter Manager health.

.DESCRIPTION
Validates expected VHD files, attachment state, mounted volumes, AutoMount task
results, recent FilterManager Event ID 3 entries, and boot diagnostics
freshness.

.PARAMETER OutputJson
Optional machine-readable report path. It is not written when DryRun is set.

.PARAMETER DryRun
Run checks without writing the optional report file. The long CLI form
`--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding()]
param(
    [switch]$PassThru,
    [switch]$FailOnIssue,
    [int]$SinceMinutes = 0,
    [string]$OutputJson,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

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
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

$expectedVhds = @(
    [pscustomobject]@{ Name = 'cloud-cache-disk'; Path = 'T:\vm\cloud-cache-disk.vhdx'; ExpectedDriveLetter = 'F'; ExpectedState = 'mounted-volume' },
    [pscustomobject]@{ Name = 'share-ext4'; Path = 'T:\vm\share-ext4.vhdx'; ExpectedDriveLetter = $null; ExpectedState = 'attached-disk-only' },
    [pscustomobject]@{ Name = 'shared-dev'; Path = 'T:\vm\shared-dev.vhdx'; ExpectedDriveLetter = 'W'; ExpectedState = 'mounted-volume' }
)

function Add-Issue {
    param(
        [System.Collections.Generic.List[string]]$Target,
        [Parameter(Mandatory)] [string]$Message
    )
    [void]$Target.Add($Message)
}

function Get-LastBootTime {
    try {
        return (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
    } catch {
        return (Get-Date).AddMinutes(-[Math]::Max(1, $SinceMinutes))
    }
}

$failures = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$lastBoot = Get-LastBootTime
$eventStart = if ($SinceMinutes -gt 0) { (Get-Date).AddMinutes(-$SinceMinutes) } else { $lastBoot }

$vhdResults = foreach ($expected in $expectedVhds) {
    $exists = Test-Path -LiteralPath $expected.Path
    $vhdInfo = $null
    $diskInfo = $null
    $volumes = @()
    $vhdError = $null

    if (-not $exists) {
        Add-Issue -Target $failures -Message "Missing expected VHD: $($expected.Path)"
    } elseif (-not (Get-Command Get-VHD -ErrorAction SilentlyContinue)) {
        Add-Issue -Target $warnings -Message "Get-VHD is unavailable; cannot validate attachment state for $($expected.Path)"
    } else {
        try {
            $vhdInfo = Get-VHD -Path $expected.Path -ErrorAction Stop
            if (-not $vhdInfo.Attached) {
                Add-Issue -Target $failures -Message "VHD is not attached: $($expected.Path)"
            }

            if ($vhdInfo.Attached -and $vhdInfo.PSObject.Properties['DiskNumber']) {
                $diskInfo = Get-Disk -Number $vhdInfo.DiskNumber -ErrorAction Stop
                $volumes = @(Get-Partition -DiskNumber $vhdInfo.DiskNumber -ErrorAction Stop |
                    Get-Volume -ErrorAction SilentlyContinue |
                    Select-Object DriveLetter, FileSystemLabel, FileSystem, HealthStatus, OperationalStatus, Path)
            }
        } catch {
            $vhdError = $_.Exception.Message
            Add-Issue -Target $failures -Message "VHD validation failed for $($expected.Path): $vhdError"
        }
    }

    if ($expected.ExpectedState -eq 'mounted-volume') {
        $matchingVolume = @($volumes | Where-Object { $_.DriveLetter -eq $expected.ExpectedDriveLetter })
        if ($exists -and $vhdInfo -and $vhdInfo.Attached -and $matchingVolume.Count -eq 0) {
            Add-Issue -Target $failures -Message "Expected drive $($expected.ExpectedDriveLetter): not found for $($expected.Name)"
        }
    } elseif ($expected.ExpectedState -eq 'attached-disk-only' -and $volumes.Count -gt 0) {
        Add-Issue -Target $warnings -Message "$($expected.Name) is expected as attached disk only but Windows volumes are visible"
    }

    [pscustomobject]@{
        Name = $expected.Name
        Path = $expected.Path
        Exists = $exists
        ExpectedDriveLetter = $expected.ExpectedDriveLetter
        ExpectedState = $expected.ExpectedState
        Attached = if ($vhdInfo) { $vhdInfo.Attached } else { $null }
        DiskNumber = if ($vhdInfo -and $vhdInfo.PSObject.Properties['DiskNumber']) { $vhdInfo.DiskNumber } else { $null }
        DiskOperationalStatus = if ($diskInfo) { $diskInfo.OperationalStatus } else { $null }
        Volumes = $volumes
        Error = $vhdError
    }
}

$taskResults = @()
try {
    $taskResults = @(Get-ScheduledTask -TaskName 'AutoMount_VHDX_*' -ErrorAction Stop | ForEach-Object {
        $info = $null
        try { $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath -ErrorAction Stop } catch { }
        if ($info -and $null -ne $info.LastTaskResult -and $info.LastTaskResult -ne 0) {
            Add-Issue -Target $failures -Message "Task $($_.TaskPath)$($_.TaskName) returned $($info.LastTaskResult)"
        }

        [pscustomobject]@{
            TaskPath = $_.TaskPath
            TaskName = $_.TaskName
            State = [string]$_.State
            LastRunTime = if ($info) { $info.LastRunTime } else { $null }
            LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
            NextRunTime = if ($info) { $info.NextRunTime } else { $null }
        }
    })
} catch {
    Add-Issue -Target $warnings -Message "Could not query AutoMount_VHDX_* scheduled tasks: $($_.Exception.Message)"
}

$filterEvents = @()
try {
    $filterEvents = @(Get-WinEvent -FilterHashtable @{
        LogName = 'System'
        ProviderName = 'Microsoft-Windows-FilterManager'
        Id = 3
        StartTime = $eventStart
    } -ErrorAction Stop | Select-Object -First 100 TimeCreated, Id, ProviderName, LevelDisplayName, Message)
    if ($filterEvents.Count -gt 0) {
        Add-Issue -Target $failures -Message "FilterManager Event ID 3 occurred after $($eventStart.ToString('o'))"
    }
} catch {
    if ($_.FullyQualifiedErrorId -notlike '*NoMatchingEventsFound*') {
        Add-Issue -Target $warnings -Message "Could not query FilterManager Event ID 3: $($_.Exception.Message)"
    }
}

$reportRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'Reports\boot-diagnostics'
$latestReport = if (Test-Path -LiteralPath $reportRoot) {
    Get-ChildItem -LiteralPath $reportRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
} else {
    $null
}

if (-not $latestReport -or $latestReport.LastWriteTime -lt $lastBoot) {
    Add-Issue -Target $warnings -Message 'No boot diagnostics report newer than the last boot was found'
}

$result = [pscustomobject]@{
    Status = if ($failures.Count -eq 0) { 'Pass' } else { 'Fail' }
    DryRun = [bool]$DryRun
    GeneratedAt = (Get-Date).ToString('o')
    LastBootTime = $lastBoot
    EventStartTime = $eventStart
    Vhds = $vhdResults
    Tasks = $taskResults
    FilterManagerEventId3 = $filterEvents
    LatestBootDiagnosticsReport = if ($latestReport) { $latestReport.FullName } else { $null }
    Failures = @($failures)
    Warnings = @($warnings)
}

if ($OutputJson -and -not $DryRun) {
    $parent = Split-Path -Parent $OutputJson
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputJson -Encoding UTF8
}

if ($PassThru) {
    $result
} else {
    $result | Format-List
}

if ($FailOnIssue -and $failures.Count -gt 0) {
    exit 1
}
