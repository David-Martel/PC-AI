<#
.SYNOPSIS
Collects drive-performance and cloud-sync risk evidence.

.DESCRIPTION
Captures read-only evidence for registry tuning, sync providers, Task
Scheduler, Process Lasso, Defender, Windows Search, filter drivers, disk state,
and recent OneDrive/storage events. Use this before and after registry or script
changes to validate the effect on OneDrive and UI responsiveness.

.PARAMETER OutputRoot
Root directory for report output. A timestamped subdirectory is created unless
-DryRun is supplied.

.PARAMETER SinceMinutes
Lookback window for relevant Windows event logs.

.PARAMETER DryRun
Preview the planned collection without creating report directories or files.
The long CLI form --DryRun is also accepted.

.PARAMETER OutputJson
Optional JSON output file. Ignored in dry-run mode.

.PARAMETER PassThru
Return the result object to the pipeline.

.PARAMETER Help
Print script help and exit. The aliases -h and --help are also accepted.
#>
[CmdletBinding()]
param(
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'Reports\drive-performance-sync-risk'),
    [int]$SinceMinutes = 240,
    [switch]$DryRun,
    [string]$OutputJson = '',
    [switch]$PassThru,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($CliArgs -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if ($CliArgs -contains '--DryRun') {
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

function Get-RegistryValueState {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ Path = $Path; Name = $Name; Exists = $false; Value = $null }
    }
    $item = Get-ItemProperty -LiteralPath $Path -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $item) {
        return [pscustomobject]@{ Path = $Path; Name = $Name; Exists = $false; Value = $null }
    }
    [pscustomobject]@{ Path = $Path; Name = $Name; Exists = $true; Value = $item.$Name }
}

function Get-SyncRoots {
    $roots = @()
    $rootKey = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager'
    if (Test-Path -LiteralPath $rootKey) {
        foreach ($provider in Get-ChildItem -LiteralPath $rootKey -ErrorAction SilentlyContinue) {
            $userRootsKey = Join-Path $provider.PSPath 'UserSyncRoots'
            if (Test-Path -LiteralPath $userRootsKey) {
                $props = Get-ItemProperty -LiteralPath $userRootsKey
                foreach ($property in $props.PSObject.Properties) {
                    if ($property.Name -notmatch '^PS') {
                        $roots += [pscustomobject]@{
                            Provider = $provider.PSChildName
                            Sid = $property.Name
                            Path = [string]$property.Value
                            Exists = Test-Path -LiteralPath ([string]$property.Value)
                        }
                    }
                }
            }
        }
    }
    $roots
}

function Get-OneDriveDiagnostics {
    $diagPath = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Personal\SyncDiagnostics.log'
    $metrics = [ordered]@{
        Path = $diagPath
        Exists = Test-Path -LiteralPath $diagPath
        LastWriteTime = $null
        Values = @{}
    }
    if (-not $metrics.Exists) {
        return [pscustomobject]$metrics
    }
    $item = Get-Item -LiteralPath $diagPath
    $metrics.LastWriteTime = $item.LastWriteTime
    $wanted = @(
        'driveChangesToSend',
        'driveSentChanges',
        'files',
        'folders',
        'scanState',
        'scanStateStallDetected',
        'syncStallDetected',
        'syncProgressState',
        'threadCount',
        'totalDoScanWorkCpuTimeInMs',
        'timeUtc',
        'version'
    )
    foreach ($line in Get-Content -LiteralPath $diagPath -ErrorAction SilentlyContinue) {
        if ($line -match '^\s*([^=]+?)\s*=\s*(.*)\s*$') {
            $key = $matches[1].Trim()
            if ($key -in $wanted) {
                $metrics.Values[$key] = $matches[2].Trim()
            }
        }
    }
    [pscustomobject]$metrics
}

function Get-RecentEvents {
    param([datetime]$StartTime)
    $queries = @(
        @{ LogName = 'Application'; ProviderName = 'Windows Error Reporting'; Label = 'WER' },
        @{ LogName = 'Application'; ProviderName = 'Application Error'; Label = 'ApplicationError' },
        @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-FilterManager'; Label = 'FilterManager' },
        @{ LogName = 'System'; ProviderName = 'disk'; Label = 'disk' },
        @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Ntfs'; Label = 'Ntfs' },
        @{ LogName = 'Microsoft-Windows-TaskScheduler/Operational'; ProviderName = 'Microsoft-Windows-TaskScheduler'; Label = 'TaskScheduler' }
    )
    $events = @()
    foreach ($query in $queries) {
        try {
            $items = Get-WinEvent -FilterHashtable @{
                LogName = $query.LogName
                ProviderName = $query.ProviderName
                StartTime = $StartTime
            } -ErrorAction Stop
            foreach ($event in $items) {
                if ($event.Message -match 'OneDrive|FileSyncHelper|GoogleDriveFS|Dropbox|iCloud|Proton|Filter Manager|Harddisk|VHD|Process Lasso|ProcessGovernor|UDM|rclone|disk identifiers') {
                    $events += [pscustomobject]@{
                        Label = $query.Label
                        TimeCreated = $event.TimeCreated
                        Id = $event.Id
                        ProviderName = $event.ProviderName
                        LevelDisplayName = $event.LevelDisplayName
                        Message = $event.Message
                    }
                }
            }
        }
        catch {
        }
    }
    $events | Sort-Object TimeCreated -Descending
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$reportDir = Join-Path $OutputRoot $stamp
$eventStart = (Get-Date).AddMinutes(-1 * $SinceMinutes)

$registryItems = @(
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled'
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'NtfsMemoryUsage'
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'NtfsDisableLastAccessUpdate'
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'NtfsDisable8dot3NameCreation'
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' -Name 'LargeSystemCache'
    Get-RegistryValueState -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' -Name 'DisablePagingExecutive'
    Get-RegistryValueState -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer' -Name 'NoRemoteChangeNotify'
    Get-RegistryValueState -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer' -Name 'NoRemoteChangeNotify'
    Get-RegistryValueState -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer' -Name 'NoRemoteRecursiveEvents'
    Get-RegistryValueState -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer' -Name 'NoRemoteRecursiveEvents'
)

$processNames = @('OneDrive', 'OneDrive.Sync.Service', 'FileSyncHelper', 'GoogleDriveFS', 'Dropbox', 'iCloudDrive', 'ProtonDrive', 'ProcessGovernor', 'ProcessLasso', 'MsMpEng')
$syncProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $processNames } |
    Select-Object ProcessName, Id, CPU, WorkingSet64, Path, StartTime, Responding)
$taskPatterns = 'OneDrive|Google|Dropbox|iCloud|Proton|Process Lasso|UDM|VHD|ProfileLogSync'
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.TaskName -match $taskPatterns -or $_.TaskPath -match $taskPatterns
} | ForEach-Object {
    $info = $_ | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
    [pscustomobject]@{
        TaskName = $_.TaskName
        TaskPath = $_.TaskPath
        State = $_.State
        LastRunTime = if ($info) { $info.LastRunTime } else { $null }
        LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
        NextRunTime = if ($info) { $info.NextRunTime } else { $null }
    }
})

$defender = $null
try {
    $defender = Get-MpPreference | Select-Object DisableRealtimeMonitoring, DisableIOAVProtection, EnableControlledFolderAccess, ExclusionPath, ExclusionProcess, ScanAvgCPULoadFactor
}
catch {
}

$searchService = Get-Service WSearch -ErrorAction SilentlyContinue | Select-Object Name, Status, StartType
$filters = @(fltmc filters 2>$null)
$disks = @(Get-Disk -ErrorAction SilentlyContinue | Select-Object Number, FriendlyName, UniqueId, Guid, PartitionStyle, BusType, OperationalStatus, HealthStatus, IsOffline, IsReadOnly, Size)
$volumes = @(Get-Volume -ErrorAction SilentlyContinue | Select-Object DriveLetter, FileSystemLabel, FileSystem, Path, HealthStatus, OperationalStatus, SizeRemaining, Size)
$syncRoots = @(Get-SyncRoots)
$oneDriveDiagnostics = Get-OneDriveDiagnostics
$events = @(Get-RecentEvents -StartTime $eventStart)

$warnings = @()
if (@($events | Where-Object { $_.Message -match 'OneDrive|FileSyncHelper' }).Count -gt 0) {
    $warnings += 'Recent OneDrive/FileSyncHelper event evidence exists.'
}
if (@($events | Where-Object { $_.ProviderName -eq 'Microsoft-Windows-FilterManager' }).Count -gt 0) {
    $warnings += 'Recent FilterManager evidence exists.'
}
if (@($syncRoots | Where-Object { $_.Path -like 'F:\*' }).Count -gt 0) {
    $warnings += 'One or more cloud-sync roots are on F:, the mounted cloud-cache VHD.'
}

$result = [ordered]@{
    GeneratedAt = (Get-Date).ToString('o')
    DryRun = [bool]$DryRun
    ReportDirectory = $reportDir
    SinceMinutes = $SinceMinutes
    EventStartTime = $eventStart
    Registry = $registryItems
    SyncProcesses = $syncProcesses
    SyncRoots = $syncRoots
    ScheduledTasks = $tasks
    ProcessLasso = @{
        Governor = @($syncProcesses | Where-Object { $_.ProcessName -eq 'ProcessGovernor' })
        LogPath = 'C:\ProgramData\ProcessLasso\logs\processlasso.log'
    }
    Defender = $defender
    WindowsSearch = $searchService
    FilterDrivers = $filters
    Disks = $disks
    Volumes = $volumes
    OneDriveDiagnostics = $oneDriveDiagnostics
    RecentEvents = $events
    Warnings = $warnings
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $jsonPath = if ($OutputJson) { $OutputJson } else { Join-Path $reportDir 'drive-performance-sync-risk.json' }
    $jsonDir = Split-Path -Parent $jsonPath
    if ($jsonDir -and -not (Test-Path -LiteralPath $jsonDir)) {
        New-Item -ItemType Directory -Path $jsonDir -Force | Out-Null
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    $summaryPath = Join-Path $reportDir 'summary.md'
    @(
        '# Drive Performance And Sync Risk Capture'
        ''
        "- Generated: $($result.GeneratedAt)"
        "- SinceMinutes: $SinceMinutes"
        "- Warnings: $(@($warnings).Count)"
        "- Sync roots: $(@($syncRoots).Count)"
        "- Sync/provider processes: $(@($syncProcesses).Count)"
        "- Scheduled tasks captured: $(@($tasks).Count)"
        "- Events captured: $(@($events).Count)"
        ''
        '## Warnings'
        @($warnings | ForEach-Object { "- $_" })
    ) | Set-Content -LiteralPath $summaryPath -Encoding UTF8
}

$object = [pscustomobject]$result
if ($PassThru -or $DryRun) {
    $object
}
else {
    Write-Host "Drive performance sync-risk report: $reportDir"
}
