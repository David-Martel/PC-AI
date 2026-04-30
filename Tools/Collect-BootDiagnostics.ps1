#Requires -Version 7.0
<#
.SYNOPSIS
Collects boot, mount, sync-provider, and Process Lasso diagnostics.

.DESCRIPTION
Writes a timestamped boot diagnostics bundle with scheduled task inventory,
startup commands, services, mounted VHD state, sync roots, shell overlays,
filesystem filter state, event-log profile output, and optional post-reboot
health checks.

.PARAMETER SinceMinutes
Lookback window for event-log queries.

.PARAMETER OutputRoot
Root directory for timestamped diagnostic output.

.PARAMETER PostRebootVerify
Run boot mount and sync-provider health checks and include their results.

.PARAMETER DryRun
Preview planned captures without writing files or querying live diagnostics.
The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding()]
param(
    [int]$SinceMinutes = 180,
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'Reports\boot-diagnostics'),
    [switch]$PostRebootVerify,
    [switch]$FailOnIssue,
    [int]$TaskRunningTooLongMinutes = 30,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = Join-Path $OutputRoot $stamp
$rawDir = Join-Path $outDir 'raw'

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

$interestingTaskPatterns = @(
    'AutoMount_VHDX_*',
    'UnifiUdmDriveStackStartup',
    'WSL-Docker-Startup',
    '*OneDrive*',
    '*Google*Drive*',
    '*Dropbox*',
    '*Process*Lasso*'
)

$expectedVhds = @(
    [pscustomobject]@{ Name = 'cloud-cache-disk'; Path = 'T:\vm\cloud-cache-disk.vhdx'; ExpectedDriveLetter = 'F'; ExpectedState = 'mounted-volume' },
    [pscustomobject]@{ Name = 'share-ext4'; Path = 'T:\vm\share-ext4.vhdx'; ExpectedDriveLetter = $null; ExpectedState = 'attached-disk-only' },
    [pscustomobject]@{ Name = 'shared-dev'; Path = 'T:\vm\shared-dev.vhdx'; ExpectedDriveLetter = 'W'; ExpectedState = 'mounted-volume' }
)

if ($DryRun) {
    [pscustomobject]@{
        DryRun = $true
        PlannedOutputDirectory = $outDir
        PlannedRawDirectory = $rawDir
        SinceMinutes = $SinceMinutes
        PostRebootVerify = [bool]$PostRebootVerify
        InterestingTaskPatterns = $interestingTaskPatterns
        ExpectedVhds = $expectedVhds
        RawCaptures = @('get-vhd', 'get-disk', 'get-partition', 'get-volume', 'fltmc-filters', 'fltmc-volumes')
        EventProfiles = @('FilterManager', 'Kernel-PnP', 'disk', 'volmgr', 'Ntfs', 'ReFS', 'VHDMP', 'TaskScheduler', 'WER', 'ApplicationError', 'ProcessLasso')
    }
    return
}

New-Item -ItemType Directory -Path $rawDir -Force | Out-Null

function Invoke-BootSafe {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock
    )

    try {
        [pscustomobject]@{
            Name = $Name
            Ok = $true
            Data = & $ScriptBlock
            Error = $null
        }
    } catch {
        [pscustomobject]@{
            Name = $Name
            Ok = $false
            Data = @()
            Error = $_.Exception.Message
        }
    }
}

function Write-RawCapture {
    param(
        [Parameter(Mandatory)] [string]$FileName,
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock
    )

    $path = Join-Path $rawDir $FileName
    try {
        & $ScriptBlock | Out-File -LiteralPath $path -Encoding UTF8
    } catch {
        "ERROR: $($_.Exception.Message)" | Out-File -LiteralPath $path -Encoding UTF8
    }

    return $path
}

function Get-RegistryValues {
    param([Parameter(Mandatory)] [string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }

    $item = Get-ItemProperty -LiteralPath $Path -ErrorAction Stop
    foreach ($property in $item.PSObject.Properties) {
        if ($property.Name -in @('PSPath', 'PSParentPath', 'PSChildName', 'PSDrive', 'PSProvider')) {
            continue
        }

        [pscustomobject]@{
            Scope = $Path
            Name = $property.Name
            Command = [string]$property.Value
        }
    }
}

function Get-StartupCommands {
    $runKeys = @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run',
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run'
    )

    foreach ($key in $runKeys) {
        Get-RegistryValues -Path $key
    }

    $startupFolders = @(
        [Environment]::GetFolderPath('Startup'),
        [Environment]::GetFolderPath('CommonStartup')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    foreach ($folder in $startupFolders) {
        Get-ChildItem -LiteralPath $folder -Force -ErrorAction SilentlyContinue |
            Select-Object @{ Name = 'Scope'; Expression = { $folder } }, Name, FullName, LastWriteTime
    }
}

function Get-InterestingScheduledTasks {
    $tasks = Get-ScheduledTask -ErrorAction Stop
    foreach ($task in $tasks) {
        $taskName = $task.TaskName
        $actionText = (@($task.Actions | ForEach-Object {
            $execute = if ($_.PSObject.Properties['Execute']) { $_.Execute } else { '' }
            $arguments = if ($_.PSObject.Properties['Arguments']) { $_.Arguments } else { '' }
            "$execute $arguments"
        }) -join ' ')
        $matchesPattern = $false
        foreach ($pattern in $interestingTaskPatterns) {
            if ($taskName -like $pattern -or $task.TaskPath -like $pattern -or $actionText -like $pattern) {
                $matchesPattern = $true
                break
            }
        }

        if (-not $matchesPattern) {
            continue
        }

        $info = $null
        try { $info = Get-ScheduledTaskInfo -TaskName $task.TaskName -TaskPath $task.TaskPath -ErrorAction Stop } catch { }

        $issues = [System.Collections.Generic.List[string]]::new()
        if ($info -and $null -ne $info.LastTaskResult -and $info.LastTaskResult -ne 0) {
            [void]$issues.Add("LastTaskResult=$($info.LastTaskResult)")
        }

        if ($task.State -eq 'Running' -and $info -and $info.LastRunTime -and
            $info.LastRunTime -lt (Get-Date).AddMinutes(-[Math]::Abs($TaskRunningTooLongMinutes))) {
            [void]$issues.Add("RunningLongerThanMinutes=$TaskRunningTooLongMinutes")
        }

        [pscustomobject]@{
            TaskPath = $task.TaskPath
            TaskName = $task.TaskName
            State = [string]$task.State
            LastRunTime = if ($info) { $info.LastRunTime } else { $null }
            LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
            NextRunTime = if ($info) { $info.NextRunTime } else { $null }
            Actions = @($task.Actions | ForEach-Object {
                [pscustomobject]@{
                    Execute = if ($_.PSObject.Properties['Execute']) { $_.Execute } else { $null }
                    Arguments = if ($_.PSObject.Properties['Arguments']) { $_.Arguments } else { $null }
                    WorkingDirectory = if ($_.PSObject.Properties['WorkingDirectory']) { $_.WorkingDirectory } else { $null }
                }
            })
            Triggers = @($task.Triggers | ForEach-Object { $_ | Select-Object * })
            Issues = @($issues)
        }
    }
}

function Get-SyncRoots {
    $roots = [System.Collections.Generic.List[object]]::new()
    $syncRootBase = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager'
    if (Test-Path -LiteralPath $syncRootBase) {
        Get-ChildItem -LiteralPath $syncRootBase -ErrorAction SilentlyContinue | ForEach-Object {
            $props = Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue
            [void]$roots.Add([pscustomobject]@{
                ProviderId = $_.PSChildName
                DisplayName = $props.DisplayNameResource
                UserSyncRoot = $props.UserSyncRoot
                IconResource = $props.IconResource
            })
        }
    }

    foreach ($candidate in @(
        (Join-Path $env:USERPROFILE 'OneDrive'),
        (Join-Path $env:USERPROFILE 'Google Drive'),
        (Join-Path $env:USERPROFILE 'Dropbox'),
        (Join-Path $env:USERPROFILE 'iCloudDrive'),
        (Join-Path $env:USERPROFILE 'Proton Drive')
    )) {
        if (Test-Path -LiteralPath $candidate) {
            [void]$roots.Add([pscustomobject]@{
                ProviderId = Split-Path -Leaf $candidate
                DisplayName = Split-Path -Leaf $candidate
                UserSyncRoot = $candidate
                IconResource = $null
            })
        }
    }

    return $roots
}

function Get-ShellOverlayHandlers {
    $overlayKeys = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\ShellIconOverlayIdentifiers',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Explorer\ShellIconOverlayIdentifiers',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\ShellIconOverlayIdentifiers'
    )

    foreach ($key in $overlayKeys) {
        if (-not (Test-Path -LiteralPath $key)) {
            continue
        }

        Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue | ForEach-Object {
            $default = (Get-ItemProperty -LiteralPath $_.PSPath -Name '(default)' -ErrorAction SilentlyContinue).'(default)'
            [pscustomobject]@{
                Scope = $key
                Name = $_.PSChildName
                Clsid = $default
            }
        }
    }
}

function Get-ExpectedVhdState {
    foreach ($vhd in $expectedVhds) {
        $exists = Test-Path -LiteralPath $vhd.Path
        $vhdInfo = $null
        $diskInfo = $null
        $volumes = @()
        if ($exists -and (Get-Command Get-VHD -ErrorAction SilentlyContinue)) {
            try { $vhdInfo = Get-VHD -Path $vhd.Path -ErrorAction Stop } catch { $vhdInfo = [pscustomobject]@{ Error = $_.Exception.Message } }
        }

        if ($vhdInfo -and $vhdInfo.PSObject.Properties['DiskNumber']) {
            try { $diskInfo = Get-Disk -Number $vhdInfo.DiskNumber -ErrorAction Stop } catch { }
            try {
                $volumes = @(Get-Partition -DiskNumber $vhdInfo.DiskNumber -ErrorAction Stop |
                    Get-Volume -ErrorAction SilentlyContinue |
                    Select-Object DriveLetter, FileSystemLabel, FileSystem, HealthStatus, OperationalStatus, Path)
            } catch { $volumes = @() }
        }

        [pscustomobject]@{
            Name = $vhd.Name
            Path = $vhd.Path
            Exists = $exists
            ExpectedDriveLetter = $vhd.ExpectedDriveLetter
            ExpectedState = $vhd.ExpectedState
            Vhd = $vhdInfo
            Disk = $diskInfo
            Volumes = $volumes
        }
    }
}

function Get-BootEvents {
    $startTime = (Get-Date).AddMinutes(-[Math]::Max(1, $SinceMinutes))
    $queries = @(
        @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-FilterManager'; Label = 'FilterManager'; Optional = $false },
        @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-PnP'; Label = 'Kernel-PnP'; Optional = $false },
        @{ LogName = 'System'; ProviderName = 'disk'; Label = 'disk'; Optional = $false },
        @{ LogName = 'System'; ProviderName = 'volmgr'; Label = 'volmgr'; Optional = $false },
        @{ LogName = 'System'; ProviderName = 'Ntfs'; Label = 'Ntfs'; Optional = $false },
        @{ LogName = 'System'; ProviderName = 'ReFS'; Label = 'ReFS'; Optional = $true },
        @{ LogName = 'System'; ProviderName = 'VHDMP'; Label = 'VHDMP'; Optional = $true },
        @{ LogName = 'Microsoft-Windows-TaskScheduler/Operational'; ProviderName = 'Microsoft-Windows-TaskScheduler'; Label = 'TaskScheduler'; Optional = $false },
        @{ LogName = 'Application'; ProviderName = 'Windows Error Reporting'; Label = 'WER'; Optional = $false },
        @{ LogName = 'Application'; ProviderName = 'Application Error'; Label = 'ApplicationError'; Optional = $false },
        @{ LogName = 'Application'; ProviderName = 'Process Lasso'; Label = 'ProcessLasso'; Optional = $true }
    )

    foreach ($query in $queries) {
        try {
            Get-WinEvent -FilterHashtable @{ LogName = $query.LogName; ProviderName = $query.ProviderName; StartTime = $startTime } -ErrorAction Stop |
                Select-Object -First 300 |
                ForEach-Object {
                    [pscustomobject]@{
                        Profile = $query.Label
                        LogName = $_.LogName
                        ProviderName = $_.ProviderName
                        Id = $_.Id
                        LevelDisplayName = $_.LevelDisplayName
                        TimeCreated = $_.TimeCreated
                        TimeCreatedUtc = $_.TimeCreated.ToUniversalTime().ToString('o')
                        Message = ($_.Message -replace '\s+', ' ').Trim()
                    }
                }
        } catch {
            if ($_.FullyQualifiedErrorId -like '*NoMatchingEventsFound*') {
                continue
            }

            $isMissingOptionalProvider = [bool]$query.Optional -and
                $_.Exception.Message -like '*There is not an event provider*'

            [pscustomobject]@{
                Profile = $query.Label
                LogName = $query.LogName
                ProviderName = $query.ProviderName
                Id = $null
                LevelDisplayName = if ($isMissingOptionalProvider) { 'QuerySkipped' } else { 'QueryError' }
                TimeCreated = $null
                TimeCreatedUtc = $null
                Message = $_.Exception.Message
            }
        }
    }
}

function Invoke-ProcessLassoBootValidation {
    $validator = Join-Path $PSScriptRoot 'Test-ProcessLassoBootSafety.ps1'
    if (-not (Test-Path -LiteralPath $validator)) {
        return [pscustomobject]@{
            Status = 'NotPresent'
            Script = $validator
            OutputPath = $null
            ExitCode = $null
        }
    }

    $outputPath = Join-Path $outDir 'processlasso-validation.txt'
    try {
        & $validator 2>&1 | Out-File -LiteralPath $outputPath -Encoding UTF8
        return [pscustomobject]@{
            Status = 'Ran'
            Script = $validator
            OutputPath = $outputPath
            ExitCode = $LASTEXITCODE
        }
    } catch {
        $_.Exception.Message | Out-File -LiteralPath $outputPath -Encoding UTF8
        return [pscustomobject]@{
            Status = 'Failed'
            Script = $validator
            OutputPath = $outputPath
            ExitCode = 1
        }
    }
}

Write-RawCapture -FileName 'get-vhd.txt' -ScriptBlock {
    foreach ($vhd in $expectedVhds) {
        if ((Test-Path -LiteralPath $vhd.Path) -and (Get-Command Get-VHD -ErrorAction SilentlyContinue)) {
            Get-VHD -Path $vhd.Path | Format-List *
        } else {
            "Get-VHD unavailable or VHD missing: $($vhd.Path)"
        }
    }
} | Out-Null
Write-RawCapture -FileName 'get-disk.txt' -ScriptBlock { Get-Disk | Format-List * } | Out-Null
Write-RawCapture -FileName 'get-partition.txt' -ScriptBlock { Get-Partition | Format-List * } | Out-Null
Write-RawCapture -FileName 'get-volume.txt' -ScriptBlock { Get-Volume | Format-List * } | Out-Null
Write-RawCapture -FileName 'fltmc-filters.txt' -ScriptBlock { fltmc filters } | Out-Null
Write-RawCapture -FileName 'fltmc-volumes.txt' -ScriptBlock { fltmc volumes } | Out-Null

$startupTasks = @(Invoke-BootSafe -Name 'ScheduledTasks' -ScriptBlock { Get-InterestingScheduledTasks })
$startupCommands = Invoke-BootSafe -Name 'StartupCommands' -ScriptBlock { Get-StartupCommands }
$services = Invoke-BootSafe -Name 'Services' -ScriptBlock {
    Get-CimInstance Win32_Service |
        Where-Object {
            $_.StartMode -in @('Auto', 'Automatic') -or
            $_.Name -match 'OneDrive|Google|Dropbox|iCloud|Proton|ProcessLasso|Docker|WSL|rclone|WinFsp|Hyper-V|vmcompute'
        } |
        Select-Object Name, DisplayName, State, StartMode, StartName, PathName, ProcessId
}
$vhdState = Invoke-BootSafe -Name 'MountedVHDs' -ScriptBlock { Get-ExpectedVhdState }
$syncRoots = Invoke-BootSafe -Name 'SyncRoots' -ScriptBlock { Get-SyncRoots }
$overlayHandlers = Invoke-BootSafe -Name 'ShellOverlayHandlers' -ScriptBlock { Get-ShellOverlayHandlers }
$events = Invoke-BootSafe -Name 'BootEventProfile' -ScriptBlock { Get-BootEvents }
$processLassoValidation = Invoke-ProcessLassoBootValidation

$verifier = $null
if ($PostRebootVerify) {
    $mountScript = Join-Path $PSScriptRoot 'Test-BootMountHealth.ps1'
    $syncScript = Join-Path $PSScriptRoot 'Test-SyncProviderHealth.ps1'
    $mount = if (Test-Path -LiteralPath $mountScript) { & $mountScript -PassThru } else { [pscustomobject]@{ Status = 'NotPresent'; Script = $mountScript; Failures = @("Missing $mountScript") } }
    $sync = if (Test-Path -LiteralPath $syncScript) { & $syncScript -PassThru } else { [pscustomobject]@{ Status = 'NotPresent'; Script = $syncScript; Failures = @("Missing $syncScript") } }
    $verifier = [pscustomobject]@{
        MountHealth = $mount
        SyncProviderHealth = $sync
        FailureCount = @($mount.Failures).Count + @($sync.Failures).Count
    }
}

$inventory = [pscustomobject]@{
    SchemaVersion = 1
    GeneratedAt = (Get-Date).ToString('o')
    Hostname = $env:COMPUTERNAME
    SinceMinutes = $SinceMinutes
    OutputDirectory = $outDir
    StartupInventory = [pscustomobject]@{
        ScheduledTasks = $startupTasks[0]
        RunAndStartupCommands = $startupCommands
        Services = $services
        MountedVHDs = $vhdState
        SyncRoots = $syncRoots
        ShellOverlayHandlers = $overlayHandlers
    }
    ActiveFilesystemFilters = [pscustomobject]@{
        FiltersTextPath = Join-Path $rawDir 'fltmc-filters.txt'
        VolumesTextPath = Join-Path $rawDir 'fltmc-volumes.txt'
    }
    EventProfile = $events
    ProcessLassoValidation = $processLassoValidation
    PostRebootVerifier = $verifier
}

$jsonPath = Join-Path $outDir 'startup-inventory.json'
$eventsPath = Join-Path $outDir 'boot-events.json'
$summaryPath = Join-Path $outDir 'startup-inventory.md'

$inventory | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
$events.Data | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $eventsPath -Encoding UTF8

$taskRows = @($startupTasks[0].Data)
$issueTasks = @($taskRows | Where-Object { @($_.Issues).Count -gt 0 })
$eventRows = @($events.Data | Where-Object { $_.LevelDisplayName -notin @('QueryError', 'QuerySkipped') })
$eventErrors = @($events.Data | Where-Object { $_.LevelDisplayName -eq 'QueryError' })
$eventSkipped = @($events.Data | Where-Object { $_.LevelDisplayName -eq 'QuerySkipped' })

$markdown = [System.Collections.Generic.List[string]]::new()
[void]$markdown.Add("# Boot Diagnostics $stamp")
[void]$markdown.Add("")
[void]$markdown.Add("- Generated: $((Get-Date).ToString('o'))")
[void]$markdown.Add("- Output: ``$outDir``")
[void]$markdown.Add("- Window: last $SinceMinutes minutes")
[void]$markdown.Add("- Post-reboot verifier: $PostRebootVerify")
[void]$markdown.Add("")
[void]$markdown.Add("## Task Issues")
if ($issueTasks.Count -eq 0) {
    [void]$markdown.Add("")
    [void]$markdown.Add("No scheduled task issues were detected in the inventory.")
} else {
    [void]$markdown.Add("")
    [void]$markdown.Add("| Task | State | Last Run | Last Result | Issues |")
    [void]$markdown.Add("| --- | --- | --- | --- | --- |")
    foreach ($task in $issueTasks) {
        [void]$markdown.Add("| $($task.TaskPath)$($task.TaskName) | $($task.State) | $($task.LastRunTime) | $($task.LastTaskResult) | $(@($task.Issues) -join '; ') |")
    }
}

[void]$markdown.Add("")
[void]$markdown.Add("## Expected VHDs")
[void]$markdown.Add("")
[void]$markdown.Add("| Name | Path | Exists | Expected | Attached | Volumes |")
[void]$markdown.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($vhd in @($vhdState.Data)) {
    $attached = if ($vhd.Vhd -and $vhd.Vhd.PSObject.Properties['Attached']) { $vhd.Vhd.Attached } else { 'unknown' }
    $volumesText = (@($vhd.Volumes) | ForEach-Object { "$($_.DriveLetter): $($_.FileSystemLabel) $($_.FileSystem)" }) -join '<br>'
    [void]$markdown.Add("| $($vhd.Name) | ``$($vhd.Path)`` | $($vhd.Exists) | $($vhd.ExpectedState) $($vhd.ExpectedDriveLetter) | $attached | $volumesText |")
}

[void]$markdown.Add("")
[void]$markdown.Add("## Event Profile")
[void]$markdown.Add("")
[void]$markdown.Add("- Events captured: $($eventRows.Count)")
[void]$markdown.Add("- Query errors: $($eventErrors.Count)")
if ($eventSkipped.Count -gt 0) {
    [void]$markdown.Add("- Optional providers skipped: $($eventSkipped.Count)")
}
[void]$markdown.Add("- Event JSON: ``$eventsPath``")

if ($verifier) {
    [void]$markdown.Add("")
    [void]$markdown.Add("## Post-Reboot Verifier")
    [void]$markdown.Add("")
    [void]$markdown.Add("- Failure count: $($verifier.FailureCount)")
    [void]$markdown.Add("- Mount failures: $(@($verifier.MountHealth.Failures) -join '; ')")
    [void]$markdown.Add("- Sync failures: $(@($verifier.SyncProviderHealth.Failures) -join '; ')")
}

[void]$markdown.Add("")
[void]$markdown.Add("## Process Lasso Validation")
[void]$markdown.Add("")
[void]$markdown.Add("- Status: $($processLassoValidation.Status)")
[void]$markdown.Add("- Script: ``$($processLassoValidation.Script)``")
[void]$markdown.Add("- Output: ``$($processLassoValidation.OutputPath)``")

$markdown | Set-Content -LiteralPath $summaryPath -Encoding UTF8

$result = [pscustomobject]@{
    OutputDirectory = $outDir
    JsonPath = $jsonPath
    MarkdownPath = $summaryPath
    EventsPath = $eventsPath
    TaskIssueCount = $issueTasks.Count
    EventQueryErrorCount = $eventErrors.Count
    OptionalEventProviderSkipCount = $eventSkipped.Count
    PostRebootFailureCount = if ($verifier) { $verifier.FailureCount } else { $null }
}

$result

if ($FailOnIssue) {
    $failures = $issueTasks.Count
    if ($verifier) { $failures += $verifier.FailureCount }
    if ($failures -gt 0) {
        exit 1
    }
}
