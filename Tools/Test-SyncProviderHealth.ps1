#Requires -Version 7.0
<#
.SYNOPSIS
Checks OneDrive and sync-provider health for boot/UI diagnostics.

.DESCRIPTION
Inventories sync-provider processes, sync roots, shell overlay handlers,
scheduled tasks, OneDrive diagnostics, and recent WER evidence for OneDrive,
Google Drive, Dropbox, iCloud, and Proton Drive.

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
    [string[]]$RequiredProviders = @('OneDrive'),
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

$providers = @(
    [pscustomobject]@{ Name = 'OneDrive'; Processes = @('OneDrive', 'OneDrive.Sync.Service', 'FileSyncHelper'); RootPatterns = @('*OneDrive*') },
    [pscustomobject]@{ Name = 'GoogleDrive'; Processes = @('GoogleDriveFS'); RootPatterns = @('*Google*Drive*') },
    [pscustomobject]@{ Name = 'Dropbox'; Processes = @('Dropbox', 'DropboxUpdate'); RootPatterns = @('*Dropbox*') },
    [pscustomobject]@{ Name = 'iCloud'; Processes = @('iCloudDrive', 'iCloudServices', 'ApplePhotoStreams'); RootPatterns = @('*iCloud*') },
    [pscustomobject]@{ Name = 'ProtonDrive'; Processes = @('ProtonDrive'); RootPatterns = @('*Proton*Drive*') }
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
                Source = 'SyncRootManager'
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
                Source = 'Filesystem'
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
            [pscustomobject]@{
                Scope = $key
                Name = $_.PSChildName
                Clsid = (Get-ItemProperty -LiteralPath $_.PSPath -Name '(default)' -ErrorAction SilentlyContinue).'(default)'
            }
        }
    }
}

$failures = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$lastBoot = Get-LastBootTime
$eventStart = if ($SinceMinutes -gt 0) { (Get-Date).AddMinutes(-$SinceMinutes) } else { $lastBoot }
$syncRoots = @(Get-SyncRoots)
$overlayHandlers = @(Get-ShellOverlayHandlers)

$providerResults = foreach ($provider in $providers) {
    $runningProcesses = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $provider.Processes -contains $_.ProcessName } |
        Select-Object Id, ProcessName, Responding, StartTime, Path)

    $matchedRoots = @($syncRoots | Where-Object {
        $rootText = "$($_.ProviderId) $($_.DisplayName) $($_.UserSyncRoot)"
        foreach ($pattern in $provider.RootPatterns) {
            if ($rootText -like $pattern) { return $true }
        }
        return $false
    })

    $isRequired = $RequiredProviders -contains $provider.Name
    if ($isRequired -and $runningProcesses.Count -eq 0) {
        Add-Issue -Target $failures -Message "Required sync provider is not running: $($provider.Name)"
    } elseif (-not $isRequired -and $matchedRoots.Count -gt 0 -and $runningProcesses.Count -eq 0) {
        Add-Issue -Target $warnings -Message "Sync root exists but provider process is not running: $($provider.Name)"
    }

    [pscustomobject]@{
        Name = $provider.Name
        Required = $isRequired
        Processes = $runningProcesses
        SyncRoots = $matchedRoots
    }
}

$taskPatterns = @('*OneDrive*', '*Google*Drive*', '*Dropbox*', '*iCloud*', '*Proton*')
$syncTasks = @()
try {
    $syncTasks = @(Get-ScheduledTask -ErrorAction Stop | Where-Object {
        $taskName = $_.TaskName
        $actionText = (@($_.Actions | ForEach-Object {
            $execute = if ($_.PSObject.Properties['Execute']) { $_.Execute } else { '' }
            $arguments = if ($_.PSObject.Properties['Arguments']) { $_.Arguments } else { '' }
            "$execute $arguments"
        }) -join ' ')
        foreach ($pattern in $taskPatterns) {
            if ($taskName -like $pattern -or $actionText -like $pattern) { return $true }
        }
        return $false
    } | ForEach-Object {
        $info = $null
        try { $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath -ErrorAction Stop } catch { }
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
    Add-Issue -Target $warnings -Message "Could not query sync-provider scheduled tasks: $($_.Exception.Message)"
}

$werEvents = @()
try {
    $werEvents = @(Get-WinEvent -FilterHashtable @{
        LogName = 'Application'
        ProviderName = 'Windows Error Reporting'
        StartTime = $eventStart
    } -ErrorAction Stop | Where-Object {
        $_.Message -match 'OneDrive|GoogleDriveFS|Dropbox|iCloud|ProtonDrive|FileSyncHelper'
    } | Select-Object -First 100 TimeCreated, Id, ProviderName, LevelDisplayName, Message)

    $oneDriveWer = @($werEvents | Where-Object { $_.Message -match 'OneDrive|FileSyncHelper' })
    if ($oneDriveWer.Count -gt 0) {
        Add-Issue -Target $failures -Message "OneDrive/FileSyncHelper WER events occurred after $($eventStart.ToString('o'))"
    }
} catch {
    if ($_.FullyQualifiedErrorId -notlike '*NoMatchingEventsFound*') {
        Add-Issue -Target $warnings -Message "Could not query sync-provider WER events: $($_.Exception.Message)"
    }
}

$syncDiagnostics = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Personal\SyncDiagnostics.log'
$syncDiagnosticsState = if (Test-Path -LiteralPath $syncDiagnostics) {
    $item = Get-Item -LiteralPath $syncDiagnostics
    [pscustomobject]@{
        Path = $syncDiagnostics
        Exists = $true
        LastWriteTime = $item.LastWriteTime
        FreshSinceBoot = $item.LastWriteTime -ge $lastBoot
        Tail = @(Get-Content -LiteralPath $syncDiagnostics -Tail 50 -ErrorAction SilentlyContinue)
    }
} else {
    Add-Issue -Target $warnings -Message "OneDrive SyncDiagnostics.log was not found: $syncDiagnostics"
    [pscustomobject]@{
        Path = $syncDiagnostics
        Exists = $false
        LastWriteTime = $null
        FreshSinceBoot = $false
        Tail = @()
    }
}

$result = [pscustomobject]@{
    Status = if ($failures.Count -eq 0) { 'Pass' } else { 'Fail' }
    DryRun = [bool]$DryRun
    GeneratedAt = (Get-Date).ToString('o')
    LastBootTime = $lastBoot
    EventStartTime = $eventStart
    Providers = $providerResults
    SyncRoots = $syncRoots
    ShellOverlayHandlers = $overlayHandlers
    ScheduledTasks = $syncTasks
    WerEvents = $werEvents
    OneDriveSyncDiagnostics = $syncDiagnosticsState
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
