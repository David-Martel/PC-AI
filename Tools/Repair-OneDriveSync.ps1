#Requires -Version 7.0
<#
.SYNOPSIS
Captures and repairs OneDrive sync-client state with reset kept opt-in.

.DESCRIPTION
Collects OneDrive process, task, version, WER, sync diagnostics, and known
restriction evidence into a structured report. By default the script is
evidence-only. Use -DownloadInstaller to download the current Microsoft
OneDrive installer, -InstallLatest to run it, and -ResetOneDrive to run the
Microsoft-documented reset command. Reset is intentionally opt-in because it
causes OneDrive to rebuild sync state and perform a full sync.

.PARAMETER OutputDirectory
Directory for captured evidence and downloaded installer.

.PARAMETER SinceMinutes
Lookback window for OneDrive/FileSyncHelper Windows Error Reporting events.

.PARAMETER InstallerUrl
Microsoft OneDrive installer URL.

.PARAMETER DownloadInstaller
Download the OneDrive installer into OutputDirectory.

.PARAMETER InstallLatest
Run the downloaded installer. If InstallerPath is not supplied, the script
downloads the installer first.

.PARAMETER InstallerPath
Existing OneDriveSetup.exe path to run with -InstallLatest.

.PARAMETER StopBeforeInstall
Stop OneDrive-related user processes before running the installer.

.PARAMETER ResetOneDrive
Run onedrive.exe /reset. This is opt-in because reset triggers a full sync
state rebuild.

.PARAMETER StartAfterRepair
Start OneDrive after installer or reset actions complete.

.PARAMETER DryRun
Collect evidence and report planned actions without stopping processes,
downloading, installing, resetting, or starting OneDrive. The long CLI form
`--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$OutputDirectory = (Join-Path (Join-Path $PSScriptRoot '..\Reports\onedrive-repair') (Get-Date -Format 'yyyyMMdd-HHmmss')),

    [int]$SinceMinutes = 240,

    [string]$InstallerUrl = 'https://go.microsoft.com/fwlink/p/?LinkID=2182910',

    [switch]$DownloadInstaller,

    [switch]$InstallLatest,

    [string]$InstallerPath,

    [switch]$StopBeforeInstall,

    [switch]$ResetOneDrive,

    [switch]$StartAfterRepair,

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

$dryRunActive = [bool]($DryRun -or $WhatIfPreference)

function Add-ActionRecord {
    param(
        [Parameter(Mandatory)] [object]$Actions,
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Status,
        [AllowNull()] [object]$Detail
    )

    [void]$Actions.Add([pscustomobject]@{
            Name      = $Name
            Status    = $Status
            Detail    = $Detail
            Timestamp = (Get-Date).ToString('o')
        })
}

function Get-OneDriveExecutableCandidates {
    $candidates = [System.Collections.Generic.List[string]]::new()
    foreach ($path in @(
            (Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\OneDrive.exe'),
            'C:\Program Files\Microsoft OneDrive\OneDrive.exe',
            'C:\Program Files (x86)\Microsoft OneDrive\OneDrive.exe'
        )) {
        if (Test-Path -LiteralPath $path) {
            [void]$candidates.Add($path)
        }
    }

    foreach ($root in @('C:\Program Files\Microsoft OneDrive', (Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive'))) {
        if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Filter OneDrive.exe -Recurse -ErrorAction SilentlyContinue |
                ForEach-Object {
                    if (-not $candidates.Contains($_.FullName)) {
                        [void]$candidates.Add($_.FullName)
                    }
                }
        }
    }

    $candidates
}

function Get-FileVersionInfoObject {
    param([Parameter(Mandatory)] [string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $item = Get-Item -LiteralPath $Path
    [pscustomobject]@{
        Path             = $item.FullName
        Length           = $item.Length
        LastWriteTime    = $item.LastWriteTime.ToString('o')
        ProductVersion   = $item.VersionInfo.ProductVersion
        FileVersion      = $item.VersionInfo.FileVersion
        CompanyName      = $item.VersionInfo.CompanyName
        ProductName      = $item.VersionInfo.ProductName
    }
}

function Read-SyncDiagnostics {
    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Personal\SyncDiagnostics.log'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Business1\SyncDiagnostics.log'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Business2\SyncDiagnostics.log')
    )

    foreach ($path in $paths) {
        $exists = Test-Path -LiteralPath $path
        $values = [ordered]@{}
        $tail = @()
        $lastWrite = $null
        if ($exists) {
            $item = Get-Item -LiteralPath $path
            $lastWrite = $item.LastWriteTime.ToString('o')
            $content = Get-Content -LiteralPath $path -ErrorAction SilentlyContinue
            $tail = @($content | Select-Object -Last 80)
            foreach ($line in $content) {
                if ($line -match '^\s*([^=]+?)\s*=\s*(.*?)\s*$') {
                    $key = $Matches[1].Trim()
                    $value = $Matches[2].Trim()
                    if ($key.Length -gt 0) {
                        $values[$key] = $value
                    }
                }
            }
        }

        [pscustomobject]@{
            Path          = $path
            Exists        = $exists
            LastWriteTime = $lastWrite
            Values        = $values
            Tail          = $tail
        }
    }
}

function Get-OneDriveTasks {
    Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object { $_.TaskName -like '*OneDrive*' -or $_.TaskPath -like '*OneDrive*' } |
        ForEach-Object {
            $info = $null
            try { $info = $_ | Get-ScheduledTaskInfo -ErrorAction Stop } catch { }
            $state = if ($null -ne $_.State) { $_.State.ToString() } else { $null }
            $principalUser = if ($null -ne $_.Principal) { $_.Principal.UserId } else { $null }
            [pscustomobject]@{
                TaskPath       = $_.TaskPath
                TaskName       = $_.TaskName
                State          = $state
                LastRunTime    = if ($info -and $null -ne $info.LastRunTime) { $info.LastRunTime.ToString('o') } else { $null }
                LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
                NextRunTime    = if ($info -and $null -ne $info.NextRunTime) { $info.NextRunTime.ToString('o') } else { $null }
                Author         = $_.Author
                UserId         = $principalUser
            }
        }
}

function Get-OneDriveWerEvents {
    param([Parameter(Mandatory)] [int]$LookbackMinutes)

    $start = (Get-Date).AddMinutes(-1 * $LookbackMinutes)
    $events = @()
    foreach ($logName in @('Application')) {
        try {
            $events += Get-WinEvent -FilterHashtable @{
                LogName   = $logName
                StartTime = $start
            } -ErrorAction Stop |
                Where-Object {
                    ($_.ProviderName -in @('Windows Error Reporting', 'Application Error')) -and
                    ($_.Message -match 'OneDrive\.exe|FileSyncHelper\.exe|OneDrive\.Sync\.Service\.exe')
                } |
                Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message
        } catch {
            $events += [pscustomobject]@{
                TimeCreated      = (Get-Date)
                Id               = -1
                ProviderName     = 'QueryError'
                LevelDisplayName = 'Error'
                Message          = $_.Exception.Message
            }
        }
    }

    $events
}

function Get-OneDriveWerQueues {
    $root = 'C:\ProgramData\Microsoft\Windows\WER\ReportQueue'
    if (-not (Test-Path -LiteralPath $root)) {
        return @()
    }

    Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'OneDrive|FileSyncHelper|OneDrive\.Sync\.Service' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 40 |
        ForEach-Object {
            [pscustomobject]@{
                Name          = $_.Name
                FullName      = $_.FullName
                LastWriteTime = $_.LastWriteTime.ToString('o')
                FileCount     = @((Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue)).Count
            }
        }
}

function Get-SyncRootInventory {
    $roots = [System.Collections.Generic.List[object]]::new()
    foreach ($path in @(
            (Join-Path $env:USERPROFILE 'OneDrive'),
            (Join-Path $env:USERPROFILE 'OneDrive - Personal'),
            (Join-Path $env:USERPROFILE 'iCloudDrive'),
            'F:\Auricle Dropbox',
            'F:\Proton-Drive\My files'
        )) {
        [void]$roots.Add([pscustomobject]@{
                Path   = $path
                Exists = Test-Path -LiteralPath $path
            })
    }

    $roots
}

function Invoke-OneDriveInstaller {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [object]$Actions
    )

    $arguments = '/allusers'
    if ($dryRunActive) {
        Add-ActionRecord -Actions $Actions -Name 'InstallLatest' -Status 'SkippedDryRun' -Detail @{
            Path      = $Path
            Arguments = $arguments
        }
        return $null
    }

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Installer not found: $Path"
    }

    if ($PSCmdlet.ShouldProcess($Path, "Run OneDrive installer $arguments")) {
        $process = Start-Process -FilePath $Path -ArgumentList $arguments -Wait -PassThru
        Add-ActionRecord -Actions $Actions -Name 'InstallLatest' -Status 'Completed' -Detail @{
            Path      = $Path
            Arguments = $arguments
            ExitCode  = $process.ExitCode
        }
        return $process.ExitCode
    }
}

function Invoke-OneDriveInteractiveTask {
    param(
        [Parameter(Mandatory)] [string]$Executable,
        [AllowNull()] [string]$Arguments,
        [Parameter(Mandatory)] [string]$Purpose,
        [Parameter(Mandatory)] [object]$Actions
    )

    $taskName = 'PC-AI OneDrive ' + $Purpose + ' ' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    $taskPath = '\PC-AI\'
    $detail = @{
        TaskPath   = $taskPath
        TaskName   = $taskName
        Executable = $Executable
        Arguments  = $Arguments
        UserId     = $env:USERNAME
    }

    if ($dryRunActive) {
        Add-ActionRecord -Actions $Actions -Name $Purpose -Status 'SkippedDryRun' -Detail $detail
        return $null
    }

    if (-not (Test-Path -LiteralPath $Executable)) {
        Add-ActionRecord -Actions $Actions -Name $Purpose -Status 'Failed' -Detail ($detail + @{ Error = 'Executable not found.' })
        return $null
    }

    if (-not $PSCmdlet.ShouldProcess("$taskPath$taskName", "Run $Executable $Arguments as interactive user")) {
        return $null
    }

    $registered = $false
    try {
        $action = New-ScheduledTaskAction -Execute $Executable -Argument $Arguments
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
        Register-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
        $registered = $true
        Start-ScheduledTask -TaskPath $taskPath -TaskName $taskName

        $deadline = (Get-Date).AddSeconds(60)
        do {
            Start-Sleep -Seconds 2
            $task = Get-ScheduledTask -TaskPath $taskPath -TaskName $taskName -ErrorAction SilentlyContinue
            $info = $task | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
        } while ($task -and $task.State -eq 'Running' -and (Get-Date) -lt $deadline)

        $detail.LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
        $detail.LastTaskResultHex = if ($info -and $null -ne $info.LastTaskResult) {
            '0x{0:X8}' -f ([uint32]$info.LastTaskResult)
        } else {
            $null
        }
        $detail.State = if ($task -and $task.State) { $task.State.ToString() } else { $null }
        $detail.ProcessesAfterAction = @(Get-Process -Name OneDrive, FileSyncHelper, FileCoAuth, 'OneDrive.Sync.Service' -ErrorAction SilentlyContinue |
            Select-Object ProcessName, Id, CPU, StartTime, Responding, Path)

        $status = 'Completed'
        if ($info -and $null -ne $info.LastTaskResult -and $info.LastTaskResult -ne 0) {
            $status = 'CompletedWithWarning'
            $detail.Warning = "Scheduled task returned non-zero LastTaskResult $($detail.LastTaskResultHex). Check WER/process evidence before treating the action as healthy."
        }

        Add-ActionRecord -Actions $Actions -Name $Purpose -Status $status -Detail $detail
        return $detail
    } catch {
        Add-ActionRecord -Actions $Actions -Name $Purpose -Status 'Failed' -Detail ($detail + @{ Error = $_.Exception.Message })
        return $null
    } finally {
        if ($registered) {
            Unregister-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        }
    }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$actions = [System.Collections.Generic.List[object]]::new()

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

$preEvidence = [pscustomobject]@{
    GeneratedAt          = (Get-Date).ToString('o')
    ComputerName         = $env:COMPUTERNAME
    UserName             = $env:USERNAME
    IsAdministrator      = $isAdmin
    SinceMinutes         = $SinceMinutes
    Processes            = @(Get-Process -Name OneDrive, FileSyncHelper, 'OneDrive.Sync.Service' -ErrorAction SilentlyContinue |
        Select-Object ProcessName, Id, CPU, WorkingSet64, StartTime, Responding, Path)
    Executables          = @(Get-OneDriveExecutableCandidates | ForEach-Object { Get-FileVersionInfoObject -Path $_ })
    Tasks                = @(Get-OneDriveTasks)
    SyncDiagnostics      = @(Read-SyncDiagnostics)
    WerEvents            = @(Get-OneDriveWerEvents -LookbackMinutes $SinceMinutes)
    WerQueues            = @(Get-OneDriveWerQueues)
    SyncRoots            = @(Get-SyncRootInventory)
    ProgramFilesChildren = @(Get-ChildItem -LiteralPath 'C:\Program Files\Microsoft OneDrive' -ErrorAction SilentlyContinue |
        Select-Object Name, FullName, LastWriteTime, Length)
}

$prePath = Join-Path $OutputDirectory 'pre-repair-evidence.json'
$preEvidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $prePath -Encoding UTF8
Add-ActionRecord -Actions $actions -Name 'CapturePreEvidence' -Status 'Completed' -Detail $prePath

$effectiveInstallerPath = $InstallerPath
if (($DownloadInstaller -or $InstallLatest) -and [string]::IsNullOrWhiteSpace($effectiveInstallerPath)) {
    $effectiveInstallerPath = Join-Path $OutputDirectory 'OneDriveSetup.exe'
}

if ($DownloadInstaller -or ($InstallLatest -and -not (Test-Path -LiteralPath $effectiveInstallerPath -ErrorAction SilentlyContinue))) {
    if ($dryRunActive) {
        Add-ActionRecord -Actions $actions -Name 'DownloadInstaller' -Status 'SkippedDryRun' -Detail @{
            Url  = $InstallerUrl
            Path = $effectiveInstallerPath
        }
    } elseif ($PSCmdlet.ShouldProcess($effectiveInstallerPath, "Download OneDrive installer from $InstallerUrl")) {
        Invoke-WebRequest -Uri $InstallerUrl -OutFile $effectiveInstallerPath -UseBasicParsing
        Add-ActionRecord -Actions $actions -Name 'DownloadInstaller' -Status 'Completed' -Detail @{
            Url  = $InstallerUrl
            Path = $effectiveInstallerPath
            File = Get-FileVersionInfoObject -Path $effectiveInstallerPath
        }
    }
}

if ($StopBeforeInstall -and ($InstallLatest -or $ResetOneDrive)) {
    $targets = @(Get-Process -Name OneDrive, FileSyncHelper, 'OneDrive.Sync.Service' -ErrorAction SilentlyContinue)
    if ($dryRunActive) {
        Add-ActionRecord -Actions $actions -Name 'StopBeforeRepair' -Status 'SkippedDryRun' -Detail @{
            ProcessIds = @($targets | Select-Object ProcessName, Id)
        }
    } elseif ($PSCmdlet.ShouldProcess('OneDrive processes', 'Stop before repair')) {
        foreach ($process in $targets) {
            try {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
            } catch {
                Add-ActionRecord -Actions $actions -Name 'StopProcess' -Status 'Failed' -Detail @{
                    ProcessName = $process.ProcessName
                    Id          = $process.Id
                    Error       = $_.Exception.Message
                }
            }
        }
        Add-ActionRecord -Actions $actions -Name 'StopBeforeRepair' -Status 'Completed' -Detail @{
            ProcessIds = @($targets | Select-Object ProcessName, Id)
        }
    }
}

if ($InstallLatest) {
    Invoke-OneDriveInstaller -Path $effectiveInstallerPath -Actions $actions | Out-Null
}

if ($ResetOneDrive) {
    $resetExe = @(Get-OneDriveExecutableCandidates | Where-Object { $_ -match '\\OneDrive\.exe$' } | Select-Object -First 1)
    if ($resetExe.Count -eq 0) {
        Add-ActionRecord -Actions $actions -Name 'ResetOneDrive' -Status 'Failed' -Detail 'No OneDrive.exe candidate found.'
    } else {
        Invoke-OneDriveInteractiveTask -Executable $resetExe[0] -Arguments '/reset' -Purpose 'ResetOneDrive' -Actions $actions | Out-Null
    }
}

if ($StartAfterRepair -and ($InstallLatest -or $ResetOneDrive)) {
    $startExe = @(Get-OneDriveExecutableCandidates | Where-Object { $_ -match '\\OneDrive\.exe$' } | Select-Object -First 1)
    if ($startExe.Count -eq 0) {
        Add-ActionRecord -Actions $actions -Name 'StartOneDrive' -Status 'Failed' -Detail 'No OneDrive.exe candidate found.'
    } else {
        Invoke-OneDriveInteractiveTask -Executable $startExe[0] -Arguments '/background' -Purpose 'StartOneDrive' -Actions $actions | Out-Null
        Start-Sleep -Seconds 10
    }
}

$postEvidence = [pscustomobject]@{
    GeneratedAt     = (Get-Date).ToString('o')
    Processes       = @(Get-Process -Name OneDrive, FileSyncHelper, 'OneDrive.Sync.Service' -ErrorAction SilentlyContinue |
        Select-Object ProcessName, Id, CPU, WorkingSet64, StartTime, Responding, Path)
    Executables     = @(Get-OneDriveExecutableCandidates | ForEach-Object { Get-FileVersionInfoObject -Path $_ })
    Tasks           = @(Get-OneDriveTasks)
    SyncDiagnostics = @(Read-SyncDiagnostics)
    WerEvents       = @(Get-OneDriveWerEvents -LookbackMinutes $SinceMinutes)
    WerQueues       = @(Get-OneDriveWerQueues)
}

$postPath = Join-Path $OutputDirectory 'post-repair-evidence.json'
$postEvidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $postPath -Encoding UTF8
Add-ActionRecord -Actions $actions -Name 'CapturePostEvidence' -Status 'Completed' -Detail $postPath

$preSyncDiagnostic = @($preEvidence.SyncDiagnostics | Where-Object { $_.Exists } | Select-Object -First 1)
$postSyncDiagnostic = @($postEvidence.SyncDiagnostics | Where-Object { $_.Exists } | Select-Object -First 1)
$warnings = @($actions | Where-Object { $_.Status -match 'Warning|Failed' })

$summary = [pscustomobject]@{
    OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
    DryRun          = $dryRunActive
    GeneratedAt     = (Get-Date).ToString('o')
    InstallerUrl    = $InstallerUrl
    InstallerPath   = $effectiveInstallerPath
    Actions         = @($actions)
    PreEvidence     = $prePath
    PostEvidence    = $postPath
    PreWerCount     = @($preEvidence.WerEvents).Count
    PostWerCount    = @($postEvidence.WerEvents).Count
    WarningCount     = @($warnings).Count
    Warnings         = @($warnings)
    PreSyncValues   = if ($preSyncDiagnostic.Count -gt 0) { $preSyncDiagnostic[0].Values } else { $null }
    PostSyncValues  = if ($postSyncDiagnostic.Count -gt 0) { $postSyncDiagnostic[0].Values } else { $null }
}

$summaryPath = Join-Path $OutputDirectory 'summary.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

$summary
