#Requires -Version 5.1
<#
.SYNOPSIS
Mounts and validates a persistent VHDX used during workstation boot.

.DESCRIPTION
Mount-PersistentVHDX.ps1 is the maintained Task Scheduler wrapper for boot-time
VHDX attachment. It writes a transcript, a structured JSON result, and Windows
event-log entries, then exits nonzero when the mount is degraded or failed.

.PARAMETER VhdPath
Path to the VHD or VHDX file.

.PARAMETER ExpectedVolumeLabel
Expected Windows volume label when a mounted volume is required.

.PARAMETER ExpectedDriveLetter
Expected drive letter without a colon when a mounted volume is required.

.PARAMETER ExpectedFileSystem
Expected filesystem such as NTFS when a mounted volume is required.

.PARAMETER ExpectedDiskUniqueId
Optional expected disk unique identifier.

.PARAMETER ExpectedState
Volume requires a Windows volume. AttachedDiskOnly only requires the VHDX to be
attached and discoverable as a disk.

.PARAMETER DryRun
Validate inputs and inspect current VHD state without calling Mount-VHD or
writing Windows event-log entries. The long CLI form `--DryRun` is also
accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Mount-PersistentVHDX.ps1 -VhdPath T:\vm\shared-dev.vhdx -ExpectedVolumeLabel WSL-Shared-Dev -ExpectedDriveLetter W -ExpectedFileSystem NTFS -TaskName AutoMount_VHDX_shared-dev
#>

[CmdletBinding()]
param(
    [string]$VhdPath,
    [string]$ExpectedVolumeLabel,
    [ValidatePattern('^[A-Za-z]$')]
    [string]$ExpectedDriveLetter,
    [string]$ExpectedFileSystem,
    [string]$ExpectedDiskUniqueId,
    [ValidateSet('Volume', 'AttachedDiskOnly')]
    [string]$ExpectedState = 'Volume',
    [string]$TaskName = 'Manual-PersistentVHDXMount',
    [int]$StartupDelaySeconds = 0,
    [int]$MountTimeoutSeconds = 120,
    [int]$FilterManagerEventLookbackSeconds = 300,
    [string]$LogRoot = (Join-Path $PSScriptRoot '..\Logs\VHDMount'),
    [switch]$SkipEventSourceRegistration,
    [switch]$PassThru,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0

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

$script:VhdMountEventSource = 'PC-AI-VHDMount'
$script:VhdMountEventLogName = 'Application'
$script:VhdMountExitCodes = @{
    Success = 0
    Degraded = 40
    InvalidInput = 50
    MissingVhd = 51
    MissingHyperV = 52
    MountTimeout = 53
    VerificationFailed = 54
    UnhandledException = 99
}

function New-PersistentVHDXRunResult {
    param(
        [string]$VhdPath,
        [string]$TaskName,
        [string]$LogRoot
    )

    $started = Get-Date
    $stamp = $started.ToString('yyyyMMdd-HHmmss')
    $safeTaskName = if ([string]::IsNullOrWhiteSpace($TaskName)) { 'Manual-PersistentVHDXMount' } else { $TaskName -replace '[^A-Za-z0-9_.-]', '_' }
    $runId = '{0}-{1}' -f $stamp, ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $runRoot = Join-Path $LogRoot $safeTaskName

    if (-not (Test-Path -LiteralPath $runRoot)) {
        New-Item -Path $runRoot -ItemType Directory -Force | Out-Null
    }

    [ordered]@{
        SchemaVersion = 1
        RunId = $runId
        TaskName = $TaskName
        VhdPath = $VhdPath
        Expected = [ordered]@{}
        Status = 'Started'
        ExitCode = $script:VhdMountExitCodes.UnhandledException
        StartedAt = $started.ToString('o')
        CompletedAt = $null
        AlreadyAttached = $false
        MountAttempted = $false
        HyperV = [ordered]@{
            GetVHD = $false
            MountVHD = $false
        }
        Vhd = $null
        Disk = $null
        Partitions = @()
        Volumes = @()
        FilterManager = [ordered]@{
            FltmcVolumesQueried = $false
            MountedVolumeVisible = $null
            EventId3Count = 0
            EventId3 = @()
            Error = $null
        }
        DegradedReasons = @()
        Errors = @()
        Logs = [ordered]@{
            Root = $runRoot
            Transcript = Join-Path $runRoot ('{0}.transcript.log' -f $runId)
            Json = Join-Path $runRoot ('{0}.result.json' -f $runId)
        }
    }
}

function Add-PersistentVHDXIssue {
    param(
        [System.Collections.IDictionary]$Result,
        [string]$Message,
        [switch]$Degraded
    )

    if ($Degraded) {
        $items = @($Result.DegradedReasons)
        $items += $Message
        $Result.DegradedReasons = $items
    } else {
        $items = @($Result.Errors)
        $items += $Message
        $Result.Errors = $items
    }
}

function ConvertTo-PersistentVHDXSimpleObject {
    param([object]$InputObject)

    if ($null -eq $InputObject) {
        return $null
    }

    $properties = [ordered]@{}
    foreach ($name in @('Path', 'Attached', 'DiskNumber', 'DiskIdentifier', 'ComputerName', 'VhdFormat', 'VhdType', 'FileSize', 'Size', 'Number', 'UniqueId', 'FriendlyName', 'PartitionNumber', 'DriveLetter', 'Type', 'Guid', 'FileSystemLabel', 'FileSystem', 'HealthStatus')) {
        if ($InputObject.PSObject.Properties.Name -contains $name) {
            $properties[$name] = $InputObject.$name
        }
    }

    [pscustomobject]$properties
}

function Save-PersistentVHDXRunResult {
    param([hashtable]$Result)

    $Result.CompletedAt = (Get-Date).ToString('o')
    $json = $Result | ConvertTo-Json -Depth 8
    Set-Content -LiteralPath $Result.Logs.Json -Value $json -Encoding UTF8
}

function Register-PersistentVHDXEventSource {
    param([switch]$Skip)

    if ($Skip) {
        return
    }

    try {
        if (-not [System.Diagnostics.EventLog]::SourceExists($script:VhdMountEventSource)) {
            New-EventLog -LogName $script:VhdMountEventLogName -Source $script:VhdMountEventSource -ErrorAction Stop
        }
    } catch {
        Write-Warning ("Unable to register event source {0}: {1}" -f $script:VhdMountEventSource, $_.Exception.Message)
    }
}

function Write-PersistentVHDXEvent {
    param(
        [ValidateSet('Information', 'Warning', 'Error')]
        [string]$EntryType,
        [int]$EventId,
        [string]$Message
    )

    try {
        Write-EventLog -LogName $script:VhdMountEventLogName -Source $script:VhdMountEventSource -EntryType $EntryType -EventId $EventId -Message $Message -ErrorAction Stop
    } catch {
        Write-Verbose ("Unable to write event {0}: {1}" -f $EventId, $_.Exception.Message)
    }
}

function Test-PersistentVHDXHyperVCommands {
    $getVhd = Get-Command -Name Get-VHD -ErrorAction SilentlyContinue
    $mountVhd = Get-Command -Name Mount-VHD -ErrorAction SilentlyContinue

    [pscustomobject]@{
        GetVHD = $null -ne $getVhd
        MountVHD = $null -ne $mountVhd
        Available = ($null -ne $getVhd -and $null -ne $mountVhd)
    }
}

function Wait-PersistentVHDXAttached {
    param(
        [string]$Path,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds([Math]::Max(0, $TimeoutSeconds))
    do {
        $vhd = Get-VHD -Path $Path -ErrorAction Stop
        if ($vhd.Attached) {
            return $vhd
        }

        if ((Get-Date) -ge $deadline) {
            break
        }

        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    return $vhd
}

function Resolve-PersistentVHDXDisk {
    param(
        [object]$Vhd,
        [string]$ExpectedDiskUniqueId
    )

    if ($null -ne $Vhd -and ($Vhd.PSObject.Properties.Name -contains 'DiskNumber') -and $null -ne $Vhd.DiskNumber) {
        return Get-Disk -Number $Vhd.DiskNumber -ErrorAction Stop
    }

    $disks = @(Get-Disk -ErrorAction Stop)
    if (-not [string]::IsNullOrWhiteSpace($ExpectedDiskUniqueId)) {
        return $disks | Where-Object { $_.UniqueId -eq $ExpectedDiskUniqueId } | Select-Object -First 1
    }

    if ($null -ne $Vhd -and ($Vhd.PSObject.Properties.Name -contains 'DiskIdentifier') -and -not [string]::IsNullOrWhiteSpace($Vhd.DiskIdentifier)) {
        return $disks | Where-Object { $_.UniqueId -eq $Vhd.DiskIdentifier } | Select-Object -First 1
    }

    return $null
}

function Resolve-PersistentVHDXVolumes {
    param(
        [object]$Disk,
        [string]$ExpectedDriveLetter
    )

    $volumes = @()
    $partitions = @()

    if ($null -ne $Disk) {
        $partitions = @(Get-Partition -DiskNumber $Disk.Number -ErrorAction Stop)
        foreach ($partition in $partitions) {
            try {
                $volume = Get-Volume -Partition $partition -ErrorAction Stop
                if ($null -ne $volume) {
                    $volumes += $volume
                }
            } catch {
                Write-Verbose ("No Windows volume for disk {0} partition {1}: {2}" -f $Disk.Number, $partition.PartitionNumber, $_.Exception.Message)
            }
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($ExpectedDriveLetter)) {
        try {
            $expectedVolume = Get-Volume -DriveLetter $ExpectedDriveLetter -ErrorAction Stop
            if ($null -ne $expectedVolume -and -not ($volumes | Where-Object { $_.Path -eq $expectedVolume.Path })) {
                $volumes += $expectedVolume
            }
        } catch {
            Write-Verbose ("Expected drive {0}: was not resolved: {1}" -f $ExpectedDriveLetter, $_.Exception.Message)
        }
    }

    [pscustomobject]@{
        Partitions = $partitions
        Volumes = $volumes
    }
}

function Test-PersistentVHDXFilterManagerVisibility {
    param([object[]]$Volumes)

    $output = $null
    try {
        $output = fltmc volumes 2>&1
        $text = ($output | Out-String)
        $visible = $null

        foreach ($volume in @($Volumes)) {
            $candidates = @()
            if ($volume.PSObject.Properties.Name -contains 'Path') {
                $candidates += $volume.Path
            }
            if ($volume.PSObject.Properties.Name -contains 'DriveLetter' -and -not [string]::IsNullOrWhiteSpace([string]$volume.DriveLetter)) {
                $candidates += ('{0}:' -f $volume.DriveLetter)
            }

            foreach ($candidate in $candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) {
                if ($text -like ('*{0}*' -f $candidate)) {
                    $visible = $true
                    break
                }
            }
        }

        if ($null -eq $visible -and @($Volumes).Count -gt 0) {
            $visible = $false
        }

        [pscustomobject]@{
            Queried = $true
            Visible = $visible
            Output = $text
            Error = $null
        }
    } catch {
        [pscustomobject]@{
            Queried = $false
            Visible = $null
            Output = $null
            Error = $_.Exception.Message
        }
    }
}

function Get-PersistentVHDXFilterManagerEventId3 {
    param(
        [datetime]$StartTime,
        [int]$LookbackSeconds
    )

    $queryStart = $StartTime
    if ($LookbackSeconds -gt 0) {
        $lookbackStart = (Get-Date).AddSeconds(-1 * $LookbackSeconds)
        if ($lookbackStart -lt $queryStart) {
            $queryStart = $lookbackStart
        }
    }

    try {
        @(Get-WinEvent -FilterHashtable @{
                ProviderName = 'Microsoft-Windows-FilterManager'
                Id = 3
                StartTime = $queryStart
            } -ErrorAction SilentlyContinue | ForEach-Object {
                [pscustomobject]@{
                    TimeCreated = $_.TimeCreated
                    ProviderName = $_.ProviderName
                    Id = $_.Id
                    LevelDisplayName = $_.LevelDisplayName
                    Message = $_.Message
                }
            })
    } catch {
        Write-Verbose ("Unable to query FilterManager Event ID 3: {0}" -f $_.Exception.Message)
        @()
    }
}

function Invoke-PersistentVHDXMount {
    [CmdletBinding()]
    param(
        [string]$VhdPath,
        [string]$ExpectedVolumeLabel,
        [ValidatePattern('^[A-Za-z]$')]
        [string]$ExpectedDriveLetter,
        [string]$ExpectedFileSystem,
        [string]$ExpectedDiskUniqueId,
        [ValidateSet('Volume', 'AttachedDiskOnly')]
        [string]$ExpectedState = 'Volume',
        [string]$TaskName = 'Manual-PersistentVHDXMount',
        [int]$StartupDelaySeconds = 0,
        [int]$MountTimeoutSeconds = 120,
        [int]$FilterManagerEventLookbackSeconds = 300,
        [string]$LogRoot = (Join-Path $PSScriptRoot '..\Logs\VHDMount'),
        [switch]$SkipEventSourceRegistration,
        [switch]$DryRun
    )

    $result = New-PersistentVHDXRunResult -VhdPath $VhdPath -TaskName $TaskName -LogRoot $LogRoot
    $result.Expected = [ordered]@{
        VolumeLabel = $ExpectedVolumeLabel
        DriveLetter = $ExpectedDriveLetter
        FileSystem = $ExpectedFileSystem
        DiskUniqueId = $ExpectedDiskUniqueId
        State = $ExpectedState
        StartupDelaySeconds = $StartupDelaySeconds
        DryRun = [bool]$DryRun
    }

    $transcriptStarted = $false
    try {
        try {
            Start-Transcript -Path $result.Logs.Transcript -Force -ErrorAction Stop | Out-Null
            $transcriptStarted = $true
        } catch {
            Add-PersistentVHDXIssue -Result $result -Message ("Transcript failed to start: {0}" -f $_.Exception.Message) -Degraded
        }

        if (-not $DryRun) {
            Register-PersistentVHDXEventSource -Skip:$SkipEventSourceRegistration
            Write-PersistentVHDXEvent -EntryType Information -EventId 1000 -Message ("Starting VHDX mount validation for {0} from task {1}" -f $VhdPath, $TaskName)
        }

        if ([string]::IsNullOrWhiteSpace($VhdPath)) {
            $result.Status = 'Failed'
            $result.ExitCode = $script:VhdMountExitCodes.InvalidInput
            Add-PersistentVHDXIssue -Result $result -Message 'VhdPath is required.'
            Write-PersistentVHDXEvent -EntryType Error -EventId 3000 -Message 'VHDX mount failed: VhdPath is required.'
            return [pscustomobject]$result
        }

        if (-not (Test-Path -LiteralPath $VhdPath)) {
            $result.Status = 'Failed'
            $result.ExitCode = $script:VhdMountExitCodes.MissingVhd
            Add-PersistentVHDXIssue -Result $result -Message ("Missing VHDX file: {0}" -f $VhdPath)
            Write-PersistentVHDXEvent -EntryType Error -EventId 3000 -Message ("VHDX mount failed because the file is missing: {0}" -f $VhdPath)
            return [pscustomobject]$result
        }

        $hyperV = Test-PersistentVHDXHyperVCommands
        $result.HyperV.GetVHD = $hyperV.GetVHD
        $result.HyperV.MountVHD = $hyperV.MountVHD
        if (-not $hyperV.Available) {
            $result.Status = 'Failed'
            $result.ExitCode = $script:VhdMountExitCodes.MissingHyperV
            Add-PersistentVHDXIssue -Result $result -Message 'Hyper-V PowerShell commands Get-VHD and Mount-VHD are required.'
            Write-PersistentVHDXEvent -EntryType Error -EventId 3004 -Message 'VHDX mount failed because Hyper-V PowerShell commands are unavailable.'
            return [pscustomobject]$result
        }

        $mountStart = Get-Date
        $vhd = Get-VHD -Path $VhdPath -ErrorAction Stop
        $result.Vhd = ConvertTo-PersistentVHDXSimpleObject -InputObject $vhd
        if ($vhd.Attached) {
            $result.AlreadyAttached = $true
            if (-not $DryRun) {
                Write-PersistentVHDXEvent -EntryType Information -EventId 1001 -Message ("VHDX was already attached: {0}" -f $VhdPath)
            }
        } else {
            if ($DryRun) {
                $result.Status = 'DryRun'
                $result.ExitCode = $script:VhdMountExitCodes.Success
                Add-PersistentVHDXIssue -Result $result -Message ("Dry run: VHDX is detached and would be mounted: {0}" -f $VhdPath) -Degraded
                return [pscustomobject]$result
            }
            $result.MountAttempted = $true
            Mount-VHD -Path $VhdPath -ErrorAction Stop
            $vhd = Wait-PersistentVHDXAttached -Path $VhdPath -TimeoutSeconds $MountTimeoutSeconds
            $result.Vhd = ConvertTo-PersistentVHDXSimpleObject -InputObject $vhd
            if (-not $vhd.Attached) {
                $result.Status = 'Failed'
                $result.ExitCode = $script:VhdMountExitCodes.MountTimeout
                Add-PersistentVHDXIssue -Result $result -Message ("Timed out waiting for VHDX attachment after {0} seconds: {1}" -f $MountTimeoutSeconds, $VhdPath)
                Write-PersistentVHDXEvent -EntryType Error -EventId 3005 -Message ("VHDX mount timed out after {0} seconds: {1}" -f $MountTimeoutSeconds, $VhdPath)
                return [pscustomobject]$result
            }
        }

        $disk = Resolve-PersistentVHDXDisk -Vhd $vhd -ExpectedDiskUniqueId $ExpectedDiskUniqueId
        if ($null -eq $disk) {
            $result.Status = 'Failed'
            $result.ExitCode = $script:VhdMountExitCodes.VerificationFailed
            Add-PersistentVHDXIssue -Result $result -Message 'Attached VHDX could not be resolved to a Windows disk.'
            Write-PersistentVHDXEvent -EntryType Error -EventId 3000 -Message ("VHDX verification failed because no Windows disk was resolved: {0}" -f $VhdPath)
            return [pscustomobject]$result
        }

        $result.Disk = ConvertTo-PersistentVHDXSimpleObject -InputObject $disk
        if (-not [string]::IsNullOrWhiteSpace($ExpectedDiskUniqueId) -and $disk.UniqueId -ne $ExpectedDiskUniqueId) {
            Add-PersistentVHDXIssue -Result $result -Message ("Expected disk UniqueId {0}, found {1}." -f $ExpectedDiskUniqueId, $disk.UniqueId)
        }

        $resolved = Resolve-PersistentVHDXVolumes -Disk $disk -ExpectedDriveLetter $ExpectedDriveLetter
        $result.Partitions = @($resolved.Partitions | ForEach-Object { ConvertTo-PersistentVHDXSimpleObject -InputObject $_ })
        $result.Volumes = @($resolved.Volumes | ForEach-Object { ConvertTo-PersistentVHDXSimpleObject -InputObject $_ })

        if ($ExpectedState -eq 'Volume') {
            if (@($resolved.Volumes).Count -eq 0) {
                Add-PersistentVHDXIssue -Result $result -Message 'Expected a Windows volume but none was resolved.'
                Write-PersistentVHDXEvent -EntryType Error -EventId 3002 -Message ("VHDX verification failed because no Windows volume was resolved: {0}" -f $VhdPath)
            }

            $selectedVolume = $null
            if (-not [string]::IsNullOrWhiteSpace($ExpectedDriveLetter)) {
                $selectedVolume = @($resolved.Volumes | Where-Object { [string]$_.DriveLetter -ieq $ExpectedDriveLetter }) | Select-Object -First 1
                if ($null -eq $selectedVolume) {
                    Add-PersistentVHDXIssue -Result $result -Message ("Expected drive letter {0}: was not present." -f $ExpectedDriveLetter)
                    Write-PersistentVHDXEvent -EntryType Error -EventId 3003 -Message ("VHDX verification failed for {0}: expected drive {1}: was not present." -f $VhdPath, $ExpectedDriveLetter)
                }
            } elseif (@($resolved.Volumes).Count -gt 0) {
                $selectedVolume = @($resolved.Volumes)[0]
            }

            if ($null -ne $selectedVolume) {
                if (-not [string]::IsNullOrWhiteSpace($ExpectedVolumeLabel) -and $selectedVolume.FileSystemLabel -ne $ExpectedVolumeLabel) {
                    Add-PersistentVHDXIssue -Result $result -Message ("Expected volume label {0}, found {1}." -f $ExpectedVolumeLabel, $selectedVolume.FileSystemLabel)
                }
                if (-not [string]::IsNullOrWhiteSpace($ExpectedFileSystem) -and $selectedVolume.FileSystem -ne $ExpectedFileSystem) {
                    Add-PersistentVHDXIssue -Result $result -Message ("Expected filesystem {0}, found {1}." -f $ExpectedFileSystem, $selectedVolume.FileSystem)
                }
            }
        }

        $filterVisibility = Test-PersistentVHDXFilterManagerVisibility -Volumes $resolved.Volumes
        $result.FilterManager.FltmcVolumesQueried = $filterVisibility.Queried
        $result.FilterManager.MountedVolumeVisible = $filterVisibility.Visible
        $result.FilterManager.Error = $filterVisibility.Error
        if ($ExpectedState -eq 'Volume' -and $filterVisibility.Visible -eq $false) {
            Add-PersistentVHDXIssue -Result $result -Message 'Mounted volume was not visible in fltmc volumes output.' -Degraded
        }
        if (-not [string]::IsNullOrWhiteSpace($filterVisibility.Error)) {
            Add-PersistentVHDXIssue -Result $result -Message ("fltmc volumes query failed: {0}" -f $filterVisibility.Error) -Degraded
        }

        $filterEvents = @(Get-PersistentVHDXFilterManagerEventId3 -StartTime $mountStart -LookbackSeconds $FilterManagerEventLookbackSeconds)
        $result.FilterManager.EventId3Count = $filterEvents.Count
        $result.FilterManager.EventId3 = @($filterEvents)
        if ($filterEvents.Count -gt 0) {
            Add-PersistentVHDXIssue -Result $result -Message ("FilterManager Event ID 3 occurred {0} time(s) after the mount window." -f $filterEvents.Count) -Degraded
            Write-PersistentVHDXEvent -EntryType Warning -EventId 3001 -Message ("FilterManager Event ID 3 occurred after mounting {0}. See JSON result: {1}" -f $VhdPath, $result.Logs.Json)
        }

        if (@($result.Errors).Count -gt 0) {
            $result.Status = 'Failed'
            $result.ExitCode = $script:VhdMountExitCodes.VerificationFailed
            if (-not $DryRun) {
                Write-PersistentVHDXEvent -EntryType Error -EventId 3000 -Message ("VHDX verification failed for {0}: {1}" -f $VhdPath, (@($result.Errors) -join '; '))
            }
        } elseif (@($result.DegradedReasons).Count -gt 0) {
            $result.Status = 'Degraded'
            $result.ExitCode = $script:VhdMountExitCodes.Degraded
            if (-not $DryRun) {
                Write-PersistentVHDXEvent -EntryType Warning -EventId 2000 -Message ("VHDX mount degraded for {0}: {1}" -f $VhdPath, (@($result.DegradedReasons) -join '; '))
            }
        } else {
            $result.Status = if ($DryRun) { 'DryRun' } else { 'Success' }
            $result.ExitCode = $script:VhdMountExitCodes.Success
            if (-not $DryRun) {
                Write-PersistentVHDXEvent -EntryType Information -EventId 1002 -Message ("VHDX mount validated successfully for {0}" -f $VhdPath)
            }
        }

        return [pscustomobject]$result
    } catch {
        $result.Status = 'Failed'
        $result.ExitCode = $script:VhdMountExitCodes.UnhandledException
        Add-PersistentVHDXIssue -Result $result -Message ("Unhandled exception: {0}" -f $_.Exception.Message)
        Write-PersistentVHDXEvent -EntryType Error -EventId 3099 -Message ("Unhandled VHDX mount exception for {0}: {1}" -f $VhdPath, $_.Exception.Message)
        return [pscustomobject]$result
    } finally {
        try {
            Save-PersistentVHDXRunResult -Result $result
        } catch {
            Write-Warning ("Unable to save VHDX JSON result: {0}" -f $_.Exception.Message)
        }

        if ($transcriptStarted) {
            try {
                Stop-Transcript | Out-Null
            } catch {
                Write-Verbose ("Unable to stop transcript: {0}" -f $_.Exception.Message)
            }
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ([string]::IsNullOrWhiteSpace($LogRoot)) {
        $LogRoot = Join-Path $PSScriptRoot '..\Logs\VHDMount'
    }
    $mountParameters = @{} + $PSBoundParameters
    $mountParameters.Remove('PassThru')
    $mountParameters.Remove('Help')
    $mountParameters.Remove('CliArgs')
    $mountParameters['LogRoot'] = $LogRoot
    $runResult = Invoke-PersistentVHDXMount @mountParameters
    if ($PassThru) {
        $runResult
    }
    exit $runResult.ExitCode
}
