#Requires -Version 5.1
<#
.SYNOPSIS
Repair a missing cloud cache after proving its image and volume identity.
.DESCRIPTION
Healthy volumes must be F:, NTFS and associated with the expected attached image.
Repairs require live identity and a post-request successful receipt within one
bounded deadline. Conflicting identities fail closed. Healthy runs leave clients
and cache configuration alone. Incomplete Storage associations and query errors
wait within that same deadline; initial pending observations wait passively.
.PARAMETER DryRun
Report the decision without filesystem writes or task, mount or process changes.
.PARAMETER Help
Print help. -h and --help are also supported.
.EXAMPLE
pwsh -File .\Tools\Repair-CloudCacheMount.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateNotNullOrEmpty()][string]$Label = 'cloud-cache-disk',
    [ValidateNotNullOrEmpty()][string]$ExpectedVhd = 'T:\vm\cloud-cache-disk.vhdx',
    [ValidatePattern('^[A-Za-z]$')][string]$ExpectedDriveLetter = 'F',
    [ValidateNotNullOrEmpty()][string]$MountTask = 'AutoMount_VHDX_cloud-cache-disk',
    [ValidateNotNullOrEmpty()][string]$LauncherTask = 'CloudClients-AfterVHDX',
    [ValidateRange(1, 3600)][int]$MountWaitSeconds = 180,
    [ValidateRange(100, 5000)][int]$PollMilliseconds = 1000,
    [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
    [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudCacheRepair',
    [switch]$DryRun, [Alias('h', '?')][switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs
)

function Get-CacheVolumeIdentity {
    param($Volume)
    $identity = $null
    if ($null -eq $Volume) { return $null }
    foreach ($name in @('Path', 'UniqueId')) {
        $property = $Volume.PSObject.Properties[$name]
        if ($property -and -not [string]::IsNullOrWhiteSpace([string]$property.Value)) {
            $guid = [guid]::Empty
            if ([string]$property.Value -notmatch '^\\\\\?\\Volume\{([0-9a-fA-F-]{36})\}\\$' -or
                -not [guid]::TryParse($Matches[1], [ref]$guid) -or $guid -eq [guid]::Empty) { return $null }
            if ($identity -and $identity -ne $guid.ToString('D')) { return $null }
            $identity = $guid.ToString('D')
        }
    }
    return $identity
}

function Get-CacheVolumeState {
    [CmdletBinding()]
    param([string]$Label, [string]$ExpectedVhd, [string]$ExpectedDriveLetter = 'F')
    $state = @{ State = 'Pending'; Volume = $null; Identity = $null; Reason = ''; Backing = $ExpectedVhd }
    try {
        $volumes = @(Get-Volume -ErrorAction Stop | Where-Object { $null -ne $_ })
        $state.State = 'InvalidIdentity'
        $named = @($volumes | Where-Object { [string]$_.FileSystemLabel -eq $Label })
        $atDrive = @($volumes | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter })
        if ($atDrive.Count -gt 1) { throw 'Expected drive letter resolves to multiple volumes.' }
        if ($atDrive.Count -eq 1 -and -not [string]::IsNullOrWhiteSpace([string]$atDrive[0].FileSystemLabel) -and
            [string]$atDrive[0].FileSystemLabel -ne $Label) {
            throw 'Expected drive letter is occupied by a different volume.'
        }
        if ($named.Count -eq 0) {
            if ($atDrive.Count -eq 0) { $state.State = 'Missing'; return $state }
            # An unlabeled row at the expected drive is a provisional candidate,
            # never a healthy identity. Continue the available conflict checks.
            $named = @($atDrive[0])
        }
        if ($named.Count -ne 1) { throw 'The label resolves to multiple volumes.' }
        $volume = $named[0]
        $state.Volume = $volume
        $identity = Get-CacheVolumeIdentity $volume
        if (-not $identity) { throw 'The volume lacks a valid nonempty GUID.' }
        $letter = [string]$volume.DriveLetter
        $letterPending = [string]::IsNullOrWhiteSpace($letter) -or $letter -eq [string][char]0
        if (-not $letterPending -and $letter -ne $ExpectedDriveLetter) { throw 'The labeled volume has a conflicting drive letter.' }
        $fileSystem = [string]$volume.FileSystem
        if (-not [string]::IsNullOrWhiteSpace($fileSystem) -and $fileSystem -ne 'NTFS') { throw 'The labeled volume has a conflicting filesystem.' }
        $volumePending = $letterPending -or [string]::IsNullOrWhiteSpace($fileSystem) -or
            [string]::IsNullOrWhiteSpace([string]$volume.FileSystemLabel)
        if (-not $volumePending -and $atDrive.Count -ne 1) { throw 'Expected drive letter does not resolve to one volume.' }
        $state.State = 'Pending'
        $images = @(Get-DiskImage -ImagePath $ExpectedVhd -ErrorAction Stop | Where-Object { $null -ne $_ })
        $state.State = 'InvalidIdentity'
        if ($images.Count -eq 0) { $state.State = 'Pending'; $state.Reason = 'Expected image association is not yet available.'; return $state }
        if ($images.Count -ne 1 -or $images[0].Attached -ne $true -or
            [string]$images[0].ImagePath -ne $ExpectedVhd) { throw 'Expected image is detached, ambiguous or mismatched.' }
        # MSFT_DiskImage.Number may be null; use the actual Storage association.
        $state.State = 'Pending'
        $disks = @($images[0] | Get-Disk -ErrorAction Stop | Where-Object { $null -ne $_ })
        $state.State = 'InvalidIdentity'
        if ($disks.Count -eq 0 -or ($disks.Count -eq 1 -and $null -eq $disks[0].Number)) {
            $state.State = 'Pending'; $state.Reason = 'Expected image disk association is incomplete.'; return $state
        }
        $number = 0
        if ($disks.Count -ne 1 -or
            -not [int]::TryParse([string]$disks[0].Number, [ref]$number) -or $number -lt 0) {
            throw 'Expected image does not resolve to one valid disk number.'
        }
        $state.State = 'Pending'
        $diskPartitions = @(Get-Partition -DiskNumber $number -ErrorAction Stop | Where-Object { $null -ne $_ })
        $state.State = 'InvalidIdentity'
        if ($diskPartitions.Count -eq 0) { $state.State = 'Pending'; $state.Reason = 'Expected image partition association is not yet available.'; return $state }
        foreach ($diskPartition in $diskPartitions) {
            if ($null -ne $diskPartition.DiskNumber) {
                $observedDisk = 0
                if (-not [int]::TryParse([string]$diskPartition.DiskNumber, [ref]$observedDisk) -or $observedDisk -ne $number) {
                    throw 'Returned partition belongs to a conflicting or malformed disk.'
                }
            }
        }
        $partitions = @($diskPartitions | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter })
        if ($partitions.Count -eq 0) {
            $assigned = @($diskPartitions | Where-Object {
                -not [string]::IsNullOrWhiteSpace([string]$_.DriveLetter) -and [string]$_.DriveLetter -ne [string][char]0
            })
            if ($assigned.Count -gt 0) { throw 'Expected image partitions have conflicting drive assignments.' }
            $state.State = 'Pending'; $state.Reason = 'Expected image partition drive assignment is incomplete.'; return $state
        }
        if ($partitions.Count -eq 1 -and $null -eq $partitions[0].DiskNumber) {
            $state.State = 'Pending'; $state.Reason = 'Matching partition disk association is incomplete.'; return $state
        }
        $partitionDisk = 0
        if ($partitions.Count -ne 1 -or
            -not [int]::TryParse([string]$partitions[0].DiskNumber, [ref]$partitionDisk) -or $partitionDisk -ne $number) {
            throw 'Expected image has no unique matching partition on its disk.'
        }
        $state.State = 'Pending'
        $associated = @(Get-Volume -Partition $partitions[0] -ErrorAction Stop | Where-Object { $null -ne $_ })
        $state.State = 'InvalidIdentity'
        if ($associated.Count -eq 0) { $state.State = 'Pending'; $state.Reason = 'Partition volume association is not yet available.'; return $state }
        if ($associated.Count -ne 1 -or (Get-CacheVolumeIdentity $associated[0]) -ne $identity) {
            throw 'Associated volume identity changed or does not match the expected cache.'
        }
        $associatedLetter = [string]$associated[0].DriveLetter
        $associatedFs = [string]$associated[0].FileSystem
        $associatedLabel = [string]$associated[0].FileSystemLabel
        if ((-not [string]::IsNullOrWhiteSpace($associatedLetter) -and $associatedLetter -ne [string][char]0 -and $associatedLetter -ne $ExpectedDriveLetter) -or
            (-not [string]::IsNullOrWhiteSpace($associatedFs) -and $associatedFs -ne 'NTFS') -or
            (-not [string]::IsNullOrWhiteSpace($associatedLabel) -and $associatedLabel -ne $Label)) {
            throw 'Associated volume drive, filesystem or label conflicts with the expected cache.'
        }
        if ([string]::IsNullOrWhiteSpace($associatedLetter) -or $associatedLetter -eq [string][char]0 -or
            [string]::IsNullOrWhiteSpace($associatedFs) -or [string]::IsNullOrWhiteSpace($associatedLabel)) {
            $state.State = 'Pending'; $state.Reason = 'Associated volume properties are incomplete.'; return $state
        }
        if ($volumePending) {
            $state.State = 'Pending'; $state.Reason = 'Primary volume label, drive or filesystem assignment is incomplete.'; return $state
        }
        $state.State = 'Healthy'
        $state.Identity = $identity
    } catch {
        $state.Reason = if ($state.State -eq 'Pending') { 'Storage query is pending: ' + $_.Exception.Message } else { $_.Exception.Message }
    }
    return $state
}

function Get-CacheRepairReceipt {
    [CmdletBinding()]
    param([string]$MountLogRoot, [datetime]$RequestedAt, [string]$ExpectedVhd,
        [string]$Label, [string]$ExpectedDriveLetter, [string]$Identity)
    $pending = @{ State = 'Pending'; Reason = 'No completed post-request receipt.' }
    try {
        $latest = Get-ChildItem -LiteralPath $MountLogRoot -Filter '*.result.json' -ErrorAction Stop |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latest) { return $pending }
        $receipt = Get-Content -LiteralPath $latest.FullName -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        # Direct producer writes can temporarily expose partial JSON. Keep the
        # original deadline while waiting for readable, complete evidence.
        foreach ($field in @('StartedAt', 'CompletedAt', 'VhdPath', 'Status', 'ExitCode')) {
            if (-not $receipt.PSObject.Properties[$field] -or $null -eq $receipt.$field) { return $pending }
        }
        $started = [datetime]$receipt.StartedAt
        $completed = [datetime]$receipt.CompletedAt
        if ($started -lt $RequestedAt) { return $pending }
        if ($started -gt (Get-Date) -or $completed -gt (Get-Date) -or $completed -lt $started) {
            return @{ State = 'Denied'; Reason = 'Receipt timestamps are inconsistent.' }
        }
        if ([string]$receipt.VhdPath -ne $ExpectedVhd -or [string]$receipt.Status -ne 'Success' -or
            [string]$receipt.ExitCode -ne '0') {
            return @{ State = 'Denied'; Reason = 'Fresh mount receipt reports failure or another image.' }
        }
        if (-not $receipt.PSObject.Properties['Volumes']) { return $pending }
        if (-not $Identity) { return $pending }
        $matching = @($receipt.Volumes | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter })
        if ($matching.Count -ne 1 -or [string]$matching[0].FileSystemLabel -ne $Label -or
            [string]$matching[0].FileSystem -ne 'NTFS' -or (Get-CacheVolumeIdentity $matching[0]) -ne $Identity) {
            return @{ State = 'Denied'; Reason = 'Fresh receipt does not bind the exact live volume GUID.' }
        }
        return @{ State = 'Success'; Reason = '' }
    } catch { $pending.Reason = 'Receipt read or parse is pending: ' + $_.Exception.Message; return $pending }
}

function Test-CacheInteractiveSession {
    try {
        if ((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).UserName) { return $true }
    } catch { Write-Verbose ('Interactive session query failed: ' + $_.Exception.Message) }
    return [bool](Get-Process -Name explorer -ErrorAction SilentlyContinue)
}

function Invoke-CloudCacheRepair {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [ValidateNotNullOrEmpty()][string]$Label = 'cloud-cache-disk',
        [ValidateNotNullOrEmpty()][string]$ExpectedVhd = 'T:\vm\cloud-cache-disk.vhdx',
        [ValidatePattern('^[A-Za-z]$')][string]$ExpectedDriveLetter = 'F',
        [ValidateNotNullOrEmpty()][string]$MountTask = 'AutoMount_VHDX_cloud-cache-disk',
        [ValidateNotNullOrEmpty()][string]$LauncherTask = 'CloudClients-AfterVHDX',
        [ValidateRange(1, 3600)][int]$MountWaitSeconds = 180,
        [ValidateRange(100, 5000)][int]$PollMilliseconds = 1000,
        [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
        [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudCacheRepair',
        [switch]$DryRun, [Alias('h', '?')][switch]$Help,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs
    )
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    if (@($CliArgs) -contains '--help') { $Help = $true }
    if (@($CliArgs) -contains '--DryRun') { $DryRun = $true }
    if ($Help) { Get-Help $script:CloudCacheRepairSource -Detailed | Out-String | Write-Information -InformationAction Continue; return 0 }
    $readOnly = $DryRun -or $WhatIfPreference
    $logFile = $null
    if (-not $readOnly) {
        New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
        $logFile = Join-Path $LogRoot 'current.log'
    }
    function Write-CacheRepairLog {
        param([string]$Message)
        $line = '{0:o} {1}' -f (Get-Date), $Message
        Write-Information $line -InformationAction Continue
        if ($logFile) { Add-Content -LiteralPath $logFile -Value $line }
    }
    $stateArgs = @{ Label = $Label; ExpectedVhd = $ExpectedVhd; ExpectedDriveLetter = $ExpectedDriveLetter }
    $state = Get-CacheVolumeState @stateArgs
    if ($state.State -eq 'Healthy') { Write-CacheRepairLog "Healthy volume $($state.Identity); no repair needed."; return 0 }
    if ($state.State -eq 'InvalidIdentity') { Write-CacheRepairLog "Refusing repair: $($state.Reason)"; return 3 }
    if (-not (Test-Path -LiteralPath $ExpectedVhd -PathType Leaf)) { Write-CacheRepairLog 'Backing image is missing.'; return 2 }
    if ($state.State -eq 'Pending' -and $readOnly) { Write-CacheRepairLog "Dry run: would passively wait for complete mount identity: $($state.Reason)"; return 0 }
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    # A pre-existing asynchronous mount/query may be settling. Observe it without
    # requesting a competing mount or restarting clients. Missing recovery uses
    # the remainder of this same deadline for the mount request and receipt.
    while ($state.State -eq 'Pending' -and $timer.Elapsed.TotalSeconds -lt $MountWaitSeconds) {
        $remaining = [int][math]::Ceiling(($MountWaitSeconds - $timer.Elapsed.TotalSeconds) * 1000)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([math]::Min($PollMilliseconds, $remaining)) }
        if ($timer.Elapsed.TotalSeconds -lt $MountWaitSeconds) { $state = Get-CacheVolumeState @stateArgs }
    }
    if ($state.State -eq 'InvalidIdentity') { Write-CacheRepairLog "Refusing repair: $($state.Reason)"; return 3 }
    if ($timer.Elapsed.TotalSeconds -ge $MountWaitSeconds) {
        Write-CacheRepairLog "Identity observation timed out; live state=$($state.State); reason=$($state.Reason)"
        return 1
    }
    if ($state.State -eq 'Healthy') { Write-CacheRepairLog "Existing mount settled for volume $($state.Identity); no client restart requested."; return 0 }
    if ($state.State -ne 'Missing') { Write-CacheRepairLog "Refusing unknown mount state: $($state.State)"; return 3 }
    if ($readOnly) { Write-CacheRepairLog 'Dry run: would request mount and validate its fresh receipt.'; return 0 }
    if (-not $PSCmdlet.ShouldProcess($MountTask, 'Start-ScheduledTask')) { return 0 }
    if ($timer.Elapsed.TotalSeconds -ge $MountWaitSeconds) { Write-CacheRepairLog 'Deadline elapsed before mount request.'; return 1 }
    $requested = Get-Date
    try { Start-ScheduledTask -TaskName $MountTask -ErrorAction Stop }
    catch { Write-CacheRepairLog ('Mount request failed: ' + $_.Exception.Message); return 1 }
    $receiptArgs = @{ MountLogRoot = $MountLogRoot; RequestedAt = $requested; ExpectedVhd = $ExpectedVhd;
        Label = $Label; ExpectedDriveLetter = $ExpectedDriveLetter }
    $receipt = @{ State = 'Pending'; Reason = 'Deadline elapsed before mount evidence was available.' }
    while ($timer.Elapsed.TotalSeconds -lt $MountWaitSeconds) {
        $state = Get-CacheVolumeState @stateArgs
        if ($state.State -eq 'InvalidIdentity') { Write-CacheRepairLog "Repair identity denied: $($state.Reason)"; return 3 }
        $receipt = Get-CacheRepairReceipt @receiptArgs -Identity $state.Identity
        if ($receipt.State -eq 'Denied') { Write-CacheRepairLog $receipt.Reason; return 3 }
        if ($state.State -eq 'Healthy' -and $receipt.State -eq 'Success' -and $timer.Elapsed.TotalSeconds -lt $MountWaitSeconds) {
            Write-CacheRepairLog "Repair validated for volume $($state.Identity)."
            if ((Test-CacheInteractiveSession) -and $PSCmdlet.ShouldProcess($LauncherTask, 'Start-ScheduledTask')) {
                try { Start-ScheduledTask -TaskName $LauncherTask -ErrorAction Stop }
                catch { Write-CacheRepairLog ('Launcher request failed: ' + $_.Exception.Message); return 1 }
                Write-CacheRepairLog 'Requested gated cloud launcher; client readiness is checked by that launcher.'
            }
            return 0
        }
        $remaining = [int][math]::Ceiling(($MountWaitSeconds - $timer.Elapsed.TotalSeconds) * 1000)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([math]::Min($PollMilliseconds, $remaining)) }
    }
    Write-CacheRepairLog "Repair timed out; live state=$($state.State); receipt=$($receipt.Reason)"
    return 1
}

$script:CloudCacheRepairSource = $PSCommandPath
if ($MyInvocation.InvocationName -ne '.') { exit (Invoke-CloudCacheRepair @PSBoundParameters) }
