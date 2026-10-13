#Requires -Version 5.1
<#
.SYNOPSIS
Repair a missing cloud cache after proving its image and volume identity.
.DESCRIPTION
Healthy volumes must be F:, NTFS and associated with the expected attached image.
Repairs require live identity and a post-request successful receipt within one
bounded deadline. Conflicting identities fail closed. Healthy runs leave clients
and cache configuration alone.
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
    $state = @{ State = 'InvalidIdentity'; Volume = $null; Identity = $null; Reason = ''; Backing = $ExpectedVhd }
    try {
        $volumes = @(Get-Volume -ErrorAction Stop)
        $named = @($volumes | Where-Object { [string]$_.FileSystemLabel -eq $Label })
        if ($named.Count -eq 0) {
            if (@($volumes | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter }).Count -gt 0) {
                $state.Reason = 'Expected drive letter is occupied by a different volume.'
                return $state
            }
            $state.State = 'Missing'
            return $state
        }
        if ($named.Count -ne 1) { throw 'The label resolves to multiple volumes.' }
        if (@($volumes | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter }).Count -ne 1) {
            throw 'Expected drive letter does not resolve to one volume.'
        }
        $volume = $named[0]
        $state.Volume = $volume
        if ([string]$volume.DriveLetter -ne $ExpectedDriveLetter -or [string]$volume.FileSystem -ne 'NTFS') {
            throw 'The labeled volume must have the expected drive letter and NTFS filesystem.'
        }
        $identity = Get-CacheVolumeIdentity $volume
        if (-not $identity) { throw 'The volume lacks a valid nonempty GUID.' }
        $images = @(Get-DiskImage -ImagePath $ExpectedVhd -ErrorAction Stop)
        if ($images.Count -ne 1 -or $images[0].Attached -ne $true -or
            [string]$images[0].ImagePath -ne $ExpectedVhd) { throw 'Expected image is absent, detached or mismatched.' }
        # MSFT_DiskImage.Number may be null; use the actual Storage association.
        $disks = @($images[0] | Get-Disk -ErrorAction Stop)
        $number = 0
        if ($disks.Count -ne 1 -or $null -eq $disks[0].Number -or
            -not [int]::TryParse([string]$disks[0].Number, [ref]$number) -or $number -lt 0) {
            throw 'Expected image does not resolve to one valid disk number.'
        }
        $partitions = @(Get-Partition -DiskNumber $number -ErrorAction Stop | Where-Object {
            [string]$_.DriveLetter -eq $ExpectedDriveLetter
        })
        $partitionDisk = 0
        if ($partitions.Count -ne 1 -or $null -eq $partitions[0].DiskNumber -or
            -not [int]::TryParse([string]$partitions[0].DiskNumber, [ref]$partitionDisk) -or $partitionDisk -ne $number) {
            throw 'Expected image has no unique matching partition on its disk.'
        }
        $associated = @(Get-Volume -Partition $partitions[0] -ErrorAction Stop)
        if ($associated.Count -ne 1 -or [string]$associated[0].DriveLetter -ne $ExpectedDriveLetter -or
            [string]$associated[0].FileSystem -ne 'NTFS' -or [string]$associated[0].FileSystemLabel -ne $Label -or
            (Get-CacheVolumeIdentity $associated[0]) -ne $identity) {
            throw 'Associated volume identity changed or does not match the expected cache.'
        }
        $state.State = 'Healthy'
        $state.Identity = $identity
    } catch { $state.Reason = $_.Exception.Message }
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
    if ($state.State -ne 'Missing') { Write-CacheRepairLog "Refusing repair: $($state.Reason)"; return 3 }
    if (-not (Test-Path -LiteralPath $ExpectedVhd -PathType Leaf)) { Write-CacheRepairLog 'Backing image is missing.'; return 2 }
    if ($readOnly) { Write-CacheRepairLog 'Dry run: would request mount and validate its fresh receipt.'; return 0 }
    if (-not $PSCmdlet.ShouldProcess($MountTask, 'Start-ScheduledTask')) { return 0 }
    $requested = Get-Date
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
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
