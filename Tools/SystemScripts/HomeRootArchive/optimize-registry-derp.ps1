<#
.SYNOPSIS
Audits and optionally applies workstation filesystem registry tuning.

.DESCRIPTION
This archived script is intentionally report-first. It no longer applies broad
filesystem/cache changes by default. Use -DryRun or the default invocation to
inspect the planned changes. Use -Apply only after reviewing cloud-sync risk.

The safe workstation profile avoids undocumented Disk, FltMgr, and ReFS cache
tweaks. It keeps long paths enabled and proposes conservative cache defaults
that are easier to validate with OneDrive and other cloud-sync providers.

.PARAMETER Profile
The tuning profile to evaluate. ReportOnly makes no proposed changes.
GeneralWorkstationSafe proposes conservative Windows workstation values.
ExperimentalFilesystemCache proposes higher-risk filesystem cache values.

.PARAMETER Apply
Apply the proposed changes. Without -Apply, the script is a dry run.

.PARAMETER DryRun
Preview planned changes without writing registry keys, snapshots, or reports.
The long CLI form --DryRun is also accepted.

.PARAMETER SnapshotPath
Path for touched-key snapshots used by -Apply and -RestoreFromSnapshot.

.PARAMETER RestoreFromSnapshot
Restore registry values from a prior touched-key snapshot.

.PARAMETER ForceCloudSyncRisk
Allow applying changes while cloud-sync providers are running or sync roots are
registered. Without this switch, -Apply is blocked when cloud-sync risk exists.

.PARAMETER OutputJson
Optional JSON report path. Reports are only written when not in dry-run mode.

.PARAMETER Help
Print script help and exit. The aliases -h and --help are also accepted.

.EXAMPLE
.\optimize-registry-derp.ps1
Preview the safe workstation profile without changing the machine.

.EXAMPLE
.\optimize-registry-derp.ps1 -Profile GeneralWorkstationSafe -Apply -ForceCloudSyncRisk
Apply conservative workstation values after explicitly accepting sync risk.

.EXAMPLE
.\optimize-registry-derp.ps1 -RestoreFromSnapshot -SnapshotPath .\snapshot.json -Apply
Restore values captured by an earlier apply run.
#>
[CmdletBinding()]
param(
    [ValidateSet('ReportOnly', 'GeneralWorkstationSafe', 'ExperimentalFilesystemCache')]
    [string]$Profile = 'GeneralWorkstationSafe',
    [switch]$Apply,
    [switch]$DryRun,
    [string]$SnapshotPath = (Join-Path $env:USERPROFILE 'RegistryTuningSnapshot.json'),
    [switch]$RestoreFromSnapshot,
    [switch]$ForceCloudSyncRisk,
    [string]$OutputJson = '',
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
if ($CliArgs -contains '--Apply') {
    $Apply = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--Apply' })
}
if ($CliArgs -contains '--ForceCloudSyncRisk') {
    $ForceCloudSyncRisk = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--ForceCloudSyncRisk' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}
if (-not $Apply) {
    $DryRun = $true
}

function Get-TuningSettings {
    param([string]$SelectedProfile)

    $safe = @(
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
            Name = 'LongPathsEnabled'
            Value = 1
            Rationale = 'Supported workstation compatibility setting.'
            Risk = 'Low'
        },
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
            Name = 'NtfsMemoryUsage'
            Value = 1
            Rationale = 'Conservative NTFS cache behavior for cloud-sync stability.'
            Risk = 'Medium'
        },
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
            Name = 'LargeSystemCache'
            Value = 0
            Rationale = 'Workstation/application bias rather than server-style file cache.'
            Risk = 'Medium'
        },
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
            Name = 'DisablePagingExecutive'
            Value = 0
            Rationale = 'Avoid broad kernel memory residency tuning during sync debugging.'
            Risk = 'Medium'
        }
    )

    $experimental = @(
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
            Name = 'NtfsMemoryUsage'
            Value = 2
            Rationale = 'High file-count cache experiment; not recommended while OneDrive is unstable.'
            Risk = 'High'
        },
        [pscustomobject]@{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
            Name = 'LargeSystemCache'
            Value = 1
            Rationale = 'Server-style file cache experiment; not a cloud-sync default.'
            Risk = 'High'
        }
    )

    switch ($SelectedProfile) {
        'ReportOnly' { @() }
        'GeneralWorkstationSafe' { $safe }
        'ExperimentalFilesystemCache' { $experimental }
    }
}

function Get-RegistryValueState {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ Exists = $false; Value = $null }
    }

    $item = Get-ItemProperty -LiteralPath $Path -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $item) {
        return [pscustomobject]@{ Exists = $false; Value = $null }
    }

    return [pscustomobject]@{ Exists = $true; Value = $item.$Name }
}

function Get-CloudSyncState {
    $processNames = @('OneDrive', 'OneDrive.Sync.Service', 'FileSyncHelper', 'GoogleDriveFS', 'Dropbox', 'iCloudDrive', 'ProtonDrive')
    $processes = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $processNames } |
        Select-Object ProcessName, Id, Path)

    $syncRoots = @()
    $rootKey = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager'
    if (Test-Path -LiteralPath $rootKey) {
        foreach ($provider in Get-ChildItem -LiteralPath $rootKey -ErrorAction SilentlyContinue) {
            $userRootsKey = Join-Path $provider.PSPath 'UserSyncRoots'
            if (Test-Path -LiteralPath $userRootsKey) {
                $props = Get-ItemProperty -LiteralPath $userRootsKey
                foreach ($property in $props.PSObject.Properties) {
                    if ($property.Name -notmatch '^PS') {
                        $syncRoots += [pscustomobject]@{
                            Provider = $provider.PSChildName
                            Sid = $property.Name
                            Path = [string]$property.Value
                        }
                    }
                }
            }
        }
    }

    [pscustomobject]@{
        RunningProcesses = $processes
        SyncRoots = $syncRoots
        HasRisk = (@($processes).Count -gt 0 -or @($syncRoots).Count -gt 0)
    }
}

function New-TouchedKeySnapshot {
    param([object[]]$Settings)

    foreach ($setting in $Settings) {
        $state = Get-RegistryValueState -Path $setting.Path -Name $setting.Name
        [pscustomobject]@{
            Path = $setting.Path
            Name = $setting.Name
            Existed = [bool]$state.Exists
            Value = $state.Value
            CapturedAt = (Get-Date).ToString('o')
        }
    }
}

function Apply-RegistrySetting {
    param([object]$Setting)

    if (-not (Test-Path -LiteralPath $Setting.Path)) {
        New-Item -Path $Setting.Path -Force | Out-Null
    }
    Set-ItemProperty -LiteralPath $Setting.Path -Name $Setting.Name -Value $Setting.Value -Type DWord -Force
}

function Restore-RegistrySnapshot {
    param([object[]]$Snapshot)

    foreach ($entry in $Snapshot) {
        if (-not (Test-Path -LiteralPath $entry.Path)) {
            New-Item -Path $entry.Path -Force | Out-Null
        }
        if ([bool]$entry.Existed) {
            Set-ItemProperty -LiteralPath $entry.Path -Name $entry.Name -Value $entry.Value -Type DWord -Force
        }
        else {
            Remove-ItemProperty -LiteralPath $entry.Path -Name $entry.Name -ErrorAction SilentlyContinue
        }
    }
}

$cloud = Get-CloudSyncState
$settings = @(Get-TuningSettings -SelectedProfile $Profile)
$changes = @()
foreach ($setting in $settings) {
    $state = Get-RegistryValueState -Path $setting.Path -Name $setting.Name
    $changes += [pscustomobject]@{
        Path = $setting.Path
        Name = $setting.Name
        CurrentValue = $state.Value
        CurrentExists = [bool]$state.Exists
        TargetValue = $setting.Value
        WouldChange = (-not $state.Exists) -or ([string]$state.Value -ne [string]$setting.Value)
        Risk = $setting.Risk
        Rationale = $setting.Rationale
    }
}

$warnings = @()
if ($cloud.HasRisk) {
    $warnings += 'Cloud-sync providers or registered sync roots are present.'
}
if ($Profile -eq 'ExperimentalFilesystemCache') {
    $warnings += 'ExperimentalFilesystemCache is not recommended while OneDrive is unstable.'
}

$result = [ordered]@{
    Script = $PSCommandPath
    GeneratedAt = (Get-Date).ToString('o')
    Profile = $Profile
    Apply = [bool]$Apply
    DryRun = [bool]$DryRun
    SnapshotPath = $SnapshotPath
    RestoreFromSnapshot = [bool]$RestoreFromSnapshot
    ForceCloudSyncRisk = [bool]$ForceCloudSyncRisk
    CloudSync = $cloud
    Changes = $changes
    Warnings = $warnings
    Actions = @()
}

if ($RestoreFromSnapshot) {
    if (-not (Test-Path -LiteralPath $SnapshotPath)) {
        throw "Snapshot not found: $SnapshotPath"
    }
    $snapshot = @(Get-Content -LiteralPath $SnapshotPath -Raw | ConvertFrom-Json)
    $result.Actions += "Restore snapshot entries: $(@($snapshot).Count)"
    if (-not $DryRun) {
        Restore-RegistrySnapshot -Snapshot $snapshot
    }
}
elseif (@($changes | Where-Object { $_.WouldChange }).Count -gt 0) {
    if ($cloud.HasRisk -and -not $ForceCloudSyncRisk -and -not $DryRun) {
        throw 'Cloud-sync risk detected. Re-run with -ForceCloudSyncRisk after reviewing the dry-run report.'
    }
    $result.Actions += "Planned registry changes: $(@($changes | Where-Object { $_.WouldChange }).Count)"
    if (-not $DryRun) {
        $snapshot = @(New-TouchedKeySnapshot -Settings $settings)
        $snapshotDir = Split-Path -Parent $SnapshotPath
        if ($snapshotDir -and -not (Test-Path -LiteralPath $snapshotDir)) {
            New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
        }
        $snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SnapshotPath -Encoding UTF8
        foreach ($setting in $settings) {
            Apply-RegistrySetting -Setting $setting
        }
    }
}
else {
    $result.Actions += 'No registry changes needed.'
}

if ($OutputJson -and -not $DryRun) {
    $outputDir = Split-Path -Parent $OutputJson
    if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputJson -Encoding UTF8
}

[pscustomobject]$result
