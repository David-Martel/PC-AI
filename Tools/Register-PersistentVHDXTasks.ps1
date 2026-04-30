#Requires -Version 5.1
<#
.SYNOPSIS
Generates or registers PC-AI persistent VHDX startup tasks.

.DESCRIPTION
Creates maintained Task Scheduler definitions for the boot-time VHDX mounts.
By default the script previews the task plans. Use -Register to create or
update the tasks on the local machine.

.PARAMETER Register
Actually register or update scheduled tasks and the event-log source.

.PARAMETER DryRun
Preview task definitions even if -Register is supplied. The long CLI form
`--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.PARAMETER ScriptPath
Path to Tools\Mount-PersistentVHDX.ps1.

.EXAMPLE
pwsh -File .\Tools\Register-PersistentVHDXTasks.ps1

.EXAMPLE
pwsh -File .\Tools\Register-PersistentVHDXTasks.ps1 -Register
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Register,
    [string]$ScriptPath = (Join-Path $PSScriptRoot 'Mount-PersistentVHDX.ps1'),
    [string]$LogRoot = (Join-Path $PSScriptRoot '..\Logs\VHDMount'),
    [string]$TaskPath = '\',
    [string]$PowerShellExe = 'pwsh.exe',
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

$script:PersistentVHDXEventSource = 'PC-AI-VHDMount'
$script:PersistentVHDXEventLogName = 'Application'

function Get-PersistentVHDXTaskConfig {
    @(
        [pscustomobject]@{
            TaskName = 'AutoMount_VHDX_cloud-cache-disk'
            VhdPath = 'T:\vm\cloud-cache-disk.vhdx'
            ExpectedVolumeLabel = 'cloud-cache-disk'
            ExpectedDriveLetter = 'F'
            ExpectedFileSystem = 'NTFS'
            ExpectedDiskUniqueId = $null
            ExpectedState = 'Volume'
            StartupDelaySeconds = 30
            Description = 'Mount and validate cloud-cache-disk.vhdx through Tools\Mount-PersistentVHDX.ps1.'
        }
        [pscustomobject]@{
            TaskName = 'AutoMount_VHDX_shared-dev'
            VhdPath = 'T:\vm\shared-dev.vhdx'
            ExpectedVolumeLabel = 'WSL-Shared-Dev'
            ExpectedDriveLetter = 'W'
            ExpectedFileSystem = 'NTFS'
            ExpectedDiskUniqueId = $null
            ExpectedState = 'Volume'
            StartupDelaySeconds = 60
            Description = 'Mount and validate shared-dev.vhdx through Tools\Mount-PersistentVHDX.ps1.'
        }
        [pscustomobject]@{
            TaskName = 'AutoMount_VHDX_share-ext4'
            VhdPath = 'T:\vm\share-ext4.vhdx'
            ExpectedVolumeLabel = $null
            ExpectedDriveLetter = $null
            ExpectedFileSystem = $null
            ExpectedDiskUniqueId = $null
            ExpectedState = 'AttachedDiskOnly'
            StartupDelaySeconds = 90
            Description = 'Mount share-ext4.vhdx as an attached ext4/WSL disk without requiring a Windows volume.'
        }
    )
}

function ConvertTo-Iso8601DurationSeconds {
    param([int]$Seconds)

    'PT{0}S' -f ([Math]::Max(0, $Seconds))
}

function ConvertTo-TaskArgumentToken {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    '"' + ($Value -replace '"', '\"') + '"'
}

function New-PersistentVHDXTaskArgument {
    param(
        [pscustomobject]$Config,
        [string]$ScriptPath,
        [string]$LogRoot
    )

    $args = @(
        '-NoLogo'
        '-NoProfile'
        '-ExecutionPolicy'
        'Bypass'
        '-File'
        (ConvertTo-TaskArgumentToken -Value $ScriptPath)
        '-VhdPath'
        (ConvertTo-TaskArgumentToken -Value $Config.VhdPath)
        '-TaskName'
        (ConvertTo-TaskArgumentToken -Value $Config.TaskName)
        '-ExpectedState'
        $Config.ExpectedState
        '-StartupDelaySeconds'
        ([string]$Config.StartupDelaySeconds)
        '-LogRoot'
        (ConvertTo-TaskArgumentToken -Value $LogRoot)
    )

    if (-not [string]::IsNullOrWhiteSpace($Config.ExpectedVolumeLabel)) {
        $args += @('-ExpectedVolumeLabel', (ConvertTo-TaskArgumentToken -Value $Config.ExpectedVolumeLabel))
    }
    if (-not [string]::IsNullOrWhiteSpace($Config.ExpectedDriveLetter)) {
        $args += @('-ExpectedDriveLetter', $Config.ExpectedDriveLetter)
    }
    if (-not [string]::IsNullOrWhiteSpace($Config.ExpectedFileSystem)) {
        $args += @('-ExpectedFileSystem', $Config.ExpectedFileSystem)
    }
    if (-not [string]::IsNullOrWhiteSpace($Config.ExpectedDiskUniqueId)) {
        $args += @('-ExpectedDiskUniqueId', (ConvertTo-TaskArgumentToken -Value $Config.ExpectedDiskUniqueId))
    }

    $args -join ' '
}

function Register-PersistentVHDXTaskEventSource {
    param([switch]$Skip)

    if ($Skip) {
        return
    }

    try {
        if (-not [System.Diagnostics.EventLog]::SourceExists($script:PersistentVHDXEventSource)) {
            New-EventLog -LogName $script:PersistentVHDXEventLogName -Source $script:PersistentVHDXEventSource -ErrorAction Stop
        }
    } catch {
        Write-Warning ("Unable to register event source {0}: {1}" -f $script:PersistentVHDXEventSource, $_.Exception.Message)
    }
}

function New-PersistentVHDXTaskPlan {
    [CmdletBinding()]
    param(
        [string]$ScriptPath,
        [string]$LogRoot,
        [string]$TaskPath = '\',
        [string]$PowerShellExe = 'pwsh.exe'
    )

    foreach ($config in Get-PersistentVHDXTaskConfig) {
        $argument = New-PersistentVHDXTaskArgument -Config $config -ScriptPath $ScriptPath -LogRoot $LogRoot
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $trigger.Delay = ConvertTo-Iso8601DurationSeconds -Seconds $config.StartupDelaySeconds
        $settings = New-ScheduledTaskSettingsSet `
            -MultipleInstances IgnoreNew `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
            -RestartCount 3 `
            -RestartInterval (New-TimeSpan -Minutes 1) `
            -StartWhenAvailable `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries
        $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest
        $action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $argument -WorkingDirectory (Split-Path -Parent $ScriptPath)
        [pscustomobject]@{
            TaskName = $config.TaskName
            TaskPath = $TaskPath
            VhdPath = $config.VhdPath
            ExpectedState = $config.ExpectedState
            StartupDelaySeconds = $config.StartupDelaySeconds
            Delay = $trigger.Delay
            RestartCount = 3
            RestartInterval = 'PT1M'
            ExecutionTimeLimit = 'PT10M'
            MultipleInstances = 'IgnoreNew'
            Description = $config.Description
            Execute = $PowerShellExe
            Argument = $argument
            Action = $action
            Trigger = $trigger
            Settings = $settings
            Principal = $principal
            Definition = $null
        }
    }
}

function New-PersistentVHDXScheduledTaskDefinition {
    param(
        [object]$Action,
        [object]$Trigger,
        [object]$Settings,
        [object]$Principal,
        [string]$Description
    )

    New-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description $Description
}

function Register-PersistentVHDXScheduledTask {
    param(
        [string]$TaskName,
        [string]$TaskPath,
        [object]$Definition
    )

    Register-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -InputObject $Definition -Force
}

function Invoke-PersistentVHDXTaskRegistration {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [switch]$Register,
        [string]$ScriptPath,
        [string]$LogRoot,
        [string]$TaskPath = '\',
        [string]$PowerShellExe = 'pwsh.exe',
        [switch]$SkipEventSourceRegistration,
        [switch]$DryRun
    )

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "Mount wrapper script not found: $ScriptPath"
    }

    $plans = @(New-PersistentVHDXTaskPlan -ScriptPath $ScriptPath -LogRoot $LogRoot -TaskPath $TaskPath -PowerShellExe $PowerShellExe)

    if ($Register -and -not $DryRun) {
        Register-PersistentVHDXTaskEventSource -Skip:$SkipEventSourceRegistration
        foreach ($plan in $plans) {
            $target = Join-Path $TaskPath $plan.TaskName
            if ($PSCmdlet.ShouldProcess($target, 'Register or update persistent VHDX startup task')) {
                $definition = New-PersistentVHDXScheduledTaskDefinition -Action $plan.Action -Trigger $plan.Trigger -Settings $plan.Settings -Principal $plan.Principal -Description $plan.Description
                Register-PersistentVHDXScheduledTask -TaskName $plan.TaskName -TaskPath $TaskPath -Definition $definition | Out-Null
            }
        }
    }

    $plans
}

if ($MyInvocation.InvocationName -ne '.') {
    if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
        $ScriptPath = Join-Path $PSScriptRoot 'Mount-PersistentVHDX.ps1'
    }
    if ([string]::IsNullOrWhiteSpace($LogRoot)) {
        $LogRoot = Join-Path $PSScriptRoot '..\Logs\VHDMount'
    }
    $registrationParameters = @{} + $PSBoundParameters
    $registrationParameters.Remove('PassThru')
    $registrationParameters.Remove('Help')
    $registrationParameters.Remove('CliArgs')
    $registrationParameters['ScriptPath'] = $ScriptPath
    $registrationParameters['LogRoot'] = $LogRoot
    $plans = Invoke-PersistentVHDXTaskRegistration @registrationParameters
    if ($PassThru -or -not $Register) {
        $plans | Select-Object TaskName, VhdPath, ExpectedState, Delay, RestartCount, RestartInterval, ExecutionTimeLimit, MultipleInstances, Execute, Argument
    }
}
