<#
.SYNOPSIS
Registers a Process Lasso governor watchdog scheduled task.

.DESCRIPTION
Creates or updates delayed logon and recurring scheduled triggers that run
Ensure-ProcessLassoGovernor.ps1. The watchdog checks whether
ProcessGovernor.exe is running, starts it when missing, and writes loud
Application event-log entries for remediation/failure cases.
Use named parameters for registration; the standalone --help token is supported.

.PARAMETER TaskName
Scheduled task name.

.PARAMETER ScriptPath
Path to Ensure-ProcessLassoGovernor.ps1.

.PARAMETER StartupDelaySeconds
Delay on the logon trigger and initial offset for the independent periodic
schedule. Periodic checks can run earlier than this delay after future logons;
this is not a minimum grace period for every session.

.PARAMETER RepetitionIntervalMinutes
Interval between watchdog checks while the interactive user is logged on.
The default is five minutes; repetition has no end duration.

.PARAMETER ReportPath
Optional watchdog JSON report path. Empty uses the portable repo Reports path.

.PARAMETER RunNow
Start the task immediately after registration.

.PARAMETER Disable
Register the task and then leave it disabled.

.PARAMETER Unregister
Remove the scheduled task.

.PARAMETER DryRun
Preview registration without mutating Task Scheduler or event-log sources. The
long CLI form --DryRun is also accepted.

.PARAMETER Help
Print script help and exit. The aliases -h and --help are also accepted.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$TaskName = 'PC-AI Process Lasso Governor Watchdog',
    [string]$ScriptPath = (Join-Path $PSScriptRoot 'Ensure-ProcessLassoGovernor.ps1'),
    [int]$StartupDelaySeconds = 180,
    [ValidateRange(1, 1440)]
    [int]$RepetitionIntervalMinutes = 5,
    [string]$ReportPath = '',
    [int]$ExecutionTimeLimitMinutes = 5,
    [int]$RestartCount = 3,
    [int]$RestartIntervalMinutes = 1,
    [switch]$RunNow,
    [switch]$Disable,
    [switch]$Unregister,
    [switch]$DryRun,
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

function ConvertTo-Iso8601Duration {
    param([timespan]$Duration)
    if ($Duration.TotalSeconds -lt 1) { return 'PT0S' }
    $parts = 'PT'
    if ($Duration.Hours -gt 0) { $parts += "$($Duration.Hours)H" }
    if ($Duration.Minutes -gt 0) { $parts += "$($Duration.Minutes)M" }
    if ($Duration.Seconds -gt 0) { $parts += "$($Duration.Seconds)S" }
    $parts
}

function Write-Step {
    param([string]$Message)
    Write-Host "[processlasso-watchdog] $Message"
}

if ($Unregister) {
    if ($DryRun) {
        Write-Step "would unregister scheduled task: $TaskName"
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Step "scheduled task removed: $TaskName"
    return
}

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Watchdog script not found: $ScriptPath"
}

$pwsh = Join-Path $PSHOME 'pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh)) {
    $pwsh = 'pwsh.exe'
}

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $ReportPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'Reports\processlasso-governor-watchdog.json'
}
$arguments = @(
    '-NoLogo',
    '-NoProfile',
    '-WindowStyle', 'Hidden',
    '-ExecutionPolicy', 'Bypass',
    '-File', "`"$ScriptPath`"",
    '-ReportPath', "`"$reportPath`""
) -join ' '

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
$logonTrigger.Delay = ConvertTo-Iso8601Duration -Duration (New-TimeSpan -Seconds $StartupDelaySeconds)
$periodicStart = (Get-Date).AddSeconds($StartupDelaySeconds)
$periodicTrigger = New-ScheduledTaskTrigger -Once -At $periodicStart -RepetitionInterval (New-TimeSpan -Minutes $RepetitionIntervalMinutes)
$triggers = @($logonTrigger, $periodicTrigger)
$action = New-ScheduledTaskAction -Execute $pwsh -Argument $arguments -WorkingDirectory (Split-Path -Parent $ScriptPath)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount $RestartCount `
    -RestartInterval (New-TimeSpan -Minutes $RestartIntervalMinutes) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes $ExecutionTimeLimitMinutes)

if ($DryRun) {
    Write-Step "would register scheduled task: $TaskName"
    Write-Host "     trigger: at logon, delayed $StartupDelaySeconds seconds"
    Write-Host "     trigger: every $RepetitionIntervalMinutes minutes starting $($periodicStart.ToString('o')), indefinitely"
    Write-Host "     action: $pwsh $arguments"
    Write-Host "     disabled after registration: $Disable"
    return
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $triggers `
    -Principal $principal `
    -Settings $settings `
    -Description 'Ensures ProcessGovernor.exe is running after logon and every five minutes by default; logs remediation events.' `
    -Force | Out-Null

if ($Disable) {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
}

Write-Step "scheduled task registered: $TaskName"
if ($Disable) {
    Write-Step "scheduled task disabled: $TaskName"
}

if ($RunNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Step "scheduled task started: $TaskName"
}
