<#
.SYNOPSIS
Ensures the Process Lasso governor is running.

.DESCRIPTION
Checks for ProcessGovernor.exe and starts it when missing. The script writes a
structured result object, optional JSON report, and Windows Application events
when a restart or failure occurs. Dry-run mode performs all checks without
starting processes or writing event-log/report output.

.PARAMETER GovernorPath
Path to ProcessGovernor.exe.

.PARAMETER EventSource
Application event-log source used for warnings and failures.

.PARAMETER ReportPath
Optional JSON report path. Not written in dry-run mode.

.PARAMETER DryRun
Preview behavior without starting ProcessGovernor.exe or writing report/event
output. The long CLI form --DryRun is also accepted.

.PARAMETER PassThru
Return the result object.

.PARAMETER Help
Print script help and exit. The aliases -h and --help are also accepted.
#>
[CmdletBinding()]
param(
    [string]$GovernorPath = 'C:\Program Files\Process Lasso\ProcessGovernor.exe',
    [string]$EventSource = 'PC-AI-ProcessLasso',
    [string]$ReportPath = '',
    [switch]$DryRun,
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

function Ensure-EventSource {
    param([string]$Source)
    if ($DryRun) {
        return $false
    }
    try {
        if (-not [System.Diagnostics.EventLog]::SourceExists($Source)) {
            New-EventLog -LogName Application -Source $Source -ErrorAction Stop
        }
        return $true
    }
    catch {
        return $false
    }
}

function Write-GovernorEvent {
    param(
        [string]$Message,
        [System.Diagnostics.EventLogEntryType]$EntryType,
        [int]$EventId
    )
    if ($script:EventSourceReady) {
        try {
            Write-EventLog -LogName Application -Source $EventSource -EntryType $EntryType -EventId $EventId -Message $Message -ErrorAction Stop
        }
        catch {
        }
    }
}

$startedAt = Get-Date
$before = Get-Process -Name ProcessGovernor -ErrorAction SilentlyContinue | Select-Object -First 1
$actions = [System.Collections.Generic.List[string]]::new()
$failures = [System.Collections.Generic.List[string]]::new()
$script:EventSourceReady = Ensure-EventSource -Source $EventSource

if (-not (Test-Path -LiteralPath $GovernorPath)) {
    [void]$failures.Add("ProcessGovernor.exe not found: $GovernorPath")
}
elseif (-not $before) {
    [void]$actions.Add('ProcessGovernor.exe was not running.')
    if ($DryRun) {
        [void]$actions.Add("Would start: $GovernorPath")
    }
    else {
        Write-GovernorEvent -EntryType Warning -EventId 4701 -Message "ProcessGovernor.exe was not running; starting $GovernorPath."
        try {
            Start-Process -FilePath $GovernorPath -WindowStyle Hidden -ErrorAction Stop
            Start-Sleep -Seconds 2
            [void]$actions.Add("Started: $GovernorPath")
        }
        catch {
            [void]$failures.Add("Failed to start ProcessGovernor.exe: $($_.Exception.Message)")
        }
    }
}
else {
    [void]$actions.Add("ProcessGovernor.exe already running (PID $($before.Id)).")
}

$after = Get-Process -Name ProcessGovernor -ErrorAction SilentlyContinue | Select-Object -First 1
$status = if ($failures.Count -gt 0) { 'failed' } elseif ($after) { 'ok' } else { 'missing' }
if ($status -eq 'failed' -and -not $DryRun) {
    Write-GovernorEvent -EntryType Error -EventId 4702 -Message ($failures -join '; ')
}
elseif ($status -eq 'ok' -and -not $DryRun -and -not $before) {
    Write-GovernorEvent -EntryType Information -EventId 4703 -Message "ProcessGovernor.exe is running after watchdog remediation (PID $($after.Id))."
}

$result = [pscustomobject]@{
    Ok = ($status -eq 'ok')
    DryRun = [bool]$DryRun
    GeneratedAt = (Get-Date).ToString('o')
    StartedAt = $startedAt.ToString('o')
    GovernorPath = $GovernorPath
    EventSource = $EventSource
    EventSourceReady = [bool]$script:EventSourceReady
    Status = $status
    Before = if ($before) { [pscustomobject]@{ Id = $before.Id; StartTime = $before.StartTime; Responding = $before.Responding; Path = $before.Path } } else { $null }
    After = if ($after) { [pscustomobject]@{ Id = $after.Id; StartTime = $after.StartTime; Responding = $after.Responding; Path = $after.Path } } else { $null }
    Actions = @($actions)
    Failures = @($failures)
}

if ($ReportPath -and -not $DryRun) {
    $reportDirectory = Split-Path -Parent $ReportPath
    if ($reportDirectory -and -not (Test-Path -LiteralPath $reportDirectory)) {
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
    }
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}

if ($PassThru -or $DryRun) {
    $result
}
else {
    if ($result.Ok) {
        Write-Host "Process Lasso governor status: $($result.Status)"
    }
    else {
        Write-Error "Process Lasso governor status: $($result.Status)"
    }
}

if (-not $result.Ok) {
    exit 1
}
