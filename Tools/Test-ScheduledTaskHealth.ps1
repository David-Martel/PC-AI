#Requires -Version 7.0
<#
.SYNOPSIS
Reports which Scheduled Tasks are failing, stalled, or silently not running.

.DESCRIPTION
Windows Task Scheduler has no health surface. A task can fail on every run, or
stop running entirely, and nothing surfaces it -- on this machine AgentHubRunner
sat terminated for five weeks and DTMVHeadscalePullMonitor accumulated over
twenty thousand missed runs, both undetected.

This script classifies every non-Microsoft task on two independent axes:

  Result    the action's last exit code, split into success, the documented
            SCHED_S_* informational codes, and everything else (a failure).
  Staleness whether the task has actually run when its own triggers say it
            should have. A boot- or logon-triggered task that has not run since
            the last boot is stalled even if its last recorded result was 0 --
            this is the axis that catches a task which quietly stopped firing.

Both matter. A task can be green on result and still be dead.

.PARAMETER IncludeMicrosoft
Also report tasks under \Microsoft\. Off by default -- the OS ships hundreds of
them and their failures are not actionable here.

.PARAMETER LocalOnly
Only report tasks whose action runs something from a path we own (C:\codedev,
C:\actions-runner, ~\bin, ~\.claude, T:\projects). Vendor tasks -- Lenovo, NI,
OneDrive, npcap -- fail constantly and drown the signal. Ownership is derived
from the action itself, not a name list, so it does not rot. This is the form a
watchdog should gate on.

.PARAMETER TaskPathFilter
Only report tasks whose full path matches this wildcard, e.g. '\Lenovo\*'.

.PARAMETER StaleFactor
How many times its own interval a repeating task may overrun before being called
stalled. Default 3, so an hourly task is stale after three hours. Tolerates a
missed run or a laptop that was asleep without crying wolf.

.PARAMETER Expected
Task names that are deliberately disabled or dormant. These are reported as
'Ignored' and never affect the exit code. Supply your own list to override the
built-in defaults.

.PARAMETER PassThru
Emit the per-task result objects to the pipeline.

.PARAMETER FailOnIssue
Exit 1 when any task is Failed or Stalled. Use this to gate CI or a watchdog.

.PARAMETER OutputJson
Write the machine-readable report to this path. Not written when DryRun is set.

.PARAMETER DryRun
Run every check but write no report file. The long form `--DryRun` also works.

.PARAMETER Help
Print this help and exit. The aliases `-h`, `-?` and `--help` also work.

.EXAMPLE
./Test-ScheduledTaskHealth.ps1
Print the health table for every non-Microsoft task.

.EXAMPLE
./Test-ScheduledTaskHealth.ps1 -FailOnIssue -OutputJson Reports\scheduled-task-health.json
Write a report and exit non-zero if anything is broken -- the watchdog form.

.NOTES
Read-only. This script inspects Task Scheduler and never registers, modifies,
starts or stops a task.
#>
[CmdletBinding()]
param(
    [switch]$IncludeMicrosoft,
    [switch]$LocalOnly,
    [string]$TaskPathFilter,
    [ValidateRange(1, 100)]
    [int]$StaleFactor = 3,
    [string[]]$Expected,
    [switch]$PassThru,
    [switch]$FailOnIssue,
    [string]$OutputJson,
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

# Tasks that are dormant on purpose. Reported as Ignored so they cannot mask a
# real regression, but never counted against the exit code. Documented here
# rather than silently skipped, so the list stays reviewable.
$DefaultExpectedDormant = @(
    'DevEnvironmentStartup'        # superseded; last ran 2026-01-25 exit 0
    'UnifiUdmDriveStackStartup'    # UDM drive stack retired
    'DTMVHeadscalePullMonitor'     # headscale pull moved off-box
)
if (-not $PSBoundParameters.ContainsKey('Expected')) { $Expected = $DefaultExpectedDormant }
$Expected = @($Expected)

# Result codes are keyed by normalised hex STRING, not by number. LastTaskResult
# is a CIM uint32, so 0xC000013A arrives as 3,221,225,786 -- larger than
# Int32.MaxValue. Casting it to [int] throws, and mixing [int] and [int64] keys
# in a hashtable makes ContainsKey miss silently, which would have quietly
# downgraded exactly the killed-process codes this script exists to catch.
$InformationalCodes = @{
    '0x41300'    = 'Ready (not yet run)'
    '0x41301'    = 'Currently running'
    '0x41302'    = 'Disabled'
    '0x41303'    = 'Has not run'
    '0x41304'    = 'No more runs scheduled'
    '0x41305'    = 'Not scheduled'
    '0x41325'    = 'Queued'
}
# Codes worth naming in the report rather than printing bare hex.
$KnownFailureCodes = @{
    '0x1'        = 'Incorrect function / generic script failure'
    '0x2'        = 'File not found'
    '0xA'        = 'Environment is incorrect'
    '0x32'       = 'Not supported'
    '0x33'       = 'Resource not available (used here as a script exit code)'
    '0x41306'    = 'Terminated before completion'
    '0x8004131F' = 'An instance is already running'
    '0x800710E0' = 'Operator or administrator refused the request'
    '0x80070420' = 'An instance is already running'
    '0xC000013A' = 'Terminated by Ctrl+C / killed'
}
# Terminated is "informational" to Task Scheduler but is a real failure for a
# long-running task: something killed it.
$TerminationCodes = @('0x41306', '0xC000013A')

# A task is "Local" when its action runs something from a path we own. Derived
# from the action, not from a hand-maintained name list, so it cannot rot as
# tasks are added or renamed. Vendor tasks (Lenovo, NI, OneDrive, npcap...) fail
# constantly and are not actionable here; -LocalOnly is what a watchdog gates on.
$LocalActionRoots = @(
    'C:\codedev'
    'C:\actions-runner'
    "$env:USERPROFILE\bin"
    "$env:USERPROFILE\.local\bin"
    "$env:USERPROFILE\.claude"
    'T:\projects'
)

function Get-TaskOwner {
    <#
    Classify a task as Local or Vendor by looking at what its actions execute --
    both the executable and its arguments, since our tasks are overwhelmingly
    `pwsh.exe -File <a path we own>`, where the executable alone says nothing.
    #>
    param($Task)

    foreach ($action in @($Task.Actions)) {
        if ($null -eq $action) { continue }
        $haystack = ''
        if ($action.PSObject.Properties.Name -contains 'Execute' -and $action.Execute) {
            $haystack += ' ' + $action.Execute
        }
        if ($action.PSObject.Properties.Name -contains 'Arguments' -and $action.Arguments) {
            $haystack += ' ' + $action.Arguments
        }
        foreach ($root in $LocalActionRoots) {
            if ([string]::IsNullOrWhiteSpace($root)) { continue }
            if ($haystack -like "*$root*") { return 'Local' }
        }
    }
    return 'Vendor'
}

function Get-TriggerExpectation {
    <#
    Derive, from a task's own triggers, how long it may legitimately go without
    running. Returns a hashtable with Kind and, where one applies, Interval.
    Returns Kind = 'Unknown' when no cadence can be inferred, so an unschedulable
    task is never reported as stalled on a guess.
    #>
    param($Task)

    $bestInterval = $null
    $sawBootOrLogon = $false
    $sawOnDemandOnly = $true

    foreach ($trigger in @($Task.Triggers)) {
        if ($null -eq $trigger) { continue }
        $className = $trigger.CimClass.CimClassName
        $sawOnDemandOnly = $false

        # A repetition interval, when present, is the tightest real cadence and
        # overrides the trigger's own kind.
        $repetition = $null
        if ($trigger.PSObject.Properties.Name -contains 'Repetition') { $repetition = $trigger.Repetition }
        if ($null -ne $repetition -and
            $repetition.PSObject.Properties.Name -contains 'Interval' -and
            -not [string]::IsNullOrWhiteSpace($repetition.Interval)) {
            try {
                $span = [System.Xml.XmlConvert]::ToTimeSpan($repetition.Interval)
                if ($span -gt [TimeSpan]::Zero -and ($null -eq $bestInterval -or $span -lt $bestInterval)) {
                    $bestInterval = $span
                }
                continue
            } catch {
                # An unparseable interval is not a reason to fail the whole run.
                Write-Verbose "Unparseable repetition interval '$($repetition.Interval)' on $($Task.TaskName)"
            }
        }

        switch -Wildcard ($className) {
            'MSFT_TaskBootTrigger'  { $sawBootOrLogon = $true }
            'MSFT_TaskLogonTrigger' { $sawBootOrLogon = $true }
            'MSFT_TaskDailyTrigger' {
                $span = [TimeSpan]::FromDays(1)
                if ($null -eq $bestInterval -or $span -lt $bestInterval) { $bestInterval = $span }
            }
            'MSFT_TaskWeeklyTrigger' {
                $span = [TimeSpan]::FromDays(7)
                if ($null -eq $bestInterval -or $span -lt $bestInterval) { $bestInterval = $span }
            }
            default { }
        }
    }

    if ($null -ne $bestInterval) { return @{ Kind = 'Interval'; Interval = $bestInterval } }
    if ($sawBootOrLogon) { return @{ Kind = 'BootOrLogon'; Interval = $null } }
    if ($sawOnDemandOnly) { return @{ Kind = 'OnDemand'; Interval = $null } }
    return @{ Kind = 'Unknown'; Interval = $null }
}

$bootTime = $null
try {
    $bootTime = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
} catch {
    Write-Warning "Could not read LastBootUpTime - boot/logon staleness will not be evaluated."
}

$now = Get-Date
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue)
if (-not $IncludeMicrosoft) {
    $tasks = @($tasks | Where-Object { $_.TaskPath -notlike '\Microsoft\*' })
}
if (-not [string]::IsNullOrWhiteSpace($TaskPathFilter)) {
    $tasks = @($tasks | Where-Object { "$($_.TaskPath)$($_.TaskName)" -like $TaskPathFilter })
}

$results = foreach ($task in $tasks) {
    $info = $null
    try { $info = $task | Get-ScheduledTaskInfo -ErrorAction Stop } catch { }

    $lastRun = $null
    $nextRun = $null
    $lastResult = $null
    $missed = 0
    if ($null -ne $info) {
        if ($info.NextRunTime -and $info.NextRunTime.Year -gt 1999) { $nextRun = $info.NextRunTime }
        # Task Scheduler reports "never run" as a sentinel date near 1899/1601.
        if ($info.LastRunTime -and $info.LastRunTime.Year -gt 1999) { $lastRun = $info.LastRunTime }
        # [int64], never [int]: LastTaskResult is a uint32 and 0xC000013A
        # (3,221,225,786) overflows Int32.
        if ($null -ne $info.LastTaskResult) { $lastResult = [int64]$info.LastTaskResult }
        if ($null -ne $info.NumberOfMissedRuns) { $missed = [int]$info.NumberOfMissedRuns }
    }

    $expectation = Get-TriggerExpectation -Task $task
    $owner = Get-TaskOwner -Task $task
    $status = 'Healthy'
    $reasons = [System.Collections.Generic.List[string]]::new()

    # --- Axis 1: the last result code -------------------------------------
    $resultText = 'No result recorded'
    if ($null -ne $lastResult) {
        $hex = '0x{0:X}' -f $lastResult
        if ($lastResult -eq 0) {
            $resultText = 'Success'
        } elseif ($hex -eq '0x41301') {
            $resultText = $InformationalCodes[$hex]
        } elseif ($TerminationCodes -contains $hex) {
            $resultText = "$hex $($KnownFailureCodes[$hex])"
            $status = 'Failed'
            $reasons.Add("Last run $resultText")
        } elseif ($InformationalCodes.ContainsKey($hex)) {
            $resultText = "$hex $($InformationalCodes[$hex])"
        } else {
            $named = if ($KnownFailureCodes.ContainsKey($hex)) { $KnownFailureCodes[$hex] } else { 'Action returned a nonzero exit code' }
            $resultText = "$hex $named"
            $status = 'Failed'
            $reasons.Add("Last run $resultText")
        }
    }

    # --- Axis 2: has it actually been running? -----------------------------
    # Evaluated independently. A task can report result 0 and still be stalled.
    $staleness = 'n/a'
    if ($task.State -ne 'Disabled') {
        switch ($expectation.Kind) {
            'Interval' {
                $budget = $expectation.Interval * $StaleFactor
                if ($null -eq $lastRun) {
                    if ($null -ne $lastResult -and $lastResult -eq 0x41303) {
                        $staleness = 'never run'
                    } else {
                        $staleness = 'never run'
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add("Has a $([math]::Round($expectation.Interval.TotalMinutes)) min cadence but has never run")
                    }
                } else {
                    $age = $now - $lastRun
                    $staleness = "{0:N1} h since last run" -f $age.TotalHours
                    if ($age -gt $budget) {
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add(("Last ran {0:N1} h ago but its cadence is {1:N1} h (allowed {2}x = {3:N1} h)" -f `
                            $age.TotalHours, $expectation.Interval.TotalHours, $StaleFactor, $budget.TotalHours))
                        # The cause is almost always this: a repetition pattern
                        # only arms when its trigger FIRES. A task registered
                        # with boot/logon triggers after the current boot has
                        # armed nothing and will not repeat until the next one.
                        if ($null -eq $nextRun) {
                            $reasons.Add('No next run is scheduled - its repetition is not armed, so it will not run again until a trigger fires (e.g. the next boot or logon)')
                        }
                    }
                }
            }
            'BootOrLogon' {
                if ($null -ne $bootTime) {
                    if ($null -eq $lastRun) {
                        $staleness = 'never run'
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add('Triggers on boot or logon but has never run')
                    } elseif ($lastRun -lt $bootTime) {
                        $staleness = 'predates last boot'
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add(("Triggers on boot or logon but last ran {0:yyyy-MM-dd HH:mm}, before the current boot at {1:yyyy-MM-dd HH:mm}" -f `
                            $lastRun, $bootTime))
                    } else {
                        $staleness = 'ran this boot'
                    }
                }
            }
            default { $staleness = 'no inferable cadence' }
        }
    }

    if ($missed -gt 0) { $reasons.Add("$missed missed run(s) recorded") }

    if ($task.State -eq 'Disabled') { $status = 'Disabled' }
    if ($Expected -contains $task.TaskName) { $status = 'Ignored' }

    [pscustomobject]@{
        TaskPath    = $task.TaskPath
        TaskName    = $task.TaskName
        Owner       = $owner
        State       = [string]$task.State
        Status      = $status
        LastRunTime = $lastRun
        NextRunTime = $nextRun
        LastResult  = $lastResult
        ResultText  = $resultText
        Cadence     = $expectation.Kind
        Staleness   = $staleness
        MissedRuns  = $missed
        Reasons     = @($reasons)
    }
}

$results = @($results)
$vendorSuppressed = 0
if ($LocalOnly) {
    $vendorSuppressed = @($results | Where-Object { $_.Owner -ne 'Local' }).Count
    $results = @($results | Where-Object { $_.Owner -eq 'Local' })
}
$failed = @($results | Where-Object { $_.Status -eq 'Failed' })
$stalled = @($results | Where-Object { $_.Status -eq 'Stalled' })
$issues = @($failed) + @($stalled)

$summary = [pscustomobject]@{
    GeneratedAt  = $now.ToString('o')
    Computer     = $env:COMPUTERNAME
    LastBootUp   = if ($null -ne $bootTime) { $bootTime.ToString('o') } else { $null }
    StaleFactor  = $StaleFactor
    LocalOnly    = [bool]$LocalOnly
    VendorSuppressed = $vendorSuppressed
    TotalTasks   = $results.Count
    Healthy      = @($results | Where-Object { $_.Status -eq 'Healthy' }).Count
    Failed       = $failed.Count
    Stalled      = $stalled.Count
    Disabled     = @($results | Where-Object { $_.Status -eq 'Disabled' }).Count
    Ignored      = @($results | Where-Object { $_.Status -eq 'Ignored' }).Count
    Tasks        = $results
}

if ($issues.Count -gt 0) {
    Write-Host ''
    Write-Host "Scheduled task issues on $($env:COMPUTERNAME):" -ForegroundColor Yellow
    foreach ($r in ($issues | Sort-Object Status, TaskName)) {
        $colour = if ($r.Status -eq 'Failed') { 'Red' } else { 'Yellow' }
        Write-Host ("  [{0}] {1}{2}" -f $r.Status.ToUpper(), $r.TaskPath, $r.TaskName) -ForegroundColor $colour
        foreach ($reason in $r.Reasons) { Write-Host "      - $reason" }
    }
} else {
    Write-Host "All $($results.Count) task(s) healthy." -ForegroundColor Green
}

Write-Host ''
Write-Host ("Total {0} | healthy {1} | failed {2} | stalled {3} | disabled {4} | ignored {5}" -f `
    $summary.TotalTasks, $summary.Healthy, $summary.Failed, $summary.Stalled, $summary.Disabled, $summary.Ignored)

if (-not [string]::IsNullOrWhiteSpace($OutputJson) -and -not $DryRun) {
    $jsonDir = Split-Path -Parent $OutputJson
    if (-not [string]::IsNullOrWhiteSpace($jsonDir) -and -not (Test-Path -LiteralPath $jsonDir)) {
        New-Item -ItemType Directory -Path $jsonDir -Force | Out-Null
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputJson -Encoding utf8
    Write-Host "Report written to $OutputJson"
}

if ($PassThru) { $results }

if ($FailOnIssue -and $issues.Count -gt 0) { exit 1 }
exit 0
