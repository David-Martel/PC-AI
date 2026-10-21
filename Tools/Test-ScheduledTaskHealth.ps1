#Requires -Version 7.0
<#
.SYNOPSIS
Reports which Scheduled Tasks are failing, stalled, or silently not running.

.DESCRIPTION
Windows Task Scheduler has no health surface. A task can fail on every run, or
stop running entirely, and nothing surfaces it -- on this machine AgentHubRunner
sat terminated for five weeks and DTMVHeadscalePullMonitor accumulated over
twenty thousand missed runs, both undetected.

This script classifies every non-Microsoft task on three independent axes:

  Result    the action's last exit code, split into success, the documented
            SCHED_S_* informational codes, and everything else (a failure).
  Staleness whether the task has actually run when its own triggers say it
            should have. A boot- or logon-triggered task that has not run since
            the last boot is stalled even if its last recorded result was 0 --
            this is the axis that catches a task which quietly stopped firing.
  Launcher  for tasks started through dtm-task-run.exe, the status JSON the
            launcher writes (docs/scheduled-tasks/LAUNCHER.md): a nonzero or
            timed-out last run, a run that was interrupted, and a status file
            that is stale against the task's cadence are all reported.

Both Result and Staleness matter. A task can be green on result and still be
dead. Tasks wrapped in `conhost.exe --headless` are counted and named in the
summary: that wrapper hides the window but also hides the exit code, so their
Result is always 0 and means nothing.

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

.PARAMETER StatusDir
Directory (or directories) holding the dtm-task-run status files (<task id>.json). With no
value, the current user's %LOCALAPPDATA%\dtm-tasks\status plus every
<SystemDrive>\Users\*\AppData\Local\dtm-tasks\status that exists, so the SYSTEM watchdog
(whose own %LOCALAPPDATA% is the SYSTEM profile) still finds the owner's files; the newest
file for a task wins. Separate several directories with ';' when running through `pwsh -File`
(which binds only the first value of an array parameter). Only tasks whose action is dtm-task-run.exe are checked against it; a
direct task is never flagged for lacking a status file.

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
    [string[]]$StatusDir,
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
    '0x41300' = 'Ready (not yet run)'
    '0x41301' = 'Currently running'
    '0x41302' = 'Disabled'
    '0x41303' = 'Has not run'
    '0x41304' = 'No more runs scheduled'
    '0x41305' = 'Not scheduled'
    '0x41325' = 'Queued'
}
# Codes worth naming in the report rather than printing bare hex.
$KnownFailureCodes = @{
    '0x1'        = 'Incorrect function / generic script failure'
    '0x2'        = 'File not found'
    '0xA'        = 'Environment is incorrect'
    '0x32'       = 'Not supported'
    '0x33'       = 'Resource not available (used here as a script exit code)'
    '0x41306'    = 'Terminated before completion'
    # 0x8004131F is SCHED_E_ALREADY_RUNNING - the TASK. 0x80070420 wraps Win32
    # 1056 ERROR_SERVICE_ALREADY_RUNNING - a SERVICE the action tried to start.
    # Same words, different subject and different fix, so name them apart.
    '0x8004131F' = 'This task is already running'
    '0x800710E0' = 'Operator or administrator refused the request'
    '0x80070420' = 'A service the action starts is already running'
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
    "$env:USERPROFILE\.local\share\pc-ai\deployed"
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

        # Second signal: the action runs a SCRIPT from somewhere that is not a
        # vendor install root. Vendors ship compiled executables; a loose .ps1 or
        # .py on this machine is almost always ours. This catches the tasks an
        # explicit root list misses - Update-BwArchive.ps1 under ~\.machine,
        # linklocal-guard.ps1 under C:\ProgramData\dtm-netfix, recruitment_job.py
        # under %LOCALAPPDATA%\vigil - without needing every such directory
        # enumerated in advance, which is what would rot.
        if ($haystack -match '(?i)([A-Z]:\\[^"'']*?\.(?:ps1|py|cmd|bat|sh))') {
            $scriptPath = $Matches[1]
            if ($scriptPath -notmatch '(?i)^[A-Z]:\\(?:Program Files( \(x86\))?|Windows)\\') {
                return 'Local'
            }
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
            }
            catch {
                # An unparseable interval is not a reason to fail the whole run.
                Write-Verbose "Unparseable repetition interval '$($repetition.Interval)' on $($Task.TaskName)"
            }
        }

        switch -Wildcard ($className) {
            'MSFT_TaskBootTrigger' { $sawBootOrLogon = $true }
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

function Get-LauncherInfo {
    <#
    Classify how a task is started: 'dtm-task-run' (exit code preserved, status JSON written),
    'conhost' (`conhost.exe --headless`: window hidden but exit code masked, LastTaskResult is
    always 0x0) or 'direct'. For dtm-task-run also returns the task id from `--name`; only the
    part before the first bare `--` is searched, so a child's own --name is never mistaken for it.
    #>
    param($Task)

    foreach ($action in @($Task.Actions)) {
        if ($null -eq $action) { continue }
        $names = $action.PSObject.Properties.Name
        if ($names -notcontains 'Execute' -or [string]::IsNullOrWhiteSpace($action.Execute)) { continue }
        $leaf = [IO.Path]::GetFileName(([string]$action.Execute).Trim('"'))
        $argText = if ($names -contains 'Arguments' -and $action.Arguments) { [string]$action.Arguments } else { '' }
        if ($leaf -ieq 'dtm-task-run.exe') {
            $head = ($argText -split '\s--(?:\s|$)', 2)[0]
            $id = $null
            if ($head -match '(?i)--name\s+(?:"(?<q>[^"]+)"|(?<b>\S+))') {
                $id = if ($Matches['q']) { $Matches['q'] } else { $Matches['b'] }
            }
            return @{ Kind = 'dtm-task-run'; TaskId = $id }
        }
        if ($leaf -ieq 'conhost.exe' -and $argText -match '(?i)^\s*--headless\b') {
            return @{ Kind = 'conhost'; TaskId = $null }
        }
    }
    return @{ Kind = 'direct'; TaskId = $null }
}

function ConvertTo-LauncherStem {
    <# The launcher's own rule (TaskNames.ToFileStem): invalid file-name characters and spaces become '_'. #>
    param([string]$Name)
    $invalid = [IO.Path]::GetInvalidFileNameChars()
    $chars = $Name.Trim().ToCharArray()
    for ($i = 0; $i -lt $chars.Length; $i++) {
        if ($chars[$i] -eq ' ' -or $invalid -contains $chars[$i]) { $chars[$i] = '_' }
    }
    $stem = (-join $chars).Trim('.')
    if ($stem.Length -eq 0) { '_' } else { $stem }
}

function Get-DefaultStatusDirs {
    <#
    The current user's status directory plus every other profile's that exists. The SYSTEM watchdog
    runs with its own %LOCALAPPDATA%, but the tasks (and so their status files) belong to the owner.
    #>
    $dirs = [System.Collections.Generic.List[string]]::new()
    if ($env:LOCALAPPDATA) { $dirs.Add((Join-Path $env:LOCALAPPDATA 'dtm-tasks\status')) }
    $usersRoot = Join-Path $(if ($env:SystemDrive) { $env:SystemDrive } else { 'C:' }) 'Users'
    if (Test-Path -LiteralPath $usersRoot) {
        foreach ($profileDir in Get-ChildItem -LiteralPath $usersRoot -Directory -ErrorAction SilentlyContinue) {
            $candidate = Join-Path $profileDir.FullName 'AppData\Local\dtm-tasks\status'
            if ((Test-Path -LiteralPath $candidate -PathType Container) -and $dirs -notcontains $candidate) { $dirs.Add($candidate) }
        }
    }
    @($dirs)
}

function Read-LauncherStatus {
    <# Reads <Dir>\<stem>.json from the first/newest of Dir. State is 'missing', 'unreadable' or 'ok'. #>
    param([string[]]$Dir, [string]$TaskId)
    $leaf = (ConvertTo-LauncherStem $TaskId) + '.json'
    $found = @($Dir | Where-Object { $_ } | ForEach-Object { Join-Path $_ $leaf } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Sort-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc } -Descending)
    if ($found.Count -eq 0) { return @{ State = 'missing'; Path = (Join-Path @($Dir)[0] $leaf); Data = $null } }
    $path = $found[0]
    try {
        $data = Get-Content -LiteralPath $path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        foreach ($required in 'state', 'startUtc') {
            if ($null -eq $data -or $data.PSObject.Properties.Name -notcontains $required) { throw "missing '$required'" }
        }
        return @{ State = 'ok'; Path = $path; Data = $data }
    }
    catch {
        return @{ State = 'unreadable'; Path = $path; Data = $null; Error = $_.Exception.Message }
    }
}

function ConvertFrom-IsoUtc {
    param($Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return $null }
    if ($Value -is [datetime]) { return $Value.ToUniversalTime() }
    [datetime]::Parse([string]$Value, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}

if (-not $PSBoundParameters.ContainsKey('StatusDir')) { $StatusDir = Get-DefaultStatusDirs }
# `pwsh -File` binds only the first value of an array parameter, so also accept ';'-separated paths.
$StatusDir = @($StatusDir | ForEach-Object { $_ -split ';' } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

$bootTime = $null
try {
    $bootTime = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
}
catch {
    Write-Warning "Could not read LastBootUpTime - boot/logon staleness will not be evaluated."
}

$now = Get-Date
# -ErrorAction Stop, NOT SilentlyContinue. Silencing this turns a Task Scheduler
# provider or service fault into an EMPTY set, which then reports "all healthy"
# and exits 0 under -FailOnIssue - a health check that cannot report its own
# failure, which is the exact defect this tool exists to catch. Fail loudly.
try {
    $tasks = @(Get-ScheduledTask -ErrorAction Stop)
}
catch {
    Write-Host "FATAL: could not enumerate scheduled tasks - $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '       This is NOT a clean result. Task Scheduler could not be queried.'
    exit 2
}
if ($tasks.Count -eq 0) {
    # Windows always ships tasks. Zero means the query silently returned nothing.
    Write-Host 'FATAL: Get-ScheduledTask returned no tasks at all - refusing to report health.' -ForegroundColor Red
    exit 2
}
if (-not $IncludeMicrosoft) {
    $tasks = @($tasks | Where-Object { $_.TaskPath -notlike '\Microsoft\*' })
}
if (-not [string]::IsNullOrWhiteSpace($TaskPathFilter)) {
    $tasks = @($tasks | Where-Object { "$($_.TaskPath)$($_.TaskName)" -like $TaskPathFilter })
}

$results = foreach ($task in $tasks) {
    $info = $null
    $inspectionStatus = 'Available'
    $inspectionReason = $null
    try {
        $info = $task | Get-ScheduledTaskInfo -ErrorAction Stop
    }
    catch {
        $inspectionStatus = 'Unavailable'
        $inspectionReason = "Task inspection unavailable: Get-ScheduledTaskInfo failed: $($_.Exception.Message)"
    }
    if ($null -eq $info -and $inspectionStatus -eq 'Available') {
        $inspectionStatus = 'Unavailable'
        $inspectionReason = 'Task inspection unavailable: Get-ScheduledTaskInfo returned no metadata'
    }

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
    if ($inspectionStatus -eq 'Unavailable') {
        $status = 'Failed'
        $reasons.Add($inspectionReason)
    }

    # --- Axis 1: the last result code -------------------------------------
    $resultText = 'No result recorded'
    if ($null -ne $lastResult) {
        $hex = '0x{0:X}' -f $lastResult
        if ($lastResult -eq 0) {
            $resultText = 'Success'
        }
        elseif ($hex -eq '0x41301') {
            $resultText = $InformationalCodes[$hex]
        }
        elseif ($TerminationCodes -contains $hex) {
            $resultText = "$hex $($KnownFailureCodes[$hex])"
            $status = 'Failed'
            $reasons.Add("Last run $resultText")
        }
        elseif ($InformationalCodes.ContainsKey($hex)) {
            $resultText = "$hex $($InformationalCodes[$hex])"
        }
        else {
            $named = if ($KnownFailureCodes.ContainsKey($hex)) { $KnownFailureCodes[$hex] } else { 'Action returned a nonzero exit code' }
            $resultText = "$hex $named"
            $status = 'Failed'
            $reasons.Add("Last run $resultText")
        }
    }

    # --- Axis 2: has it actually been running? -----------------------------
    # Evaluated independently. A task can report result 0 and still be stalled.
    $staleness = 'n/a'
    if ($inspectionStatus -eq 'Unavailable') { $staleness = 'not evaluated (inspection unavailable)' }
    if ($task.State -ne 'Disabled' -and $inspectionStatus -eq 'Available') {
        switch ($expectation.Kind) {
            'Interval' {
                $budget = $expectation.Interval * $StaleFactor
                if ($null -eq $lastRun) {
                    # "Has not run" (0x41303) used to be accepted unconditionally,
                    # which let an interval task that never fires sit Healthy
                    # forever. A newly registered task legitimately has not run
                    # yet - the thing that distinguishes the two is whether a
                    # next run is actually scheduled.
                    $staleness = 'never run'
                    if ($null -ne $nextRun -and $nextRun -gt $now) {
                        $staleness = "never run; first run due {0:yyyy-MM-dd HH:mm}" -f $nextRun
                    }
                    else {
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add("Has a $([math]::Round($expectation.Interval.TotalMinutes)) min cadence, has never run, and has no future run scheduled")
                    }
                }
                else {
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
                    }
                    elseif ($lastRun -lt $bootTime) {
                        $staleness = 'predates last boot'
                        if ($status -eq 'Healthy') { $status = 'Stalled' }
                        $reasons.Add(("Triggers on boot or logon but last ran {0:yyyy-MM-dd HH:mm}, before the current boot at {1:yyyy-MM-dd HH:mm}" -f `
                                    $lastRun, $bootTime))
                    }
                    else {
                        $staleness = 'ran this boot'
                    }
                }
            }
            default { $staleness = 'no inferable cadence' }
        }
    }

    # --- Axis 3: what the launcher itself recorded ---------------------------
    # Only for tasks started through dtm-task-run.exe; a direct task is never flagged for
    # lacking a status file. 'missing' is informational: a freshly migrated task has not
    # run through the launcher yet.
    $launcher = Get-LauncherInfo -Task $task
    $statusState = 'n/a'
    $statusExit = $null
    $statusEnd = $null
    if ($launcher.Kind -eq 'dtm-task-run' -and $task.State -ne 'Disabled' -and $inspectionStatus -eq 'Available') {
        if ([string]::IsNullOrWhiteSpace($launcher.TaskId) -or @($StatusDir).Count -eq 0) {
            $statusState = 'not evaluated (no task id or status directory)'
        }
        else {
            $read = Read-LauncherStatus -Dir $StatusDir -TaskId $launcher.TaskId
            switch ($read.State) {
                'missing' { $statusState = 'missing' }
                'unreadable' {
                    $statusState = 'unreadable'
                    $status = 'Failed'
                    $reasons.Add("Launcher status file is unreadable ($($read.Error)): $($read.Path)")
                }
                'ok' {
                    $s = $read.Data
                    $statusState = [string]$s.state
                    $startUtc = ConvertFrom-IsoUtc $s.startUtc
                    $nowUtc = $now.ToUniversalTime()
                    if ($statusState -eq 'running') {
                        $alive = ($task.State -eq 'Running')
                        if (-not $alive -and $s.PSObject.Properties.Name -contains 'pid' -and $s.pid) {
                            $p = Get-Process -Id ([int]$s.pid) -ErrorAction SilentlyContinue
                            $alive = ($null -ne $p -and $p.ProcessName -like 'dtm-task-run*')
                        }
                        if (-not $alive) {
                            $status = 'Failed'
                            $reasons.Add(("Launcher status is still 'running' (started {0:yyyy-MM-dd HH:mm} UTC) but the task is not: the launcher was killed mid-run" -f $startUtc))
                        }
                    }
                    elseif ($statusState -eq 'completed') {
                        $endUtc = ConvertFrom-IsoUtc $s.endUtc
                        $statusEnd = $endUtc
                        $statusExit = [string]$s.exitCodeHex
                        if ($s.timedOut) {
                            $status = 'Failed'
                            $reasons.Add(("Launcher killed the task on timeout ({0}) at {1:yyyy-MM-dd HH:mm} UTC; log: {2}" -f $statusExit, $endUtc, $s.logPath))
                        }
                        elseif ([int64]$s.exitCode -ne 0) {
                            $status = 'Failed'
                            $reasons.Add(("Launcher recorded exit {0} at {1:yyyy-MM-dd HH:mm} UTC; log: {2}" -f $statusExit, $endUtc, $s.logPath))
                        }
                        if ($task.State -ne 'Running') {
                            # A status file that is older than the task's own cadence means the launcher
                            # has not recorded a run recently, whatever Task Scheduler remembers.
                            if ($expectation.Kind -eq 'Interval' -and $null -ne $endUtc) {
                                $age = $nowUtc - $endUtc
                                if ($age -gt ($expectation.Interval * $StaleFactor)) {
                                    if ($status -eq 'Healthy') { $status = 'Stalled' }
                                    $reasons.Add(("Launcher status is {0:N1} h old but the cadence is {1:N1} h (allowed {2}x)" -f $age.TotalHours, $expectation.Interval.TotalHours, $StaleFactor))
                                }
                            }
                            elseif ($expectation.Kind -eq 'BootOrLogon' -and $null -ne $bootTime -and $null -ne $endUtc -and $endUtc.ToLocalTime() -lt $bootTime) {
                                if ($status -eq 'Healthy') { $status = 'Stalled' }
                                $reasons.Add(("Launcher status predates the current boot ({0:yyyy-MM-dd HH:mm} UTC)" -f $endUtc))
                            }
                            # Task Scheduler ran it more recently than the launcher recorded: the launcher
                            # died before it could write (or the status directory is not writable).
                            if ($null -ne $lastRun -and $null -ne $startUtc -and ($lastRun.ToUniversalTime() - $startUtc) -gt [TimeSpan]::FromMinutes(5)) {
                                if ($status -eq 'Healthy') { $status = 'Stalled' }
                                $reasons.Add(("Task Scheduler last ran it {0:yyyy-MM-dd HH:mm} UTC but the launcher's latest record started {1:yyyy-MM-dd HH:mm} UTC" -f $lastRun.ToUniversalTime(), $startUtc))
                            }
                        }
                    }
                    else {
                        $status = 'Failed'
                        $reasons.Add("Launcher status has unknown state '$statusState'")
                    }
                }
            }
        }
    }

    if ($missed -gt 0) { $reasons.Add("$missed missed run(s) recorded") }

    if ($task.State -eq 'Disabled') { $status = 'Disabled' }
    if ($Expected -contains $task.TaskName) { $status = 'Ignored' }

    [pscustomobject]@{
        TaskPath         = $task.TaskPath
        TaskName         = $task.TaskName
        Owner            = $owner
        State            = [string]$task.State
        Status           = $status
        InspectionStatus = $inspectionStatus
        LastRunTime      = $lastRun
        NextRunTime      = $nextRun
        LastResult       = $lastResult
        ResultText       = $resultText
        Cadence          = $expectation.Kind
        Staleness        = $staleness
        Launcher         = $launcher.Kind
        LauncherStatus   = $statusState
        LauncherExit     = $statusExit
        LauncherEndUtc   = if ($null -ne $statusEnd) { $statusEnd.ToString('o') } else { $null }
        MissedRuns       = $missed
        Reasons          = @($reasons)
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
# conhost --headless hides the window AND the exit code, so these tasks always show Success.
# Not counted as a failure (they are not known to be broken), but never let them look verified.
$conhostWrapped = @($results | Where-Object { $_.Launcher -eq 'conhost' -and $_.Status -notin 'Disabled', 'Ignored' })

$summary = [pscustomobject]@{
    GeneratedAt      = $now.ToString('o')
    Computer         = $env:COMPUTERNAME
    LastBootUp       = if ($null -ne $bootTime) { $bootTime.ToString('o') } else { $null }
    StaleFactor      = $StaleFactor
    LocalOnly        = [bool]$LocalOnly
    VendorSuppressed = $vendorSuppressed
    TotalTasks       = $results.Count
    Healthy          = @($results | Where-Object { $_.Status -eq 'Healthy' }).Count
    Failed           = $failed.Count
    Stalled          = $stalled.Count
    Disabled         = @($results | Where-Object { $_.Status -eq 'Disabled' }).Count
    Ignored          = @($results | Where-Object { $_.Status -eq 'Ignored' }).Count
    ConhostWrapped   = $conhostWrapped.Count
    Tasks            = $results
}

if ($issues.Count -gt 0) {
    Write-Host ''
    Write-Host "Scheduled task issues on $($env:COMPUTERNAME):" -ForegroundColor Yellow
    foreach ($r in ($issues | Sort-Object Status, TaskName)) {
        $colour = if ($r.Status -eq 'Failed') { 'Red' } else { 'Yellow' }
        Write-Host ("  [{0}] {1}{2}" -f $r.Status.ToUpper(), $r.TaskPath, $r.TaskName) -ForegroundColor $colour
        foreach ($reason in $r.Reasons) { Write-Host "      - $reason" }
    }
}
else {
    Write-Host "All $($results.Count) task(s) healthy." -ForegroundColor Green
}

if ($conhostWrapped.Count -gt 0) {
    Write-Host ''
    Write-Host ("{0} task(s) still run behind 'conhost --headless', which masks their exit code (always 0x0), so their Result above is unverified:" -f $conhostWrapped.Count) -ForegroundColor DarkYellow
    foreach ($r in ($conhostWrapped | Sort-Object TaskName)) { Write-Host ("  {0}{1}" -f $r.TaskPath, $r.TaskName) }
    Write-Host '  Convert them with Tools/Convert-TaskToLauncher.ps1 (see docs/scheduled-tasks/LAUNCHER.md).'
}

Write-Host ''
Write-Host ("Total {0} | healthy {1} | failed {2} | stalled {3} | disabled {4} | ignored {5} | conhost-wrapped {6}" -f `
        $summary.TotalTasks, $summary.Healthy, $summary.Failed, $summary.Stalled, $summary.Disabled, $summary.Ignored, $summary.ConhostWrapped)

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
