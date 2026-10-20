#Requires -Version 7.0
<#
.SYNOPSIS
Regenerates docs/scheduled-tasks/task-registry.json and TASK_REGISTRY.md from the live Task Scheduler.

.DESCRIPTION
The registry answers "what is this task for, who owns it, how does it start, how long may it run and
how will I know it failed?" without guessing. Facts that Task Scheduler knows (triggers, delays, time
limit, instance policy, principal, launcher, last result) are read live, so they cannot drift. Facts
that only a person reading the script can know (purpose, owning repo/script, failure signal, whether
the sibling host needs it) come from the curated notes file, docs/scheduled-tasks/task-registry-notes.json.
Measured run time comes from the Task Scheduler operational log (events 100 and 102, paired by
instance id) over the last -HistoryDays days.

A task with no entry in the notes file is written with purpose UNKNOWN rather than a guess.

.PARAMETER NotesPath
Curated notes (one object per task, keyed by path + name).

.PARAMETER OutputJson
Machine-readable registry. Default: docs/scheduled-tasks/task-registry.json next to the notes.

.PARAMETER OutputMarkdown
Human-readable registry. Default: docs/scheduled-tasks/TASK_REGISTRY.md next to the notes.

.PARAMETER BackupDir
Where Convert-TaskToLauncher.ps1 saved pre-launcher__*.xml (used to record each migrated task's original action).

.PARAMETER HistoryDays
How far back to measure run times. 0 or -SkipHistory skips the (slow, ~15 s) event-log read.

.EXAMPLE
./Export-ScheduledTaskRegistry.ps1
#>
[CmdletBinding()]
param(
    [string]$NotesPath = (Join-Path $PSScriptRoot '..\docs\scheduled-tasks\task-registry-notes.json'),
    [string]$OutputJson,
    [string]$OutputMarkdown,
    [string]$BackupDir = (Join-Path $env:USERPROFILE '.local\state\task-backups'),
    [ValidateRange(0, 60)][int]$HistoryDays = 14,
    [switch]$SkipHistory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$notesDir = Split-Path -Parent (Resolve-Path -LiteralPath $NotesPath)
if (-not $OutputJson) { $OutputJson = Join-Path $notesDir 'task-registry.json' }
if (-not $OutputMarkdown) { $OutputMarkdown = Join-Path $notesDir 'TASK_REGISTRY.md' }

function Format-Span {
    param([string]$Iso)
    if ([string]::IsNullOrWhiteSpace($Iso)) { return $null }
    $t = [System.Xml.XmlConvert]::ToTimeSpan($Iso)
    if ($t -le [TimeSpan]::Zero) { return $null }
    if ($t.TotalDays -ge 1 -and $t.TotalDays -eq [math]::Floor($t.TotalDays)) { return '{0}d' -f [int]$t.TotalDays }
    if ($t.TotalHours -ge 1 -and $t.TotalHours -eq [math]::Floor($t.TotalHours)) { return '{0}h' -f [int]$t.TotalHours }
    if ($t.TotalMinutes -ge 1 -and $t.TotalMinutes -eq [math]::Floor($t.TotalMinutes)) { return '{0}m' -f [int]$t.TotalMinutes }
    return '{0}s' -f [int]$t.TotalSeconds
}

function Get-TriggerSummary {
    param($Trigger)
    $kind = ($Trigger.CimClass.CimClassName -replace '^MSFT_Task', '' -replace 'Trigger$', '')
    $props = $Trigger.PSObject.Properties.Name
    $delay = if ($props -contains 'Delay') { Format-Span $Trigger.Delay } else { $null }
    $start = $null
    if ($props -contains 'StartBoundary' -and $Trigger.StartBoundary) { $start = [datetime]::Parse($Trigger.StartBoundary, [Globalization.CultureInfo]::InvariantCulture) }
    $every = $null; $for = $null
    if ($props -contains 'Repetition' -and $Trigger.Repetition) {
        $every = Format-Span $Trigger.Repetition.Interval
        $for = Format-Span $Trigger.Repetition.Duration
    }
    $text = switch ($kind) {
        'Logon' { 'at logon' + $(if ($delay) { " +$delay" }) }
        'Boot' { 'at boot' + $(if ($delay) { " +$delay" }) }
        'Daily' { 'daily' + $(if ($start) { " $($start.ToString('HH:mm'))" }) }
        'Weekly' { 'weekly' + $(if ($start) { " $($start.ToString('HH:mm'))" }) }
        'Time' { if ($every) { "every $every" + $(if ($for) { " for $for" }) } else { 'once' + $(if ($start) { " $($start.ToString('yyyy-MM-dd HH:mm'))" }) } }
        'Idle' { 'when idle' }
        'Event' { 'on event' }
        'Registration' { 'at registration' }
        default { $kind.ToLowerInvariant() }
    }
    if ($kind -in 'Daily', 'Weekly', 'Logon', 'Boot' -and $every) { $text += ", repeating every $every" + $(if ($for) { " for $for" }) }
    [pscustomobject]@{ type = $kind; delay = $delay; repeatEvery = $every; repeatFor = $for; start = if ($start) { $start.ToString('s') } else { $null }; text = $text }
}

function Get-LauncherInfo {
    param($Task)
    $a = @($Task.Actions)[0]
    # COM-handler actions have no Execute property.
    if ($null -eq $a -or $a.PSObject.Properties.Name -notcontains 'Execute' -or -not $a.Execute) { return @{ Kind = 'com-handler'; Id = $null; Timeout = $null; Notify = $false } }
    $leaf = [IO.Path]::GetFileName(([string]$a.Execute).Trim('"'))
    $argText = [string]$a.Arguments
    if ($leaf -ieq 'dtm-task-run.exe') {
        $head = ($argText -split '\s--(?:\s|$)', 2)[0]
        $id = $null; $timeout = $null
        if ($head -match '(?i)--name\s+(?:"(?<q>[^"]+)"|(?<b>\S+))') { $id = if ($Matches['q']) { $Matches['q'] } else { $Matches['b'] } }
        if ($head -match '(?i)--timeout\s+(?<t>\d+)') { $timeout = [int]$Matches['t'] }
        return @{ Kind = 'dtm-task-run'; Id = $id; Timeout = $timeout; Notify = ($head -match '(?i)--notify-bus') }
    }
    if ($leaf -ieq 'conhost.exe' -and $argText -match '(?i)^\s*--headless\b') { return @{ Kind = 'conhost --headless'; Id = $null; Timeout = $null; Notify = $false } }
    @{ Kind = 'direct'; Id = $null; Timeout = $null; Notify = $false }
}

function ConvertTo-LauncherStem {
    <# dtm-task-run's rule for file names: invalid characters and spaces become '_' (see LAUNCHER.md). #>
    param([string]$Name)
    $invalid = [IO.Path]::GetInvalidFileNameChars()
    $chars = $Name.Trim().ToCharArray()
    for ($i = 0; $i -lt $chars.Length; $i++) { if ($chars[$i] -eq ' ' -or $invalid -contains $chars[$i]) { $chars[$i] = '_' } }
    $stem = (-join $chars).Trim('.')
    if ($stem.Length -eq 0) { '_' } else { $stem }
}

function Get-RunHistory {
    <# Duration per task from TaskScheduler/Operational: event 100 (started) paired with 102 (finished) by instance id. #>
    param([int]$Days)
    $byTask = @{}
    if ($Days -le 0) { return $byTask }
    $events = @(Get-WinEvent -FilterHashtable @{ LogName = 'Microsoft-Windows-TaskScheduler/Operational'; Id = 100, 102; StartTime = (Get-Date).AddDays(-$Days) } -ErrorAction SilentlyContinue)
    $starts = @{}
    foreach ($e in ($events | Sort-Object TimeCreated)) {
        $task = [string]$e.Properties[0].Value
        $instance = [string]$e.Properties[2].Value
        if ($e.Id -eq 100) { $starts[$instance] = $e.TimeCreated; continue }
        if ($starts.ContainsKey($instance)) {
            if (-not $byTask.ContainsKey($task)) { $byTask[$task] = [System.Collections.Generic.List[double]]::new() }
            $byTask[$task].Add(($e.TimeCreated - $starts[$instance]).TotalSeconds)
        }
    }
    $byTask
}

function Format-Seconds {
    param([double]$S)
    if ($S -lt 1) { return '<1s' }
    if ($S -lt 90) { return ('{0:N0}s' -f $S) }
    if ($S -lt 5400) { return ('{0:N0}m' -f ($S / 60)) }
    return ('{0:N1}h' -f ($S / 3600))
}

# --- Gather ---------------------------------------------------------------------------------
$notes = @{}
foreach ($n in (Get-Content -LiteralPath $NotesPath -Raw | ConvertFrom-Json)) { $notes["$($n.path)$($n.name)"] = $n }
$history = if ($SkipHistory) { @{} } else { Get-RunHistory -Days $HistoryDays }
$live = @(Get-ScheduledTask -ErrorAction Stop | Where-Object { $_.TaskPath -notlike '\Microsoft\*' } | Sort-Object TaskPath, TaskName)

$rows = foreach ($t in $live) {
    $key = "$($t.TaskPath)$($t.TaskName)"
    $info = $t | Get-ScheduledTaskInfo
    $n = $notes[$key]
    $launcher = Get-LauncherInfo $t
    $action = @($t.Actions)[0]
    $triggers = @(@($t.Triggers) | Where-Object { $null -ne $_ } | ForEach-Object { Get-TriggerSummary $_ })
    $limit = if ($t.Settings.ExecutionTimeLimit) { Format-Span ([string]$t.Settings.ExecutionTimeLimit) } else { $null }
    $runs = $history[$key]
    $measured = $null
    if ($runs -and $runs.Count -gt 0) {
        $s = @($runs | Sort-Object)
        $median = if ($s.Count % 2) { $s[($s.Count - 1) / 2] } else { ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2 }
        $measured = [pscustomobject]@{ runs = $s.Count; medianSeconds = [math]::Round($median, 1); maxSeconds = [math]::Round($s[-1], 1); windowDays = $HistoryDays }
    }
    $pre = Join-Path $BackupDir ("pre-launcher__" + ((($t.TaskPath.Trim('\') + '\' + $t.TaskName).Trim('\')) -replace '[\\ ]', '_') + '.xml')
    $originalAction = $null
    if (Test-Path -LiteralPath $pre) {
        [xml]$x = Get-Content -LiteralPath $pre -Raw
        $ns = [System.Xml.XmlNamespaceManager]::new($x.NameTable); $ns.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $e = $x.SelectSingleNode('//t:Actions/t:Exec', $ns)
        if ($e) { $originalAction = "$($e.Command) $($e.Arguments)".Trim() }
    }
    $vendor = ($null -ne $n -and $n.confidence -eq 'vendor')
    $p1gen9Raw = if ($n -and $n.needsOnDp1gen9) { [string]$n.needsOnDp1gen9 } else { 'unknown - not assessed' }
    $notifyReason = if ($n -and $n.PSObject.Properties.Name -contains 'notifyBusReason') { [string]$n.notifyBusReason } else { $null }
    $launcherNote = if ($n -and $n.PSObject.Properties.Name -contains 'launcherNote') { [string]$n.launcherNote } else { $null }
    $knownIssue = if ($n -and $n.PSObject.Properties.Name -contains 'knownIssue') { [string]$n.knownIssue } else { $null }
    [pscustomobject]([ordered]@{
            task               = $key
            path               = $t.TaskPath
            name               = $t.TaskName
            category           = if ($vendor) { 'vendor' } else { 'own' }
            state              = [string]$t.State
            purpose            = if ($n) { [string]$n.purpose } else { 'UNKNOWN - no entry in task-registry-notes.json' }
            confidence         = if ($n) { [string]$n.confidence } else { 'unknown' }
            owner              = if ($n) { [string]$n.owner } else { 'UNKNOWN' }
            script             = if ($n) { $n.script } else { $null }
            command            = if ($action -and $action.PSObject.Properties.Name -contains 'Execute') { "$($action.Execute) $($action.Arguments)".Trim() } elseif ($action) { "COM handler $($action.ClassId)".Trim() } else { $null }
            triggers           = $triggers
            executionTimeLimit = $limit
            multipleInstances  = [string]$t.Settings.MultipleInstances
            principal          = [pscustomobject]@{ user = ([string]$t.Principal.UserId).Split('\')[-1]; logonType = [string]$t.Principal.LogonType; runLevel = [string]$t.Principal.RunLevel }
            launcher           = $launcher.Kind
            launcherId         = $launcher.Id
            launcherTimeoutSeconds = $launcher.Timeout
            notifyBus          = [bool]$launcher.Notify
            notifyBusReason    = $notifyReason
            launcherNote       = $launcherNote
            knownIssue         = $knownIssue
            originalAction     = $originalAction
            expectedRuntime    = if ($n) { [string]$n.expectedRuntimeHint } else { $null }
            measuredRuntime    = $measured
            failureSignal      = if ($n) { [string]$n.failureSignal } else { $null }
            statusFile         = if ($launcher.Id) { '%LOCALAPPDATA%\dtm-tasks\status\' + (ConvertTo-LauncherStem $launcher.Id) + '.json' } else { $null }
            logFile            = if ($launcher.Id) { '%LOCALAPPDATA%\dtm-tasks\logs\' + (ConvertTo-LauncherStem $launcher.Id) + '.log' } else { $null }
            lastRun            = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 1999) { $info.LastRunTime.ToString('s') } else { $null }
            lastResult         = ('0x{0:X}' -f [uint32]$info.LastTaskResult)
            p1gen9             = $p1gen9Raw
            evidence           = if ($n) { @($n.checked) } else { @() }
        })
}
$rows = @($rows)

# --- JSON -----------------------------------------------------------------------------------
$registry = [pscustomobject]@{
    schemaVersion = 1
    host          = $env:COMPUTERNAME.ToLowerInvariant()
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    generator     = 'Tools/Export-ScheduledTaskRegistry.ps1'
    notes         = 'docs/scheduled-tasks/task-registry-notes.json'
    runtimeHistoryDays = $HistoryDays
    counts        = [pscustomobject]@{
        total = $rows.Count
        own = @($rows | Where-Object category -eq 'own').Count
        vendor = @($rows | Where-Object category -eq 'vendor').Count
        viaLauncher = @($rows | Where-Object launcher -eq 'dtm-task-run').Count
        conhostHeadless = @($rows | Where-Object launcher -eq 'conhost --headless').Count
        notifyBus = @($rows | Where-Object notifyBus).Count
        purposeUnknown = @($rows | Where-Object { $_.purpose -match '^UNKNOWN' }).Count
    }
    tasks         = $rows
}
[IO.File]::WriteAllText($OutputJson, ($registry | ConvertTo-Json -Depth 8) + "`n", [Text.UTF8Encoding]::new($false))

# --- Markdown -------------------------------------------------------------------------------
function Escape-Cell { param($s) if ($null -eq $s) { return '' }; ([string]$s -replace '\|', '\|' -replace '\r?\n', ' ').Trim() }
function Get-Anchor { param($Row) ($Row.task.TrimStart('\') -replace '[^A-Za-z0-9]+', '-').Trim('-').ToLowerInvariant() }

$md = [System.Text.StringBuilder]::new()
[void]$md.AppendLine('# Scheduled task registry (dtm-p1gen7)')
[void]$md.AppendLine()
[void]$md.AppendLine('Generated by [`Tools/Export-ScheduledTaskRegistry.ps1`](../../Tools/Export-ScheduledTaskRegistry.ps1) from the live Task Scheduler plus the curated')
[void]$md.AppendLine('[`task-registry-notes.json`](task-registry-notes.json); the machine-readable form is [`task-registry.json`](task-registry.json).')
[void]$md.AppendLine('Do not edit the tables by hand: change the notes file (purpose, owner, failure signal) or the task, then regenerate.')
[void]$md.AppendLine()
[void]$md.AppendLine("Generated $($registry.generatedAtUtc) on $($registry.host). $($registry.counts.total) non-Microsoft tasks: $($registry.counts.own) of ours, $($registry.counts.vendor) vendor/app updaters.")
$bt = '`'
[void]$md.AppendLine(('{1} run through [{0}dtm-task-run{0}](LAUNCHER.md) (exit code and output preserved, {2} also notify the agent bus), {3} still run behind {0}conhost.exe --headless{0}' -f $bt, $registry.counts.viaLauncher, $registry.counts.notifyBus, $registry.counts.conhostHeadless))
[void]$md.AppendLine('(window hidden, exit code masked: their Task Scheduler result is always 0x0 and proves nothing).')
[void]$md.AppendLine()
[void]$md.AppendLine("Expected runtime is Task Scheduler's own start/finish events for the last $HistoryDays days (median and max) followed by what the script's header says. The window")
[void]$md.AppendLine('can include runs made before the 2026-10-20 scheduler changes (new delays, limits and wrappers), so read it as a range, not a promise.')
[void]$md.AppendLine()
[void]$md.AppendLine('How to read it: each task has one row in **Launch and schedule** and one in **Intent and failure signals**, in the same order.')
[void]$md.AppendLine('`UNKNOWN` means the purpose could not be established from the script; the `evidence` field of the JSON lists what was read.')
[void]$md.AppendLine('The failure signal column says where to look: for a launcher task that is always `%LOCALAPPDATA%\dtm-tasks\status\<id>.json` (exit code, timeout, stderr tail),')
[void]$md.AppendLine('`logs\<id>.log` (all output) and the Application event log, source `dtm-task-run`, ids 1001 failed / 1002 timed out / 1003 could not start.')
[void]$md.AppendLine()

$own = @($rows | Where-Object category -eq 'own')
$vendor = @($rows | Where-Object category -eq 'vendor')

[void]$md.AppendLine('## Our tasks')
[void]$md.AppendLine()
[void]$md.AppendLine('### Launch and schedule')
[void]$md.AppendLine()
[void]$md.AppendLine('| Task | State | Trigger(s) | Time limit | Instances | Launcher | Bus | Expected runtime | Last result |')
[void]$md.AppendLine('|---|---|---|---|---|---|---|---|---|')
foreach ($r in $own) {
    $trg = ($r.triggers | ForEach-Object { $_.text }) -join '; '
    if (-not $trg) { $trg = 'on demand (no trigger)' }
    $lim = if ($r.executionTimeLimit) { $r.executionTimeLimit } else { 'none' }
    if ($r.launcherTimeoutSeconds) { $lim += " (launcher kills at $($r.launcherTimeoutSeconds)s)" }
    $rt = ''
    if ($r.measuredRuntime) { $rt = "median $(Format-Seconds $r.measuredRuntime.medianSeconds), max $(Format-Seconds $r.measuredRuntime.maxSeconds) ($($r.measuredRuntime.runs) runs/$($r.measuredRuntime.windowDays)d)" }
    if ($r.expectedRuntime) { $rt = if ($rt) { "$rt. $($r.expectedRuntime)" } else { $r.expectedRuntime } }
    $busCell = if ($r.notifyBus) { 'yes' } else { 'no' }
    $name = ('[{0}](#{1})' -f (Escape-Cell $r.task), (Get-Anchor $r))
    $last = if ($r.launcher -eq 'conhost --headless') { "$($r.lastResult) (masked)" } else { $r.lastResult }
    [void]$md.AppendLine(('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |' -f $name, $r.state, (Escape-Cell $trg), (Escape-Cell $lim), $r.multipleInstances, $r.launcher, $busCell, (Escape-Cell $rt), $last))
}
[void]$md.AppendLine()
[void]$md.AppendLine('### Intent and failure signals')
[void]$md.AppendLine()
[void]$md.AppendLine('| Task | Purpose | Owner / script | Failure signal | On dtm-p1gen9 |')
[void]$md.AppendLine('|---|---|---|---|---|')
foreach ($r in $own) {
    $owner = Escape-Cell $r.owner
    $bt = '`'
    if ($r.script) { $owner += '<br>' + $bt + (Escape-Cell $r.script) + $bt }
    $sig = Escape-Cell $r.failureSignal
    if ($r.statusFile) { $sig = 'Status {0}{1}{0}, log {0}{2}{0}; {3}' -f $bt, $r.statusFile, $r.logFile, $sig }
    elseif ($r.launcher -eq 'conhost --headless') { $sig = "Exit code is masked (always 0x0); $sig" }
    $name = ('<a id="{0}"></a>{1}' -f (Get-Anchor $r), (Escape-Cell $r.task))
    [void]$md.AppendLine(('| {0} | {1} | {2} | {3} | {4} |' -f $name, (Escape-Cell $r.purpose), $owner, $sig, (Escape-Cell $r.p1gen9)))
}
[void]$md.AppendLine()

$withNotes = @($own | Where-Object { $_.notifyBusReason -or $_.launcherNote -or $_.originalAction })
if ($withNotes.Count) {
    [void]$md.AppendLine('### Launcher decisions')
    [void]$md.AppendLine()
    [void]$md.AppendLine('Why a task does or does not notify the agent bus, and what it ran before the migration.')
    [void]$md.AppendLine()
    [void]$md.AppendLine('| Task | Bus | Decision | Original action (before migration) |')
    [void]$md.AppendLine('|---|---|---|---|')
    foreach ($r in $withNotes) {
        $reason = if ($r.notifyBusReason) { $r.notifyBusReason } else { '' }
        if ($r.launcherNote) { $reason = (@($reason, $r.launcherNote) | Where-Object { $_ }) -join ' ' }
        $orig = if ($r.originalAction) { '`' + (Escape-Cell $r.originalAction) + '`' } else { '' }
        [void]$md.AppendLine(('| {0} | {1} | {2} | {3} |' -f (Escape-Cell $r.task), $(if ($r.notifyBus) { 'yes' } else { 'no' }), (Escape-Cell $reason), $orig))
    }
    [void]$md.AppendLine()
}

$withIssues = @($own | Where-Object { $_.knownIssue })
if ($withIssues.Count) {
    [void]$md.AppendLine('### Known issues found while building the registry')
    [void]$md.AppendLine()
    [void]$md.AppendLine('"Verified" means re-checked on the live system when the registry was written; the rest come from reading the scripts and should be re-checked before acting.')
    [void]$md.AppendLine()
    [void]$md.AppendLine('| Task | Issue |')
    [void]$md.AppendLine('|---|---|')
    foreach ($r in $withIssues) { [void]$md.AppendLine(('| {0} | {1} |' -f (Escape-Cell $r.task), (Escape-Cell $r.knownIssue))) }
    [void]$md.AppendLine()
}

[void]$md.AppendLine('## Vendor and application tasks')
[void]$md.AppendLine()
[void]$md.AppendLine('Installed by vendors or apps; listed so the registry covers every non-Microsoft task. Not researched beyond the task description.')
[void]$md.AppendLine()
[void]$md.AppendLine('| Task | State | Trigger(s) | What it is | Owner |')
[void]$md.AppendLine('|---|---|---|---|---|')
foreach ($r in $vendor) {
    $trg = ($r.triggers | ForEach-Object { $_.text }) -join '; '
    if (-not $trg) { $trg = 'on demand' }
    [void]$md.AppendLine(('| {0} | {1} | {2} | {3} | {4} |' -f (Escape-Cell $r.task), $r.state, (Escape-Cell $trg), (Escape-Cell $r.purpose), (Escape-Cell $r.owner)))
}
[void]$md.AppendLine()
[void]$md.AppendLine('## Moving this to dtm-p1gen9')
[void]$md.AppendLine()
[void]$md.AppendLine('1. Deploy `dtm-task-run.exe` to the same relative path (`~\.local\share\pc-ai\deployed\bin`) and register the event-log source once from an elevated shell: `dtm-task-run.exe --install-eventlog-source`.')
[void]$md.AppendLine('2. Recreate only the tasks marked `yes` in the last column, then wrap them with [`Tools/Convert-TaskToLauncher.ps1`](../../Tools/Convert-TaskToLauncher.ps1) (it carries the child command line over verbatim).')
[void]$md.AppendLine('3. Tasks marked `no` are tied to this host (VHDX mounts, drive letters, local runner registrations, local GPUs); do not copy their XML.')
[void]$md.AppendLine('4. Regenerate this file on p1gen9 with its own notes; `Tools/Test-ScheduledTaskHealth.ps1` works unchanged there.')
[IO.File]::WriteAllText($OutputMarkdown, $md.ToString(), [Text.UTF8Encoding]::new($false))

"Registry: $($rows.Count) tasks ($($registry.counts.own) own, $($registry.counts.vendor) vendor); $($registry.counts.viaLauncher) via launcher, $($registry.counts.conhostHeadless) conhost; $($registry.counts.purposeUnknown) purpose unknown"
"  $OutputJson"
"  $OutputMarkdown"
