#Requires -Version 7.0
<#
.SYNOPSIS
Moves scheduled tasks from `conhost.exe --headless <cmd>` to `dtm-task-run.exe -- <cmd>` (or back).

.DESCRIPTION
`conhost.exe --headless` hides a task's window but masks its exit code: LastTaskResult is always
0x0. dtm-task-run keeps the window hidden and exits with the child's real code (see
docs/scheduled-tasks/LAUNCHER.md). This script rewrites the single Exec action of each named task:

    conhost.exe --headless <child command line>
 -> dtm-task-run.exe --name <TaskName> [--timeout N] [--notify-bus] -- <child command line>

The child command line is copied into the new action's arguments character for character. Note that
dtm-task-run then splits it into separate arguments and quotes each one again for the child
(ProcessStartInfo.ArgumentList), whereas conhost passed the line through unchanged: ordinary quoting
survives, but a command whose meaning depends on unusual quoting (nested `cmd /c ""..." "...""`) should
be run once after conversion and its log checked. Triggers, principal, settings and run history are
untouched (Set-ScheduledTask updates in place). Before any
change the task XML is exported to <BackupDir>\pre-launcher__<task>.xml, and afterwards to
after-launcher__<task>.xml.

--timeout defaults to the task's own ExecutionTimeLimit minus 10 s, so the launcher reports a hang as
exit 1460 with a status file and event-log entry instead of Task Scheduler killing it silently
(LastTaskResult 0x41306). Tasks with no limit (daemons, runners) get no --timeout. dtm-task-run refuses
a --timeout above 86400 s with exit 87 and never starts the child (LEARNED_RULES 85), so a task whose
limit is longer than that (Task Scheduler's default is PT72H) also gets no --timeout and keeps relying
on its own limit; an explicit -TimeoutSeconds above 86400 is an error.

.PARAMETER TaskName
Full task names, e.g. '\Bitwarden-Archive-Refresh', '\PowerShell\ProfileLogSync'.

.PARAMETER NotifyBus
Subset of -TaskName whose failures should also be posted to the agent bus.

.PARAMETER TimeoutSeconds
Per-task override: @{ '\Name' = 120 }. 0 disables the launcher timeout for that task.

.PARAMETER LauncherPath
Path of dtm-task-run.exe. Defaults to the deployed copy.

.PARAMETER BackupDir
Where XML exports go.

.PARAMETER Revert
Restore each task's action (command, arguments and working directory) from its pre-launcher__ XML
backup instead.

.PARAMETER DryRun
Non-mutating preview: report each new action without changing any task and without creating the backup
directory or writing XML files. The long CLI form `--DryRun` is also accepted. Equivalent to -WhatIf,
exposed under the name the repo's session-script contract requires (AGENTS.md).

.PARAMETER Help
Print this help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
./Convert-TaskToLauncher.ps1 -TaskName '\LspmuxServer','\Bitwarden-Archive-Refresh' -NotifyBus '\Bitwarden-Archive-Refresh' -DryRun
.EXAMPLE
./Convert-TaskToLauncher.ps1 -TaskName '\Bitwarden-Archive-Refresh' -Revert -DryRun
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string[]]$TaskName = @(),
    [string[]]$NotifyBus = @(),
    [hashtable]$TimeoutSeconds = @{},
    [string]$LauncherPath = (Join-Path $env:USERPROFILE '.local\share\pc-ai\deployed\bin\dtm-task-run.exe'),
    [string]$BackupDir = (Join-Path $env:USERPROFILE '.local\state\task-backups'),
    [switch]$Revert,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The launcher's own upper bound (Native/DtmTaskRun/Options.cs MaxTimeoutSeconds). A larger --timeout makes
# dtm-task-run exit 87 without starting the child, so the converted task would never run (LEARNED_RULES 85).
$script:MaxLauncherTimeoutSeconds = 86400

# CLI contract (AGENTS.md): -h/--help and -DryRun/--DryRun are handled before anything touches Task
# Scheduler, so help works on a machine where the conversion could not run.
$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if (@($CliArgs).Count -gt 0) { throw "Unknown arguments: $($CliArgs -join ' ')" }
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
# Every task change and every file write in Invoke-Conversion is ShouldProcess-guarded (New-Item honours
# $WhatIfPreference itself), so forcing $WhatIfPreference makes -DryRun genuinely non-mutating.
if ($DryRun) { $WhatIfPreference = $true }

function Get-TaskBackupName {
    <# Matches the scheduler agent's convention: path separators and spaces become underscores. #>
    param([string]$Path, [string]$Name)
    (($Path.Trim('\') + '\' + $Name).Trim('\')) -replace '[\\ ]', '_'
}

function ConvertTo-LauncherArguments {
    <#
    Builds the dtm-task-run argument string from a conhost action's argument string.
    Returns $null when the action is not a `conhost.exe --headless ...` wrapper.
    #>
    param(
        [Parameter(Mandatory)][string]$Execute,
        [AllowEmptyString()][string]$Arguments,
        [Parameter(Mandatory)][string]$Name,
        [int]$TimeoutSeconds = 0,
        [switch]$NotifyBus
    )
    if ($Execute -notmatch '(?i)(^|[\\/])conhost\.exe"?$') { return $null }
    if ($Arguments -notmatch '^\s*--headless\s+(?<child>\S.*)$') { return $null }
    $child = $Matches['child']
    $parts = [System.Collections.Generic.List[string]]::new()
    $parts.Add('--name'); $parts.Add('"' + ($Name -replace '"', '') + '"')
    if ($TimeoutSeconds -gt 0) { $parts.Add('--timeout'); $parts.Add([string]$TimeoutSeconds) }
    if ($NotifyBus) { $parts.Add('--notify-bus') }
    $parts.Add('--')
    ($parts -join ' ') + ' ' + $child
}

function Get-DefaultTimeoutSeconds {
    <#
    ExecutionTimeLimit minus 10 s. 0 (no launcher timeout) when the task has no finite limit, and also when
    the result would exceed the launcher's 86400 s maximum: that task keeps relying on Task Scheduler's own
    limit rather than getting a --timeout that dtm-task-run rejects.
    #>
    param([string]$ExecutionTimeLimit)
    if ([string]::IsNullOrWhiteSpace($ExecutionTimeLimit)) { return 0 }
    $span = [System.Xml.XmlConvert]::ToTimeSpan($ExecutionTimeLimit)
    if ($span -le [TimeSpan]::Zero) { return 0 }
    $seconds = [math]::Max(1, $span.TotalSeconds - 10)
    if ($seconds -gt $script:MaxLauncherTimeoutSeconds) { return 0 }
    [int]$seconds
}

function Get-BackupExecAction {
    <# Command, Arguments and WorkingDirectory of the single Exec action in an exported task XML file. #>
    param([Parameter(Mandatory)][string]$Path)
    [xml]$old = Get-Content -LiteralPath $Path -Raw
    $ns = [System.Xml.XmlNamespaceManager]::new($old.NameTable)
    $ns.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
    $exec = $old.SelectSingleNode('//t:Actions/t:Exec', $ns)
    if ($null -eq $exec) { throw "No Exec action in $Path" }
    $values = @{}
    foreach ($field in 'Command', 'Arguments', 'WorkingDirectory') {
        $node = $exec.SelectSingleNode("t:$field", $ns)
        $values[$field] = if ($node) { $node.InnerText } else { '' }
    }
    [pscustomobject]$values
}

function Invoke-Conversion {
    [CmdletBinding(SupportsShouldProcess)]
    param()
    if (@($TaskName).Count -eq 0) { throw '-TaskName is required (see -h)' }
    if (-not $Revert -and -not (Test-Path -LiteralPath $LauncherPath -PathType Leaf)) {
        throw "Launcher not found: $LauncherPath (deploy it first)"
    }
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

    foreach ($full in $TaskName) {
        $path = (Split-Path $full -Parent).Replace('/', '\')
        if (-not $path.EndsWith('\')) { $path += '\' }
        if (-not $path.StartsWith('\')) { $path = '\' + $path }
        $name = Split-Path $full -Leaf
        $task = Get-ScheduledTask -TaskPath $path -TaskName $name -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskPath $task.TaskPath -TaskName $task.TaskName
        $backup = Get-TaskBackupName $task.TaskPath $task.TaskName
        $preFile = Join-Path $BackupDir "pre-launcher__$backup.xml"
        $postFile = Join-Path $BackupDir "after-launcher__$backup.xml"

        if ($Revert) {
            if (-not (Test-Path -LiteralPath $preFile)) { throw "No backup to restore for $full ($preFile)" }
            $exec = Get-BackupExecAction -Path $preFile
            if ($PSCmdlet.ShouldProcess($full, "restore action from $preFile ($($exec.Command) $($exec.Arguments))")) {
                # The forward conversion keeps WorkingDirectory, so the rollback must restore it too; a task that
                # resolves relative paths against it would otherwise come back broken.
                $actionArgs = @{ Execute = $exec.Command }
                if ($exec.Arguments) { $actionArgs.Argument = $exec.Arguments }
                if ($exec.WorkingDirectory) { $actionArgs.WorkingDirectory = $exec.WorkingDirectory }
                $restore = New-ScheduledTaskAction @actionArgs
                Set-ScheduledTask -TaskPath $task.TaskPath -TaskName $task.TaskName -Action $restore | Out-Null
            }
            continue
        }

        $actions = @($task.Actions)
        if ($actions.Count -ne 1) { Write-Warning "$full has $($actions.Count) actions; skipped"; continue }
        $limit = if ($task.Settings.ExecutionTimeLimit) { [string]$task.Settings.ExecutionTimeLimit } else { '' }
        $timeout = if ($TimeoutSeconds.ContainsKey($full)) { [int]$TimeoutSeconds[$full] } else { Get-DefaultTimeoutSeconds $limit }
        if ($timeout -lt 0 -or $timeout -gt $script:MaxLauncherTimeoutSeconds) {
            throw "-TimeoutSeconds for $full is $timeout; dtm-task-run accepts 0..$($script:MaxLauncherTimeoutSeconds)"
        }
        $newArgs = ConvertTo-LauncherArguments -Execute $actions[0].Execute -Arguments $actions[0].Arguments -Name $task.TaskName `
            -TimeoutSeconds $timeout -NotifyBus:($NotifyBus -contains $full)
        if ($null -eq $newArgs) { Write-Warning "$full is not a conhost --headless wrapper; skipped"; continue }

        $row = [pscustomobject]@{
            Task = $full; BeforeResult = ('0x{0:X}' -f [uint32]$info.LastTaskResult); TimeoutSeconds = $timeout
            NotifyBus = [bool]($NotifyBus -contains $full); NewArguments = $newArgs
        }
        if ($PSCmdlet.ShouldProcess($full, "wrap with dtm-task-run ($newArgs)")) {
            Export-ScheduledTask -TaskPath $task.TaskPath -TaskName $task.TaskName | Set-Content -LiteralPath $preFile -Encoding utf8
            $new = New-ScheduledTaskAction -Execute $LauncherPath -Argument $newArgs
            if ($actions[0].WorkingDirectory) { $new.WorkingDirectory = $actions[0].WorkingDirectory }
            Set-ScheduledTask -TaskPath $task.TaskPath -TaskName $task.TaskName -Action $new | Out-Null
            Export-ScheduledTask -TaskPath $task.TaskPath -TaskName $task.TaskName | Set-Content -LiteralPath $postFile -Encoding utf8
        }
        $row
    }
}

# Dot-sourcing (for the Pester tests) defines the functions without running the conversion.
if ($MyInvocation.InvocationName -ne '.') { Invoke-Conversion -WhatIf:$WhatIfPreference }
