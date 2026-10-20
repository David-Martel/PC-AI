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

The child command line is carried over character for character, so quoting is preserved. Triggers,
principal, settings and run history are untouched (Set-ScheduledTask updates in place). Before any
change the task XML is exported to <BackupDir>\pre-launcher__<task>.xml, and afterwards to
after-launcher__<task>.xml.

--timeout defaults to the task's own ExecutionTimeLimit minus 10 s, so the launcher reports a hang as
exit 1460 with a status file and event-log entry instead of Task Scheduler killing it silently
(LastTaskResult 0x41306). Tasks with no limit (daemons, runners) get no --timeout.

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
Restore each task's action from its pre-launcher__ XML backup instead.

.PARAMETER WhatIf
Show the new action without changing anything.

.EXAMPLE
./Convert-TaskToLauncher.ps1 -TaskName '\LspmuxServer','\Bitwarden-Archive-Refresh' -NotifyBus '\Bitwarden-Archive-Refresh' -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string[]]$TaskName,
    [string[]]$NotifyBus = @(),
    [hashtable]$TimeoutSeconds = @{},
    [string]$LauncherPath = (Join-Path $env:USERPROFILE '.local\share\pc-ai\deployed\bin\dtm-task-run.exe'),
    [string]$BackupDir = (Join-Path $env:USERPROFILE '.local\state\task-backups'),
    [switch]$Revert
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

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
    <# ExecutionTimeLimit minus 10 s; 0 (no launcher timeout) when the task has no finite limit. #>
    param([string]$ExecutionTimeLimit)
    if ([string]::IsNullOrWhiteSpace($ExecutionTimeLimit)) { return 0 }
    $span = [System.Xml.XmlConvert]::ToTimeSpan($ExecutionTimeLimit)
    if ($span -le [TimeSpan]::Zero) { return 0 }
    [int][math]::Max(1, $span.TotalSeconds - 10)
}

function Invoke-Conversion {
    [CmdletBinding(SupportsShouldProcess)]
    param()
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
        $info = $task | Get-ScheduledTaskInfo
        $backup = Get-TaskBackupName $task.TaskPath $task.TaskName
        $preFile = Join-Path $BackupDir "pre-launcher__$backup.xml"
        $postFile = Join-Path $BackupDir "after-launcher__$backup.xml"

        if ($Revert) {
            if (-not (Test-Path -LiteralPath $preFile)) { throw "No backup to restore for $full ($preFile)" }
            if ($PSCmdlet.ShouldProcess($full, "restore action from $preFile")) {
                [xml]$old = Get-Content -LiteralPath $preFile -Raw
                $ns = [System.Xml.XmlNamespaceManager]::new($old.NameTable); $ns.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
                $exec = $old.SelectSingleNode('//t:Actions/t:Exec', $ns)
                $restore = New-ScheduledTaskAction -Execute $exec.Command -Argument $exec.Arguments
                Set-ScheduledTask -TaskPath $task.TaskPath -TaskName $task.TaskName -Action $restore | Out-Null
            }
            continue
        }

        $actions = @($task.Actions)
        if ($actions.Count -ne 1) { Write-Warning "$full has $($actions.Count) actions; skipped"; continue }
        $limit = if ($task.Settings.ExecutionTimeLimit) { [string]$task.Settings.ExecutionTimeLimit } else { '' }
        $timeout = if ($TimeoutSeconds.ContainsKey($full)) { [int]$TimeoutSeconds[$full] } else { Get-DefaultTimeoutSeconds $limit }
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
