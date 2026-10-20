#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: Test-ScheduledTaskHealth.ps1 grew a third axis that reads the status JSON
# written by dtm-task-run.exe (docs/scheduled-tasks/LAUNCHER.md). Behind `conhost --headless` every
# task reported 0x0, so these tests pin the cases that matter: a nonzero or timed-out last run, a
# launcher killed mid-run, a stale status file, and -- just as important -- the cases that must NOT
# be flagged (direct tasks, a task that has not run through the launcher yet).

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ToolPath = Join-Path $script:RepoRoot 'Tools\Test-ScheduledTaskHealth.ps1'
    $script:FixturePath = Join-Path $TestDrive 'launcher-health-fixture.ps1'

    # Only the OS reads are replaced; the child runs the real script, classifier and exit.
    @'
param(
    [string]$ToolPath,
    [string[]]$StatusDir,
    [string]$ReportPath,
    [string]$TaskState = 'Ready',
    [string]$ActionKind = 'launcher',
    [string]$ActionArguments = '--name "Fixture Task" --timeout 290 -- pwsh.exe -File C:\codedev\fixture\job.ps1',
    [int]$LastRunMinutesAgo = 2,
    [string]$TriggerKind = 'Interval15',
    [int]$FailOnIssue = 1
)
function Get-CimInstance {
    [CmdletBinding()] param([string]$ClassName)
    [pscustomobject]@{ LastBootUpTime = (Get-Date).AddHours(-4) }
}
function Get-ScheduledTask {
    [CmdletBinding()] param()
    $action = switch ($ActionKind) {
        'launcher' { [pscustomobject]@{ Execute = 'C:\Users\x\.local\share\pc-ai\deployed\bin\dtm-task-run.exe'; Arguments = $ActionArguments } }
        'conhost'  { [pscustomobject]@{ Execute = 'C:\WINDOWS\System32\conhost.exe'; Arguments = '--headless "pwsh.exe" -File C:\codedev\fixture\job.ps1' } }
        default    { [pscustomobject]@{ Execute = 'pwsh.exe'; Arguments = '-File C:\codedev\fixture\job.ps1' } }
    }
    $triggers = switch ($TriggerKind) {
        'Interval15' { @([pscustomobject]@{ CimClass = [pscustomobject]@{ CimClassName = 'MSFT_TaskTimeTrigger' }; Repetition = [pscustomobject]@{ Interval = 'PT15M' } }) }
        'Logon'      { @([pscustomobject]@{ CimClass = [pscustomobject]@{ CimClassName = 'MSFT_TaskLogonTrigger' }; Repetition = $null }) }
        default      { @() }
    }
    [pscustomobject]@{ TaskName = 'FixtureTask'; TaskPath = '\'; State = $TaskState; Actions = @($action); Triggers = $triggers }
}
function Get-ScheduledTaskInfo {
    [CmdletBinding()] param([Parameter(ValueFromPipeline)]$InputObject)
    process {
        [pscustomobject]@{
            NextRunTime = (Get-Date).AddMinutes(10)
            LastRunTime = (Get-Date).AddMinutes(-$LastRunMinutesAgo)
            LastTaskResult = [uint32]0
            NumberOfMissedRuns = 0
        }
    }
}
$parameters = @{ OutputJson = $ReportPath; FailOnIssue = [bool]$FailOnIssue }
if ($PSBoundParameters.ContainsKey('StatusDir')) { $parameters.StatusDir = $StatusDir }
& $ToolPath @parameters
exit $LASTEXITCODE
'@ | Set-Content -LiteralPath $script:FixturePath -Encoding utf8

    function New-StatusFile {
        param(
            [string]$Dir, [string]$Stem = 'Fixture_Task', [string]$State = 'completed', [int]$ExitCode = 0,
            [bool]$TimedOut = $false, [int]$EndMinutesAgo = 2, [int]$LauncherPid = 0
        )
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        $end = (Get-Date).ToUniversalTime().AddMinutes(-$EndMinutesAgo)
        $running = ($State -eq 'running')
        $hex = '0x{0:X}' -f [uint32]$ExitCode
        $body = [ordered]@{
            schemaVersion = 1; task = 'Fixture Task'; state = $State; pid = $LauncherPid
            startUtc = $end.AddSeconds(-5).ToString('o')
            endUtc = if ($running) { $null } else { $end.ToString('o') }
            durationMs = if ($running) { $null } else { 5000 }
            exitCode = if ($running) { $null } else { $ExitCode }
            exitCodeHex = if ($running) { $null } else { $hex }
            timedOut = $TimedOut; cmdline = 'pwsh.exe -File job.ps1'; stderrTail = @(); host = 'fixture'
            logPath = 'C:\fixture\Fixture_Task.log'
        }
        $body | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Dir "$Stem.json") -Encoding utf8
    }

    function Invoke-LauncherHealth {
        param([string[]]$StatusDir, [hashtable]$Fixture = @{}, [string]$LocalAppData)
        $report = Join-Path $TestDrive "$([guid]::NewGuid()).json"
        $psi = [Diagnostics.ProcessStartInfo]::new((Get-Command pwsh -CommandType Application | Select-Object -First 1).Source)
        $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
        if ($LocalAppData) { $psi.Environment['LOCALAPPDATA'] = $LocalAppData }
        foreach ($a in @('-NoLogo', '-NoProfile', '-File', $script:FixturePath, '-ToolPath', $script:ToolPath, '-ReportPath', $report)) { $psi.ArgumentList.Add($a) }
        # `pwsh -File` binds one value per array parameter, so several directories travel ';'-joined.
        if ($StatusDir) { $psi.ArgumentList.Add('-StatusDir'); $psi.ArgumentList.Add(($StatusDir -join ';')) }
        foreach ($k in $Fixture.Keys) { $psi.ArgumentList.Add("-$k"); $psi.ArgumentList.Add([string]$Fixture[$k]) }
        $child = [Diagnostics.Process]::Start($psi)
        try {
            $out = $child.StandardOutput.ReadToEndAsync(); $err = $child.StandardError.ReadToEndAsync()
            if (-not $child.WaitForExit(30000)) { $child.Kill($true); throw 'fixture exceeded 30 s' }
            [pscustomobject]@{
                ExitCode = $child.ExitCode
                Output   = $out.GetAwaiter().GetResult() + $err.GetAwaiter().GetResult()
                Summary  = if (Test-Path -LiteralPath $report) { Get-Content -LiteralPath $report -Raw | ConvertFrom-Json } else { $null }
                Row      = if (Test-Path -LiteralPath $report) { @((Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).Tasks)[0] } else { $null }
            }
        }
        finally { $child.Dispose() }
    }
}

Describe 'Test-ScheduledTaskHealth launcher status axis' -Tag 'LauncherStatus' {

    BeforeEach { $script:Dir = Join-Path $TestDrive "status-$([guid]::NewGuid())" }

    It 'treats a clean, recent launcher run as healthy' {
        New-StatusFile -Dir $script:Dir
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.ExitCode | Should -Be 0
        $r.Row.Status | Should -Be 'Healthy'
        $r.Row.Launcher | Should -Be 'dtm-task-run'
        $r.Row.LauncherStatus | Should -Be 'completed'
        $r.Row.LauncherExit | Should -Be '0x0'
    }

    It 'fails a task whose launcher recorded a nonzero exit even though Task Scheduler says 0x0' {
        # The exact blind spot of conhost --headless, now visible through the status file.
        New-StatusFile -Dir $script:Dir -ExitCode 3
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.ExitCode | Should -Be 1
        $r.Row.Status | Should -Be 'Failed'
        $r.Row.LauncherExit | Should -Be '0x3'
        ($r.Row.Reasons -join ' ') | Should -Match 'Launcher recorded exit 0x3'
    }

    It 'fails a task the launcher killed on timeout and says so' {
        New-StatusFile -Dir $script:Dir -ExitCode 1460 -TimedOut $true
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.Row.Status | Should -Be 'Failed'
        ($r.Row.Reasons -join ' ') | Should -Match 'timeout'
        ($r.Row.Reasons -join ' ') | Should -Match '0x5B4'
    }

    It 'flags a launcher that was killed mid-run (status still running, task not running)' {
        New-StatusFile -Dir $script:Dir -State running -EndMinutesAgo 3 -LauncherPid 2147480000
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ TaskState = 'Ready' }
        $r.Row.Status | Should -Be 'Failed'
        ($r.Row.Reasons -join ' ') | Should -Match 'launcher was killed mid-run'
    }

    It 'does not flag a task that is genuinely still running' {
        New-StatusFile -Dir $script:Dir -State running -EndMinutesAgo 1 -LauncherPid 2147480000
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ TaskState = 'Running' }
        $r.Row.Status | Should -Be 'Healthy'
        $r.Row.LauncherStatus | Should -Be 'running'
    }

    It 'reports a status file older than the task cadence as Stalled' {
        # 15 minute cadence, default StaleFactor 3 => 45 minutes. The record is 5 hours old.
        New-StatusFile -Dir $script:Dir -EndMinutesAgo 300
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.Row.Status | Should -Be 'Stalled'
        ($r.Row.Reasons -join ' ') | Should -Match 'Launcher status is .* h old but the cadence'
    }

    It 'flags a status record that predates the last Task Scheduler run by more than five minutes' {
        New-StatusFile -Dir $script:Dir -EndMinutesAgo 20 -ExitCode 0
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ LastRunMinutesAgo = 2 }
        $r.Row.Status | Should -Be 'Stalled'
        ($r.Row.Reasons -join ' ') | Should -Match "launcher's latest record"
    }

    It 'does not flag a missing status file (a freshly migrated task has not run through the launcher yet)' {
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.ExitCode | Should -Be 0
        $r.Row.Status | Should -Be 'Healthy'
        $r.Row.LauncherStatus | Should -Be 'missing'
    }

    It 'fails an unreadable status file instead of silently passing it' {
        New-Item -ItemType Directory -Path $script:Dir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $script:Dir 'Fixture_Task.json') -Value '{ not json' -Encoding utf8
        $r = Invoke-LauncherHealth -StatusDir $script:Dir
        $r.Row.Status | Should -Be 'Failed'
        $r.Row.LauncherStatus | Should -Be 'unreadable'
    }

    It 'finds the status file for a task id with spaces (spaces become underscores)' {
        New-StatusFile -Dir $script:Dir -Stem 'Fixture_Task' -ExitCode 7
        (Invoke-LauncherHealth -StatusDir $script:Dir).Row.LauncherExit | Should -Be '0x7'
    }

    It 'defaults to %LOCALAPPDATA%\dtm-tasks\status when -StatusDir is not given' {
        $root = Join-Path $TestDrive "appdata-$([guid]::NewGuid())"
        New-StatusFile -Dir (Join-Path $root 'dtm-tasks\status') -ExitCode 6
        $r = Invoke-LauncherHealth -LocalAppData $root
        $r.Row.LauncherExit | Should -Be '0x6'
        $r.Row.Status | Should -Be 'Failed'
    }

    It 'uses the newest status file when several directories are searched' {
        $old = Join-Path $TestDrive "old-$([guid]::NewGuid())"
        $new = Join-Path $TestDrive "new-$([guid]::NewGuid())"
        New-StatusFile -Dir $old -ExitCode 9
        (Get-Item (Join-Path $old 'Fixture_Task.json')).LastWriteTimeUtc = (Get-Date).AddHours(-1).ToUniversalTime()
        New-StatusFile -Dir $new -ExitCode 0
        $r = Invoke-LauncherHealth -StatusDir @($old, $new)
        $r.Row.LauncherExit | Should -Be '0x0'
    }

    It 'reads --name only from the launcher options, never from the child command line' {
        New-StatusFile -Dir $script:Dir -Stem 'Real' -ExitCode 0
        New-StatusFile -Dir $script:Dir -Stem 'decoy' -ExitCode 9
        $args = '--name Real -- tool.exe --name decoy'
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ ActionArguments = $args }
        $r.Row.Status | Should -Be 'Healthy'
        $r.Row.LauncherExit | Should -Be '0x0'
    }

    It 'never flags a direct task, even when an unrelated failing status file shares its name' {
        New-StatusFile -Dir $script:Dir -Stem 'FixtureTask' -ExitCode 5
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ ActionKind = 'direct' }
        $r.Row.Launcher | Should -Be 'direct'
        $r.Row.LauncherStatus | Should -Be 'n/a'
        $r.Row.Status | Should -Be 'Healthy'
    }

    It 'names conhost --headless tasks in the summary without failing them' {
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ ActionKind = 'conhost' }
        $r.ExitCode | Should -Be 0
        $r.Row.Launcher | Should -Be 'conhost'
        $r.Summary.ConhostWrapped | Should -Be 1
        $r.Output | Should -Match "conhost --headless"
        $r.Output | Should -Match 'FixtureTask'
    }

    It 'does not evaluate launcher status for a disabled task' {
        New-StatusFile -Dir $script:Dir -ExitCode 3
        $r = Invoke-LauncherHealth -StatusDir $script:Dir -Fixture @{ TaskState = 'Disabled' }
        $r.Row.Status | Should -Be 'Disabled'
        $r.Row.LauncherStatus | Should -Be 'n/a'
    }

    It 'POSITIVE CONTROL: the same task goes from Healthy to Failed only because the status file changed' {
        New-StatusFile -Dir $script:Dir -ExitCode 0
        $ok = Invoke-LauncherHealth -StatusDir $script:Dir
        New-StatusFile -Dir $script:Dir -ExitCode 2
        $bad = Invoke-LauncherHealth -StatusDir $script:Dir
        $ok.Row.Status | Should -Be 'Healthy'
        $bad.Row.Status | Should -Be 'Failed'
    }
}
