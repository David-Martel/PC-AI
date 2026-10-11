#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ToolPath = Join-Path $script:RepoRoot 'Tools\Test-ScheduledTaskHealth.ps1'
}

Describe 'Scheduled task inspection metadata regression' -Tag 'InspectionMetadata' {
    BeforeAll {
        $script:FixturePath = Join-Path $TestDrive 'task-health-fixture.ps1'
        # Only the external Task Scheduler / OS reads are replaced. The child
        # executes the actual diagnostic, including its classifier and exit.
        @'
param(
    [string]$ToolPath,
    [string]$Mode,
    [string]$TaskState,
    [uint32]$ResultCode,
    [string]$ReportPath,
    [int]$EnableFailOnIssue,
    [int]$EnableDryRun,
    [int]$ExpectedDormant
)
function Get-CimInstance {
    [CmdletBinding()]
    param([string]$ClassName)
    if ($ClassName -ne 'Win32_OperatingSystem') { throw 'Unexpected OS query' }
    [pscustomobject]@{ LastBootUpTime = (Get-Date).AddHours(-4) }
}
function Get-ScheduledTask {
    [CmdletBinding()]
    param()
    if ($Mode -eq 'Empty') { return }
    if ($Mode -eq 'EnumerationThrow') { throw 'Fixture enumeration denied' }
    [pscustomobject]@{
        TaskName = 'FixtureTask'
        TaskPath = '\'
        State = $TaskState
        Actions = @([pscustomobject]@{
            Execute = 'pwsh.exe'
            Arguments = '-File C:\codedev\fixture\healthcheck.ps1'
        })
        Triggers = @()
    }
}
function Get-ScheduledTaskInfo {
    [CmdletBinding()]
    param([Parameter(ValueFromPipeline)]$InputObject)
    process {
        if ($InputObject.TaskName -ne 'FixtureTask') { throw 'Unexpected task inspection' }
        if ($Mode -eq 'Throw') { throw 'Fixture task metadata denied' }
        if ($Mode -eq 'Null') { return }
        [pscustomobject]@{
            NextRunTime = [datetime]::MinValue
            LastRunTime = (Get-Date).AddMinutes(-1)
            LastTaskResult = $ResultCode
            NumberOfMissedRuns = 0
        }
    }
}
$parameters = @{
    OutputJson = $ReportPath
    FailOnIssue = [bool]$EnableFailOnIssue
    DryRun = [bool]$EnableDryRun
}
if ($ExpectedDormant) { $parameters.Expected = @('FixtureTask') }
& $ToolPath @parameters
exit $LASTEXITCODE
'@ | Set-Content -LiteralPath $script:FixturePath -Encoding utf8

        function Invoke-TaskHealthFixture {
            param(
                [string]$Mode = 'Healthy',
                [string]$TaskState = 'Ready',
                [uint32]$ResultCode = 0,
                [int]$EnableFailOnIssue = 1,
                [switch]$DryRun,
                [switch]$ExpectedDormant
            )
            $reportPath = Join-Path $TestDrive "$([guid]::NewGuid())/report.json"
            $startInfo = [Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            foreach ($argument in @(
                    '-NoLogo', '-NoProfile', '-File', $script:FixturePath,
                    '-ToolPath', $script:ToolPath, '-Mode', $Mode,
                    '-TaskState', $TaskState, '-ResultCode', [string]$ResultCode,
                    '-ReportPath', $reportPath, '-EnableFailOnIssue', [string]$EnableFailOnIssue,
                    '-EnableDryRun', [string][int]$DryRun.IsPresent,
                    '-ExpectedDormant', [string][int]$ExpectedDormant.IsPresent)) {
                $startInfo.ArgumentList.Add($argument)
            }
            $child = [Diagnostics.Process]::new()
            $child.StartInfo = $startInfo
            try {
                [void]$child.Start()
                $stdout = $child.StandardOutput.ReadToEndAsync()
                $stderr = $child.StandardError.ReadToEndAsync()
                if (-not $child.WaitForExit(20000)) {
                    $child.Kill($true)
                    $child.WaitForExit()
                    throw 'Task-health fixture exceeded its 20 second deadline'
                }
                $reportExists = Test-Path -LiteralPath $reportPath
                [pscustomobject]@{
                    ExitCode              = $child.ExitCode
                    Output                = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
                    ReportExists          = $reportExists
                    ReportDirectoryExists = Test-Path -LiteralPath (Split-Path -Parent $reportPath)
                    Report                = if ($reportExists) { Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json } else { $null }
                }
            }
            finally {
                $child.Dispose()
            }
        }
    }

    It 'fails closed when enabled task metadata is <Mode>' -ForEach @(
        @{ Mode = 'Throw'; Reason = 'Fixture task metadata denied' }
        @{ Mode = 'Null'; Reason = 'returned no metadata' }
    ) {
        $result = Invoke-TaskHealthFixture -Mode $Mode
        $result.ExitCode | Should -Be 1
        $result.Report.Failed | Should -Be 1
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Failed'
        $row.InspectionStatus | Should -Be 'Unavailable'
        $row.LastResult | Should -BeNullOrEmpty
        $row.ResultText | Should -Be 'No result recorded'
        ($row.Reasons -join ' ') | Should -Match $Reason
        $row.Staleness | Should -Be 'not evaluated (inspection unavailable)'
    }

    It 'keeps valid successful metadata healthy' {
        $result = Invoke-TaskHealthFixture
        $result.ExitCode | Should -Be 0
        $result.Report.Healthy | Should -Be 1
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Healthy'
        $row.InspectionStatus | Should -Be 'Available'
        $row.LastResult | Should -Be 0
        $row.ResultText | Should -Be 'Success'
        $row.Reasons | Should -BeNullOrEmpty
    }

    It 'keeps the active running informational code healthy' {
        $result = Invoke-TaskHealthFixture -TaskState Running -ResultCode 267009
        $result.ExitCode | Should -Be 0
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Healthy'
        $row.LastResult | Should -Be 267009
        $row.ResultText | Should -Be 'Currently running'
    }

    It 'retains the full unsigned failure code even for a running task' {
        $result = Invoke-TaskHealthFixture -TaskState Running -ResultCode 3221225786
        $result.ExitCode | Should -Be 1
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Failed'
        $row.LastResult | Should -Be 3221225786
        $row.ResultText | Should -Match '0xC000013A'
        ($row.Reasons -join ' ') | Should -Match 'Last run 0xC000013A'
    }

    It 'retains Disabled precedence when inspection is <Mode>' -ForEach @(
        @{ Mode = 'Throw' }; @{ Mode = 'Null' }
    ) {
        $result = Invoke-TaskHealthFixture -Mode $Mode -TaskState Disabled
        $result.ExitCode | Should -Be 0
        $result.Report.Disabled | Should -Be 1
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Disabled'
        $row.InspectionStatus | Should -Be 'Unavailable'
        ($row.Reasons -join ' ') | Should -Match 'inspection unavailable'
    }

    It 'retains Expected precedence over <TaskState> when inspection is <Mode>' -ForEach @(
        @{ Mode = 'Throw'; TaskState = 'Ready' }
        @{ Mode = 'Null'; TaskState = 'Ready' }
        @{ Mode = 'Throw'; TaskState = 'Disabled' }
        @{ Mode = 'Null'; TaskState = 'Disabled' }
    ) {
        $result = Invoke-TaskHealthFixture -Mode $Mode -TaskState $TaskState -ExpectedDormant
        $result.ExitCode | Should -Be 0
        $result.Report.Ignored | Should -Be 1
        $row = @($result.Report.Tasks)[0]
        $row.Status | Should -Be 'Ignored'
        $row.InspectionStatus | Should -Be 'Unavailable'
    }

    It 'reports unavailable inspection without changing the optional exit contract' {
        $result = Invoke-TaskHealthFixture -Mode Throw -EnableFailOnIssue 0
        $result.ExitCode | Should -Be 0
        @($result.Report.Tasks)[0].Status | Should -Be 'Failed'
        $result.Report.Failed | Should -Be 1
    }

    It 'fails enumeration clearly for <Mode>' -ForEach @(
        @{ Mode = 'Empty'; Reason = 'no tasks' }
        @{ Mode = 'EnumerationThrow'; Reason = 'Fixture enumeration denied' }
    ) {
        $result = Invoke-TaskHealthFixture -Mode $Mode
        $result.ExitCode | Should -Be 2
        $result.ReportExists | Should -BeFalse
        $result.ReportDirectoryExists | Should -BeFalse
        $result.Output | Should -Match $Reason
    }

    It 'DryRun retains issue exit semantics without writing a report or directory for <Mode>' -ForEach @(
        @{ Mode = 'Throw'; ExpectedExit = 1 }
        @{ Mode = 'Null'; ExpectedExit = 1 }
        @{ Mode = 'Healthy'; ExpectedExit = 0 }
    ) {
        $result = Invoke-TaskHealthFixture -Mode $Mode -DryRun
        $result.ExitCode | Should -Be $ExpectedExit
        $result.ReportExists | Should -BeFalse
        $result.ReportDirectoryExists | Should -BeFalse
        if ($ExpectedExit -eq 1) { $result.Output | Should -Match 'inspection unavailable' }
    }
}

Describe 'Test-ScheduledTaskHealth' {

    Context 'Static checks' {

        It 'exists' {
            Test-Path -LiteralPath $script:ToolPath | Should -BeTrue
        }

        It 'parses without error' {
            $errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseFile(
                $script:ToolPath, [ref]$null, [ref]$errors)
            @($errors) | Should -BeNullOrEmpty
        }

        It 'is read-only - never registers, starts, stops or unregisters a task' {
            $content = Get-Content -LiteralPath $script:ToolPath -Raw
            # This tool is a diagnostic. If it ever gains a mutating cmdlet the
            # watchdog that runs it as SYSTEM becomes a liability.
            foreach ($forbidden in 'Register-ScheduledTask', 'Unregister-ScheduledTask',
                'Start-ScheduledTask', 'Stop-ScheduledTask',
                'Set-ScheduledTask', 'Disable-ScheduledTask') {
                $content | Should -Not -Match $forbidden
            }
        }

        It 'never casts a task result to [int] - 0xC000013A overflows Int32' {
            # Regression guard. LastTaskResult is a uint32; [int] throws on the
            # killed-process codes this tool exists to surface.
            $content = Get-Content -LiteralPath $script:ToolPath -Raw
            $content | Should -Not -Match '\[int\]\$info\.LastTaskResult'
        }

        It 'never enumerates tasks with -ErrorAction SilentlyContinue (must fail closed)' {
            # Regression guard. Silencing the enumeration turns a Task Scheduler
            # provider fault into an empty set, which then reports "all healthy"
            # and exits 0 under -FailOnIssue - a health check that cannot report
            # its own failure. It must fail loudly instead.
            $content = Get-Content -LiteralPath $script:ToolPath -Raw
            $content | Should -Not -Match 'Get-ScheduledTask\s+-ErrorAction\s+SilentlyContinue'
            $content | Should -Match 'Get-ScheduledTask\s+-ErrorAction\s+Stop'
        }

        It 'supports --help without touching the system' {
            $out = & pwsh -NoLogo -NoProfile -File $script:ToolPath '--help' 2>&1
            ($out | Out-String) | Should -Match 'SYNOPSIS'
        }
    }

    # Every assertion below queries the live Task Scheduler through CIM, which
    # exists only on Windows. Without this guard the whole context fails on a
    # Linux runner at the first Get-Content of a JSON the script never wrote.
    Context 'Live behaviour' -Skip:(-not $IsWindows -or -not (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue)) {

        BeforeAll {
            # Assert against the JSON report, not -PassThru. The script ends in
            # `exit`, so it cannot be dot-sourced into the Pester session, and
            # `pwsh -File` serialises objects to display text across the process
            # boundary - $row.Status would be $null. The JSON is the real
            # machine-readable contract, so test that.
            $script:JsonPath = Join-Path ([IO.Path]::GetTempPath()) "sth-live-$([guid]::NewGuid()).json"
            & pwsh -NoLogo -NoProfile -File $script:ToolPath -OutputJson $script:JsonPath *> $null
            $script:Summary = Get-Content -LiteralPath $script:JsonPath -Raw | ConvertFrom-Json
            $script:Report = @($script:Summary.Tasks)
        }

        AfterAll {
            Remove-Item -LiteralPath $script:JsonPath -ErrorAction SilentlyContinue
        }

        It 'returns task objects' {
            @($script:Report).Count | Should -BeGreaterThan 0
        }

        It 'reports counts that add up to the total' {
            $s = $script:Summary
            ($s.Healthy + $s.Failed + $s.Stalled + $s.Disabled + $s.Ignored) |
                Should -Be $s.TotalTasks
        }

        It 'classifies every task into a known status' {
            $valid = 'Healthy', 'Failed', 'Stalled', 'Disabled', 'Ignored'
            foreach ($row in @($script:Report)) {
                $valid | Should -Contain $row.Status
            }
        }

        It 'assigns every task an owner' {
            foreach ($row in @($script:Report)) {
                'Local', 'Vendor' | Should -Contain $row.Owner
            }
        }

        It 'POSITIVE CONTROL: detects a task that is known to be failing' {
            # The whole point of this tool. If this assertion cannot fail, the
            # tool is decorative. We assert that a task whose LastTaskResult is
            # a genuine nonzero action exit code is reported as Failed - not
            # that any *specific* task is broken, so the test stays valid once
            # the underlying tasks are repaired.
            $knownBad = @(Get-ScheduledTask -ErrorAction SilentlyContinue |
                    Where-Object { $_.TaskPath -notlike '\Microsoft\*' } |
                    ForEach-Object {
                        $i = $_ | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
                        if ($null -ne $i -and $null -ne $i.LastTaskResult) {
                            $code = [int64]$i.LastTaskResult
                            # 0x1 is unambiguous: a generic action failure, never
                            # one of the SCHED_S_* informational codes.
                            if ($code -eq 0x1) { "$($_.TaskPath)$($_.TaskName)" }
                        }
                    })

            if (@($knownBad).Count -eq 0) {
                Set-ItResult -Skipped -Because 'no task on this machine currently reports 0x1'
                return
            }

            foreach ($name in $knownBad) {
                $row = @($script:Report | Where-Object { "$($_.TaskPath)$($_.TaskName)" -eq $name })[0]
                if ($null -eq $row) { continue }
                # Disabled/Ignored take precedence by design; only assert on the
                # ones still in scope.
                if ($row.Status -in 'Disabled', 'Ignored') { continue }
                $row.Status | Should -Be 'Failed' -Because "$name last returned 0x1"
            }
        }

        It 'exits 0 without -FailOnIssue even when issues exist' {
            & pwsh -NoLogo -NoProfile -File $script:ToolPath -LocalOnly *> $null
            $LASTEXITCODE | Should -Be 0
        }

        It 'DryRun writes no report file' {
            $tmp = Join-Path ([IO.Path]::GetTempPath()) "sth-$([guid]::NewGuid()).json"
            & pwsh -NoLogo -NoProfile -File $script:ToolPath -OutputJson $tmp -DryRun *> $null
            Test-Path -LiteralPath $tmp | Should -BeFalse
        }

        It 'writes valid JSON when asked' {
            $tmp = Join-Path ([IO.Path]::GetTempPath()) "sth-$([guid]::NewGuid()).json"
            try {
                & pwsh -NoLogo -NoProfile -File $script:ToolPath -OutputJson $tmp *> $null
                Test-Path -LiteralPath $tmp | Should -BeTrue
                { Get-Content -LiteralPath $tmp -Raw | ConvertFrom-Json } | Should -Not -Throw
            }
            finally {
                Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
            }
        }
    }
}
