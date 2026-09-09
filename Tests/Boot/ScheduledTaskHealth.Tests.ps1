#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ToolPath = Join-Path $script:RepoRoot 'Tools\Test-ScheduledTaskHealth.ps1'
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
            } finally {
                Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
            }
        }
    }
}
