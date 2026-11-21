#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
<#
.SYNOPSIS
    Unit tests for Modules/PC-AI.Reliability (fleet failure ledger, sweeper, playbooks).

.DESCRIPTION
    Why these tests exist: on 2026-11-21 the fleet audit found tasks that exited 0 while failing,
    tasks that never ran, alert paths that never delivered, runners offline for days and a disk
    watchdog that read a missing drive as 0 GB. Each detector here has a positive control (it
    fires on the failure it exists for) and a negative control (it stays quiet when healthy),
    because a check that cannot fail is not a check.

    Every external call (agent-bus, gh, Task Scheduler, TLS, Discord) is mocked through the one
    function the module routes it through; nothing here touches the network or the scheduler.
    Contract under test: docs/reliability/fleet-ledger.md.
#>

BeforeAll {
    $script:ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1'
    Import-Module $script:ModulePath -Force
    $script:Calls = New-Object System.Collections.ArrayList
    $script:Fake = @{}

    function script:Reset-FleetFake {
        $script:Calls.Clear()
        $script:Fake = @{ Send = 0; Spool = 0; Replay = 0; ReplayErr = ''; ReadOut = '[]'; Task = $null }
    }
    function script:Get-BusCall([string]$Verb) {
        return @($script:Calls | Where-Object { @($_.Args)[0] -eq $Verb })
    }
    function script:Get-ArgValue($Call, [string]$Flag) {
        $a = @($Call.Args)
        $vals = @()
        for ($i = 0; $i -lt $a.Count - 1; $i++) { if ($a[$i] -eq $Flag) { $vals += $a[$i + 1] } }
        return $vals
    }
    function script:Read-TestLedger {
        $p = Join-Path $env:FLEET_RELIABILITY_HOME 'ledger\ledger.jsonl'
        if (-not (Test-Path -LiteralPath $p)) { return @() }
        return @(Get-Content -LiteralPath $p | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
    }
    function script:New-TestObservation {
        param([string]$Status = 'failed', [string]$Severity = 'medium', [int]$ResolveAfter = 1, [string]$Name = 'demo', [string]$Check = 'run')
        New-FleetObservation -Source task -Name $Name -Check $Check -Status $Status -Severity $Severity -Message "status $Status" -ResolveAfter $ResolveAfter
    }
}

Describe 'PC-AI.Reliability' {
    BeforeEach {
        $env:FLEET_RELIABILITY_HOME = Join-Path $TestDrive ('home-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        $env:FLEET_HOST = 'testhost'
        $env:AGENT_BUS_EXE = 'C:\fake\agent-bus.exe'
        $env:FLEET_NOW = '2026-11-21T12:00:00Z'
        Remove-Item Env:FLEET_LEDGER_MAX_BYTES -ErrorAction SilentlyContinue
        Remove-Item Env:DISCORD_BOT_TOKEN -ErrorAction SilentlyContinue
        Reset-FleetFake
        Mock -ModuleName PC-AI.Reliability Invoke-FleetProcess {
            [void]$script:Calls.Add([pscustomobject]@{ FilePath = $FilePath; Args = @($ArgumentList) })
            $verb = @($ArgumentList)[0]
            $r = [pscustomobject]@{ ExitCode = 0; StdOut = ''; StdErr = ''; TimedOut = $false }
            if ($script:Fake.Task -and $FilePath -eq $script:Fake.Task.FilePath) {
                $r.ExitCode = $script:Fake.Task.ExitCode; $r.StdOut = $script:Fake.Task.StdOut; return $r
            }
            switch ($verb) {
                'send' { $r.ExitCode = $script:Fake.Send }
                'spool-send' { $r.ExitCode = $script:Fake.Spool }
                'spool-replay' { $r.ExitCode = $script:Fake.Replay; $r.StdErr = $script:Fake.ReplayErr }
                'read' { $r.StdOut = $script:Fake.ReadOut }
            }
            return $r
        }
    }

    Describe 'Ledger events (fleet-ledger/1)' -Tag 'Portable' {
        It 'writes one line carrying every contract field, with nulls instead of missing keys' {
            # Protects the cross-host contract: the bash emitter and the digest read the same fields.
            $ev = New-FleetLedgerEvent -Source task -Name '\x\demo' -Check run -Kind run -Status failed -Severity medium -Message 'boom' -ExitCode 3
            $null = Write-FleetLedgerEvent -LedgerEvent $ev
            $line = @(Read-TestLedger)
            $line.Count | Should -Be 1
            $names = $line[0].PSObject.Properties.Name
            foreach ($f in 'schema', 'event_id', 'ts', 'host', 'source', 'name', 'check', 'kind', 'status', 'severity', 'start', 'end', 'exit_code', 'message', 'log_path', 'incident_key', 'remediation', 'details') {
                $names | Should -Contain $f
            }
            $line[0].schema | Should -Be 'fleet-ledger/1'
            $line[0].host | Should -Be 'testhost'
            $line[0].exit_code | Should -Be 3
            $line[0].event_id | Should -Match '^testhost-20261121T120000000Z-[0-9a-f]{8}$'
        }

        It 'rotates at the size limit and keeps exactly five old files' {
            # Protects disk: the ledger must not grow without bound on a host that fails all day.
            $env:FLEET_LEDGER_MAX_BYTES = '300'
            1..12 | ForEach-Object { $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name "n$_" -Message ('x' * 200)) }
            $dir = Join-Path $env:FLEET_RELIABILITY_HOME 'ledger'
            1..5 | ForEach-Object { Test-Path (Join-Path $dir "ledger.$_.jsonl") | Should -BeTrue }
            Test-Path (Join-Path $dir 'ledger.6.jsonl') | Should -BeFalse
        }

        It 'keeps run events local and forwards incidents to agent-bus with the contract shape' {
            # Detects noise (every run on the bus) and mis-shaped messages the hub would reject.
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind run)
            @(Get-BusCall 'send').Count | Should -Be 0
            $ev = New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Severity high -Message 'it broke' -IncidentKey 'testhost|task|demo|run'
            $null = Write-FleetLedgerEvent -LedgerEvent $ev
            $send = @(Get-BusCall 'send')
            $send.Count | Should -Be 1
            (Get-ArgValue $send[0] '--topic') | Should -Be 'fleet-failures'
            (Get-ArgValue $send[0] '--thread-id') | Should -Be 'fleet-failures'
            (Get-ArgValue $send[0] '--schema') | Should -Be 'finding'
            (Get-ArgValue $send[0] '--from-agent') | Should -Be 'fleet-ledger-testhost'
            (Get-ArgValue $send[0] '--priority') | Should -Be 'high'
            (Get-ArgValue $send[0] '--tag') | Should -Contain 'host:testhost'
            $body = [string](Get-ArgValue $send[0] '--body')
            $body | Should -Match '^FINDING: \[failed\] testhost task/demo \(run\): it broke'
            $body | Should -Match 'SEVERITY: HIGH'
            ((Get-ArgValue $send[0] '--metadata') | ConvertFrom-Json).event_id | Should -Be $ev.event_id
        }

        It 'sends resolutions with the status schema and a RESOLVED prefix' {
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind incident-resolved -Message 'recovered')
            $send = @(Get-BusCall 'send')[0]
            (Get-ArgValue $send '--schema') | Should -Be 'status'
            [string](Get-ArgValue $send '--body') | Should -Match '^RESOLVED: '
        }

        It 'keeps a long finding body under 1900 characters and keeps the SEVERITY line' {
            # Detects truncation that would drop SEVERITY: and fail the hub's finding schema.
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Message ('y' * 1990))
            $body = [string](Get-ArgValue @(Get-BusCall 'send')[0] '--body')
            $body.Length | Should -BeLessOrEqual 1900
            $body | Should -Match 'SEVERITY: INFO'
        }

        It 'records each forward with its per-transport delivery outcome' {
            # Detects silent alert loss: before this, a failed delivery was only a line in a text log.
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Severity high)
            $d = @(Read-TestLedger | Where-Object { $_.source -eq 'alert' -and $_.check -eq 'delivery' })
            $d.Count | Should -Be 1
            $d[0].message | Should -Be 'bus=sent discord=skipped(no recipient)'
            $d[0].status | Should -Be 'ok'
            $script:Fake.Send = 1; $script:Fake.Spool = 1
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo2 -Kind incident-open -Status failed -Severity high) -WarningAction SilentlyContinue
            (@(Read-TestLedger | Where-Object { $_.source -eq 'alert' }))[-1].status | Should -Be 'failed'
        }

        It 'spools the message when the bus send fails' {
            # Protects alert delivery across a hub outage: the spool is replayed by the next sweep.
            $script:Fake.Send = 1
            $r = Send-FleetBusEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed)
            $r | Should -Be 'spooled'
            $spool = @(Get-BusCall 'spool-send')
            $spool.Count | Should -Be 1
            (Split-Path -Leaf (Get-ArgValue $spool[0] '--spool')) | Should -Be 'bus-spool.ndjson'
            (Get-ArgValue $spool[0] '--topic') | Should -Be 'fleet-failures'
        }

        It 'still records the local line when both bus and spool fail' {
            $script:Fake.Send = 1; $script:Fake.Spool = 1
            $null = Write-FleetLedgerEvent -LedgerEvent (New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed) -WarningAction SilentlyContinue
            @(Read-TestLedger | Where-Object { $_.kind -eq 'incident-open' }).Count | Should -Be 1
        }

        It 'quotes argv so JSON metadata survives CreateProcess' {
            # Detects the PowerShell 5.1 quoting trap: agent-bus --metadata is JSON full of quotes.
            InModuleScope PC-AI.Reliability {
                # A bare token keeps its trailing backslash; a quoted one must double it before the closing quote.
                ConvertTo-FleetCommandLine -Arguments @('plain', 'two words', '{"a":"b"}', 'C:\dir\', 'C:\my dir\', '') |
                    Should -Be 'plain "two words" "{\"a\":\"b\"}" C:\dir\ "C:\my dir\\" ""'
            }
        }
    }

    Describe 'Spool replay' -Tag 'Portable' {
        It 'renames the live spool before replaying and removes it after success' {
            # Protects messages appended during a replay: spool-replay deletes the whole file it read.
            $dir = Join-Path $env:FLEET_RELIABILITY_HOME 'spool'
            New-Item -ItemType Directory -Force $dir | Out-Null
            Set-Content -LiteralPath (Join-Path $dir 'bus-spool.ndjson') -Value '{"sender":"a"}'
            $s = Invoke-FleetSpoolReplay
            $s.Replayed | Should -Be 1
            $replay = @(Get-BusCall 'spool-replay')
            (Get-ArgValue $replay[0] '--spool') | Should -BeLike '*bus-spool.*.replaying.ndjson'
            @(Get-ChildItem $dir -Filter 'bus-spool*').Count | Should -Be 0
        }

        It 'leaves the renamed spool for the next sweep while the bus is down' {
            $dir = Join-Path $env:FLEET_RELIABILITY_HOME 'spool'
            New-Item -ItemType Directory -Force $dir | Out-Null
            Set-Content -LiteralPath (Join-Path $dir 'bus-spool.ndjson') -Value '{"sender":"a"}'
            $script:Fake.Replay = 1
            (Invoke-FleetSpoolReplay).Pending | Should -Be 1
            @(Get-ChildItem $dir -Filter '*.replaying.ndjson').Count | Should -Be 1
        }

        It 'quarantines a spool that does not parse instead of failing every replay forever' {
            $dir = Join-Path $env:FLEET_RELIABILITY_HOME 'spool'
            New-Item -ItemType Directory -Force $dir | Out-Null
            Set-Content -LiteralPath (Join-Path $dir 'bus-spool.ndjson') -Value 'not json'
            $script:Fake.Replay = 1; $script:Fake.ReplayErr = 'line 1: JSON parse error'
            (Invoke-FleetSpoolReplay).Quarantined | Should -Be 1
            @(Get-ChildItem (Join-Path $dir 'quarantine')).Count | Should -Be 1
        }
    }

    Describe 'Incident state' -Tag 'Portable' {
        It 'opens once and does not re-alert while the failure persists' {
            # Detects the repeat-every-run noise pattern; one incident, one alert.
            (Submit-FleetObservation (New-TestObservation)).Transition | Should -Be 'opened'
            (Submit-FleetObservation (New-TestObservation)).Transition | Should -Be 'still-open'
            @(Get-BusCall 'send').Count | Should -Be 1
        }

        It 'sends a reminder after 24 hours open' {
            $null = Submit-FleetObservation (New-TestObservation)
            $env:FLEET_NOW = '2026-11-22T12:30:00Z'
            (Submit-FleetObservation (New-TestObservation)).Transition | Should -Be 'reminder'
            $kinds = @(Read-TestLedger | ForEach-Object { $_.kind })
            $kinds | Should -Contain 'incident-reminder'
        }

        It 're-alerts at once when severity rises' {
            $null = Submit-FleetObservation (New-TestObservation -Severity medium)
            (Submit-FleetObservation (New-TestObservation -Severity high)).Transition | Should -Be 'reminder'
        }

        It 'resolves only after ResolveAfter consecutive good observations' {
            $null = Submit-FleetObservation (New-TestObservation -ResolveAfter 2)
            (Submit-FleetObservation (New-TestObservation -Status ok -ResolveAfter 2)).Transition | Should -Be 'still-open'
            (Submit-FleetObservation (New-TestObservation -Status ok -ResolveAfter 2)).Transition | Should -Be 'resolved'
            $res = @(Get-BusCall 'send' | Where-Object { (Get-ArgValue $_ '--body') -match '^RESOLVED' })
            $res.Count | Should -Be 1
        }

        It 'stays quiet for a healthy subject that never failed' {
            (Submit-FleetObservation (New-TestObservation -Status ok)).Transition | Should -Be 'quiet'
            @(Read-TestLedger).Count | Should -Be 0
        }

        It 'treats skipped as neutral (planned DRAIN), neither opening nor resolving' {
            $null = Submit-FleetObservation (New-TestObservation)
            (Submit-FleetObservation (New-TestObservation -Status skipped)).Transition | Should -Be 'neutral'
            (Get-FleetIncidentState)['testhost|task|demo|run'].state | Should -Be 'open'
        }

        It 'declares flapping after three opens in six hours and then suppresses forwarding' {
            # Detects flap storms: one flapping alert instead of an open/resolve pair every sweep.
            $t = @()
            foreach ($m in 0, 20, 40) {
                $env:FLEET_NOW = ('2026-11-21T12:{0:D2}:00Z' -f $m)
                $t += (Submit-FleetObservation (New-TestObservation)).Transition
                $env:FLEET_NOW = ('2026-11-21T12:{0:D2}:00Z' -f ($m + 10))
                $null = Submit-FleetObservation (New-TestObservation -Status ok)
            }
            $t | Should -Be @('opened', 'opened', 'flapping')
            $env:FLEET_NOW = '2026-11-21T13:30:00Z'
            $before = @(Get-BusCall 'send').Count
            (Submit-FleetObservation (New-TestObservation)).Transition | Should -Be 'suppressed'
            @(Get-BusCall 'send').Count | Should -Be $before
        }

        It 'resolves an open incident whose subject is no longer observed by a detector that ran' {
            $null = Submit-FleetObservation (New-TestObservation -Name 'gone-task')
            $r = @(Resolve-FleetUnobservedIncidents -ObservedKeys @() -CompletedSources @('task'))
            $r.Count | Should -Be 1
            $null = Resolve-FleetUnobservedIncidents -ObservedKeys @() -CompletedSources @('task')
            (Get-FleetIncidentState)['testhost|task|gone-task|run'].state | Should -Be 'resolved'
        }

        It 'does not resolve an incident when its detector did not complete' {
            $null = Submit-FleetObservation (New-TestObservation -Name 'kept')
            @(Resolve-FleetUnobservedIncidents -ObservedKeys @() -CompletedSources @('disk')).Count | Should -Be 0
        }
    }

    Describe 'Scheduled task detector' -Tag 'Portable' {
        BeforeEach {
            $script:Report = @{
                Tasks = @(
                    @{ TaskPath = '\'; TaskName = 'broken'; Owner = 'Local'; State = 'Ready'; Status = 'Failed'; LastResult = 1; ResultText = '0x1'; Reasons = @('Last run 0x1'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' }
                    @{ TaskPath = '\'; TaskName = 'keepalive'; Owner = 'Local'; State = 'Running'; Status = 'Failed'; LastResult = 2147946720; ResultText = '0x800710E0'; Reasons = @('Last run 0x800710E0'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' }
                    @{ TaskPath = '\'; TaskName = 'launcher-failed'; Owner = 'Local'; State = 'Ready'; Status = 'Failed'; LastResult = 0; ResultText = 'Success'; Reasons = @('launcher exit 3'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' }
                    @{ TaskPath = '\'; TaskName = 'never'; Owner = 'Local'; State = 'Ready'; Status = 'Stalled'; LastResult = $null; ResultText = 'No result'; Reasons = @('never run'); LastRunTime = $null; Staleness = 'never run' }
                    @{ TaskPath = '\'; TaskName = 'off'; Owner = 'Local'; State = 'Disabled'; Status = 'Disabled'; LastResult = $null; Reasons = @(); LastRunTime = $null; Staleness = '' }
                    @{ TaskPath = '\Vendor\'; TaskName = 'noise'; Owner = 'Vendor'; State = 'Ready'; Status = 'Failed'; LastResult = 1; Reasons = @('x'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = '' }
                    @{ TaskPath = '\'; TaskName = 'healthy'; Owner = 'Local'; State = 'Ready'; Status = 'Healthy'; LastResult = 0; ResultText = 'Success'; Reasons = @(); LastRunTime = '2026-11-21T11:00:00Z'; Staleness = 'fresh' }
                )
            }
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskHealthData { return $script:Report }
            Mock -ModuleName PC-AI.Reliability Get-FleetScheduledTaskData { return @() }
            Mock -ModuleName PC-AI.Reliability Get-FleetInteractiveUserData { return @() }
            $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json')
        }

        It 'reports a non-zero last result as a failed run' {
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\broken' -and $_.Check -eq 'run' })
            $o[0].Status | Should -Be 'failed'
        }

        It 'honours per-task benign codes for forwarders that end at logoff' {
            # Negative control for long-running forwarders that end with 0xC000013A at logoff by design.
            $script:Report.Tasks[0].LastResult = 3221225786
            $script:Policy.task_ok_codes_by_task = @{ '\broken' = @('0xC000013A') }
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\broken' -and $_.Check -eq 'run' }))[0].Status | Should -Be 'ok'
            $script:Policy.task_ok_codes_by_task = @{}
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\broken' -and $_.Check -eq 'run' }))[0].Status | Should -Be 'failed'
        }

        It 'treats a benign IgnoreNew result (0x800710E0) as healthy' {
            # Negative control: keepalive tasks read like failures in every naive scan (audit #12).
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\keepalive' -and $_.Check -eq 'run' })
            $o[0].Status | Should -Be 'ok'
        }

        It 'does not let a benign nonzero code mask a launcher failure reported alongside it' {
            # Detects the masking found by the second review of #346: an allowlisted result code turned every
            # Failed report into ok, including one that also carried a launcher timeout.
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskHealthData {
                [pscustomobject]@{ Tasks = @(
                        [pscustomobject]@{ TaskPath = '\'; TaskName = 'keepalive-plus-launcher'; Owner = 'Local'; State = 'Ready'; Status = 'Failed'; LastResult = 2147946720; ResultText = '0x800710E0'; Reasons = @('Last run 0x800710E0 The operator or administrator has refused the request', 'Launcher killed the task on timeout (124) at 2026-11-21 10:00 UTC; log: x'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' },
                        [pscustomobject]@{ TaskPath = '\'; TaskName = 'keepalive-only'; Owner = 'Local'; State = 'Ready'; Status = 'Failed'; LastResult = 2147946720; ResultText = '0x800710E0'; Reasons = @('Last run 0x800710E0 The operator or administrator has refused the request'); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' }) }
            }
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'run' })
            ($o | Where-Object Name -EQ '\keepalive-plus-launcher').Status | Should -Be 'failed'
            ($o | Where-Object Name -EQ '\keepalive-only').Status | Should -Be 'ok'
        }

        It 'does not let a benign 0x0 mask a launcher-reported failure' {
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\launcher-failed' -and $_.Check -eq 'run' })
            $o[0].Status | Should -Be 'failed'
        }

        It 'reports a task that has never run as never-ran' {
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\never' -and $_.Check -eq 'task-stale' })
            $o[0].Status | Should -Be 'never-ran'
        }

        It 'skips disabled tasks and vendor tasks unless the policy includes them' {
            $all = @(Get-FleetTaskObservations -Policy $script:Policy)
            @($all | Where-Object { $_.Name -in '\off', '\Vendor\noise' }).Count | Should -Be 0
            $script:Policy.task_include = @('\Vendor\*')
            @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\Vendor\noise' }).Count | Should -BeGreaterThan 0
        }

        It 'limits a staged rollout to task_scope when it is set' {
            # Protects a staged install: only the named tasks may open incidents until the owner widens it.
            $script:Policy.task_scope = @('\healthy')
            $names = @(Get-FleetTaskObservations -Policy $script:Policy | ForEach-Object { $_.Name } | Sort-Object -Unique)
            $names | Should -Be @('\healthy')
            $script:Policy.task_scope = @()
            @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Name -eq '\broken' }).Count | Should -BeGreaterThan 0
        }

        It 'flags a task that must be enabled but is Disabled and never ran' {
            # Positive control for an offsite backup task found Disabled and never run.
            $script:Policy.expect_enabled = @('\off')
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'task-expected' })
            $o[0].Status | Should -Be 'never-ran'
            $o[0].Message | Should -Match 'Disabled'
        }

        It 'flags an Interactive task when its user has no interactive session' {
            # Positive control: about 25 Interactive tasks on one host silently stopped firing.
            Mock -ModuleName PC-AI.Reliability Get-FleetScheduledTaskData { @([pscustomobject]@{ FullName = '\healthy'; State = 'Ready'; LogonType = 'Interactive'; UserId = 'HOST\david' }) }
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'task-interactive-no-session' })
            $o[0].Status | Should -Be 'degraded'
            Mock -ModuleName PC-AI.Reliability Get-FleetInteractiveUserData { @('host\david') }
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'task-interactive-no-session' })
            $o[0].Status | Should -Be 'ok'
        }

        It 'turns exit 0 plus a failure line in the latest log into a failed log-rule' {
            # Positive control for a database pull that returned 0x0 with "integrity check did not pass".
            $log = Join-Path $TestDrive 'pull.log'
            Set-Content -LiteralPath $log -Value @('start', '[ERROR] integrity check did not pass', 'done')
            $script:Policy.log_rules = @(@{ task = '\healthy'; log = $log; patterns = @('integrity check did not pass') })
            $script:Report.Tasks[6].LastRunTime = (Get-Date).ToUniversalTime().AddMinutes(-5).ToString('o')
            $o = @(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'log-rule' })
            $o[0].Status | Should -Be 'failed'
            $o[0].LogPath | Should -Be $log
        }

        It 'uses the default patterns when a rule names none, instead of matching every line' {
            # Detects the @($null) trap: an absent patterns list became one empty regex that matches anything.
            $log = Join-Path $TestDrive 'nopatterns.log'
            Set-Content -LiteralPath $log -Value 'all good'
            $script:Policy.log_rules = @(@{ task = '\healthy'; log = $log })
            $script:Report.Tasks[6].LastRunTime = (Get-Date).ToUniversalTime().AddMinutes(-5).ToString('o')
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'log-rule' }))[0].Status | Should -Be 'ok'
            Add-Content -LiteralPath $log -Value 'export STALE since 2026-11-12'
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'log-rule' }))[0].Status | Should -Be 'failed'
        }

        It 'stays quiet when the latest log is clean, and skips a log older than the last run' {
            $log = Join-Path $TestDrive 'clean.log'
            Set-Content -LiteralPath $log -Value 'all good'
            $script:Policy.log_rules = @(@{ task = '\healthy'; log = $log; patterns = @('ERROR') })
            $script:Report.Tasks[6].LastRunTime = (Get-Date).ToUniversalTime().AddMinutes(-5).ToString('o')
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'log-rule' }))[0].Status | Should -Be 'ok'
            (Get-Item $log).LastWriteTimeUtc = (Get-Date).ToUniversalTime().AddDays(-2)
            (@(Get-FleetTaskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'log-rule' }))[0].Status | Should -Be 'skipped'
        }
    }

    Describe 'Disk detector' -Tag 'Portable' {
        BeforeEach { $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json') }

        It 'reports a drive below its floor as degraded/high' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDriveData { @([pscustomobject]@{ Name = 'T'; FreeBytes = 10GB; TotalBytes = 500GB }) }
            $script:Policy.disks = @(@{ name = 'T'; min_free_gb = 15 })
            $o = @(Get-FleetDiskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'disk-free' })
            $o[0].Status | Should -Be 'degraded'
            $o[0].Severity | Should -Be 'high'
            $o[0].Playbook | Should -Be 'disk-low'
        }

        It 'reports a healthy drive as ok' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDriveData { @([pscustomobject]@{ Name = 'T'; FreeBytes = 100GB; TotalBytes = 500GB }) }
            (@(Get-FleetDiskObservations -Policy $script:Policy))[0].Status | Should -Be 'ok'
        }

        It 'fails closed on a missing required drive instead of reading it as 0 GB' {
            # Positive control for the t-free-watchdog bug: a missing T: must be a failure, not "0 GB".
            Mock -ModuleName PC-AI.Reliability Get-FleetDriveData { @() }
            $script:Policy.disks = @(@{ name = 'T'; min_free_gb = 15; required = $true })
            $o = @(Get-FleetDiskObservations -Policy $script:Policy)
            $o[0].Status | Should -Be 'failed'
            $o[0].Message | Should -Match 'missing'
            $o[0].Playbook | Should -BeNullOrEmpty
        }

        It 'projects when a falling drive reaches its floor' {
            # 49 GB falling 4 GB/h with a 15 GB floor reaches it in 8.5 h.
            $now = [datetime]::Parse('2026-11-21T12:00:00Z').ToUniversalTime()
            $samples = 0..6 | ForEach-Object { @{ ts = $now.AddHours(-6 + $_).ToString('o'); free = [int64]((49 + 4 * (6 - $_)) * 1GB) } }
            $t = Get-FleetDiskTrend -Samples $samples -Now $now -WindowHours 6 -FloorBytes 15GB
            [math]::Round($t.EtaHours, 1) | Should -Be 8.5
            [math]::Round($t.SlopeBytesPerHour / 1GB, 1) | Should -Be -4
        }

        It 'gives no ETA for a rising drive and no trend from too few samples' {
            $now = [datetime]::Parse('2026-11-21T12:00:00Z').ToUniversalTime()
            $rising = 0..4 | ForEach-Object { @{ ts = $now.AddHours(-4 + $_).ToString('o'); free = [int64]((20 + $_) * 1GB) } }
            (Get-FleetDiskTrend -Samples $rising -Now $now -FloorBytes 15GB).EtaHours | Should -BeNullOrEmpty
            Get-FleetDiskTrend -Samples @($rising[0], $rising[1]) -Now $now -FloorBytes 15GB | Should -BeNullOrEmpty
        }

        It 'raises a disk-trend incident from persisted samples across sweeps' {
            $script:Free = 40GB
            Mock -ModuleName PC-AI.Reliability Get-FleetDriveData { @([pscustomobject]@{ Name = 'T'; FreeBytes = $script:Free; TotalBytes = 500GB }) }
            $script:Policy.disks = @(@{ name = 'T'; min_free_gb = 15 })
            $last = $null
            foreach ($h in 0..4) {
                $env:FLEET_NOW = ('2026-11-21T{0:D2}:00:00Z' -f (8 + $h))
                $script:Free = [int64]((40 - 4 * $h) * 1GB)
                $last = @(Get-FleetDiskObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'disk-trend' })
            }
            $last[0].Status | Should -Be 'degraded'
            $last[0].Message | Should -Match '^T: hits 15 GB in '
        }
    }

    Describe 'Runner and workflow detectors' -Tag 'Portable' {
        BeforeEach {
            $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json')
            $script:Policy.runner_scopes = @('repos/o/r')
            $script:Policy.drain = @{ bus_agent = 'vct-nightly'; max_minutes = 90; runner_patterns = @('hostA-*') }
            Mock -ModuleName PC-AI.Reliability Invoke-FleetGhApi { @(@{ runners = @(@{ name = 'hostA-1'; status = 'offline'; busy = $false; labels = @() }, @{ name = 'hostB-1'; status = 'online'; busy = $true; labels = @() }) }) }
        }

        It 'reports an offline runner as failed and an online one as ok' {
            # Positive control for a runner that sat offline for three days unnoticed.
            Mock -ModuleName PC-AI.Reliability Get-FleetDrainState { @{ Active = $false; Minutes = 0 } }
            $o = @(Get-FleetRunnerObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'runner-offline' })
            ($o | Where-Object { $_.Name -eq 'o/r/hostA-1' }).Status | Should -Be 'failed'
            ($o | Where-Object { $_.Name -eq 'o/r/hostB-1' }).Status | Should -Be 'ok'
        }

        It 'treats a matching runner as expected-down during a DRAIN inside its budget' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDrainState { @{ Active = $true; Minutes = 20 } }
            $o = @(Get-FleetRunnerObservations -Policy $script:Policy)
            ($o | Where-Object { $_.Name -eq 'o/r/hostA-1' }).Status | Should -Be 'skipped'
            ($o | Where-Object { $_.Check -eq 'drain-overrun' }).Status | Should -Be 'ok'
        }

        It 'reports a DRAIN that outlives its budget, and the runner behind it' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDrainState { @{ Active = $true; Minutes = 140 } }
            $o = @(Get-FleetRunnerObservations -Policy $script:Policy)
            ($o | Where-Object { $_.Check -eq 'drain-overrun' }).Status | Should -Be 'failed'
            ($o | Where-Object { $_.Name -eq 'o/r/hostA-1' }).Status | Should -Be 'failed'
        }

        It 'reads the drain lifecycle case-sensitively from real vct-nightly messages' {
            # Regression from live data (2026-11-21): an ALERT containing "restart" was read as an active DRAIN.
            $msgs = @(
                @{ timestamp_utc = '2026-11-21T14:20:53Z'; body = 'vct-nightly START on dtm-work: DRAIN, then compact vigil-ci-t' }
                @{ timestamp_utc = '2026-11-21T14:56:35Z'; body = 'vct-nightly ended without a compact (rc=7) but restored cleanly' }
                @{ timestamp_utc = '2026-11-21T15:14:22Z'; body = 'ALERT vct-nightly on dtm-work: the previous run died before its restore. Check the STOP/DRAIN files before a restart.' }
            )
            # Built element by element: Windows PowerShell 5.1 wraps a piped array as {value, Count}.
            $toJson = { param($items) '[' + ((@($items) | ForEach-Object { $_ | ConvertTo-Json -Compress }) -join ',') + ']' }
            $script:Fake.ReadOut = & $toJson $msgs
            $s = InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Get-FleetDrainState -Policy $P }
            $s.Active | Should -BeFalse
            $s.Alert | Should -Match '^ALERT vct-nightly'
            $script:Fake.ReadOut = & $toJson $msgs[0..1]
            (InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Get-FleetDrainState -Policy $P }).Active | Should -BeFalse
            $script:Fake.ReadOut = & $toJson @($msgs[0])
            $env:FLEET_NOW = '2026-11-21T14:40:53Z'
            $a = InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Get-FleetDrainState -Policy $P }
            $a.Active | Should -BeTrue
            $a.Minutes | Should -Be 20
        }

        It 'surfaces a drain-agent ALERT as a degraded observation, not as healthy' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDrainState { @{ Active = $false; Minutes = 0; Alert = 'ALERT vct-nightly: died before its restore' } }
            $o = @(Get-FleetRunnerObservations -Policy $script:Policy | Where-Object { $_.Check -eq 'drain-overrun' })
            $o[0].Status | Should -Be 'degraded'
            $o[0].Message | Should -Match 'state unknown'
        }

        It 'reports an expected workflow whose newest run is too old as stale' {
            # Positive control for a QA workflow that created no runs for over three weeks.
            $script:Policy.workflow_expectations = @(@{ repo = 'o/r'; workflow = 'qa.yml'; max_age_days = 3 })
            Mock -ModuleName PC-AI.Reliability Invoke-FleetGhApi { @(@{ workflow_runs = @(@{ created_at = '2026-10-29T00:00:00Z'; conclusion = 'success' }) }) }
            (@(Get-FleetWorkflowObservations -Policy $script:Policy))[0].Status | Should -Be 'stale'
            Mock -ModuleName PC-AI.Reliability Invoke-FleetGhApi { @(@{ workflow_runs = @(@{ created_at = '2026-11-21T06:00:00Z'; conclusion = 'success' }) }) }
            (@(Get-FleetWorkflowObservations -Policy $script:Policy))[0].Status | Should -Be 'ok'
            # The newest-run query must not paginate (it walked the whole run history and timed out live).
            Should -Invoke -ModuleName PC-AI.Reliability Invoke-FleetGhApi -ParameterFilter { $NoPaginate -and $Path -like '*runs?per_page=1' }
        }
    }

    Describe 'Postgres, orphan process and certificate detectors' -Tag 'Portable' {
        BeforeEach { $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json') }

        It 'reports connections at or above warn_pct of max_connections from the conf file' {
            # Positive control for the agent-bus Postgres exhausted by leaked MCP processes.
            $conf = Join-Path $TestDrive 'postgresql.conf'
            Set-Content -LiteralPath $conf -Value @('#max_connections = 10', 'max_connections = 40')
            $script:Policy.postgres = @(@{ name = 'bus'; port = 5300; conf_files = @($conf); reaper = '{profile}\reaper.ps1' })
            Mock -ModuleName PC-AI.Reliability Get-FleetTcpListenerConnectionCount { 36 }
            $o = (@(Get-FleetPostgresObservations -Policy $script:Policy))[0]
            $o.Status | Should -Be 'degraded'
            $o.Message | Should -Match '^36/40'
            $o.Playbook | Should -Be 'reap-orphans'
            Mock -ModuleName PC-AI.Reliability Get-FleetTcpListenerConnectionCount { 10 }
            (@(Get-FleetPostgresObservations -Policy $script:Policy))[0].Status | Should -Be 'ok'
        }

        It 'flags a postgres conf changed after the server started (restart pending)' {
            # Positive control for a max_connections raise that sat unapplied: the conf said 100, the server ran 40.
            $data = Join-Path $TestDrive 'pgdata'
            New-Item -ItemType Directory -Force $data | Out-Null
            Set-Content (Join-Path $data 'postmaster.pid') '1234'
            Set-Content (Join-Path $data 'postgresql.conf') 'max_connections = 100'
            (Get-Item (Join-Path $data 'postmaster.pid')).LastWriteTimeUtc = [datetime]::UtcNow.AddHours(-2)
            $script:Policy.postgres = @(@{ name = 'bus'; port = 5300; data_dir = $data })
            Mock -ModuleName PC-AI.Reliability Get-FleetTcpListenerConnectionCount { 32 }
            $o = @(Get-FleetPostgresObservations -Policy $script:Policy)
            ($o | Where-Object { $_.Check -eq 'pg-config-pending' }).Status | Should -Be 'degraded'
            # Never a false "32/100 ok" while the live limit may still be 40.
            ($o | Where-Object { $_.Check -eq 'pg-connections' }).Status | Should -Be 'skipped'
            $script:Policy.postgres = @(@{ name = 'bus'; port = 5300; data_dir = $data; max_connections = 40 })
            $o = @(Get-FleetPostgresObservations -Policy $script:Policy)
            ($o | Where-Object { $_.Check -eq 'pg-connections' }).Message | Should -Match '^32/40'
            ($o | Where-Object { $_.Check -eq 'pg-connections' }).Status | Should -Be 'degraded'
            $script:Policy.postgres = @(@{ name = 'bus'; port = 5300; data_dir = $data })
            (Get-Item (Join-Path $data 'postmaster.pid')).LastWriteTimeUtc = [datetime]::UtcNow
            (@(Get-FleetPostgresObservations -Policy $script:Policy) | Where-Object { $_.Check -eq 'pg-config-pending' }).Status | Should -Be 'ok'
        }

        It 'finds orphans whose parent died or whose parent PID was reused' {
            $t0 = Get-Date '2026-11-21T10:00:00'
            Mock -ModuleName PC-AI.Reliability Get-FleetProcessData {
                @(
                    [pscustomobject]@{ ProcessId = 10; ParentProcessId = 999; Name = 'litho-qmd-mcp.exe'; CreationDate = $t0 }
                    [pscustomobject]@{ ProcessId = 11; ParentProcessId = 20; Name = 'litho-qmd-mcp.exe'; CreationDate = $t0 }
                    [pscustomobject]@{ ProcessId = 20; ParentProcessId = 1; Name = 'codex.exe'; CreationDate = $t0.AddHours(1) }
                    [pscustomobject]@{ ProcessId = 12; ParentProcessId = 21; Name = 'litho-qmd-mcp.exe'; CreationDate = $t0 }
                    [pscustomobject]@{ ProcessId = 21; ParentProcessId = 1; Name = 'claude.exe'; CreationDate = $t0.AddHours(-1) }
                )
            }
            $script:Policy.orphan_processes = @(@{ pattern = 'litho-qmd-mcp*'; reaper = '{profile}\.machine\qmd\Stop-OrphanQmdMcp.ps1' })
            $o = (@(Get-FleetOrphanProcessObservations -Policy $script:Policy))[0]
            $o.Status | Should -Be 'degraded'
            $o.Message | Should -Match 'PIDs 10,11$'
        }

        It 'grades certificate expiry and fails on a broken chain' {
            $script:Policy.cert_endpoints = @('a.example:443')
            Mock -ModuleName PC-AI.Reliability Get-FleetRemoteCertificate { [pscustomobject]@{ NotAfter = [datetime]::Parse('2026-11-26T00:00:00Z').ToUniversalTime(); PolicyErrors = 'None' } }
            $o = (@(Get-FleetCertObservations -Policy $script:Policy))[0]
            $o.Status | Should -Be 'degraded'; $o.Severity | Should -Be 'high'
            Mock -ModuleName PC-AI.Reliability Get-FleetRemoteCertificate { [pscustomobject]@{ NotAfter = [datetime]::Parse('2027-06-01T00:00:00Z').ToUniversalTime(); PolicyErrors = 'None' } }
            (@(Get-FleetCertObservations -Policy $script:Policy))[0].Status | Should -Be 'ok'
            Mock -ModuleName PC-AI.Reliability Get-FleetRemoteCertificate { [pscustomobject]@{ NotAfter = [datetime]::Parse('2027-06-01T00:00:00Z').ToUniversalTime(); PolicyErrors = 'RemoteCertificateNameMismatch' } }
            (@(Get-FleetCertObservations -Policy $script:Policy))[0].Severity | Should -Be 'critical'
        }
    }

    Describe 'Bounded playbooks' {
        BeforeEach { $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json'); $script:Ran = 0 }

        It 'consumes the budget even when recording the remediation fails' -Tag 'Portable' {
            # Detects the unbounded loop from the second review of #346: the action ran before its attempt was
            # persisted, so a throwing ledger write left the budget unspent and every sweep acted again.
            $script:Ran = 0
            $action = { param($a) $script:Ran++; @{ outcome = 'triggered'; detail = 'x' } }
            Mock -ModuleName PC-AI.Reliability Write-FleetLedgerEvent { throw 'disk full' } -ParameterFilter { $LedgerEvent.kind -eq 'remediation' }
            $null = Invoke-FleetPlaybook -Playbook rerun-task -IncidentKey 'k-ledger' -Policy $script:Policy -Action $action
            Invoke-FleetPlaybook -Playbook rerun-task -IncidentKey 'k-ledger' -Policy $script:Policy -Action $action | Should -Be 'budget-exhausted'
            $script:Ran | Should -Be 1
        }

        It 'does not act when its attempt cannot be reserved' -Tag 'WindowsOnly' {
            # Detects the fail-open from the sixth review of #346: Move-Item had no -ErrorAction Stop, so a state
            # file that could not be replaced left the attempt unrecorded while the action still ran.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows share modes'; return }
            $script:Ran = 0
            $action = { param($a) $script:Ran++; @{ outcome = 'triggered'; detail = 'x' } }
            $path = InModuleScope PC-AI.Reliability { Join-Path (Join-Path (Get-FleetPath -Kind state) 'playbooks') ('{0}.json' -f (Get-FleetShortHash 'p|k-locked')) }
            New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
            Set-Content -LiteralPath $path -Value '{"key":"p|k-locked","attempts":[]}'
            $fs = [IO.File]::Open($path, 'Open', 'Read', 'Read')   # readable, but cannot be replaced
            try { { Invoke-FleetPlaybook -Playbook p -IncidentKey 'k-locked' -Policy $script:Policy -Action $action } | Should -Throw }
            finally { $fs.Dispose() }
            $script:Ran | Should -Be 0
        }

        It 'acts once, then escalates exactly once when the budget is spent' -Tag 'Portable' {
            # Detects runaway remediation: a broken fix must not loop, and a human must hear about it.
            $action = { param($a) $script:Ran++; @{ outcome = 'triggered'; detail = 'x' } }
            Invoke-FleetPlaybook -Playbook rerun-task -IncidentKey 'k' -Policy $script:Policy -Action $action | Should -Be 'triggered'
            Invoke-FleetPlaybook -Playbook rerun-task -IncidentKey 'k' -Policy $script:Policy -Action $action | Should -Be 'budget-exhausted'
            Invoke-FleetPlaybook -Playbook rerun-task -IncidentKey 'k' -Policy $script:Policy -Action $action | Should -Be 'budget-exhausted'
            $script:Ran | Should -Be 1
            @(Read-TestLedger | Where-Object { $_.kind -eq 'escalation' }).Count | Should -Be 1
            @(Read-TestLedger | Where-Object { $_.kind -eq 'remediation' }).Count | Should -Be 1
        }

        It 'backs off between attempts and acts again after the backoff' -Tag 'Portable' {
            $script:Policy.playbooks = @{ budget = 3; window_hours = 24; backoff_minutes = 30 }
            $action = { param($a) $script:Ran++; @{ outcome = 'triggered'; detail = 'x' } }
            $null = Invoke-FleetPlaybook -Playbook p -IncidentKey 'k' -Policy $script:Policy -Action $action
            $env:FLEET_NOW = '2026-11-21T12:10:00Z'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k' -Policy $script:Policy -Action $action | Should -Be 'backoff'
            $env:FLEET_NOW = '2026-11-21T12:45:00Z'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k' -Policy $script:Policy -Action $action | Should -Be 'triggered'
            $script:Ran | Should -Be 2
        }

        It 'does nothing while another sweep holds the playbook lock' -Tag 'Portable' {
            $lockDir = Join-Path $env:FLEET_RELIABILITY_HOME 'locks'
            New-Item -ItemType Directory -Force $lockDir | Out-Null
            $hash = InModuleScope PC-AI.Reliability { Get-FleetShortHash 'p|k' }
            $fs = [IO.File]::Open((Join-Path $lockDir "playbook-$hash.lock"), 'OpenOrCreate', 'ReadWrite', 'None')
            try { Invoke-FleetPlaybook -Playbook p -IncidentKey 'k' -Policy $script:Policy -Action { @{ outcome = 'triggered' } } | Should -Be 'locked' }
            finally { $fs.Dispose() }
        }

        It 're-runs an allowlisted task only when it is not running or disabled' -Tag 'Portable' {
            Mock -ModuleName PC-AI.Reliability Start-FleetScheduledTask { }
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskStateData { 'Running' }
            (InModuleScope PC-AI.Reliability { Invoke-FleetRerunTaskAction -Arguments @{ TaskPath = '\'; TaskName = 't' } }).outcome | Should -Be 'skipped'
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskStateData { 'Ready' }
            (InModuleScope PC-AI.Reliability { Invoke-FleetRerunTaskAction -Arguments @{ TaskPath = '\'; TaskName = 't' } }).outcome | Should -Be 'triggered'
            Should -Invoke -ModuleName PC-AI.Reliability Start-FleetScheduledTask -Times 1 -Exactly
        }

        It 'allows only cache directories for pruning (any OS)' -Tag 'Portable' {
            # Protects user data on Linux too: the same rule must hold with / separators.
            Test-FleetCacheDirAllowed (Join-Path $TestDrive 'a/b/pipcache') | Should -BeTrue
            Test-FleetCacheDirAllowed (Join-Path $TestDrive 'a/Documents/cache') | Should -BeFalse
            Test-FleetCacheDirAllowed (Join-Path $TestDrive 'a/b/plain') | Should -BeFalse
        }

        It 'allows only cache directories for pruning' -Tag 'WindowsOnly' {
            # Protects user data: a policy typo must not turn disk-low into a delete of real files.
            Test-FleetCacheDirAllowed 'C:\Users\x\AppData\Local\uv\cache' | Should -BeTrue
            Test-FleetCacheDirAllowed 'C:\' | Should -BeFalse
            Test-FleetCacheDirAllowed 'C:\Users\x\Documents\cache' | Should -BeFalse
            Test-FleetCacheDirAllowed 'C:\codedev\repo\.cache' | Should -BeFalse
            Test-FleetCacheDirAllowed 'C:\Users\x\AppData\Local\Temp' | Should -BeFalse
            Test-FleetCacheDirAllowed 'relative\cache\dir' | Should -BeFalse
        }

        It 'prunes only files older than the age limit' -Tag 'Portable' {
            $dir = Join-Path $TestDrive 'a\b\pipcache'
            New-Item -ItemType Directory -Force $dir | Out-Null
            $old = Join-Path $dir 'old.bin'; $new = Join-Path $dir 'new.bin'
            Set-Content $old ('x' * 100); Set-Content $new ('y' * 100)
            (Get-Item $old).LastWriteTimeUtc = [datetime]::Parse('2026-11-01T00:00:00Z').ToUniversalTime()
            (Get-Item $new).LastWriteTimeUtc = [datetime]::Parse('2026-11-21T11:00:00Z').ToUniversalTime()
            Invoke-FleetCachePrune -Path $dir -MinAgeDays 7 | Should -BeGreaterThan 0
            Test-Path $old | Should -BeFalse
            Test-Path $new | Should -BeTrue
        }

        It 'never follows a junction or symlink out of the cache, at the root or below it' -Tag 'Portable' {
            # Detects the deletion of user data found by the third review of #346: a cache root (or a directory
            # under it) that is a junction to Documents passed the lexical checks and its targets were pruned.
            $onWin = ($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows
            $linkType = $(if ($onWin) { 'Junction' } else { 'SymbolicLink' })
            $docs = Join-Path $TestDrive 'userdocs'
            New-Item -ItemType Directory -Force $docs | Out-Null
            $precious = Join-Path $docs 'thesis.docx'
            Set-Content $precious 'keep me'
            (Get-Item $precious).LastWriteTimeUtc = [datetime]::Parse('2026-01-01T00:00:00Z').ToUniversalTime()

            $linkedRoot = Join-Path $TestDrive 'a\b\linkcache'
            New-Item -ItemType Directory -Force (Split-Path $linkedRoot) | Out-Null
            New-Item -ItemType $linkType -Path $linkedRoot -Target $docs | Out-Null
            { Invoke-FleetCachePrune -Path $linkedRoot -MinAgeDays 7 } | Should -Throw '*reparse*'
            Test-Path $precious | Should -BeTrue

            $realRoot = Join-Path $TestDrive 'a\b\realcache'
            New-Item -ItemType Directory -Force $realRoot | Out-Null
            $stale = Join-Path $realRoot 'stale.bin'
            Set-Content $stale 'old'
            (Get-Item $stale).LastWriteTimeUtc = [datetime]::Parse('2026-01-01T00:00:00Z').ToUniversalTime()
            New-Item -ItemType $linkType -Path (Join-Path $realRoot 'escape') -Target $docs | Out-Null
            $null = Invoke-FleetCachePrune -Path $realRoot -MinAgeDays 7
            Test-Path $stale | Should -BeFalse
            Test-Path $precious | Should -BeTrue
        }

        It 'requests compaction from the storage lane instead of stopping anything' -Tag 'Portable' {
            $script:Policy.compaction = @{ request_agent = 'storage-lane'; drives = @('T'); task = $null }
            $r = InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Invoke-FleetDiskLowAction -Arguments @{ Drive = 'T'; FreeGb = 12; MinGb = 15 } -Policy $P }
            $r.outcome | Should -Be 'requested'
            $send = @(Get-BusCall 'send')
            (Get-ArgValue $send[0] '--to-agent') | Should -Be 'storage-lane'
            [string](Get-ArgValue $send[0] '--body') | Should -Match 'never stops runners'
        }
    }

    Describe 'Task wrapper (Invoke-FleetTask)' -Tag 'Portable' {
        It 'records exit 0 with "integrity check did not pass" as a failure and exits non-zero' {
            # Positive control for a pull job and a logger that exited 0 while failing.
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 0; StdOut = "pulling`n[ERROR] integrity check did not pass`n" }
            Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe' | Should -Be 1
            $run = @(Read-TestLedger | Where-Object { $_.kind -eq 'run' })[0]
            $run.status | Should -Be 'failed'
            $run.message | Should -Match 'integrity check did not pass'
            @(Read-TestLedger | Where-Object { $_.kind -eq 'incident-open' }).Count | Should -Be 1
            @(Get-BusCall 'send').Count | Should -Be 1
        }

        It 'passes a clean run through as exit 0 and resolves the open incident' {
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 0; StdOut = "bad`nERROR: nope" }
            $null = Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe'
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 0; StdOut = 'all fine' }
            Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe' | Should -Be 0
            @(Read-TestLedger | Where-Object { $_.kind -eq 'incident-resolved' }).Count | Should -Be 1
        }

        It 'keeps a real non-zero exit code' {
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 7; StdOut = '' }
            Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe' | Should -Be 7
        }

        It 'decodes what a tool appends to an existing UTF-16 log' {
            # Detects the hidden failure from the sixth review of #346: the wrapper sought past the byte-order mark
            # and read the appended UTF-16 text as UTF-8, so '[ERROR]' never matched and exit 0 passed.
            $log = Join-Path $TestDrive 'utf16.log'
            [IO.File]::WriteAllText($log, "started`r`n", [Text.Encoding]::Unicode)
            $script:Utf16Log = $log
            Mock -ModuleName PC-AI.Reliability Invoke-FleetProcess {
                if ($FilePath -eq 'C:\fake\utf16job.exe') {
                    $bytes = [Text.Encoding]::Unicode.GetBytes("[ERROR] integrity check did not pass`r`n")
                    $fs = [IO.File]::Open($script:Utf16Log, 'Append', 'Write', 'ReadWrite'); try { $fs.Write($bytes, 0, $bytes.Length) } finally { $fs.Dispose() }
                }
                [pscustomobject]@{ ExitCode = 0; StdOut = ''; StdErr = ''; TimedOut = $false; OutputComplete = $true }
            }
            Invoke-FleetTask -Name '\t\utf16' -FilePath 'C:\fake\utf16job.exe' -LogPath $log | Should -Be 1
        }

        It 'scans a log the job replaced with an equal or larger file from the start' {
            # Detects the hidden failure from the ninth review of #346: the offset was reset only when the log got
            # SHORTER, so a job that replaced its log with a bigger file had its early "[ERROR]" line skipped.
            $log = Join-Path $TestDrive 'replaced.log'
            Set-Content -LiteralPath $log -Value ('old clean line ' * 4)
            $script:ReplacedLog = $log
            Mock -ModuleName PC-AI.Reliability Invoke-FleetProcess {
                if ($FilePath -eq 'C:\fake\replacejob.exe') {
                    Remove-Item -LiteralPath $script:ReplacedLog -Force
                    Set-Content -LiteralPath $script:ReplacedLog -Value ("[ERROR] integrity check did not pass`n" + ('padding ' * 40))
                }
                [pscustomobject]@{ ExitCode = 0; StdOut = ''; StdErr = ''; TimedOut = $false; OutputComplete = $true }
            }
            Invoke-FleetTask -Name '\t\replaced' -FilePath 'C:\fake\replacejob.exe' -LogPath $log | Should -Be 1
        }

        It 'scans only the bytes the tool appended to its own log during this run' {
            $log = Join-Path $TestDrive 'tool.log'
            Set-Content -LiteralPath $log -Value 'STALE from yesterday'
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 0; StdOut = 'ok' }
            Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe' -LogPath $log | Should -Be 0
        }
    }

    Describe 'Sweep orchestration' -Tag 'Portable' {
        It 'turns a throwing detector into a visible detector-error and keeps sweeping' {
            # Detects the silent-pass failure mode: a detector that cannot run must not read as healthy.
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskObservations { throw 'scheduler unavailable' }
            Mock -ModuleName PC-AI.Reliability Get-FleetDiskObservations { @(New-FleetObservation -Source disk -Name 'C:' -Check disk-free -Status ok) }
            $s = Invoke-FleetHealthSweep -Detector task, disk -PolicyPath (Join-Path $TestDrive 'none.json')
            $s.detector_errors | Should -Match 'scheduler unavailable'
            (Get-FleetIncidentState)['testhost|sweeper|task|detector-error'].state | Should -Be 'open'
        }

        It 'runs the playbook for an open incident and not with -NoPlaybooks' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDiskObservations { @(New-FleetObservation -Source disk -Name 'T:' -Check disk-free -Status degraded -Severity high -Playbook 'disk-low' -PlaybookArgs @{ Drive = 'T'; FreeGb = 1; MinGb = 15 }) }
            Mock -ModuleName PC-AI.Reliability Invoke-FleetPlaybookForObservation { 'requested' }
            $null = Invoke-FleetHealthSweep -Detector disk -NoPlaybooks -PolicyPath (Join-Path $TestDrive 'none.json')
            Should -Invoke -ModuleName PC-AI.Reliability Invoke-FleetPlaybookForObservation -Times 0 -Exactly
            $s = Invoke-FleetHealthSweep -Detector disk -PolicyPath (Join-Path $TestDrive 'none.json')
            Should -Invoke -ModuleName PC-AI.Reliability Invoke-FleetPlaybookForObservation -Times 1 -Exactly
            $s.playbooks | Should -Be @('disk-low:T:=requested')
        }
    }

    Describe 'Digest' -Tag 'Portable' {
        It 'builds the fleet digest from every host on agent-bus' {
            # Detects the Windows PowerShell 5.1 trap: ConvertFrom-Json returned the bus array as ONE object,
            # so a fleet digest on 5.1 would have read three hosts' messages as a single malformed one.
            $mk = { param($h, $k, $key, $ts) '{"metadata":' + (@{ schema = 'fleet-ledger/1'; host = $h; kind = $k; incident_key = $key; ts = $ts; severity = 'medium'; source = 'task'; name = 'n'; check = 'run'; message = 'm' } | ConvertTo-Json -Compress) + '}' }
            $script:Fake.ReadOut = '[' + (((& $mk 'h1' 'incident-open' 'h1|task|n|run' '2026-11-21T10:00:00Z'), (& $mk 'h2' 'incident-open' 'h2|task|n|run' '2026-11-21T10:05:00Z'), (& $mk 'h1' 'incident-resolved' 'h1|task|n|run' '2026-11-21T11:00:00Z')) -join ',') + ']'
            $d = Send-FleetDigest -Fleet -NoSend -PolicyPath (Join-Path $TestDrive 'none.json')
            $d.Message | Should -Match 'hosts \[h1 h2\]: opened 2, resolved 1'
            $d.StillOpen | Should -Be 1
            $d.Message | Should -Match 'OPEN \[medium\] h2 task/n'
        }

        It 'reports an incident that is still open after the lookback window as open' {
            # Detects the false healthy digest from the seventh review of #346: open incidents were rebuilt from the
            # window's events only, so an incident opened 30 h ago with no later event read as "still open 0, ok".
            $null = Submit-FleetObservation (New-TestObservation -Name 'old')
            $env:FLEET_NOW = '2026-11-22T18:00:00Z'
            $d = Send-FleetDigest -NoSend -PolicyPath (Join-Path $TestDrive 'none.json')
            $d.StillOpen | Should -Be 1
            $d.Event.status | Should -Be 'degraded'
            $d.Message | Should -Match 'OPEN .*old'
        }

        It 'summarises opened, resolved and still-open incidents and posts a DIGEST status' {
            $null = Submit-FleetObservation (New-TestObservation -Name 'a')
            $null = Submit-FleetObservation (New-TestObservation -Name 'b')
            $null = Submit-FleetObservation (New-TestObservation -Name 'b' -Status ok)
            $d = Send-FleetDigest -PolicyPath (Join-Path $TestDrive 'none.json')
            $d.StillOpen | Should -Be 1
            $d.Message | Should -Match 'opened 2, resolved 1'
            $d.Message | Should -Match 'OPEN \[medium\] testhost task/a'
            $digest = @(Get-BusCall 'send' | Where-Object { (Get-ArgValue $_ '--body') -match '^DIGEST: ' })
            $digest.Count | Should -Be 1
        }
    }

    Describe 'Review fixes and dry-run side effects' -Tag 'Portable' {
        # Each test pins one finding of the independent review of #346, or the AGENTS.md rule that a
        # write-capable script's -DryRun must provably write nothing.
        BeforeEach { $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json') }

        It 'keeps attempt history per incident when several incidents remediate' {
            # Protects playbook budgets: one shared state file under per-incident locks lost attempts.
            $act = { param($x) @{ outcome = 'triggered'; detail = 'x' } }
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k1' -Policy $script:Policy -Action $act | Should -Be 'triggered'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k2' -Policy $script:Policy -Action $act | Should -Be 'triggered'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k1' -Policy $script:Policy -Action $act | Should -Be 'budget-exhausted'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k2' -Policy $script:Policy -Action $act | Should -Be 'budget-exhausted'
            @(Get-ChildItem (Join-Path (Join-Path $env:FLEET_RELIABILITY_HOME 'state') 'playbooks') -Filter '*.json').Count | Should -Be 2
        }

        It 'never reuses a previous task-health report when the child fails' {
            # Detects false recovery: a failed child left the last good report looking current.
            $tsh = Join-Path $TestDrive 'tsh.ps1'; Set-Content -LiteralPath $tsh -Value '# stub'
            $script:Policy.task_health_script = $tsh
            $stateDir = Join-Path $env:FLEET_RELIABILITY_HOME 'state'
            New-Item -ItemType Directory -Force $stateDir | Out-Null
            $report = Join-Path $stateDir 'scheduled-task-health.json'
            Set-Content -LiteralPath $report -Value '{"Tasks":[]}'
            $pw = (Get-Process -Id $PID).Path
            if ($pw -notmatch '(?i)pwsh(\.exe)?$') { $pw = (Get-Command pwsh).Source }
            $script:Fake.Task = @{ FilePath = $pw; ExitCode = 2; StdOut = '' }
            { InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Get-FleetTaskHealthData -Policy $P } } | Should -Throw '*exit 2*'
            Test-Path -LiteralPath $report | Should -BeFalse
            $script:Fake.Task.ExitCode = 0
            { InModuleScope PC-AI.Reliability -Parameters @{ P = $script:Policy } { param($P) Get-FleetTaskHealthData -Policy $P } } | Should -Throw '*report written=False*'
        }

        It 'does not let a sweep resolve an incident that only the wrapper observes' {
            # Detects a false RESOLVED: a wrapped task outside task_scope (or with a -Name that is not the exact
            # task path) is never seen by the task detector, which used to count it as "no longer observed".
            $script:Fake.Task = @{ FilePath = 'C:\fake\job.exe'; ExitCode = 0; StdOut = '[ERROR] still broken' }
            $null = Invoke-FleetTask -Name '\outside\scope' -FilePath 'C:\fake\job.exe'
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskObservations { @() }
            $none = Join-Path $TestDrive 'none.json'
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath $none
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath $none
            (Get-FleetIncidentState)['testhost|task|\outside\scope|run'].state | Should -Be 'open'
        }

        It 'keeps an incident event that could not be written and writes it on the next submission' {
            # Detects the lost alert from the third review of #346: incident state was committed before its event
            # was appended, so a failed append left the incident open with no opening alert until the 24 h
            # reminder (and a resolution could vanish for good).
            # Fault injection without mocks: a directory where ledger.jsonl should be makes the real append throw.
            $ledgerFile = Join-Path (Join-Path $env:FLEET_RELIABILITY_HOME 'ledger') 'ledger.jsonl'
            New-Item -ItemType Directory -Force $ledgerFile | Out-Null
            { $null = Submit-FleetObservation (New-TestObservation -Name 'outbox') } | Should -Not -Throw
            Remove-Item -LiteralPath $ledgerFile -Recurse -Force
            $null = Submit-FleetObservation (New-TestObservation -Name 'other' -Status ok)
            @(Read-TestLedger | Where-Object { $_.kind -eq 'incident-open' -and $_.name -eq 'outbox' }).Count | Should -Be 1
            $null = Submit-FleetObservation (New-TestObservation -Name 'other' -Status ok)
            @(Read-TestLedger | Where-Object { $_.kind -eq 'incident-open' -and $_.name -eq 'outbox' }).Count | Should -Be 1
        }

        It 'does not resolve an incident whose task a narrower task_scope no longer covers' {
            # Detects the false recovery from the fourth review of #346: after a full sweep opened '\a\job', sweeps
            # with task_scope narrowed to '\b\*' completed the task detector without seeing it and resolved it.
            $null = Submit-FleetObservation (New-FleetObservation -Source task -Name '\a\job' -Check run -Status failed -Severity medium -Message 'x')
            $narrow = Join-Path $TestDrive 'narrow.json'
            '{"task_scope":["\\b\\*"]}' | Set-Content -LiteralPath $narrow
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskObservations { @() }
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath $narrow
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath $narrow
            (Get-FleetIncidentState)['testhost|task|\a\job|run'].state | Should -Be 'open'
        }

        It 'treats a running or queued task as neutral, so a retry in progress does not clear the failure' {
            # Detects the premature recovery from the fourth review of #346: 0x41301 (running) and 0x41325 (queued)
            # became ok observations with ResolveAfter 1, so a sweep during the retry resolved the earlier failure.
            $null = Submit-FleetObservation (New-FleetObservation -Source task -Name '\r\retry' -Check run -Status failed -Severity medium -Message 'x')
            Mock -ModuleName PC-AI.Reliability Get-FleetScheduledTaskData { @() }
            Mock -ModuleName PC-AI.Reliability Get-FleetInteractiveUserData { @() }
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskHealthData {
                [pscustomobject]@{ Tasks = @(
                        [pscustomobject]@{ TaskPath = '\r\'; TaskName = 'retry'; Owner = 'Local'; State = 'Running'; Status = 'Healthy'; LastResult = 267009; ResultText = 'running'; Reasons = @(); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' },
                        [pscustomobject]@{ TaskPath = '\r\'; TaskName = 'queued'; Owner = 'Local'; State = 'Ready'; Status = 'Healthy'; LastResult = 267045; ResultText = 'queued'; Reasons = @(); LastRunTime = '2026-11-21T10:00:00Z'; Staleness = 'fresh' }) }
            }
            $o = @(Get-FleetTaskObservations -Policy (Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json')) | Where-Object { $_.Check -eq 'run' })
            ($o | Where-Object Name -EQ '\r\retry').Status | Should -Be 'skipped'
            ($o | Where-Object Name -EQ '\r\queued').Status | Should -Be 'skipped'
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath (Join-Path $TestDrive 'none.json')
            (Get-FleetIncidentState)['testhost|task|\r\retry|run'].state | Should -Be 'open'
        }

        It 'does not read an unreadable log as a clean one' {
            # Detects the false recovery from the fifth review of #346: Get-Content errors were suppressed, so a
            # locked or unreadable log gave no lines and an ok log-rule observation that resolved the incident.
            $log = Join-Path $TestDrive 'job.log'
            Set-Content -LiteralPath $log -Value '[ERROR] still broken'
            # Mocked at the module's own read function: how a mocked cmdlet's non-terminating error reaches the
            # caller differs between hosts (it passed locally and failed in CI), a throw does not.
            Mock -ModuleName PC-AI.Reliability Read-FleetLogTail { throw 'The process cannot access the file because it is being used by another process' }
            $o = @(InModuleScope PC-AI.Reliability -Parameters @{ L = $log } { param($L) Get-FleetLogRuleObservation -Rule @{ log = $L } -Tasks @{} })
            @($o | Where-Object { $_.Check -eq 'log-rule' -and $_.Status -eq 'ok' }).Count | Should -Be 0
            ($o | Where-Object Check -EQ 'log-rule').Status | Should -Be 'skipped'
            ($o | Where-Object Check -EQ 'log-read').Status | Should -Be 'failed'
        }

        It 'reads a log another process holds exclusively as a read failure' -Tag 'WindowsOnly' {
            # Positive control for Read-FleetLogTail against a REAL locked file, not a mock.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows share modes'; return }
            $log = Join-Path $TestDrive 'locked.log'
            Set-Content -LiteralPath $log -Value 'line'
            $fs = [IO.File]::Open($log, 'Open', 'ReadWrite', 'None')
            try { { InModuleScope PC-AI.Reliability -Parameters @{ L = $log } { param($L) Read-FleetLogTail -Path $L } } | Should -Throw }
            finally { $fs.Dispose() }
            InModuleScope PC-AI.Reliability -Parameters @{ L = $log } { param($L) Read-FleetLogTail -Path $L } | Should -Be 'line'
        }

        It 'does not act again when its budget state exists but cannot be read' {
            # Detects the reset budget from the fifth review of #346: an unreadable state file was moved aside and
            # treated as "no attempts yet", so the playbook acted again.
            $script:Ran = 0
            $action = { param($a) $script:Ran++; @{ outcome = 'triggered'; detail = 'x' } }
            $path = InModuleScope PC-AI.Reliability { Join-Path (Join-Path (Get-FleetPath -Kind state) 'playbooks') ('{0}.json' -f (Get-FleetShortHash 'p|k-unreadable')) }
            New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
            Set-Content -LiteralPath $path -Value '{ truncated'
            Invoke-FleetPlaybook -Playbook p -IncidentKey 'k-unreadable' -Policy $script:Policy -Action $action | Should -Be 'state-unreadable'
            $script:Ran | Should -Be 0
            Get-Content -LiteralPath $path -Raw | Should -Match 'truncated'
        }

        It 'keeps expect_enabled, log_rules and interactive checks inside task_scope' {
            # Detects the staged-rollout leak from the sixth review of #346: these loops ignored task_scope and
            # task_exclude, so a rollout limited to one task still opened incidents for unrelated tasks.
            $pol = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json')
            $pol.task_scope = @('\in\*')
            $pol.expect_enabled = @('\out\backup', '\in\backup')
            $outLog = Join-Path $TestDrive 'out.log'; Set-Content -LiteralPath $outLog -Value '[ERROR] x'
            $pol.log_rules = @(@{ task = '\out\job'; log = $outLog })
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskHealthData { [pscustomobject]@{ Tasks = @() } }
            Mock -ModuleName PC-AI.Reliability Get-FleetScheduledTaskData { @([pscustomobject]@{ FullName = '\out\gui'; State = 'Ready'; LogonType = 'Interactive'; UserId = 'someone' }) }
            Mock -ModuleName PC-AI.Reliability Get-FleetInteractiveUserData { @() }
            $o = @(Get-FleetTaskObservations -Policy $pol)
            @($o | Where-Object { $_.Name -like '\out\*' }).Count | Should -Be 0
            @($o | Where-Object { $_.Name -eq '\in\backup' -and $_.Check -eq 'task-expected' }).Count | Should -Be 1
        }

        It 'applies task_scope to the tasks an expect_enabled wildcard matches, not to the wildcard' {
            # Detects the skipped check from the ninth review of #346: with expect_enabled '\backups\*' and task_scope
            # '\backups\offsite-copy', the wildcard itself was out of scope, so the disabled in-scope task was never
            # reported (and two sweeps resolved its incident).
            $pol = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json')
            $pol.task_scope = @('\backups\offsite-copy')
            $pol.expect_enabled = @('\backups\*')
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskHealthData {
                [pscustomobject]@{ Tasks = @(
                        [pscustomobject]@{ TaskPath = '\backups\'; TaskName = 'offsite-copy'; Owner = 'Local'; State = 'Disabled'; Status = 'Disabled'; LastResult = 0; ResultText = 'Success'; Reasons = @(); LastRunTime = '2026-11-20T00:00:00Z'; Staleness = 'fresh' },
                        [pscustomobject]@{ TaskPath = '\backups\'; TaskName = 'local-copy'; Owner = 'Local'; State = 'Disabled'; Status = 'Disabled'; LastResult = 0; ResultText = 'Success'; Reasons = @(); LastRunTime = $null; Staleness = 'never run' }) }
            }
            Mock -ModuleName PC-AI.Reliability Get-FleetScheduledTaskData { @() }
            Mock -ModuleName PC-AI.Reliability Get-FleetInteractiveUserData { @() }
            $o = @(Get-FleetTaskObservations -Policy $pol | Where-Object { $_.Check -eq 'task-expected' })
            ($o | Where-Object Name -EQ '\backups\offsite-copy').Status | Should -Be 'failed'
            @($o | Where-Object Name -EQ '\backups\local-copy').Count | Should -Be 0
        }

        It 'clears a detector-error only when that detector itself runs successfully' {
            # Detects false recovery: sweeps limited to other detectors used to resolve the error.
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskObservations { throw 'scheduler unavailable' }
            Mock -ModuleName PC-AI.Reliability Get-FleetDiskObservations { @() }
            $none = Join-Path $TestDrive 'none.json'
            $null = Invoke-FleetHealthSweep -Detector task, disk -NoPlaybooks -PolicyPath $none
            $null = Invoke-FleetHealthSweep -Detector disk -NoPlaybooks -PolicyPath $none
            $null = Invoke-FleetHealthSweep -Detector disk -NoPlaybooks -PolicyPath $none
            (Get-FleetIncidentState)['testhost|sweeper|task|detector-error'].state | Should -Be 'open'
            Mock -ModuleName PC-AI.Reliability Get-FleetTaskObservations { @() }
            $null = Invoke-FleetHealthSweep -Detector task -NoPlaybooks -PolicyPath $none
            (Get-FleetIncidentState)['testhost|sweeper|task|detector-error'].state | Should -Be 'resolved'
        }

        It 'dry-run sweep returns observations and writes, sends and remediates nothing' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDiskObservations { @(New-FleetObservation -Source disk -Name 'T:' -Check disk-free -Status degraded -Severity high -Playbook 'disk-low' -PlaybookArgs @{ Drive = 'T' }) }
            Mock -ModuleName PC-AI.Reliability Invoke-FleetPlaybookForObservation { 'requested' }
            $o = @(Invoke-FleetHealthSweep -DryRun -Detector disk -PolicyPath (Join-Path $TestDrive 'none.json'))
            $o.Count | Should -Be 1
            $o[0].Status | Should -Be 'degraded'
            Test-Path -LiteralPath $env:FLEET_RELIABILITY_HOME | Should -BeFalse
            $script:Calls.Count | Should -Be 0
            Should -Invoke -ModuleName PC-AI.Reliability Invoke-FleetPlaybookForObservation -Times 0 -Exactly
        }

        It 'dry-run sweep leaves the real disk samples untouched' {
            Mock -ModuleName PC-AI.Reliability Get-FleetDriveData { @([pscustomobject]@{ Name = 'T'; FreeBytes = 10GB; TotalBytes = 500GB }) }
            $stateDir = Join-Path $env:FLEET_RELIABILITY_HOME 'state'
            New-Item -ItemType Directory -Force $stateDir | Out-Null
            $samples = Join-Path $stateDir 'disk-samples.json'
            Set-Content -LiteralPath $samples -Value '{"T":[]}'
            $before = (Get-FileHash -LiteralPath $samples).Hash
            $o = @(Invoke-FleetHealthSweep -DryRun -Detector disk -PolicyPath (Join-Path $TestDrive 'none.json'))
            @($o | Where-Object { $_.Check -eq 'disk-free' }).Count | Should -Be 1
            (Get-FileHash -LiteralPath $samples).Hash | Should -Be $before
            @(Get-ChildItem -LiteralPath $env:FLEET_RELIABILITY_HOME -Recurse -File).Count | Should -Be 1
        }

        It 'dry-run leaves a malformed policy and state file in place' {
            # Detects the preview that moved live files: corrupt-file recovery renamed policy.json and
            # last-sweep.json during a dry run (second review of #346). Since the eighth review an unreadable
            # policy fails the run explicitly; it must still be left exactly where it is.
            $policy = Join-Path $TestDrive 'bad-policy.json'
            Set-Content -LiteralPath $policy -Value '{ not json'
            $stateDir = Join-Path $env:FLEET_RELIABILITY_HOME 'state'
            New-Item -ItemType Directory -Force $stateDir | Out-Null
            $last = Join-Path $stateDir 'last-sweep.json'
            Set-Content -LiteralPath $last -Value '{ not json'
            Mock -ModuleName PC-AI.Reliability Get-FleetDiskObservations { @() }
            { $null = @(Invoke-FleetHealthSweep -DryRun -Detector disk -PolicyPath $policy) } | Should -Throw '*policy*'
            { $null = Send-FleetDigest -DryRun -PolicyPath $policy } | Should -Throw '*policy*'
            $good = Join-Path $TestDrive 'good-policy.json'
            Set-Content -LiteralPath $good -Value '{}'
            $null = Send-FleetDigest -DryRun -PolicyPath $good
            Test-Path -LiteralPath $policy | Should -BeTrue
            Test-Path -LiteralPath $last | Should -BeTrue
            @(Get-ChildItem -LiteralPath $TestDrive, $stateDir -Filter '*.corrupt-*').Count | Should -Be 0
        }

        It 'fails a sweep whose policy exists but cannot be read, instead of falling back to defaults' {
            # Detects the silent downgrade from the eighth review of #346: an unreadable policy.json became the
            # defaults, dropping required drives, fleet checks and recipients, so two sweeps resolved a still-missing
            # required drive.
            $policy = Join-Path $TestDrive 'truncated-policy.json'
            Set-Content -LiteralPath $policy -Value '{"disks": {"Q": {"min_free_gb": 5, "requ'
            { Get-FleetPolicy -Path $policy } | Should -Throw '*policy*'
            { $null = Invoke-FleetHealthSweep -Detector disk -NoPlaybooks -PolicyPath $policy } | Should -Throw '*policy*'
            Test-Path -LiteralPath $policy | Should -BeTrue
            (Get-FleetPolicy -Path (Join-Path $TestDrive 'absent.json')).Count | Should -BeGreaterThan 0
        }

        It 'dry-run digest builds the message and records or sends nothing' {
            $null = Submit-FleetObservation (New-TestObservation -Name 'a')
            $lines = @(Read-TestLedger).Count
            $sends = @(Get-BusCall 'send').Count
            $d = Send-FleetDigest -DryRun -PolicyPath (Join-Path $TestDrive 'none.json')
            $d.Message | Should -Match 'opened 1'
            @(Read-TestLedger).Count | Should -Be $lines
            @(Get-BusCall 'send').Count | Should -Be $sends
        }

        It 'dry-run wrapper runs nothing and writes nothing' {
            Invoke-FleetTask -Name '\t\job' -FilePath 'C:\fake\job.exe' -DryRun 6>$null | Should -Be 0
            $script:Calls.Count | Should -Be 0
            Test-Path -LiteralPath $env:FLEET_RELIABILITY_HOME | Should -BeFalse
        }
    }

    Describe 'Collection failures (Windows data sources)' {
        # Detects the false health from the fourth review of #346: a CIM or socket query that failed was read as
        # "no processes" or "zero connections", so the detector reported ok and resolved open incidents.
        BeforeEach { $script:Policy = Get-FleetPolicy -Path (Join-Path $TestDrive 'none.json') }

        It 'raises when process enumeration fails instead of reporting no orphans' {
            if (-not (Get-Command Get-CimInstance -ErrorAction SilentlyContinue)) { Set-ItResult -Skipped -Because 'no CIM'; return }
            # A real CIM failure is a NON-terminating error, which -ErrorAction SilentlyContinue used to swallow.
            Mock -ModuleName PC-AI.Reliability Get-CimInstance { Write-Error -Message 'WMI repository is broken' -ErrorId 'HRESULT 0x80041010' -Category InvalidOperation }
            { InModuleScope PC-AI.Reliability { Get-FleetProcessData } } | Should -Throw '*WMI*'
        }

        It 'raises when the connection query fails, and still counts a port with no connections as zero' {
            if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { Set-ItResult -Skipped -Because 'no NetTCPIP'; return }
            Mock -ModuleName PC-AI.Reliability Get-NetTCPConnection { Write-Error -Message 'CIM query failed' -ErrorId 'HRESULT 0x80041003,Get-NetTCPConnection' -Category InvalidOperation }
            { InModuleScope PC-AI.Reliability { Get-FleetTcpListenerConnectionCount -Port 5300 } } | Should -Throw '*CIM query failed*'
            Mock -ModuleName PC-AI.Reliability Get-NetTCPConnection { Write-Error -Message 'No matching MSFT_NetTCPConnection objects found' -ErrorId 'CmdletizationQuery_NotFound_LocalPort,Get-NetTCPConnection' -Category ObjectNotFound }
            InModuleScope PC-AI.Reliability { Get-FleetTcpListenerConnectionCount -Port 5300 } | Should -Be 0
        }
    }

    Describe 'Discord transport' {
        BeforeEach {
            $script:PolicyPath = Join-Path $env:FLEET_RELIABILITY_HOME 'policy.json'
            New-Item -ItemType Directory -Force $env:FLEET_RELIABILITY_HOME | Out-Null
            '{"alerts":{"discord_min_severity":"medium","discord_channel_id":"123"}}' | Set-Content -LiteralPath $script:PolicyPath
            $script:Posted = New-Object System.Collections.ArrayList
            Mock -ModuleName PC-AI.Reliability Invoke-FleetDiscordPost { [void]$script:Posted.Add(@{ Uri = $Uri; Auth = $Headers.Authorization }); @{ id = '1' } }
        }

        It 'reads a LocalMachine-DPAPI token file (S4U-safe) and sends with it' {
            # Positive control for "ALERT not sent: no bot token": no vault, no env var, still delivers.
            $secure = New-Object System.Security.SecureString
                foreach ($ch in 'dummy-token-value'.ToCharArray()) { $secure.AppendChar($ch) }
                $null = Set-FleetDiscordToken -Token $secure
            $ev = New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Severity high
            Send-FleetDiscordEvent -LedgerEvent $ev -Policy (Get-FleetPolicy -Path $script:PolicyPath) | Should -Be 'sent'
            $script:Posted[0].Auth | Should -Be 'Bot dummy-token-value'
            $script:Posted[0].Uri | Should -Match '/channels/123/messages$'
        }

        It 'refuses to write the token when the directory ACL cannot be restricted' {
            # Protects the secret: the ACL is the only boundary of a LocalMachine-DPAPI blob.
            Mock -ModuleName PC-AI.Reliability Set-FleetSecretDirAcl { 5 }
            $secure = New-Object System.Security.SecureString
            foreach ($ch in 'dummy-token-value'.ToCharArray()) { $secure.AppendChar($ch) }
            { Set-FleetDiscordToken -Token $secure } | Should -Throw '*NOT written*'
            Test-Path (Join-Path $env:FLEET_RELIABILITY_HOME 'secrets\discord-bot-token.dpapi') | Should -BeFalse
        }

        It 'replaces an existing token file together with its permissive ACL' -Tag 'WindowsOnly' {
            # Detects the readable secret from the third review of #346: rotating the token overwrote the file in
            # place, so an explicit Everyone:R on the old file survived next to the restricted directory.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows ACLs'; return }
            $dir = Join-Path $env:FLEET_RELIABILITY_HOME 'secrets'
            New-Item -ItemType Directory -Force $dir | Out-Null
            $file = Join-Path $dir 'discord-bot-token.dpapi'
            [IO.File]::WriteAllBytes($file, [byte[]](1, 2, 3))
            & icacls.exe $file /grant '*S-1-1-0:R' | Out-Null
            $LASTEXITCODE | Should -Be 0
            $secure = New-Object System.Security.SecureString
            foreach ($ch in 'dummy-token-value'.ToCharArray()) { $secure.AppendChar($ch) }
            $null = Set-FleetDiscordToken -Token $secure
            $everyone = @((Get-Acl -LiteralPath $file).Access | Where-Object { $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -eq 'S-1-1-0' })
            $everyone.Count | Should -Be 0
            (Get-Item $file).Length | Should -BeGreaterThan 3
        }

        It 'removes explicit grants of other identities from the home and its files before elevated tasks use it' -Tag 'WindowsOnly' {
            # Detects the writable policy from the fifth review of #346: /inheritance:r kept an explicit Users:Modify
            # on an existing home and its policy.json, while the tasks run with Highest privileges.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows ACLs'; return }
            $homeDir = Join-Path $TestDrive 'acl-home'
            New-Item -ItemType Directory -Force $homeDir | Out-Null
            $policy = Join-Path $homeDir 'policy.json'
            Set-Content -LiteralPath $policy -Value '{}'
            & icacls.exe $homeDir /grant '*S-1-5-32-545:(OI)(CI)M' | Out-Null
            & icacls.exe $policy /grant '*S-1-5-32-545:M' | Out-Null
            @(Get-FleetSecretDirAclViolation -Path $policy).Count | Should -BeGreaterThan 0
            Set-FleetHomeAcl -Path $homeDir
            @(Get-FleetSecretDirAclViolation -Path $homeDir).Count | Should -Be 0
            @(Get-FleetSecretDirAclViolation -Path $policy).Count | Should -Be 0
        }

        It 'takes ownership away from an untrusted owner, and fails closed when it cannot' -Tag 'WindowsOnly' {
            # Detects the regained access from the seventh review of #346: a foreign owner of the home or policy file
            # keeps the right to rewrite its ACL after hardening, and the tasks run with Highest privileges.
            # Owner lookup and change are the module's own seams; making another account the owner needs privileges
            # a test does not have.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows ACLs'; return }
            $homeDir = Join-Path $TestDrive 'owner-home'
            New-Item -ItemType Directory -Force $homeDir | Out-Null
            Set-Content -LiteralPath (Join-Path $homeDir 'policy.json') -Value '{}'
            Mock -ModuleName PC-AI.Reliability Get-FleetItemOwnerSid { 'S-1-5-21-1-2-3-1001' }
            Mock -ModuleName PC-AI.Reliability Set-FleetItemOwner { 0 }
            { Set-FleetHomeAcl -Path $homeDir } | Should -Throw '*owned by*'
            Should -Invoke -ModuleName PC-AI.Reliability Set-FleetItemOwner -Times 2 -Exactly

            $script:Owned = @{}
            Mock -ModuleName PC-AI.Reliability Get-FleetItemOwnerSid { if ($script:Owned[$Path]) { 'S-1-5-32-544' } else { 'S-1-5-21-1-2-3-1001' } }
            Mock -ModuleName PC-AI.Reliability Set-FleetItemOwner { $script:Owned[$Path] = $true; 0 }
            { Set-FleetHomeAcl -Path $homeDir } | Should -Not -Throw
        }

        It 'accepts a home the installing user owns, with no ownership change' -Tag 'WindowsOnly' {
            # Positive control on REAL owners: a fresh directory and file are owned by the current user.
            if (-not (($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows)) { Set-ItResult -Skipped -Because 'Windows ACLs'; return }
            $homeDir = Join-Path $TestDrive 'own-home'
            New-Item -ItemType Directory -Force $homeDir | Out-Null
            Set-Content -LiteralPath (Join-Path $homeDir 'policy.json') -Value '{}'
            Mock -ModuleName PC-AI.Reliability Set-FleetItemOwner { 0 }
            { Set-FleetHomeAcl -Path $homeDir } | Should -Not -Throw
            Should -Invoke -ModuleName PC-AI.Reliability Set-FleetItemOwner -Times 0 -Exactly
        }

        It 'sends to the recipient of the policy passed with -PolicyPath, not the default policy' {
            # Detects the dropped-policy bug: forwarding reloaded the default policy.
            Remove-Item -LiteralPath $script:PolicyPath -Force
            $other = Join-Path $TestDrive 'other-policy.json'
            '{"alerts":{"discord_min_severity":"medium","discord_channel_id":"456"}}' | Set-Content -LiteralPath $other
            $env:DISCORD_BOT_TOKEN = 'dummy'
            $null = Send-FleetDigest -PolicyPath $other
            @($script:Posted | Where-Object { $_.Uri -match '/channels/456/messages$' }).Count | Should -Be 1
        }

        It 'skips without a token and below the severity floor, without throwing' {
            $ev = New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Severity high
            Send-FleetDiscordEvent -LedgerEvent $ev -Policy (Get-FleetPolicy -Path $script:PolicyPath) | Should -Be 'skipped(no token)'
            $env:DISCORD_BOT_TOKEN = 'x'
            $low = New-FleetLedgerEvent -Source task -Name demo -Kind incident-open -Status failed -Severity low
            Send-FleetDiscordEvent -LedgerEvent $low -Policy (Get-FleetPolicy -Path $script:PolicyPath) | Should -Be 'skipped(severity)'
            $script:Posted.Count | Should -Be 0
        }
    }

    Describe 'Scheduling priority' {
        It 'registers the sweeper at normal priority and the CLIs raise themselves from BelowNormal' -Tag 'Portable' {
            # Detects starvation: at the default priority 7 an S4U sweep needed 86 s just to start pwsh on a
            # host at 100% CPU, so the monitor went quiet exactly when the host was in trouble.
            $tools = Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Reliability'
            Get-Content -Raw (Join-Path $tools 'Install-FleetReliabilityTasks.ps1') | Should -Match 'New-ScheduledTaskSettingsSet[^\r\n]*-Priority 4'
            foreach ($cli in 'Invoke-FleetHealthSweep.ps1', 'Send-FleetDigest.ps1') {
                Get-Content -Raw (Join-Path $tools $cli) | Should -Match 'Set-FleetNormalPriority'
            }
        }

        It 'raises BelowNormal to Normal and never lowers a priority' {
            $p = Get-Process -Id $PID
            $saved = $p.PriorityClass
            try {
                $p.PriorityClass = 'BelowNormal'
                Set-FleetNormalPriority | Should -Be 'BelowNormal'
                (Get-Process -Id $PID).PriorityClass | Should -Be 'Normal'
                $p.PriorityClass = 'AboveNormal'
                Set-FleetNormalPriority | Should -Be 'AboveNormal'
                (Get-Process -Id $PID).PriorityClass | Should -Be 'AboveNormal'
            }
            finally { (Get-Process -Id $PID).PriorityClass = $saved }
        }
    }

    Describe 'Profile resolution' {
        It 'resolves the real profile even when USERPROFILE is redirected' {
            # Detects audit finding #15: agent sessions redirect USERPROFILE and scripts scanned nothing.
            $saved = $env:USERPROFILE
            try {
                $env:USERPROFILE = Join-Path $TestDrive 'redirected'
                Get-FleetRealProfile | Should -Be ([Environment]::GetFolderPath('UserProfile'))
                Get-FleetRealProfile | Should -Not -Be $env:USERPROFILE
            }
            finally { $env:USERPROFILE = $saved }
        }
    }
}

Describe 'PC-AI.Reliability child processes (unmocked)' -Tag 'Portable' {
    # Detects the hang found on 2026-11-21: Kill() left a grandchild holding the output pipe, so a 2 s timeout
    # took 26 s, and a job that exited at once but left a background process blocked for as long as it lived.
    # Real processes, no mocks; the 20 s bound separates "bounded" (about 5 s) from "hung" (30 s sleeper).
    BeforeAll {
        $env:FLEET_PIPE_DRAIN_MS = '3000'
        $script:OnWindows = ($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows
    }
    AfterAll { Remove-Item Env:FLEET_PIPE_DRAIN_MS -ErrorAction SilentlyContinue }

    It 'returns promptly on timeout even when a descendant holds the pipe' {
        if ($script:OnWindows) { $exe = Join-Path $env:SystemRoot 'System32\cmd.exe'; $argv = @('/c', 'ping -n 30 127.0.0.1') }
        else { $exe = '/bin/sh'; $argv = @('-c', 'sleep 30; echo late') }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = InModuleScope PC-AI.Reliability -Parameters @{ E = $exe; A = $argv } { param($E, $A) Invoke-FleetProcess -FilePath $E -ArgumentList $A -TimeoutSeconds 2 }
        $r.TimedOut | Should -BeTrue
        $sw.Elapsed.TotalSeconds | Should -BeLessThan 20
    }

    It 'returns promptly when the job exits but leaves a background process holding the pipe' {
        if ($script:OnWindows) { $exe = Join-Path $env:SystemRoot 'System32\cmd.exe'; $argv = @('/c', 'start /b ping -n 30 127.0.0.1 & echo parent-done') }
        else { $exe = '/bin/sh'; $argv = @('-c', 'sleep 30 & echo parent-done') }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = InModuleScope PC-AI.Reliability -Parameters @{ E = $exe; A = $argv } { param($E, $A) Invoke-FleetProcess -FilePath $E -ArgumentList $A -TimeoutSeconds 60 }
        $r.ExitCode | Should -Be 0
        $r.TimedOut | Should -BeFalse
        $sw.Elapsed.TotalSeconds | Should -BeLessThan 20
    }

    It 'keeps output printed before a descendant holds the pipe, so a failure line is not lost' {
        # Detects the false success from the second review of #346: the job printed [ERROR], exited 0 and left
        # a background process on the pipe; the whole-stream read was discarded at the drain deadline.
        if ($script:OnWindows) { $exe = Join-Path $env:SystemRoot 'System32\cmd.exe'; $argv = @('/c', 'echo [ERROR] integrity check did not pass & start /b ping -n 30 127.0.0.1 & exit 0') }
        else { $exe = '/bin/sh'; $argv = @('-c', 'echo "[ERROR] integrity check did not pass"; sleep 30 & exit 0') }
        $r = InModuleScope PC-AI.Reliability -Parameters @{ E = $exe; A = $argv } { param($E, $A) Invoke-FleetProcess -FilePath $E -ArgumentList $A -TimeoutSeconds 60 }
        $r.ExitCode | Should -Be 0
        $r.StdOut | Should -Match '\[ERROR\] integrity check'
        $r.OutputComplete | Should -BeFalse
    }

    It 'captures output and the exit code of a normal job' {
        if ($script:OnWindows) { $exe = Join-Path $env:SystemRoot 'System32\cmd.exe'; $argv = @('/c', 'echo hello & exit 3') }
        else { $exe = '/bin/sh'; $argv = @('-c', 'echo hello; exit 3') }
        $r = InModuleScope PC-AI.Reliability -Parameters @{ E = $exe; A = $argv } { param($E, $A) Invoke-FleetProcess -FilePath $E -ArgumentList $A -TimeoutSeconds 30 }
        $r.ExitCode | Should -Be 3
        $r.StdOut | Should -Match 'hello'
        $r.OutputComplete | Should -BeTrue
    }
}

Describe 'PC-AI.Reliability CLIs: help and dry-run' {
    # AGENTS.md: write-capable scripts expose -h, --help and a non-mutating -DryRun whose side effects
    # are proven absent. These run the real scripts in a child pwsh with a throwaway home.
    BeforeAll {
        $script:Tools = Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Reliability'
        $script:Pwsh = (Get-Process -Id $PID).Path
        if ($script:Pwsh -notmatch '(?i)pwsh(\.exe)?$') { $script:Pwsh = (Get-Command pwsh).Source }
        function script:Invoke-Cli([string]$Script, [string[]]$CliArgs) {
            $out = & $script:Pwsh -NoProfile -NonInteractive -File (Join-Path $script:Tools $Script) @CliArgs 2>&1 | Out-String
            return [pscustomobject]@{ Code = $LASTEXITCODE; Out = $out }
        }
    }
    BeforeEach {
        $script:SavedHome = $env:FLEET_RELIABILITY_HOME
        $script:SavedBus = $env:AGENT_BUS_EXE
        $env:FLEET_RELIABILITY_HOME = Join-Path $TestDrive ('cli-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        $env:AGENT_BUS_EXE = Join-Path $TestDrive 'no-such-agent-bus.exe'
    }
    AfterEach { $env:FLEET_RELIABILITY_HOME = $script:SavedHome; $env:AGENT_BUS_EXE = $script:SavedBus }

    It 'prints help for -h and --help on every CLI' -Tag 'Portable' {
        foreach ($cli in 'Invoke-FleetHealthSweep.ps1', 'Invoke-FleetTask.ps1', 'Send-FleetDigest.ps1', 'Install-FleetReliabilityTasks.ps1') {
            foreach ($flag in '--help', '-h') {
                $r = Invoke-Cli $cli @($flag)
                $r.Code | Should -Be 0 -Because "$cli $flag"
                $r.Out | Should -Match 'SYNOPSIS' -Because "$cli $flag"
            }
        }
    }

    It 'binds comma lists passed through pwsh -File, and refuses a quoted -ArgumentList' -Tag 'Portable' {
        # Detects the -File array trap found by a real run: "-Detector disk,cert" matched no detector, and
        # "-ArgumentList '/c','x'" reached the job as one literal string.
        $r = Invoke-Cli 'Invoke-FleetHealthSweep.ps1' @('--DryRun', '-Detector', 'cert,disk')
        $r.Code | Should -Be 0
        $r.Out | Should -Match 'DRY RUN: [1-9]\d* observations'
        $r = Invoke-Cli 'Invoke-FleetTask.ps1' @('-Name', '\t\job', '-FilePath', 'no-such-job', '-ArgumentList', "'/c','echo x'")
        $r.Code | Should -Be 2
        $r.Out | Should -Match 'cannot pass arrays'
        Test-Path -LiteralPath $env:FLEET_RELIABILITY_HOME | Should -BeFalse
    }

    It 'sweep --DryRun writes nothing under the home' -Tag 'Portable' {
        $r = Invoke-Cli 'Invoke-FleetHealthSweep.ps1' @('--DryRun', '-Detector', 'cert')
        $r.Code | Should -Be 0
        $r.Out | Should -Match 'DRY RUN'
        Test-Path -LiteralPath $env:FLEET_RELIABILITY_HOME | Should -BeFalse
    }

    It 'wrapper and digest --DryRun write nothing under the home' -Tag 'Portable' {
        $r = Invoke-Cli 'Invoke-FleetTask.ps1' @('--DryRun', '-Name', '\t\job', '-FilePath', 'no-such-job')
        $r.Code | Should -Be 0
        $r = Invoke-Cli 'Send-FleetDigest.ps1' @('--DryRun')
        $r.Code | Should -Be 0
        $r.Out | Should -Match 'DRY RUN'
        Test-Path -LiteralPath (Join-Path $env:FLEET_RELIABILITY_HOME 'ledger') | Should -BeFalse
    }

    It 'installer --DryRun registers no task and changes no ACL' -Tag 'WindowsOnly' {
        $folder = '\fleet-reliability-dryrun-test-' + [guid]::NewGuid().ToString('N').Substring(0, 6) + '\'
        $r = Invoke-Cli 'Install-FleetReliabilityTasks.ps1' @('--DryRun', '-Digest', '-TaskFolder', $folder)
        $r.Code | Should -Be 0
        $r.Out | Should -Match 'What if'
        @(Get-ScheduledTask -TaskPath $folder -ErrorAction SilentlyContinue).Count | Should -Be 0
        Test-Path -LiteralPath $env:FLEET_RELIABILITY_HOME | Should -BeFalse
    }
}

AfterAll {
    foreach ($v in 'FLEET_RELIABILITY_HOME', 'FLEET_HOST', 'AGENT_BUS_EXE', 'FLEET_NOW', 'FLEET_LEDGER_MAX_BYTES', 'DISCORD_BOT_TOKEN') { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    Remove-Module PC-AI.Reliability -ErrorAction SilentlyContinue
}
