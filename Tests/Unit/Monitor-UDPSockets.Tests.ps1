#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
<#
  Tests for Tools/SystemScripts/Machine/Monitor-UDPSockets.ps1.

  Why these exist: on dtm-work (2026-11-21) the log held 0 "OK" lines because the heartbeat ran only
  when the minute was 0, and a 5-minute task that starts at 06:14:34 never runs at minute 0. The monitor
  also force-stops dllhost, so its decision logic must be tested without touching a real process, and it
  counted 0 endpoints (a healthy-looking result) when Get-NetUDPEndpoint failed.
#>
BeforeAll {
    $script:Script = (Resolve-Path (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine\Monitor-UDPSockets.ps1')).Path
    . $script:Script
    function New-Endpoints([int]$Count, [int]$OwningProcess = 1234) {
        1..$Count | ForEach-Object { [pscustomobject]@{ OwningProcess = $OwningProcess } }
    }
    function Get-LogText([string]$Dir) {
        $p = Join-Path $Dir 'udp-monitor.log'
        if (Test-Path $p) { Get-Content -LiteralPath $p -Raw } else { '' }
    }
}

Describe 'Test-HeartbeatDue' {
    It 'is due when no stamp exists (first run must write a heartbeat)' {
        Test-HeartbeatDue -LogDir $TestDrive -Minutes 60 | Should -BeTrue
    }

    It 'is not due 5 minutes after a heartbeat and is due after 60' {
        $stamp = Join-Path $TestDrive 'udp-monitor.heartbeat'
        Set-Content -LiteralPath $stamp -Value 'x'
        (Get-Item $stamp).LastWriteTime = [datetime]'2026-11-21 06:00:00'
        Test-HeartbeatDue -LogDir $TestDrive -Minutes 60 -Now ([datetime]'2026-11-21 06:05:00') | Should -BeFalse
        Test-HeartbeatDue -LogDir $TestDrive -Minutes 60 -Now ([datetime]'2026-11-21 07:00:00') | Should -BeTrue
    }

    It 'fires once per hour for the real 5-minute trigger that starts at :14:34 (regression for the Minute -eq 0 bug)' {
        # Positive control: the OLD rule never fires for this phase, which is why the log had no OK lines.
        $runs = 0..23 | ForEach-Object { ([datetime]'2026-11-21 06:14:34').AddMinutes(5 * $_) }
        @($runs | Where-Object { $_.Minute -eq 0 }).Count | Should -Be 0

        $stamp = Join-Path $TestDrive 'udp-monitor.heartbeat'
        Remove-Item $stamp -ErrorAction SilentlyContinue
        $fired = 0
        foreach ($t in $runs) {
            if (Test-HeartbeatDue -LogDir $TestDrive -Minutes 60 -Now $t) {
                $fired++
                Set-Content -LiteralPath $stamp -Value 'x'
                (Get-Item $stamp).LastWriteTime = $t
            }
        }
        # 24 runs span 2 hours: first run plus one hour later (06:14 -> 07:14) plus the 08:14 run is out of range.
        $fired | Should -Be 2
    }
}

Describe 'Invoke-UdpMonitor' {
    BeforeEach {
        $script:dir = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        Mock Get-NetUDPEndpoint { New-Endpoints 150 }
        Mock Stop-Process {}
        Mock ipconfig {}
    }

    It 'writes an OK heartbeat and returns 0 when the count is under the threshold' {
        (Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60) | Should -Be 0
        Get-LogText $script:dir | Should -Match 'OK: 150 UDP sockets'
    }

    It 'does not repeat the OK line on the next run inside the heartbeat window' {
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60
        ([regex]::Matches((Get-LogText $script:dir), 'OK:')).Count | Should -Be 1
    }

    It 'returns 1 and logs an ERROR when the endpoint list cannot be read (a failure must not look healthy)' {
        Mock Get-NetUDPEndpoint { throw 'CIM unavailable' }
        (Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60) | Should -Be 1
        Get-LogText $script:dir | Should -Match 'ERROR: cannot read UDP endpoints: CIM unavailable'
    }

    It 'logs an ALERT with the top owner above the threshold but stops nothing for a non-dllhost owner' {
        Mock Get-NetUDPEndpoint { New-Endpoints 6000 -OwningProcess 4242 }
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'chrome' } }
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60
        Get-LogText $script:dir | Should -Match 'ALERT: UDP socket count 6000'
        Get-LogText $script:dir | Should -Match 'PID 4242 \(chrome\): 6000 sockets'
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'does not stop dllhost at 10000 sockets or fewer (the rule is strictly greater than 10000)' {
        Mock Get-NetUDPEndpoint { New-Endpoints 10000 -OwningProcess 99 }
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'dllhost' } }
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'stops dllhost above 10000 sockets and flushes DNS' {
        Mock Get-NetUDPEndpoint { New-Endpoints 10001 -OwningProcess 99 }
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'dllhost' } }
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60
        Should -Invoke Stop-Process -Times 1 -Exactly -ParameterFilter { $Id -eq '99' -and $Force }
        Should -Invoke ipconfig -Times 1 -Exactly
        Get-LogText $script:dir | Should -Match 'AUTO-KILL: dllhost PID 99 with 10001 sockets'
    }

    It '-DryRun creates no directory, log or heartbeat, and says what it would log (AGENTS.md dry-run rule)' {
        $out = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60 -DryRun
        Test-Path -LiteralPath $script:dir | Should -BeFalse
        ($out | Where-Object { $_ -is [string] }) -join "`n" | Should -Match 'DRYRUN would log: .*OK: 150 UDP sockets'
        $out | Where-Object { $_ -is [int] } | Should -Be 0
    }

    It '-DryRun leaves an existing log and heartbeat byte-for-byte and mtime unchanged, so it cannot suppress the next real OK line' {
        $null = New-Item -ItemType Directory $script:dir
        $log = Join-Path $script:dir 'udp-monitor.log'; $hb = Join-Path $script:dir 'udp-monitor.heartbeat'
        Set-Content $log 'old line'; Set-Content $hb 'old'
        (Get-Item $hb).LastWriteTime = [datetime]'2026-01-01 00:00:00'
        $before = @((Get-FileHash $log).Hash, (Get-FileHash $hb).Hash, (Get-Item $hb).LastWriteTime.Ticks)
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60 -DryRun
        @((Get-FileHash $log).Hash, (Get-FileHash $hb).Hash, (Get-Item $hb).LastWriteTime.Ticks) | Should -Be $before
    }

    It '-DryRun above 10000 dllhost sockets reports the kill but stops nothing and flushes nothing' {
        Mock Get-NetUDPEndpoint { New-Endpoints 10001 -OwningProcess 99 }
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'dllhost' } }
        $out = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60 -DryRun
        Should -Invoke Stop-Process -Times 0 -Exactly
        Should -Invoke ipconfig -Times 0 -Exactly
        Test-Path -LiteralPath $script:dir | Should -BeFalse
        ($out | Where-Object { $_ -is [string] }) -join "`n" | Should -Match 'DRYRUN would log: .*AUTO-KILL: dllhost PID 99'
    }

    It '-WhatIf is a dry run too: no files, no kill' {
        Mock Get-NetUDPEndpoint { New-Endpoints 10001 -OwningProcess 99 }
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'dllhost' } }
        $null = Invoke-UdpMonitor -Threshold 5000 -LogDir $script:dir -HeartbeatMinutes 60 -WhatIf
        Should -Invoke Stop-Process -Times 0 -Exactly
        Should -Invoke ipconfig -Times 0 -Exactly
        Test-Path -LiteralPath $script:dir | Should -BeFalse
    }
}