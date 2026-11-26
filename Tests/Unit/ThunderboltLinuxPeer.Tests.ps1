BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1')
    $profilePath = Join-Path $TestDrive 'linux-peer-profiles.json'
    @{
        schemaVersion = 1
        peers = @{
            millylaptop1 = @{
                sshAlias = 'millylaptop1'; expectedHostname = 'millylaptop1'
                localAddress = '172.31.240.1'; peerAddress = '172.31.240.2'
                prefixLength = 30; mtuBytes = 1500
            }
            asuspro13 = @{
                sshAlias = 'asuspro13'; expectedHostname = 'asuspro13'
                localAddress = '172.31.240.5'; peerAddress = '172.31.240.6'
                prefixLength = 30; mtuBytes = 1500
            }
        }
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $profilePath
    $adapter = [pscustomobject]@{ Name = 'Thunderbolt peer'; InterfaceDescription = 'Microsoft USB4 P2P Network Adapter'; Status = 'Up'; ifIndex = 42 }
    $linux = [pscustomobject]@{
        hostname   = 'millylaptop1'
        interfaces = @([pscustomobject]@{ name = 'thunderbolt0'; driver = 'thunderbolt-net'; carrier = '1'; mtu = '1500' })
        addresses  = @([pscustomobject]@{ ifname = 'thunderbolt0'; addr_info = @() })
        routes = @(); iperf3Available = $true; networkManagerAvailable = $true
    }
}

Describe 'Explicit Linux peer selection' -Tag 'Unit', 'Portable' {
    # Protects: both CLI entrypoints require an operator-selected peer.
    # Detects: falling through to profile reads or connecting to a historical default host.
    # Needs: PowerShell and a deliberately absent profile file. No SSH or hardware is used.
    # Breadcrumb: Tools/Invoke-ThunderboltNetworking.ps1 LinuxPeer and Invoke-ThunderboltLinuxPeer.ps1 entrypoint.
    It 'refuses an omitted peer before reading a profile through <Entry>' -ForEach @(
        @{ Entry = 'wrapper'; ScriptPath = 'Tools/Invoke-ThunderboltNetworking.ps1'; Mode = @{ Mode = 'LinuxPeer' } }
        @{ Entry = 'helper'; ScriptPath = 'Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1'; Mode = @{} }
    ) {
        $modes = $Mode
        $entrypoint = Join-Path $repoRoot $ScriptPath
        $absentProfile = Join-Path $TestDrive 'absent-peer-profile.json'
        { & $entrypoint @modes -Action Prepare -Apply -ConfigPath $absentProfile } | Should -Throw '*Specify -Peer*'
    }
}

Describe 'Thunderbolt TCP benchmark modes' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $benchmarkProfile = @{ sshAlias = 'test-peer'; expectedHostname = 'test-peer'; localAddress = '172.31.240.1'; peerAddress = '172.31.240.2' }
        $session = @{ Profile = $benchmarkProfile; Adapter = $adapter; RemoteInterface = 'thunderbolt0'; Executable = 'test-iperf3'; Seconds = 4; ServerPort = 55201 }
        Mock Assert-TbBenchmarkRoute {}
        Mock Invoke-TbNative { 'iperf 3.16 (cJSON 1.7.15)' }
        Mock Invoke-TbSsh { 'iperf 3.16 (cJSON 1.7.15)' }
    }

    # Protects: the default keeps separate forward and reverse measurements.
    # Detects: accidentally opting existing callers into concurrent traffic or changing result order.
    # Needs: mocked process and inventory providers; no network transport.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    It 'keeps the default benchmark as forward and reverse trials' {
        Mock Invoke-TbBenchmarkTrial { '{"end":{"sum_received":{"bits_per_second":20}}}' }
        $result = @(Invoke-TbBenchmark @session)
        $result.Count | Should -Be 2
        $result[0].Direction | Should -Be 'WindowsToLinux'
        $result[1].Direction | Should -Be 'LinuxToWindows'
        Should -Invoke Invoke-TbBenchmarkTrial -Times 1 -Exactly -ParameterFilter { -not $ClientOptions }
        Should -Invoke Invoke-TbBenchmarkTrial -Times 1 -Exactly -ParameterFilter { $ClientOptions -contains '-R' }
        Should -Invoke Invoke-TbNative -Times 0 -Exactly
        Should -Invoke Invoke-TbSsh -Times 0 -Exactly
    }

    # Protects: true simultaneous tests launch one source-bound client and report rates from both receivers.
    # Detects: treating successive one-way tests as duplex, or swapping the Linux and Windows receiver rows.
    # Needs: mocked process and inventory providers plus the iperf3 JSON contract; no traffic.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    It 'runs one bidirectional trial and reports both receiver rates' {
        Mock Invoke-TbBenchmarkTrial {
            '{"end":{"sum_received":{"bits_per_second":8000,"bytes":4000,"seconds":4},"sum_received_bidir_reverse":{"bits_per_second":6000,"bytes":3000,"seconds":4}}}'
        }
        $result = @(Invoke-TbBenchmark @session -FullDuplex)
        $result.Count | Should -Be 1
        $result[0].Direction | Should -Be 'FullDuplex'
        $result[0].ReceiverBitsPerSecond.WindowsToLinux | Should -Be 8000
        $result[0].ReceiverBitsPerSecond.LinuxToWindows | Should -Be 6000
        Should -Invoke Invoke-TbBenchmarkTrial -Times 1 -Exactly -ParameterFilter {
            $ClientOptions -contains '--bidir' -and $ClientOptions -notcontains '-R' -and
            $Profile.peerAddress -eq '172.31.240.2' -and $Profile.localAddress -eq '172.31.240.1' -and
            $RemoteInterface -eq 'thunderbolt0' -and $ServerPort -eq 55201 -and $Seconds -eq 4
        }
    }

    # Protects: duplex evidence is two real receiver rows from a successful JSON result.
    # Detects: reporting a one-way, failed, or nonnumeric measurement as full duplex.
    # Needs: mocked one-shot process and fixed JSON fixtures; no network transport.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    It 'rejects incomplete duplex evidence from <Case>' -ForEach @(
        @{ Case = 'an iperf error'; Payload = '{"error":"stream failed"}'; Expected = '*stream failed*' }
        @{ Case = 'a missing reverse receiver'; Payload = '{"end":{"sum_received":{"bits_per_second":8000,"bytes":4000,"seconds":4}}}'; Expected = '*sum_received_bidir_reverse*' }
        @{ Case = 'a missing forward receiver'; Payload = '{"end":{"sum_received_bidir_reverse":{"bits_per_second":6000,"bytes":3000,"seconds":4}}}'; Expected = '*sum_received*' }
        @{ Case = 'an invalid receiver rate'; Payload = '{"end":{"sum_received":{"bits_per_second":"8000","bytes":4000,"seconds":4},"sum_received_bidir_reverse":{"bits_per_second":6000,"bytes":3000,"seconds":4}}}'; Expected = '*sum_received*bits_per_second*' }
    ) {
        Mock Invoke-TbBenchmarkTrial { $Payload }
        { Invoke-TbBenchmark @session -FullDuplex } | Should -Throw $Expected
    }

    # Protects: both endpoints provide the separately named bidirectional JSON summaries.
    # Detects: starting a test against iperf3 versions that lack the required evidence shape.
    # Needs: mocked Windows and Linux version commands and process boundary.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    It 'refuses an older <Endpoint> iperf3 before opening the one-shot listener' -ForEach @(
        @{ Endpoint = 'Windows' }; @{ Endpoint = 'Linux' }
    ) {
        if ($Endpoint -eq 'Windows') { Mock Invoke-TbNative { 'iperf 3.10.1' } }
        else { Mock Invoke-TbSsh { 'iperf 3.10.1' } }
        Mock Invoke-TbBenchmarkTrial { throw 'Listener must not start.' }
        { Invoke-TbBenchmark @session -FullDuplex } | Should -Throw "*$Endpoint*iperf3 3.11 or newer*"
        Should -Invoke Invoke-TbBenchmarkTrial -Times 0 -Exactly
    }
}

Describe 'USB4 listener script text' -Tag 'Unit', 'Portable' {
    # Protects: the remote script that opens the unauthenticated iperf3 listener keeps its safety shape.
    # Detects: removing or reordering the return-route check, or weakening the bound, time-limited, cleaned-up listener.
    # Needs: PowerShell only. New-TbIperfServerScript is a pure text builder; no ssh, bash or network is used.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    BeforeAll {
        $peer = @{ sshAlias = 'test-peer'; expectedHostname = 'test-peer'; localAddress = '172.31.240.1'; peerAddress = '172.31.240.2' }
        $serverScript = (New-TbIperfServerScript -Profile $peer -RemoteInterface 'thunderbolt0' -Seconds 4 -ServerPort 55201).Replace("`r`n", "`n")
        $routeCheck = "ip -j -4 route get '172.31.240.1' from '172.31.240.2'"
    }

    # Protects: the return path is proven direct before anything listens, so a reply cannot leave through the LAN.
    # Detects: deleting the `ip -j route get ... from <peer>` check or moving it after the `iperf3 -s` line.
    # Needs: the rendered server script as text; compares string positions only.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'checks the source-bound return route before it opens the listener' {
        $check = $serverScript.IndexOf($routeCheck, [StringComparison]::Ordinal)
        $listener = $serverScript.IndexOf('iperf3 -s', [StringComparison]::Ordinal)
        $check | Should -BeGreaterThan -1
        $listener | Should -BeGreaterThan $check
    }

    # Protects: a route mismatch ends the remote script with a nonzero status before the listener line.
    # Detects: dropping `set -e`/pipefail, dropping the refusal exits, or binding the check to a different interface.
    # Needs: the rendered server script as text. The bash tests below execute these same lines.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'aborts on a route mismatch before the listener line' {
        $listener = $serverScript.IndexOf('iperf3 -s', [StringComparison]::Ordinal)
        $serverScript.Split("`n")[0] | Should -Be 'set -euo pipefail'
        foreach ($refusal in 'sys.exit("Expected exactly one source-bound USB4 return route")',
            'sys.exit("USB4 return route is not direct on the selected interface")') {
            $at = $serverScript.IndexOf($refusal, [StringComparison]::Ordinal)
            $at | Should -BeGreaterThan -1 -Because $refusal
            $at | Should -BeLessThan $listener -Because $refusal
        }
        $serverScript | Should -BeLike "*' 'thunderbolt0'`n*" -Because 'the check compares the route device with the selected interface'
    }

    # Protects: the listener stays one-shot, bound to the peer /30 address, time-limited and killed on every exit path.
    # Detects: an unbound or unbounded listener, a lost trap, or a lost readiness handshake after refactoring the builder.
    # Needs: the rendered server script as text.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'keeps the listener one-shot, bound, time-limited and cleaned up' {
        $lines = $serverScript.Split("`n")
        $lines | Should -Contain "timeout --kill-after=3s 19s iperf3 -s -1 -B '172.31.240.2' -p 55201 &"
        $lines | Should -Contain 'trap ''kill "$child" 2>/dev/null || true'' EXIT HUP INT TERM'
        $lines | Should -Contain 'kill -0 "$child"'
        $lines | Should -Contain 'echo PCAI_IPERF_READY'
        $lines | Should -Contain 'wait "$child"'
    }

}

Describe 'Duplex return-route guard executed under bash' -Tag 'Unit', 'Portable' {
    # Protects: the rendered remote script refuses to open the listener unless the return route is direct on the USB4 interface.
    # Detects: a guard that is present as text but does not stop the script, for example a lost pipefail or a loose interface match.
    # Needs: bash with a working python3, found by a functional probe. hostname, ip, timeout and iperf3 are stubs placed first on
    #        PATH inside the script's own shell, so no real route, listener or host is touched. Skipped with a reason otherwise.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    BeforeAll {
        $peer = @{ sshAlias = 'test-peer'; expectedHostname = 'test-peer'; localAddress = '172.31.240.1'; peerAddress = '172.31.240.2' }
        $isWindowsHost = [Environment]::OSVersion.Platform -eq 'Win32NT'
        $duplexScript = (New-TbIperfServerScript -Profile $peer -RemoteInterface 'thunderbolt0' -Seconds 4 -ServerPort 55201).Replace("`r`n", "`n")
        $directRoute = '[{"dst":"172.31.240.1","dev":"thunderbolt0","prefsrc":"172.31.240.2","flags":[],"uid":1000,"cache":[]}]'

        function Find-TbFixtureTool {
            param([string[]]$Name, [string[]]$Exclude = @(), [scriptblock]$Accept = { $true })
            foreach ($candidate in $Name) {
                foreach ($command in @(Get-Command $candidate -CommandType Application -All -ErrorAction SilentlyContinue)) {
                    if ($Exclude | Where-Object { $command.Source -like $_ }) { continue }
                    if (& $Accept $command.Source) { return $command.Source }
                }
            }
        }

        function Invoke-TbBashFixture {
            param([string]$Bash, [string]$Python, [string]$WorkDir, [string]$Script, [string]$RouteJson = '[]', [int]$IpExit = 0)
            $stub = Join-Path $WorkDir ([guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path $stub | Out-Null
            $marker = Join-Path $stub 'listener-calls.txt'
            $pythonPath = ($Python -replace '\\', '/').Replace("'", "'\''")
            $stubs = [ordered]@{
                hostname = "#!/bin/sh`necho 'test-peer'"
                ip       = @'
#!/bin/sh
case "$*" in
  *'addr show'*) echo '    inet 172.31.240.2/30 brd 172.31.240.3 scope global thunderbolt0' ;;
  *'route get'*) cat "$PCAI_FIXTURE_ROUTE"; exit "$PCAI_FIXTURE_IP_EXIT" ;;
  *) echo "unexpected ip call: $*" >&2; exit 99 ;;
esac
'@
                timeout  = "#!/bin/sh`necho `"timeout `$*`" >> `"`$PCAI_FIXTURE_MARKER`"`nsleep 2"
                iperf3   = "#!/bin/sh`necho `"iperf3 `$*`" >> `"`$PCAI_FIXTURE_MARKER`"`nsleep 2"
                python3  = "#!/bin/sh`nexec '$pythonPath' `"`$@`""
            }
            foreach ($name in $stubs.Keys) {
                $file = Join-Path $stub $name
                [IO.File]::WriteAllText($file, ($stubs[$name].Replace("`r`n", "`n") + "`n"), [Text.UTF8Encoding]::new($false))
                if (-not $isWindowsHost) { [IO.File]::SetUnixFileMode($file, [IO.UnixFileMode]'UserRead, UserWrite, UserExecute') }
            }
            $routeFile = Join-Path $stub 'route.json'
            [IO.File]::WriteAllText($routeFile, $RouteJson, [Text.UTF8Encoding]::new($false))

            $start = [Diagnostics.ProcessStartInfo]::new($Bash)
            $start.ArgumentList.Add('-s')
            $start.UseShellExecute = $false
            $start.RedirectStandardInput = $true
            $start.RedirectStandardOutput = $true
            $start.RedirectStandardError = $true
            $start.Environment['PCAI_FIXTURE_ROUTE'] = $routeFile -replace '\\', '/'
            $start.Environment['PCAI_FIXTURE_MARKER'] = $marker -replace '\\', '/'
            $start.Environment['PCAI_FIXTURE_IP_EXIT'] = "$IpExit"
            # Put the stub directory first on PATH from inside the shell, whatever order this bash flavour gives PATH.
            $prelude = 'S=''' + ($stub -replace '\\', '/') + '''; if command -v cygpath >/dev/null 2>&1; then S="$(cygpath -u "$S")"; fi; PATH="$S:$PATH"; export PATH' + "`n"
            $process = [Diagnostics.Process]::Start($start)
            try {
                $stdout = $process.StandardOutput.ReadToEndAsync()
                $stderr = $process.StandardError.ReadToEndAsync()
                $process.StandardInput.Write($prelude + $Script)
                $process.StandardInput.Close()
                $finished = $process.WaitForExit(30000)
                if (-not $finished) { $process.Kill($true); $process.WaitForExit(5000) | Out-Null }
                [pscustomobject]@{
                    TimedOut = -not $finished
                    ExitCode = $(if ($finished) { $process.ExitCode } else { -1 })
                    Stdout = $stdout.GetAwaiter().GetResult()
                    Stderr = $stderr.GetAwaiter().GetResult()
                    ListenerCalls = @($(if (Test-Path -LiteralPath $marker) { [IO.File]::ReadAllLines($marker) }))
                }
            }
            finally { $process.Dispose() }
        }

        # Find a bash and a python3 that can run the stubs; otherwise explain why the tests are skipped.
        $fixtureHost = @{ Bash = $null; Python = $null; WorkDir = $TestDrive; Reason = $null }
        $python = Find-TbFixtureTool -Name $(if ($isWindowsHost) { 'python3.exe', 'python.exe' } else { 'python3', 'python' }) -Accept {
            param($path) & $path -c 'import json, sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' 2>$null; $LASTEXITCODE -eq 0 }
        if (-not $python) { $fixtureHost.Reason = 'no working python3 was found; the guard under test is Python-based' }
        else {
            $fixtureHost.Python = $python
            $bashNames = if ($isWindowsHost) { 'bash.exe' } else { 'bash' }
            $known = if ($isWindowsHost) { @("$env:ProgramFiles\Git\bin\bash.exe", "$env:ProgramFiles\Git\usr\bin\bash.exe", "${env:ProgramFiles(x86)}\Git\bin\bash.exe") | Where-Object { $_ -and (Test-Path -LiteralPath $_) } }
            $probe = "test `"`$(hostname)`" = 'test-peer'`necho '[]' | python3 -c 'import json, sys; json.load(sys.stdin); print(`"PROBE_OK`")'`n"
            $candidates = @(@($known) + @(Get-Command $bashNames -CommandType Application -All -ErrorAction SilentlyContinue |
                    Where-Object { $_.Source -notlike '*\System32\*' -and $_.Source -notlike '*\WindowsApps\*' } | ForEach-Object Source) | Where-Object { $_ })
            $failures = @()
            foreach ($candidate in $candidates) {
                $result = Invoke-TbBashFixture -Bash $candidate -Python $python -WorkDir $TestDrive -Script $probe
                if ($result.ExitCode -eq 0 -and $result.Stdout -match 'PROBE_OK') { $fixtureHost.Bash = $candidate; break }
                $failures += "$candidate exited $($result.ExitCode): $($result.Stderr.Trim())"
            }
            if (-not $fixtureHost.Bash) {
                $fixtureHost.Reason = $(if ($failures) { 'no bash could run the stub probe: ' + ($failures -join '; ') } else { 'bash is not available' })
            }
        }
    }

    # Protects: a direct source-bound route on the USB4 interface still reaches the listener, with the bound and time limit intact.
    # Detects: a guard so strict that it blocks the supported topology, which would also make every refusal test below vacuous.
    # Needs: the bash fixture host described above; this is the positive control for the refusal cases.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'lets a direct route on thunderbolt0 through to the bounded listener' {
        if ($fixtureHost.Reason) { Set-ItResult -Skipped -Because $fixtureHost.Reason; return }
        $run = Invoke-TbBashFixture -Bash $fixtureHost.Bash -Python $fixtureHost.Python -WorkDir $fixtureHost.WorkDir -Script $duplexScript -RouteJson $directRoute
        $run.TimedOut | Should -BeFalse
        $run.ExitCode | Should -Be 0 -Because $run.Stderr
        $run.Stdout | Should -Match 'PCAI_IPERF_READY'
        @($run.ListenerCalls).Count | Should -Be 1
        $run.ListenerCalls[0] | Should -Be 'timeout --kill-after=3s 19s iperf3 -s -1 -B 172.31.240.2 -p 55201'
    }

    # Protects: every unsafe or unreadable return route stops the script before the listener, and for the stated reason.
    # Detects: deleting the check, loosening the interface comparison, accepting a gateway, or ignoring a failed or garbled `ip`.
    # Needs: the bash fixture host; fixture JSON stands in for `ip -j -4 route get`. A stub marker proves the listener never ran.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'refuses <Case> before opening the listener' -ForEach @(
        @{ Case = 'a route through the LAN NIC'; Json = '[{"dst":"172.31.240.1","gateway":"192.168.1.1","dev":"enp3s0","prefsrc":"192.168.1.20"}]'; IpExit = 0; Reason = '*not direct*' }
        @{ Case = 'a gateway on the USB4 interface'; Json = '[{"dst":"172.31.240.1","gateway":"172.31.240.9","dev":"thunderbolt0","prefsrc":"172.31.240.2"}]'; IpExit = 0; Reason = '*not direct*' }
        @{ Case = 'a via route on the USB4 interface'; Json = '[{"dst":"172.31.240.1","via":{"family":"inet","host":"172.31.240.9"},"dev":"thunderbolt0"}]'; IpExit = 0; Reason = '*not direct*' }
        @{ Case = 'a route with no interface'; Json = '[{"dst":"172.31.240.1","prefsrc":"172.31.240.2"}]'; IpExit = 0; Reason = '*not direct*' }
        @{ Case = 'an interface named thunderbolt00'; Json = '[{"dst":"172.31.240.1","dev":"thunderbolt00","prefsrc":"172.31.240.2"}]'; IpExit = 0; Reason = '*not direct*' }
        @{ Case = 'an empty route list'; Json = '[]'; IpExit = 0; Reason = '*exactly one*' }
        @{ Case = 'two candidate routes'; Json = '[{"dst":"172.31.240.1","dev":"thunderbolt0"},{"dst":"172.31.240.1","dev":"thunderbolt0"}]'; IpExit = 0; Reason = '*exactly one*' }
        @{ Case = 'garbled route JSON'; Json = 'garbled {'; IpExit = 0; Reason = '*JSON*' }
        @{ Case = 'no route output from a failing ip'; Json = ''; IpExit = 2; Reason = '*JSON*' }
        @{ Case = 'a failing ip that already printed a direct route'; Json = '[{"dst":"172.31.240.1","dev":"thunderbolt0","prefsrc":"172.31.240.2","flags":[],"uid":1000,"cache":[]}]'; IpExit = 1; Reason = $null }
    ) {
        if ($fixtureHost.Reason) { Set-ItResult -Skipped -Because $fixtureHost.Reason; return }
        $run = Invoke-TbBashFixture -Bash $fixtureHost.Bash -Python $fixtureHost.Python -WorkDir $fixtureHost.WorkDir -Script $duplexScript -RouteJson $Json -IpExit $IpExit
        $run.TimedOut | Should -BeFalse
        $run.ExitCode | Should -Not -Be 0
        $run.Stdout | Should -Not -Match 'PCAI_IPERF_READY'
        @($run.ListenerCalls).Count | Should -Be 0 -Because "the listener ran: $($run.ListenerCalls -join '; ')"
        if ($Reason) { $run.Stderr | Should -BeLike $Reason }
    }

    # Protects: the default benchmark stops before listening if the return route cannot be read.
    # Detects: leaving default sequential traffic without the same return-path proof used for duplex.
    # Needs: the bash fixture host. Only the stub `ip route get` fails and prints garbage.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, New-TbIperfServerScript.
    It 'refuses an invalid return route for the default trial' {
        if ($fixtureHost.Reason) { Set-ItResult -Skipped -Because $fixtureHost.Reason; return }
        $run = Invoke-TbBashFixture -Bash $fixtureHost.Bash -Python $fixtureHost.Python -WorkDir $fixtureHost.WorkDir -Script $duplexScript -RouteJson 'garbled {' -IpExit 7
        $run.ExitCode | Should -Not -Be 0
        $run.Stdout | Should -Not -Match 'PCAI_IPERF_READY'
        @($run.ListenerCalls).Count | Should -Be 0
    }
}

Describe 'Duplex receiver rate guard' -Tag 'Unit', 'Portable' {
    # Protects: a duplex result needs a strictly positive receiver rate on both rows.
    # Detects: accepting a zero or negative bits_per_second as measurement evidence, for example by relaxing `-le 0` to `-lt 0`.
    # Needs: mocked trial output and tool versions; no process, network or hardware.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    BeforeEach {
        $benchmarkProfile = @{ sshAlias = 'test-peer'; expectedHostname = 'test-peer'; localAddress = '172.31.240.1'; peerAddress = '172.31.240.2' }
        $session = @{ Profile = $benchmarkProfile; Adapter = $adapter; RemoteInterface = 'thunderbolt0'; Executable = 'test-iperf3'; Seconds = 4; ServerPort = 55201 }
        Mock Assert-TbBenchmarkRoute {}
        Mock Invoke-TbNative { 'iperf 3.16 (cJSON 1.7.15)' }
        Mock Invoke-TbSsh { 'iperf 3.16 (cJSON 1.7.15)' }
    }

    # Protects: the exact row and field are named when a receiver reports no throughput, and no result object escapes.
    # Detects: zero or negative rates passing, or being reported against the wrong summary.
    # Needs: mocked one-shot trial JSON in which only the named row is invalid.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-TbBenchmark.
    It 'rejects a bits_per_second of <Rate> in <Row> with no result' -ForEach @(
        @{ Row = 'sum_received'; Rate = 0 }
        @{ Row = 'sum_received'; Rate = -1 }
        @{ Row = 'sum_received_bidir_reverse'; Rate = 0 }
        @{ Row = 'sum_received_bidir_reverse'; Rate = -1 }
    ) {
        $good = '{"bits_per_second":8000,"bytes":4000,"seconds":4}'
        $bad = '{"bits_per_second":' + $Rate + ',"bytes":4000,"seconds":4}'
        $forward = if ($Row -eq 'sum_received') { $bad } else { $good }
        $reverse = if ($Row -eq 'sum_received') { $good } else { $bad }
        $payload = '{"end":{"sum_received":' + $forward + ',"sum_received_bidir_reverse":' + $reverse + '}}'
        Mock Invoke-TbBenchmarkTrial { $payload }
        $result = $null
        $failure = $null
        try { $result = @(Invoke-TbBenchmark @session -FullDuplex) } catch { $failure = $_ }
        $failure | Should -Not -BeNullOrEmpty -Because 'a non-positive receiver rate must not become a measurement'
        $failure.Exception.Message | Should -Be "iperf3 full-duplex JSON has invalid end.$Row.bits_per_second."
        $result | Should -BeNullOrEmpty
    }
}

Describe 'Thunderbolt profile safety' {
    It 'uses separate isolated /30 profiles at conservative MTU' {
        $milly = Get-TbProfile -Path $profilePath -Name millylaptop1
        $asus = Get-TbProfile -Path $profilePath -Name asuspro13
        $milly.localAddress | Should -Be '172.31.240.1'
        $milly.peerAddress | Should -Be '172.31.240.2'
        $milly.mtuBytes | Should -Be 1500
        $asus.localAddress | Should -Be '172.31.240.5'
    }
    It 'rejects unknown profile' {
        { Get-TbProfile -Path $profilePath -Name missing } | Should -Throw '*Unknown peer*'
    }
    It 'rejects command injection in SSH alias' {
        $config = Get-Content $profilePath -Raw | ConvertFrom-Json -AsHashtable
        $config.peers.millylaptop1.sshAlias = 'host;touch /tmp/unsafe'
        $path = Join-Path $TestDrive 'unsafe.json'
        $config | ConvertTo-Json -Depth 5 | Set-Content $path
        { Get-TbProfile -Path $path -Name millylaptop1 } | Should -Throw '*Unsafe*'
    }
    It 'rejects overlapping profiles' {
        $config = Get-Content $profilePath -Raw | ConvertFrom-Json -AsHashtable
        $config.peers.asuspro13.localAddress = '172.31.240.2'
        $path = Join-Path $TestDrive 'overlap.json'
        $config | ConvertTo-Json -Depth 5 | Set-Content $path
        { Get-TbProfile -Path $path -Name millylaptop1 } | Should -Throw '*overlap*'
    }
    It 'rejects an MTU shell payload instead of performing lexical range comparisons' {
        $config = Get-Content $profilePath -Raw | ConvertFrom-Json -AsHashtable
        $config.peers.millylaptop1.mtuBytes = "1500'; sudo -n true; #"
        $path = Join-Path $TestDrive 'mtu-injection.json'
        $config | ConvertTo-Json -Depth 5 | Set-Content $path
        { Get-TbProfile -Path $path -Name millylaptop1 } | Should -Throw '*must be an integer*'
    }
    It 'rejects a quoted numeric prefix and fractional MTU' {
        $config = Get-Content $profilePath -Raw | ConvertFrom-Json -AsHashtable
        $config.peers.millylaptop1.prefixLength = '30'
        $path = Join-Path $TestDrive 'numeric-types.json'
        $config | ConvertTo-Json -Depth 5 | Set-Content $path
        { Get-TbProfile -Path $path -Name millylaptop1 } | Should -Throw '*must be an integer*'
        $config.peers.millylaptop1.prefixLength = 30
        $config.peers.millylaptop1.mtuBytes = 1500.5
        $config | ConvertTo-Json -Depth 5 | Set-Content $path
        { Get-TbProfile -Path $path -Name millylaptop1 } | Should -Throw '*must be an integer*'
    }
}

Describe 'Hardware interface selection' {
    It 'selects a genuine USB4 P2P network interface' {
        (Select-TbWindowsAdapter -Adapters @($adapter)).ifIndex | Should -Be 42
    }
    It 'rejects a USB Ethernet adapter even with an explicit alias' {
        $ordinary = [pscustomobject]@{ Name = 'USB Ethernet'; InterfaceDescription = 'Realtek USB GbE'; Status = 'Up'; ifIndex = 7 }
        { Select-TbWindowsAdapter -Adapters @($ordinary) -Alias 'USB Ethernet' } | Should -Throw '*genuine*'
    }
    It 'requires an exact override when multiple peer interfaces exist' {
        $other = [pscustomobject]@{ Name = 'Peer two'; InterfaceDescription = 'Thunderbolt(TM) Networking'; Status = 'Up'; ifIndex = 43 }
        { Select-TbWindowsAdapter -Adapters @($adapter, $other) } | Should -Throw '*exactly one*'
        (Select-TbWindowsAdapter -Adapters @($adapter, $other) -Alias 'Peer two').ifIndex | Should -Be 43
    }
    It 'requires an Up Windows link' {
        $down = [pscustomobject]@{ Name = 'Peer'; InterfaceDescription = 'Thunderbolt Networking'; Status = 'Disconnected'; ifIndex = 42 }
        { Select-TbWindowsAdapter -Adapters @($down) } | Should -Throw '*not Up*'
    }
    It 'requires correct Linux hostname' {
        { Select-TbLinuxInterface -Inventory $linux -ExpectedHostname otherhost } | Should -Throw '*identity mismatch*'
    }
    It 'requires real thunderbolt-net driver and carrier' {
        $inventory = [pscustomobject]@{ hostname = 'millylaptop1'; interfaces = @([pscustomobject]@{ name = 'enx123'; driver = 'r8152'; carrier = '1' }) }
        { Select-TbLinuxInterface -Inventory $inventory -ExpectedHostname millylaptop1 } | Should -Throw '*thunderbolt-net*'
        $inventory.interfaces[0].driver = 'thunderbolt-net'
        $inventory.interfaces[0].carrier = '0'
        { Select-TbLinuxInterface -Inventory $inventory -ExpectedHostname millylaptop1 } | Should -Throw '*carrier*'
    }
}

Describe 'Configuration isolation and dry-run' {
    BeforeEach {
        Mock Get-TbLinuxInventory { $linux }
        Mock Get-NetAdapter { @($adapter) }
        Mock Get-NetIPAddress { @() }
        Mock Get-NetRoute { @() }
        Mock Set-NetIPInterface {}
        Mock New-NetIPAddress {}
        Mock Invoke-TbSsh { throw 'Mutation must not be called in planning tests.' }
        Mock Invoke-TbBenchmark { throw 'Benchmark must not be called in planning tests.' }
    }
    It 'Configure dry-run does not mutate either machine or report success' {
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply -IsDryRun
        $result.Applied | Should -BeFalse
        $result.State | Should -Be Observed
        Should -Invoke Invoke-TbSsh -Times 0
        Should -Invoke Set-NetIPInterface -Times 0
        Should -Invoke New-NetIPAddress -Times 0
    }
    It 'Prepare dry-run does not write or run remote commands' {
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Prepare -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply -IsDryRun
        $result.Applied | Should -BeFalse
        Should -Invoke Invoke-TbSsh -Times 0
    }
    # Protects: previewing a duplex benchmark creates no traffic even with -Apply.
    # Detects: starting the one-shot benchmark in the planning path or omitting its concurrent plan.
    # Needs: mocked adapter and Linux inventory; no peer or iperf3 installation.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1, Invoke-ThunderboltLinuxPeerMain.
    It 'previews full-duplex traffic without running the benchmark' {
        Mock Assert-TbBenchmarkRoute {}
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Benchmark -SelectedPeer millylaptop1 -ProfilePath $profilePath -FullDuplex -EnableApply -IsDryRun
        $result.Applied | Should -BeFalse
        $result.Plan -join ' ' | Should -Match 'simultaneous'
        Should -Invoke Invoke-TbBenchmark -Times 0 -Exactly
    }
    It 'WhatIf suppresses Configure writes even with Apply' {
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply -WhatIf
        $result.Applied | Should -BeFalse
        Should -Invoke Invoke-TbSsh -Times 0
        Should -Invoke Set-NetIPInterface -Times 0
    }
    It 'returns Blocked Status when Windows has no peer adapter' {
        Mock Get-NetAdapter { @() }
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Status -SelectedPeer millylaptop1 -ProfilePath $profilePath
        $result.State | Should -Be Blocked
        $result.Blockers.Count | Should -Be 1
    }
    It 'refuses Configure before any mutation when peer hardware is absent' {
        Mock Get-NetAdapter { @() }
        { Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply } | Should -Throw '*Cannot Configure*'
        Should -Invoke Invoke-TbSsh -Times 0
        Should -Invoke Set-NetIPInterface -Times 0
    }
    It 'preserves unrelated Windows addresses by refusing to change their adapter' {
        Mock Get-NetIPAddress { @([pscustomobject]@{ IPAddress = '192.168.10.1'; InterfaceIndex = 42; PrefixLength = 24 }) }
        { Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply } | Should -Throw '*Unrelated IPv4*'
        Should -Invoke Invoke-TbSsh -Times 0
    }
    It 'refuses address or exact subnet route already on another interface' {
        Mock Get-NetRoute { @([pscustomobject]@{ DestinationPrefix = '172.31.240.0/30'; InterfaceIndex = 7 }) }
        { Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply } | Should -Throw '*route belongs*'
    }
    It 'refuses a broader nondefault conflicting route and preserves the other interface' {
        Mock Get-NetRoute { @([pscustomobject]@{ DestinationPrefix = '172.31.0.0/16'; InterfaceIndex = 7 }) }
        { Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply } | Should -Throw '*route belongs*'
        Should -Invoke Invoke-TbSsh -Times 0
    }
    It 'detects host routes and ignores the ordinary default route' {
        Test-TbRouteContainsAddress -Prefix '172.31.240.2' -Address '172.31.240.2' | Should -BeTrue
        Test-TbRouteContainsAddress -Prefix '172.31.240.4/30' -Address '172.31.240.2' | Should -BeFalse
        Test-TbRouteContainsAddress -Prefix '0.0.0.0/0' -Address '172.31.240.2' | Should -BeFalse
    }
    It 'refuses an existing default route on the Thunderbolt adapter' {
        Mock Get-NetRoute { @([pscustomobject]@{ DestinationPrefix = '0.0.0.0/0'; InterfaceIndex = 42 }) }
        { Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply } | Should -Throw '*already has a default route*'
        Should -Invoke Invoke-TbSsh -Times 0
    }
    It 'builds a persistent Linux profile without DNS or a gateway' {
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        $script = New-TbLinuxConfigureScript -Profile $profile -Name millylaptop1 -Interface thunderbolt0
        $script | Should -Match "ipv4.gateway '' ipv4.dns ''"
        $script | Should -Match 'ipv4.never-default yes'
        $script | Should -Match '802-3-ethernet.mtu ''1500'''
        $script | Should -Not -Match 'ip route add default|systemctl|ufw|firewall'
    }
}

Describe 'Benchmark route and native failures' {
    BeforeAll {
        if (-not ('VigilProcessCustodyFixture' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class VigilProcessCustodyFixture {
    delegate void TryCloser(ref IntPtr handle, List<Exception> failures);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool DuplicateHandle(IntPtr sourceProcess, SafeProcessHandle source,
        IntPtr targetProcess, out SafeProcessHandle copy, uint access, bool inherit, uint options);
    [DllImport("kernel32.dll")]
    static extern IntPtr GetCurrentProcess();
    public static SafeProcessHandle Duplicate(Process process) {
        SafeProcessHandle copy;
        if (!DuplicateHandle(GetCurrentProcess(), process.SafeHandle, GetCurrentProcess(), out copy, 0, false, 2))
            throw new Win32Exception(Marshal.GetLastWin32Error());
        return copy;
    }
    public static Tuple<long, long, int> ObserveIndependentCloses(Type owner, Process process) {
        var close = (TryCloser)owner.GetMethod("TryClose", BindingFlags.NonPublic | BindingFlags.Static)
            .CreateDelegate(typeof(TryCloser));
        using (var copy = Duplicate(process)) {
            // Handle index zero is invalid; pseudo-handles can be accepted by
            // CloseHandle and therefore cannot establish its failure path.
            IntPtr invalid = new IntPtr(1), valid = copy.DangerousGetHandle();
            var failures = new List<Exception>();
            close(ref invalid, failures);
            close(ref valid, failures);
            if (valid == IntPtr.Zero) copy.SetHandleAsInvalid();
            return Tuple.Create(invalid.ToInt64(), valid.ToInt64(), failures.Count);
        }
    }
}
'@
        }
    }
    It 'binds direct Thunderbolt SSH to its source and disables inherited proxy routing' {
        Mock Invoke-TbNative { 'remote reply' }
        Invoke-TbSsh -SshAlias millylaptop1 -Address '172.31.240.2' -KnownHostIdentity millylaptop1 `
            -SourceAddress '172.31.240.1' -Script 'hostname' | Should -Be 'remote reply'
        Should -Invoke Invoke-TbNative -Times 1 -ParameterFilter {
            $Arguments -contains 'HostName=172.31.240.2' -and $Arguments -contains 'HostKeyAlias=millylaptop1' -and
            $Arguments -contains 'ProxyJump=none' -and $Arguments -contains 'ProxyCommand=none' -and
            $Arguments -contains '-b' -and $Arguments -contains '172.31.240.1' -and
            $Arguments -contains 'StrictHostKeyChecking=yes' -and $InputText -eq 'hostname'
        }
    }
    It 'uses the original effective host-key identity and checks the Linux return route' {
        Mock Invoke-TbNative { "hostname 192.168.50.43`r`nhostkeyalias millylaptop1`r`n" }
        Mock Invoke-TbSsh { '{"dev":"thunderbolt0","src":"172.31.240.2"}' }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        $endpoint = Test-TbSshEndpoint -Profile $profile -RemoteInterface thunderbolt0
        $endpoint.KnownHostIdentity | Should -Be millylaptop1
        $endpoint.SourceAddress | Should -Be '172.31.240.1'
        $endpoint.Command | Should -Be 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ProxyJump=none -o ProxyCommand=none -b 172.31.240.1 -o HostName=172.31.240.2 -o HostKeyAlias=millylaptop1 millylaptop1'
        Should -Invoke Invoke-TbSsh -Times 1 -ParameterFilter {
            $KnownHostIdentity -eq 'millylaptop1' -and $SourceAddress -eq '172.31.240.1' -and
            $Address -eq '172.31.240.2' -and $Script -match 'r\["dev"\]=="thunderbolt0"'
        }
    }
    It 'rejects a source-bound route through LAN' {
        Mock Find-NetRoute { @([pscustomobject]@{ InterfaceIndex = 7 }) }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        { Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter } | Should -Throw '*does not use*'
    }
    It 'rejects missing local configured source address' {
        Mock Find-NetRoute {
            [pscustomobject]@{ InterfaceIndex = 42; IPAddress = '172.31.240.1' }
            [pscustomobject]@{ InterfaceIndex = 42; DestinationPrefix = '172.31.240.0/30'; NextHop = '0.0.0.0' }
        }
        Mock Get-NetIPAddress { @() }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        { Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter } | Should -Throw '*address is missing*'
    }
    # Protects: receiver traffic is admitted only on the selected USB4 link and source.
    # Detects: a same-index gateway, default route, stale source or ambiguous provider rows.
    # Needs: fake Windows route and address rows. Breadcrumb: Invoke-ThunderboltLinuxPeer.ps1::Assert-TbBenchmarkRoute.
    It 'requires an unambiguous source and on-link peer route for <Case>' -ForEach @(
        @{ Case = 'connected subnet'; SelectedSource = '172.31.240.1'; Prefix = '172.31.240.0/30'; Hop = '0.0.0.0'; Extra = $false; State = 'Preferred'; Admit = $true }
        @{ Case = 'direct host route'; SelectedSource = '172.31.240.1'; Prefix = '172.31.240.2/32'; Hop = '0.0.0.0'; Extra = $false; State = 'Preferred'; Admit = $true }
        @{ Case = 'gateway on the same interface'; SelectedSource = '172.31.240.1'; Prefix = '172.31.240.2/32'; Hop = '172.31.240.2'; Extra = $false; State = 'Preferred'; Admit = $false }
        @{ Case = 'unrelated default route'; SelectedSource = '172.31.240.1'; Prefix = '0.0.0.0/0'; Hop = '0.0.0.0'; Extra = $false; State = 'Preferred'; Admit = $false }
        @{ Case = 'different source'; SelectedSource = '172.31.240.2'; Prefix = '172.31.240.0/30'; Hop = '0.0.0.0'; Extra = $false; State = 'Preferred'; Admit = $false }
        @{ Case = 'ambiguous route rows'; SelectedSource = '172.31.240.1'; Prefix = '172.31.240.0/30'; Hop = '0.0.0.0'; Extra = $true; State = 'Preferred'; Admit = $false }
        @{ Case = 'tentative local address'; SelectedSource = '172.31.240.1'; Prefix = '172.31.240.0/30'; Hop = '0.0.0.0'; Extra = $false; State = 'Tentative'; Admit = $false }
    ) {
        $rows = @(
            [pscustomobject]@{ InterfaceIndex = 42; IPAddress = $SelectedSource }
            [pscustomobject]@{ InterfaceIndex = 42; DestinationPrefix = $Prefix; NextHop = $Hop }
        )
        if ($Extra) { $rows += $rows[1] }
        $live = [pscustomobject]@{ InterfaceIndex = 42; IPAddress = '172.31.240.1'; PrefixLength = 30; AddressState = $State }
        Mock Find-NetRoute { $rows }
        Mock Get-NetIPAddress { $live }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        if ($Admit) {
            { Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter } | Should -Not -Throw
        } else {
            { Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter } | Should -Throw '*Benchmark refused*'
        }
    }
    It 'surfaces actual native nonzero exit and stderr' {
        { Invoke-TbNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', '[Console]::Error.WriteLine("controlled failure"); exit 7') } | Should -Throw '*exited 7*controlled failure*'
    }
    It 'terminates an actual native timeout' {
        { Invoke-TbNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', 'Start-Sleep 10') -TimeoutSeconds 1 } | Should -Throw '*timed out*'
        @(Get-VigilPendingProcessCustody).Count | Should -Be 0
    }
    It 'confirms an already exited owned process without another termination request' {
        Initialize-VigilBoundedProcessRunner
        $info = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-Command', 'exit 0')) { $info.ArgumentList.Add($argument) }
        $owned = [Diagnostics.Process]::Start($info)
        try {
            $handle = $owned.Handle
            $owned.WaitForExit(15000) | Should -BeTrue
            $confirm = [Vigil.BoundedProcess].GetMethod('ConfirmExit', [Reflection.BindingFlags]'NonPublic,Static')
            { $confirm.Invoke($null, @($handle)) } | Should -Not -Throw
        } finally {
            if (-not $owned.HasExited) { $owned.Kill($true); $owned.WaitForExit(1000) | Out-Null }
            $owned.Dispose()
        }
    }
    It 'preserves a failed native close while independently closing a valid owned handle' {
        Initialize-VigilBoundedProcessRunner
        $current = [Diagnostics.Process]::GetCurrentProcess()
        try {
            $observation = [VigilProcessCustodyFixture]::ObserveIndependentCloses([Vigil.BoundedProcess], $current)
            $observation.Item1 | Should -Be 1
            $observation.Item2 | Should -Be 0
            $observation.Item3 | Should -Be 1
            $current.HasExited | Should -BeFalse
        } finally { $current.Dispose() }
    }
    It 'retains exact handle custody through WhatIf, blocks replacement, and releases it only after confirmed exit' {
        Initialize-VigilBoundedProcessRunner
        $info = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardInput = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-Command', '[Console]::In.ReadToEnd() | Out-Null')) { $info.ArgumentList.Add($argument) }
        $owned = [Diagnostics.Process]::Start($info)
        $identity = [guid]::NewGuid()
        $pending = [Vigil.BoundedProcess].GetField('PendingCleanup', [Reflection.BindingFlags]'NonPublic,Static').GetValue($null)
        # A separately owned duplicate lets retry close its exact handle while
        # the Process object retains an independent observation/cleanup handle.
        $retained = [VigilProcessCustodyFixture]::Duplicate($owned)
        try {
            $pending.Add($identity, $retained)
            @(Get-VigilPendingProcessCustody) | Should -Contain $identity
            Stop-VigilPendingProcessCustody -CustodyId $identity -WhatIf
            $owned.HasExited | Should -BeFalse
            @(Get-VigilPendingProcessCustody) | Should -Contain $identity
            { Invoke-VigilBoundedProcess -FilePath (Join-Path $PSHOME 'pwsh.exe') } | Should -Throw '*blocks another launch*'
            { Stop-VigilPendingProcessCustody -CustodyId ([guid]::NewGuid()) -Confirm:$false } | Should -Throw '*Unknown owned process custody*'
            $owned.HasExited | Should -BeFalse
            Stop-VigilPendingProcessCustody -CustodyId $identity -Confirm:$false
            $owned.WaitForExit(1000) | Should -BeTrue
            @(Get-VigilPendingProcessCustody) | Should -Not -Contain $identity
        } finally {
            if (-not $owned.HasExited) { $owned.Kill($true); $owned.WaitForExit(1000) | Out-Null }
            $pending.Remove($identity) | Out-Null
            $retained.Dispose()
            $owned.Dispose()
        }
    }
    # Protects: a stdin write the child never reads is ended by the operation's own timeout, and the child that was started is terminated.
    # Detects: removing the timeout or the process kill, or losing the process-tree cleanup; a timeout that fires without killing.
    # Needs: PowerShell 7 and a Windows job object. The runner is compiled before timing, so the outer bound covers the operation
    #        and its teardown only. The 8 s operational timeout is asserted separately from the generous elapsed upper bound.
    # Breadcrumb: Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1, Invoke-TbNative.
    It 'bounds a large stdin write when the actual child never reads it' {
        $timeoutSeconds = 8
        $slackSeconds = 10
        $pidFile = Join-Path $TestDrive 'nonreading-child.pid'
        # The child records its PID and start time, then sleeps 40 s, far longer than timeout plus slack, so only a real kill ends it.
        $child = '$me = [Diagnostics.Process]::GetCurrentProcess(); [string]::Join('','', $me.Id, $me.StartTime.ToUniversalTime().Ticks) | Set-Content -LiteralPath ''' + $pidFile.Replace("'", "''") + '''; Start-Sleep 40'
        Initialize-VigilBoundedProcessRunner
        $timer = [Diagnostics.Stopwatch]::StartNew()
        { Invoke-TbNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', $child) -InputText ('x' * 1048576) -TimeoutSeconds $timeoutSeconds } |
            Should -Throw "*timed out after $timeoutSeconds seconds; process tree terminated*"
        $timer.Stop()
        $timer.Elapsed.TotalSeconds | Should -BeGreaterOrEqual $timeoutSeconds -Because 'the operation timeout, not an early exit, ended the call'
        $timer.Elapsed.TotalSeconds | Should -BeLessThan ($timeoutSeconds + $slackSeconds) -Because 'teardown on a loaded host needs slack, but the 40 s child must not run to completion'
        Test-Path -LiteralPath $pidFile | Should -BeTrue -Because 'the child must have started before the timeout for its termination to be provable'
        $childId, $startTicks = (Get-Content -LiteralPath $pidFile -Raw).Trim() -split ','
        $survivor = Get-Process -Id ([int]$childId) -ErrorAction SilentlyContinue |
            Where-Object { $_.StartTime.ToUniversalTime().Ticks -eq [long]$startTicks }
        $survivor | Should -BeNullOrEmpty -Because 'the exact child process that was started must be gone'
    }
    It 'preserves large stdin and simultaneous output without deadlock' {
        $text = 'z' * 1048576
        $child = '[Console]::Error.WriteLine("e" * 131072); $inputText = [Console]::In.ReadToEnd(); [Console]::Out.Write($inputText.Length)'
        Invoke-TbNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', $child) -InputText $text -TimeoutSeconds 10 | Should -Be '1048576'
    }
    It 'terminates an owned descendant after its parent exits normally' {
        $witnessPath = Join-Path $TestDrive 'owned-descendant.json'
        $child = @'
$info = [Diagnostics.ProcessStartInfo]::new('__SHELL__')
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
foreach ($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep 40')) { $info.ArgumentList.Add($argument) }
$owned = [Diagnostics.Process]::Start($info)
$witness = @{ Id = $owned.Id; StartTicks = $owned.StartTime.ToUniversalTime().Ticks } | ConvertTo-Json
[IO.File]::WriteAllText('__WITNESS__', $witness)
[Console]::Out.Write('spawned')
exit 0
'@
        $child = $child.Replace('__SHELL__', (Join-Path $PSHOME 'pwsh.exe').Replace("'", "''")).Replace('__WITNESS__', $witnessPath.Replace("'", "''"))
        $descendant = $null
        try {
            $result = Invoke-VigilBoundedProcess -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoLogo','-NoProfile','-Command',$child) -TimeoutSeconds 15
            $result.ExitCode | Should -Be 0
            $result.Stdout | Should -BeExactly 'spawned'
            $witness = Get-Content -LiteralPath $witnessPath -Raw | ConvertFrom-Json
            $observed = Get-Process -Id $witness.Id -ErrorAction SilentlyContinue
            if ($observed) {
                if ($observed.StartTime.ToUniversalTime().Ticks -ne $witness.StartTicks) {
                    $observed.Dispose()
                    throw 'The descendant PID was reused; no unrelated process will be terminated.'
                }
                $descendant = $observed
                $descendant.WaitForExit(1000) | Should -BeTrue
            }
            @(Get-VigilPendingProcessCustody).Count | Should -Be 0
        } finally {
            if ($descendant) {
                if (-not $descendant.HasExited) { $descendant.Kill($true); $descendant.WaitForExit(1000) | Out-Null }
                $descendant.Dispose()
            }
        }
    }
}
