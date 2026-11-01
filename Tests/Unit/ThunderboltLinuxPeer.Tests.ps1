BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1')
    $profilePath = Join-Path $repoRoot 'Config/thunderbolt-peers.json'
    $adapter = [pscustomobject]@{ Name = 'Thunderbolt peer'; InterfaceDescription = 'Microsoft USB4 P2P Network Adapter'; Status = 'Up'; ifIndex = 42 }
    $linux = [pscustomobject]@{
        hostname   = 'millylaptop1'
        interfaces = @([pscustomobject]@{ name = 'thunderbolt0'; driver = 'thunderbolt-net'; carrier = '1'; mtu = '1500' })
        addresses  = @([pscustomobject]@{ ifname = 'thunderbolt0'; addr_info = @() })
        routes = @(); iperf3Available = $true; networkManagerAvailable = $true
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
        $result.State | Should -Be Planned
        Should -Invoke Invoke-TbSsh -Times 0
        Should -Invoke Set-NetIPInterface -Times 0
        Should -Invoke New-NetIPAddress -Times 0
        Should -Invoke Get-TbLinuxInventory -Times 0
        Should -Invoke Get-NetAdapter -Times 0
    }
    It 'Prepare dry-run does not write or run remote commands' {
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Prepare -SelectedPeer millylaptop1 -ProfilePath $profilePath -EnableApply -IsDryRun
        $result.Applied | Should -BeFalse
        Should -Invoke Invoke-TbSsh -Times 0
        Should -Invoke Get-TbLinuxInventory -Times 0
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

Describe 'Per-invocation SSH application selection' {
    BeforeAll {
        $selectedApplication = Join-Path $PSHOME 'pwsh.exe'
        $selectedConfig = Join-Path $TestDrive 'selected ssh config'
        'Host millylaptop1' | Set-Content -LiteralPath $selectedConfig
    }
    It 'resolves an explicit application and literal config file while disabling only the agent' {
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $selectedConfig -DisableAgent
        $transport.FilePath | Should -Be $selectedApplication
        $transport.Arguments.Count | Should -Be 6
        $transport.Arguments[0] | Should -Be '-F'
        $transport.Arguments[1] | Should -Be $selectedConfig
        $transport.Arguments[2] | Should -Be '-o'
        $transport.Arguments[3] | Should -Be 'IdentityAgent=none'
        $transport.Arguments[4] | Should -Be '-o'
        $transport.Arguments[5] | Should -Be 'IdentitiesOnly=yes'
    }
    It 'keeps the default config and agent intact unless explicitly overridden' {
        $transport = Resolve-TbSshTransport -Path $selectedApplication
        $transport.Arguments.Count | Should -Be 0
    }
    It 'rejects absent applications and directories or absent config files' {
        { Resolve-TbSshTransport -Path (Join-Path $TestDrive 'absent.exe') } | Should -Throw
        { Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $TestDrive } | Should -Throw '*filesystem file*'
        { Resolve-TbSshTransport -Path $selectedApplication -ConfigFile (Join-Path $TestDrive 'absent-config') } | Should -Throw
    }
    It 'accepts configless mode only when explicitly requested' {
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile none
        $transport.Arguments.Count | Should -Be 2
        $transport.Arguments[1] | Should -Be none
    }
    It 'propagates selection and agent control to ordinary remote inventory commands' {
        Mock Invoke-TbNative { '{"hostname":"millylaptop1"}' }
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $selectedConfig -DisableAgent
        (Get-TbLinuxInventory -SshAlias millylaptop1 -SshTransport $transport).hostname | Should -Be millylaptop1
        Should -Invoke Invoke-TbNative -Times 1 -ParameterFilter {
            $FilePath -eq $selectedApplication -and $Arguments[0] -eq '-F' -and $Arguments[1] -eq $selectedConfig -and
            $Arguments -contains 'IdentityAgent=none' -and $Arguments -contains 'StrictHostKeyChecking=yes' -and
            $Arguments -contains 'millylaptop1' -and $TimeoutSeconds -eq 30
        }
    }
    It 'uses the same selected executable and options in the actual benchmark process start information' {
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $selectedConfig -DisableAgent
        $start = New-TbSshStartInfo -SshTransport $transport -Arguments @('-o', 'StrictHostKeyChecking=yes', 'millylaptop1', 'bash', '-s')
        $start.FileName | Should -Be $selectedApplication
        @($start.ArgumentList) | Should -Be @('-F', $selectedConfig, '-o', 'IdentityAgent=none', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', 'millylaptop1', 'bash', '-s')
        $start.UseShellExecute | Should -BeFalse
        $start.CreateNoWindow | Should -BeTrue
        $start.RedirectStandardInput | Should -BeTrue
        $start.RedirectStandardOutput | Should -BeTrue
        $start.RedirectStandardError | Should -BeTrue
    }
    It 'bounds selected SSH effective-config lookup and stops before any remote connection on timeout' {
        Mock Invoke-VigilBoundedProcess { throw 'controlled effective-config timed out' }
        Mock Invoke-TbSsh { throw 'Must not reach remote endpoint after config timeout.' }
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $selectedConfig -DisableAgent
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        { Test-TbSshEndpoint -Profile $profile -RemoteInterface thunderbolt0 -SshTransport $transport } | Should -Throw '*effective-config timed out*'
        Should -Invoke Invoke-VigilBoundedProcess -Times 1 -ParameterFilter {
            $FilePath -eq $selectedApplication -and $TimeoutSeconds -eq 10 -and
            $Arguments[0] -eq '-F' -and $Arguments[1] -eq $selectedConfig -and
            $Arguments -contains 'IdentityAgent=none' -and $Arguments -contains '-G'
        }
        Should -Invoke Invoke-TbSsh -Times 0
    }
    It 'passes the selected transport through config lookup and source-bound direct endpoint verification' {
        Mock Invoke-TbNative { "hostname 192.168.50.66`nhostkeyalias millylaptop1`n" }
        Mock Invoke-TbSsh { '{"dev":"thunderbolt0"}' }
        $transport = Resolve-TbSshTransport -Path $selectedApplication -ConfigFile $selectedConfig -DisableAgent
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        $endpoint = Test-TbSshEndpoint -Profile $profile -RemoteInterface thunderbolt0 -SshTransport $transport
        $endpoint.KnownHostIdentity | Should -Be millylaptop1
        $endpoint.Command | Should -Match ([regex]::Escape($selectedApplication))
        $endpoint.Command | Should -Match ([regex]::Escape($selectedConfig))
        $endpoint.Command | Should -Match 'IdentityAgent=none'
        Should -Invoke Invoke-TbSsh -Times 1 -ParameterFilter {
            $SshTransport.FilePath -eq $selectedApplication -and $SshTransport.Arguments -contains 'IdentityAgent=none' -and
            $Address -eq '172.31.240.2' -and $SourceAddress -eq '172.31.240.1' -and $KnownHostIdentity -eq 'millylaptop1'
        }
    }
    It 'dry-run avoids executable resolution, SSH and adapter discovery even with unavailable explicit paths' {
        Mock Resolve-TbSshTransport { throw 'Must not resolve in dry-run.' }
        Mock Get-TbLinuxInventory { throw 'Must not contact a peer in dry-run.' }
        Mock Get-NetAdapter { throw 'Must not query adapters in dry-run.' }
        $result = Invoke-ThunderboltLinuxPeerMain -SelectedAction Configure -SelectedPeer millylaptop1 -ProfilePath $profilePath `
            -EnableApply -IsDryRun -SshPath 'absent-client.exe' -SshConfigFile 'absent-config' -DisableSshAgent
        $result.Applied | Should -BeFalse
        $result.State | Should -Be Planned
        $result.Linux | Should -BeNullOrEmpty
        Should -Invoke Resolve-TbSshTransport -Times 0
        Should -Invoke Get-TbLinuxInventory -Times 0
        Should -Invoke Get-NetAdapter -Times 0
    }
    It 'starts the selected real benchmark process and reports its actual nonzero exit' {
        Mock Assert-TbBenchmarkRoute {}
        Mock Invoke-TbNative { '{"end":{"sum_received":{"bits_per_second":123}}}' }
        $fakeClient = Join-Path $TestDrive 'benchmark-client.ps1'
        @'
if ($args -notcontains 'millylaptop1' -or $args -notcontains 'StrictHostKeyChecking=yes') {
    throw 'Selected benchmark process lost its SSH arguments.'
}
$null = [Console]::In.ReadToEnd()
[Console]::Out.WriteLine('PCAI_IPERF_READY')
[Console]::Error.WriteLine('selected benchmark client sentinel')
exit 7
'@ | Set-Content -LiteralPath $fakeClient
        $transport = @{ FilePath = $selectedApplication; Arguments = @('-NoProfile', '-File', $fakeClient) }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        { Invoke-TbBenchmark -Profile $profile -Adapter $adapter -RemoteInterface thunderbolt0 `
            -Executable 'controlled-iperf' -Seconds 1 -ServerPort 5201 -SshTransport $transport } |
            Should -Throw '*exited 7*selected benchmark client sentinel*'
    }
    It 'supports actual help aliases before invalid SSH/config paths can be resolved' {
        $controller = Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1'
        foreach ($helpArgument in @('-h', '--help')) {
            $reply = Invoke-TbNative -FilePath $selectedApplication -Arguments @('-NoProfile', '-File', $controller,
                $helpArgument, '-SshPath', 'absent-client.exe', '-SshConfigFile', 'absent-config') -TimeoutSeconds 10
            $reply | Should -Match 'Invoke-ThunderboltLinuxPeer'
        }
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
        Mock Find-NetRoute { @([pscustomobject]@{ InterfaceIndex = 42 }) }
        Mock Get-NetIPAddress { @() }
        $profile = Get-TbProfile -Path $profilePath -Name millylaptop1
        { Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter } | Should -Throw '*address is missing*'
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
    It 'bounds a large stdin write when the actual child never reads it' {
        $pidFile = Join-Path $TestDrive 'nonreading-child.pid'
        # Allow a cold PowerShell child to initialize under concurrent build load.
        # A synchronous blocked stdin write would still wait the full 30 seconds.
        $child = '$PID | Set-Content -LiteralPath ''' + $pidFile.Replace("'", "''") + '''; Start-Sleep 30'
        $timer = [Diagnostics.Stopwatch]::StartNew()
        { Invoke-TbNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', $child) -InputText ('x' * 1048576) -TimeoutSeconds 10 } | Should -Throw '*timed out*'
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 13
        Test-Path -LiteralPath $pidFile | Should -BeTrue
        $childId = [int](Get-Content -LiteralPath $pidFile)
        Get-Process -Id $childId -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
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
foreach ($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep 30')) { $info.ArgumentList.Add($argument) }
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
