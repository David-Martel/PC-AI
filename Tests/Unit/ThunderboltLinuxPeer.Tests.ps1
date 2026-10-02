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
    }
}
