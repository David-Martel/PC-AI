#Requires -Version 7.0
<#
.SYNOPSIS
    Exercises network identity, readiness, transport and repair contracts without hardware I/O.
.DESCRIPTION
    Loads the real source functions. Only OS commands, time and native process boundaries
    are replaced. Fail-closed boundary guards prevent unmocked adapter or transport actions.
#>
param([string]$NetworkSourceRoot)

BeforeAll {
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $networkRoot = if ($NetworkSourceRoot) { $NetworkSourceRoot } else { Join-Path $repoRoot 'Modules/PC-AI.Network' }
    . (Join-Path $networkRoot 'Private/Adapter-Readiness.ps1')
    . (Join-Path $networkRoot 'Private/Network-Helpers.ps1')
    . (Join-Path $networkRoot 'Public/Get-StableNetAdapter.ps1')
    . (Join-Path $networkRoot 'Public/Test-NetPathHealth.ps1')
    . (Join-Path $networkRoot 'Public/Repair-UsbNetAdapter.ps1')

    function Get-NetAdapter { [CmdletBinding()] param([switch]$IncludeHidden) throw 'Unmocked adapter enumeration forbidden.' }
    function Get-NetIPAddress { [CmdletBinding()] param($InterfaceIndex, $AddressFamily) throw 'Unmocked address enumeration forbidden.' }
    function Get-NetAdapterRss { [CmdletBinding()] param($Name) throw 'Unmocked RSS query forbidden.' }
    function Get-NetAdapterRsc { [CmdletBinding()] param($Name) throw 'Unmocked RSC query forbidden.' }
    function Get-NetAdapterAdvancedProperty { [CmdletBinding()] param($Name, $RegistryKeyword) throw 'Unmocked advanced-property query forbidden.' }
    function Get-NetAdapterBinding { [CmdletBinding()] param($Name) throw 'Unmocked binding query forbidden.' }
    function Disable-PnpDevice { [CmdletBinding(SupportsShouldProcess)] param($InstanceId) throw 'Unmocked device disable forbidden.' }
    function Enable-PnpDevice { [CmdletBinding(SupportsShouldProcess)] param($InstanceId) throw 'Unmocked device enable forbidden.' }
    function netsh { param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments) throw 'Unmocked netsh forbidden.' }
    function iperf3 { param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments) throw 'Unmocked throughput transport forbidden.' }
    function ping.exe { param([int]$n, [int]$w, [string]$Target) throw 'Unmocked ping forbidden.' }
    function New-FixtureAdapter {
        param([string]$Name = 'Ethernet 17', [int]$Index = 17, [string]$Status = 'Up')
        [pscustomobject]@{
            Name = $Name; ifIndex = $Index; Status = $Status
            InterfaceGuid = '{22222222-2222-4222-8222-222222222222}'
            MacAddress = '02-00-00-00-00-17'; InterfaceDescription = 'Synthetic USB NIC'
            LinkSpeed = '5 Gbps'; MtuSize = 9014; PnPDeviceID = 'SYNTHETIC\NIC\17'
        }
    }
}

Describe 'Stable adapter identity and IPv4 authority' -Tag 'Unit', 'Network', 'Windows' {
    BeforeEach {
        $script:Adapters = @(New-FixtureAdapter)
        Mock Get-NetAdapter { $script:Adapters }
        Mock Get-NetIPAddress {
            @([pscustomobject]@{ IPAddress = '169.254.1.2'; PrefixLength = 16 },
              [pscustomobject]@{ IPAddress = '10.42.0.1'; PrefixLength = 30 })
        }
        Mock netsh { '1500 1 10 20 Ethernet 170'; '62000 1 10 20 Ethernet 17' }
    }
    It 'resolves a GUID with its current name/index and exact subinterface row' {
        $result = Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}'
        $result.Name | Should -Be 'Ethernet 17'
        $result.ifIndex | Should -Be 17
        $result.IPv4Mtu | Should -Be '62000'
        $result.AdapterMtu | Should -Be 9014
        $result.IPv4Address | Should -Be '10.42.0.1/30'
        Should -Invoke Get-NetAdapter -Exactly -Times 1 -ParameterFilter { $IncludeHidden }
        Should -Invoke Get-NetIPAddress -Exactly -Times 1 -ParameterFilter { $InterfaceIndex -eq 17 -and $AddressFamily -eq 'IPv4' }
    }
    It 'survives name and index reassignment without selecting an unrelated adapter' {
        $script:Adapters = @((New-FixtureAdapter -Name 'Ethernet 18' -Index 28), (New-FixtureAdapter))
        $script:Adapters[1].InterfaceGuid = '{33333333-3333-4333-8333-333333333333}'
        Mock netsh { '1500 1 0 0 Ethernet 17'; '62000 1 0 0 Ethernet 18' }
        $result = Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}'
        $result.Name | Should -Be 'Ethernet 18'
        $result.ifIndex | Should -Be 28
        $result.IPv4Mtu | Should -Be '62000'
        Should -Invoke Get-NetIPAddress -Exactly -Times 1 -ParameterFilter { $InterfaceIndex -eq 28 }
    }
    It 'refuses zero and duplicate GUID matches before querying MTU or addresses' -ForEach @(
        @{ Count = 0; Message = '*No adapter matches*' },
        @{ Count = 2; Message = '*matched 2 adapters*Refusing to guess*' }
    ) {
        $script:Adapters = if ($Count -eq 0) { @() } else { @((New-FixtureAdapter), (New-FixtureAdapter -Index 18)) }
        { Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' } | Should -Throw $Message
        Should -Invoke netsh -Exactly -Times 0
        Should -Invoke Get-NetIPAddress -Exactly -Times 0
    }
    It 'validates GUID format before OS discovery' {
        { Get-StableNetAdapter -InterfaceGuid 'not-a-guid' } | Should -Throw '*does not match*'
        Should -Invoke Get-NetAdapter -Exactly -Times 0
    }
    It 'warns for unstable names and still rejects duplicate name matches' {
        $script:Adapters += New-FixtureAdapter -Index 18
        Mock Write-Warning {}
        { Get-StableNetAdapter -Name 'Ethernet 17' } | Should -Throw '*Refusing to guess*'
        Should -Invoke Write-Warning -Exactly -Times 1 -ParameterFilter { $Message -like "Resolving by Name is unstable: 'Ethernet 17'*" }
    }
    It 'does not substitute adapter MTU when netsh fails or has only a prefix match' -ForEach @(
        @{ Failure = $true }, @{ Failure = $false }
    ) {
        Mock netsh { if ($Failure) { throw 'synthetic query refusal' }; '1500 1 10 20 Ethernet 170' }
        $result = Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}'
        $result.IPv4Mtu | Should -BeNullOrEmpty
        $result.AdapterMtu | Should -Be 9014
    }
    It 'reports no-RSS nonoperational RSC risk without blaming a bound filter' {
        Mock Get-NetAdapterRss { [pscustomobject]@{ Enabled = $false; NumberOfReceiveQueues = 0 } }
        Mock Get-NetAdapterRsc { [pscustomobject]@{ IPv4Enabled = $true; IPv4OperationalState = $false; IPv4FailureReason = 'NDISCompatibility' } }
        Mock Get-NetAdapterAdvancedProperty { [pscustomobject]@{ DisplayValue = '9014'; ValidDisplayValues = @('1514', '9014') } }
        Mock Get-NetAdapterBinding { @([pscustomobject]@{ Enabled = $true; ComponentID = 'ms_tcpip' }, [pscustomobject]@{ Enabled = $false; ComponentID = 'excluded_filter' }) }
        $result = Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -Detailed
        $result.ReceiveRisk | Should -BeLike 'HIGH:*NDISCompatibility*'
        $result.BoundComponents | Should -Be @('ms_tcpip')
        $result.JumboSupported | Should -Be '1514 | 9014'
        Should -Invoke Get-NetAdapterAdvancedProperty -Exactly -Times 1 -ParameterFilter { $Name -eq 'Ethernet 17' -and $RegistryKeyword -eq '*JumboPacket' }
    }
}

Describe 'Adapter readiness follows identity and expected address' -Tag 'Unit', 'Network', 'Windows' {
    BeforeEach {
        $script:Clock = [datetime]'2026-01-01T00:00:00Z'
        $script:Poll = 0
        Mock Get-Date { $script:Clock }
        Mock Start-Sleep { $script:Clock = $script:Clock.AddSeconds($Seconds); $script:Poll++ }
        Mock Get-NetAdapter {
            if ($script:Poll -eq 1) { throw 'temporarily absent' }
            New-FixtureAdapter -Name 'Ethernet 29' -Index 29
        }
        Mock netsh { '62000 1 0 0 Ethernet 29' }
        Mock Get-NetIPAddress {
            [pscustomobject]@{ IPAddress = if ($script:Poll -lt 3) { '10.42.0.99' } else { '10.42.0.1' }; PrefixLength = 30 }
        }
        Mock ping.exe { 'synthetic warmup' }
    }
    It 'waits through disappearance and Up without expected IP before warming only the peer' {
        $ready = Wait-PcaiAdapterReady -Resolve { Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' } -ExpectedIPv4 '10.42.0.1' -WarmPingTarget '10.42.0.2' -TimeoutSeconds 15
        $ready | Should -BeTrue
        $script:Poll | Should -Be 3
        Should -Invoke ping.exe -Exactly -Times 1 -ParameterFilter { $n -eq 2 -and $w -eq 1500 -and $Target -eq '10.42.0.2' }
        Should -Invoke Get-NetIPAddress -Exactly -Times 4 -ParameterFilter { $InterfaceIndex -eq 29 -and $AddressFamily -eq 'IPv4' }
    }
    It 'times out honestly when expected IP never appears and never warms a peer' {
        Mock Get-NetIPAddress { [pscustomobject]@{ IPAddress = '10.42.0.99'; PrefixLength = 30 } }
        Wait-PcaiAdapterReady -Resolve { Get-StableNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' } -ExpectedIPv4 '10.42.0.1' -WarmPingTarget '10.42.0.2' -TimeoutSeconds 6 | Should -BeFalse
        $script:Poll | Should -Be 2
        Should -Invoke ping.exe -Exactly -Times 0
    }
    It 'does not warm the local address when no peer is provided' {
        Wait-PcaiAdapterReady -Resolve { New-FixtureAdapter } -TimeoutSeconds 5 | Should -BeTrue
        Should -Invoke Get-NetIPAddress -Exactly -Times 0
        Should -Invoke ping.exe -Exactly -Times 0
    }
    It 'does not treat a disconnected adapter as ready' {
        Wait-PcaiAdapterReady -Resolve { New-FixtureAdapter -Status 'Disconnected' } -TimeoutSeconds 6 | Should -BeFalse
        Should -Invoke ping.exe -Exactly -Times 0
    }
}

Describe 'Native throughput results and attribution' -Tag 'Unit', 'Network', 'Windows' {
    BeforeEach {
        $script:Rates = @(1000, 1000)
        $script:NativeExit = 0
        $script:NativeExits = @(0, 0, 0, 0)
        $script:TransportCalls = [Collections.Generic.List[object]]::new()
        Mock Start-Sleep {}
        Mock iperf3 {
            $script:TransportCalls.Add(@($Arguments))
            $global:LASTEXITCODE = if ($script:NativeExit) { $script:NativeExit } else { $script:NativeExits[$script:TransportCalls.Count - 1] }
            $rate = $script:Rates[$script:TransportCalls.Count - 1]
            if ($null -eq $rate) { 'iperf3: error - synthetic refusal' } else { "[SUM] 0.00-1.00 sec 100 MBytes $rate Mbits/sec receiver" }
        }
    }
    It 'passes exact peer port stream duration and reverse arguments and reports symmetry' {
        $result = Test-NetPathHealth -Peer 'primary.fixture' -Port 5301 -Streams 2 -Seconds 1
        $result.Verdict | Should -BeLike 'SYMMETRIC:*'
        $result.Ratio | Should -Be 1
        $result.Asymmetric | Should -BeFalse
        ($script:TransportCalls[0] -join '|') | Should -Be '-c|primary.fixture|-p|5301|-P|2|-t|1|-f|m'
        ($script:TransportCalls[1] -join '|') | Should -Be '-c|primary.fixture|-p|5301|-P|2|-t|1|-f|m|-R'
    }
    It 'does not attribute asymmetry without a control peer' {
        $script:Rates = @(1000, 100)
        $result = Test-NetPathHealth -Peer 'primary.fixture'
        $result.Asymmetric | Should -BeTrue
        $result.Verdict | Should -Match 'CAUSE NOT ESTABLISHED'
        $script:TransportCalls.Count | Should -Be 2
    }
    It 'reports incomplete transport output without fabricated throughput' {
        $script:Rates = @($null, 1000)
        $script:NativeExit = 7
        $result = Test-NetPathHealth -Peer 'primary.fixture'
        $result.LocalToPeerMbits | Should -BeNullOrEmpty
        $result.Verdict | Should -BeLike 'INCOMPLETE:*'
        $result.Errors -join ' ' | Should -Match 'synthetic refusal'
    }
    It 'rejects a nonzero native exit even when stdout contains a plausible receiver result' {
        $script:NativeExit = 7
        $result = Test-NetPathHealth -Peer 'primary.fixture'
        $result.Verdict | Should -BeLike 'INCOMPLETE:*'
        $result.LocalToPeerMbits | Should -BeNullOrEmpty
    }
    It 'checks the exit from each primary direction separately: call <FailAt>' -ForEach @(
        @{ FailAt = 1 }, @{ FailAt = 2 }
    ) {
        $script:NativeExits[$FailAt - 1] = 7
        $result = Test-NetPathHealth -Peer 'primary.fixture'
        $result.Verdict | Should -BeLike 'INCOMPLETE:*'
        $result.Errors -join ' ' | Should -Match '7'
        if ($FailAt -eq 1) {
            $result.LocalToPeerMbits | Should -BeNullOrEmpty
            $result.PeerToLocalMbits | Should -Be 1000
        } else {
            $result.LocalToPeerMbits | Should -Be 1000
            $result.PeerToLocalMbits | Should -BeNullOrEmpty
        }
    }
    It 'refuses attribution after a failed native control call <FailAt> for <Slow>' -ForEach @(
        @{ FailAt = 3; Slow = 'receive'; Rates = @(1000, 100, 1000, 1000) },
        @{ FailAt = 4; Slow = 'receive'; Rates = @(1000, 100, 1000, 1000) },
        @{ FailAt = 3; Slow = 'transmit'; Rates = @(100, 1000, 1000, 1000) },
        @{ FailAt = 4; Slow = 'transmit'; Rates = @(100, 1000, 1000, 1000) }
    ) {
        $script:Rates = $Rates
        $script:NativeExits[$FailAt - 1] = 7
        $result = Test-NetPathHealth -Peer 'primary.fixture' -ControlPeer 'control.fixture'
        $result.Verdict | Should -Match 'control test.*did not complete.*7'
        $result.Verdict | Should -Not -Match 'LOCALISED'
        if ($FailAt -eq 3) { $result.ControlCapabilityMbits | Should -BeNullOrEmpty }
        else { $result.ControlMbits | Should -BeNullOrEmpty }
    }
    It 'keeps incapable controls inconclusive in either slow direction' -ForEach @(
        @{ Rates = @(1000, 100, 100, 100) }, @{ Rates = @(100, 1000, 100, 100) }
    ) {
        $script:Rates = $Rates
        $result = Test-NetPathHealth -Peer 'primary.fixture' -ControlPeer 'control.fixture'
        $result.Verdict | Should -BeLike 'INCONCLUSIVE:*'
        $result.ControlCapabilityMbits | Should -Be 100
        $result.ControlMbits | Should -Be 100
        $script:TransportCalls.Count | Should -Be 4
    }
    It 'attributes only after the opposite control direction demonstrates capability: <Verdict>' -ForEach @(
        @{ Rates = @(1000, 100, 1000, 100); Verdict = "LOCALISED TO THIS HOST'S RECEIVE PATH:*"; CapabilityReverse = $false },
        @{ Rates = @(1000, 100, 1000, 1000); Verdict = 'LOCALISED TO THE PEER OR THE PATH TO IT:*'; CapabilityReverse = $false },
        @{ Rates = @(100, 1000, 1000, 100); Verdict = "LOCALISED TO THIS HOST'S TRANSMIT PATH:*"; CapabilityReverse = $true },
        @{ Rates = @(100, 1000, 1000, 1000); Verdict = "LOCALISED TO THE PEER'S RECEIVE PATH:*"; CapabilityReverse = $true }
    ) {
        $script:Rates = $Rates
        $result = Test-NetPathHealth -Peer 'primary.fixture' -ControlPeer 'control.fixture'
        $result.Verdict | Should -BeLike $Verdict
        $result.ControlCapabilityMbits | Should -Be 1000
        ($script:TransportCalls[2] -contains '-R') | Should -Be $CapabilityReverse
        ($script:TransportCalls[3] -contains '-R') | Should -Be (-not $CapabilityReverse)
        $script:TransportCalls[2][1] | Should -Be 'control.fixture'
        $script:TransportCalls[3][1] | Should -Be 'control.fixture'
    }
    It 'does not attribute when either control direction fails to return a measurement' -ForEach @(
        @{ Rates = @(1000, 100, $null, 1000) }, @{ Rates = @(100, 1000, 1000, $null) }
    ) {
        $script:Rates = $Rates
        $result = Test-NetPathHealth -Peer 'primary.fixture' -ControlPeer 'control.fixture'
        $result.Verdict | Should -Match 'control test.*did not complete'
        $result.Verdict | Should -Not -Match 'LOCALISED'
    }
}

Describe 'Registry and latency helper contracts' -Tag 'Unit', 'Network', 'Windows' {
    It 'formats byte rates at binary unit boundaries' -ForEach @(
        @{ Rate = 0; Expected = '0.00 B/s' }, @{ Rate = 1023; Expected = '1,023.00 B/s' },
        @{ Rate = 1024; Expected = '1.00 KB/s' }, @{ Rate = 1048576; Expected = '1.00 MB/s' },
        @{ Rate = 1073741824; Expected = '1.00 GB/s' }, @{ Rate = 1099511627776; Expected = '1.00 TB/s' }
    ) {
        $oldCulture = [Threading.Thread]::CurrentThread.CurrentCulture
        try {
            [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]'en-US'
            Format-BytesPerSecond -BytesPerSecond $Rate | Should -Be $Expected
        } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $oldCulture }
    }
    It 'formats latency without confusing microseconds milliseconds and seconds' -ForEach @(
        @{ Value = 0.5; Expected = '500.00 us' }, @{ Value = 10; Expected = '10.00 ms' },
        @{ Value = 1000; Expected = '1.00 s' }
    ) {
        $oldCulture = [Threading.Thread]::CurrentThread.CurrentCulture
        try {
            [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]'en-US'
            Format-Latency -Milliseconds $Value | Should -Be $Expected
        } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $oldCulture }
    }
    It 'reports an empty section explicitly and preserves supplied text' -ForEach @(
        @{ Data = $null; Text = 'No observations.' }, @{ Data = @(); Text = 'No observations.' },
        @{ Data = 'observed link state only'; Text = 'observed link state only' }
    ) {
        ConvertTo-NetworkReportSection -Title 'Synthetic' -Data $Data -EmptyMessage 'No observations.' |
            Should -Be ("== Synthetic ==`r`n`r`n$Text`r`n")
    }
    It 'maps known and unknown adapter status values' -ForEach @(
        @{ Code = 1; Expected = 'Up' }, @{ Code = 7; Expected = 'LowerLayerDown' }, @{ Code = 99; Expected = 'Unknown (99)' }
    ) { Get-AdapterStatusDescription -StatusCode $Code | Should -Be $Expected }
    It 'maps issue severity without treating an unknown issue as healthy' -ForEach @(
        @{ Issue = 'IPConflict'; Expected = 'Critical' }, @{ Issue = 'PacketLoss'; Expected = 'Warning' },
        @{ Issue = 'UnusedAdapter'; Expected = 'Info' }, @{ Issue = 'Unexpected'; Expected = 'Unknown' }
    ) { Get-NetworkSeverity -IssueType $Issue | Should -Be $Expected }
    It 'does not mutate a missing registry key under WhatIf' {
        Mock Test-Path { $false }
        Mock New-Item { throw 'registry create attempted' }
        Mock Set-ItemProperty { throw 'registry write attempted' }
        Set-RegistryValueSafe -Path 'HKCU:\SyntheticOnly' -Name 'Value' -Value 17 -PropertyType DWord -WhatIf | Should -BeFalse
        Should -Invoke New-Item -Exactly -Times 0
        Should -Invoke Set-ItemProperty -Exactly -Times 0
    }
    It 'returns a specified registry fallback when the external read fails' {
        Mock Test-Path { $true }
        Mock Get-ItemProperty { throw 'synthetic access refusal' }
        Get-RegistryValueSafe -Path 'HKCU:\SyntheticOnly' -Name 'Value' -DefaultValue 37 | Should -Be 37
        Should -Invoke Get-ItemProperty -Exactly -Times 1 -ParameterFilter { $Path -eq 'HKCU:\SyntheticOnly' -and $Name -eq 'Value' }
    }
    It 'returns an existing zero registry value instead of the nonzero fallback' {
        Mock Test-Path { $true }
        Mock Get-ItemProperty { [pscustomobject]@{ Value = 0 } }
        Get-RegistryValueSafe -Path 'HKCU:\SyntheticOnly' -Name 'Value' -DefaultValue 37 | Should -Be 0
    }
    It 'writes only the exact requested registry value through the external boundary' {
        Mock Test-Path { $true }
        Mock Set-ItemProperty {}
        Set-RegistryValueSafe -Path 'HKCU:\SyntheticOnly' -Name 'Value' -Value 17 -PropertyType DWord -Confirm:$false | Should -BeTrue
        Should -Invoke Set-ItemProperty -Exactly -Times 1 -ParameterFilter { $Path -eq 'HKCU:\SyntheticOnly' -and $Name -eq 'Value' -and $Value -eq 17 -and $Type -eq 'DWord' -and $Force }
    }
    It 'surfaces an external registry write refusal and never reports success' {
        Mock Test-Path { $true }
        Mock Set-ItemProperty { throw 'synthetic write refusal' }
        Mock Write-Warning {}
        Set-RegistryValueSafe -Path 'HKCU:\SyntheticOnly' -Name 'Value' -Value 17 -PropertyType DWord -Confirm:$false | Should -BeFalse
        Should -Invoke Write-Warning -Exactly -Times 1 -ParameterFilter { $Message -like '*synthetic write refusal*' }
    }
    It 'reports actual latency samples and missing replies as packet loss' {
        Mock Test-Connection { @([pscustomobject]@{ ResponseTime = 10 }, [pscustomobject]@{ ResponseTime = 30 }) }
        $result = Measure-NetworkLatency -Target 'peer.fixture' -Count 4
        $result.Success | Should -BeTrue
        $result.MinLatency | Should -Be 10
        $result.MaxLatency | Should -Be 30
        $result.AvgLatency | Should -Be 20
        $result.PacketLoss | Should -Be 50
        Should -Invoke Test-Connection -Exactly -Times 1 -ParameterFilter { $ComputerName -eq 'peer.fixture' -and $Count -eq 4 }
    }
    It 'reports a failed ping honestly without inventing latency' {
        Mock Test-Connection { throw 'synthetic transport refusal' }
        $result = Measure-NetworkLatency -Target 'peer.fixture'
        $result.Success | Should -BeFalse
        $result.PacketLoss | Should -Be 100
        $result.AvgLatency | Should -BeNullOrEmpty
        $result.Error | Should -Be 'synthetic transport refusal'
    }
}

Describe 'TCP connection endpoint and completion contracts' -Tag 'Unit', 'Network', 'Windows' {
    It 'distinguishes connection success timeout and failed EndConnect, closing its exact client' -ForEach @(
        @{ WaitResult = $true; EndFailure = $false; ExpectedSuccess = $true; ExpectedMessage = 'Connected' },
        @{ WaitResult = $false; EndFailure = $false; ExpectedSuccess = $false; ExpectedMessage = 'Connection timeout' },
        @{ WaitResult = $true; EndFailure = $true; ExpectedSuccess = $false; ExpectedMessage = '*synthetic connection refusal*' }
    ) {
        $script:TcpState = @{ WaitResult = $WaitResult; EndFailure = $EndFailure; Closed = 0; EndCalls = 0 }
        $waitHandle = [pscustomobject]@{}
        $waitHandle | Add-Member ScriptMethod WaitOne {
            param($Milliseconds, $ExitContext)
            $script:TcpState.Timeout = $Milliseconds
            $script:TcpState.ExitContext = $ExitContext
            $script:TcpState.WaitResult
        }
        $script:ConnectResult = [pscustomobject]@{ AsyncWaitHandle = $waitHandle }
        $script:TcpClient = [pscustomobject]@{}
        $script:TcpClient | Add-Member ScriptMethod BeginConnect {
            param($HostName, $Port, $Callback, $State)
            $script:TcpState.HostName = $HostName
            $script:TcpState.Port = $Port
            $script:TcpState.Callback = $Callback
            $script:TcpState.State = $State
            $script:ConnectResult
        }
        $script:TcpClient | Add-Member ScriptMethod EndConnect {
            param($Result)
            $script:TcpState.EndCalls++
            $script:TcpState.ResultSame = [object]::ReferenceEquals($Result, $script:ConnectResult)
            if ($script:TcpState.EndFailure) { throw 'synthetic connection refusal' }
        }
        $script:TcpClient | Add-Member ScriptMethod Close { $script:TcpState.Closed++ }
        Mock New-Object { $script:TcpClient } -ParameterFilter { $TypeName -eq 'System.Net.Sockets.TcpClient' }
        $result = Test-PortConnectivity -HostName 'peer.fixture' -Port 31415 -TimeoutMs 123
        $result.Host | Should -Be 'peer.fixture'
        $result.Port | Should -Be 31415
        $result.Success | Should -Be $ExpectedSuccess
        $result.Message | Should -BeLike $ExpectedMessage
        $script:TcpState.HostName | Should -Be 'peer.fixture'
        $script:TcpState.Port | Should -Be 31415
        $script:TcpState.Timeout | Should -Be 123
        $script:TcpState.ExitContext | Should -BeFalse
        $script:TcpState.Callback | Should -BeNullOrEmpty
        $script:TcpState.State | Should -BeNullOrEmpty
        $script:TcpState.Closed | Should -Be 1
        $script:TcpState.EndCalls | Should -Be ([int]$WaitResult)
        if ($WaitResult) { $script:TcpState.ResultSame | Should -BeTrue }
        Should -Invoke New-Object -Exactly -Times 1 -ParameterFilter { $TypeName -eq 'System.Net.Sockets.TcpClient' }
    }
}

Describe 'USB repair WhatIf and threshold contracts' -Tag 'Unit', 'Network', 'Windows' {
    BeforeAll {
        # GetNewClosure resolves the exported identity function through the actual module.
        # The dot-sourced function remains the repair unit; its identity dependency is real.
        Import-Module (Join-Path $networkRoot 'PC-AI.Network.psd1') -Force -Global -ErrorAction Stop
    }
    BeforeEach {
        Mock Get-NetAdapter { New-FixtureAdapter }
        Mock Get-NetIPAddress { [pscustomobject]@{ IPAddress = '10.42.0.1'; PrefixLength = 30 } }
        Mock netsh { '62000 1 0 0 Ethernet 17' }
        Mock Get-NetAdapter { New-FixtureAdapter } -ModuleName PC-AI.Network
        Mock Get-NetIPAddress { [pscustomobject]@{ IPAddress = '10.42.0.1'; PrefixLength = 30 } } -ModuleName PC-AI.Network
        Mock netsh { '62000 1 0 0 Ethernet 17' } -ModuleName PC-AI.Network
        Mock Disable-PnpDevice { throw 'device disable must not run in preview' }
        Mock Enable-PnpDevice { throw 'device enable must not run in preview' }
        Mock Start-Sleep { throw 'repair must not wait in preview' }
    }
    It 'performs no PnP cycle in preview, or fails the privilege boundary before discovery' {
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        if ($isAdmin) {
            $result = Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -WhatIf
            $result.Recovered | Should -BeFalse
            $result.Verified | Should -BeFalse
            $result.Cycles.Count | Should -Be 0
            $result.Message | Should -BeLike 'Skipped:*'
        } else {
            { Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -WhatIf } | Should -Throw '*requires an elevated session*'
            Should -Invoke Get-NetAdapter -Exactly -Times 0
        }
        Should -Invoke Disable-PnpDevice -Exactly -Times 0
        Should -Invoke Enable-PnpDevice -Exactly -Times 0
        Should -Invoke Start-Sleep -Exactly -Times 0
    }
    It 'requires an explicit throughput pass threshold before any measurement or cycle' {
        $script:Measurements = 0
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        $message = if ($isAdmin) { '*MinAcceptableMbits*threshold*' } else { '*requires an elevated session*' }
        { Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -ThroughputTest { $script:Measurements++; 100 } -WhatIf } | Should -Throw $message
        $script:Measurements | Should -Be 0
        Should -Invoke Disable-PnpDevice -Exactly -Times 0
    }
    It 'does not run a valid supplied throughput measurement under WhatIf' {
        $script:Measurements = 0
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        if ($isAdmin) {
            $result = Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -ThroughputTest { $script:Measurements++; 100 } -MinAcceptableMbits 700 -WhatIf
            $result.Verified | Should -BeFalse
        } else {
            { Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -ThroughputTest { $script:Measurements++; 100 } -MinAcceptableMbits 700 -WhatIf } | Should -Throw '*requires an elevated session*'
        }
        $script:Measurements | Should -Be 0
        Should -Invoke Disable-PnpDevice -Exactly -Times 0
        Should -Invoke Enable-PnpDevice -Exactly -Times 0
    }
    It 'never verifies recovery from plausible throughput emitted by a failed native child' {
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        if (-not $isAdmin) {
            { Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -TestPeer 'never-network.invalid' -MinAcceptableMbits 700 -WhatIf } | Should -Throw '*requires an elevated session*'
        } else {
            $fixture = Join-Path $TestDrive 'failed-native-repair'
            New-Item -ItemType Directory -Path $fixture | Out-Null
            $native = Join-Path $fixture 'iperf3.cmd'
            @('@echo off', 'echo [SUM] 0.00-1.00 sec 100 MBytes 1000 Mbits/sec receiver', 'exit /b 7') | Set-Content $native -Encoding ascii
            $oldPath = $env:PATH
            try {
                $env:PATH = $fixture + [IO.Path]::PathSeparator + $oldPath
                $result = Repair-UsbNetAdapter -InterfaceGuid '{22222222-2222-4222-8222-222222222222}' -TestPeer 'never-network.invalid' -MinAcceptableMbits 700 -Confirm:$false
                $LASTEXITCODE | Should -Be 7
                $result.Verified | Should -BeFalse
                $result.Recovered | Should -BeFalse
            } finally { $env:PATH = $oldPath }
        }
        Should -Invoke Disable-PnpDevice -Exactly -Times 0
        Should -Invoke Enable-PnpDevice -Exactly -Times 0
    }
}

AfterAll {
    Remove-Module PC-AI.Network -Force -ErrorAction SilentlyContinue
}
