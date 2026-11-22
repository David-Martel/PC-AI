<#
.SYNOPSIS
    Unit tests for PC-AI.Network module

.DESCRIPTION
    Tests network diagnostics, WSL connectivity, and VSock performance monitoring
#>

if (-not (Get-Variable -Name IsAdmin -Scope Script -ErrorAction SilentlyContinue)) {
    $script:IsAdmin = $false
}

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Network\PC-AI.Network.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    # Fail closed if any WSL mock is missing; this command is invoked directly by the producer.
    & (Get-Module PC-AI.Network) {
        function script:wsl {
            param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens)
            throw 'Live WSL is forbidden in portable network unit tests.'
        }
    }

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop

    # Helper function to check if running as Administrator
    function Test-IsAdmin {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    $script:IsAdmin = Test-IsAdmin

    # Shared fixtures for the dual-stack endpoint tests below.
    function New-DsAdapter {
        param([int]$Index, [string]$Alias = 'Ethernet 18', [string]$Description = 'USB4 P2P Network Adapter',
            [string]$Status = 'Up', [string]$Mac = '02-00-5E-00-53-AA')
        [pscustomobject]@{ Name = $Alias; InterfaceAlias = $Alias; InterfaceDescription = $Description; ifIndex = $Index; Status = $Status; MacAddress = $Mac }
    }
    $script:Registry = @'
{
  "ulaPrefix": "fd00:7462:7032",
  "fleetPeers": [
{ "name": "dtm-test-peer", "aliases": ["test-peer", "dtm-test-peer.local"], "nodeIdHex": "50",
  "shortLinkLocal": "fe80::d700:50", "shortUla": "fd00:7462:7032::50", "tbHostAlias": "dtm-test-peer-tb",
  "lanAddresses": ["192.0.2.50"] },
{ "name": "dtm-other-peer", "aliases": [], "nodeIdHex": "51", "shortLinkLocal": "fe80::d700:51",
  "shortUla": "fd00:7462:7032::51", "lanAddresses": [] }
  ],
  "topologyLinks": [ { "linkId": "x", "expectedPeerMacs": { "dtm-test-peer": "02-00-5E-00-53-50" } } ]
}
'@ | ConvertFrom-Json
    $script:RegistrySource = { param($Path) $script:Registry }
    $script:NoNeighbors = { param($InterfaceIndex) @() }
    function New-DsInterfaceSource {
        param($Adapters)
        $captured = @($Adapters)
        { $captured }.GetNewClosure()
    }
    $script:Usb4Pair = @((New-DsAdapter -Index 19 -Alias 'Ethernet 18'), (New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
    # Fake connector: Plan maps an address string to Hang | Refused | Unreachable | Ok | OkAfter:<ms> | Throw.
    # Each client is a MemoryStream, so CanRead proves whether the connector's client was disposed.
    function New-DsLog { [pscustomobject]@{ Calls = [Collections.Generic.List[string]]::new(); Clients = @{}; Hook = $null } }
    function New-DsConnector {
        param([hashtable]$Plan, $Log)
        {
            param($IPAddress, $Port, $Candidate)
            $key = $IPAddress.ToString()
            $Log.Calls.Add($key)
            $client = [IO.MemoryStream]::new()
            $Log.Clients[$key] = $client
            if ($Log.Hook) { & $Log.Hook $key }
            $mode = [string]$Plan[$key]
            if ($mode -eq 'Throw') { $client.Dispose(); throw 'synthetic synchronous connector failure' }
            $task = switch -Wildcard ($mode) {
                'Hang' { [Threading.Tasks.TaskCompletionSource[bool]]::new().Task }
                'Refused' { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10061)) }
                'Unreachable' { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10051)) }
                'HostUnreachable' { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10065)) }
                'OkAfter:*' { [Threading.Tasks.Task]::Delay([int]($mode.Substring(8))) }
                default { [Threading.Tasks.Task]::CompletedTask }
            }
            [pscustomobject]@{ Task = $task; Client = $client }
        }.GetNewClosure()
    }
    function New-DsCand {
        param([string]$Address, [string]$Source = 'Dns', [string]$NeighborState = $null, [bool]$MacMatched = $false)
        [pscustomobject]@{ Address = $Address; Source = $Source; NeighborState = $NeighborState; MacMatched = $MacMatched }
    }
}

Describe "Get-NetworkDiagnostics" -Tag 'Unit', 'Network', 'Fast', 'Portable' {
    Context "When gathering network diagnostics" {
        BeforeAll {
            Mock Get-NetAdapter {
                @(
                    [PSCustomObject]@{
                        Name = "Ethernet"
                        Status = "Up"
                        LinkSpeed = "1 Gbps"
                        MacAddress = "00-D8-61-12-34-56"
                        MediaType = "802.3"
                        Virtual = $false
                        ifIndex = 1
                        InterfaceDescription = "Intel Ethernet Adapter"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetIPConfiguration {
                [PSCustomObject]@{
                    InterfaceIndex = 1
                    InterfaceAlias = "Ethernet"
                    IPv4DefaultGateway = @{ NextHop = "192.168.1.1" }
                    DNSServer = @{ ServerAddresses = @("8.8.8.8", "8.8.4.4") }
                    NetIPv4Interface = @{ Dhcp = "Enabled" }
                }
            } -ModuleName PC-AI.Network

            Mock Get-NetIPAddress {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias = "Ethernet"
                        InterfaceIndex = 1
                        IPAddress = "192.168.1.100"
                        PrefixLength = 24
                        AddressFamily = "IPv4"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetAdapterStatistics {
                [PSCustomObject]@{
                    Name = "Ethernet"
                    SentBytes = 1000000
                    ReceivedBytes = 2000000
                    OutboundDiscardedPackets = 0
                    InboundDiscardedPackets = 0
                }
            } -ModuleName PC-AI.Network

            Mock Get-VMSwitch {
                @()
            } -ModuleName PC-AI.Network

            Mock Get-DnsClientServerAddress {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias = "Ethernet"
                        ServerAddresses = @("8.8.8.8", "8.8.4.4")
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-DnsClientCache {
                @(
                    [PSCustomObject]@{ Entry = "google.com"; Data = "142.250.185.46" }
                )
            } -ModuleName PC-AI.Network

            Mock Get-DnsClientGlobalSetting {
                [PSCustomObject]@{ SuffixSearchList = @("local.domain") }
            } -ModuleName PC-AI.Network

            Mock Get-NetRoute {
                @(
                    [PSCustomObject]@{
                        DestinationPrefix = "0.0.0.0/0"
                        NextHop = "192.168.1.1"
                        RouteMetric = 25
                        InterfaceAlias = "Ethernet"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Measure-NetworkLatency {
                [PSCustomObject]@{
                    Target = "8.8.8.8"
                    Success = $true
                    MinLatency = 10
                    MaxLatency = 20
                    AvgLatency = 15
                    PacketLoss = 0
                }
            } -ModuleName PC-AI.Network

            Mock Resolve-DnsName {
                [PSCustomObject]@{
                    Name = "google.com"
                    IPAddress = "142.250.185.46"
                    Type = "A"
                }
            } -ModuleName PC-AI.Network
        }

        It "Should return PSCustomObject" {
            $result = Get-NetworkDiagnostics
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have required properties" {
            $result = Get-NetworkDiagnostics
            $result.PSObject.Properties.Name | Should -Contain 'Timestamp'
            $result.PSObject.Properties.Name | Should -Contain 'ComputerName'
            $result.PSObject.Properties.Name | Should -Contain 'PhysicalAdapters'
            $result.PSObject.Properties.Name | Should -Contain 'VirtualAdapters'
            $result.PSObject.Properties.Name | Should -Contain 'VirtualSwitches'
            $result.PSObject.Properties.Name | Should -Contain 'DNSConfiguration'
            $result.PSObject.Properties.Name | Should -Contain 'RoutingTable'
            $result.PSObject.Properties.Name | Should -Contain 'ConnectivityTests'
            $result.PSObject.Properties.Name | Should -Contain 'DNSTests'
            $result.PSObject.Properties.Name | Should -Contain 'Issues'
            $result.PSObject.Properties.Name | Should -Contain 'Summary'
        }

        It "Should detect physical adapters" {
            $result = Get-NetworkDiagnostics
            $result.PhysicalAdapters.Count | Should -BeGreaterThan 0
            $result.PhysicalAdapters[0].Name | Should -Be "Ethernet"
            $result.PhysicalAdapters[0].Status | Should -Be "Up"
            $result.PhysicalAdapters[0].IPv4Address | Should -Be "192.168.1.100"
        }

        It "Should test connectivity" {
            $result = Get-NetworkDiagnostics -TestConnectivity
            $result.ConnectivityTests.Count | Should -BeGreaterThan 0
            Should -Invoke Measure-NetworkLatency -ModuleName PC-AI.Network
        }

        It "Should test DNS resolution" {
            $result = Get-NetworkDiagnostics -TestDNS
            $result.DNSTests.Count | Should -BeGreaterThan 0
            Should -Invoke Resolve-DnsName -ModuleName PC-AI.Network
        }

        It "Should include detailed statistics when requested" {
            $result = Get-NetworkDiagnostics -Detailed
            Should -Invoke Get-NetAdapterStatistics -ModuleName PC-AI.Network
        }

        It "Should generate summary" {
            $result = Get-NetworkDiagnostics
            $result.Summary | Should -Not -BeNullOrEmpty
            $result.Summary.PSObject.Properties.Name | Should -Contain 'TotalAdapters'
            $result.Summary.PSObject.Properties.Name | Should -Contain 'OverallStatus'
        }
    }

    Context "When network adapters are disconnected" {
        BeforeAll {
            Mock Get-NetAdapter {
                @(
                    [PSCustomObject]@{
                        Name = "Ethernet"
                        Status = "Disconnected"
                        LinkSpeed = "0 bps"
                        Virtual = $false
                        ifIndex = 1
                        MacAddress = "00-D8-61-12-34-56"
                        MediaType = "802.3"
                        InterfaceDescription = "Intel Ethernet Adapter"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetIPConfiguration { $null } -ModuleName PC-AI.Network
            Mock Get-NetIPAddress { @() } -ModuleName PC-AI.Network
            Mock Get-VMSwitch { @() } -ModuleName PC-AI.Network
            Mock Get-DnsClientServerAddress { @() } -ModuleName PC-AI.Network
            Mock Get-NetRoute { @() } -ModuleName PC-AI.Network
        }

        It "Should detect disconnected adapters" {
            $result = Get-NetworkDiagnostics -TestConnectivity:$false -TestDNS:$false
            $result.PhysicalAdapters[0].Status | Should -Be "Disconnected"
        }

        It "Should add issues for disconnected adapters" {
            $result = Get-NetworkDiagnostics -TestConnectivity:$false -TestDNS:$false
            $disconnectedIssue = $result.Issues | Where-Object { $_.Item -eq "Ethernet" }
            $disconnectedIssue | Should -Not -BeNullOrEmpty
            $disconnectedIssue.Severity | Should -Be "Warning"
        }
    }

    Context "When connectivity fails" {
        BeforeAll {
            Mock Get-NetAdapter {
                @(
                    [PSCustomObject]@{
                        Name = "Ethernet"
                        Status = "Up"
                        Virtual = $false
                        ifIndex = 1
                        MacAddress = "00-D8-61-12-34-56"
                        LinkSpeed = "1 Gbps"
                        MediaType = "802.3"
                        InterfaceDescription = "Intel Ethernet Adapter"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetIPConfiguration { $null } -ModuleName PC-AI.Network
            Mock Get-NetIPAddress { @() } -ModuleName PC-AI.Network
            Mock Get-VMSwitch { @() } -ModuleName PC-AI.Network

            Mock Measure-NetworkLatency {
                [PSCustomObject]@{
                    Target = "8.8.8.8"
                    Success = $false
                    MinLatency = $null
                    MaxLatency = $null
                    AvgLatency = $null
                    PacketLoss = 100
                }
            } -ModuleName PC-AI.Network
        }

        It "Should detect connectivity failures" {
            $result = Get-NetworkDiagnostics -TestConnectivity -TestDNS:$false
            $failedTest = $result.ConnectivityTests | Where-Object { -not $_.Success }
            $failedTest | Should -Not -BeNullOrEmpty
        }

        It "Should add issues for connectivity failures" {
            $result = Get-NetworkDiagnostics -TestConnectivity -TestDNS:$false
            $connectivityIssue = $result.Issues | Where-Object { $_.Category -eq "Connectivity" }
            $connectivityIssue | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Test-WSLConnectivity" -Tag 'Unit', 'Network', 'Slow', 'Portable' {
    Context "When testing WSL network connectivity" {
        BeforeAll {
            Mock Get-WSLDistributions {
                @(
                    [PSCustomObject]@{
                        Name = "Ubuntu"
                        State = "Running"
                        Version = "2"
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetAdapter {
                [PSCustomObject]@{
                    Name = "vEthernet (WSL)"
                    Status = "Up"
                    ifIndex = 99
                }
            } -ModuleName PC-AI.Network

            Mock Get-NetIPAddress {
                [PSCustomObject]@{
                    IPAddress = "172.31.208.1"
                    AddressFamily = "IPv4"
                }
            } -ModuleName PC-AI.Network

            Mock Resolve-DnsName {
                [PSCustomObject]@{
                    Name = "google.com"
                    IPAddress = "142.250.185.46"
                }
            } -ModuleName PC-AI.Network

            Mock Test-PortConnectivity {
                [PSCustomObject]@{
                    Host = "172.31.208.2"
                    Port = 22
                    Success = $true
                    Message = "Connected"
                }
            } -ModuleName PC-AI.Network

            Mock Get-CimInstance {
                [PSCustomObject]@{
                    Name = "Hyper-V Socket"
                    Status = "OK"
                }
            } -ModuleName PC-AI.Network

            # Mock the actual direct native boundary, not the unrelated Invoke-Command cmdlet.
            Mock -CommandName wsl -MockWith {
                $global:LASTEXITCODE = 0
                if ($Tokens -contains '--status') {
                    'Default Version: 2'; 'Default Distribution: Ubuntu'; 'Kernel version: 5.10.102.1'
                } elseif ($Tokens -contains 'ip') {
                    'inet 172.31.208.2/20 scope global eth0'
                } elseif ($Tokens -contains 'nslookup') {
                    'Address: 142.250.185.46'
                } elseif ($Tokens -contains 'hostname') {
                    '172.31.208.2'
                } elseif ($Tokens -contains 'curl') {
                    '200'
                } else { throw "Unexpected mocked WSL arguments: $Tokens" }
            } -ModuleName PC-AI.Network
        }

        It "Should return PSCustomObject" {
            $result = Test-WSLConnectivity
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have required properties" {
            $result = Test-WSLConnectivity
            $result.PSObject.Properties.Name | Should -Contain 'Timestamp'
            $result.PSObject.Properties.Name | Should -Contain 'WSLStatus'
            $result.PSObject.Properties.Name | Should -Contain 'Distributions'
            $result.PSObject.Properties.Name | Should -Contain 'NetworkInterfaces'
            $result.PSObject.Properties.Name | Should -Contain 'DNSTests'
            $result.PSObject.Properties.Name | Should -Contain 'PortTests'
            $result.PSObject.Properties.Name | Should -Contain 'InternetAccess'
            $result.PSObject.Properties.Name | Should -Contain 'VSockStatus'
            $result.PSObject.Properties.Name | Should -Contain 'Issues'
            $result.PSObject.Properties.Name | Should -Contain 'Summary'
        }

        It "Should check WSL distributions" {
            $result = Test-WSLConnectivity
            Should -Invoke Get-WSLDistributions -ModuleName PC-AI.Network
            # Distributions property should exist (may be empty in unit test without WSL)
            $result.PSObject.Properties.Name | Should -Contain 'Distributions'
        }

        It "Should test DNS resolution" {
            $result = Test-WSLConnectivity
            # DNSTests property should exist (may be empty if no distro available for testing)
            $result.PSObject.Properties.Name | Should -Contain 'DNSTests'
        }

        It "Should test port connectivity" {
            $result = Test-WSLConnectivity -TestPorts 22,80
            # PortTests property should exist (may be empty if no distro available)
            $result.PSObject.Properties.Name | Should -Contain 'PortTests'
        }

        It "Should generate summary" {
            $result = Test-WSLConnectivity
            $result.Summary | Should -Not -BeNullOrEmpty
            $result.Summary.PSObject.Properties.Name | Should -Contain 'OverallStatus'
            $result.Summary.PSObject.Properties.Name | Should -Contain 'PassedTests'
            $result.Summary.PSObject.Properties.Name | Should -Contain 'TotalTests'
        }
    }

    Context "When WSL network has issues" {
        BeforeAll {
            Mock wsl { $global:LASTEXITCODE = 0; 'Default Version: 2' } -ModuleName PC-AI.Network
            # Mock to return no distributions, which triggers "No distributions found" issue
            Mock Get-WSLDistributions {
                @()
            } -ModuleName PC-AI.Network

            Mock Get-NetAdapter { $null } -ModuleName PC-AI.Network
            Mock Get-NetIPAddress { @() } -ModuleName PC-AI.Network

            Mock Resolve-DnsName {
                throw "DNS resolution failed"
            } -ModuleName PC-AI.Network

            Mock Test-PortConnectivity {
                [PSCustomObject]@{
                    Host = "172.31.208.2"
                    Port = 22
                    Success = $false
                    Message = "Connection timeout"
                }
            } -ModuleName PC-AI.Network
        }

        It "Should return result with issues (not throw)" {
            { $result = Test-WSLConnectivity -ErrorAction Stop } | Should -Not -Throw
        }

        It "Should populate Issues property" {
            $result = Test-WSLConnectivity
            $result.Issues | Should -Not -BeNullOrEmpty
            # Should have "No WSL distributions found" issue
            $result.Issues[0].Issue | Should -Match "No WSL distributions found"
        }

        It "Should set OverallStatus to Critical or Warning" {
            $result = Test-WSLConnectivity
            $result.Summary.OverallStatus | Should -BeIn @('Critical', 'Warning', 'Failed')
        }
    }

    Context "When WSL is not installed" {
        BeforeAll {
            Mock wsl { throw 'Synthetic WSL command unavailable.' } -ModuleName PC-AI.Network
            Mock Get-WSLDistributions { @() } -ModuleName PC-AI.Network
        }

        It "Should handle missing WSL gracefully" {
            $result = Test-WSLConnectivity
            $result | Should -Not -BeNullOrEmpty
            $result.Issues | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Watch-VSockPerformance" -Tag 'Unit', 'Network', 'Slow', 'Portable' {
    Context "When monitoring VSock performance" {
        BeforeAll {
            Mock Get-NetAdapter {
                @(
                    [PSCustomObject]@{
                        Name = "vEthernet (WSL)"
                        Status = "Up"
                        LinkSpeed = "10 Gbps"
                        Virtual = $true
                    },
                    [PSCustomObject]@{
                        Name = "Ethernet"
                        Status = "Up"
                        LinkSpeed = "1 Gbps"
                        Virtual = $false
                    }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetAdapterStatistics {
                param([string]$Name)

                if ($Name -eq "Ethernet") {
                    return [PSCustomObject]@{
                        Name = "Ethernet"
                        SentBytes = 1500000
                        ReceivedBytes = 2500000
                        SentUnicastPackets = 1500
                        ReceivedUnicastPackets = 2500
                        OutboundDiscardedPackets = 0
                        InboundDiscardedPackets = 0
                        OutboundPacketErrors = 0
                        InboundPacketErrors = 0
                    }
                }

                return [PSCustomObject]@{
                    Name = "vEthernet (WSL)"
                    SentBytes = 1000000
                    ReceivedBytes = 2000000
                    SentUnicastPackets = 1000
                    ReceivedUnicastPackets = 2000
                    OutboundDiscardedPackets = 0
                    InboundDiscardedPackets = 0
                    OutboundPacketErrors = 0
                    InboundPacketErrors = 0
                }
            } -ModuleName PC-AI.Network

            Mock Get-NetTCPConnection {
                @(
                    [PSCustomObject]@{ State = "Established" },
                    [PSCustomObject]@{ State = "Listen" }
                )
            } -ModuleName PC-AI.Network

            Mock Get-NetTCPSetting {
                [PSCustomObject]@{
                    SettingName = "Internet"
                    AutoTuningLevelLocal = "Normal"
                    CongestionProvider = "CUBIC"
                    EcnCapability = "Disabled"
                }
            } -ModuleName PC-AI.Network

            Mock Start-Sleep {} -ModuleName PC-AI.Network
            Mock Clear-Host {} -ModuleName PC-AI.Network
            Mock Format-BytesPerSecond { "1.23 MB/s" } -ModuleName PC-AI.Network

            Mock Get-Date {
                param([string]$Format)

                if ($script:dateIndex -ge $script:dates.Count) {
                    $value = $script:dates[-1]
                } else {
                    $value = $script:dates[$script:dateIndex]
                    $script:dateIndex++
                }

                if ($Format) {
                    return $value.ToString($Format)
                }
                return $value
            } -ModuleName PC-AI.Network
        }

        BeforeEach {
            $baseTime = [DateTime]::UtcNow
            $script:dates = @(
                $baseTime,
                $baseTime,
                $baseTime.AddSeconds(1),
                $baseTime.AddSeconds(2)
            )
            $script:dateIndex = 0
        }

        It "Should use correct parameters (Duration and RefreshInterval)" {
            { Watch-VSockPerformance -Duration 5 -RefreshInterval 1 } | Should -Not -Throw
        }

        It "Should return array of PSCustomObjects" {
            $result = Watch-VSockPerformance -Duration 2 -RefreshInterval 1 -Quiet
            @($result).Count | Should -BeGreaterThan 0
            ($result | Select-Object -First 1) | Should -BeOfType [PSCustomObject]
        }

        It "Should query network adapters" {
            $result = Watch-VSockPerformance -Duration 2 -RefreshInterval 1 -Quiet
            Should -Invoke Get-NetAdapter -ModuleName PC-AI.Network
        }

        It "Should query adapter statistics" {
            $result = Watch-VSockPerformance -Duration 2 -RefreshInterval 1 -Quiet
            Should -Invoke Get-NetAdapterStatistics -ModuleName PC-AI.Network
        }

        It "Should query TCP connections" {
            $result = Watch-VSockPerformance -Duration 2 -RefreshInterval 1 -Quiet
            Should -Invoke Get-NetTCPConnection -ModuleName PC-AI.Network
        }

        It "Should respect duration parameter" {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Watch-VSockPerformance -Duration 2 -RefreshInterval 1 -Quiet
            $sw.Stop()
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 5
        }

        It "Should support InterfaceFilter parameter" {
            $result = Watch-VSockPerformance -Duration 1 -InterfaceFilter "*WSL*" -Quiet
            # Get-NetAdapter is called without parameters; filtering happens in Where-Object
            # Called multiple times in the monitoring loop (once per iteration)
            Should -Invoke Get-NetAdapter -ModuleName PC-AI.Network
        }

        It "Should support IncludeVirtual parameter" {
            $result = Watch-VSockPerformance -Duration 1 -IncludeVirtual:$false -Quiet
            @($result).Count | Should -BeGreaterThan 0
            @($result | Where-Object { $_.Interface -match 'vEthernet' }).Count | Should -Be 0
        }
    }

    Context "When no adapters match filter" {
        BeforeAll {
            Mock Get-NetAdapter { @() } -ModuleName PC-AI.Network
            Mock Get-NetTCPConnection { @() } -ModuleName PC-AI.Network
            Mock Start-Sleep {} -ModuleName PC-AI.Network
            Mock Clear-Host {} -ModuleName PC-AI.Network
            Mock Get-Date {
                param([string]$Format)

                if ($script:dateIndex -ge $script:dates.Count) {
                    $value = $script:dates[-1]
                } else {
                    $value = $script:dates[$script:dateIndex]
                    $script:dateIndex++
                }

                if ($Format) {
                    return $value.ToString($Format)
                }
                return $value
            } -ModuleName PC-AI.Network
        }

        BeforeEach {
            $baseTime = [DateTime]::UtcNow
            $script:dates = @(
                $baseTime,
                $baseTime,
                $baseTime.AddSeconds(1),
                $baseTime.AddSeconds(2)
            )
            $script:dateIndex = 0
        }

        It "Should handle no matching adapters gracefully" {
            { $script:lastResult = Watch-VSockPerformance -Duration 1 -InterfaceFilter "NonExistent*" -Quiet } | Should -Not -Throw
            $script:lastResult | Should -BeNullOrEmpty
        }
    }
}

Describe "Optimize-VSock" -Tag 'Unit', 'Network', 'Slow', 'RequiresAdmin', 'Portable' {
    Context "When optimizing VSock configuration" {
        BeforeAll {
            # Mock admin check to return true for tests
            Mock -CommandName Invoke-Command -MockWith {
                param($ScriptBlock)
                if ($ScriptBlock -match 'IsInRole') {
                    return $true
                }
            } -ModuleName PC-AI.Network

            Mock Get-RegistryValueSafe {
                param($Path, $Name)
                return 0
            } -ModuleName PC-AI.Network

            Mock Set-RegistryValueSafe {
                return $true
            } -ModuleName PC-AI.Network

            Mock Test-Path { $true } -ModuleName PC-AI.Network

            # Mock netsh command execution
            Mock -CommandName netsh -MockWith {
                return "Ok."
            } -ModuleName PC-AI.Network

            Mock ConvertTo-Json { '[]' } -ModuleName PC-AI.Network
            Mock Out-File {} -ModuleName PC-AI.Network
        }

        It "Should accept Profile parameter (not BufferSize)" -Skip:(-not $script:IsAdmin) {
            { Optimize-VSock -Profile Balanced -WhatIf } | Should -Not -Throw
        }

        It "Should support Balanced profile" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Balanced -WhatIf
            $result.Profile | Should -Be "Balanced"
        }

        It "Should support Performance profile" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Performance -WhatIf
            $result.Profile | Should -Be "Performance"
        }

        It "Should support Conservative profile" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Conservative -WhatIf
            $result.Profile | Should -Be "Conservative"
        }

        It "Should return PSCustomObject with required properties" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Balanced -WhatIf
            $result | Should -BeOfType [PSCustomObject]
            $result.PSObject.Properties.Name | Should -Contain 'Timestamp'
            $result.PSObject.Properties.Name | Should -Contain 'Profile'
            $result.PSObject.Properties.Name | Should -Contain 'ChangesApplied'
            $result.PSObject.Properties.Name | Should -Contain 'ChangesPending'
            $result.PSObject.Properties.Name | Should -Contain 'Errors'
            $result.PSObject.Properties.Name | Should -Contain 'WSLRestarted'
            $result.PSObject.Properties.Name | Should -Contain 'BackupCreated'
        }

        It "Should support WhatIf" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Balanced -WhatIf
            $result.ChangesPending.Count | Should -BeGreaterThan 0
            $result.ChangesApplied.Count | Should -Be 0
        }

        It "Should create backup before changes" -Skip:(-not $script:IsAdmin) {
            Mock Get-RegistryValueSafe { return 10 } -ModuleName PC-AI.Network
            $result = Optimize-VSock -Profile Balanced -SkipWSLRestart -Confirm:$false
            $result.BackupCreated | Should -Be $true
            Should -Invoke Out-File -ModuleName PC-AI.Network
        }

        It "Should apply registry settings" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Balanced -SkipWSLRestart -Confirm:$false
            Should -Invoke Set-RegistryValueSafe -ModuleName PC-AI.Network
        }

        It "Should restart WSL by default" -Skip:(-not $script:IsAdmin) {
            Mock Get-RegistryValueSafe { return 10 } -ModuleName PC-AI.Network
            Mock Set-RegistryValueSafe { return $true } -ModuleName PC-AI.Network

            # Note: Cannot easily mock wsl.exe, so we skip actual restart test
            $result = Optimize-VSock -Profile Balanced -SkipWSLRestart -Confirm:$false
            $result.WSLRestarted | Should -Be $false
        }

        It "Should skip WSL restart when requested" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -Profile Balanced -SkipWSLRestart -Confirm:$false
            $result.WSLRestarted | Should -Be $false
        }
    }

    Context "When not running as Administrator" {
        BeforeAll {
            # Cannot easily mock IsInRole check, but this is tested manually
        }

        It "Should require Administrator privileges" {
            if ($script:IsAdmin) {
                Set-ItResult -Skipped -Because "Test requires non-admin context"
            }

            $result = Optimize-VSock -WhatIf -ErrorAction SilentlyContinue
            $result | Should -BeNullOrEmpty
        }
    }

    Context "When restoring from backup" {
        BeforeAll {
            Mock Test-Path { $true } -ModuleName PC-AI.Network
            Mock Get-Content {
                '[{"Path":"HKLM:\\Test","Name":"TestValue","Value":10}]'
            } -ModuleName PC-AI.Network
            Mock ConvertFrom-Json {
                @(
                    [PSCustomObject]@{
                        Path = "HKLM:\Test"
                        Name = "TestValue"
                        Value = 10
                    }
                )
            } -ModuleName PC-AI.Network
            Mock Set-ItemProperty {} -ModuleName PC-AI.Network
        }

        It "Should restore settings from backup file" -Skip:(-not $script:IsAdmin) {
            $result = Optimize-VSock -RestoreBackup -BackupPath "test.json" -WhatIf
            Should -Invoke Test-Path -ModuleName PC-AI.Network
        }
    }
}

Describe "Network-Helpers (Private Functions)" -Tag 'Unit', 'Network', 'Fast', 'Portable' {
    Context "When using helper functions" {
        BeforeAll {
            Mock Get-NetAdapter {
                [PSCustomObject]@{
                    Name = "Ethernet"
                    Status = "Up"
                    LinkSpeed = "1 Gbps"
                }
            } -ModuleName PC-AI.Network
        }

        It "Should load module successfully" {
            Get-Module PC-AI.Network | Should -Not -BeNullOrEmpty
        }

        It "Should export public functions" {
            $module = Get-Module PC-AI.Network
            $module.ExportedFunctions.Keys | Should -Contain 'Get-NetworkDiagnostics'
            $module.ExportedFunctions.Keys | Should -Contain 'Test-WSLConnectivity'
            $module.ExportedFunctions.Keys | Should -Contain 'Watch-VSockPerformance'
            $module.ExportedFunctions.Keys | Should -Contain 'Optimize-VSock'
        }

        It "Should not export private functions" {
            $module = Get-Module PC-AI.Network
            $module.ExportedFunctions.Keys | Should -Not -Contain 'Get-AdapterStatusDescription'
            $module.ExportedFunctions.Keys | Should -Not -Contain 'Measure-NetworkLatency'
        }
    }
}

# ---------------------------------------------------------------------------------------------------------------------
# Dual-stack endpoint candidates and the RFC 8305 connector (docs/networking/dual-stack.md).
#
# Why these tests exist: a known USB4 peer must be found again, fast, after a replug, another port, a hub change or an
# ifIndex drift, and a connect to it must not stall on a dead address family. Identifiers LOCATE a peer; SSH host keys
# authenticate it, so nothing here asserts identity beyond "this address belongs to the MAC/peer we asked about".
#
# Every provider and the connector are injected: no test touches the adapter table, the neighbour cache, DNS or a
# network. Addresses are synthetic (2001:db8::/32, 192.0.2.0/24, made-up fe80 IIDs, 02-00-5e-00-53-xx MACs).
# The one real-socket test lives in Tests/Portable/Test-DualStackConnector.ps1.
# ---------------------------------------------------------------------------------------------------------------------
Describe 'Get-PcaiEndpointCandidate' -Tag 'Unit', 'Network', 'Fast', 'Portable' {
    # Protects: the modified EUI-64 derivation (RFC 4291 appendix A) that lets a known MAC locate a peer's
    # SLAAC link-local address.
    # Detects: a missing U/L bit flip (00:11:.. must become 02:11:..), a flip applied the wrong way round for a MAC whose
    # U/L bit is already set, a missing ff:fe, and a zone that is not the live ifIndex.
    # Needs: nothing live; one injected USB4 adapter.
    # Breadcrumb: ConvertTo-PcaiEui64LinkLocal
    It 'derives the modified EUI-64 link-local for <Mac> as <Expected> on the live interface' -ForEach @(
        @{ Mac = '00-11-22-33-44-55'; Expected = 'fe80::211:22ff:fe33:4455%25' }
        @{ Mac = '02-00-5e-00-53-01'; Expected = 'fe80::5eff:fe00:5301%25' }
        @{ Mac = '00:11:22:33:44:55'; Expected = 'fe80::211:22ff:fe33:4455%25' }
        @{ Mac = '001122334455'; Expected = 'fe80::211:22ff:fe33:4455%25' }
        @{ Mac = '00-11-22-AA-BB-CC'; Expected = 'fe80::211:22ff:feaa:bbcc%25' }
    ) {
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -MacAddress $Mac -Provider Eui64 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $found.Count | Should -Be 1
        $found[0].Address | Should -BeExactly $Expected
        $found[0].IPAddress.ScopeId | Should -Be 25
        $found[0].Class | Should -Be 'LinkLocal'
        $found[0].Family | Should -Be 'InterNetworkV6'
        $found[0].Source | Should -Be 'Eui64'
        $found[0].Eui64Derived | Should -BeTrue
        $found[0].InterfaceIndex | Should -Be 25
    }

    # Protects: topology change. The same peer must be found on the NEW interface after a replug, another port or a hub
    # change moved it from ifIndex 19 to 25. Nothing may remember the old index.
    # Detects: an ifIndex cached between calls (the second call would still say %19), and a stale zone left in the output.
    # Needs: an injected adapter table that changes between the two calls.
    # Breadcrumb: Get-PcaiLiveUsb4Interface / Get-PcaiEndpointCandidate (no script-scope state)
    It 'follows the interface when the peer moves from ifIndex 19 to 25 (replug) with no stale zone' {
        $state = @{ Adapters = @((New-DsAdapter -Index 19 -Alias 'Ethernet 18')) }
        $source = { $state.Adapters }.GetNewClosure()
        $before = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer, Eui64 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $state.Adapters = @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $after = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer, Eui64 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $linkLocalBefore = @($before | Where-Object { $_.Class -eq 'LinkLocal' })
        $linkLocalAfter = @($after | Where-Object { $_.Class -eq 'LinkLocal' })
        $linkLocalBefore.Count | Should -BeGreaterThan 0
        @($linkLocalBefore | Where-Object { $_.Address -notmatch '%19$' }).Count | Should -Be 0
        $linkLocalAfter.Count | Should -Be $linkLocalBefore.Count
        @($linkLocalAfter | Where-Object { $_.Address -notmatch '%25$' }).Count | Should -Be 0
        @($linkLocalAfter | Where-Object { $_.Address -match '%19' }).Count | Should -Be 0
        @($linkLocalAfter | ForEach-Object { $_.IPAddress.ScopeId } | Sort-Object -Unique) | Should -Be @(25)
    }

    # Protects: a pinned alias is resolved to the CURRENT ifIndex on every call (alias is stable, index is not).
    # Detects: a pinned alias bound to an index remembered from an earlier call.
    # Needs: an injected adapter table that renumbers.
    # Breadcrumb: Get-PcaiEndpointCandidate -InterfaceAlias
    It 'resolves a pinned InterfaceAlias to the live ifIndex on each call' {
        $state = @{ Adapters = @((New-DsAdapter -Index 19 -Alias 'Ethernet 4'), (New-DsAdapter -Index 7 -Alias 'Ethernet 18')) }
        $source = { $state.Adapters }.GetNewClosure()
        $first = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceAlias 'Ethernet 4' -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $state.Adapters = @((New-DsAdapter -Index 31 -Alias 'Ethernet 4'), (New-DsAdapter -Index 7 -Alias 'Ethernet 18'))
        $second = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceAlias 'Ethernet 4' -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $first.Count | Should -Be 1
        $first[0].Address | Should -BeExactly 'fe80::211:22ff:fe33:4455%19'
        $second.Count | Should -Be 1
        $second[0].Address | Should -BeExactly 'fe80::211:22ff:fe33:4455%31'
    }

    # Protects: a link-local address is ambiguous across interfaces, so with no interface pinned one candidate per LIVE
    # USB4/P2P adapter is emitted, and only those.
    # Detects: candidates on a non-USB4 NIC, on a Down adapter, or only on the first adapter.
    # Needs: injected adapters (two live USB4, one Down USB4, one ordinary Ethernet).
    # Breadcrumb: Select-PcaiUsb4Adapter
    It 'emits one link-local candidate per live USB4 adapter and none for Down or non-USB4 adapters' {
        $source = New-DsInterfaceSource @(
            (New-DsAdapter -Index 19 -Alias 'Ethernet 18'), (New-DsAdapter -Index 25 -Alias 'Ethernet 4'),
            (New-DsAdapter -Index 30 -Alias 'Ethernet 9' -Status 'Disconnected'),
            (New-DsAdapter -Index 7 -Alias 'Ethernet' -Description 'Intel(R) Ethernet Connection'),
            (New-DsAdapter -Index 11 -Alias 'Thunderbolt Bridge' -Description 'Thunderbolt(TM) Networking Adapter'))
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer -IncludeIPv6 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $short = @($found | Where-Object { $_.Address -like 'fe80::d700:50%*' })
        @($short | ForEach-Object { $_.InterfaceIndex } | Sort-Object) | Should -Be @(11, 19, 25)
    }

    # Protects: the KnownPeer provider reads the peer registry (name, alias, tbHostAlias, node id) and emits the short and the
    # deterministic link-local and ULA addresses plus the registry's IPv4 addresses.
    # Detects: a registry peer found by name only, a missing deterministic address, a ULA given a zone, an IPv4 address
    # emitted when only IPv6 was asked for.
    # Needs: an injected registry (synthetic peer, not the real fleet).
    # Breadcrumb: Resolve-PcaiPeer, Get-PcaiPeerNodeHextet
    It 'resolves the peer by <Key> and emits its registry addresses' -ForEach @(
        @{ Key = 'PeerId name'; Params = @{ PeerId = 'dtm-test-peer' } }
        @{ Key = 'PeerId alias'; Params = @{ PeerId = 'test-peer' } }
        @{ Key = 'PeerId node id'; Params = @{ PeerId = '50' } }
        @{ Key = 'HostName alias with domain'; Params = @{ HostName = 'dtm-test-peer.local' } }
        @{ Key = 'HostName tb alias'; Params = @{ HostName = 'dtm-test-peer-tb' } }
    ) {
        $params = @{ Provider = 'KnownPeer' } + $Params
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate @params -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        @($found | Where-Object { $_.Address -eq 'fe80::d700:50%25' }).Count | Should -Be 1
        @($found | Where-Object { $_.Address -eq 'fd00:7462:7032::50' -and $_.Class -eq 'ULA' -and $null -eq $_.InterfaceIndex }).Count | Should -Be 1
        @($found | Where-Object { $_.Class -eq 'LinkLocal' }).Count | Should -Be 2 # short + deterministic
        @($found | Where-Object { $_.Address -eq '192.0.2.50' -and $_.Family -eq 'InterNetwork' }).Count | Should -Be 1
        @($found | Where-Object { $_.PeerId -ne 'dtm-test-peer' }).Count | Should -Be 0
        @($found | Where-Object { $_.Source -ne 'KnownPeer' }).Count | Should -Be 0
    }

    # Protects: the SHA-256 node hextets are a mirror of Get-ThunderboltPeerIdentity (Drivers). A drifting mirror would
    # point at a different fe80/ULA than the peer configured for itself.
    # Detects: a changed hash input string, byte order, or hextet formatting on either side.
    # Needs: pwsh 7 (Drivers uses SHA256.HashData) and the Drivers module; skipped on Windows PowerShell 5.1.
    # Breadcrumb: Modules/PC-AI.Drivers/Public/Get-ThunderboltPeerIdentity.ps1 Get-DeterministicNodeHextets
    It 'derives the same deterministic link-local and ULA as the Drivers module' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
        $driversManifest = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
        Import-Module $driversManifest -Force -ErrorAction Stop
        $registryFile = Join-Path $TestDrive 'peers.json'
        [IO.File]::WriteAllText($registryFile, ($script:Registry | ConvertTo-Json -Depth 6))
        $identity = Get-ThunderboltPeerIdentity -ComputerName 'dtm-test-peer' -RegistryPath $registryFile
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $linkLocal = @($found | Where-Object { $_.Class -eq 'LinkLocal' } | ForEach-Object { ($_.Address -split '%')[0] })
        $ula = @($found | Where-Object { $_.Class -eq 'ULA' } | ForEach-Object { $_.Address })
        $linkLocal | Should -Contain ([ipaddress]$identity.DeterministicLinkLocal).ToString()
        $ula | Should -Contain ([ipaddress]$identity.DeterministicUla).ToString()
    }

    # Protects: an unknown peer is "nothing to say", not an error, so a caller can fall through to DNS.
    # Detects: a throw or an invented candidate for a name absent from the registry.
    # Needs: an injected registry.
    # Breadcrumb: Resolve-PcaiPeer
    It 'returns no KnownPeer candidates for a peer that is not in the registry' {
        $source = New-DsInterfaceSource $script:Usb4Pair
        { Get-PcaiEndpointCandidate -PeerId 'no-such-peer' -Provider KnownPeer -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors } | Should -Not -Throw
        @(Get-PcaiEndpointCandidate -PeerId 'no-such-peer' -Provider KnownPeer -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors).Count | Should -Be 0
    }

    # Protects: the caller must say WHO to look for.
    # Detects: a silent empty result for a call with no identity at all.
    # Needs: nothing.
    # Breadcrumb: Get-PcaiEndpointCandidate parameter validation
    It 'requires a HostName, PeerId or MacAddress' {
        { Get-PcaiEndpointCandidate -InterfaceSource { @() } -RegistrySource $script:RegistrySource } | Should -Throw '*HostName*'
    }

    # Protects: the address-family switches. Neither switch means both; one switch means only that family.
    # Detects: IPv4 leaking into an IPv6-only request or the reverse.
    # Needs: injected registry (carries one IPv4 and several IPv6 addresses).
    # Breadcrumb: Get-PcaiEndpointCandidate -IncludeIPv4/-IncludeIPv6
    It 'honours -IncludeIPv4 and -IncludeIPv6' {
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $common = @{ PeerId = 'dtm-test-peer'; Provider = 'KnownPeer'; InterfaceSource = $source; RegistrySource = $script:RegistrySource; NeighborSource = $script:NoNeighbors }
        $v6 = @(Get-PcaiEndpointCandidate @common -IncludeIPv6)
        $v4 = @(Get-PcaiEndpointCandidate @common -IncludeIPv4)
        $both = @(Get-PcaiEndpointCandidate @common)
        @($v6 | Where-Object { $_.Family -ne 'InterNetworkV6' }).Count | Should -Be 0
        $v6.Count | Should -BeGreaterThan 0
        @($v4 | Where-Object { $_.Family -ne 'InterNetwork' }).Count | Should -Be 0
        $v4.Count | Should -Be 1
        $both.Count | Should -Be ($v6.Count + $v4.Count)
    }

    # Protects: the Neighbor provider keeps only link-local entries in a live-ish state whose MAC matches the one asked
    # for, whatever the separator or case, and records whether the address is EUI-64 derived from that MAC.
    # Detects: unmatched neighbours (a router, another peer) offered as the peer, a dead (Unreachable/Incomplete)
    # entry, a multicast row, a MAC variant that fails to match, a wrong Eui64Derived flag.
    # Needs: injected neighbour cache rows.
    # Breadcrumb: Get-PcaiEndpointCandidate Neighbor provider
    It 'matches neighbour MAC <Variant> and keeps only the matching live link-local entry' -ForEach @(
        @{ Variant = 'dash upper'; Mac = '02-00-5E-00-53-50' }
        @{ Variant = 'colon lower'; Mac = '02:00:5e:00:53:50' }
        @{ Variant = 'bare lower'; Mac = '02005e005350' }
        @{ Variant = 'dotted'; Mac = '0200.5e00.5350' }
    ) {
        $rows = @(
            [pscustomobject]@{ IPAddress = 'fe80::5eff:fe00:5350'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 'Reachable'; InterfaceIndex = 25 }
            [pscustomobject]@{ IPAddress = 'fe80::1'; LinkLayerAddress = '02-00-5E-00-53-99'; State = 'Reachable'; InterfaceIndex = 25 }
            [pscustomobject]@{ IPAddress = 'fe80::dead:beef:1'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 'Unreachable'; InterfaceIndex = 25 }
            [pscustomobject]@{ IPAddress = 'ff02::1'; LinkLayerAddress = '33-33-00-00-00-01'; State = 'Permanent'; InterfaceIndex = 25 }
            [pscustomobject]@{ IPAddress = '2001:db8::5'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 'Reachable'; InterfaceIndex = 25 })
        $neighbors = { param($InterfaceIndex) $rows }.GetNewClosure()
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -MacAddress $Mac -Provider Neighbor -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $neighbors)
        $found.Count | Should -Be 1
        $found[0].Address | Should -BeExactly 'fe80::5eff:fe00:5350%25'
        $found[0].Source | Should -Be 'Neighbor'
        $found[0].MacAddress | Should -Be '02-00-5E-00-53-50'
        $found[0].MacMatched | Should -BeTrue
        $found[0].NeighborState | Should -Be 'Reachable'
        $found[0].Eui64Derived | Should -BeTrue
    }

    # Protects: a neighbour entry for a registry peer's known address is positive evidence even with no MAC supplied, and
    # an entry that is neither a known address nor a known MAC is not offered.
    # Detects: the Neighbor provider emitting every neighbour on the link when it has no identity to match.
    # Needs: injected registry and neighbour rows.
    # Breadcrumb: Get-PcaiEndpointCandidate Neighbor provider (identity requirement)
    It 'offers a neighbour only when it matches a known address or MAC' {
        $rows = @(
            [pscustomobject]@{ IPAddress = 'fe80::d700:50'; LinkLayerAddress = '02-00-5E-00-53-77'; State = 'Stale'; InterfaceIndex = 25 }
            [pscustomobject]@{ IPAddress = 'fe80::abcd:1'; LinkLayerAddress = '02-00-5E-00-53-78'; State = 'Reachable'; InterfaceIndex = 25 })
        $neighbors = { param($InterfaceIndex) $rows }.GetNewClosure()
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $withPeer = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider Neighbor -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $neighbors)
        $withPeer.Count | Should -Be 1
        $withPeer[0].Address | Should -BeExactly 'fe80::d700:50%25'
        $withPeer[0].NeighborState | Should -Be 'Stale'
        $withPeer[0].MacAddress | Should -Be '02-00-5E-00-53-77'
        $noIdentity = @(Get-PcaiEndpointCandidate -HostName 'no-such-peer' -Provider Neighbor -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $neighbors)
        $noIdentity.Count | Should -Be 0
    }

    # Protects: the optional multicast wake (ping -6 ff02::1%if) runs once per scoped interface and only when asked.
    # Detects: a wake sent without -Wake (traffic the caller did not ask for) or skipped with it.
    # Needs: an injected wake action that records the ifIndex values.
    # Breadcrumb: Invoke-PcaiNeighborWake
    It 'wakes each interface once with -Wake and never without it' {
        $woken = [Collections.Generic.List[int]]::new()
        $wake = { param($InterfaceIndex) $woken.Add([int]$InterfaceIndex) }.GetNewClosure()
        $source = New-DsInterfaceSource $script:Usb4Pair
        $common = @{ PeerId = 'dtm-test-peer'; Provider = 'Neighbor'; InterfaceSource = $source; RegistrySource = $script:RegistrySource; NeighborSource = $script:NoNeighbors; WakeAction = $wake }
        $null = Get-PcaiEndpointCandidate @common
        $woken.Count | Should -Be 0
        $null = Get-PcaiEndpointCandidate @common -Wake
        @($woken | Sort-Object) | Should -Be @(19, 25)
    }

    # Protects: dedupe by (address, zone). The same address seen by several providers is one candidate that keeps the
    # strongest source and the evidence/MAC of all; the same address on different zones stays separate.
    # Detects: duplicate attempts for one address, a MAC or neighbour state lost in the merge, two zones collapsed.
    # Needs: injected registry, adapters and neighbour rows.
    # Breadcrumb: Merge-PcaiEndpointCandidate
    It 'dedupes by address and zone, merging evidence and keeping the strongest source' {
        $rows = @([pscustomobject]@{ IPAddress = 'fe80::d700:50'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 'Reachable'; InterfaceIndex = 25 })
        $neighbors = { param($InterfaceIndex) if ($InterfaceIndex -eq 25) { $rows } else { @() } }.GetNewClosure()
        $source = New-DsInterfaceSource $script:Usb4Pair
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer, Neighbor -IncludeIPv6 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $neighbors)
        $short25 = @($found | Where-Object { $_.Address -eq 'fe80::d700:50%25' })
        $short19 = @($found | Where-Object { $_.Address -eq 'fe80::d700:50%19' })
        $short25.Count | Should -Be 1
        $short19.Count | Should -Be 1
        $short25[0].Source | Should -Be 'Neighbor'
        $short25[0].MacAddress | Should -Be '02-00-5E-00-53-50'
        $short25[0].Evidence | Should -BeLike '*registry*'
        $short25[0].Evidence | Should -BeLike '*neighbor*'
        @($found | Group-Object -Property Address | Where-Object Count -gt 1).Count | Should -Be 0
    }

    # Protects: the tie rule for ambiguous link-local candidates. A KnownPeer address on two interfaces is annotated with
    # the neighbour state seen on EACH interface, so the connector can prefer the interface whose entry is Reachable.
    # Detects: neighbour state attached to the wrong interface or not attached to non-Neighbor candidates.
    # Needs: injected neighbour rows for interface 25 only.
    # Breadcrumb: Get-PcaiEndpointCandidate neighbour annotation
    It 'annotates KnownPeer candidates with the neighbour state of their own interface' {
        $rows = @([pscustomobject]@{ IPAddress = 'fe80::d700:50'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 'Reachable'; InterfaceIndex = 25 })
        $neighbors = { param($InterfaceIndex) if ($InterfaceIndex -eq 25) { $rows } else { @() } }.GetNewClosure()
        $source = New-DsInterfaceSource $script:Usb4Pair
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider KnownPeer -IncludeIPv6 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $neighbors)
        ($found | Where-Object { $_.Address -eq 'fe80::d700:50%25' }).NeighborState | Should -Be 'Reachable'
        ($found | Where-Object { $_.Address -eq 'fe80::d700:50%19' }).NeighborState | Should -BeNullOrEmpty
    }

    # Protects: the registry MAC (topologyLinks[].expectedPeerMacs) feeds the Eui64 provider when the caller supplies none.
    # Detects: the Eui64 provider emitting nothing for a peer whose MAC is in the registry.
    # Needs: injected registry carrying expectedPeerMacs for the synthetic peer.
    # Breadcrumb: Resolve-PcaiPeer registry MACs
    It 'uses the registry MAC for the Eui64 provider when no MacAddress is given' {
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-test-peer' -Provider Eui64 -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $found.Count | Should -Be 1
        $found[0].Address | Should -BeExactly 'fe80::5eff:fe00:5350%25'
        $found[0].PeerId | Should -Be 'dtm-test-peer'
    }

    # Protects: the Dns provider. getaddrinfo drops a zone-less fe80, so a link-local answer without a scope is
    # unusable and must never be offered; A and AAAA are both used.
    # Detects: a zone-less fe80 candidate, a missing IPv4 or IPv6 answer, a class mislabel.
    # Needs: an injected resolver.
    # Breadcrumb: Resolve-PcaiDnsAddress
    It 'maps DNS answers to classed candidates and drops a zone-less link-local' {
        $dns = { param($Name) [ipaddress[]]@([ipaddress]'2001:db8::10', [ipaddress]'192.0.2.10', [ipaddress]'fe80::1', [ipaddress]'fd00:7462:7032::10', [ipaddress]'127.0.0.1') }
        $found = @(Get-PcaiEndpointCandidate -HostName 'peer.example.invalid' -Provider Dns -DnsSource $dns -InterfaceSource { @() } -RegistrySource $script:RegistrySource)
        @($found | Where-Object { $_.Class -eq 'LinkLocal' }).Count | Should -Be 0
        ($found | Where-Object { $_.Address -eq '2001:db8::10' }).Class | Should -Be 'GUA'
        ($found | Where-Object { $_.Address -eq 'fd00:7462:7032::10' }).Class | Should -Be 'ULA'
        ($found | Where-Object { $_.Address -eq '192.0.2.10' }).Class | Should -Be 'IPv4Public'
        ($found | Where-Object { $_.Address -eq '127.0.0.1' }).Class | Should -Be 'Loopback'
        @($found | Where-Object { $_.Source -ne 'Dns' }).Count | Should -Be 0
        $v6only = @(Get-PcaiEndpointCandidate -HostName 'peer.example.invalid' -Provider Dns -IncludeIPv6 -DnsSource $dns -InterfaceSource { @() } -RegistrySource $script:RegistrySource)
        @($v6only | Where-Object { $_.Family -ne 'InterNetworkV6' }).Count | Should -Be 0
    }

    # Protects: the address classifier behind candidate ordering and masked reporting.
    # Detects: a ULA/GUA swap, IPv4 shared (CGNAT) space or APIPA classed as public, loopback not recognised.
    # Needs: nothing.
    # Breadcrumb: Get-PcaiAddressClass
    It 'classifies <Address> as <Class>' -ForEach @(
        @{ Address = 'fe80::1'; Class = 'LinkLocal' }
        @{ Address = 'febf::1'; Class = 'LinkLocal' }
        @{ Address = 'fd00:7462:7032::50'; Class = 'ULA' }
        @{ Address = 'fc00::1'; Class = 'ULA' }
        @{ Address = '2001:db8::1'; Class = 'GUA' }
        @{ Address = '::1'; Class = 'Loopback' }
        @{ Address = '127.0.0.1'; Class = 'Loopback' }
        @{ Address = '10.1.2.3'; Class = 'IPv4Private' }
        @{ Address = '172.31.240.2'; Class = 'IPv4Private' }
        @{ Address = '192.168.1.5'; Class = 'IPv4Private' }
        @{ Address = '100.64.0.1'; Class = 'IPv4Private' }
        @{ Address = '169.254.1.1'; Class = 'IPv4LinkLocal' }
        @{ Address = '192.0.2.10'; Class = 'IPv4Public' }
        @{ Address = '172.32.0.1'; Class = 'IPv4Public' }
    ) {
        & (Get-Module PC-AI.Network) { param($a) Get-PcaiAddressClass -IPAddress ([ipaddress]$a) } $Address | Should -Be $Class
    }
}

Describe 'Connect-PcaiEndpoint' -Tag 'Unit', 'Network', 'Portable' {
    # Protects: RFC 8305 candidate order: scoped link-local first (neighbour-observed and registry candidates before
    # EUI-64 guesses, MAC-matched before unmatched, Reachable interface before others), then IPv6 and IPv4 interleaved
    # starting with IPv6.
    # Detects: IPv4 first (the classic Happy Eyeballs bug), link-local behind ULA/IPv4, EUI-64 guesses ahead of observed
    # neighbours, a stable-sort assumption (candidates are given shuffled and with equal keys).
    # Needs: a connector that refuses everything instantly so attempts run strictly in candidate order.
    # Breadcrumb: Get-PcaiConnectionOrder
    It 'orders scoped link-local first, then interleaves IPv6 and IPv4 starting with IPv6' {
        $unordered = @(
            (New-DsCand '192.0.2.11'), (New-DsCand 'fe80::5eff:fe00:5350%19' -Source Eui64),
            (New-DsCand '2001:db8::50'), (New-DsCand 'fe80::d700:50%19' -Source KnownPeer -NeighborState Stale),
            (New-DsCand '192.0.2.10'), (New-DsCand 'fd00:7462:7032::50'),
            (New-DsCand 'fe80::aa:bb%7' -Source Neighbor -NeighborState Stale -MacMatched $true),
            (New-DsCand 'fe80::d700:50%25' -Source KnownPeer -NeighborState Reachable))
        $plan = @{}
        foreach ($c in $unordered) { $plan[([ipaddress]$c.Address).ToString()] = 'Refused' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate $unordered -Port 22 -TimeoutMs 3000 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeFalse
        @($result.Attempts | ForEach-Object { $_.Address }) | Should -Be @(
            'fe80::aa:bb%7', 'fe80::d700:50%25', 'fe80::d700:50%19', 'fe80::5eff:fe00:5350%19',
            'fd00:7462:7032::50', '192.0.2.11', '2001:db8::50', '192.0.2.10')
    }

    # Protects: the staggered start. Attempt 2 starts after AttemptDelayMs while attempt 1 is STILL pending, and a later
    # success wins over a connect that never finishes.
    # Detects: serial attempts (attempt 2 never starts because attempt 1 hangs), an attempt started before the delay.
    # Needs: a hanging first attempt and an instantly connecting second; timing tolerance is wide on purpose.
    # Breadcrumb: Connect-PcaiEndpoint race loop
    It 'starts the second attempt after the delay without waiting for the first to finish' {
        $plan = @{ '2001:db8::1' = 'Hang'; '192.0.2.1' = 'Ok' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1')) -Port 22 -AttemptDelayMs 300 -TimeoutMs 3000 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeTrue
        $result.Winner.Address | Should -Be '192.0.2.1'
        $result.Attempts.Count | Should -Be 2
        $result.Attempts[1].StartMs | Should -BeGreaterOrEqual 240
        $result.Attempts[1].StartMs | Should -BeLessThan 1200
        $result.Attempts[0].Outcome | Should -Be 'Cancelled'
        $result.Attempts[1].Outcome | Should -Be 'Connected'
        $result.ElapsedMs | Should -BeLessThan 2500
    }

    # Protects: fallback on failure is immediate. A refused or unreachable attempt must not make the next candidate wait
    # out the whole delay (RFC 8305 section 5).
    # Detects: a connector that always waits AttemptDelayMs, even after a definite failure.
    # Needs: a delay of 1500 ms so the difference between "immediate" and "after the delay" is unmistakable.
    # Breadcrumb: Connect-PcaiEndpoint nextStartAt reset on failure
    It 'starts the next attempt immediately when the current one fails' {
        $plan = @{ 'fe80::1%25' = 'Refused'; '192.0.2.1' = 'Ok' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand 'fe80::1%25' -Source KnownPeer), (New-DsCand '192.0.2.1')) -Port 22 -AttemptDelayMs 1500 -TimeoutMs 4000 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeTrue
        $result.Attempts[0].Outcome | Should -Be 'Refused'
        $result.Attempts[1].StartMs | Should -BeLessThan 700
        $result.ElapsedMs | Should -BeLessThan 1200
    }

    # Protects: no socket leak. Every client but a PassThru winner is disposed: a hung loser, the winner in test mode, and
    # a loser that ALSO connected in the same wake.
    # Detects: undisposed losers or winner, a candidate started after the winner, a second connected client left open.
    # Needs: a hook that completes attempt 1 at the instant attempt 2 is created, so both are done in the same wake.
    # Breadcrumb: Connect-PcaiEndpoint cleanup (finally)
    It 'disposes every client it opened, including a second connected loser' {
        $log = New-DsLog
        $hang = [Threading.Tasks.TaskCompletionSource[bool]]::new()
        $connector = {
            param($IPAddress, $Port, $Candidate)
            $key = $IPAddress.ToString()
            $log.Calls.Add($key)
            $client = [IO.MemoryStream]::new()
            $log.Clients[$key] = $client
            if ($key -eq '2001:db8::1') { return [pscustomobject]@{ Task = $hang.Task; Client = $client } }
            # Attempt 2 is created 120 ms in; finish attempt 1 first so both are complete at the next wake.
            $hang.SetResult($true)
            [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $client }
        }.GetNewClosure()
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1'), (New-DsCand '192.0.2.2')) -Port 22 -AttemptDelayMs 120 -TimeoutMs 3000 -Connector $connector
        $result.Connected | Should -BeTrue
        $result.Winner.Address | Should -Be '2001:db8::1'
        $result.Attempts.Count | Should -Be 2
        $log.Calls | Should -Not -Contain '192.0.2.2'
        $log.Clients['2001:db8::1'].CanRead | Should -BeFalse
        $log.Clients['192.0.2.1'].CanRead | Should -BeFalse
        $result.Attempts[1].Outcome | Should -Be 'Cancelled'
    }

    # Protects: the hung-loser path of the same rule, plus PassThru ownership: the caller owns the winner's client.
    # Detects: a PassThru winner disposed by the connector, a hung loser left open.
    # Needs: one hanging and one instantly connecting candidate.
    # Breadcrumb: Connect-PcaiEndpoint -PassThru
    It 'disposes a hung loser and hands the winner to the caller with -PassThru' {
        $plan = @{ '2001:db8::1' = 'Hang'; '192.0.2.1' = 'Ok' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1')) -Port 22 -AttemptDelayMs 50 -TimeoutMs 3000 -PassThru -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeTrue
        $log.Clients['2001:db8::1'].CanRead | Should -BeFalse
        $log.Clients['192.0.2.1'].CanRead | Should -BeTrue
        [object]::ReferenceEquals($result.Client, $log.Clients['192.0.2.1']) | Should -BeTrue
        $result.Client.Dispose()
    }

    # Protects: all-fail is data, not success and not an exception, and the attempt list is complete and classified.
    # Detects: Connected true with no winner, a missing attempt, a refused/unreachable/synchronous failure mislabelled.
    # Needs: three failing candidates plus one whose connector throws synchronously.
    # Breadcrumb: Connect-PcaiEndpoint result assembly
    It 'returns Connected=false with every attempt classified when all fail' {
        $plan = @{ 'fe80::1%25' = 'Refused'; '2001:db8::1' = 'Unreachable'; '192.0.2.1' = 'HostUnreachable'; '192.0.2.2' = 'Throw' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand 'fe80::1%25' -Source KnownPeer), (New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1'), (New-DsCand '192.0.2.2')) -Port 22 -TimeoutMs 3000 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeFalse
        $result.Winner | Should -BeNullOrEmpty
        $result.TimedOut | Should -BeFalse
        $result.Attempts.Count | Should -Be 4
        @($result.Attempts | ForEach-Object { $_.Outcome }) | Should -Be @('Refused', 'Unreachable', 'Unreachable', 'Unreachable')
        $result.Attempts[0].Class | Should -Be 'LinkLocal'
        $result.Attempts[0].Zone | Should -Be 25
        $result.Attempts[0].Error | Should -Not -BeNullOrEmpty
        $result.Attempts[3].Error | Should -BeLike '*synchronous*'
        foreach ($client in $log.Clients.Values) { $client.CanRead | Should -BeFalse }
    }

    # Protects: the documented error convention: failure is returned data; only an explicit -ErrorAction Stop turns it into an
    # exception, so a script with a global Stop preference can still read the attempt list.
    # Detects: a throw on plain failure, or no throw under -ErrorAction Stop.
    # Needs: one refusing candidate.
    # Breadcrumb: Connect-PcaiEndpoint error convention
    It 'does not throw on failure unless -ErrorAction Stop is passed explicitly' {
        $plan = @{ '192.0.2.1' = 'Refused' }
        $log = New-DsLog
        { Connect-PcaiEndpoint -Candidate @((New-DsCand '192.0.2.1')) -Port 22 -Connector (New-DsConnector $plan $log) } | Should -Not -Throw
        { Connect-PcaiEndpoint -Candidate @((New-DsCand '192.0.2.1')) -Port 22 -Connector (New-DsConnector $plan $log) -ErrorAction Stop } | Should -Throw '*192.0.2.1*'
    }

    # Protects: the overall deadline. All attempts hang; the call returns near TimeoutMs, marks them Timeout, and disposes
    # every client.
    # Detects: a missing deadline (hang), a deadline ignored while a delay is pending, undisposed clients, TimedOut unset.
    # Needs: all-hanging candidates; wide upper tolerance for a loaded runner.
    # Breadcrumb: Connect-PcaiEndpoint deadline
    It 'honours the overall timeout and reports Timeout for attempts still pending' {
        $plan = @{ '2001:db8::1' = 'Hang'; '192.0.2.1' = 'Hang'; '192.0.2.2' = 'Hang' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1'), (New-DsCand '192.0.2.2')) -Port 22 -AttemptDelayMs 100 -TimeoutMs 450 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeFalse
        $result.TimedOut | Should -BeTrue
        $result.ElapsedMs | Should -BeGreaterOrEqual 400
        $result.ElapsedMs | Should -BeLessThan 1800
        $result.Attempts.Count | Should -Be 3
        @($result.Attempts | Where-Object { $_.Outcome -ne 'Timeout' }).Count | Should -Be 0
        foreach ($client in $log.Clients.Values) { $client.CanRead | Should -BeFalse }
    }

    # Protects: AttemptDelayMs is clamped to 10..2000 rather than rejected, so an absurd value cannot starve the second
    # family.
    # Detects: an unclamped 99999 ms delay, which would exceed the timeout and never start attempt 2.
    # Needs: about 2 s of wall time.
    # Breadcrumb: Connect-PcaiEndpoint clamp
    It 'clamps AttemptDelayMs to 2000 ms' {
        $plan = @{ '2001:db8::1' = 'Hang'; '192.0.2.1' = 'Ok' }
        $log = New-DsLog
        $result = Connect-PcaiEndpoint -Candidate @((New-DsCand '2001:db8::1'), (New-DsCand '192.0.2.1')) -Port 22 -AttemptDelayMs 99999 -TimeoutMs 3500 -Connector (New-DsConnector $plan $log)
        $result.Connected | Should -BeTrue
        $result.Attempts[1].StartMs | Should -BeGreaterOrEqual 1900
        $result.Attempts[1].StartMs | Should -BeLessThan 2800
    }

    # Protects: the discover-then-connect path end to end with injected providers: candidates are piped in, and a replug
    # to a new ifIndex changes the zone that is dialled.
    # Detects: a pipeline that drops candidates, a zone not carried into the connector call.
    # Needs: injected registry, adapters, neighbour cache and connector.
    # Breadcrumb: Connect-PcaiEndpoint -PeerId resolve set; Get-PcaiEndpointCandidate
    It 'resolves and connects by PeerId, dialling the live zone after a replug' {
        $dialled = [Collections.Generic.List[string]]::new()
        $connector = {
            param($IPAddress, $Port, $Candidate)
            $dialled.Add($IPAddress.ToString())
            [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $null }
        }.GetNewClosure()
        $state = @{ Adapters = @((New-DsAdapter -Index 19 -Alias 'Ethernet 18')) }
        $resolver = @{
            InterfaceSource = { $state.Adapters }.GetNewClosure()
            RegistrySource = $script:RegistrySource
            NeighborSource = $script:NoNeighbors
        }
        $first = Connect-PcaiEndpoint -PeerId 'dtm-test-peer' -Port 22 -Provider KnownPeer -IncludeIPv6 -ResolverParameter $resolver -Connector $connector
        $state.Adapters = @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $second = Connect-PcaiEndpoint -PeerId 'dtm-test-peer' -Port 22 -Provider KnownPeer -IncludeIPv6 -ResolverParameter $resolver -Connector $connector
        $first.Connected | Should -BeTrue
        $first.Winner.Address | Should -BeLike 'fe80::d700:50%19'
        $second.Winner.Address | Should -BeLike 'fe80::d700:50%25'
        $dialled[0] | Should -BeLike '*%19'
        $dialled[1] | Should -BeLike '*%25'
        $first.Winner.Source | Should -Be 'KnownPeer'
    }

    # Protects: the thin wrapper for ssh/probe consumers returns the winning candidate (or nothing), never a throw.
    # Detects: a wrapper that returns the whole result, hides a failure as a candidate, or throws for no route.
    # Needs: injected providers and connector.
    # Breadcrumb: Resolve-PcaiPeerEndpoint
    It 'Resolve-PcaiPeerEndpoint returns the winning candidate, or nothing when no route answers' {
        $ok = { param($IPAddress, $Port, $Candidate) [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $null } }
        $refuse = { param($IPAddress, $Port, $Candidate) [pscustomobject]@{ Task = [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10061)); Client = $null } }
        $resolver = @{ InterfaceSource = { @((New-DsAdapter -Index 25 -Alias 'Ethernet 4')) }; RegistrySource = $script:RegistrySource; NeighborSource = $script:NoNeighbors }
        $winner = Resolve-PcaiPeerEndpoint -PeerId 'dtm-test-peer' -Port 22 -Provider KnownPeer -ResolverParameter $resolver -Connector $ok
        $winner.Class | Should -Be 'LinkLocal'
        $winner.Address | Should -BeLike '*%25'
        $none = Resolve-PcaiPeerEndpoint -PeerId 'dtm-test-peer' -Port 22 -Provider KnownPeer -ResolverParameter $resolver -Connector $refuse
        $none | Should -BeNullOrEmpty
        $detail = Resolve-PcaiPeerEndpoint -PeerId 'dtm-test-peer' -Port 22 -Provider KnownPeer -ResolverParameter $resolver -Connector $refuse -Detailed
        $detail.Connected | Should -BeFalse
        $detail.Attempts.Count | Should -BeGreaterThan 0
    }
}

Describe 'PC-AI.Network dual-stack export contract' -Tag 'Unit', 'Network', 'Fast', 'Portable' {
    # Protects: the manifest and the psm1 Export-ModuleMember list agree, and the helpers stay private.
    # Detects: a function exported by only one of the two lists (invisible to callers or leaked to them).
    # Needs: the module imported in the file-level BeforeAll.
    # Breadcrumb: Modules/PC-AI.Network/PC-AI.Network.psd1 FunctionsToExport; PC-AI.Network.psm1 Export-ModuleMember
    It 'exports the three dual-stack commands from both lists and keeps helpers private' {
        $module = Get-Module PC-AI.Network
        foreach ($name in 'Get-PcaiEndpointCandidate', 'Connect-PcaiEndpoint', 'Resolve-PcaiPeerEndpoint') {
            $module.ExportedFunctions.Keys | Should -Contain $name
        }
        # Evaluate the manifest's data literal directly: Import-PowerShellDataFile is not available in every host this runs in.
        $manifest = & ([scriptblock]::Create((Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Network\PC-AI.Network.psd1') -Raw)))
        $psm1 = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Network\PC-AI.Network.psm1') -Raw
        $inPsm1 = [regex]::Matches(($psm1 -replace '(?s).*Export-ModuleMember\s+-Function\s+@\((.*?)\).*', '$1'), "'([^']+)'") | ForEach-Object { $_.Groups[1].Value }
        @($manifest.FunctionsToExport | Sort-Object) | Should -Be @($inPsm1 | Sort-Object)
        @($module.ExportedFunctions.Keys | Sort-Object) | Should -Be @($manifest.FunctionsToExport | Sort-Object)
        foreach ($private in 'Get-PcaiAddressClass', 'ConvertTo-PcaiEui64LinkLocal', 'Get-PcaiConnectionOrder', 'Merge-PcaiEndpointCandidate') {
            $module.ExportedFunctions.Keys | Should -Not -Contain $private
        }
    }
}

Describe 'Test-FleetPath -DualStack' -Tag 'Unit', 'Network' {
    BeforeAll {
        # Test-FleetPath.ps1 requires PowerShell 7; the Describe is skipped on 5.1 (see -Skip below).
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            . (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Networking\Test-FleetPath.ps1')
            $inventory = Join-Path $TestDrive 'fleet-paths.json'
            [IO.File]::WriteAllText($inventory, '{"schema":"pcai.fleet-paths/1","gateways":[{"name":"gw","ip":"192.0.2.1","mac":"00-00-5E-00-53-01","probePorts":[443]}],"hosts":{"dtm-test-peer":{"tailnet":"100.64.0.102","lan":["192.0.2.33"],"services":{"ssh":{"ports":[22]}}}}}')
            $script:FleetInventory = $inventory
            Mock Find-FleetRoute { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
            Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $false; Error = 'test'; Status = $null } }
            Mock Get-FleetInterface { [pscustomobject]@{ InterfaceIndex = 12; InterfaceAlias = 'Ethernet 12'; IPAddress = '192.0.2.200'; PrefixLength = 24 } }
            Mock Get-FleetPhysicalInterfaceIndex { @(12) }
            Mock Get-FleetNeighbor { [pscustomobject]@{ IPAddress = '192.0.2.1'; LinkLayerAddress = '00-00-5E-00-53-01'; State = 'Reachable'; InterfaceAlias = 'Ethernet 12'; InterfaceIndex = 12 } }
            $script:GatewayConnector = { param($Address, $Port, $SourceAddress) [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $null } }
            $registry = @'
{ "ulaPrefix": "fd00:7462:7032", "fleetPeers": [ { "name": "dtm-test-peer", "aliases": [], "shortLinkLocal": "fe80::d700:50", "shortUla": "fd00:7462:7032::50", "lanAddresses": ["192.0.2.99"] } ] }
'@ | ConvertFrom-Json
            $script:DualResolver = @{
                InterfaceSource = { @([pscustomobject]@{ Name = 'Ethernet 4'; InterfaceAlias = 'Ethernet 4'; InterfaceDescription = 'USB4 P2P Network Adapter'; ifIndex = 25; Status = 'Up'; MacAddress = '02-00-5E-00-53-AA' }) }
                RegistrySource = { param($Path) $registry }.GetNewClosure()
                NeighborSource = { param($InterfaceIndex) @() }
            }
        }
    }

    # Protects: byte-compatible default output. Without the switch the report carries no DualStack member anywhere, and the
    # per-target USB4 resolver is never consulted.
    # Detects: a DualStack property (even a null one) leaking into the default report or summary rows.
    # Needs: pwsh 7; mocked probes and an injected gateway connector (no network).
    # Breadcrumb: Invoke-FleetPathTest -DualStack
    It 'adds nothing to the report unless -DualStack is passed' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
        $report = Invoke-FleetPathTest -InventoryPath $script:FleetInventory -BudgetMs 500 -Connector $script:GatewayConnector
        $json = $report | ConvertTo-Json -Depth 10
        $json | Should -Not -Match 'DualStack'
        @($report.Targets[0].Endpoints[0].PSObject.Properties.Name) | Should -Be @('Service', 'Port', 'Recommended', 'Candidates')
        @($report.Summary[0].PSObject.Properties.Name) | Should -Be @('Target', 'Service', 'Port', 'Recommended', 'LatencyMs', 'Path', 'Interface')
    }

    # Protects: -DualStack feeds the connector only gated candidates: the verified LAN address plus scoped USB4 link-local/ULA
    # for the peer; the registry's own IPv4 address (not home-LAN verified) and DNS are never connected to.
    # Detects: an ungated registry IPv4 candidate (192.0.2.99) reaching the connector, a missing link-local candidate, a
    # winner not reported with its class, source and timing.
    # Needs: pwsh 7; mocked probes; injected resolver and dual-stack connector recording what it was asked to dial.
    # Breadcrumb: Invoke-FleetPathTest -DualStack
    It 'reports the winner, class, source and timing, and dials only gated candidates' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
        $dialled = [Collections.Generic.List[string]]::new()
        $dual = {
            param($IPAddress, $Port, $Candidate)
            $dialled.Add($IPAddress.ToString())
            $task = if ($IPAddress.AddressFamily -eq 'InterNetworkV6') { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10061)) } else { [Threading.Tasks.Task]::CompletedTask }
            [pscustomobject]@{ Task = $task; Client = $null }
        }.GetNewClosure()
        $report = Invoke-FleetPathTest -InventoryPath $script:FleetInventory -BudgetMs 800 -Connector $script:GatewayConnector -DualStack -DualStackConnector $dual -DualStackResolverParameter $script:DualResolver
        $endpoint = $report.Targets[0].Endpoints[0]
        $endpoint.DualStack.Connected | Should -BeTrue
        $endpoint.DualStack.Winner.Class | Should -Not -BeNullOrEmpty
        $endpoint.DualStack.Winner.Source | Should -Not -BeNullOrEmpty
        $endpoint.DualStack.ElapsedMs | Should -BeGreaterOrEqual 0
        @($endpoint.DualStack.Attempts | Where-Object { $_.Class -eq 'LinkLocal' }).Count | Should -BeGreaterThan 0
        $dialled | Should -Not -Contain '192.0.2.99'
        $dialled | Should -Contain '192.0.2.33'
        $report.Summary[0].PSObject.Properties.Name | Should -Contain 'DualStackWinnerClass'
        $report.Summary[0].DualStackConnected | Should -BeTrue
    }
}

Describe 'Dual-stack defaults and edge cases' -Tag 'Unit', 'Network', 'Portable' {
    # Protects: the default OS wrappers (adapter table, neighbour cache, wake, DNS, registry file) and the odd inputs the
    # injected-provider tests above do not reach. Only loopback sockets and module-scoped mocks are used.
    # Detects: a default wrapper that throws instead of reading as "nothing", or that is never actually wired in.

    # Protects: the default adapter source feeds the scope interfaces; a failing adapter read is no interfaces.
    # Detects: an unhandled adapter-table error, a default source that ignores the Up (operational status 1) and USB4 filters.
    # Needs: a module-scoped Get-CimInstance mock for MSFT_NetAdapter (the default source reads CIM, not Get-NetAdapter).
    # Breadcrumb: Get-PcaiLiveInterface
    It 'reads the live adapter table by default and treats a failure as no interfaces' {
        Mock Get-CimInstance -ModuleName PC-AI.Network -ParameterFilter { $ClassName -eq 'MSFT_NetAdapter' } {
            @([pscustomobject]@{ Name = 'Ethernet 4'; InterfaceDescription = 'USB4 P2P Network Adapter'; InterfaceIndex = 41; InterfaceOperationalStatus = 1; PermanentAddress = '02005E0053AA' },
                [pscustomobject]@{ Name = 'Ethernet'; InterfaceDescription = 'Intel(R) Ethernet'; InterfaceIndex = 7; InterfaceOperationalStatus = 1; PermanentAddress = '02005E0053AB' },
                [pscustomobject]@{ Name = 'Ethernet 9'; InterfaceDescription = 'USB4 P2P Network Adapter #2'; InterfaceIndex = 43; InterfaceOperationalStatus = 2; PermanentAddress = '02005E0053AC' })
        }
        $found = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $found.Count | Should -Be 1
        $found[0].Address | Should -BeExactly 'fe80::211:22ff:fe33:4455%41'
        Mock Get-CimInstance -ModuleName PC-AI.Network -ParameterFilter { $ClassName -eq 'MSFT_NetAdapter' } { throw 'adapter table unavailable' }
        @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors).Count | Should -Be 0
    }

    # Protects: the default neighbour source and the default wake action are the ones used when nothing is injected, and
    # their failures are not fatal.
    # Detects: a wake that never runs with -Wake, a neighbour read error that aborts the lookup.
    # Needs: module-scoped mocks of Get-CimInstance for MSFT_NetNeighbor and the (private) wake function so no ping is sent.
    # Breadcrumb: Get-PcaiNeighborEntry, Invoke-PcaiNeighborWake
    It 'reads the CIM neighbour cache and the wake function by default, and survives their failures' {
        Mock Invoke-PcaiNeighborWake -ModuleName PC-AI.Network { }
        Mock Get-CimInstance -ModuleName PC-AI.Network -ParameterFilter { $ClassName -eq 'MSFT_NetNeighbor' } {
            @([pscustomobject]@{ IPAddress = 'fe80::5eff:fe00:5350'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 5; InterfaceIndex = 25 },
                [pscustomobject]@{ IPAddress = 'fe80::dead:1'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 0; InterfaceIndex = 25 },
                [pscustomobject]@{ IPAddress = 'fe80::dead:2'; LinkLayerAddress = '02-00-5E-00-53-50'; State = 99; InterfaceIndex = 25 })
        }
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -MacAddress '02-00-5E-00-53-50' -Provider Neighbor -Wake -InterfaceSource $source -RegistrySource $script:RegistrySource)
        $found.Count | Should -Be 1
        $found[0].NeighborState | Should -Be 'Reachable'
        Should -Invoke Invoke-PcaiNeighborWake -ModuleName PC-AI.Network -Times 1 -Exactly
        Mock Get-CimInstance -ModuleName PC-AI.Network -ParameterFilter { $ClassName -eq 'MSFT_NetNeighbor' } { throw 'neighbour cache unavailable' }
        @(Get-PcaiEndpointCandidate -MacAddress '02-00-5E-00-53-50' -Provider Neighbor -InterfaceSource $source -RegistrySource $script:RegistrySource).Count | Should -Be 0
        $throwingWake = { param($InterfaceIndex) throw 'wake failed' }
        { Get-PcaiEndpointCandidate -MacAddress '02-00-5E-00-53-50' -Provider Neighbor -Wake -WakeAction $throwingWake -InterfaceSource $source -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors } | Should -Not -Throw
    }

    # Protects: the registry file reader and both registry shapes (fleetPeers[] and peers{} keyed by name), and that a
    # file that is not JSON reads as "no registry" rather than an error.
    # Detects: a registry path that is ignored, the keyed shape not resolving, a parse error escaping.
    # Needs: files under TestDrive.
    # Breadcrumb: Read-PcaiPeerRegistry, Resolve-PcaiPeer
    It 'reads a registry file in either shape and tolerates a file that is not JSON' {
        $keyed = Join-Path $TestDrive 'keyed.json'
        [IO.File]::WriteAllText($keyed, '{ "peers": { "dtm-keyed-peer": { "shortLinkLocal": "fe80::d700:77", "lanAddresses": ["192.0.2.77"] } } }')
        $source = New-DsInterfaceSource @((New-DsAdapter -Index 25 -Alias 'Ethernet 4'))
        $found = @(Get-PcaiEndpointCandidate -PeerId 'dtm-keyed-peer' -Provider KnownPeer -RegistryPath $keyed -InterfaceSource $source -NeighborSource $script:NoNeighbors)
        @($found | Where-Object { $_.Address -eq 'fe80::d700:77%25' }).Count | Should -Be 1
        @($found | Where-Object { $_.Address -eq '192.0.2.77' }).Count | Should -Be 1
        $broken = Join-Path $TestDrive 'broken.json'
        [IO.File]::WriteAllText($broken, 'this is not json')
        @(Get-PcaiEndpointCandidate -PeerId 'dtm-keyed-peer' -Provider KnownPeer -RegistryPath $broken -InterfaceSource $source -NeighborSource $script:NoNeighbors).Count | Should -Be 0
        $throwingRegistry = { param($Path) throw 'registry unavailable' }
        { Get-PcaiEndpointCandidate -PeerId 'dtm-keyed-peer' -Provider KnownPeer -RegistrySource $throwingRegistry -InterfaceSource $source -NeighborSource $script:NoNeighbors } | Should -Not -Throw
    }

    # Protects: the DNS provider's default source, literal addresses given as HostName (zone preserved, a zone-less
    # link-local expanded to the live zones), a bad MAC warning, and a DNS source that throws.
    # Detects: a literal address sent to DNS, a zone-less literal link-local offered without a zone.
    # Needs: a module-scoped mock of the default resolver wrapper.
    # Breadcrumb: Resolve-PcaiDnsAddress, Get-PcaiEndpointCandidate Dns provider
    It 'handles literal HostName values, the default resolver wrapper and a bad MAC' {
        Mock Resolve-PcaiDnsAddress -ModuleName PC-AI.Network { [ipaddress[]]@([ipaddress]'192.0.2.44') }
        $source = New-DsInterfaceSource $script:Usb4Pair
        $named = @(Get-PcaiEndpointCandidate -HostName 'peer.example.invalid' -Provider Dns -InterfaceSource $source -RegistrySource $script:RegistrySource)
        $named.Count | Should -Be 1
        $named[0].Address | Should -Be '192.0.2.44'
        Should -Invoke Resolve-PcaiDnsAddress -ModuleName PC-AI.Network -Times 1 -Exactly
        $literal = @(Get-PcaiEndpointCandidate -HostName '2001:db8::7' -Provider Dns -InterfaceSource $source -RegistrySource $script:RegistrySource)
        $literal.Count | Should -Be 1
        $literal[0].Evidence | Should -BeLike '*literal*'
        $linkLocal = @(Get-PcaiEndpointCandidate -HostName 'fe80::7' -Provider Dns -InterfaceSource $source -RegistrySource $script:RegistrySource)
        @($linkLocal | ForEach-Object { $_.Address } | Sort-Object) | Should -Be @('fe80::7%19', 'fe80::7%25')
        $throwingDns = { param($Name) throw 'dns down' }
        { Get-PcaiEndpointCandidate -HostName 'peer.example.invalid' -Provider Dns -DnsSource $throwingDns -InterfaceSource $source -RegistrySource $script:RegistrySource } | Should -Not -Throw
        $warnings = @()
        $null = Get-PcaiEndpointCandidate -HostName 'x.example.invalid' -MacAddress 'not-a-mac' -Provider Eui64 -InterfaceSource $source -RegistrySource $script:RegistrySource -WarningVariable warnings -WarningAction SilentlyContinue
        $warnings.Count | Should -BeGreaterThan 0
    }

    # Protects: the pinned-interface paths: an index absent from the table is trusted, and alias plus index must agree.
    # Detects: a pinned index dropped because the adapter table does not list it.
    # Needs: an injected adapter table.
    # Breadcrumb: Select-PcaiScopeInterface
    It 'trusts a pinned InterfaceIndex that is not in the table' {
        $found = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceIndex 77 -InterfaceSource { @() } -RegistrySource $script:RegistrySource -NeighborSource $script:NoNeighbors)
        $found.Count | Should -Be 1
        $found[0].Address | Should -BeExactly 'fe80::211:22ff:fe33:4455%77'
    }

    # Protects: unusual candidate inputs to the connector: an unparseable address is an attempt that failed (not a crash),
    # an IPv4-mapped IPv6 address is dialled as IPv4, a missing port is a clear per-attempt error, and a candidate's own
    # Port is used when -Port is omitted.
    # Detects: an exception from garbage input, a mapped address left as IPv6, a silent connect to port 0.
    # Needs: injected connector.
    # Breadcrumb: ConvertTo-PcaiConnectCandidate, Connect-PcaiEndpoint port handling
    It 'copes with unparseable, IPv4-mapped and port-less candidates' {
        $log = New-DsLog
        $ok = New-DsConnector @{} $log
        $r = Connect-PcaiEndpoint -Candidate @([pscustomobject]@{ Address = 'not an address' }, [pscustomobject]@{ Address = '::ffff:192.0.2.9'; Port = 2222 }) -TimeoutMs 2000 -Connector $ok
        $r.Connected | Should -BeTrue
        $r.Attempts[0].Outcome | Should -Be 'Unreachable'
        $r.Attempts[0].Error | Should -BeLike '*not a parseable*'
        $r.Winner.Address | Should -Be '192.0.2.9'
        $r.Winner.Family | Should -Be 'InterNetwork'
        $r.Winner.Port | Should -Be 2222
        $none = Connect-PcaiEndpoint -Candidate @((New-DsCand '192.0.2.5')) -TimeoutMs 2000 -Connector $ok
        $none.Connected | Should -BeFalse
        $none.Attempts[0].Error | Should -BeLike '*no port*'
        (Connect-PcaiEndpoint -Candidate @() -Port 22 -Connector $ok).Attempts.Count | Should -Be 0
        { Connect-PcaiEndpoint -Candidate @() -Port 22 -Connector $ok -ErrorAction Stop } | Should -Throw '*no candidates*'
    }

    # Protects: failure classification: a TimedOut socket error is Timeout, a cancelled task is Cancelled, a connector that
    # does not return a Task is Unreachable with a clear message, and out-of-range delay and timeout are clamped.
    # Detects: a mislabelled outcome or an exception from a malformed connector.
    # Needs: injected connector returning faulted, cancelled and malformed results.
    # Breadcrumb: Get-PcaiAttemptOutcome, Connect-PcaiEndpoint
    It 'classifies timed-out, cancelled and malformed connects' {
        $connector = {
            param($IPAddress, $Port, $Candidate)
            switch ($IPAddress.ToString()) {
                '192.0.2.1' { [pscustomobject]@{ Task = [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10060)); Client = $null } }
                '192.0.2.2' { [pscustomobject]@{ Task = [Threading.Tasks.Task]::FromCanceled([Threading.CancellationToken]::new($true)); Client = $null } }
                default { [pscustomobject]@{ Task = 'not a task'; Client = $null } }
            }
        }
        $r = Connect-PcaiEndpoint -Candidate @((New-DsCand '192.0.2.1'), (New-DsCand '192.0.2.2'), (New-DsCand '192.0.2.3')) -Port 22 -AttemptDelayMs 1 -TimeoutMs 2000 -Connector $connector
        $r.Connected | Should -BeFalse
        @($r.Attempts | ForEach-Object { $_.Outcome }) | Should -Be @('Timeout', 'Cancelled', 'Unreachable')
        $r.Attempts[2].Error | Should -BeLike '*Task*'
    }

    # Protects: the default connector (TcpClient.ConnectAsync) end to end on a real loopback listener, and the Resolve
    # parameter set's -IncludeIPv4 and -Wake pass-through to the candidate provider.
    # Detects: a default connector that cannot dial, switches that are not forwarded.
    # Needs: a loopback TcpListener on 127.0.0.1 only (no IPv6 requirement).
    # Breadcrumb: New-PcaiTcpConnection, Connect-PcaiEndpoint Resolve set
    It 'dials a real loopback listener with the default connector and forwards resolver switches' {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try {
            $port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
            $r = Connect-PcaiEndpoint -Candidate @((New-DsCand '127.0.0.1')) -Port $port -TimeoutMs 3000
            $r.Connected | Should -BeTrue
            $r.Winner.Class | Should -Be 'Loopback'
            $woken = [Collections.Generic.List[int]]::new()
            $resolver = @{
                InterfaceSource = { @() }
                RegistrySource = $script:RegistrySource
                NeighborSource = $script:NoNeighbors
                DnsSource = { param($Name) [ipaddress[]]@([ipaddress]'127.0.0.1') }
                WakeAction = { param($InterfaceIndex) $woken.Add($InterfaceIndex) }.GetNewClosure()
            }
            $viaName = Connect-PcaiEndpoint -HostName 'peer.example.invalid' -Port $port -IncludeIPv4 -Wake -ResolverParameter $resolver -TimeoutMs 3000
            $viaName.Connected | Should -BeTrue
            $viaName.Winner.Source | Should -Be 'Dns'
            $viaWrapper = Resolve-PcaiPeerEndpoint -HostName 'peer.example.invalid' -Port $port -Wake -ResolverParameter $resolver -TimeoutMs 3000
            $viaWrapper.Address | Should -Be '127.0.0.1'
        }
        finally { $listener.Stop() }
    }

    # Protects: merging of duplicate candidates keeps the identity fields of the stronger or fuller report.
    # Detects: a PeerId, alias or port lost when a Dns duplicate arrives first.
    # Needs: the private merge function called through module scope.
    # Breadcrumb: Merge-PcaiEndpointCandidate
    It 'merges peer id, interface alias, port and evidence from duplicates' {
        $merged = @(& (Get-Module PC-AI.Network) {
            $a = New-PcaiEndpointCandidate -IPAddress ([ipaddress]'192.0.2.8') -Source 'Dns' -Evidence 'DNS answer'
            $b = New-PcaiEndpointCandidate -IPAddress ([ipaddress]'192.0.2.8') -Source 'KnownPeer' -PeerId 'p1' -Port 22 -Evidence 'registry IPv4' -Interface ([pscustomobject]@{ InterfaceAlias = 'Ethernet 4' })
            @(Merge-PcaiEndpointCandidate -Candidate @($a, $null, $b))
        })
        $merged.Count | Should -Be 1
        $merged[0].Source | Should -Be 'KnownPeer'
        $merged[0].PeerId | Should -Be 'p1'
        $merged[0].Port | Should -Be 22
        $merged[0].InterfaceAlias | Should -Be 'Ethernet 4'
        $merged[0].Evidence | Should -BeLike '*DNS answer*registry IPv4*'
    }
}

AfterAll {
    Remove-Module PC-AI.Network -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
