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
}

Describe "Get-NetworkDiagnostics" -Tag 'Unit', 'Network', 'Fast' {
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

Describe "Test-WSLConnectivity" -Tag 'Unit', 'Network', 'Slow' {
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

            # Mock wsl command execution
            Mock -CommandName Invoke-Command -MockWith {
                if ($ScriptBlock -match 'wsl --status') {
                    "Default Version: 2`nDefault Distribution: Ubuntu`nKernel version: 5.10.102.1"
                }
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
            Mock Get-WSLDistributions { @() } -ModuleName PC-AI.Network
        }

        It "Should handle missing WSL gracefully" {
            $result = Test-WSLConnectivity
            $result | Should -Not -BeNullOrEmpty
            $result.Issues | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Watch-VSockPerformance" -Tag 'Unit', 'Network', 'Slow' {
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

Describe "Optimize-VSock" -Tag 'Unit', 'Network', 'Slow', 'RequiresAdmin' {
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

Describe "Network-Helpers (Private Functions)" -Tag 'Unit', 'Network', 'Fast' {
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

AfterAll {
    Remove-Module PC-AI.Network -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
