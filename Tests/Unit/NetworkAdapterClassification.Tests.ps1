param([string]$ProducerPath)

BeforeDiscovery {
    $classificationCases = @(
        @{ Mode = 'default'; IncludeArgument = $false; IncludeValue = $true; ExpectedVirtual = 'vEthernet (owned),Owned VPN'; ExpectedTotal = 4 },
        @{ Mode = 'explicit true'; IncludeArgument = $true; IncludeValue = $true; ExpectedVirtual = 'vEthernet (owned),Owned VPN'; ExpectedTotal = 4 },
        @{ Mode = 'explicit false'; IncludeArgument = $true; IncludeValue = $false; ExpectedVirtual = ''; ExpectedTotal = 2 }
    )
}

Describe 'Network diagnostics adapter classification: <Mode>' -ForEach $classificationCases -Tag 'Unit', 'Network', 'Fast', 'Portable' {
    BeforeAll {
        if (-not $ProducerPath) {
            $ProducerPath = Join-Path $PSScriptRoot '../../Modules/PC-AI.Network/Public/Get-NetworkDiagnostics.ps1'
        }
        # Load the exact public producer, without importing aggregate modules or OS collectors.
        $script:classificationModule = New-Module -Name NetworkClassificationFixture -ArgumentList $ProducerPath -ScriptBlock {
            param($source)
            $script:Denied = [Collections.Generic.List[string]]::new()
            $script:Calls = [Collections.Generic.List[string]]::new()
            $script:Adapters = @(
                [pscustomobject]@{Name='Owned Ethernet'; Virtual=$false; ifIndex=101; Status='Up'; LinkSpeed='1 Gbps'; MacAddress='02-00-00-00-00-01'; MediaType='802.3'; InterfaceDescription='Owned physical one'},
                [pscustomobject]@{Name='Owned Wi-Fi'; Virtual=$false; ifIndex=102; Status='Disconnected'; LinkSpeed='0 bps'; MacAddress='02-00-00-00-00-02'; MediaType='802.11'; InterfaceDescription='Owned physical two'},
                [pscustomobject]@{Name='vEthernet (owned)'; Virtual=$true; ifIndex=201; Status='Up'; LinkSpeed='10 Gbps'; MacAddress='02-00-00-00-00-03'; MediaType='802.3'; InterfaceDescription='Owned virtual one'},
                [pscustomobject]@{Name='Owned VPN'; Virtual=$true; ifIndex=202; Status='Up'; LinkSpeed='1 Gbps'; MacAddress='02-00-00-00-00-04'; MediaType='802.3'; InterfaceDescription='Owned virtual two'}
            )
            function Get-NetAdapter { [CmdletBinding()] param() $script:Calls.Add('Get-NetAdapter'); $script:Adapters }
            function Get-NetIPConfiguration {
                [CmdletBinding()] param($InterfaceIndex)
                $script:Calls.Add("IPConfig:$InterfaceIndex")
                [pscustomobject]@{IPv4DefaultGateway=@{NextHop='192.0.2.1'}; DNSServer=@{ServerAddresses=@('192.0.2.53')}; NetIPv4Interface=@{Dhcp='Enabled'}}
            }
            function Get-NetIPAddress {
                [CmdletBinding()] param($InterfaceIndex)
                $script:Calls.Add("IPAddress:$InterfaceIndex")
                [pscustomobject]@{AddressFamily='IPv4'; IPAddress="192.0.2.$($InterfaceIndex % 100 + 10)"}
            }
            function Get-VMSwitch { [CmdletBinding()] param() $script:Calls.Add('Get-VMSwitch'); @() }
            function Get-DnsClientServerAddress { [CmdletBinding()] param() $script:Calls.Add('Get-DnsClientServerAddress'); @() }
            function Get-DnsClientCache { [CmdletBinding()] param() $script:Calls.Add('Get-DnsClientCache'); @() }
            function Get-DnsClientGlobalSetting { [CmdletBinding()] param() $script:Calls.Add('Get-DnsClientGlobalSetting'); [pscustomobject]@{SuffixSearchList=@('owned.invalid')} }
            function Get-NetRoute { [CmdletBinding()] param($AddressFamily) $script:Calls.Add('Get-NetRoute'); [pscustomobject]@{DestinationPrefix='0.0.0.0/0';NextHop='192.0.2.1';RouteMetric=1;InterfaceAlias='Owned Ethernet'} }
            # Unexpected external boundaries fail closed and remain detectable even if the producer catches the throw.
            foreach ($name in @('Get-NetAdapterStatistics','Measure-NetworkLatency','Resolve-DnsName','Get-CimInstance','Get-WmiObject','Get-Service','Test-NetConnection','Test-Connection','Invoke-WebRequest','Invoke-RestMethod','netsh','wsl')) {
                $body = "[CmdletBinding()] param([Parameter(ValueFromRemainingArguments)][object[]]`$Tokens) `$script:Denied.Add('$name'); throw 'Live external boundary forbidden: $name'"
                Set-Item -LiteralPath "Function:script:$name" -Value ([scriptblock]::Create($body))
            }
            . $source
            Export-ModuleMember -Function Get-NetworkDiagnostics
        }
        $script:classificationResult = & $script:classificationModule {
            param($includeArgument, $includeValue)
            $arguments = @{TestConnectivity=$false; TestDNS=$false}
            if ($includeArgument) { $arguments.IncludeVirtual = $includeValue }
            Get-NetworkDiagnostics @arguments
        } $IncludeArgument $IncludeValue
        $script:classificationDenied = @(& $script:classificationModule { $script:Denied.ToArray() })
        $script:classificationCalls = @(& $script:classificationModule { $script:Calls.ToArray() })
    }
    It 'keeps the two physical names in physical output' {
        ($script:classificationResult.PhysicalAdapters.Name -join ',') | Should -BeExactly 'Owned Ethernet,Owned Wi-Fi'
    }
    It 'counts only the two physical adapters as physical' {
        @($script:classificationResult.PhysicalAdapters).Count | Should -Be 2
    }
    It 'includes virtual adapters only when requested' {
        (@($script:classificationResult.VirtualAdapters | ForEach-Object { $_.Name }) -join ',') | Should -BeExactly $ExpectedVirtual
    }
    It 'summarizes the requested adapter population' {
        $script:classificationResult.Summary.TotalAdapters | Should -Be $ExpectedTotal
    }
    It 'preserves the existing active physical adapter summary' {
        $script:classificationResult.Summary.ActiveAdapters | Should -Be 1
    }
    It 'uses inert collectors and never reaches an external boundary' {
        $script:classificationDenied.Count | Should -Be 0
        $script:classificationCalls | Should -Contain 'Get-NetAdapter'
        $script:classificationCalls | Should -Contain 'IPConfig:101'
        $script:classificationCalls | Should -Contain 'Get-NetRoute'
    }
    It 'collects virtual adapter IP configuration only when requested' {
        if (-not $IncludeValue) {
            $script:classificationCalls | Should -Not -Contain 'IPConfig:201'
            $script:classificationCalls | Should -Not -Contain 'IPConfig:202'
        }
        else {
            $script:classificationCalls | Should -Contain 'IPConfig:201'
            $script:classificationCalls | Should -Contain 'IPConfig:202'
        }
    }
    AfterAll { Remove-Module $script:classificationModule -ErrorAction Stop }
}
