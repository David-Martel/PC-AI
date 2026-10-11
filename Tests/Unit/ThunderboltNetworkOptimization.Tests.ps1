#Requires -Version 7.0
Describe 'Set-ThunderboltNetworkOptimization' -Tag 'Unit', 'Drivers', 'Portable' {
    BeforeAll {
        $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        . (Join-Path $repoRoot 'Modules/PC-AI.Drivers/Private/Assert-ThunderboltIPv4Safety.ps1')
        . (Join-Path $repoRoot 'Modules/PC-AI.Drivers/Public/Set-ThunderboltNetworkOptimization.ps1')
        # External command guards: no device or network command can execute in this suite.
        function Get-ThunderboltNetworkStatus {
            param([string]$InterfaceAlias)
            throw 'Unmocked adapter discovery is forbidden in this suite.'
        }
        function netsh {
            param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
            throw 'Unmocked netsh is forbidden in this suite.'
        }
        function Get-NetAdapter { param($Name, [switch]$IncludeHidden) throw 'Unmocked adapter I/O forbidden.' }
        function Get-NetIPAddress { param($AddressFamily) throw 'Unmocked address I/O forbidden.' }
        function Get-NetRoute { param($AddressFamily) throw 'Unmocked route I/O forbidden.' }
        function New-FixtureAddress {
            param($Index, $Address, $Prefix)
            [pscustomobject]@{ InterfaceIndex = $Index; IPAddress = $Address; PrefixLength = $Prefix; SkipAsSource = $false; PrefixOrigin = 'Manual'; SuffixOrigin = 'Manual'; AddressState = 'Preferred' }
        }
        function New-FixtureRoute {
            param($Index, $Destination, $Hop = '0.0.0.0')
            [pscustomobject]@{ InterfaceIndex = $Index; DestinationPrefix = $Destination; NextHop = $Hop; RouteMetric = 0; Protocol = 'Local' }
        }
    }
    BeforeEach {
        $script:BeforeStatus = @([pscustomobject]@{
                InterfaceAlias = 'Ethernet 11'; IPv4Metric = 25; IPv6Metric = 25
                IPv4Mtu = 1500; IPv4Addresses = @()
            })
        $script:AfterStatus = @([pscustomobject]@{
                InterfaceAlias = 'Ethernet 11'; IPv4Metric = 15; IPv6Metric = 15
                IPv4Mtu = 62000; IPv4Addresses = @()
            })
        $script:StatusCalls = 0
        $script:NativeCalls = [System.Collections.Generic.List[object]]::new()
        $script:FailAt = 0
        $script:Adapters = @([pscustomobject]@{ Name = 'Ethernet 11'; InterfaceDescription = 'USB4 P2P Network Adapter'; PnPDeviceID = 'ROOT\PROT_USB4NET\1'; Status = 'Up'; ifIndex = 7; InterfaceGuid = 'peer-fixture'; MacAddress = '02-00-00-00-00-07' })
        $script:Addresses = @(New-FixtureAddress 9 '192.168.50.79' 24)
        $script:Routes = @((New-FixtureRoute 9 '192.168.50.0/24'), (New-FixtureRoute 9 '0.0.0.0/0' '192.168.50.1'))
        $script:AfterNative = $null
        Mock Get-NetAdapter { $script:Adapters }
        Mock Get-NetIPAddress { $script:Addresses }
        Mock Get-NetRoute { $script:Routes }
        Mock Get-ThunderboltNetworkStatus {
            $script:StatusCalls++
            if ($script:StatusCalls -eq 1) { $script:BeforeStatus } else { $script:AfterStatus }
        }
        Mock netsh {
            param($Arguments)
            $script:NativeCalls.Add(@($Arguments))
            $global:LASTEXITCODE = if ($script:NativeCalls.Count -eq $script:FailAt) { 5 } else { 0 }
            if ($global:LASTEXITCODE -eq 0 -and $Arguments -contains 'address') {
                $address = ($Arguments | Where-Object { $_ -like 'address=*' }).Substring(8)
                $mask = [Net.IPAddress]::Parse(($Arguments | Where-Object { $_ -like 'mask=*' }).Substring(5)).GetAddressBytes()
                $prefix = ($mask | ForEach-Object { @([Convert]::ToString($_, 2).ToCharArray() | Where-Object { $_ -eq '1' }).Count } | Measure-Object -Sum).Sum
                $bytes = [Net.IPAddress]::Parse($address).GetAddressBytes()
                $network = (0..3 | ForEach-Object { $bytes[$_] -band $mask[$_] }) -join '.'
                $script:Addresses += New-FixtureAddress 7 $address $prefix
                $script:Routes += New-FixtureRoute 7 "$network/$prefix"
                $script:AfterStatus[0].IPv4Addresses = @($address)
            }
            if ($script:AfterNative) { & $script:AfterNative $script:NativeCalls.Count }
            'external diagnostic output'
        }
    }

    It 'returns the existing default plan for the only adapter without executing netsh' {
        $result = Set-ThunderboltNetworkOptimization
        $result.InterfaceAlias | Should -Be 'Ethernet 11'
        $result.CurrentStatus.IPv4Mtu | Should -Be 1500
        $result.PlannedActions.Count | Should -Be 3
        $result.PlannedActions[2].Arguments | Should -Contain 'mtu=62000'
        $script:NativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 1
    }

    It 'matches an explicit alias exactly and does not fall back on an empty or different result' -ForEach @(
        @{ Kind = 'empty' }, @{ Kind = 'nonmatching' }
    ) {
        $script:BeforeStatus = if ($Kind -eq 'empty') { @() } else {
            @([pscustomobject]@{ InterfaceAlias = 'Ethernet 99' })
        }
        { Set-ThunderboltNetworkOptimization -InterfaceAlias 'Ethernet 11' -Apply } |
            Should -Throw '*No Thunderbolt / USB4 interface matched*'
        $script:NativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 1 -ParameterFilter { $InterfaceAlias -eq 'Ethernet 11' }
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 0 -ParameterFilter { -not $InterfaceAlias }
    }

    It 'rejects no adapters when alias is omitted' {
        $script:BeforeStatus = @()
        { Set-ThunderboltNetworkOptimization } | Should -Throw '*No Thunderbolt / USB4 interface matched*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects ambiguous automatic adapter selection' {
        $script:BeforeStatus += [pscustomobject]@{ InterfaceAlias = 'Ethernet 12' }
        { Set-ThunderboltNetworkOptimization -Apply } | Should -Throw '*Multiple Thunderbolt / USB4 interfaces matched*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects duplicate exact alias matches' {
        $script:BeforeStatus += $script:BeforeStatus[0]
        { Set-ThunderboltNetworkOptimization -InterfaceAlias 'Ethernet 11' } |
            Should -Throw '*Multiple Thunderbolt / USB4 interfaces matched*'
    }

    It 'rejects devices without a network alias' {
        $script:BeforeStatus[0].InterfaceAlias = $null
        { Set-ThunderboltNetworkOptimization } | Should -Throw '*no network interface alias*'
    }

    It 'does not execute the plan under Apply WhatIf' {
        $result = Set-ThunderboltNetworkOptimization -InterfaceAlias 'Ethernet 11' -Apply -WhatIf
        $result | Should -BeNullOrEmpty
        $script:NativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 1
    }

    It 'applies all three steps and returns independently verified status' {
        $global:LASTEXITCODE = 97
        $result = Set-ThunderboltNetworkOptimization -Apply -Confirm:$false
        $script:NativeCalls.Count | Should -Be 3
        ($script:NativeCalls[0] -join '|') | Should -Be 'interface|ipv4|set|interface|name=Ethernet 11|metric=15'
        ($script:NativeCalls[1] -join '|') | Should -Be 'interface|ipv6|set|interface|Ethernet 11|metric=15'
        ($script:NativeCalls[2] -join '|') | Should -Be 'interface|ipv4|set|subinterface|Ethernet 11|mtu=62000|store=persistent'
        $result.IPv4Metric | Should -Be 15
        $result.IPv6Metric | Should -Be 15
        $result.IPv4Mtu | Should -Be 62000
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 2
    }

    It 'stops immediately at <Step> with its exit code and diagnostic' -ForEach @(
        @{ FailAt = 1; Step = 'SetIPv4Metric' }, @{ FailAt = 2; Step = 'SetIPv6Metric' }
        @{ FailAt = 3; Step = 'SetMtu' }, @{ FailAt = 4; Step = 'SetStaticIPv4' }
    ) {
        $script:FailAt = $FailAt
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } |
            Should -Throw "*$Step*netsh exit 5*Remaining steps were not attempted*external diagnostic output*"
        $script:NativeCalls.Count | Should -Be $FailAt
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 1
    }

    It 'builds and applies the correct IPv4 mask for prefix <Prefix>' -ForEach @(
        @{ Prefix = 8; Mask = '255.0.0.0' }, @{ Prefix = 16; Mask = '255.255.0.0' }
        @{ Prefix = 24; Mask = '255.255.255.0' }, @{ Prefix = 30; Mask = '255.255.255.252' }
    ) {
        $plan = Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -PrefixLength $Prefix
        $plan.PlannedActions.Count | Should -Be 4
        $plan.PlannedActions[3].Arguments | Should -Contain "mask=$Mask"
        $script:NativeCalls.Count | Should -Be 0
        $script:StatusCalls = 0
        $result = Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -PrefixLength $Prefix -Apply -Confirm:$false
        $script:NativeCalls.Count | Should -Be 4
        $script:NativeCalls[3] | Should -Contain "mask=$Mask"
        $script:NativeCalls[3] | Should -Contain 'address=10.42.0.1'
        $script:NativeCalls[3] | Should -Contain 'gateway=none'
        $result.IPv4Addresses | Should -Contain '10.42.0.1'
    }

    It 'rejects non-IPv4 input before discovery or mutations' -ForEach @(
        @{ Address = '10.42.0.999' }, @{ Address = '::1' }
    ) {
        { Set-ThunderboltNetworkOptimization -IPv4Address $Address -Apply } |
            Should -Throw '*IPv4Address must be a valid IPv4 address*'
        $script:NativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 0
    }

    It 'rejects unsuccessful observed <Field> after successful commands' -ForEach @(
        @{ Field = 'IPv4Metric'; Value = 25; Message = 'IPv4 metric expected 15' }
        @{ Field = 'IPv6Metric'; Value = 25; Message = 'IPv6 metric expected 15' }
        @{ Field = 'IPv4Mtu'; Value = 1500; Message = 'IPv4 MTU expected 62000' }
    ) {
        $script:AfterStatus[0].$Field = $Value
        { Set-ThunderboltNetworkOptimization -Apply -Confirm:$false } | Should -Throw "*postcondition failed*$Message*"
        $script:NativeCalls.Count | Should -Be 3
        Should -Invoke Get-ThunderboltNetworkStatus -Exactly -Times 2
    }

    It 'rejects a static address absent from observed status' {
        $script:AfterStatus[0].IPv4Addresses = @('169.254.1.2')
        $script:AfterNative = { param($Step) if ($Step -eq 4) { $script:AfterStatus[0].IPv4Addresses = @('169.254.1.2') } }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } |
            Should -Throw '*postcondition failed*IPv4 address 10.42.0.1 was not observed*'
        $script:NativeCalls.Count | Should -Be 4
    }

    It 'rejects a missing or ambiguous postcondition adapter' -ForEach @(
        @{ Kind = 'missing' }, @{ Kind = 'ambiguous' }
    ) {
        $script:AfterStatus = if ($Kind -eq 'missing') { @() } else { @($script:AfterStatus[0], $script:AfterStatus[0]) }
        { Set-ThunderboltNetworkOptimization -Apply -Confirm:$false } |
            Should -Throw '*selected adapter could not be uniquely verified*'
        $script:NativeCalls.Count | Should -Be 3
    }

    It 'adds the requested prefix through the real guard and preserves LAN recovery state' {
        $beforeAddresses = $script:Addresses | ConvertTo-Json -Compress
        $beforeRoutes = $script:Routes | ConvertTo-Json -Compress
        $result = Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -PrefixLength 30 -Apply -Confirm:$false
        $result.IPv4Addresses | Should -Be @('10.42.0.1')
        @($script:Addresses | Where-Object InterfaceIndex -EQ 7)[0].PrefixLength | Should -Be 30
        ($script:Addresses | Where-Object InterfaceIndex -EQ 9 | ConvertTo-Json -Compress) | Should -BeExactly $beforeAddresses
        ($script:Routes | Where-Object InterfaceIndex -EQ 9 | ConvertTo-Json -Compress) | Should -BeExactly $beforeRoutes
        $script:NativeCalls.Count | Should -Be 4
    }

    It 'preserves an unrelated selected address by refusing static replacement: <Address>/<Prefix>' -ForEach @(
        @{ Address = '169.254.1.2'; Prefix = 16 }, @{ Address = '192.168.70.2'; Prefix = 24 },
        @{ Address = '10.42.0.1'; Prefix = 24 }
    ) {
        $script:Addresses += New-FixtureAddress 7 $Address $Prefix
        $script:BeforeStatus[0].IPv4Addresses = @($Address)
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*Unrelated IPv4 address or mismatched prefix*'
        $script:NativeCalls.Count | Should -Be 0
        @($script:Addresses | Where-Object InterfaceIndex -EQ 7)[0].PrefixLength | Should -Be $Prefix
    }

    It 'keeps existing exact address/prefix idempotent without set-address' {
        $script:Addresses += New-FixtureAddress 7 '10.42.0.1' 30
        $script:Routes += New-FixtureRoute 7 '10.42.0.0/30'
        $script:BeforeStatus[0].IPv4Addresses = @('10.42.0.1')
        $script:AfterStatus[0].IPv4Addresses = @('10.42.0.1')
        $before = $script:Addresses | ConvertTo-Json -Compress
        Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false | Out-Null
        $script:NativeCalls.Count | Should -Be 3
        ($script:Addresses | ConvertTo-Json -Compress) | Should -BeExactly $before
    }

    It 'rejects selected-interface default routes before metrics' {
        $script:Routes += New-FixtureRoute 7 '0.0.0.0/0' '10.42.0.2'
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*peer interface has a default route*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects overlapping foreign routes before metrics: <Destination>' -ForEach @(
        @{ Destination = '10.42.0.2/32' }, @{ Destination = '10.42.0.0/24' }, @{ Destination = '10.0.0.0/8' }
    ) {
        $script:Routes += New-FixtureRoute 9 $Destination '192.168.50.1'
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*overlaps a route on another interface*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects overlapping foreign address prefixes without relying on routes: <Address>/<Prefix>' -ForEach @(
        @{ Address = '10.42.0.2'; Prefix = 30 }, @{ Address = '10.42.0.1'; Prefix = 32 }, @{ Address = '10.20.0.1'; Prefix = 8 }
    ) {
        $script:Addresses += New-FixtureAddress 9 $Address $Prefix
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*overlaps an address on another interface*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects fresh OS adapter identity problems before mutation: <Kind>' -ForEach @(
        @{ Kind = 'absent' }, @{ Kind = 'different' }, @{ Kind = 'duplicate' }, @{ Kind = 'ordinary' }, @{ Kind = 'down' }, @{ Kind = 'badindex' }
    ) {
        switch ($Kind) {
            absent { $script:Adapters = @() }
            different { $script:Adapters[0].Name = 'LAN' }
            duplicate { $script:Adapters += $script:Adapters[0] }
            ordinary { $script:Adapters[0].InterfaceDescription = 'Intel Ethernet'; $script:Adapters[0].PnPDeviceID = 'PCI\ordinary' }
            down { $script:Adapters[0].Status = 'Disconnected' }
            badindex { $script:Adapters[0].ifIndex = 0 }
        }
        $message = switch ($Kind) {
            absent { '*exactly one current adapter*' }
            different { '*exactly one current adapter*' }
            duplicate { '*exactly one current adapter*' }
            ordinary { '*not a genuine Thunderbolt*' }
            down { '*must be Up*' }
            badindex { '*valid interface index*' }
        }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw $message
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects unusable desired host <Address> before native I/O' -ForEach @(
        @{ Address = '10.42.0.0' }, @{ Address = '10.42.0.3' }, @{ Address = '127.0.0.1' }, @{ Address = '224.0.0.1' }, @{ Address = '0.0.0.1' }, @{ Address = '' }
    ) {
        $message = if ($Address -eq '') { '*IPv4Address must be a valid IPv4*' } else { '*usable unicast host address*' }
        { Set-ThunderboltNetworkOptimization -IPv4Address $Address -Apply -Confirm:$false } | Should -Throw $message
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects unsupported prefixes before native I/O: <Prefix>' -ForEach @(@{ Prefix = 7 }, @{ Prefix = 31 }) {
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -PrefixLength $Prefix -Apply -Confirm:$false } | Should -Throw
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'refuses inconsistent preflight status and fresh addresses' {
        $script:BeforeStatus[0].IPv4Addresses = @('10.42.0.1')
        { Set-ThunderboltNetworkOptimization -Apply -Confirm:$false } | Should -Throw '*status and fresh IPv4 address inventory disagree*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'catches a new overlap before address assignment after metrics' {
        $script:AfterNative = { param($Step) if ($Step -eq 3) { $script:Routes += New-FixtureRoute 9 '10.42.0.0/30' } }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*overlaps a route on another interface*'
        $script:NativeCalls.Count | Should -Be 3
        @($script:Addresses | Where-Object InterfaceIndex -EQ 7).Count | Should -Be 0
    }

    It 'does not report success when actual post-state is wrong: <Kind>' -ForEach @(
        @{ Kind = 'wrongprefix' }, @{ Kind = 'foreignaddressremoved' }, @{ Kind = 'foreignprefixchanged' },
        @{ Kind = 'foreignaddressadded' }, @{ Kind = 'foreignroutechanged' }, @{ Kind = 'defaultremoved' },
        @{ Kind = 'unrelatedrouteadded' }, @{ Kind = 'missingsubnetroute' }, @{ Kind = 'tentative' }, @{ Kind = 'skipassource' }, @{ Kind = 'identitychanged' }, @{ Kind = 'dhcporigin' }
    ) {
        $script:PostCorruption = $Kind
        $script:AfterNative = {
            param($Step)
            if ($Step -ne 4) { return }
            switch ($script:PostCorruption) {
                wrongprefix { $script:Addresses[-1].PrefixLength = 24 }
                foreignaddressremoved { $script:Addresses = @($script:Addresses | Where-Object InterfaceIndex -EQ 7) }
                foreignprefixchanged { $script:Addresses[0].PrefixLength = 25 }
                foreignaddressadded { $script:Addresses += New-FixtureAddress 10 '192.168.90.2' 24 }
                foreignroutechanged { $script:Routes[0].RouteMetric = 21 }
                defaultremoved { $script:Routes = @($script:Routes | Where-Object DestinationPrefix -NE '0.0.0.0/0') }
                unrelatedrouteadded { $script:Routes += New-FixtureRoute 7 '172.16.0.0/16' }
                missingsubnetroute { $script:Routes = @($script:Routes | Where-Object InterfaceIndex -NE 7) }
                tentative { $script:Addresses[-1].AddressState = 'Tentative' }
                skipassource { $script:Addresses[-1].SkipAsSource = $true }
                identitychanged { $script:Adapters[0].InterfaceGuid = 'replacement-adapter' }
                dhcporigin { $script:Addresses[-1].PrefixOrigin = 'Dhcp' }
            }
        }
        $message = switch ($Kind) {
            wrongprefix { '*Unrelated IPv4 address or mismatched prefix*' }
            foreignaddressremoved { '*unrelated IPv4 address state changed*' }
            foreignprefixchanged { '*unrelated IPv4 address state changed*' }
            foreignaddressadded { '*unrelated IPv4 address state changed*' }
            foreignroutechanged { '*existing IPv4 route changed or disappeared*' }
            defaultremoved { '*existing IPv4 route changed or disappeared*' }
            unrelatedrouteadded { '*unrelated IPv4 route appeared*' }
            missingsubnetroute { '*requires exactly one current selected-interface on-link subnet route*' }
            tentative { '*not Preferred and usable as a source*' }
            skipassource { '*not Preferred and usable as a source*' }
            identitychanged { '*adapter identity changed after preflight*' }
            dhcporigin { '*not manually configured*' }
        }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw $message
        $script:NativeCalls.Count | Should -Be 4
    }

    It 'preserves pre-existing selected routes while admitting only new on-link routes' {
        $script:Routes += New-FixtureRoute 7 '172.16.0.0/16' '172.16.0.1'
        $script:Routes += New-FixtureRoute 9 '10.42.0.4/30' '192.168.50.1'
        Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false | Out-Null
        @($script:Routes | Where-Object DestinationPrefix -EQ '172.16.0.0/16').Count | Should -Be 1
        $script:NativeCalls.Count | Should -Be 4
    }

    It 'leaves static requests nonmutating with WhatIf' {
        Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -WhatIf | Out-Null
        $script:NativeCalls.Count | Should -Be 0
        Should -Invoke Get-NetAdapter -Times 0 -Exactly
    }

    It 'supports a genuinely empty IPv4 inventory and only the new peer route' {
        $script:Addresses = @()
        $script:Routes = @()
        Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false | Out-Null
        $script:NativeCalls.Count | Should -Be 4
        $script:Addresses.Count | Should -Be 1
        $script:Routes[0].DestinationPrefix | Should -BeExactly '10.42.0.0/30'
    }

    It 'does not hide failed live discovery: <Boundary>' -ForEach @(
        @{ Boundary = 'Get-NetAdapter' }, @{ Boundary = 'Get-NetIPAddress' }, @{ Boundary = 'Get-NetRoute' }
    ) {
        Mock $Boundary { throw 'fixture OS discovery failed' }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw '*fixture OS discovery failed*'
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'refuses duplicate or unusable already-present desired addresses: <Kind>' -ForEach @(
        @{ Kind = 'duplicate' }, @{ Kind = 'tentative' }, @{ Kind = 'skipassource' }, @{ Kind = 'dhcp' }, @{ Kind = 'missingroute' }, @{ Kind = 'duplicateroute' }
    ) {
        $script:Addresses += New-FixtureAddress 7 '10.42.0.1' 30
        $script:BeforeStatus[0].IPv4Addresses = @('10.42.0.1')
        if ($Kind -ne 'missingroute') { $script:Routes += New-FixtureRoute 7 '10.42.0.0/30' }
        switch ($Kind) {
            duplicate { $script:Addresses += New-FixtureAddress 7 '10.42.0.1' 30 }
            tentative { $script:Addresses[-1].AddressState = 'Tentative' }
            skipassource { $script:Addresses[-1].SkipAsSource = $true }
            dhcp { $script:Addresses[-1].PrefixOrigin = 'Dhcp' }
            duplicateroute { $script:Routes += New-FixtureRoute 7 '10.42.0.0/30' }
        }
        $message = switch ($Kind) {
            duplicate { '*address is ambiguous*' }
            dhcp { '*not manually configured*' }
            missingroute { '*requires exactly one current selected-interface on-link subnet route*' }
            duplicateroute { '*requires exactly one current selected-interface on-link subnet route*' }
            default { '*not Preferred and usable as a source*' }
        }
        { Set-ThunderboltNetworkOptimization -IPv4Address '10.42.0.1' -Apply -Confirm:$false } | Should -Throw $message
        $script:NativeCalls.Count | Should -Be 0
    }

    It 'rejects unrelated state loss during metric-only apply' {
        $script:AfterNative = { param($Step) if ($Step -eq 3) { $script:Routes = @() } }
        { Set-ThunderboltNetworkOptimization -Apply -Confirm:$false } | Should -Throw '*IPv4 route state changed*'
        $script:NativeCalls.Count | Should -Be 3
    }
}
