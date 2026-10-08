#Requires -Version 7.0
Describe 'Set-ThunderboltNetworkOptimization' -Tag 'Unit', 'Drivers', 'Portable' {
    BeforeAll {
        $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
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
    }
    BeforeEach {
        $script:BeforeStatus = @([pscustomobject]@{
            InterfaceAlias = 'Ethernet 11'; IPv4Metric = 25; IPv6Metric = 25
            IPv4Mtu = 1500; IPv4Addresses = @('169.254.1.2')
        })
        $script:AfterStatus = @([pscustomobject]@{
            InterfaceAlias = 'Ethernet 11'; IPv4Metric = 15; IPv6Metric = 15
            IPv4Mtu = 62000; IPv4Addresses = @('10.42.0.1')
        })
        $script:StatusCalls = 0
        $script:NativeCalls = [System.Collections.Generic.List[object]]::new()
        $script:FailAt = 0
        Mock Get-ThunderboltNetworkStatus {
            $script:StatusCalls++
            if ($script:StatusCalls -eq 1) { $script:BeforeStatus } else { $script:AfterStatus }
        }
        Mock netsh {
            param($Arguments)
            $script:NativeCalls.Add(@($Arguments))
            $global:LASTEXITCODE = if ($script:NativeCalls.Count -eq $script:FailAt) { 5 } else { 0 }
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
}
