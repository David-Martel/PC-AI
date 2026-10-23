#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Drivers = Import-Module (Join-Path $script:RepoRoot 'Modules/PC-AI.Drivers/PC-AI.Drivers.psd1') -Force -PassThru
    # Only external boundaries are replaced. Both bridges, the operator entrypoint
    # and Set-ThunderboltNetworkOptimization execute their actual implementations.
    & $script:Drivers {
        function script:netsh {
            param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
            throw 'Unmocked netsh is forbidden in this suite.'
        }
        function script:Get-NetAdapter { param($Name, [switch]$IncludeHidden) throw 'Unmocked adapter I/O forbidden.' }
        function script:Get-NetIPAddress { param($AddressFamily) throw 'Unmocked address I/O forbidden.' }
        function script:Get-NetRoute { param($AddressFamily) throw 'Unmocked route I/O forbidden.' }
    }
    function powershell.exe { throw 'Legacy shell launch is forbidden in this suite.' }
    function Remove-NetIPAddress { throw 'Address deletion is forbidden in this suite.' }
    function Set-NetConnectionProfile { throw 'Network profile mutation is forbidden in this suite.' }
    $script:PreviousGlobalNetsh = Get-Item -LiteralPath Function:global:netsh -ErrorAction SilentlyContinue
    function global:netsh {
        param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
        throw 'Real netsh is forbidden even if a module reload displaces a proxy.'
    }
}

AfterAll {
    if ($script:PreviousGlobalNetsh) {
        Set-Item -LiteralPath Function:global:netsh -Value $script:PreviousGlobalNetsh.ScriptBlock
    }
    else {
        Remove-Item -LiteralPath Function:global:netsh
    }
    Remove-Variable -Scope Global -Name PcaiLegacyFixtureStatusCalls, PcaiLegacyFixtureNativeCalls, PcaiLegacyFixtureFailAt, PcaiLegacyFixtureBeforeStatus, PcaiLegacyFixtureAfterStatus, PcaiLegacyFixtureAddresses, PcaiLegacyFixtureRoutes
}

Describe 'Legacy bridge <Name>' -ForEach @(
    @{ Name = 'Initialize-ThunderboltLink.ps1'; DefaultAddress = '172.31.240.1'; Role = 'Local' },
    @{ Name = 'Bootstrap-ThunderboltPeerRemote.ps1'; DefaultAddress = '172.31.240.2'; Role = 'WindowsPeer' }
) -Tag 'Unit', 'Drivers', 'Portable' {
    BeforeAll {
        $script:Bridge = Join-Path $script:RepoRoot "Tools/$Name"
        function Assert-NoLegacyMutation {
            $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 0
            Should -Invoke Import-Module -Exactly -Times 0
            Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 0
            Should -Invoke powershell.exe -Exactly -Times 0
            Should -Invoke Remove-NetIPAddress -Exactly -Times 0
            Should -Invoke Set-NetConnectionProfile -Exactly -Times 0
            Should -Invoke Enable-PSRemoting -Exactly -Times 0
        }
    }
    BeforeEach {
        $global:PcaiLegacyFixtureStatusCalls = 0
        $global:PcaiLegacyFixtureNativeCalls = [Collections.Generic.List[object]]::new()
        $global:PcaiLegacyFixtureFailAt = 0
        $global:PcaiLegacyFixtureBeforeStatus = @([pscustomobject]@{
                InterfaceAlias = 'Ethernet 11'; IPv4Metric = 25; IPv6Metric = 25
                IPv4Mtu = 1500; IPv4Addresses = @('192.168.50.79')
            })
        $global:PcaiLegacyFixtureAfterStatus = @([pscustomobject]@{
                InterfaceAlias = 'Ethernet 11'; IPv4Metric = 19; IPv6Metric = 19
                IPv4Mtu = 9000; IPv4Addresses = @('192.168.50.79')
            })
        $global:PcaiLegacyFixtureAddresses = @([pscustomobject]@{ InterfaceIndex = 7; IPAddress = '192.168.50.79'; PrefixLength = 24; SkipAsSource = $false; PrefixOrigin = 'Manual'; SuffixOrigin = 'Manual'; AddressState = 'Preferred' })
        $global:PcaiLegacyFixtureRoutes = @([pscustomobject]@{ InterfaceIndex = 9; DestinationPrefix = '0.0.0.0/0'; NextHop = '192.168.50.1'; RouteMetric = 0; Protocol = 'Local' })
        Mock Get-NetAdapter -ModuleName PC-AI.Drivers {
            [pscustomobject]@{ Name = 'Ethernet 11'; InterfaceDescription = 'USB4 P2P Network Adapter'; PnPDeviceID = 'ROOT\PROT_USB4NET\1'; Status = 'Up'; ifIndex = 7; InterfaceGuid = 'peer-fixture'; MacAddress = '02-00-00-00-00-07' }
        }
        Mock Get-NetIPAddress -ModuleName PC-AI.Drivers { $global:PcaiLegacyFixtureAddresses }
        Mock Get-NetRoute -ModuleName PC-AI.Drivers { $global:PcaiLegacyFixtureRoutes }
        # Module loading is an external runtime boundary. The real module is
        # already imported; retain its external-call proxies when the real
        # operator entrypoint requests a Force reload.
        Mock Import-Module { } -ParameterFilter { $Name -like '*PC-AI.Drivers.psd1' }
        Mock Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers {
            $global:PcaiLegacyFixtureStatusCalls++
            if ($global:PcaiLegacyFixtureStatusCalls -eq 1) { $global:PcaiLegacyFixtureBeforeStatus } else { $global:PcaiLegacyFixtureAfterStatus }
        }
        Mock netsh -ModuleName PC-AI.Drivers {
            param($Arguments)
            $global:PcaiLegacyFixtureNativeCalls.Add(@($Arguments))
            $global:LASTEXITCODE = if ($global:PcaiLegacyFixtureNativeCalls.Count -eq $global:PcaiLegacyFixtureFailAt) { 5 } else { 0 }
            if ($global:LASTEXITCODE -eq 0 -and $Arguments -contains 'address') {
                $address = ($Arguments | Where-Object { $_ -like 'address=*' }).Substring(8)
                $mask = [Net.IPAddress]::Parse(($Arguments | Where-Object { $_ -like 'mask=*' }).Substring(5)).GetAddressBytes()
                $prefix = ($mask | ForEach-Object { @([Convert]::ToString($_, 2).ToCharArray() | Where-Object { $_ -eq '1' }).Count } | Measure-Object -Sum).Sum
                $bytes = [Net.IPAddress]::Parse($address).GetAddressBytes()
                $network = (0..3 | ForEach-Object { $bytes[$_] -band $mask[$_] }) -join '.'
                $global:PcaiLegacyFixtureAddresses += [pscustomobject]@{ InterfaceIndex = 7; IPAddress = $address; PrefixLength = $prefix; SkipAsSource = $false; PrefixOrigin = 'Manual'; SuffixOrigin = 'Manual'; AddressState = 'Preferred' }
                $global:PcaiLegacyFixtureRoutes += [pscustomobject]@{ InterfaceIndex = 7; DestinationPrefix = "$network/$prefix"; NextHop = '0.0.0.0'; RouteMetric = 0; Protocol = 'Local' }
                $global:PcaiLegacyFixtureAfterStatus[0].IPv4Addresses = @($address)
            }
            'external diagnostic output'
        }
        Mock powershell.exe { throw 'Shell launch is forbidden.' }
        Mock Remove-NetIPAddress { throw 'Address deletion is forbidden.' }
        Mock Set-NetConnectionProfile { throw 'Network profile mutation is forbidden.' }
        Mock Enable-PSRemoting { throw 'Remoting mutation is forbidden.' }
    }

    It 'returns the correct legacy Windows address intent without discovery or mutation' {
        $result = & $script:Bridge
        $result.State | Should -Be 'Planned'
        $result.Applied | Should -BeFalse
        $result.IPv4Address | Should -BeExactly $DefaultAddress
        $result.PrefixLength | Should -Be 30
        $result.CompatibilityRole | Should -BeExactly $Role
        $result.SetPrivateProfile | Should -BeFalse
        $result.EnablePsRemoting | Should -BeFalse
        $result.StaticIPv4Safety | Should -Match 'a plan does not establish live safety'
        $result.ApplyBlockers -join ' ' | Should -Match 'explicit InterfaceAlias'
        Assert-NoLegacyMutation
    }

    It 'implements actual -h and --help before validation or module loading' {
        foreach ($text in @((& $script:Bridge -h -IPv4Address 'invalid-ip' -Apply), (& $script:Bridge --help -IPv4Address 'invalid-ip' -Apply))) {
            $text | Should -Match '^Usage: pwsh'
            $text | Should -Match ([regex]::Escape($Name))
            $text | Should -Match 'PowerShell 7 is required'
            $text | Should -Match 'Windows peers require pwsh'
            $text | Should -Match 'preservation guards'
        }
        Assert-NoLegacyMutation
    }

    It 'rejects unexpected positional settings before invoking the maintained path' {
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply 'unexpected' -Confirm:$false } |
            Should -Throw '*Use named parameters*'
        Assert-NoLegacyMutation
    }

    It 'retains explicit static and security requests in a dry-run plan' {
        $result = & $script:Bridge -InterfaceAlias LAN -IPv4Address '10.42.0.1' -PrefixLength 24 -InterfaceMetric 31 -MtuBytes 1500 -SetPrivateProfile -EnablePsRemoting -Apply -DryRun
        $result.IPv4Address | Should -BeExactly '10.42.0.1'
        $result.PrefixLength | Should -Be 24
        $result.InterfaceMetric | Should -Be 31
        $result.MtuBytes | Should -Be 1500
        $result.SetPrivateProfile | Should -BeTrue
        $result.EnablePsRemoting | Should -BeTrue
        $result.Applied | Should -BeFalse
        $result.ApplyBlockers.Count | Should -Be 2
        Assert-NoLegacyMutation
    }

    It 'refuses replacing an existing unrelated address through the actual central guard' {
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -IPv4Address '10.42.0.1' -Apply -Confirm:$false } |
            Should -Throw '*Unrelated IPv4 address or mismatched prefix*'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 0
        $global:PcaiLegacyFixtureAddresses[0].IPAddress | Should -BeExactly '192.168.50.79'
        Should -Invoke Remove-NetIPAddress -Exactly -Times 0
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 1
    }

    It 'rejects invalid IPv4 before any external call' -ForEach @(
        @{ Address = 'invalid-ip' }, @{ Address = '::1' }, @{ Address = '' }
    ) {
        { & $script:Bridge -InterfaceAlias LAN -IPv4Address $Address -Apply -Confirm:$false } |
            Should -Throw '*IPv4Address must be a valid IPv4*'
        Assert-NoLegacyMutation
    }

    It 'does not silently drop explicit static settings under MetricOnly' -ForEach @(
        @{ Extra = @{ IPv4Address = '10.42.0.1' } }, @{ Extra = @{ PrefixLength = 24 } }
    ) {
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply @Extra -Confirm:$false } |
            Should -Throw '*MetricOnly cannot be combined*'
        Assert-NoLegacyMutation
    }

    It 'requires an explicit adapter before applying metric-only tuning' {
        { & $script:Bridge -MetricOnly -Apply -Confirm:$false } | Should -Throw '*explicit InterfaceAlias*'
        Assert-NoLegacyMutation
    }

    It 'refuses explicit security mutations without changing the host policy' -ForEach @(
        @{ Extra = @{ EnablePsRemoting = $true }; Message = '*WinRM enablement*' },
        @{ Extra = @{ SetPrivateProfile = $true }; Message = '*Private network profile*' }
    ) {
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply @Extra -Confirm:$false } |
            Should -Throw $Message
        Assert-NoLegacyMutation
    }

    It 'keeps metric-only Apply DryRun and WhatIf nonmutating' {
        foreach ($extra in @(@{ DryRun = $true }, @{ WhatIf = $true })) {
            $result = & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply @extra
            $result.Applied | Should -BeFalse
            $result.MetricOnly | Should -BeTrue
            $result.IPv4Address | Should -BeNullOrEmpty
            $result.SuggestedCompatibilityIPv4 | Should -BeExactly $DefaultAddress
        }
        Assert-NoLegacyMutation
    }

    It 'retains guarded static intent under Apply WhatIf without mutation' {
        $result = & $script:Bridge -InterfaceAlias LAN -Apply -WhatIf
        $result.Applied | Should -BeFalse
        $result.IPv4Address | Should -BeExactly $DefaultAddress
        $result.StaticIPv4Safety | Should -Match 'preservation checks'
        Assert-NoLegacyMutation
    }

    It 'respects an actual declined ShouldProcess confirmation in a guarded child (MetricOnly=<MetricMode>)' -ForEach @(
        @{ MetricMode = $true }, @{ MetricMode = $false }
    ) {
        $escapedBridge = $script:Bridge.Replace("'", "''")
        $requestedOption = if ($MetricMode) { '-MetricOnly' } else { "-IPv4Address '10.42.0.1'" }
        $child = "function Invoke-DeniedImportBoundary { throw 'Optimizer invoked after declined confirmation.' }; Set-Alias -Name Import-Module -Value Invoke-DeniedImportBoundary; `$result = & '$escapedBridge' -InterfaceAlias 'Ethernet 11' $requestedOption -Apply -Confirm:`$true; 'PCAI_LEGACY_DENIED:' + (`$result | ConvertTo-Json -Compress)"
        $shellName = if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' }
        $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME $shellName))
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardInput = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-InputFormat', 'Text', '-OutputFormat', 'Text', '-EncodedCommand', [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child)))) {
            $start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            [void]$process.Start()
            $output = $process.StandardOutput.ReadToEndAsync()
            $errorOutput = $process.StandardError.ReadToEndAsync()
            $process.StandardInput.WriteLine('N')
            $process.StandardInput.Close()
            if (-not $process.WaitForExit(15000)) {
                $process.Kill($true)
                throw 'Declined confirmation witness timed out.'
            }
            $stdout = $output.GetAwaiter().GetResult()
            $stderr = $errorOutput.GetAwaiter().GetResult()
            $process.ExitCode | Should -Be 0 -Because $stderr
            $stdout | Should -Match 'PCAI_LEGACY_DENIED:(\{[^\r\n]*\})'
            $result = [regex]::Match($stdout, 'PCAI_LEGACY_DENIED:(\{[^\r\n]*\})').Groups[1].Value | ConvertFrom-Json
            $result.State | Should -BeExactly 'Planned'
            $result.Applied | Should -BeFalse
            $result.MetricOnly | Should -Be $MetricMode
            if (-not $MetricMode) { $result.IPv4Address | Should -BeExactly '10.42.0.1' }
            $result.InterfaceAlias | Should -BeExactly 'Ethernet 11'
        }
        finally {
            $process.Dispose()
        }
        Assert-NoLegacyMutation
    }

    It 'runs the actual maintained optimizer and verifies exactly three metric/MTU calls' {
        $result = & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false
        $result.State | Should -BeExactly 'Applied'
        $result.Applied | Should -BeTrue
        $result.VerifiedStatus.IPv4Metric | Should -Be 19
        $result.VerifiedStatus.IPv6Metric | Should -Be 19
        $result.VerifiedStatus.IPv4Mtu | Should -Be 9000
        $result.VerifiedStatus.IPv4Addresses | Should -Be @('192.168.50.79')
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 3
        $global:PcaiLegacyFixtureNativeCalls[0] | Should -Be @('interface', 'ipv4', 'set', 'interface', 'name=Ethernet 11', 'metric=19')
        $global:PcaiLegacyFixtureNativeCalls[1] | Should -Be @('interface', 'ipv6', 'set', 'interface', 'Ethernet 11', 'metric=19')
        $global:PcaiLegacyFixtureNativeCalls[2] | Should -Be @('interface', 'ipv4', 'set', 'subinterface', 'Ethernet 11', 'mtu=9000', 'store=persistent')
        Should -Invoke Import-Module -Exactly -Times 1 -ParameterFilter { $Name -like '*PC-AI.Drivers.psd1' -and $Force }
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 2 -ParameterFilter { $InterfaceAlias -eq 'Ethernet 11' }
        Should -Invoke Remove-NetIPAddress -Exactly -Times 0
        Should -Invoke Enable-PSRemoting -Exactly -Times 0
        Should -Invoke powershell.exe -Exactly -Times 0
    }

    It 'propagates the actual optimizer failure and stops at native step <Step>' -ForEach @(
        @{ Step = 1 }, @{ Step = 2 }, @{ Step = 3 }
    ) {
        $global:PcaiLegacyFixtureFailAt = $Step
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false } |
            Should -Throw '*netsh exit 5*external diagnostic output*'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be $Step
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 1
    }

    It 'propagates actual central adapter refusal before netsh' -ForEach @(
        @{ Kind = 'empty' }, @{ Kind = 'nonmatching' }, @{ Kind = 'duplicate' }
    ) {
        $global:PcaiLegacyFixtureBeforeStatus = switch ($Kind) {
            'empty' { @() }
            'nonmatching' { @([pscustomobject]@{ InterfaceAlias = 'LAN' }) }
            'duplicate' { @($global:PcaiLegacyFixtureBeforeStatus[0], $global:PcaiLegacyFixtureBeforeStatus[0]) }
        }
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply -Confirm:$false } |
            Should -Throw '*Thunderbolt / USB4 interface*'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 1 -ParameterFilter { $InterfaceAlias -eq 'Ethernet 11' }
    }

    It 'does not report applied when the actual optimizer postcondition fails' {
        $global:PcaiLegacyFixtureAfterStatus[0].IPv4Mtu = 1500
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -MetricOnly -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false } |
            Should -Throw '*postcondition failed*MTU expected 9000, observed 1500*'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 3
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 2
    }

    It 'applies the legacy default address through automatically loaded private guards' {
        $global:PcaiLegacyFixtureAddresses = @()
        $global:PcaiLegacyFixtureBeforeStatus[0].IPv4Addresses = @()
        $result = & $script:Bridge -InterfaceAlias 'Ethernet 11' -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false
        $result.Applied | Should -BeTrue
        $result.IPv4Address | Should -BeExactly $DefaultAddress
        $result.VerifiedStatus.IPv4Addresses | Should -Be @($DefaultAddress)
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 4
        $global:PcaiLegacyFixtureNativeCalls[3] | Should -Contain "address=$DefaultAddress"
        $global:PcaiLegacyFixtureNativeCalls[3] | Should -Contain 'mask=255.255.255.252'
        $global:PcaiLegacyFixtureAddresses[0].PrefixLength | Should -Be 30
        $global:PcaiLegacyFixtureRoutes[0].DestinationPrefix | Should -BeExactly '0.0.0.0/0'
        Should -Invoke Get-NetAdapter -ModuleName PC-AI.Drivers -Exactly -Times 3
        Should -Invoke Remove-NetIPAddress -Exactly -Times 0
        Should -Invoke Enable-PSRemoting -Exactly -Times 0
    }

    It 'propagates static native failure without reporting applied' {
        $global:PcaiLegacyFixtureAddresses = @()
        $global:PcaiLegacyFixtureBeforeStatus[0].IPv4Addresses = @()
        $global:PcaiLegacyFixtureFailAt = 4
        { & $script:Bridge -InterfaceAlias 'Ethernet 11' -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false } | Should -Throw '*SetStaticIPv4*netsh exit 5*'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 4
        $global:PcaiLegacyFixtureAddresses.Count | Should -Be 0
    }

    It 'retains explicit static DryRun without any live guard or mutation' {
        $result = & $script:Bridge -InterfaceAlias 'Ethernet 11' -IPv4Address '10.42.0.1' -PrefixLength 24 -Apply -DryRun
        $result.Applied | Should -BeFalse
        $result.IPv4Address | Should -BeExactly '10.42.0.1'
        $result.PrefixLength | Should -Be 24
        Assert-NoLegacyMutation
        Should -Invoke Get-NetAdapter -ModuleName PC-AI.Drivers -Exactly -Times 0
    }

    It 'dispatches actual metric-only Optimize without injecting an empty IPv4 parameter' {
        $result = & (Join-Path $script:RepoRoot 'Tools/Invoke-ThunderboltNetworking.ps1') -Mode Optimize -InterfaceAlias 'Ethernet 11' -InterfaceMetric 19 -MtuBytes 9000 -Apply -Confirm:$false
        $result.IPv4Addresses | Should -Be @('192.168.50.79')
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 3
    }

    It 'dispatches explicit static prefix through the actual driver' {
        $global:PcaiLegacyFixtureAddresses = @()
        $global:PcaiLegacyFixtureBeforeStatus[0].IPv4Addresses = @()
        $result = & (Join-Path $script:RepoRoot 'Tools/Invoke-ThunderboltNetworking.ps1') -Mode Optimize -InterfaceAlias 'Ethernet 11' -InterfaceMetric 19 -MtuBytes 9000 -IPv4Address '10.42.0.1' -PrefixLength 24 -Apply -Confirm:$false
        $result.IPv4Addresses | Should -Be @('10.42.0.1')
        $global:PcaiLegacyFixtureAddresses[0].PrefixLength | Should -Be 24
        $global:PcaiLegacyFixtureNativeCalls[3] | Should -Contain 'mask=255.255.255.0'
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 4
    }

    It 'does not silently omit explicitly invalid static dispatcher input: <Kind>' -ForEach @(
        @{ Kind = 'emptyaddress'; Extra = @{ IPv4Address = '' }; Message = '*IPv4Address must be a valid IPv4*' }
        @{ Kind = 'invalidprefix'; Extra = @{ IPv4Address = '10.42.0.1'; PrefixLength = 31 }; Message = '*PrefixLength*' }
        @{ Kind = 'prefixonly'; Extra = @{ PrefixLength = 24 }; Message = '*PrefixLength requires an explicitly requested IPv4Address*' }
    ) {
        { & (Join-Path $script:RepoRoot 'Tools/Invoke-ThunderboltNetworking.ps1') -Mode Optimize -InterfaceAlias 'Ethernet 11' -Apply @Extra -Confirm:$false } | Should -Throw $Message
        $global:PcaiLegacyFixtureNativeCalls.Count | Should -Be 0
        Should -Invoke Get-ThunderboltNetworkStatus -ModuleName PC-AI.Drivers -Exactly -Times 0
    }
}
