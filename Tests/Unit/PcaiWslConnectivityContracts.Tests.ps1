BeforeAll {
    $script:WslContractSource = if ($env:PCAI_WSL_CONTRACT_SOURCE) {
        $env:PCAI_WSL_CONTRACT_SOURCE
    } else {
        Join-Path $PSScriptRoot '../../Modules/PC-AI.Network/Public/Test-WSLConnectivity.ps1'
    }
    . $script:WslContractSource

    # Local refusal stubs prevent any missing mock from reaching a live host boundary.
    function wsl { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Live WSL is forbidden in contract fixtures.' }
    function Get-WSLDistributions { throw 'Live distribution enumeration is forbidden.' }
    function Get-NetAdapter { [CmdletBinding()] param() throw 'Live adapter enumeration is forbidden.' }
    function Get-NetIPAddress { [CmdletBinding()] param($InterfaceIndex, $AddressFamily) throw 'Live IP enumeration is forbidden.' }
    function Resolve-DnsName { [CmdletBinding()] param($Name, $Type) throw 'Live DNS is forbidden.' }
    function Test-PortConnectivity { param($HostName, $Port, $TimeoutMs) throw 'Live TCP is forbidden.' }
    function Get-CimInstance { [CmdletBinding()] param($ClassName) throw 'Live CIM is forbidden.' }
    function Get-Service { [CmdletBinding()] param($Name) throw 'Live service access is forbidden.' }

    function Invoke-FixtureWsl {
        param([object[]]$Tokens)
        $stage = if ($Tokens -contains '--status') { 'status' }
            elseif ($Tokens -contains 'ip') { 'ip' }
            elseif ($Tokens -contains 'nslookup') { 'dns' }
            elseif ($Tokens -contains 'hostname') { 'hostname' }
            elseif ($Tokens -contains 'curl') { 'curl' }
            else { throw "Unexpected synthetic WSL arguments: $Tokens" }
        $script:WslCalls += $stage
        $exitCode = if ($stage -eq $script:FailedStage) { $script:NativeExit } else { 0 }
        $lines = switch ($stage) {
            'status' {
                if ($script:ArrayStatus) { 'Default Version: 2'; 'Default Distribution: FixtureDistro'; 'Kernel version: 6.6.1' }
                else { 'Default Version: 2' }
            }
            'ip' { '    inet 192.0.2.10/24 scope global eth0' }
            'dns' { 'Address: 192.0.2.53'; 'Address: 198.51.100.20' }
            'hostname' { '192.0.2.10 2001:db8::10' }
            'curl' { '200' }
        }
        if ($script:NativeChild) {
            # Only a private cmd script emits synthetic stdout and exits. It never invokes WSL.
            $shim = Join-Path $TestDrive 'wsl-contract-synthetic.cmd'
            $body = @('@echo off') + @($lines | ForEach-Object { "echo $_" }) + @("exit /b $exitCode")
            [IO.File]::WriteAllLines($shim, $body, [Text.UTF8Encoding]::new($false))
            & $env:ComSpec /d /c $shim
            $global:LASTEXITCODE = $LASTEXITCODE
        } else {
            $global:LASTEXITCODE = $exitCode
            $lines
        }
    }
}

Describe 'WSL connectivity output and native-status contracts' -Tag 'Unit', 'Network', 'Portable' {
    BeforeEach {
        $script:WslCalls = @()
        $script:ArrayStatus = $false
        $script:FailedStage = ''
        $script:NativeExit = 7
        $script:NativeChild = $false
        $global:LASTEXITCODE = 0
        Mock Write-Host { }
        Mock wsl { Invoke-FixtureWsl -Tokens $Tokens }
        Mock Get-WSLDistributions { @([pscustomobject]@{ Name = 'FixtureDistro'; State = 'Running'; Version = 2 }) }
        Mock Get-NetAdapter { [pscustomobject]@{ Name = 'vEthernet (WSL)'; Status = 'Up'; ifIndex = 999 } }
        Mock Get-NetIPAddress { [pscustomobject]@{ IPAddress = '192.0.2.1' } }
        Mock Resolve-DnsName { [pscustomobject]@{ IPAddress = '198.51.100.20' } }
        Mock Test-PortConnectivity { [pscustomobject]@{ Success = $true; Message = 'Synthetic connected' } }
        Mock Get-CimInstance { [pscustomobject]@{ Name = 'Hyper-V Socket'; Status = 'OK' } }
        Mock Get-Service { throw 'Unexpected service fallback.' }
    }

    It 'preserves healthy scalar status and Windows/WSL diagnostic outcomes' {
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @('fixture.invalid') -TestPorts @(22)
        $result.WSLStatus.Installed | Should -BeTrue
        $result.WSLStatus.Version | Should -Be '2'
        @($result.NetworkInterfaces | Where-Object Side -EQ WSL)[0].IPAddress | Should -Be '192.0.2.10'
        @($result.DNSTests | Where-Object Side -EQ WSL)[0].IPAddress | Should -Be '198.51.100.20'
        $result.PortTests[0].Success | Should -BeTrue
        $result.InternetAccess.Success | Should -BeTrue
        $result.Summary.OverallStatus | Should -Be Healthy
        $script:WslCalls | Should -Be @('status', 'ip', 'dns', 'hostname', 'curl')
    }

    It 'parses successful array-valued status output as installed' {
        $script:ArrayStatus = $true
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.Installed | Should -BeTrue
    }
    It 'extracts the version from its own status line' {
        $script:ArrayStatus = $true
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.Version | Should -Be '2'
    }
    It 'extracts the default distribution from its own status line' {
        $script:ArrayStatus = $true
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.DefaultDistro | Should -Be FixtureDistro
    }
    It 'extracts the kernel version from its own status line' {
        $script:ArrayStatus = $true
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.KernelVersion | Should -Be '6.6.1'
    }

    It 'refuses installed status from plausible stdout with nonzero native exit' {
        $script:FailedStage = 'status'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.Installed | Should -BeFalse
        @($result.Issues | Where-Object { $_.Category -eq 'WSL' -and $_.Severity -eq 'Critical' }).Count | Should -BeGreaterThan 0
    }
    It 'refuses a successful interface observation from failed ip stdout' {
        $script:FailedStage = 'ip'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        @($result.NetworkInterfaces | Where-Object { $_.Side -eq 'WSL' -and $_.Test -eq 'OK' }).Count | Should -Be 0
        @($result.Issues | Where-Object { $_.Category -eq 'Network' -and $_.Severity -eq 'Critical' }).Count | Should -BeGreaterThan 0
    }
    It 'records failed WSL DNS rather than trusting an address from failed nslookup' {
        $script:FailedStage = 'dns'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @('fixture.invalid') -TestPorts @() -SkipInternetTest
        @($result.DNSTests | Where-Object Side -EQ WSL).Count | Should -Be 1
        @($result.DNSTests | Where-Object Side -EQ WSL)[0].Success | Should -BeFalse
        @($result.DNSTests | Where-Object Side -EQ WSL)[0].IPAddress | Should -BeNullOrEmpty
    }
    It 'does not test a port against an address from failed hostname' {
        $script:FailedStage = 'hostname'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @(22) -SkipInternetTest
        Should -Invoke Test-PortConnectivity -Exactly -Times 0
        $result.PortTests | Should -HaveCount 1
        $result.PortTests[0].Success | Should -BeFalse
    }
    It 'refuses successful internet access from HTTP200 with failed curl exit' {
        $script:FailedStage = 'curl'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @()
        $result.InternetAccess.Success | Should -BeFalse
        $result.Summary.FailedTests | Should -Be 1
    }
    It 'detects the actual nonzero exit of a private native child with plausible stdout' {
        $script:NativeChild = $true
        $script:FailedStage = 'curl'
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @()
        $global:LASTEXITCODE | Should -Be 7
        $result.InternetAccess.Success | Should -BeFalse
    }
    It 'preserves actual zero exit and successful private native-child output' {
        $script:NativeChild = $true
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @()
        $result.InternetAccess.Success | Should -BeTrue
        $global:LASTEXITCODE | Should -Be 0
    }
    It 'does not dispatch curl when internet testing is skipped' {
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.InternetAccess | Should -BeNullOrEmpty
        $script:WslCalls | Should -Not -Contain curl
    }
    It 'does not dispatch WSL DNS for localhost' {
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @('localhost') -TestPorts @() -SkipInternetTest
        $script:WslCalls | Should -Not -Contain dns
        $result.DNSTests | Should -HaveCount 1
        $result.DNSTests[0].Side | Should -Be Windows
    }
    It 'handles a unavailable WSL executable without losing diagnostic output' {
        Mock wsl { throw 'Synthetic command unavailable.' }
        $result = Test-WSLConnectivity -Distribution FixtureDistro -DNSTargets @() -TestPorts @() -SkipInternetTest
        $result.WSLStatus.Installed | Should -BeFalse
        $result.Summary.OverallStatus | Should -Be Critical
    }
}
