#Requires -Version 5.1
BeforeAll {
    $script:HealthSource = if ($env:PCAI_HEALTH_CONTRACT_SOURCE) { $env:PCAI_HEALTH_CONTRACT_SOURCE }
        else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Virtualization/Public/Get-PcaiServiceHealth.ps1' }
    . $script:HealthSource
    function Get-PcaiRuntimeDefaults { param($ConfigPath) throw 'Live runtime configuration is forbidden.' }
    function wsl { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Live WSL is forbidden.' }
    function docker { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Live Docker is forbidden.' }
    function Get-CimInstance { [CmdletBinding()] param($ClassName) throw 'Live CIM is forbidden.' }

    function Invoke-HealthFixtureNative {
        param([string]$Stage)
        $exit = if ($Stage -eq $script:FailedStage) { 7 } else { 0 }
        $output = switch ($Stage) {
            'wsl-list' { $script:DistroLines }
            'bridge-count' { $script:BridgeOutput }
            'docker-id' { 'synthetic-engine' }
            'docker-runtime' { '{"nvidia":{}}' }
            default { throw 'Unknown synthetic native stage.' }
        }
        $script:NativeCalls += [pscustomobject]@{ Stage = $Stage; Exit = $exit }
        if ($script:ActualChild) {
            if (-not ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT)) { throw 'Actual cmd fixture requires Windows.' }
            $shim = Join-Path $TestDrive 'service-health-native.cmd'
            [IO.File]::WriteAllLines($shim, (@('@echo off') + @($output | ForEach-Object { "echo $_" }) + @("exit /b $exit")), [Text.UTF8Encoding]::new($false))
            & $env:ComSpec /d /c $shim
            $global:LASTEXITCODE = $LASTEXITCODE
        } else {
            $global:LASTEXITCODE = $exit
            $output
        }
    }
}

Describe 'Service health native status and exact distribution contracts' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach {
        $script:FailedStage = ''
        $script:ActualChild = $false
        $script:DistroLines = @('  NAME STATE VERSION', '* FixtureDistro Running 2')
        $script:BridgeOutput = '3'
        $script:NativeCalls = @()
        $global:LASTEXITCODE = 0
        Mock Get-PcaiRuntimeDefaults {
            [pscustomobject]@{
                PcaiInferenceUrl = 'http://fixture-inference.invalid:9080'; FunctionGemmaUrl = 'http://fixture-router.invalid:9081'
                OllamaBaseUrl = 'http://fixture-legacy.invalid:9082'; vLLMBaseUrl = 'http://fixture-legacy.invalid:9083'
                NativeDllSearchPaths = @()
            }
        }
        Mock Invoke-RestMethod { throw 'Live or synthetic HTTP provider unavailable.' }
        Mock Test-Path { $false }
        Mock Get-Process { [pscustomobject]@{ Id = -1; ProcessName = 'Synthetic Docker presence' } }
        Mock Get-CimInstance { [pscustomobject]@{ Name = 'Fixture GPU'; DriverVersion = 'fixture'; Status = 'OK'; PNPDeviceID = 'SYNTHETIC' } }
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'nvidia-smi' }
        Mock wsl {
            if ($Tokens -contains '-l') { Invoke-HealthFixtureNative -Stage 'wsl-list' }
            elseif ($Tokens -contains 'pgrep') { Invoke-HealthFixtureNative -Stage 'bridge-count' }
            else { throw "Unexpected synthetic WSL arguments: $Tokens" }
        }
        Mock docker {
            if ($Tokens[-1] -like '*Runtimes*') { Invoke-HealthFixtureNative -Stage 'docker-runtime' }
            else { Invoke-HealthFixtureNative -Stage 'docker-id' }
        }
    }

    It 'preserves successful WSL Docker GPU-runtime and bridge observations' {
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.WSL.Running | Should -BeTrue
        $result.WSL.Status | Should -Be OK
        $result.Docker.Status | Should -Be OK
        $result.Gpu.NvidiaRuntime | Should -BeTrue
        $result.Bridges.Count | Should -Be 3
        $result.Bridges.Status | Should -Be OK
        $result.OverallStatus | Should -Be Degraded
    }
    It 'does not report Running or dispatch bridge probes after failed WSL enumeration' {
        $script:FailedStage = 'wsl-list'
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.WSL.Running | Should -BeFalse
        $result.WSL.Status | Should -Not -Be OK
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains 'pgrep' }
    }
    It 'does not report three healthy bridges from failed pgrep output' {
        $script:FailedStage = 'bridge-count'
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Bridges.Count | Should -Be 0
        $result.Bridges.Status | Should -Be NotChecked
    }
    It 'does not report an NVIDIA runtime from failed docker-info JSON' {
        $script:FailedStage = 'docker-runtime'
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Gpu.NvidiaRuntime | Should -BeFalse
        $result.Gpu.Devices[0].Name | Should -Be 'Fixture GPU'
    }
    It 'preserves a failed Docker daemon status despite plausible engine text' {
        $script:FailedStage = 'docker-id'
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Docker.Status | Should -Be DaemonNotResponding
    }
    It 'does not select a running sibling instead of the stopped requested distribution' {
        $script:DistroLines = @('* FixtureDistro-Backup Running 2', '  FixtureDistro Stopped 2')
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.WSL.Running | Should -BeFalse
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains 'pgrep' }
    }
    It 'treats distribution-name square brackets literally' {
        $script:DistroLines = @('* Fixtureo Running 2', '  Fixture[one] Stopped 2')
        (Get-PcaiServiceHealth -Distribution 'Fixture[one]').WSL.Running | Should -BeFalse
    }
    It 'treats distribution-name dots literally' {
        $script:DistroLines = @('* Fixturexv2 Running 2', '  Fixture.v2 Stopped 2')
        (Get-PcaiServiceHealth -Distribution 'Fixture.v2').WSL.Running | Should -BeFalse
    }
    It 'preserves an exactly selected running name containing spaces' {
        $script:DistroLines = @('  OtherDistro Stopped 2', '* Fixture Distro Running 2')
        (Get-PcaiServiceHealth -Distribution 'Fixture Distro').WSL.Running | Should -BeTrue
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains 'pgrep' -and $Tokens -contains 'Fixture Distro' }
    }
    It 'does not accept a partial or negated running-state token' -ForEach @(
        @{ State = 'NotRunning' }, @{ State = 'RunningSlow' }, @{ State = 'Stopped' }
    ) {
        $script:DistroLines = @("* FixtureDistro $State 2")
        (Get-PcaiServiceHealth -Distribution FixtureDistro).WSL.Running | Should -BeFalse
    }
    It 'does not report an absent requested distribution as Running' {
        $script:DistroLines = @('* AnotherDistro Running 2')
        (Get-PcaiServiceHealth -Distribution FixtureDistro).WSL.Running | Should -BeFalse
    }
    It 'normalizes WSL native NUL characters before selecting the exact row' {
        $script:DistroLines = @([string]::Join("`0", '* FixtureDistro Running 2'.ToCharArray()))
        (Get-PcaiServiceHealth -Distribution FixtureDistro).WSL.Running | Should -BeTrue
    }
    It 'accepts the normal no-bridge pgrep exit one with a zero count' {
        Mock wsl {
            $global:LASTEXITCODE = if ($Tokens -contains 'pgrep') { 1 } else { 0 }
            if ($Tokens -contains 'pgrep') { '0' } else { '* FixtureDistro Running 2' }
        }
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Bridges.Count | Should -Be 0
        $result.Bridges.Status | Should -Be None
    }
    It 'refuses malformed or negative bridge counts rather than inventing a healthy count' -ForEach @(
        @{ CountOutput = 'invalid' }, @{ CountOutput = '-3' }, @{ CountOutput = '3.5' }
    ) {
        $script:BridgeOutput = $CountOutput
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Bridges.Count | Should -Be 0
        $result.Bridges.Status | Should -Be NotChecked
    }
    It 'refuses actual failed native child output for each admission boundary' -Skip:(-not ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT)) -ForEach @(
        @{ FailedStage = 'wsl-list'; Field = 'WSL' },
        @{ FailedStage = 'bridge-count'; Field = 'Bridges' },
        @{ FailedStage = 'docker-runtime'; Field = 'Gpu' }
    ) {
        $script:ActualChild = $true
        $script:FailedStage = $FailedStage
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        @($script:NativeCalls | Where-Object { $_.Stage -eq $FailedStage -and $_.Exit -eq 7 }).Count | Should -Be 1
        switch ($Field) {
            WSL { $result.WSL.Running | Should -BeFalse }
            Bridges { $result.Bridges.Count | Should -Be 0; $result.Bridges.Status | Should -Be NotChecked }
            Gpu { $result.Gpu.NvidiaRuntime | Should -BeFalse }
        }
    }
    It 'preserves actual successful private native child observations' -Skip:(-not ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT)) {
        $script:ActualChild = $true
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.WSL.Running | Should -BeTrue
        $result.Bridges.Count | Should -Be 3
        $result.Gpu.NvidiaRuntime | Should -BeTrue
        @($script:NativeCalls | Where-Object Exit -NE 0).Count | Should -Be 0
    }
}
