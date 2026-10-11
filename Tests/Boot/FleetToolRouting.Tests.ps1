#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:HelperPath = Join-Path $script:RepoRoot 'Modules\PC-AI.LLM\Private\LLM-Helpers.ps1'
    . $script:HelperPath
    function New-FleetToolDefinition {
        param([string]$Name = 'pcai_get_machine_capabilities', [string]$Path = 'Tools/Get-PcaiMachineCapabilities.ps1', $Params = @{})
        return [pscustomobject]@{
            function = [pscustomobject]@{ name = $Name }
            pcai_mapping = [pscustomobject]@{ script = $Path; params = [pscustomobject]$Params }
        }
    }
}

Describe 'Fleet diagnostic tool routing' -Tag 'Boot', 'Unit' {
    BeforeEach {
        $script:FixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $toolsFolder = Join-Path $script:FixtureRoot 'Tools'
        New-Item -ItemType Directory -Path $toolsFolder | Out-Null
        $fixture = @'
param([string]$OutputPath, [switch]$ValidateIntelQsv, [switch]$ListCaptureDevices)
if ($OutputPath -or $ValidateIntelQsv -or $ListCaptureDevices) { throw 'Read-only fixture received dangerous defaults' }
[pscustomobject]@{
    invocationId = [guid]::NewGuid().ToString()
    defaultOnly = $true
    nested = @{ l1 = @{ l2 = @{ l3 = @{ l4 = @{ l5 = @{ l6 = @{ l7 = @{ l8 = @{ value = 'deep-payload'; unknownSensor = $null } } } } } } } } }
}
'@
        foreach ($name in @('Get-PcaiMachineCapabilities.ps1', 'Test-PcaiMediaToolchain.ps1')) {
            Set-Content -LiteralPath (Join-Path $toolsFolder $name) -Value $fixture
        }
    }

    It 'registers both fixed no-parameter mappings and matching router scenarios' {
        $definitions = (Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Config\pcai-tools.json') -Raw | ConvertFrom-Json).tools
        $scenarios = (Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Deploy\rust-functiongemma-train\examples\scenarios.json') -Raw | ConvertFrom-Json).scenarios
        foreach ($name in @('pcai_get_machine_capabilities', 'pcai_test_media_toolchain')) {
            $definition = @($definitions | Where-Object { $_.function.name -eq $name })
            $definition.Count | Should -Be 1
            @($definition[0].function.parameters.properties.PSObject.Properties).Count | Should -Be 0
            $definition[0].function.parameters.additionalProperties | Should -BeFalse
            @($definition[0].pcai_mapping.params.PSObject.Properties).Count | Should -Be 0
            @($scenarios | Where-Object tool_name -eq $name).Count | Should -Be 1
        }
    }

    It 'makes fresh benign default calls and preserves deeply structured JSON and unknown values' {
        foreach ($definition in @(
                (New-FleetToolDefinition)
                (New-FleetToolDefinition -Name pcai_test_media_toolchain -Path 'Tools/Test-PcaiMediaToolchain.ps1')
            )) {
            $first = Invoke-ToolByName -Name $definition.function.name -Args @{} -Tools @($definition) -ModuleRoot $script:FixtureRoot | ConvertFrom-Json
            $second = Invoke-ToolByName -Name $definition.function.name -Args ([pscustomobject]@{}) -Tools @($definition) -ModuleRoot $script:FixtureRoot | ConvertFrom-Json
            $first.defaultOnly | Should -BeTrue
            $first.invocationId | Should -Not -Be $second.invocationId
            $first.nested.l1.l2.l3.l4.l5.l6.l7.l8.value | Should -Be 'deep-payload'
            $first.nested.l1.l2.l3.l4.l5.l6.l7.l8.unknownSensor | Should -BeNullOrEmpty
        }
    }

    It 'suppresses inherited profile defaults that could enable output, capture or encode' {
        $definition = New-FleetToolDefinition -Name pcai_test_media_toolchain -Path 'Tools/Test-PcaiMediaToolchain.ps1'
        $PSDefaultParameterValues = @{ '*:OutputPath' = (Join-Path $TestDrive 'must-not-write.json'); '*:ValidateIntelQsv' = $true; '*:ListCaptureDevices' = $true }
        try {
            $result = Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $script:FixtureRoot | ConvertFrom-Json
            $result.defaultOnly | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $TestDrive 'must-not-write.json') | Should -BeFalse
        }
        finally { $PSDefaultParameterValues = @{} }
    }

    It 'accepts an empty dictionary mapping and rejects malformed configured parameters' {
        $definition = @{
            function = @{ name = 'pcai_get_machine_capabilities' }
            pcai_mapping = @{ script = 'Tools/Get-PcaiMachineCapabilities.ps1'; params = @{} }
        }
        $result = Invoke-ToolByName -Name pcai_get_machine_capabilities -Args ([ordered]@{}) -Tools @($definition) -ModuleRoot $script:FixtureRoot | ConvertFrom-Json
        $result.defaultOnly | Should -BeTrue
        $definition.pcai_mapping.params = 'OutputPath=out.json'
        Invoke-ToolByName -Name pcai_get_machine_capabilities -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*empty object'
    }

    It 'rejects caller arguments and configured output, capture, encode or install parameters' {
        $definition = New-FleetToolDefinition
        foreach ($arguments in @(@{ OutputPath = 'out.json' }, @{ ValidateIntelQsv = $true }, @{ ListCaptureDevices = $true }, @{ Install = $true }, '{}')) {
            Invoke-ToolByName -Name $definition.function.name -Args $arguments -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*accept no arguments'
        }
        foreach ($parameters in @(@{ OutputPath = 'out.json' }, @{ ValidateIntelQsv = $true }, @{ ListCaptureDevices = $true })) {
            $definition = New-FleetToolDefinition -Params $parameters
            Invoke-ToolByName -Name $definition.function.name -Args @{} -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*no configured parameters'
        }
    }

    It 'rejects path traversal, absolute paths, wrong scripts and nonallowlisted tool names' {
        foreach ($path in @('Tools/../Tools/Get-PcaiMachineCapabilities.ps1', '../Tools/Get-PcaiMachineCapabilities.ps1', 'C:\Windows\evil.ps1', 'Tools/Watch-CrashTelemetry.ps1', 'Tools/Test-PcaiMediaToolchain.ps1')) {
            $definition = New-FleetToolDefinition -Path $path
            Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*fixed read-only allowlist'
        }
        $definition = New-FleetToolDefinition -Name 'arbitrary_script'
        Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*not allowlisted'
    }

    It 'rejects ambiguous mappings, cmdlet fallbacks and missing approved scripts' {
        $definition = New-FleetToolDefinition
        Invoke-ToolByName -Name $definition.function.name -Tools @($definition, $definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*unique'
        $definition.pcai_mapping | Add-Member -NotePropertyName cmdlet -NotePropertyValue 'Install-Anything'
        Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Error executing tool .*only a fixed script path'
        $definition = New-FleetToolDefinition
        $missingRoot = Join-Path $TestDrive 'unavailable-repo'
        Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $missingRoot | Should -Match '^Error executing tool .*unavailable'
    }

    It 'rejects a junction that redirects the approved Tools folder outside the repository' -Skip:(-not $IsWindows) {
        $redirectedRepo = Join-Path $TestDrive 'redirected-repository'
        $outside = Join-Path $TestDrive 'outside-approved-root'
        New-Item -ItemType Directory -Path $redirectedRepo, $outside | Out-Null
        Set-Content -LiteralPath (Join-Path $outside 'Get-PcaiMachineCapabilities.ps1') -Value "throw 'External script must not execute'"
        New-Item -ItemType Junction -Path (Join-Path $redirectedRepo 'Tools') -Target $outside | Out-Null
        $definition = New-FleetToolDefinition
        Invoke-ToolByName -Name $definition.function.name -Tools @($definition) -ModuleRoot $redirectedRepo | Should -Match '^Error executing tool .*reparse points'
    }

    It 'loads configured definitions and the package root for legacy fleet chat' {
        $toolsPath = Join-Path $script:FixtureRoot 'pcai-tools.json'
        @{ tools = @((New-FleetToolDefinition)) } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $toolsPath
        $result = Invoke-PcaiFleetToolCall -Name pcai_get_machine_capabilities -ArgumentJson '{}' -ProjectRoot $script:FixtureRoot -ToolsPath $toolsPath | ConvertFrom-Json
        $result.defaultOnly | Should -BeTrue
        $result.nested.l1.l2.l3.l4.l5.l6.l7.l8.unknownSensor | Should -BeNullOrEmpty
    }

    It 'rejects legacy fleet chat arguments that could request writes or workloads' -ForEach @(
        @{ Arguments = '{"OutputPath":"C:\\unsafe.json"}' }
        @{ Arguments = '[]' }
        @{ Arguments = '"text"' }
        @{ Arguments = 'not JSON' }
    ) {
        Mock Invoke-ToolByName { throw 'Router must not execute' }
        { Invoke-PcaiFleetToolCall -Name pcai_get_machine_capabilities -ArgumentJson $Arguments -ProjectRoot $script:FixtureRoot -ToolsPath (Join-Path $script:FixtureRoot 'missing.json') } | Should -Throw
        Should -Invoke Invoke-ToolByName -Times 0 -Exactly
    }

    It 'preserves existing cmdlet parameter mapping and unhandled-tool behavior' {
        function Invoke-FleetFixtureCmdlet { param([string]$Value) [pscustomobject]@{ received = $Value; source = 'existing-cmdlet' } }
        $definition = [pscustomobject]@{
            function = [pscustomobject]@{ name = 'existing_cmdlet' }
            pcai_mapping = [pscustomobject]@{ module = $null; cmdlet = 'Invoke-FleetFixtureCmdlet'; args = @(); params = [pscustomobject]@{ Value = '$value' } }
        }
        $result = Invoke-ToolByName -Name existing_cmdlet -Args @{ value = 'preserved' } -Tools @($definition) -ModuleRoot $script:FixtureRoot | ConvertFrom-Json
        $result.received | Should -Be 'preserved'
        $result.source | Should -Be 'existing-cmdlet'
        Invoke-ToolByName -Name missing -Tools @() -ModuleRoot $script:FixtureRoot | Should -Match '^Unhandled tool:'
        Invoke-ToolByName -Name missing -Tools @($definition) -ModuleRoot $script:FixtureRoot | Should -Match '^Unhandled tool:'
    }
}
