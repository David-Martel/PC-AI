#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ReleaseTool = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\New-PcaiPowerShellRelease.ps1'
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $script:ReleaseTool)
    function New-PcaiReleaseFixture {
        [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Creates disposable Pester TestDrive fixtures, not production deployments.')]
        [CmdletBinding()]
        param([string]$Root)
        $tools = Join-Path $Root 'Tools'
        $modules = Join-Path $Root 'Modules'
        $config = Join-Path $Root 'Config'
        New-Item -ItemType Directory -Path $tools, $modules, $config -Force | Out-Null
        $tool = $script:ReleaseTool
        Set-Content -LiteralPath (Join-Path $config 'fixture.json') -Value '{"fixture":true}'
        foreach ($name in @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md', 'AGENTS.md')) {
            Set-Content -LiteralPath (Join-Path $Root $name) -Value "Fixture $name"
        }
        $moduleNames = @('Acceleration', 'Cleanup', 'CLI', 'Evaluation', 'Hardware', 'LLM', 'Network', 'Performance', 'USB', 'Virtualization')
        foreach ($shortName in $moduleNames) {
            $name = "PC-AI.$shortName"
            $folder = Join-Path $modules $name
            New-Item -ItemType Directory -Path $folder | Out-Null
            $publicFunction = "Get-Fixture$shortName"
            $psm = @"
if (`$env:PCAI_FIXTURE_IMPORT_LOG) { [System.IO.File]::AppendAllText(`$env:PCAI_FIXTURE_IMPORT_LOG, '$name' + [environment]::NewLine) }
Write-Warning 'PC-AI fixture component loaded'
function $publicFunction { '$shortName' }
Export-ModuleMember -Function '$publicFunction'
"@
            Set-Content -LiteralPath (Join-Path $folder "$name.psm1") -Value $psm
            # Omitted alias/cmdlet manifest keys require static module resolution.
            $manifest = "@{ RootModule = '$name.psm1'; ModuleVersion = '1.0.0'; FunctionsToExport = @('$publicFunction'); VariablesToExport = @() }"
            Set-Content -LiteralPath (Join-Path $folder "$name.psd1") -Value $manifest
        }
        $common = Join-Path $modules 'PC-AI.Common'
        New-Item -ItemType Directory -Path $common | Out-Null
        Set-Content -LiteralPath (Join-Path $common 'PC-AI.Common.psm1') -Value @'
if ($env:PCAI_FIXTURE_IMPORT_LOG) { [System.IO.File]::AppendAllText($env:PCAI_FIXTURE_IMPORT_LOG, 'PC-AI.Common' + [environment]::NewLine) }
Write-Warning 'PC-AI fixture component loaded'
function Get-FixtureCommon { 'Common' }
Set-Alias -Name Invoke-FixtureCommon -Value Get-FixtureCommon
Export-ModuleMember -Function @('Get-FixtureCommon') -Alias @('Invoke-FixtureCommon')
'@
        Set-Content -LiteralPath (Join-Path $modules 'PcaiInference.psm1') -Value @'
if ($env:PCAI_FIXTURE_IMPORT_LOG) { [System.IO.File]::AppendAllText($env:PCAI_FIXTURE_IMPORT_LOG, 'PcaiInference' + [environment]::NewLine) }
Write-Warning 'PC-AI fixture component loaded'
function Get-FixtureInference { 'Inference' }
Export-ModuleMember -Function 'Get-FixtureInference'
'@
        Set-Content -LiteralPath (Join-Path $modules 'PcaiInference.psd1') -Value "@{ RootModule = 'PcaiInference.psm1'; ModuleVersion = '1.0.0'; FunctionsToExport = @('Get-FixtureInference'); CmdletsToExport = @(); AliasesToExport = @() }"
        Set-Content -LiteralPath (Join-Path $tools 'Get-BuildVersion.ps1') -Value "param([switch]`$Quiet); throw 'Version lookup must not run for DryRun or explicit version'"
        foreach ($name in @('Get-PcaiMachineCapabilities.ps1', 'Test-PcaiMediaToolchain.ps1')) {
            Set-Content -LiteralPath (Join-Path $tools $name) -Value @'
param([string]$OutputPath, [switch]$ValidateIntelQsv, [switch]$ListCaptureDevices)
if ($OutputPath -or $ValidateIntelQsv -or $ListCaptureDevices) { throw 'Fixture diagnostics require benign defaults' }
[pscustomobject]@{ fixture = $true; scriptRoot = $PSScriptRoot; defaultOnly = $true }
'@
        }
        return [pscustomobject]@{ Root = $Root; Tool = $tool; Modules = $modules }
    }

    function Invoke-PcaiReleaseChildProbe {
        param([string]$OutputRoot, [string]$Mode, [string]$WorkRoot)
        $probePath = Join-Path $WorkRoot 'Probe-Release.ps1'
        Set-Content -LiteralPath $probePath -Value @'
param([string]$OutputRoot, [string]$Mode, [string]$MarkerPath)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = $OutputRoot + [System.IO.Path]::PathSeparator + (Join-Path $PSHOME 'Modules')
$env:PCAI_FIXTURE_IMPORT_LOG = $MarkerPath
$PSModuleAutoLoadingPreference = 'All'
$warnings = @()
if ($Mode -eq 'Miss') {
    $command = Get-Command -Name 'PC-AI-Unrelated-Command-Definitely-Missing' -ErrorAction SilentlyContinue -WarningVariable warnings
    [pscustomobject]@{ Loaded = [bool](Get-Module -Name PC-AI); WarningCount = $warnings.Count; MarkerExists = Test-Path -LiteralPath $MarkerPath } | ConvertTo-Json -Compress
}
else {
    Import-Module (Join-Path $OutputRoot 'PC-AI\PC-AI.psd1') -Force -DisableNameChecking -WarningAction SilentlyContinue
    $releaseInfo = Get-PcaiReleaseInfo
    [pscustomobject]@{
        Loaded = [bool](Get-Module -Name PC-AI); ImportCount = @(Get-Content -LiteralPath $MarkerPath).Count
        Common = Get-FixtureCommon; Acceleration = Get-FixtureAcceleration; Inference = Get-FixtureInference
        Alias = Invoke-FixtureCommon; ReleaseInfo = $releaseInfo
        ExportedCommands = @((Get-Module -Name PC-AI).ExportedCommands.Keys)
    } | ConvertTo-Json -Depth 5 -Compress
}
'@
        $marker = Join-Path $WorkRoot "imports-$Mode.txt"
        $info = [System.Diagnostics.ProcessStartInfo]::new()
        $info.FileName = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $probePath, '-OutputRoot', $OutputRoot, '-Mode', $Mode, '-MarkerPath', $marker)) {
            $info.ArgumentList.Add($argument)
        }
        $process = [System.Diagnostics.Process]::Start($info)
        try {
            if (-not $process.WaitForExit(20000)) { $process.Kill(); throw 'Release child probe timed out.' }
            $stdout = $process.StandardOutput.ReadToEnd()
            $stderr = $process.StandardError.ReadToEnd()
            if ($process.ExitCode -ne 0) { throw "Release child probe failed: $stderr" }
            # Explicit imports deliberately produce fixture dependency warnings;
            # the compressed final line is the structured child probe result.
            return ($stdout.Trim() -split '\r?\n' | Select-Object -Last 1) | ConvertFrom-Json
        }
        finally { $process.Dispose() }
    }
}

Describe 'Unified PowerShell release generation' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $script:Fixture = New-PcaiReleaseFixture -Root (Join-Path $TestDrive ([guid]::NewGuid().ToString('N')))
        $script:OutputRoot = Join-Path $script:Fixture.Root 'ReleaseModules'
    }

    It 'supports dry-run, WhatIf and help without reads of build metadata or writes' {
        foreach ($parameters in @(@{ DryRun = $true; Clean = $true }, @{ WhatIf = $true; Clean = $true })) {
            $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot @parameters
            $result.Status | Should -Be 'DryRun'
            $result.WritesPerformed | Should -BeFalse
            Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
        }
        foreach ($alias in @('h', '-help')) {
            $parameters = @{ $alias = $true }
            (& $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot @parameters | Out-String) | Should -Match 'explicit component exports'
            Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
        }
        (& $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot --help | Out-String) |
            Should -Match 'explicit component exports'
        Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
    }

    It 'generates explicit exports with nested components and both legacy root helpers' {
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.2.3
        $manifest = Import-PowerShellDataFile -LiteralPath $result.ModuleManifest
        $manifest.ModuleVersion | Should -Be '1.2.3'
        $manifest.NestedModules.Count | Should -Be 12
        $manifest.FunctionsToExport.Count | Should -Be 14
        $manifest.FunctionsToExport | Should -Contain 'Get-PcaiReleaseInfo'
        $manifest.FunctionsToExport | Should -Contain 'Import-PcaiComponentModules'
        $manifest.FunctionsToExport | Should -Contain 'Get-FixtureCommon'
        $manifest.AliasesToExport | Should -Contain 'Invoke-FixtureCommon'
        $manifest.CmdletsToExport.Count | Should -Be 0
        foreach ($name in @($manifest.FunctionsToExport + $manifest.CmdletsToExport + $manifest.AliasesToExport)) {
            [System.Management.Automation.WildcardPattern]::ContainsWildcardCharacters($name) | Should -BeFalse
        }
        $rootModule = Get-Content -LiteralPath (Join-Path $result.ModuleRoot 'PC-AI.psm1') -Raw
        $rootModule | Should -Not -Match '(?m)^Import-PcaiComponentModules\s*$'
        $rootModule | Should -Not -Match 'Export-ModuleMember -Function \*'
        Test-Path -LiteralPath (Join-Path $result.ModuleRoot 'Config\fixture.json') | Should -BeTrue
        @(Get-ChildItem -LiteralPath (Join-Path $result.ModuleRoot 'Tools') -File).Count | Should -Be 2
        Test-Path -LiteralPath (Join-Path $result.ModuleRoot 'Tools\Get-BuildVersion.ps1') | Should -BeFalse
        foreach ($name in @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md')) {
            Get-Content -LiteralPath (Join-Path $result.ModuleRoot $name) | Should -Be "Fixture $name"
        }
        Test-Path -LiteralPath (Join-Path $result.ModuleRoot 'AGENTS.md') | Should -BeFalse
        Get-Content -LiteralPath (Join-Path $result.ModuleRoot 'README.md') -Raw | Should -Match '# PC-AI PowerShell Release Module'
        (Get-Content -LiteralPath (Join-Path $result.ModuleRoot 'RELEASE_INFO.json') -Raw | ConvertFrom-Json).ModuleVersion | Should -Be '1.2.3'
    }

    It 'packages both real diagnostic scripts with their configured release routes' {
        $result = & $script:ReleaseTool -SourceRoot $script:RepoRoot -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        $definitions = (Get-Content -LiteralPath (Join-Path $result.ModuleRoot 'Config\pcai-tools.json') -Raw | ConvertFrom-Json).tools
        foreach ($name in @('pcai_get_machine_capabilities', 'pcai_test_media_toolchain')) {
            $definition = @($definitions | Where-Object { $_.function.name -eq $name })
            $definition.Count | Should -Be 1
            $source = Join-Path $script:RepoRoot $definition[0].pcai_mapping.script
            $packaged = Join-Path $result.ModuleRoot $definition[0].pcai_mapping.script
            Test-Path -LiteralPath $packaged -PathType Leaf | Should -BeTrue
            (Get-FileHash -LiteralPath $packaged).Hash | Should -Be (Get-FileHash -LiteralPath $source).Hash
        }
        @(Get-ChildItem -LiteralPath (Join-Path $result.ModuleRoot 'Tools') -File).Count | Should -Be 2
        foreach ($name in @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md')) {
            (Get-FileHash -LiteralPath (Join-Path $result.ModuleRoot $name)).Hash |
                Should -Be (Get-FileHash -LiteralPath (Join-Path $script:RepoRoot $name)).Hash
        }
    }

    It 'executes both configured routes from a generated fixture release without native effects' {
        Copy-Item -LiteralPath (Join-Path $script:RepoRoot 'Config\pcai-tools.json') -Destination (Join-Path $script:Fixture.Root 'Config\pcai-tools.json')
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        . (Join-Path $script:RepoRoot 'Modules\PC-AI.LLM\Private\LLM-Helpers.ps1')
        $toolsPath = Join-Path $result.ModuleRoot 'Config\pcai-tools.json'
        foreach ($name in @('pcai_get_machine_capabilities', 'pcai_test_media_toolchain')) {
            $payload = Invoke-PcaiFleetToolCall -Name $name -ArgumentJson '{}' -ProjectRoot $result.ModuleRoot -ToolsPath $toolsPath | ConvertFrom-Json
            $payload.fixture | Should -BeTrue
            $payload.defaultOnly | Should -BeTrue
            $payload.scriptRoot | Should -Be (Join-Path $result.ModuleRoot 'Tools')
        }
    }

    It 'allows optional prompt documents to be absent without copying unrelated repository guidance' {
        foreach ($name in @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md')) {
            Remove-Item -LiteralPath (Join-Path $script:Fixture.Root $name)
        }
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        foreach ($name in @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md', 'AGENTS.md')) {
            Test-Path -LiteralPath (Join-Path $result.ModuleRoot $name) | Should -BeFalse
        }
        $info = Get-Content -LiteralPath (Join-Path $result.ModuleRoot 'RELEASE_INFO.json') -Raw | ConvertFrom-Json
        @($info.PromptDocuments).Count | Should -Be 0
        $info.DiagnosticScripts.Count | Should -Be 2
    }

    It 'rejects a missing or invalid diagnostic script before cleaning an existing release' {
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        $marker = Join-Path $result.ModuleRoot 'preserved-wip.txt'
        Set-Content -LiteralPath $marker -Value 'preserve old release'
        $scriptPath = Join-Path $script:Fixture.Root 'Tools\Test-PcaiMediaToolchain.ps1'
        Remove-Item -LiteralPath $scriptPath
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*Required release diagnostic script*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve old release'
        Set-Content -LiteralPath $scriptPath -Value 'function Broken {'
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*Cannot parse release diagnostic script*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve old release'
    }

    It 'does not autoload or warn on an unrelated Get-Command miss in a clean process' {
        & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 | Out-Null
        $probe = Invoke-PcaiReleaseChildProbe -OutputRoot $script:OutputRoot -Mode Miss -WorkRoot $script:Fixture.Root
        $probe.Loaded | Should -BeFalse
        $probe.WarningCount | Should -Be 0
        $probe.MarkerExists | Should -BeFalse
    }

    It 'loads each nested component once and preserves public functions and aliases' {
        & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 | Out-Null
        $probe = Invoke-PcaiReleaseChildProbe -OutputRoot $script:OutputRoot -Mode Import -WorkRoot $script:Fixture.Root
        $probe.Loaded | Should -BeTrue
        $probe.ImportCount | Should -Be 12
        $probe.Common | Should -Be 'Common'
        $probe.Acceleration | Should -Be 'Acceleration'
        $probe.Inference | Should -Be 'Inference'
        $probe.Alias | Should -Be 'Common'
        $probe.ReleaseInfo.Components.Count | Should -Be 12
        $probe.ExportedCommands | Should -Contain 'Get-PcaiReleaseInfo'
        $probe.ExportedCommands | Should -Contain 'Import-PcaiComponentModules'
    }

    It 'fails before writes for wildcard manifest or dynamic script-only exports' {
        $manifestPath = Join-Path $script:Fixture.Modules 'PC-AI.Acceleration\PC-AI.Acceleration.psd1'
        Set-Content -LiteralPath $manifestPath -Value "@{ RootModule = 'PC-AI.Acceleration.psm1'; FunctionsToExport = '*' }"
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 } | Should -Throw '*wildcard*'
        Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
        Set-Content -LiteralPath $manifestPath -Value "@{ RootModule = 'PC-AI.Acceleration.psm1'; FunctionsToExport = @('Get-FixtureAcceleration') }"
        Set-Content -LiteralPath (Join-Path $script:Fixture.Modules 'PC-AI.Common\PC-AI.Common.psm1') -Value '$exports = @(''Get-FixtureCommon''); Export-ModuleMember -Function $exports'
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 } | Should -Throw '*dynamic*'
        Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
    }

    It 'cleans only the exact generated module directory and preserves neighboring evidence' {
        & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 | Out-Null
        $neighbor = Join-Path $script:OutputRoot 'preserved.txt'
        Set-Content -LiteralPath $neighbor -Value 'preserved'
        & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.1 -Clean | Out-Null
        Get-Content -LiteralPath $neighbor | Should -Be 'preserved'
        (Import-PowerShellDataFile -LiteralPath (Join-Path $script:OutputRoot 'PC-AI\PC-AI.psd1')).ModuleVersion | Should -Be '1.0.1'
    }

    It 'rejects Windows reserved output directories before writes' -Skip:(-not $IsWindows) {
        foreach ($leaf in @('AUX', 'NUL.txt', '$null')) {
            $path = Join-Path $TestDrive $leaf
            { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $path -ModuleVersion 1.0.0 } | Should -Throw '*reserved*'
        }
    }

    It 'refuses Clean when the generated target is the source checkout' {
        $source = New-PcaiReleaseFixture -Root (Join-Path $TestDrive 'PC-AI')
        $marker = Join-Path $source.Root 'uncommitted-work.txt'
        Set-Content -LiteralPath $marker -Value 'preserve source WIP'
        { & $source.Tool -SourceRoot $source.Root -OutputRoot $TestDrive -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*overlaps the source checkout*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve source WIP'
        Test-Path -LiteralPath (Join-Path $source.Modules 'PC-AI.Hardware\PC-AI.Hardware.psm1') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $source.Root 'Config\fixture.json') | Should -BeTrue
    }

    It 'refuses a generated target containing the source checkout' {
        $output = Join-Path $TestDrive 'ancestor-output'
        $source = New-PcaiReleaseFixture -Root (Join-Path $output 'PC-AI\checkout')
        $marker = Join-Path $source.Root 'uncommitted-work.txt'
        Set-Content -LiteralPath $marker -Value 'preserve descendant source'
        { & $source.Tool -SourceRoot $source.Root -OutputRoot $output -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*overlaps the source checkout*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve descendant source'
        Test-Path -LiteralPath (Join-Path $source.Root 'Config\fixture.json') | Should -BeTrue
    }

    It 'refuses output inside protected source <Payload> before creating directories' -ForEach @(
        @{ Payload = 'Modules' }, @{ Payload = 'Config' }, @{ Payload = 'Tools' }
    ) {
        $protectedOutput = Join-Path $script:Fixture.Root $Payload
        $marker = Join-Path $protectedOutput 'uncommitted-work.txt'
        Set-Content -LiteralPath $marker -Value 'preserve source payload'
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $protectedOutput -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*overlaps a source payload*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve source payload'
        Test-Path -LiteralPath (Join-Path $protectedOutput 'PC-AI') | Should -BeFalse
    }

    It 'rejects an existing <Placement> output junction and preserves its target' -Skip:(-not $IsWindows) -ForEach @(
        @{ Placement = 'root' }, @{ Placement = 'ancestor' },
        @{ Placement = 'target' }, @{ Placement = 'component' }, @{ Placement = 'config' }, @{ Placement = 'tools' }
    ) {
        $external = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $external | Out-Null
        $marker = Join-Path $external 'local-config-and-wip.txt'
        Set-Content -LiteralPath $marker -Value 'preserve external data'
        $output = $script:OutputRoot
        switch ($Placement) {
            'root' { $link = $output }
            'ancestor' {
                $link = Join-Path $script:Fixture.Root 'redirected-parent'
                $output = Join-Path $link 'future-output'
            }
            'target' {
                New-Item -ItemType Directory -Path $output | Out-Null
                $link = Join-Path $output 'PC-AI'
            }
            'component' {
                $modules = Join-Path $output 'PC-AI\Modules'
                New-Item -ItemType Directory -Path $modules -Force | Out-Null
                $link = Join-Path $modules 'PC-AI.Hardware'
            }
            'config' {
                New-Item -ItemType Directory -Path (Join-Path $output 'PC-AI') -Force | Out-Null
                $link = Join-Path $output 'PC-AI\Config'
            }
            'tools' {
                New-Item -ItemType Directory -Path (Join-Path $output 'PC-AI') -Force | Out-Null
                $link = Join-Path $output 'PC-AI\Tools'
            }
        }
        New-Item -ItemType Junction -Path $link -Target $external | Out-Null
        foreach ($clean in @($false, $true)) {
            { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $output -ModuleVersion 1.0.0 -Clean:$clean } |
                Should -Throw '*reparse point*'
            Get-Content -LiteralPath $marker | Should -Be 'preserve external data'
            (Get-Item -LiteralPath $link).LinkType | Should -Be 'Junction'
            @(Get-ChildItem -LiteralPath $external -Force).Count | Should -Be 1
        }
    }

    It 'rejects a source payload junction instead of copying aliased data' -Skip:(-not $IsWindows) {
        $external = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $external | Out-Null
        $marker = Join-Path $external 'uncommitted-work.txt'
        Set-Content -LiteralPath $marker -Value 'preserve aliased source'
        $link = Join-Path $script:Fixture.Modules 'PC-AI.Hardware\linked-private'
        New-Item -ItemType Junction -Path $link -Target $external | Out-Null
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*reparse point*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve aliased source'
        Test-Path -LiteralPath $script:OutputRoot | Should -BeFalse
    }

    It 'rejects an allowlisted source script symlink before changing a release' -Skip:(-not $IsWindows) {
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        $marker = Join-Path $result.ModuleRoot 'preserved-wip.txt'
        Set-Content -LiteralPath $marker -Value 'preserve existing release'
        $external = Join-Path $TestDrive 'external-diagnostic.ps1'
        Set-Content -LiteralPath $external -Value "throw 'External script must not execute'"
        $link = Join-Path $script:Fixture.Root 'Tools\Get-PcaiMachineCapabilities.ps1'
        Remove-Item -LiteralPath $link
        New-Item -ItemType SymbolicLink -Path $link -Target $external | Out-Null
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*reparse point*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve existing release'
        Get-Content -LiteralPath $external | Should -Be "throw 'External script must not execute'"
    }

    It 'rejects a source prompt document symlink before cleaning existing release data' -Skip:(-not $IsWindows) {
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0
        $marker = Join-Path $result.ModuleRoot 'preserved-wip.txt'
        Set-Content -LiteralPath $marker -Value 'preserve release data'
        $external = Join-Path $TestDrive 'external-chat.md'
        Set-Content -LiteralPath $external -Value 'preserve external prompt'
        $link = Join-Path $script:Fixture.Root 'CHAT.md'
        Remove-Item -LiteralPath $link
        New-Item -ItemType SymbolicLink -Path $link -Target $external | Out-Null
        { & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $script:OutputRoot -ModuleVersion 1.0.0 -Clean } |
            Should -Throw '*prompt document contains a reparse point*'
        Get-Content -LiteralPath $marker | Should -Be 'preserve release data'
        Get-Content -LiteralPath $external | Should -Be 'preserve external prompt'
    }

    It 'permits the normal release directory below the checkout' {
        $output = Join-Path $script:Fixture.Root 'Release\PowerShell'
        $result = & $script:Fixture.Tool -SourceRoot $script:Fixture.Root -OutputRoot $output -ModuleVersion 1.0.0
        Test-Path -LiteralPath $result.ModuleManifest | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $script:Fixture.Root 'Config\fixture.json') | Should -BeTrue
    }
}
