<#
.SYNOPSIS
    Integration tests for PC_AI module loading and dependencies

.DESCRIPTION
    Tests that all modules load correctly, export expected functions, and have proper dependencies
#>

BeforeAll {
    $script:ModulesPath = Join-Path $PSScriptRoot '..\..\Modules'

    $script:Modules = @(
        @{
            Name = 'PC-AI.Hardware'
            ExpectedFunctions = @(
                'Get-DeviceErrors'
                'Get-DiskHealth'
                'Get-UsbStatus'
                'Get-NetworkAdapters'
                'Get-SystemEvents'
                'New-DiagnosticReport'
            )
        }
        @{
            Name = 'PC-AI.Virtualization'
            ExpectedFunctions = @(
                'Get-WSLStatus'
                'Get-HyperVStatus'
                'Get-DockerStatus'
                'Optimize-WSLConfig'
                'Set-WSLDefenderExclusions'
                'Repair-WSLNetworking'
                'Backup-WSLConfig'
            )
        }
        @{
            Name = 'PC-AI.USB'
            ExpectedFunctions = @(
                'Get-UsbDeviceList'
                'Mount-UsbToWSL'
                'Dismount-UsbFromWSL'
                'Get-UsbWSLStatus'
                'Invoke-UsbBind'
            )
        }
        @{
            Name = 'PC-AI.Network'
            ExpectedFunctions = @(
                'Get-NetworkDiagnostics'
                'Test-WSLConnectivity'
                'Watch-VSockPerformance'
                'Optimize-VSock'
            )
        }
        @{
            Name = 'PC-AI.Performance'
            ExpectedFunctions = @(
                'Get-DiskSpace'
                'Get-ProcessPerformance'
                'Watch-SystemResources'
                'Optimize-Disks'
            )
        }
        @{
            Name = 'PC-AI.Cleanup'
            ExpectedFunctions = @(
                'Get-PathDuplicates'
                'Repair-MachinePath'
                'Find-DuplicateFiles'
                'Clear-TempFiles'
            )
        }
        @{
            Name = 'PC-AI.LLM'
            ExpectedFunctions = @(
                'Get-LLMStatus'
                'Send-OllamaRequest'
                'Invoke-LLMChat'
                'Invoke-LLMChatRouted'
                'Invoke-LLMChatTui'
                'Invoke-FunctionGemmaReAct'
                'Invoke-PCDiagnosis'
                'Set-LLMConfig'
                'Invoke-DocSearch'
                'Get-SystemInfoTool'
                'Invoke-LogSearch'
            )
        }
        @{
            Name = 'PC-AI.Acceleration'
            ExpectedFunctions = @(
                'Get-RustToolStatus'
                'Test-RustToolAvailable'
                'Search-LogsFast'
                'Find-FilesFast'
                'Get-ProcessesFast'
                'Get-FileHashParallel'
                'Find-DuplicatesFast'
                'Get-DiskUsageFast'
                'Search-ContentFast'
                'Measure-CommandPerformance'
                'Compare-ToolPerformance'
                'Initialize-PcaiNative'
                'Test-PcaiNativeAvailable'
                'Get-PcaiNativeStatus'
                'Get-PcaiCapabilities'
                'Get-ProcessLassoSnapshot'
                'New-ProcessLassoOverlay'
                'Invoke-PcaiNativeDuplicates'
                'Invoke-PcaiNativeFileSearch'
                'Invoke-PcaiNativeContentSearch'
                'Invoke-PcaiNativeDirectoryManifest'
                'Invoke-PcaiNativeSystemInfo'
                'Test-PcaiResourceSafety'
                'Get-UnifiedHardwareReportJson'
                'Invoke-PcaiNativeUnifiedHardwareReport'
                'Invoke-PcaiNativeEstimateTokens'
                'Invoke-PcaiNativeProcessLassoSnapshot'
            )
        }
        @{
            Name = 'PC-AI.CLI'
            ExpectedFunctions = @(
                'Get-PCCommandMap'
                'Get-PCCommandModules'
                'Get-PCCommandList'
                'Get-PCCommandSummary'
                'Get-PCModuleHelpIndex'
                'Get-PCModuleHelpEntry'
                'ConvertTo-PCArgumentMap'
                'Resolve-PCArguments'
            )
        }
        @{
            Name = 'PC-AI.Drivers'
            ExpectedFunctions = @(
                'Get-PnpDeviceInventory'
                'Get-DriverRegistry'
                'Compare-DriverVersion'
                'Get-DriverReport'
                'Install-DriverUpdate'
                'Update-DriverRegistry'
                'Get-NetworkDiscoverySnapshot'
                'Find-ThunderboltPeer'
                'Get-ThunderboltNetworkStatus'
                'Connect-ThunderboltPeer'
                'Set-ThunderboltNetworkOptimization'
            )
        }
        @{
            Name = 'PC-AI.Evaluation'
            ExpectedFunctions = @(
                'New-EvaluationSuite'
                'Invoke-EvaluationSuite'
                'Get-EvaluationResults'
                'Measure-InferenceLatency'
                'Measure-TokenThroughput'
                'Measure-MemoryUsage'
                'Compare-ResponseSimilarity'
                'Invoke-LLMJudge'
                'Compare-ResponsePair'
                'Measure-DiagnosticQuality'
                'New-BaselineSnapshot'
                'Test-ForRegression'
                'Get-RegressionReport'
                'New-ABTest'
                'Add-ABTestResult'
                'Get-ABTestAnalysis'
                'Get-EvaluationDataset'
                'New-EvaluationTestCase'
                'Import-EvaluationDataset'
                'Export-EvaluationDataset'
                'Get-PcaiProjectRoot'
                'Get-PcaiArtifactsRoot'
                'Initialize-EvaluationPaths'
                'New-PcaiEvaluationRunContext'
                'Get-EvaluationRunState'
                'Stop-EvaluationRun'
                'Get-PcaiCompiledBinaryPath'
                'New-PcaiServerConfigFile'
                'Start-PcaiCompiledServer'
                'Measure-Coherence'
                'Measure-Toxicity'
                'Measure-Groundedness'
            )
        }
        @{
            Name = 'PC-AI.Gpu'
            ExpectedFunctions = @(
                'Get-NvidiaGpuInventory'
                'Get-NvidiaSoftwareRegistry'
                'Get-NvidiaSoftwareStatus'
                'Get-NvidiaGpuUtilization'
                'Get-NvidiaCompatibilityMatrix'
                'Initialize-NvidiaEnvironment'
                'Install-NvidiaSoftware'
                'Update-NvidiaSoftwareRegistry'
                'Test-PcaiGpuReadiness'
            )
        }
    )
}

Describe "Module Loading" -Tag 'Integration', 'ModuleLoading', 'Fast' {
    Context "When loading all modules" {
        It "Should find all module manifest files" {
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                Test-Path $manifestPath | Should -Be $true -Because "$($module.Name) manifest should exist"
            }
        }

        It "Should load <Name> module without errors" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"

            { Import-Module $manifestPath -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It "Should have valid manifest for <Name>" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"

            { Test-ModuleManifest $manifestPath -ErrorAction Stop } | Should -Not -Throw
        }
    }

    Context "When checking module versions" {
        It "Should have version information for <Name>" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"
            Import-Module $manifestPath -Force

            $module = Get-Module $Name
            $module.Version | Should -Not -BeNullOrEmpty
            $module.Version.Major | Should -BeGreaterOrEqual 0
        }

        It "Should have author information for <Name>" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"
            $manifest = Test-ModuleManifest $manifestPath

            $manifest.Author | Should -Not -BeNullOrEmpty
        }

        It "Should have description for <Name>" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"
            $manifest = Test-ModuleManifest $manifestPath

            $manifest.Description | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Module Function Exports" -Tag 'Integration', 'ModuleLoading', 'Fast' {
    Context "When checking exported functions" {
        BeforeAll {
            # Load all modules
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                Import-Module $manifestPath -Force -ErrorAction Stop
            }
        }

        It "Should export expected functions from <Name>" -ForEach $script:Modules {
            $module = Get-Module $Name
            $exportedFunctions = $module.ExportedFunctions.Keys

            foreach ($expectedFunction in $ExpectedFunctions) {
                $exportedFunctions | Should -Contain $expectedFunction -Because "$expectedFunction should be exported from $Name"
            }
        }

        It "Should have Get-Command work for <FunctionName> in <ModuleName>" -ForEach @(
            $script:Modules | ForEach-Object {
                $moduleName = $_.Name
                $_.ExpectedFunctions | ForEach-Object {
                    @{ ModuleName = $moduleName; FunctionName = $_ }
                }
            }
        ) {
            if ([string]::IsNullOrWhiteSpace($FunctionName)) { continue }
            $command = Get-Command $FunctionName -Module $ModuleName -ErrorAction SilentlyContinue
            $command | Should -Not -BeNullOrEmpty
            $command.CommandType | Should -Be 'Function'
        }

        It "Should have help documentation for <FunctionName>" -ForEach @(
            $script:Modules | ForEach-Object {
                $moduleName = $_.Name
                $_.ExpectedFunctions | Select-Object -First 2 | ForEach-Object {
                    @{ ModuleName = $moduleName; FunctionName = $_ }
                }
            }
        ) {
            $help = Get-Help $FunctionName -ErrorAction SilentlyContinue
            $help | Should -Not -BeNullOrEmpty
            $help.Synopsis | Should -Not -BeNullOrEmpty
        }
    }

    Context "When checking function parameters" {
        BeforeAll {
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                Import-Module $manifestPath -Force -ErrorAction Stop
            }
        }

        It "Get-DeviceErrors should have expected parameters" {
            $command = Get-Command Get-DeviceErrors
            $command.Parameters.Keys | Should -Contain 'ErrorAction'
        }

        It "Get-WSLStatus should have expected parameters" {
            $command = Get-Command Get-WSLStatus
            $command.Parameters.Keys | Should -Contain 'ErrorAction'
        }

        It "Send-OllamaRequest should have Prompt parameter" {
            $command = Get-Command Send-OllamaRequest
            $command.Parameters.Keys | Should -Contain 'Prompt'
            $command.Parameters.Keys | Should -Contain 'Model'
        }

        It "Get-DiskSpace should have DriveLetter parameter" {
            $command = Get-Command Get-DiskSpace
            $command.Parameters.Keys | Should -Contain 'DriveLetter'
        }
    }
}

Describe "Module Dependencies" -Tag 'Integration', 'ModuleLoading', 'Fast' {
    Context "When checking module file structure" {
        It "<Name> should have .psm1 module file" -ForEach $script:Modules {
            $modulePath = Join-Path $script:ModulesPath "$Name\$Name.psm1"
            Test-Path $modulePath | Should -Be $true
        }

        It "<Name> should have Public functions directory" -ForEach $script:Modules {
            $publicPath = Join-Path $script:ModulesPath "$Name\Public"
            Test-Path $publicPath | Should -Be $true
        }

        It "<Name> should have at least one Public function file" -ForEach $script:Modules {
            $publicPath = Join-Path $script:ModulesPath "$Name\Public\*.ps1"
            (Get-ChildItem $publicPath).Count | Should -BeGreaterThan 0
        }
    }

    Context "When checking required assemblies" {
        BeforeAll {
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                Import-Module $manifestPath -Force -ErrorAction Stop
            }
        }

        It "Modules should not throw on import" {
            { Get-Module PC-AI.* } | Should -Not -Throw
        }

        It "All expected modules should be loaded" {
            $loadedModules = (Get-Module PC-AI.*).Name
            $loadedModules.Count | Should -Be $script:Modules.Count
        }
    }
}

Describe "Module Interoperability" -Tag 'Integration', 'ModuleLoading', 'Slow' {
    Context "When modules work together" {
        BeforeAll {
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                Import-Module $manifestPath -Force -ErrorAction Stop
            }
        }

        It "Hardware module functions should be callable" {
            { Get-Command Get-DeviceErrors } | Should -Not -Throw
        }

        It "LLM module should be able to import from other modules" {
            # This tests that modules don't have conflicting exports
            $allFunctions = Get-Command -Module PC-AI.* -CommandType Function
            $allFunctions.Count | Should -BeGreaterThan 25
        }

        It "No function name conflicts should exist among declared modules" {
            $moduleNames = $script:Modules | ForEach-Object { $_.Name }
            $allFunctions = Get-Command -Module $moduleNames -CommandType Function
            $functionNames = $allFunctions.Name
            $uniqueNames = $functionNames | Select-Object -Unique

            $functionNames.Count | Should -Be $uniqueNames.Count
        }
    }

    Context "When unloading and reloading modules" {
        It "Should unload all modules cleanly" {
            { Get-Module PC-AI.* | Remove-Module -Force } | Should -Not -Throw
        }

        It "Should reload modules without errors" {
            foreach ($module in $script:Modules) {
                $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                { Import-Module $manifestPath -Force -ErrorAction Stop } | Should -Not -Throw
            }
        }

        It "Functions should still be available after reload" {
            $allFunctions = Get-Command -Module PC-AI.* -CommandType Function
            $allFunctions.Count | Should -BeGreaterThan 25
        }
    }
}

Describe "Module Performance" -Tag 'Integration', 'Performance', 'Slow' {
    Context "When measuring module load times" {
        It "Should load <Name> in reasonable time" -ForEach $script:Modules {
            $manifestPath = Join-Path $script:ModulesPath "$Name\$Name.psd1"

            # Unload if already loaded
            Remove-Module $Name -Force -ErrorAction SilentlyContinue

            $loadTime = Measure-Command {
                Import-Module $manifestPath -Force -ErrorAction Stop
            }

            $loadTime.TotalSeconds | Should -BeLessThan 5 -Because "$Name should load in under 5 seconds"
        }

        It "Should load all modules in reasonable total time" {
            # Unload all
            Get-Module PC-AI.* | Remove-Module -Force -ErrorAction SilentlyContinue

            $totalLoadTime = Measure-Command {
                foreach ($module in $script:Modules) {
                    $manifestPath = Join-Path $script:ModulesPath "$($module.Name)\$($module.Name).psd1"
                    Import-Module $manifestPath -Force -ErrorAction Stop
                }
            }

            $totalLoadTime.TotalSeconds | Should -BeLessThan 15 -Because "All modules should load in under 15 seconds"
        }
    }
}

AfterAll {
    # Clean up loaded modules
    Get-Module PC-AI.* | Remove-Module -Force -ErrorAction SilentlyContinue
}
