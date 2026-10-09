@{
    Modules = @(
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
