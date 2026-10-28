BeforeAll {
# Frozen public contract from observed PC-AI 0.1.0 manifest SHA256
# 5962D52ACAB8A63334CB40E8AF2AB85BBD6EF3403F427FA777611578647551A7.
# This expectation is independent of the new manifest at test runtime.
$script:LegacyAggregateNames = @(
        'Add-ABTestResult',
        'Backup-WSLConfig',
        'Clear-PcaiSharedCache',
        'Clear-TempFiles',
        'Close-PcaiInference',
        'Compare-ResponsePair',
        'Compare-ResponseSimilarity',
        'Compare-ToolPerformance',
        'ConvertTo-PCArgumentMap',
        'Dismount-UsbFromWSL',
        'Enable-WSLSystemd',
        'Export-EvaluationDataset',
        'Find-DuplicateFiles',
        'Find-DuplicatesFast',
        'Find-FilesFast',
        'Get-ABTestAnalysis',
        'Get-DeviceErrors',
        'Get-DiskHealth',
        'Get-DiskSpace',
        'Get-DiskUsageFast',
        'Get-DockerStatus',
        'Get-EvaluationDataset',
        'Get-EvaluationResults',
        'Get-EvaluationRunState',
        'Get-FileHashParallel',
        'Get-HVSockProxyStatus',
        'Get-HyperVStatus',
        'Get-LLMStatus',
        'Get-NetworkAdapters',
        'Get-NetworkDiagnostics',
        'Get-PathDuplicates',
        'Get-PcaiAccelerationProbe',
        'Get-PcaiArtifactsRoot',
        'Get-PcaiCapabilities',
        'Get-PcaiCompiledBinaryPath',
        'Get-PcaiDependencyStamp',
        'Get-PcaiDirectCoreProbe',
        'Get-PcaiDirectTokenEstimate',
        'Get-PcaiDiskUsage',
        'Get-PcaiExternalCacheStatus',
        'Get-PcaiInferenceStatus',
        'Get-PcaiMemoryPressure',
        'Get-PcaiMemoryStat',
        'Get-PcaiNativeStatus',
        'Get-PcaiNativeUsbDiagnostics',
        'Get-PcaiOptimizationPlan',
        'Get-PcaiProcessCategories',
        'Get-PcaiProjectRoot',
        'Get-PcaiReleaseInfo',
        'Get-PcaiRuntimeConfig',
        'Get-PcaiServiceHealth',
        'Get-PcaiSharedCacheEntry',
        'Get-PcaiTopProcess',
        'Get-PCCommandList',
        'Get-PCCommandMap',
        'Get-PCCommandModules',
        'Get-PCCommandSummary',
        'Get-PCModuleHelpEntry',
        'Get-PCModuleHelpIndex',
        'Get-ProcessesFast',
        'Get-ProcessPerformance',
        'Get-RegressionReport',
        'Get-RustToolStatus',
        'Get-ScriptMetadata',
        'Get-SystemEvents',
        'Get-SystemInfoTool',
        'Get-UnifiedHardwareReportJson',
        'Get-UsbDeviceList',
        'Get-UsbStatus',
        'Get-UsbWSLStatus',
        'Get-WSLEnvironmentHealth',
        'Get-WSLStatus',
        'Get-WSLVsockBridgeStatus',
        'Import-EvaluationDataset',
        'Import-PcaiAccelerationStack',
        'Import-PcaiComponentModules',
        'Import-PcaiModel',
        'Initialize-EvaluationPaths',
        'Initialize-PcaiInference',
        'Initialize-PcaiNative',
        'Install-HVSockProxy',
        'Install-WSLVsockBridge',
        'Invoke-DocSearch',
        'Invoke-EvaluationSuite',
        'Invoke-FunctionGemmaDataset',
        'Invoke-FunctionGemmaEval',
        'Invoke-FunctionGemmaReAct',
        'Invoke-FunctionGemmaTests',
        'Invoke-FunctionGemmaTokenCache',
        'Invoke-FunctionGemmaTrain',
        'Invoke-LLMChat',
        'Invoke-LLMChatRouted',
        'Invoke-LLMChatTui',
        'Invoke-LLMJudge',
        'Invoke-LogSearch',
        'Invoke-NativeSearch',
        'Invoke-PcaiDoctor',
        'Invoke-PcaiGenerate',
        'Invoke-PcaiNativeContentSearch',
        'Invoke-PcaiNativeDirectoryManifest',
        'Invoke-PcaiNativeDuplicates',
        'Invoke-PcaiNativeEstimateTokens',
        'Invoke-PcaiNativeFileSearch',
        'Invoke-PcaiNativeSystemInfo',
        'Invoke-PcaiNativeUnifiedHardwareReport',
        'Invoke-PcaiServiceHost',
        'Invoke-PCDiagnosis',
        'Invoke-SmartDiagnosis',
        'Invoke-UsbBind',
        'Invoke-WSLDockerHealthCheck',
        'Invoke-WSLNetworkToolkit',
        'Measure-Coherence',
        'Measure-CommandPerformance',
        'Measure-DiagnosticQuality',
        'Measure-Groundedness',
        'Measure-InferenceLatency',
        'Measure-MemoryUsage',
        'Measure-TokenThroughput',
        'Measure-Toxicity',
        'Mount-UsbToWSL',
        'New-ABTest',
        'New-BaselineSnapshot',
        'New-DiagnosticReport',
        'New-EvaluationSuite',
        'New-EvaluationTestCase',
        'New-PcaiEvaluationRunContext',
        'New-PcaiServerConfigFile',
        'Optimize-Disks',
        'Optimize-VSock',
        'Optimize-WSLConfig',
        'Register-HVSockServices',
        'Repair-MachinePath',
        'Repair-WSLNetworking',
        'Resolve-PcaiRepoRoot',
        'Resolve-PCArguments',
        'Search-ContentFast',
        'Search-LogsFast',
        'Send-OllamaRequest',
        'Set-LLMConfig',
        'Set-LLMProviderOrder',
        'Set-PCaiServiceState',
        'Set-PcaiSharedCacheEntry',
        'Set-WSLDefenderExclusions',
        'Start-HVSockProxy',
        'Start-PcaiCompiledServer',
        'Stop-EvaluationRun',
        'Stop-HVSockProxy',
        'Test-ForRegression',
        'Test-PcaiDllVersion',
        'Test-PcaiInference',
        'Test-PcaiNative',
        'Test-PcaiNativeAvailable',
        'Test-PcaiResourceSafety',
        'Test-RustToolAvailable',
        'Test-WSLConnectivity',
        'Watch-SystemResources',
        'Watch-VSockPerformance',
        'Write-Bullet',
        'Write-Error',
        'Write-Header',
        'Write-Info',
        'Write-SubHeader',
        'Write-Success',
        'Write-Warning'
)
$script:ExpectedAggregateProducers = @{
            'PC-AI.Common' = @('Clear-PcaiSharedCache', 'Get-PcaiAccelerationProbe', 'Get-PcaiDependencyStamp', 'Get-PcaiDirectCoreProbe', 'Get-PcaiDirectTokenEstimate', 'Get-PcaiExternalCacheStatus', 'Get-PcaiRuntimeConfig', 'Get-PcaiSharedCacheEntry', 'Get-ScriptMetadata', 'Import-PcaiAccelerationStack', 'Resolve-PcaiRepoRoot', 'Set-PcaiSharedCacheEntry', 'Write-Bullet', 'Write-Error', 'Write-Header', 'Write-Info', 'Write-SubHeader', 'Write-Success', 'Write-Warning')
            'PC-AI.Acceleration' = @('Compare-ToolPerformance', 'Find-DuplicatesFast', 'Find-FilesFast', 'Get-DiskUsageFast', 'Get-FileHashParallel', 'Get-PcaiCapabilities', 'Get-PcaiNativeStatus', 'Get-ProcessesFast', 'Get-RustToolStatus', 'Get-UnifiedHardwareReportJson', 'Initialize-PcaiNative', 'Invoke-PcaiNativeContentSearch', 'Invoke-PcaiNativeDirectoryManifest', 'Invoke-PcaiNativeDuplicates', 'Invoke-PcaiNativeEstimateTokens', 'Invoke-PcaiNativeFileSearch', 'Invoke-PcaiNativeSystemInfo', 'Invoke-PcaiNativeUnifiedHardwareReport', 'Measure-CommandPerformance', 'Search-ContentFast', 'Search-LogsFast', 'Test-PcaiNativeAvailable', 'Test-PcaiResourceSafety', 'Test-RustToolAvailable')
            'PC-AI.Cleanup' = @('Clear-TempFiles', 'Find-DuplicateFiles', 'Get-PathDuplicates', 'Repair-MachinePath')
            'PC-AI.CLI' = @('ConvertTo-PCArgumentMap', 'Get-PCCommandList', 'Get-PCCommandMap', 'Get-PCCommandModules', 'Get-PCCommandSummary', 'Get-PCModuleHelpEntry', 'Get-PCModuleHelpIndex', 'Resolve-PCArguments')
            'PC-AI.Evaluation' = @('Add-ABTestResult', 'Compare-ResponsePair', 'Compare-ResponseSimilarity', 'Export-EvaluationDataset', 'Get-ABTestAnalysis', 'Get-EvaluationDataset', 'Get-EvaluationResults', 'Get-EvaluationRunState', 'Get-PcaiArtifactsRoot', 'Get-PcaiCompiledBinaryPath', 'Get-PcaiProjectRoot', 'Get-RegressionReport', 'Import-EvaluationDataset', 'Initialize-EvaluationPaths', 'Invoke-EvaluationSuite', 'Invoke-LLMJudge', 'Measure-Coherence', 'Measure-DiagnosticQuality', 'Measure-Groundedness', 'Measure-InferenceLatency', 'Measure-MemoryUsage', 'Measure-TokenThroughput', 'Measure-Toxicity', 'New-ABTest', 'New-BaselineSnapshot', 'New-EvaluationSuite', 'New-EvaluationTestCase', 'New-PcaiEvaluationRunContext', 'New-PcaiServerConfigFile', 'Start-PcaiCompiledServer', 'Stop-EvaluationRun', 'Test-ForRegression')
            'PC-AI.Hardware' = @('Get-DeviceErrors', 'Get-DiskHealth', 'Get-NetworkAdapters', 'Get-SystemEvents', 'Get-UsbStatus', 'New-DiagnosticReport')
            'PC-AI.LLM' = @('Get-LLMStatus', 'Get-SystemInfoTool', 'Invoke-DocSearch', 'Invoke-FunctionGemmaDataset', 'Invoke-FunctionGemmaEval', 'Invoke-FunctionGemmaReAct', 'Invoke-FunctionGemmaTests', 'Invoke-FunctionGemmaTokenCache', 'Invoke-FunctionGemmaTrain', 'Invoke-LLMChat', 'Invoke-LLMChatRouted', 'Invoke-LLMChatTui', 'Invoke-LogSearch', 'Invoke-NativeSearch', 'Invoke-PCDiagnosis', 'Invoke-SmartDiagnosis', 'Send-OllamaRequest', 'Set-LLMConfig', 'Set-LLMProviderOrder')
            'PC-AI.Network' = @('Get-NetworkDiagnostics', 'Optimize-VSock', 'Test-WSLConnectivity', 'Watch-VSockPerformance')
            'PC-AI.Performance' = @('Get-DiskSpace', 'Get-PcaiDiskUsage', 'Get-PcaiMemoryPressure', 'Get-PcaiMemoryStat', 'Get-PcaiOptimizationPlan', 'Get-PcaiProcessCategories', 'Get-PcaiTopProcess', 'Get-ProcessPerformance', 'Optimize-Disks', 'Test-PcaiNative', 'Watch-SystemResources')
            'PC-AI.USB' = @('Dismount-UsbFromWSL', 'Get-PcaiNativeUsbDiagnostics', 'Get-UsbDeviceList', 'Get-UsbWSLStatus', 'Invoke-UsbBind', 'Mount-UsbToWSL')
            'PC-AI.Virtualization' = @('Backup-WSLConfig', 'Enable-WSLSystemd', 'Get-DockerStatus', 'Get-HVSockProxyStatus', 'Get-HyperVStatus', 'Get-PcaiServiceHealth', 'Get-WSLEnvironmentHealth', 'Get-WSLStatus', 'Get-WSLVsockBridgeStatus', 'Install-HVSockProxy', 'Install-WSLVsockBridge', 'Invoke-PcaiDoctor', 'Invoke-PcaiServiceHost', 'Invoke-WSLDockerHealthCheck', 'Invoke-WSLNetworkToolkit', 'Optimize-WSLConfig', 'Register-HVSockServices', 'Repair-WSLNetworking', 'Set-PCaiServiceState', 'Set-WSLDefenderExclusions', 'Start-HVSockProxy', 'Stop-HVSockProxy')
            'PcaiInference' = @('Close-PcaiInference', 'Get-PcaiInferenceStatus', 'Import-PcaiModel', 'Initialize-PcaiInference', 'Invoke-PcaiGenerate', 'Test-PcaiDllVersion', 'Test-PcaiInference')
}
    $script:AggregateRepository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    function New-AggregateFixture {
        $root = Join-Path $TestDrive ('aggregate-fixture-' + [guid]::NewGuid().ToString('N'))
        $modules = Join-Path $root 'modules'
        $aggregate = Join-Path $modules 'PC-AI'
        [void][IO.Directory]::CreateDirectory($aggregate)
        foreach ($leaf in @('PC-AI.psd1', 'PC-AI.psm1')) {
            Copy-Item -LiteralPath (Join-Path $script:AggregateRepository "Modules/PC-AI/$leaf") -Destination (Join-Path $aggregate $leaf)
        }
        $trace = Join-Path $root 'imports.txt'
        foreach ($component in $script:ExpectedAggregateProducers.Keys) {
            $directory = Join-Path $modules $component
            [void][IO.Directory]::CreateDirectory($directory)
            $names = @($script:ExpectedAggregateProducers[$component])
            if ($component -eq 'PC-AI.Common') { $names += 'Initialize-PcaiNative' }
            $body = @('[IO.File]::AppendAllText(''{0}'', "{1}`n")' -f $trace.Replace("'", "''"), $component)
            $body += "`$script:FixtureProducer = '$component'"
            foreach ($name in $names) {
                if ($name -eq 'Initialize-PcaiNative') {
                    $body += 'function {0} {{ [CmdletBinding()] param([switch]$Force) "$script:FixtureProducer|{0}|$([bool]$Force)" }}' -f $name
                } elseif ($name -eq 'Get-PcaiCapabilities') {
                    $body += 'function {0} {{ [CmdletBinding()] param([string]$Probe,[int]$Count=1) "$script:FixtureProducer|{0}|$Probe|$Count" }}' -f $name
                } else {
                    $body += 'function {0} {{ [CmdletBinding()] param([Parameter(ValueFromRemainingArguments)][object[]]$Arguments) "$script:FixtureProducer|{0}" }}' -f $name
                }
            }
            $quoted = @($names | ForEach-Object { "'$_'" }) -join ','
            $body += "Export-ModuleMember -Function @($quoted)"
            [IO.File]::WriteAllText((Join-Path $directory "$component.psm1"), ($body -join "`n"))
            if ($component -ne 'PC-AI.Common') {
                [IO.File]::WriteAllText((Join-Path $directory "$component.psd1"), "@{RootModule='$component.psm1';ModuleVersion='1.0.0';FunctionsToExport=@($quoted);CmdletsToExport=@();AliasesToExport=@()}")
            }
        }
        [pscustomobject]@{ Root = $root; Modules = $modules; Aggregate = $aggregate; Manifest = (Join-Path $aggregate 'PC-AI.psd1'); Trace = $trace }
    }
    function Import-AggregateFixture {
        Microsoft.PowerShell.Core\Import-Module -Name $script:AggregateFixture.Manifest -Scope Local -PassThru -ErrorAction Stop
    }
}

Describe 'PC-AI aggregate compatibility' -Tag 'Unit', 'Fast', 'Portable' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
    BeforeEach {
        $script:OriginalAggregateModulePath = $env:PSModulePath
        $script:AggregateFixture = New-AggregateFixture
    }
    AfterEach {
        Get-Module -All | Where-Object { $_.Path -and $_.Path.StartsWith($script:AggregateFixture.Root, [StringComparison]::OrdinalIgnoreCase) } |
            Remove-Module -Force -ErrorAction SilentlyContinue
        $env:PSModulePath = $script:OriginalAggregateModulePath
    }

    It 'preserves the independently frozen 164-name contract and contains no copied component bodies' {
        $data = Import-PowerShellDataFile -LiteralPath $script:AggregateFixture.Manifest
        @($data.FunctionsToExport).Count | Should -Be 164
        @($data.FunctionsToExport | Sort-Object) -join ',' | Should -BeExactly (@($script:LegacyAggregateNames | Sort-Object) -join ',')
        [string]$data.GUID | Should -BeExactly '02a9dcaf-0ac8-49ca-bcdd-61f39a12c721'
        $data.ContainsKey('NestedModules') | Should -BeFalse
        $data.ContainsKey('RequiredModules') | Should -BeFalse
        @(Get-ChildItem -LiteralPath (Join-Path $script:AggregateRepository 'Modules/PC-AI') -File -Recurse).Count | Should -Be 2
    }

    It 'loads the actual wrapper and maps every inherited command to its independent inert producer' {
        $aggregate = Import-AggregateFixture
        @($aggregate.ExportedFunctions.Keys | Sort-Object) -join ',' | Should -BeExactly (@($script:LegacyAggregateNames | Sort-Object) -join ',')
        foreach ($producer in $script:ExpectedAggregateProducers.Keys) {
            foreach ($name in $script:ExpectedAggregateProducers[$producer]) {
                $actual = & $aggregate.ExportedFunctions[$name]
                $actual | Should -BeLike "$producer|$name*"
            }
        }
        $command = $aggregate.ExportedFunctions['Get-PcaiCapabilities']
        $command.Parameters.Keys | Should -Contain 'Probe'
        & $command -Probe 'owned-literal' -Count 3 | Should -Match '\|Get-PcaiCapabilities\|owned-literal\|3$'
    }

    It 'chooses Acceleration explicitly for the shared Force initializer contract' {
        $aggregate = Import-AggregateFixture
        & $aggregate.ExportedFunctions['Initialize-PcaiNative'] -Force | Should -BeExactly 'PC-AI.Acceleration|Initialize-PcaiNative|True'
        & $aggregate.ExportedFunctions['Initialize-PcaiNative'] | Should -BeExactly 'PC-AI.Acceleration|Initialize-PcaiNative|False'
    }

    It 'retains release helper shape and resolves all component paths from ModuleRoot while retaining ConfigPath' {
        $aggregate = Import-AggregateFixture
        $info = & $aggregate.ExportedFunctions['Get-PcaiReleaseInfo']
        @($info.PSObject.Properties.Name | Sort-Object) -join ',' | Should -BeExactly 'Components,ConfigPath,ModuleRoot'
        $info.ModuleRoot | Should -BeExactly $script:AggregateFixture.Aggregate
        $info.ConfigPath | Should -BeExactly (Join-Path $script:AggregateFixture.Aggregate 'Config')
        @($info.Components).Count | Should -Be 12
        foreach ($relative in $info.Components) { Test-Path -LiteralPath (Join-Path $info.ModuleRoot $relative) -PathType Leaf | Should -BeTrue }
    }

    It 'reloads exactly the admitted component set without losing explicit initializer selection' {
        $aggregate = Import-AggregateFixture
        @(Get-Content -LiteralPath $script:AggregateFixture.Trace).Count | Should -Be 12
        @(& $aggregate.ExportedFunctions['Import-PcaiComponentModules']).Count | Should -Be 0
        @(Get-Content -LiteralPath $script:AggregateFixture.Trace).Count | Should -Be 24
        & $aggregate.ExportedFunctions['Initialize-PcaiNative'] -Force | Should -BeExactly 'PC-AI.Acceleration|Initialize-PcaiNative|True'
    }

    It 'refuses a missing sibling before importing any producer and cannot fall back to a foreign module path' {
        $missing = Join-Path $script:AggregateFixture.Modules 'PC-AI.USB/PC-AI.USB.psd1'
        Remove-Item -LiteralPath $missing
        $foreign = Join-Path $script:AggregateFixture.Root 'foreign/PC-AI.USB'
        [void][IO.Directory]::CreateDirectory($foreign)
        $tripwire = Join-Path $script:AggregateFixture.Root 'foreign-imported.txt'
        [IO.File]::WriteAllText((Join-Path $foreign 'PC-AI.USB.psm1'), "[IO.File]::WriteAllText('$tripwire','foreign')")
        $env:PSModulePath = (Split-Path -Parent $foreign) + [IO.Path]::PathSeparator + $env:PSModulePath
        { Import-AggregateFixture } | Should -Throw '*admitted sibling component*PC-AI.USB*'
        Test-Path -LiteralPath $script:AggregateFixture.Trace | Should -BeFalse
        Test-Path -LiteralPath $tripwire | Should -BeFalse
    }

    It 'refuses a foreign RootModule before any producer code executes' {
        $manifest = Join-Path $script:AggregateFixture.Modules 'PC-AI.USB/PC-AI.USB.psd1'
        [IO.File]::WriteAllText($manifest, "@{RootModule='../foreign.psm1';ModuleVersion='1.0.0'}")
        { Import-AggregateFixture } | Should -Throw '*foreign RootModule*'
        Test-Path -LiteralPath $script:AggregateFixture.Trace | Should -BeFalse
    }

    It 'refuses an incomplete named export contract instead of returning a partial aggregate' {
        $rootScript = Join-Path $script:AggregateFixture.Modules 'PC-AI.USB/PC-AI.USB.psm1'
        [IO.File]::WriteAllText($rootScript, "function Get-UsbStatus {}; Export-ModuleMember -Function Get-UsbStatus")
        { Import-AggregateFixture } | Should -Throw '*does not export required function*'
    }

    It 'uses actual installer staging to retain old package bytes and Config while publishing only the thin payload' {
        $repo = Join-Path $script:AggregateFixture.Root 'source-repo'
        $sourceAggregate = Join-Path $repo 'Modules/PC-AI'
        [void][IO.Directory]::CreateDirectory($sourceAggregate)
        foreach ($leaf in @('PC-AI.psd1', 'PC-AI.psm1')) { Copy-Item -LiteralPath (Join-Path $script:AggregateRepository "Modules/PC-AI/$leaf") -Destination (Join-Path $sourceAggregate $leaf) }
        $oldLoader = Join-Path $script:AggregateFixture.Aggregate 'PC-AI.psm1'
        [IO.File]::WriteAllText($oldLoader, "'owned old aggregate loader'")
        $oldHash = (Get-FileHash -LiteralPath $oldLoader -Algorithm SHA256).Hash
        $oldBody = Join-Path $script:AggregateFixture.Aggregate 'Modules/retained-old-body.psm1'
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent $oldBody))
        [IO.File]::WriteAllText($oldBody, "'owned old component body'")
        $oldBodyHash = (Get-FileHash -LiteralPath $oldBody -Algorithm SHA256).Hash
        $config = Join-Path $script:AggregateFixture.Aggregate 'Config/local-settings.json'
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent $config))
        [IO.File]::WriteAllText($config, "{`"ownedValue`":true}`r`n")
        $configHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash
        $opaqueConfig = Join-Path (Split-Path -Parent $config) 'owned-state.conf'
        [IO.File]::WriteAllBytes($opaqueConfig, [byte[]](0x4F,0x57,0x4E,0x45,0x44,0x0D,0x0A))
        $opaqueHash = (Get-FileHash -LiteralPath $opaqueConfig -Algorithm SHA256).Hash
        $beforePath = $env:PSModulePath
        $install = @(& (Join-Path $script:AggregateRepository 'Tools/Install-PcaiDevModules.ps1') -RepoRoot $repo -InstallRoot $script:AggregateFixture.Modules -BackupRoot (Join-Path $script:AggregateFixture.Root 'retained') -ModuleNames PC-AI -Mode Copy -UpdatePSModulePath:$false -Confirm:$false)
        $install.Count | Should -Be 1
        $install[0].Status | Should -BeExactly 'Installed'
        $env:PSModulePath | Should -BeExactly $beforePath
        # The maintained installer merges JSON values and retains displaced raw
        # bytes; opaque Config files remain byte-identical in both active/backup.
        (Get-Content -LiteralPath $config -Raw | ConvertFrom-Json).ownedValue | Should -BeTrue
        (Get-FileHash -LiteralPath $opaqueConfig -Algorithm SHA256).Hash | Should -BeExactly $opaqueHash
        Test-Path -LiteralPath $oldBody | Should -BeFalse
        (Get-FileHash -LiteralPath (Join-Path $install[0].BackupPath 'PC-AI.psm1') -Algorithm SHA256).Hash | Should -BeExactly $oldHash
        (Get-FileHash -LiteralPath (Join-Path $install[0].BackupPath 'Modules/retained-old-body.psm1') -Algorithm SHA256).Hash | Should -BeExactly $oldBodyHash
        (Get-FileHash -LiteralPath (Join-Path $install[0].BackupPath 'Config/local-settings.json') -Algorithm SHA256).Hash | Should -BeExactly $configHash
        (Get-FileHash -LiteralPath (Join-Path $install[0].BackupPath 'Config/owned-state.conf') -Algorithm SHA256).Hash | Should -BeExactly $opaqueHash
        $aggregate = Import-AggregateFixture
        (& $aggregate.ExportedFunctions['Get-PcaiReleaseInfo']).ConfigPath | Should -BeExactly (Split-Path -Parent $config)
    }
}
