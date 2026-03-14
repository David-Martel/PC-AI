#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ConfigPath,

    [Parameter()]
    [string]$Suite,

    [Parameter()]
    [string[]]$CaseId,

    [Parameter()]
    [string]$OutputRoot,

    [Parameter()]
    [switch]$SkipCapabilities,

    [Parameter()]
    [switch]$SkipPerformance,

    [Parameter()]
    [switch]$RefreshCoverage,

    [Parameter()]
    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

function Invoke-LegacyTokenEstimate {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return 0 }
    return ([regex]::Matches($Text, '\w+')).Count
}

function Get-PowerShellDirectoryManifest {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [uint32]$MaxDepth = 0,
        [uint64]$MaxResults = 0
    )

    $resolvedPath = (Resolve-Path -Path $Path -ErrorAction Stop).Path
    $items = if ($MaxDepth -gt 0) {
        Get-ChildItem -LiteralPath $resolvedPath -Force -ErrorAction SilentlyContinue -Recurse -Depth $MaxDepth
    } else {
        Get-ChildItem -LiteralPath $resolvedPath -Force -ErrorAction SilentlyContinue -Recurse
    }
    if ($MaxResults -gt 0) {
        $items = $items | Select-Object -First $MaxResults
    }

    $entries = @($items)
    $files = @($entries | Where-Object { -not $_.PSIsContainer })
    $directories = @($entries | Where-Object { $_.PSIsContainer })

    [PSCustomObject]@{
        EntriesReturned = $entries.Count
        FileCount       = $files.Count
        DirectoryCount  = $directories.Count
        TotalSize       = ($files | Measure-Object -Property Length -Sum).Sum
    }
}

function Invoke-LegacyFullInterrogation {
    $sys = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, CSName
    $cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, LoadPercentage
    $net = Get-NetIPConfiguration
    $wsl = wsl --list --verbose | Out-String

    @{
        System  = $sys
        CPU     = $cpu
        Network = $net
        Vmm     = $wsl
    } | ConvertTo-Json -Depth 3
}

function Get-ConfigValue {
    param(
        [Parameter(Mandatory)]
        [psobject]$Case,
        [Parameter(Mandatory)]
        [psobject]$Defaults,
        [Parameter(Mandatory)]
        [string]$Name
    )

    $caseProperty = $Case.PSObject.Properties[$Name]
    if ($caseProperty) {
        return $caseProperty.Value
    }

    $defaultProperty = $Defaults.PSObject.Properties[$Name]
    if ($defaultProperty) {
        return $defaultProperty.Value
    }

    return $null
}

function Invoke-BackendBenchmark {
    param(
        [Parameter(Mandatory)]
        [string]$CaseId,
        [Parameter(Mandatory)]
        [string]$Backend,
        [Parameter(Mandatory)]
        [object]$Command,
        [Parameter(Mandatory)]
        [int]$Iterations,
        [Parameter(Mandatory)]
        [int]$Warmup,
        [Parameter()]
        [ValidateSet('pwsh', 'powershell', 'cmd', 'bash')]
        [string]$Shell = 'pwsh'
    )

    $measureCommand = Get-ModuleFunctionCommand -Module $script:AccelerationModule -CommandName 'Measure-CommandPerformance'
    $measurement = & $measureCommand -Command $Command -Iterations $Iterations -Warmup $Warmup -Name "$CaseId/$Backend" -Shell $Shell
    if (-not $measurement) {
        return $null
    }

    [PSCustomObject]@{
        CaseId      = $CaseId
        Backend     = $Backend
        MeanMs      = $measurement.Mean
        MedianMs    = $measurement.Median
        MinMs       = $measurement.Min
        MaxMs       = $measurement.Max
        StdDevMs    = $measurement.StdDev
        Iterations  = $measurement.Iterations
        Tool        = $measurement.Tool
        WorkingSetDeltaMeanBytes = $measurement.WorkingSetDeltaMeanBytes
        WorkingSetDeltaMaxBytes  = $measurement.WorkingSetDeltaMaxBytes
        PrivateMemoryDeltaMeanBytes = $measurement.PrivateMemoryDeltaMeanBytes
        PrivateMemoryDeltaMaxBytes  = $measurement.PrivateMemoryDeltaMaxBytes
        ManagedMemoryDeltaMeanBytes = $measurement.ManagedMemoryDeltaMeanBytes
        ManagedMemoryDeltaMaxBytes  = $measurement.ManagedMemoryDeltaMaxBytes
        ManagedAllocatedMeanBytes = $measurement.ManagedAllocatedMeanBytes
        ManagedAllocatedMaxBytes  = $measurement.ManagedAllocatedMaxBytes
    }
}

function Add-BenchmarkMeasurement {
    param(
        [Parameter(Mandatory)]
        [object]$Collection,
        [AllowNull()]
        [object]$Measurement
    )

    if ($null -ne $Measurement) {
        $Collection.Add($Measurement)
    }
}

function Invoke-ImportedModuleCommand {
    param(
        [Parameter(Mandatory)]
        [System.Management.Automation.PSModuleInfo]$Module,
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList = @()
    )

    if (-not $Module) {
        throw 'Imported module handle is not available.'
    }

    & $Module $ScriptBlock @ArgumentList
}

function Invoke-AccelerationModuleCommand {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList = @()
    )

    Invoke-ImportedModuleCommand -Module $script:AccelerationModule -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList
}

function Invoke-CommonModuleCommand {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList = @()
    )

    Invoke-ImportedModuleCommand -Module $script:CommonModule -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList
}

function Invoke-CliModuleCommand {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList = @()
    )

    Invoke-ImportedModuleCommand -Module $script:CliModule -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList
}

function Get-ModuleFunctionCommand {
    param(
        [Parameter(Mandatory)]
        [System.Management.Automation.PSModuleInfo]$Module,
        [Parameter(Mandatory)]
        [string]$CommandName
    )

    $resolved = $null
    if ($Module.ExportedFunctions -and $Module.ExportedFunctions.ContainsKey($CommandName)) {
        $resolved = $Module.ExportedFunctions[$CommandName]
    }

    if (-not $resolved -and $Module.ExportedCommands -and $Module.ExportedCommands.ContainsKey($CommandName)) {
        $resolved = $Module.ExportedCommands[$CommandName]
    }

    if (-not $resolved) {
        throw "Failed to resolve function '$CommandName' from module '$($Module.Name)'."
    }

    return $resolved
}

function Get-ToolingCaseBenchmarks {
    param(
        [Parameter(Mandatory)]
        [psobject]$Case,
        [Parameter(Mandatory)]
        [psobject]$Defaults,
        [Parameter(Mandatory)]
        [string]$RepoRoot,
        [Parameter(Mandatory)]
        [psobject]$Capabilities,
        [Parameter(Mandatory)]
        [string]$TokenSample
    )

    $casePath = Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'path'
    $resolvedPath = if ($casePath) {
        if ([System.IO.Path]::IsPathRooted([string]$casePath)) {
            (Resolve-Path -Path $casePath -ErrorAction Stop).Path
        } else {
            (Resolve-Path -Path (Join-Path $RepoRoot $casePath) -ErrorAction Stop).Path
        }
    } else {
        $RepoRoot
    }

    $iterations = [int](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'iterations')
    if ($iterations -le 0) { $iterations = 5 }
    $warmup = [int](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'warmup')
    if ($warmup -lt 0) { $warmup = 1 }
    $maxDepth = [uint32](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'maxDepth')
    $maxResults = [uint64](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'maxResults')
    $filePattern = [string](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'filePattern')
    $contentPattern = [string](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'contentPattern')
    $contentFilePattern = [string](Get-ConfigValue -Case $Case -Defaults $Defaults -Name 'contentFilePattern')

    $coverageLookup = @{}
    $backendCoverage = @()
    if ($Capabilities -and $Capabilities.PSObject.Properties['BackendCoverage']) {
        $backendCoverage = @($Capabilities.BackendCoverage)
    }

    foreach ($entry in $backendCoverage) {
        $coverageLookup[$entry.Operation] = $entry
    }

    $accelerationModule = $script:AccelerationModule
    $commonModule = $script:CommonModule
    $cliModule = $script:CliModule
    $benchmarks = [System.Collections.Generic.List[object]]::new()

    switch ($Case.Id) {
        'token-estimate' {
            $nativeBlock = {
                & $accelerationModule {
                    param($InnerTokenSample)
                    Get-PcaiTokenEstimate -Text $InnerTokenSample
                } $TokenSample | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -Command $nativeBlock -Iterations $iterations -Warmup $warmup)
            $baselineBlock = { if ([string]::IsNullOrEmpty($TokenSample)) { 0 } else { ([regex]::Matches($TokenSample, '\w+')).Count } }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $baselineBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'TokenEstimate'
                Coverage    = $coverageLookup['TokenEstimate']
                Results     = @($benchmarks)
            }
        }
        'directory-manifest' {
            $nativeBlock = {
                & $accelerationModule {
                    param($InnerPath, $InnerMaxDepth, $InnerMaxResults)
                    Invoke-PcaiNativeDirectoryManifest -Path $InnerPath -MaxDepth $InnerMaxDepth -MaxResults $InnerMaxResults -StatsOnly
                } $resolvedPath $maxDepth $maxResults | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -Command $nativeBlock -Iterations $iterations -Warmup $warmup)
            $baselineBlock = {
                $items = if ($maxDepth -gt 0) {
                    Get-ChildItem -LiteralPath $resolvedPath -Force -ErrorAction SilentlyContinue -Recurse -Depth $maxDepth
                } else {
                    Get-ChildItem -LiteralPath $resolvedPath -Force -ErrorAction SilentlyContinue -Recurse
                }
                if ($maxResults -gt 0) {
                    $items = $items | Select-Object -First $maxResults
                }
                $entries = @($items)
                $files = @($entries | Where-Object { -not $_.PSIsContainer })
                $directories = @($entries | Where-Object { $_.PSIsContainer })
                [PSCustomObject]@{
                    EntriesReturned = $entries.Count
                    FileCount       = $files.Count
                    DirectoryCount  = $directories.Count
                    TotalSize       = ($files | Measure-Object -Property Length -Sum).Sum
                } | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $baselineBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'DirectoryManifest'
                Coverage    = $coverageLookup['DirectoryManifest']
                Results     = @($benchmarks)
            }
        }
        'file-search' {
            $nativeBlock = {
                & $accelerationModule {
                    param($InnerPattern, $InnerPath, $InnerMaxResults)
                    Invoke-PcaiNativeFileSearch -Pattern $InnerPattern -Path $InnerPath -MaxResults $InnerMaxResults | Out-Null
                } $filePattern $resolvedPath $maxResults
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -Command $nativeBlock -Iterations $iterations -Warmup $warmup)
            $acceleratedBlock = {
                & $accelerationModule {
                    param($InnerPath, $InnerPattern, $InnerMaxResults)
                    if ($script:PcaiQueryCache -and $script:PcaiQueryCache.Entries) {
                        $script:PcaiQueryCache.Entries.Clear()
                    }
                    $items = Find-FilesFast -Path $InnerPath -Pattern $InnerPattern
                    if ($InnerMaxResults -gt 0) {
                        $items = $items | Select-Object -First $InnerMaxResults
                    }
                    $items | Out-Null
                } $resolvedPath $filePattern $maxResults
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'accelerated' -Command $acceleratedBlock -Iterations $iterations -Warmup $warmup)
            $baselineBlock = {
                $items = Get-ChildItem -LiteralPath $resolvedPath -Recurse -File -Filter $filePattern -ErrorAction SilentlyContinue
                if ($maxResults -gt 0) {
                    $items = $items | Select-Object -First $maxResults
                }
                $items | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $baselineBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'FileSearch'
                Coverage    = $coverageLookup['FileSearch']
                Results     = @($benchmarks)
            }
        }
        'content-search' {
            $nativeBlock = {
                & $accelerationModule {
                    param($InnerPattern, $InnerPath, $InnerFilePattern, $InnerMaxResults)
                    Invoke-PcaiNativeContentSearch -Pattern $InnerPattern -Path $InnerPath -FilePattern $InnerFilePattern -MaxResults $InnerMaxResults | Out-Null
                } $contentPattern $resolvedPath $contentFilePattern $maxResults
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -Command $nativeBlock -Iterations $iterations -Warmup $warmup)
            $acceleratedBlock = {
                & $accelerationModule {
                    param($InnerPath, $InnerPattern, $InnerFilePattern, $InnerMaxResults)
                    if ($script:PcaiQueryCache -and $script:PcaiQueryCache.Entries) {
                        $script:PcaiQueryCache.Entries.Clear()
                    }
                    Search-ContentFast -Path $InnerPath -Pattern $InnerPattern -FilePattern $InnerFilePattern -MaxResults $InnerMaxResults | Out-Null
                } $resolvedPath $contentPattern $contentFilePattern $maxResults
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'accelerated' -Command $acceleratedBlock -Iterations $iterations -Warmup $warmup)
            $baselineBlock = {
                $items = Get-ChildItem -LiteralPath $resolvedPath -Recurse -File -Filter $contentFilePattern -ErrorAction SilentlyContinue |
                    Select-String -Pattern $contentPattern
                if ($maxResults -gt 0) {
                    $items = $items | Select-Object -First $maxResults
                }
                $items | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $baselineBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'ContentSearch'
                Coverage    = $coverageLookup['ContentSearch']
                Results     = @($benchmarks)
            }
        }
        'full-context' {
            if (Invoke-AccelerationModuleCommand -ScriptBlock { [PcaiNative.PcaiCore]::IsAvailable }) {
                $nativeBlock = {
                    & $accelerationModule { [PcaiNative.PcaiCore]::QueryFullContextJson() | Out-Null }
                }.GetNewClosure()
                Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -Command $nativeBlock -Iterations $iterations -Warmup $warmup)
            }
            $baselineBlock = {
                $sys = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, CSName
                $cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, LoadPercentage
                $net = Get-NetIPConfiguration
                $wsl = wsl --list --verbose | Out-String
                @{
                    System  = $sys
                    CPU     = $cpu
                    Network = $net
                    Vmm     = $wsl
                } | ConvertTo-Json -Depth 3 | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $baselineBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'FullContext'
                Coverage    = $coverageLookup['FullContext']
                Results     = @($benchmarks)
            }
        }
        'runtime-config' {
            $scriptBlock = {
                & $commonModule { Get-PcaiRuntimeConfig | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'shared-cache-hit' {
            $sharedCacheValue = [pscustomobject]@{
                Name = 'pcai-benchmark'
                Root = $RepoRoot
                Stamp = (Get-Date).ToString('yyyyMMdd')
            }
            Invoke-CommonModuleCommand -ScriptBlock {
                param($InnerValue)
                Clear-PcaiSharedCache -Namespace 'pcai-benchmark'
                Set-PcaiSharedCacheEntry -Namespace 'pcai-benchmark' -Key 'hit' -Value $InnerValue -TtlSeconds 120 | Out-Null
            } -ArgumentList @($sharedCacheValue)

            $scriptBlock = {
                & $commonModule { Get-PcaiSharedCacheEntry -Namespace 'pcai-benchmark' -Key 'hit' -TtlSeconds 120 | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'external-cache-status' {
            $scriptBlock = {
                & $commonModule { Get-PcaiExternalCacheStatus | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'acceleration-probe' {
            $scriptBlock = {
                & $commonModule { Get-PcaiAccelerationProbe | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'acceleration-stack' {
            $scriptBlock = {
                & $commonModule { Import-PcaiAccelerationStack -Modules @('ProfileAccelerator', 'PC-AI.Acceleration') -RepoRoot $RepoRoot | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'direct-core-probe' {
            $scriptBlock = {
                & $commonModule { Get-PcaiDirectCoreProbe | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'command-map' {
            $scriptBlock = {
                & $cliModule { Get-PCCommandMap | Out-Null }
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $scriptBlock -Iterations $iterations -Warmup $warmup)
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'acceleration-import' {
            $modulePath = Join-Path $RepoRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
            $importBlock = {
                pwsh -NoProfile -Command "& { Import-Module '$modulePath' -Force }" | Out-Null
            }.GetNewClosure()
            Add-BenchmarkMeasurement -Collection $benchmarks -Measurement (
                Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -Command $importBlock -Iterations $iterations -Warmup $warmup
            )
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        default {
            throw "Unsupported benchmark case: $($Case.Id)"
        }
    }
}

$PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $PcaiRoot 'Config\pcai-tooling-benchmarks.json'
}
$ConfigPath = (Resolve-Path -Path $ConfigPath -ErrorAction Stop).Path

$moduleImports = @(
    Join-Path $PcaiRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
    Join-Path $PcaiRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
    Join-Path $PcaiRoot 'Modules\PC-AI.CLI\PC-AI.CLI.psm1'
)
$script:ImportedModules = @{}
foreach ($modulePath in $moduleImports) {
    $importedModule = Import-Module $modulePath -Force -PassThru
    $script:ImportedModules[$importedModule.Name] = $importedModule
}
$script:CommonModule = $script:ImportedModules['PC-AI.Common']
$script:AccelerationModule = $script:ImportedModules['PC-AI.Acceleration']
$script:CliModule = $script:ImportedModules['PC-AI.CLI']

$config = Get-Content -Path $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$defaults = $config.defaults
$suiteName = if ($Suite) { $Suite } elseif ($config.defaultSuite) { [string]$config.defaultSuite } else { 'default' }
$selectedIds = if ($CaseId) {
    @($CaseId)
} else {
    $suiteValue = $config.suites.PSObject.Properties[$suiteName]
    if (-not $suiteValue) {
        throw "Unknown benchmark suite: $suiteName"
    }
    @($suiteValue.Value)
}

$caseLookup = @{}
foreach ($case in @($config.cases)) {
    $caseLookup[$case.id] = $case
}

$selectedCases = foreach ($id in $selectedIds) {
    if (-not $caseLookup.ContainsKey($id)) {
        throw "Unknown benchmark case: $id"
    }
    $caseLookup[$id]
}

$reportRootBase = if ($OutputRoot) {
    $OutputRoot
} else {
    Join-Path $PcaiRoot ([string]$config.reportDirectory)
}
if (-not [System.IO.Path]::IsPathRooted($reportRootBase)) {
    $reportRootBase = Join-Path $PcaiRoot $reportRootBase
}
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportRoot = Join-Path $reportRootBase $timestamp
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null

$capabilities = $null
if (-not $SkipCapabilities) {
    $capabilitiesCommand = Get-ModuleFunctionCommand -Module $script:AccelerationModule -CommandName 'Get-PcaiCapabilities'
    $capabilities = & $capabilitiesCommand
    $capabilityJsonPath = Join-Path $reportRoot 'capabilities.json'
    $capabilities | ConvertTo-Json -Depth 8 | Set-Content -Path $capabilityJsonPath -Encoding UTF8
}
if (-not $capabilities) {
    $capabilities = [PSCustomObject]@{
        BackendCoverage = @()
    }
}

$toolCoverageJson = $null
$toolCoverageScript = Join-Path $PcaiRoot 'Tools\update-tool-coverage.ps1'
if (Test-Path $toolCoverageScript) {
    $toolCoverageJsonPath = Join-Path $PcaiRoot 'Reports\TOOL_SCHEMA_REPORT.json'
    if ($RefreshCoverage -or -not (Test-Path $toolCoverageJsonPath)) {
    & $toolCoverageScript -RepoRoot $PcaiRoot | Out-Null
    }
    if (Test-Path $toolCoverageJsonPath) {
        $toolCoverageJson = Get-Content -Path $toolCoverageJsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
}

$tokenSeed = 'The quick brown fox jumps over the lazy dog. '
$tokenRepeat = [int](Get-ConfigValue -Case ([pscustomobject]@{}) -Defaults $defaults -Name 'tokenSampleRepeat')
if ($tokenRepeat -le 0) { $tokenRepeat = 4000 }
$tokenSample = $tokenSeed * $tokenRepeat

$caseResults = [System.Collections.Generic.List[object]]::new()
if (-not $SkipPerformance) {
    foreach ($case in $selectedCases) {
        $caseResults.Add((Get-ToolingCaseBenchmarks -Case $case -Defaults $defaults -RepoRoot $PcaiRoot -Capabilities $capabilities -TokenSample $tokenSample))
    }
}

$benchmarkRows = foreach ($caseResult in $caseResults) {
    $baseline = @($caseResult.Results | Where-Object Backend -eq 'powershell' | Select-Object -First 1)
    $baselineMean = if ($baseline.Count -gt 0) { [double]$baseline[0].MeanMs } else { $null }
    foreach ($row in @($caseResult.Results)) {
        [PSCustomObject]@{
            CaseId             = $caseResult.Case.id
            Name               = $caseResult.Case.name
            Category           = $caseResult.Case.category
            Path               = $caseResult.Path
            Backend            = $row.Backend
            MeanMs             = $row.MeanMs
            MedianMs           = $row.MedianMs
            StdDevMs           = $row.StdDevMs
            MinMs              = $row.MinMs
            MaxMs              = $row.MaxMs
            Iterations         = $row.Iterations
            Tool               = $row.Tool
            SpeedupVsBaseline  = if ($baselineMean -and $row.MeanMs -gt 0) { [Math]::Round($baselineMean / [double]$row.MeanMs, 2) } else { $null }
            CoverageState      = if ($caseResult.Coverage) { $caseResult.Coverage.CoverageState } else { 'PowerShellOnly' }
            WorkingSetDeltaMeanBytes = $row.WorkingSetDeltaMeanBytes
            WorkingSetDeltaMaxBytes  = $row.WorkingSetDeltaMaxBytes
            PrivateMemoryDeltaMeanBytes = $row.PrivateMemoryDeltaMeanBytes
            PrivateMemoryDeltaMaxBytes  = $row.PrivateMemoryDeltaMaxBytes
            ManagedMemoryDeltaMeanBytes = $row.ManagedMemoryDeltaMeanBytes
            ManagedMemoryDeltaMaxBytes  = $row.ManagedMemoryDeltaMaxBytes
            ManagedAllocatedMeanBytes = $row.ManagedAllocatedMeanBytes
            ManagedAllocatedMaxBytes  = $row.ManagedAllocatedMaxBytes
        }
    }
}

$summary = [PSCustomObject]@{
    Generated        = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    RepoRoot         = $PcaiRoot
    ConfigPath       = $ConfigPath
    Suite            = $suiteName
    SelectedCases    = @($selectedCases | ForEach-Object { $_.id })
    BenchmarkCount   = @($benchmarkRows).Count
    Reports          = [PSCustomObject]@{
        Root       = $reportRoot
        Json       = (Join-Path $reportRoot 'tooling-benchmark-report.json')
        Markdown   = (Join-Path $reportRoot 'tooling-benchmark-report.md')
    }
    Capabilities     = $capabilities
    ToolSchemaReport = $toolCoverageJson
    Results          = @($benchmarkRows)
}

$summary.Reports.Json = $summary.Reports.Json
$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $summary.Reports.Json -Encoding UTF8

$md = [System.Text.StringBuilder]::new()
$null = $md.AppendLine('# PC_AI Tooling Benchmark Report')
$null = $md.AppendLine()
$null = $md.AppendLine(('Generated: {0}' -f $summary.Generated))
$null = $md.AppendLine(('Suite: {0}' -f $summary.Suite))
$null = $md.AppendLine(('RepoRoot: `{0}`' -f $summary.RepoRoot))
$null = $md.AppendLine()

if ($capabilities) {
    $null = $md.AppendLine('## Backend Coverage')
    $null = $md.AppendLine()
    $null = $md.AppendLine('| Operation | Coverage | Preferred | Gap |')
    $null = $md.AppendLine('| --- | --- | --- | --- |')
    foreach ($coverageRow in @($capabilities.BackendCoverage)) {
        $gap = if ([string]::IsNullOrWhiteSpace([string]$coverageRow.Gap)) { '' } else { [string]$coverageRow.Gap }
        $null = $md.AppendLine(('| {0} | {1} | {2} | {3} |' -f $coverageRow.Operation, $coverageRow.CoverageState, $coverageRow.PreferredBackend, $gap))
    }
    $null = $md.AppendLine()
}

if ($toolCoverageJson) {
    $null = $md.AppendLine('## Tool Schema Coverage')
    $null = $md.AppendLine()
    $null = $md.AppendLine("- Tool schema count: $($toolCoverageJson.ToolSchemaCount)")
    $null = $md.AppendLine("- Tool mapping count: $($toolCoverageJson.ToolMappingCount)")
    $null = $md.AppendLine("- Backend coverage rows: $($toolCoverageJson.BackendCoverageCount)")
    $null = $md.AppendLine()
}

$null = $md.AppendLine('## Performance Results')
$null = $md.AppendLine()
$null = $md.AppendLine('| Case | Backend | Mean ms | Median ms | StdDev ms | WS delta KB | Private delta KB | Managed delta KB | Managed alloc KB | Speedup vs PS | Coverage |')
$null = $md.AppendLine('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
foreach ($row in @($benchmarkRows | Sort-Object CaseId, MeanMs)) {
    $speedup = if ($null -ne $row.SpeedupVsBaseline) { $row.SpeedupVsBaseline } else { '' }
    $workingSetKb = if ($null -ne $row.WorkingSetDeltaMeanBytes) { [Math]::Round(([double]$row.WorkingSetDeltaMeanBytes / 1KB), 2) } else { '' }
    $privateKb = if ($null -ne $row.PrivateMemoryDeltaMeanBytes) { [Math]::Round(([double]$row.PrivateMemoryDeltaMeanBytes / 1KB), 2) } else { '' }
    $managedKb = if ($null -ne $row.ManagedMemoryDeltaMeanBytes) { [Math]::Round(([double]$row.ManagedMemoryDeltaMeanBytes / 1KB), 2) } else { '' }
    $managedAllocatedKb = if ($null -ne $row.ManagedAllocatedMeanBytes) { [Math]::Round(([double]$row.ManagedAllocatedMeanBytes / 1KB), 2) } else { '' }
    $null = $md.AppendLine(('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |' -f $row.CaseId, $row.Backend, $row.MeanMs, $row.MedianMs, $row.StdDevMs, $workingSetKb, $privateKb, $managedKb, $managedAllocatedKb, $speedup, $row.CoverageState))
}
$md.ToString() | Set-Content -Path $summary.Reports.Markdown -Encoding UTF8

$resultObject = [PSCustomObject]@{
    ReportRoot     = $reportRoot
    JsonReportPath = $summary.Reports.Json
    MarkdownPath   = $summary.Reports.Markdown
    Summary        = $summary
}

if ($PassThru) {
    $resultObject
} else {
    $resultObject | Select-Object ReportRoot, JsonReportPath, MarkdownPath
}
