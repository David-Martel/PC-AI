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
        [scriptblock]$ScriptBlock,
        [Parameter(Mandatory)]
        [int]$Iterations,
        [Parameter(Mandatory)]
        [int]$Warmup
    )

    $measurement = Measure-CommandPerformance -Command $ScriptBlock -Iterations $Iterations -Warmup $Warmup -Name "$CaseId/$Backend"
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
    }
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
    foreach ($entry in @($Capabilities.BackendCoverage)) {
        $coverageLookup[$entry.Operation] = $entry
    }

    $benchmarks = [System.Collections.Generic.List[object]]::new()

    switch ($Case.Id) {
        'token-estimate' {
            if (Get-Command Get-PcaiTokenEstimate -ErrorAction SilentlyContinue) {
                $scriptBlock = { Get-PcaiTokenEstimate -Text $TokenSample }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -ScriptBlock $scriptBlock -Iterations $iterations -Warmup $warmup))
            }
            $baselineBlock = { Invoke-LegacyTokenEstimate -Text $TokenSample }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $baselineBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'TokenEstimate'
                Coverage    = $coverageLookup['TokenEstimate']
                Results     = @($benchmarks)
            }
        }
        'directory-manifest' {
            if (Get-Command Invoke-PcaiNativeDirectoryManifest -ErrorAction SilentlyContinue) {
                $nativeBlock = { Invoke-PcaiNativeDirectoryManifest -Path $resolvedPath -MaxDepth $maxDepth -MaxResults $maxResults -StatsOnly }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -ScriptBlock $nativeBlock -Iterations $iterations -Warmup $warmup))
            }
            $baselineBlock = { Get-PowerShellDirectoryManifest -Path $resolvedPath -MaxDepth $maxDepth -MaxResults $maxResults }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $baselineBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'DirectoryManifest'
                Coverage    = $coverageLookup['DirectoryManifest']
                Results     = @($benchmarks)
            }
        }
        'file-search' {
            if (Get-Command Invoke-PcaiNativeFileSearch -ErrorAction SilentlyContinue) {
                $nativeBlock = { Invoke-PcaiNativeFileSearch -Pattern $filePattern -Path $resolvedPath -MaxResults $maxResults -StatsOnly }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -ScriptBlock $nativeBlock -Iterations $iterations -Warmup $warmup))
            }
            if (Get-Command Find-FilesFast -ErrorAction SilentlyContinue) {
                $acceleratedBlock = { Find-FilesFast -Path $resolvedPath -Pattern $filePattern -MaxResults $maxResults | Out-Null }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'accelerated' -ScriptBlock $acceleratedBlock -Iterations $iterations -Warmup $warmup))
            }
            $baselineBlock = {
                $items = Get-ChildItem -LiteralPath $resolvedPath -Recurse -File -Filter $filePattern -ErrorAction SilentlyContinue
                if ($maxResults -gt 0) {
                    $items = $items | Select-Object -First $maxResults
                }
                $items | Out-Null
            }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $baselineBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'FileSearch'
                Coverage    = $coverageLookup['FileSearch']
                Results     = @($benchmarks)
            }
        }
        'content-search' {
            if (Get-Command Invoke-PcaiNativeContentSearch -ErrorAction SilentlyContinue) {
                $nativeBlock = { Invoke-PcaiNativeContentSearch -Pattern $contentPattern -Path $resolvedPath -FilePattern $contentFilePattern -MaxResults $maxResults -StatsOnly }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -ScriptBlock $nativeBlock -Iterations $iterations -Warmup $warmup))
            }
            if (Get-Command Search-ContentFast -ErrorAction SilentlyContinue) {
                $acceleratedBlock = { Search-ContentFast -Path $resolvedPath -Pattern $contentPattern -FilePattern $contentFilePattern -MaxResults $maxResults | Out-Null }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'accelerated' -ScriptBlock $acceleratedBlock -Iterations $iterations -Warmup $warmup))
            }
            $baselineBlock = {
                $items = Get-ChildItem -LiteralPath $resolvedPath -Recurse -File -Filter $contentFilePattern -ErrorAction SilentlyContinue |
                    Select-String -Pattern $contentPattern -List
                if ($maxResults -gt 0) {
                    $items = $items | Select-Object -First $maxResults
                }
                $items | Out-Null
            }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $baselineBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'ContentSearch'
                Coverage    = $coverageLookup['ContentSearch']
                Results     = @($benchmarks)
            }
        }
        'full-context' {
            if ([PcaiNative.PcaiCore]::IsAvailable) {
                $nativeBlock = { [PcaiNative.PcaiCore]::QueryFullContextJson() | Out-Null }.GetNewClosure()
                $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'native' -ScriptBlock $nativeBlock -Iterations $iterations -Warmup $warmup))
            }
            $baselineBlock = { Invoke-LegacyFullInterrogation | Out-Null }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $baselineBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = 'FullContext'
                Coverage    = $coverageLookup['FullContext']
                Results     = @($benchmarks)
            }
        }
        'runtime-config' {
            $scriptBlock = { Get-PcaiRuntimeConfig | Out-Null }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $scriptBlock -Iterations $iterations -Warmup $warmup))
            return [PSCustomObject]@{
                Case        = $Case
                Path        = $resolvedPath
                CoverageKey = $null
                Coverage    = $null
                Results     = @($benchmarks)
            }
        }
        'command-map' {
            $scriptBlock = { Get-PCCommandMap | Out-Null }.GetNewClosure()
            $benchmarks.Add((Invoke-BackendBenchmark -CaseId $Case.Id -Backend 'powershell' -ScriptBlock $scriptBlock -Iterations $iterations -Warmup $warmup))
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
    Join-Path $PcaiRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1'
    Join-Path $PcaiRoot 'Modules\PC-AI.CLI\PC-AI.CLI.psm1'
)
foreach ($modulePath in $moduleImports) {
    Import-Module $modulePath -Force
}

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
    $capabilities = Get-PcaiCapabilities
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
    & $toolCoverageScript -RepoRoot $PcaiRoot | Out-Null
    $toolCoverageJsonPath = Join-Path $PcaiRoot 'Reports\TOOL_SCHEMA_REPORT.json'
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
$null = $md.AppendLine("Generated: $($summary.Generated)")
$null = $md.AppendLine("Suite: $($summary.Suite)")
$null = $md.AppendLine("RepoRoot: `$($summary.RepoRoot)`")
$null = $md.AppendLine()

if ($capabilities) {
    $null = $md.AppendLine('## Backend Coverage')
    $null = $md.AppendLine()
    $null = $md.AppendLine('| Operation | Coverage | Preferred | Gap |')
    $null = $md.AppendLine('| --- | --- | --- | --- |')
    foreach ($coverageRow in @($capabilities.BackendCoverage)) {
        $gap = if ([string]::IsNullOrWhiteSpace([string]$coverageRow.Gap)) { '' } else { [string]$coverageRow.Gap }
        $null = $md.AppendLine("| $($coverageRow.Operation) | $($coverageRow.CoverageState) | $($coverageRow.PreferredBackend) | $gap |")
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
$null = $md.AppendLine('| Case | Backend | Mean ms | Median ms | StdDev ms | Speedup vs PS | Coverage |')
$null = $md.AppendLine('| --- | --- | ---: | ---: | ---: | ---: | --- |')
foreach ($row in @($benchmarkRows | Sort-Object CaseId, MeanMs)) {
    $speedup = if ($null -ne $row.SpeedupVsBaseline) { $row.SpeedupVsBaseline } else { '' }
    $null = $md.AppendLine("| $($row.CaseId) | $($row.Backend) | $($row.MeanMs) | $($row.MedianMs) | $($row.StdDevMs) | $speedup | $($row.CoverageState) |")
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
