#Requires -Version 7.0
<#
.SYNOPSIS
    PC-AI Inference Backend Evaluation Runner

.DESCRIPTION
    Comprehensive evaluation script for testing PC-AI inference backends:
    - pcai-inference (llama.cpp + mistral.rs)
    - HTTP/Ollama backends
    - A/B comparison between backends

.PARAMETER Backend
    Inference backend to evaluate: llamacpp, mistralrs, http, ollama, all

.PARAMETER ModelPath
    Path to GGUF model file (required for native backends)

.PARAMETER Dataset
    Evaluation dataset: diagnostic, general, safety, or path to custom JSON

.PARAMETER CreateBaseline
    Create a baseline snapshot with the given name

.PARAMETER CompareBaseline
    Compare against an existing baseline

.PARAMETER ABTest
    Run A/B test between two backends (e.g., "llamacpp:mistralrs")

.PARAMETER OutputPath
    Path to save evaluation results

.PARAMETER OutputRoot
    Root folder for evaluation run outputs (defaults to .pcai/evaluation/runs)

.PARAMETER RunLabel
    Optional label for evaluation run folders

.PARAMETER ProgressMode
    Progress output mode: auto, stream, bar, silent

.PARAMETER EmitStructuredMessages
    Emit JSON event lines to the pipeline

.PARAMETER HeartbeatSeconds
    Heartbeat interval for long-running runs

.PARAMETER RequestTimeoutSec
    Timeout for HTTP requests per test case

.PARAMETER StopSignalPath
    Stop signal file path; if present, evaluation will stop gracefully

.EXAMPLE
    .\Invoke-InferenceEvaluation.ps1 -Backend llamacpp -ModelPath "C:\models\llama-3.2-1b.gguf" -Dataset diagnostic

.EXAMPLE
    .\Invoke-InferenceEvaluation.ps1 -ABTest "llamacpp:mistralrs" -ModelPath "C:\models\model.gguf"

.EXAMPLE
    .\Invoke-InferenceEvaluation.ps1 -Backend llamacpp -CreateBaseline "v1.0.0-baseline"
#>
[CmdletBinding()]
param(
    [ValidateSet('llamacpp', 'mistralrs', 'llamacpp-bin', 'mistralrs-bin', 'http', 'ollama', 'all')]
    [string]$Backend = 'mistralrs',

    [string]$ModelPath,

    [string]$Dataset = 'diagnostic',

    [string]$CreateBaseline,

    [string]$CompareBaseline,

    [string]$ABTest,

    [string]$OutputPath,

    [string]$OutputRoot,

    [string]$RunLabel,

    [int]$MaxTokens = 512,

    [float]$Temperature = 0.7,

    [int]$GpuLayers = -1,

    [string]$BaseUrl,

    [int]$MaxTestCases = 0,

    [ValidateSet('auto', 'stream', 'bar', 'silent')]
    [string]$ProgressMode = 'auto',

    [switch]$EmitStructuredMessages,

    [int]$HeartbeatSeconds = 15,

    [int]$RequestTimeoutSec = 120,

    [string]$StopSignalPath,

    [switch]$ExerciseCSharpFfi
)

$ErrorActionPreference = 'Stop'

if (-not $PSBoundParameters.ContainsKey('ExerciseCSharpFfi')) {
    $ExerciseCSharpFfi = $true
}

# Import required modules
$scriptRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $scriptRoot
$configPath = Join-Path $projectRoot 'Config\llm-config.json'
$script:LlmConfig = $null
if (Test-Path $configPath) {
    try {
        $script:LlmConfig = Get-Content $configPath -Raw | ConvertFrom-Json
    } catch {
        Write-Verbose "Failed to parse $configPath: $_"
    }
}

$moduleRoot = Join-Path $projectRoot 'Modules'
if ($env:PSModulePath -notlike "*$moduleRoot*") {
    $env:PSModulePath = "$moduleRoot;$env:PSModulePath"
}

if (Test-Path (Join-Path $projectRoot 'Modules\PcaiInference.psd1')) {
    Import-Module (Join-Path $projectRoot 'Modules\PcaiInference.psd1') -Force -ErrorAction SilentlyContinue
} else {
    Import-Module (Join-Path $projectRoot 'Modules\PcaiInference.psm1') -Force -ErrorAction SilentlyContinue
}
Import-Module (Join-Path $projectRoot 'Modules\PC-AI.Evaluation\PC-AI.Evaluation.psd1') -Force

Write-Host @'

╔══════════════════════════════════════════════════════════════╗
║            PC-AI Inference Evaluation Framework              ║
╚══════════════════════════════════════════════════════════════╝

'@ -ForegroundColor Cyan

if (-not $OutputRoot) {
    $OutputRoot = Join-Path (Get-PcaiArtifactsRoot) 'evaluation\runs'
}
if (-not (Test-Path $OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
}
if (-not $OutputPath) {
    $batchLabel = if ($RunLabel) { "$RunLabel-summary" } else { 'batch-summary' }
    $batchContext = New-PcaiEvaluationRunContext -RunLabel $batchLabel -OutputRoot $OutputRoot -SuiteName 'BatchSummary' -Backend 'batch'
    $OutputPath = Join-Path $batchContext.RunDir 'evaluation_summary.json'
}
Write-Host "Output Root: $OutputRoot" -ForegroundColor DarkGray
Write-Host "Summary Output: $OutputPath" -ForegroundColor DarkGray

#region Helper Functions

function Write-Section {
    param([string]$Title)
    Write-Host "`n═══ $Title ═══" -ForegroundColor Yellow
}

function Format-Duration {
    param([timespan]$Duration)
    if ($Duration.TotalHours -ge 1) {
        return '{0:N0}h {1:N0}m {2:N0}s' -f $Duration.Hours, $Duration.Minutes, $Duration.Seconds
    } elseif ($Duration.TotalMinutes -ge 1) {
        return '{0:N0}m {1:N1}s' -f $Duration.Minutes, $Duration.Seconds
    } else {
        return '{0:N2}s' -f $Duration.TotalSeconds
    }
}

function Test-BackendAvailable {
    param(
        [string]$Backend,
        [string]$BaseUrl
    )

    switch ($Backend) {
        { $_ -in 'llamacpp', 'mistralrs' } {
            # Check if DLL is available
            $dllCandidates = @()
            if ($script:LlmConfig -and $script:LlmConfig.nativeInference -and $script:LlmConfig.nativeInference.dllSearchPaths) {
                foreach ($path in $script:LlmConfig.nativeInference.dllSearchPaths) {
                    if (-not $path) { continue }
                    if ([System.IO.Path]::IsPathRooted($path)) {
                        $dllCandidates += $path
                    } else {
                        $dllCandidates += (Join-Path $projectRoot $path)
                    }
                }
            }
            $dllCandidates += @(
                (Join-Path $projectRoot 'bin\Release\pcai_inference.dll'),
                (Join-Path $projectRoot 'bin\Debug\pcai_inference.dll')
            )
            return ($dllCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1) -ne $null
        }
        { $_ -in 'llamacpp-bin', 'mistralrs-bin' } {
            $binaryName = if ($Backend -eq 'llamacpp-bin') { 'pcai-llamacpp.exe' } else { 'pcai-mistralrs.exe' }
            $candidateDirs = @()
            if ($script:LlmConfig -and $script:LlmConfig.evaluation -and $script:LlmConfig.evaluation.binSearchPaths) {
                foreach ($path in $script:LlmConfig.evaluation.binSearchPaths) {
                    if (-not $path) { continue }
                    if ([System.IO.Path]::IsPathRooted($path)) {
                        $candidateDirs += $path
                    } else {
                        $candidateDirs += (Join-Path $projectRoot $path)
                    }
                }
            }
            $userProfile = [Environment]::GetFolderPath('UserProfile')
            $candidateDirs += @(
                (Join-Path $userProfile '.local\bin'),
                'T:\RustCache\cargo-target\release'
            ) | Where-Object { $_ }

            foreach ($dir in $candidateDirs) {
                if (Test-Path (Join-Path $dir $binaryName)) {
                    return $true
                }
            }
            return $false
        }
        'http' {
            try {
                $target = if ($BaseUrl) { $BaseUrl } elseif ($script:LlmConfig.providers.'pcai-inference'.baseUrl) { $script:LlmConfig.providers.'pcai-inference'.baseUrl } else { throw 'No BaseUrl' }
                $null = Invoke-RestMethod -Uri "$target/health" -TimeoutSec 2
                return $true
            } catch { return $false }
        }
        'ollama' {
            try {
                $target = if ($BaseUrl) { $BaseUrl } elseif ($script:LlmConfig.providers.ollama.baseUrl) { $script:LlmConfig.providers.ollama.baseUrl } else { throw 'No BaseUrl' }
                $null = Invoke-RestMethod -Uri "$target/api/tags" -TimeoutSec 2
                return $true
            } catch { return $false }
        }
    }
    return $false
}

function Get-ServiceHostPath {
    $candidates = @(
        (Join-Path $projectRoot 'Native\PcaiServiceHost\bin\Release\net8.0\win-x64\PcaiServiceHost.dll'),
        (Join-Path $projectRoot 'Native\PcaiServiceHost\bin\Release\net8.0\PcaiServiceHost.dll'),
        (Join-Path $env:USERPROFILE 'PC_AI\Native\PcaiServiceHost\bin\Release\net8.0\win-x64\PcaiServiceHost.dll')
    )
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Invoke-CSharpInferenceSmoke {
    param(
        [string]$Backend,
        [string]$ModelPath,
        [int]$GpuLayers
    )

    $hostPath = Get-ServiceHostPath
    if (-not $hostPath) {
        Write-Warning 'PcaiServiceHost.dll not found. Skipping C# inference smoke.'
        return
    }

    $dotnet = (Get-Command dotnet -ErrorAction SilentlyContinue)?.Source
    if (-not $dotnet) {
        Write-Warning 'dotnet not found. Skipping C# inference smoke.'
        return
    }

    Write-Host '  C# FFI smoke: status' -ForegroundColor DarkGray
    & $dotnet $hostPath inference status | Out-Host

    Write-Host "  C# FFI smoke: init $Backend" -ForegroundColor DarkGray
    & $dotnet $hostPath inference init --backend $Backend | Out-Host

    if ($ModelPath) {
        Write-Host '  C# FFI smoke: load model' -ForegroundColor DarkGray
        & $dotnet $hostPath inference load --model-path $ModelPath --gpu-layers $GpuLayers | Out-Host

        Write-Host '  C# FFI smoke: generate' -ForegroundColor DarkGray
        & $dotnet $hostPath inference generate --prompt 'Health check: summarize system readiness.' | Out-Host
    }
}

#endregion

#region Main Evaluation Logic

function Invoke-BackendEvaluation {
    param(
        [string]$Backend,
        [string]$ModelPath,
        [string]$Dataset,
        [int]$MaxTokens,
        [float]$Temperature,
        [int]$GpuLayers,
        [string]$BaseUrl,
        [int]$MaxTestCases,
        [string]$OutputRoot,
        [string]$RunLabel,
        [string]$ProgressMode,
        [switch]$EmitStructuredMessages,
        [int]$HeartbeatSeconds,
        [int]$RequestTimeoutSec,
        [string]$StopSignalPath,
        [switch]$ExerciseCSharpFfi
    )

    Write-Section "Evaluating Backend: $Backend"

    # Check availability
    if (-not (Test-BackendAvailable -Backend $Backend -BaseUrl $BaseUrl)) {
        Write-Warning "Backend '$Backend' is not available. Skipping."
        return $null
    }

    if ($Backend -in @('llamacpp', 'mistralrs', 'llamacpp-bin', 'mistralrs-bin') -and -not $ModelPath) {
        Write-Error "ModelPath is required for backend '$Backend'"
        return $null
    }

    # Create evaluation suite
    $suite = New-EvaluationSuite -Name "PC-AI-$Backend-Eval" `
        -Description "Evaluation of $Backend backend for PC-AI diagnostics" `
        -Metrics @('latency', 'throughput', 'similarity', 'coherence') `
        -IncludeDefaultMetrics

    # Load test cases
    $testCases = Get-EvaluationDataset -Name $Dataset
    if (-not $testCases) {
        Write-Error "Failed to load dataset: $Dataset"
        return $null
    }

    if ($MaxTestCases -gt 0) {
        $testCases = $testCases | Select-Object -First $MaxTestCases
    }

    Write-Host "  Loaded $($testCases.Count) test cases from '$Dataset'" -ForegroundColor Gray

    # Add test cases to suite
    foreach ($tc in $testCases) {
        $suite.AddTestCase($tc)
    }

    # Run evaluation
    $effectiveBaseUrl = $BaseUrl
    if (-not $effectiveBaseUrl -and $script:LlmConfig) {
        $effectiveBaseUrl = if ($Backend -eq 'ollama') { $script:LlmConfig.providers.ollama.baseUrl } else { $script:LlmConfig.providers.'pcai-inference'.baseUrl }
    }

    $runLabelEffective = if ($RunLabel) { "$RunLabel-$Backend-$Dataset" } else { "$Backend-$Dataset" }

    if ($ExerciseCSharpFfi -and $Backend -in @('llamacpp', 'mistralrs')) {
        Invoke-CSharpInferenceSmoke -Backend $Backend -ModelPath $ModelPath -GpuLayers $GpuLayers
    }

    $results = Invoke-EvaluationSuite -Suite $suite `
        -Backend $Backend `
        -ModelPath $ModelPath `
        -BaseUrl $effectiveBaseUrl `
        -MaxTokens $MaxTokens `
        -Temperature $Temperature `
        -GpuLayers $GpuLayers `
        -RunLabel $runLabelEffective `
        -OutputRoot $OutputRoot `
        -ProgressMode $ProgressMode `
        -EmitStructuredMessages:$EmitStructuredMessages `
        -HeartbeatSeconds $HeartbeatSeconds `
        -RequestTimeoutSec $RequestTimeoutSec `
        -StopSignalPath $StopSignalPath

    return @{
        Backend = $Backend
        Suite   = $suite
        Results = $results
    }
}

function Invoke-ABTestEvaluation {
    param(
        [string]$VariantSpec,
        [string]$ModelPath,
        [string]$Dataset,
        [int]$MaxTokens,
        [float]$Temperature
    )

    $variants = $VariantSpec -split ':'
    if ($variants.Count -ne 2) {
        Write-Error "A/B test requires two variants in format 'backendA:backendB'"
        return
    }

    $variantA, $variantB = $variants

    Write-Section "A/B Test: $variantA vs $variantB"

    if ($variantA -match '-bin' -or $variantB -match '-bin') {
        Write-Error 'A/B testing currently supports only FFI backends (llamacpp, mistralrs).'
        return
    }

    # Check both backends available
    foreach ($v in $variants) {
        if (-not (Test-BackendAvailable -Backend $v -BaseUrl $null)) {
            Write-Error "Backend '$v' is not available for A/B test"
            return
        }
    }

    # Create A/B test
    $abTest = New-ABTest -Name "pcai-ab-$variantA-vs-$variantB" `
        -VariantAName $variantA `
        -VariantBName $variantB

    # Load test cases
    $testCases = Get-EvaluationDataset -Name $Dataset

    Write-Host "  Running $($testCases.Count) test cases on both variants..." -ForegroundColor Gray

    foreach ($tc in $testCases) {
        Write-Host "    Testing: $($tc.Id)" -ForegroundColor DarkGray

        # Test Variant A
        try {
            $initA = Initialize-PcaiInference -Backend $variantA
            if ($ModelPath) { $null = Import-PcaiModel -ModelPath $ModelPath }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $responseA = Invoke-PcaiGenerate -Prompt $tc.Prompt -MaxTokens $MaxTokens -Temperature $Temperature
            $sw.Stop()

            $scoreA = if ($tc.ExpectedOutput) {
                Compare-ResponseSimilarity -Response $responseA -Expected $tc.ExpectedOutput
            } else {
                Measure-Coherence -Response $responseA
            }

            Add-ABTestResult -TestName $abTest.Name -Variant 'A' -Score $scoreA
            Close-PcaiInference
        } catch {
            Write-Warning "Variant A failed on $($tc.Id): $_"
        }

        # Test Variant B
        try {
            $initB = Initialize-PcaiInference -Backend $variantB
            if ($ModelPath) { $null = Import-PcaiModel -ModelPath $ModelPath }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $responseB = Invoke-PcaiGenerate -Prompt $tc.Prompt -MaxTokens $MaxTokens -Temperature $Temperature
            $sw.Stop()

            $scoreB = if ($tc.ExpectedOutput) {
                Compare-ResponseSimilarity -Response $responseB -Expected $tc.ExpectedOutput
            } else {
                Measure-Coherence -Response $responseB
            }

            Add-ABTestResult -TestName $abTest.Name -Variant 'B' -Score $scoreB
            Close-PcaiInference
        } catch {
            Write-Warning "Variant B failed on $($tc.Id): $_"
        }
    }

    # Analyze results
    $analysis = Get-ABTestAnalysis -TestName $abTest.Name

    return $analysis
}

#endregion

#region Main Execution

$startTime = [datetime]::UtcNow
$allResults = @{}

try {
    # A/B Test Mode
    if ($ABTest) {
        $abResults = Invoke-ABTestEvaluation -VariantSpec $ABTest `
            -ModelPath $ModelPath `
            -Dataset $Dataset `
            -MaxTokens $MaxTokens `
            -Temperature $Temperature

        $allResults['ABTest'] = $abResults
    }
    # Single/Multi Backend Mode
    else {
        $backends = if ($Backend -eq 'all') {
            @('mistralrs', 'llamacpp', 'http', 'ollama')
        } else {
            @($Backend)
        }

        foreach ($be in $backends) {
            $evalResult = Invoke-BackendEvaluation -Backend $be `
                -ModelPath $ModelPath `
                -Dataset $Dataset `
                -MaxTokens $MaxTokens `
                -Temperature $Temperature `
                -GpuLayers $GpuLayers `
                -BaseUrl $BaseUrl `
                -MaxTestCases $MaxTestCases `
                -OutputRoot $OutputRoot `
                -RunLabel $RunLabel `
                -ProgressMode $ProgressMode `
                -EmitStructuredMessages:$EmitStructuredMessages `
                -HeartbeatSeconds $HeartbeatSeconds `
                -RequestTimeoutSec $RequestTimeoutSec `
                -StopSignalPath $StopSignalPath `
                -ExerciseCSharpFfi:$ExerciseCSharpFfi

            if ($evalResult) {
                $allResults[$be] = $evalResult

                # Create baseline if requested
                if ($CreateBaseline -and $evalResult.Suite) {
                    Write-Section "Creating Baseline: $CreateBaseline"
                    $baseline = New-BaselineSnapshot -Name "$CreateBaseline-$be" `
                        -Suite $evalResult.Suite `
                        -Backend $be `
                        -ModelPath $ModelPath
                }

                # Compare to baseline if requested
                if ($CompareBaseline -and $evalResult.Suite) {
                    Write-Section "Comparing to Baseline: $CompareBaseline"
                    $regression = Test-ForRegression -BaselineName "$CompareBaseline-$be" `
                        -Suite $evalResult.Suite
                    $allResults["${be}_regression"] = $regression
                }
            }
        }
    }

    # Generate summary
    $endTime = [datetime]::UtcNow
    $totalDuration = $endTime - $startTime

    Write-Section 'Evaluation Summary'

    foreach ($key in $allResults.Keys) {
        $result = $allResults[$key]

        if ($result.Results) {
            Write-Host ("`n  {0}:" -f $key) -ForegroundColor White
            Write-Host "    Pass Rate: $($result.Results.PassRate)%"
            Write-Host "    Avg Score: $($result.Results.AverageScore)"
            Write-Host "    Avg Latency: $([math]::Round($result.Results.AverageLatency, 2))ms"
        } elseif ($result.Winner) {
            Write-Host "`n  A/B Test Result:" -ForegroundColor White
            Write-Host "    Winner: $($result.Winner)"
            Write-Host "    Effect Size: $($result.EffectSize) (d=$($result.CohensD))"
            Write-Host "    Significant: $($result.StatisticallySignificant)"
        }
    }

    Write-Host "`n  Total Duration: $(Format-Duration $totalDuration)" -ForegroundColor Gray

    # Save results if output path specified
    if ($OutputPath) {
        $outputDir = Split-Path $OutputPath -Parent
        if ($outputDir -and -not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }

        $exportData = @{
            Timestamp  = $startTime.ToString('o')
            Duration   = $totalDuration.ToString()
            Parameters = @{
                Backend     = $Backend
                ModelPath   = $ModelPath
                Dataset     = $Dataset
                MaxTokens   = $MaxTokens
                Temperature = $Temperature
            }
            Results    = $allResults.GetEnumerator() | ForEach-Object {
                @{
                    Key   = $_.Key
                    Value = if ($_.Value.Results) { $_.Value.Results } else { $_.Value }
                }
            }
        }

        $exportData | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputPath
        Write-Host "`n  Results saved: $OutputPath" -ForegroundColor Green
    }

} catch {
    Write-Error "Evaluation failed: $_"
    throw
}

#endregion

Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host '║                    Evaluation Complete                       ║' -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan
