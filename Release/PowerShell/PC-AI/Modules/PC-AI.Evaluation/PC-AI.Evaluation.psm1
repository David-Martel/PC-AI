#Requires -Version 7.0
<#
.SYNOPSIS
    PC-AI LLM Evaluation Framework

.DESCRIPTION
    Comprehensive evaluation suite for testing PC-AI inference backends:
    - pcai-inference (Rust FFI with llama.cpp/mistral.rs)
    - Ollama HTTP backend
    - OpenAI-compatible APIs

    Supports automated metrics, LLM-as-judge patterns, regression testing,
    and A/B testing between backends.

.NOTES
    Version: 1.0.0
    Author: PC-AI Team
#>

using namespace System.Diagnostics
using namespace System.Collections.Generic

#region Module State

$script:EvaluationConfig = @{
    DefaultMetrics = @('latency', 'throughput', 'memory')
    JudgeModel = 'claude-sonnet-4-5'
    JudgeProvider = 'anthropic'
    BaselinePath = Join-Path $PSScriptRoot 'Baselines'
    DatasetPath = Join-Path $PSScriptRoot 'Datasets'
    ResultsPath = $null
    ArtifactsRoot = $null
    EvaluationRoot = $null
    RunRoot = $null
    HttpBaseUrl = 'http://127.0.0.1:8080'
    OllamaBaseUrl = 'http://127.0.0.1:11434'
    OllamaModel = 'llama3.2'
    ProgressMode = 'auto'
    EmitStructuredMessages = $false
    ProgressLogPath = $null
    EventsLogPath = $null
    StopSignalPath = $null
    HeartbeatSeconds = 15
    ProgressIntervalSeconds = 2
}

$script:CurrentSuite = $null
$script:ABTests = @{}
$script:Baselines = @{}
$script:CompiledServerProcess = $null
$script:CompiledServerConfigPath = $null
$script:CompiledServerBaseUrl = $null
$script:CompiledServerBackend = $null
$script:EvaluationRunState = $null

#endregion

#region Classes
# Classes MUST remain inline in the .psm1 — PowerShell cannot dot-source class definitions.

class EvaluationMetric {
    [string]$Name
    [string]$Description
    [scriptblock]$Calculator
    [double]$Weight = 1.0

    EvaluationMetric([string]$name, [string]$desc, [scriptblock]$calc) {
        $this.Name = $name
        $this.Description = $desc
        $this.Calculator = $calc
    }
}

class EvaluationTestCase {
    [string]$Id
    [string]$Category
    [string]$Prompt
    [string]$ExpectedOutput
    [hashtable]$Context = @{}
    [string[]]$Tags = @()
    [hashtable]$Metadata = @{}
}

class EvaluationResult {
    [string]$TestCaseId
    [string]$Backend
    [string]$Model
    [datetime]$Timestamp
    [string]$Prompt
    [string]$Response
    [hashtable]$Metrics = @{}
    [double]$OverallScore
    [string]$Status  # 'pass', 'fail', 'error'
    [string]$ErrorMessage
    [timespan]$Duration
}

class EvaluationSuite {
    [string]$Name
    [string]$Description
    [EvaluationMetric[]]$Metrics = @()
    [EvaluationTestCase[]]$TestCases = @()
    [hashtable]$Config = @{}
    [List[EvaluationResult]]$Results = [List[EvaluationResult]]::new()

    [void] AddMetric([EvaluationMetric]$metric) {
        $this.Metrics += $metric
    }

    [void] AddTestCase([EvaluationTestCase]$testCase) {
        $this.TestCases += $testCase
    }

    [hashtable] GetSummary() {
        $passed = ($this.Results | Where-Object Status -eq 'pass').Count
        $failed = ($this.Results | Where-Object Status -eq 'fail').Count
        $errors = ($this.Results | Where-Object Status -eq 'error').Count

        return @{
            TotalTests = $this.Results.Count
            Passed = $passed
            Failed = $failed
            Errors = $errors
            PassRate = if ($this.Results.Count -gt 0) { [math]::Round($passed / $this.Results.Count * 100, 2) } else { 0 }
            AverageScore = if ($this.Results.Count -gt 0) {
                [math]::Round(($this.Results | Measure-Object -Property OverallScore -Average).Average, 4)
            } else { 0 }
            AverageLatency = if ($this.Results.Count -gt 0) {
                ($this.Results | Measure-Object -Property { $_.Duration.TotalMilliseconds } -Average).Average
            } else { 0 }
        }
    }
}

#endregion

$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object { . $_.FullName }
}
$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object { . $_.FullName }
}

Export-ModuleMember -Function *
