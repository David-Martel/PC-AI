#Requires -Version 5.1

<#
.SYNOPSIS
    Router eval harness for FunctionGemma against a running runtime instance.

.DESCRIPTION
    Sends test scenarios from scenarios.json to the FunctionGemma runtime's
    OpenAI-compatible /v1/chat/completions endpoint and evaluates response
    accuracy across three dimensions:

    - Tool call accuracy: does the response contain the correct tool name?
    - Argument accuracy: do the returned arguments match expected values?
    - NO_TOOL accuracy: does the model correctly return NO_TOOL for chat inputs?

    Reports per-category accuracy, overall accuracy, and latency statistics.

    This script complements Tools/run-functiongemma-eval.ps1, which runs the
    Rust-native eval binary. This harness instead exercises the live HTTP API,
    making it suitable for integration testing and runtime regression checks.

.PARAMETER Endpoint
    Base URL of the FunctionGemma runtime (default: http://127.0.0.1:8000).

.PARAMETER ScenariosPath
    Path to the scenarios JSON file
    (default: Deploy/rust-functiongemma-train/examples/scenarios.json).

.PARAMETER MaxScenarios
    Maximum number of scenarios to evaluate. 0 means all.

.PARAMETER MaxTokens
    Maximum tokens per completion request.

.PARAMETER Temperature
    Sampling temperature (low values for deterministic routing).

.PARAMETER Seed
    Seed for deterministic generation.

.PARAMETER OutputFormat
    Display format: Table, Json, or Markdown.

.PARAMETER ReportPath
    Optional path to write the structured JSON results file.

.EXAMPLE
    .\Tools\Invoke-FunctionGemmaEval.ps1
    # Runs all scenarios against localhost:8000, outputs a table.

.EXAMPLE
    .\Tools\Invoke-FunctionGemmaEval.ps1 -MaxScenarios 5 -OutputFormat Json
    # Runs 5 scenarios and outputs structured JSON.

.EXAMPLE
    .\Tools\Invoke-FunctionGemmaEval.ps1 -ReportPath Reports/fg-eval.json -OutputFormat Markdown
    # Runs all scenarios, saves JSON report, displays markdown summary.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Endpoint = 'http://127.0.0.1:8000',
    [string]$ScenariosPath = 'Deploy/rust-functiongemma-train/examples/scenarios.json',
    [int]$MaxScenarios = 0,  # 0 = all
    [int]$MaxTokens = 64,
    [double]$Temperature = 0.1,
    [int]$Seed = 42,
    [ValidateSet('Table','Json','Markdown')]
    [string]$OutputFormat = 'Table',
    [string]$ReportPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not [System.IO.Path]::IsPathRooted($ScenariosPath)) {
    $ScenariosPath = Join-Path $repoRoot $ScenariosPath
}
if (-not (Test-Path $ScenariosPath)) {
    throw "Scenarios file not found: $ScenariosPath"
}

$toolsPath = Join-Path $repoRoot 'Config/pcai-tools.json'
$toolsPayload = $null
if (Test-Path $toolsPath) {
    $toolsPayload = (Get-Content $toolsPath -Raw | ConvertFrom-Json).tools
}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
Write-Host "Checking FunctionGemma runtime at $Endpoint ..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$Endpoint/health" -Method Get -TimeoutSec 5
    Write-Host "Runtime healthy: engine=$($health.metadata.engine), model=$($health.metadata.model)" -ForegroundColor Green
} catch {
    Write-Error "FunctionGemma runtime is not reachable at $Endpoint. Start it first.`n$($_.Exception.Message)"
    return
}

# ---------------------------------------------------------------------------
# Load scenarios
# ---------------------------------------------------------------------------
$scenariosData = Get-Content $ScenariosPath -Raw | ConvertFrom-Json
$scenarios = $scenariosData.scenarios

if ($MaxScenarios -gt 0 -and $MaxScenarios -lt $scenarios.Count) {
    $scenarios = $scenarios[0..($MaxScenarios - 1)]
}

Write-Host "Evaluating $($scenarios.Count) scenarios (seed=$Seed, temp=$Temperature, max_tokens=$MaxTokens)" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------
$results = [System.Collections.ArrayList]::new()
$latencies = [System.Collections.ArrayList]::new()

foreach ($scenario in $scenarios) {
    $isChat = ($null -ne $scenario.assistant_content -and $scenario.assistant_content -eq 'NO_TOOL')
    $expectedTool = if ($isChat) { $null } else { $scenario.tool_name }

    # Build the request body
    $messages = @(
        @{ role = 'user'; content = $scenario.user_content }
    )

    $body = @{
        model      = 'functiongemma'
        messages   = $messages
        max_tokens = $MaxTokens
        temperature = $Temperature
        seed       = $Seed
    }
    if ($toolsPayload) {
        $body['tools'] = $toolsPayload
    }

    $jsonBody = $body | ConvertTo-Json -Depth 10 -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-RestMethod `
            -Uri "$Endpoint/v1/chat/completions" `
            -Method Post `
            -ContentType 'application/json' `
            -Body $jsonBody `
            -TimeoutSec 30
    } catch {
        $sw.Stop()
        $null = $results.Add([PSCustomObject]@{
            Input           = $scenario.user_content.Substring(0, [Math]::Min(60, $scenario.user_content.Length))
            Mode            = $scenario.mode
            ExpectedTool    = if ($isChat) { 'NO_TOOL' } else { $expectedTool }
            ActualTool      = 'ERROR'
            ToolCorrect     = $false
            ArgsCorrect     = $false
            LatencyMs       = $sw.ElapsedMilliseconds
            Error           = $_.Exception.Message
        })
        continue
    }
    $sw.Stop()
    $latencyMs = $sw.ElapsedMilliseconds
    $null = $latencies.Add($latencyMs)

    $choice = $response.choices[0]
    $msg = $choice.message

    # Extract actual tool name from response
    $actualTool = $null
    $actualArgs = $null
    $gotNoTool = $false

    if ($msg.tool_calls -and $msg.tool_calls.Count -gt 0) {
        $tc = $msg.tool_calls[0]
        $actualTool = $tc.function.name
        # Parse arguments -- may be string or object
        if ($tc.function.arguments -is [string]) {
            try { $actualArgs = $tc.function.arguments | ConvertFrom-Json } catch { $actualArgs = $null }
        } else {
            $actualArgs = $tc.function.arguments
        }
    } elseif ($msg.content -and $msg.content -match 'NO_TOOL') {
        $gotNoTool = $true
    } elseif ($msg.content) {
        # Check if the content contains a function call pattern
        if ($msg.content -match '(\w+)\s*\(') {
            $actualTool = $Matches[1]
        }
    }

    # Evaluate tool call accuracy
    $toolCorrect = $false
    if ($isChat) {
        $toolCorrect = $gotNoTool -or ($null -eq $msg.tool_calls -or $msg.tool_calls.Count -eq 0)
    } else {
        $toolCorrect = ($actualTool -eq $expectedTool)
    }

    # Evaluate argument accuracy (exact match on expected keys)
    $argsCorrect = $false
    if ($isChat) {
        # For NO_TOOL scenarios, args are correct if no tool was called
        $argsCorrect = $toolCorrect
    } elseif ($toolCorrect -and $null -ne $actualArgs -and $null -ne $scenario.tool_arguments) {
        $expectedArgs = $scenario.tool_arguments
        $argsCorrect = $true
        foreach ($prop in $expectedArgs.PSObject.Properties) {
            $expectedVal = "$($prop.Value)"
            $actualVal = "$($actualArgs.$($prop.Name))"
            if ($expectedVal -ne $actualVal) {
                $argsCorrect = $false
                break
            }
        }
    }

    $displayTool = if ($isChat) { 'NO_TOOL' } else { $expectedTool }
    $displayActual = if ($gotNoTool) { 'NO_TOOL' } elseif ($actualTool) { $actualTool } else { '(none)' }

    $null = $results.Add([PSCustomObject]@{
        Input           = $scenario.user_content.Substring(0, [Math]::Min(60, $scenario.user_content.Length))
        Mode            = $scenario.mode
        ExpectedTool    = $displayTool
        ActualTool      = $displayActual
        ToolCorrect     = $toolCorrect
        ArgsCorrect     = $argsCorrect
        LatencyMs       = $latencyMs
        Error           = $null
    })
}

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
$total = $results.Count
$toolCorrectCount = ($results | Where-Object { $_.ToolCorrect }).Count
$argsCorrectCount = ($results | Where-Object { $_.ArgsCorrect }).Count
$errorCount = ($results | Where-Object { $_.Error }).Count

$toolAccuracy = if ($total -gt 0) { [Math]::Round($toolCorrectCount / $total * 100, 1) } else { 0 }
$argsAccuracy = if ($total -gt 0) { [Math]::Round($argsCorrectCount / $total * 100, 1) } else { 0 }

# Per-mode accuracy
$modes = $results | Group-Object Mode
$modeMetrics = @{}
foreach ($group in $modes) {
    $modeTotal = $group.Count
    $modeToolOk = ($group.Group | Where-Object { $_.ToolCorrect }).Count
    $modeArgsOk = ($group.Group | Where-Object { $_.ArgsCorrect }).Count
    $modeMetrics[$group.Name] = @{
        total          = $modeTotal
        tool_accuracy  = if ($modeTotal -gt 0) { [Math]::Round($modeToolOk / $modeTotal * 100, 1) } else { 0 }
        args_accuracy  = if ($modeTotal -gt 0) { [Math]::Round($modeArgsOk / $modeTotal * 100, 1) } else { 0 }
    }
}

# Latency stats
$latencyStats = @{
    count  = $latencies.Count
    min_ms = 0
    max_ms = 0
    avg_ms = 0
    p50_ms = 0
    p95_ms = 0
}
if ($latencies.Count -gt 0) {
    $sorted = $latencies | Sort-Object
    $latencyStats.min_ms = $sorted[0]
    $latencyStats.max_ms = $sorted[-1]
    $latencyStats.avg_ms = [Math]::Round(($sorted | Measure-Object -Average).Average, 1)
    $p50Idx = [Math]::Floor($sorted.Count * 0.50)
    $p95Idx = [Math]::Min([Math]::Floor($sorted.Count * 0.95), $sorted.Count - 1)
    $latencyStats.p50_ms = $sorted[$p50Idx]
    $latencyStats.p95_ms = $sorted[$p95Idx]
}

# ---------------------------------------------------------------------------
# Build structured report
# ---------------------------------------------------------------------------
$report = [ordered]@{
    timestamp       = (Get-Date -Format 'o')
    endpoint        = $Endpoint
    scenarios_file  = $ScenariosPath
    settings        = [ordered]@{
        max_tokens  = $MaxTokens
        temperature = $Temperature
        seed        = $Seed
    }
    summary         = [ordered]@{
        total            = $total
        tool_correct     = $toolCorrectCount
        args_correct     = $argsCorrectCount
        errors           = $errorCount
        tool_accuracy_pct = $toolAccuracy
        args_accuracy_pct = $argsAccuracy
    }
    per_mode        = $modeMetrics
    latency         = $latencyStats
    details         = @($results | ForEach-Object {
        [ordered]@{
            input         = $_.Input
            mode          = $_.Mode
            expected_tool = $_.ExpectedTool
            actual_tool   = $_.ActualTool
            tool_correct  = $_.ToolCorrect
            args_correct  = $_.ArgsCorrect
            latency_ms    = $_.LatencyMs
            error         = $_.Error
        }
    })
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
switch ($OutputFormat) {
    'Json' {
        $report | ConvertTo-Json -Depth 10
    }
    'Markdown' {
        Write-Host ''
        Write-Host "## FunctionGemma Router Eval Results" -ForegroundColor Cyan
        Write-Host ''
        Write-Host "| Metric | Value |"
        Write-Host "|--------|-------|"
        Write-Host "| Total scenarios | $total |"
        Write-Host "| Tool accuracy | ${toolAccuracy}% ($toolCorrectCount/$total) |"
        Write-Host "| Argument accuracy | ${argsAccuracy}% ($argsCorrectCount/$total) |"
        Write-Host "| Errors | $errorCount |"
        Write-Host ''
        Write-Host "### Per-Mode Breakdown" -ForegroundColor Cyan
        Write-Host ''
        Write-Host "| Mode | Count | Tool Acc | Args Acc |"
        Write-Host "|------|-------|----------|----------|"
        foreach ($mode in $modeMetrics.Keys) {
            $m = $modeMetrics[$mode]
            Write-Host "| $mode | $($m.total) | $($m.tool_accuracy)% | $($m.args_accuracy)% |"
        }
        Write-Host ''
        Write-Host "### Latency" -ForegroundColor Cyan
        Write-Host ''
        Write-Host "| Stat | ms |"
        Write-Host "|------|-----|"
        Write-Host "| Min | $($latencyStats.min_ms) |"
        Write-Host "| Avg | $($latencyStats.avg_ms) |"
        Write-Host "| P50 | $($latencyStats.p50_ms) |"
        Write-Host "| P95 | $($latencyStats.p95_ms) |"
        Write-Host "| Max | $($latencyStats.max_ms) |"
    }
    default {
        # Table
        Write-Host ''
        Write-Host "=== FunctionGemma Router Eval ===" -ForegroundColor Cyan
        Write-Host "Scenarios: $total | Tool Accuracy: ${toolAccuracy}% | Arg Accuracy: ${argsAccuracy}% | Errors: $errorCount"
        Write-Host ''

        $results | Format-Table -AutoSize -Property Mode, ExpectedTool, ActualTool, ToolCorrect, ArgsCorrect, LatencyMs

        Write-Host "--- Per-Mode ---" -ForegroundColor Yellow
        foreach ($mode in $modeMetrics.Keys) {
            $m = $modeMetrics[$mode]
            Write-Host "  $mode : $($m.total) scenarios, tool=$($m.tool_accuracy)%, args=$($m.args_accuracy)%"
        }

        Write-Host ''
        Write-Host "--- Latency ---" -ForegroundColor Yellow
        Write-Host "  Min=$($latencyStats.min_ms)ms  Avg=$($latencyStats.avg_ms)ms  P50=$($latencyStats.p50_ms)ms  P95=$($latencyStats.p95_ms)ms  Max=$($latencyStats.max_ms)ms"
    }
}

# ---------------------------------------------------------------------------
# Optionally write JSON report to disk
# ---------------------------------------------------------------------------
if ($ReportPath) {
    if (-not [System.IO.Path]::IsPathRooted($ReportPath)) {
        $ReportPath = Join-Path $repoRoot $ReportPath
    }
    $parentDir = Split-Path -Parent $ReportPath
    if ($parentDir -and -not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $ReportPath -Encoding UTF8
    Write-Host ''
    Write-Host "Report written to: $ReportPath" -ForegroundColor Green
}

Write-Host ''
Write-Host "Eval complete." -ForegroundColor Green
