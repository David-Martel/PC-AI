#Requires -Version 7.0
<#
.SYNOPSIS
    Unified LLM performance benchmark runner for PC-AI.

.DESCRIPTION
    Orchestrates the full LLM performance measurement pipeline:

      1. Preflight GPU check (VRAM availability via Test-PcaiGpuReadiness)
      2. Run inference benchmarks against Ollama models with deterministic seeding
      3. Calculate roofline efficiency (actual tok/s vs memory-bandwidth ceiling)
      4. Compare against saved baseline for regression detection
      5. Output comprehensive performance report (Table, JSON, or Markdown)

    Three suite levels control scope and duration:
      - quick:    1 model, 3 prompts, <60s    (daily validation)
      - standard: 2 models, 8 prompts, ~5min  (pre-merge check)
      - full:     All config candidates, full prompt set (complete benchmark)

    Each model is benchmarked N times (configurable via -Iterations) to produce
    stable mean/median/stddev measurements.  Uses seed=42 for deterministic
    output across runs.

.PARAMETER Suite
    Benchmark scope: quick (default), standard, or full.

.PARAMETER OllamaUrl
    Base URL for the Ollama API.  Default: http://127.0.0.1:11434

.PARAMETER Models
    Override the model list for the selected suite.  When omitted, models are
    selected based on the suite level and pcai-ollama-benchmark.json config.

.PARAMETER MaxTokens
    Maximum tokens to generate per prompt.  Default: 64

.PARAMETER Temperature
    Sampling temperature.  Default: 0.1

.PARAMETER NumCtx
    Context window size in tokens.  Default: 4096

.PARAMETER Iterations
    Number of times to repeat each model/prompt combination for stable
    measurement.  Default: 3

.PARAMETER SaveBaseline
    Save current results as the new performance baseline at
    .pcai/benchmarks/perf-baseline.json and exit.

.PARAMETER CompareBaseline
    Compare current results against the saved baseline and report regressions.
    Uses a 10% threshold by default.

.PARAMETER IncludeRoofline
    Add theoretical memory-bandwidth ceiling and efficiency percentage to the
    output.  Uses GPU bandwidth / (params_b * quant_bits / 8) model.

.PARAMETER OutputFormat
    Report format: Table (console), Json (machine-readable), Markdown (GitHub).

.PARAMETER ReportPath
    Optional file path to write the report.  When omitted, a timestamped report
    is written to Reports/llm-perf/<timestamp>/benchmark.json.

.EXAMPLE
    .\Tools\Invoke-LlmPerfBenchmark.ps1
    Run the quick suite with default settings and display a table.

.EXAMPLE
    .\Tools\Invoke-LlmPerfBenchmark.ps1 -Suite standard -IncludeRoofline -CompareBaseline
    Run the standard suite with roofline analysis and regression detection.

.EXAMPLE
    .\Tools\Invoke-LlmPerfBenchmark.ps1 -Suite full -OutputFormat Json -ReportPath C:\Reports\bench.json
    Full benchmark with JSON output written to a specific path.

.EXAMPLE
    .\Tools\Invoke-LlmPerfBenchmark.ps1 -SaveBaseline
    Run quick suite and save results as the new performance baseline.

.EXAMPLE
    .\Tools\Invoke-LlmPerfBenchmark.ps1 -Models 'qwen2.5-coder:3b','gemma3:4b' -Iterations 5
    Benchmark specific models with 5 iterations each.
#>
[CmdletBinding()]
param(
    [ValidateSet('quick', 'standard', 'full')]
    [string]$Suite = 'quick',

    [string]$OllamaUrl = 'http://127.0.0.1:11434',

    [string[]]$Models,

    [int]$MaxTokens = 64,

    [double]$Temperature = 0.1,

    [int]$NumCtx = 4096,

    [int]$Iterations = 3,

    [switch]$SaveBaseline,

    [switch]$CompareBaseline,

    [switch]$IncludeRoofline,

    [ValidateSet('Table', 'Json', 'Markdown')]
    [string]$OutputFormat = 'Table',

    [string]$ReportPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve repo root ────────────────────────────────────────────────────────

$RepoRoot = Split-Path -Parent $PSScriptRoot

# ── Constants ─────────────────────────────────────────────────────────────────

$RegressionThreshold = 0.10  # 10%
$DefaultGpuBandwidthGbps = 448.0   # RTX 5060 Ti
$DefaultGpuName = 'NVIDIA GeForce RTX 5060 Ti'
$BaselinePath = Join-Path $RepoRoot '.pcai\benchmarks\perf-baseline.json'

# ── Model parameter lookup table (for roofline analysis) ─────────────────────

$ModelParams = @{
    'qwen2.5-coder:3b'  = @{ params_b = 3.0;  quant_bits = 4.5 }
    'qwen2.5-coder:7b'  = @{ params_b = 7.0;  quant_bits = 4.5 }
    'qwen2.5-coder:14b' = @{ params_b = 14.0; quant_bits = 4.5 }
    'gemma3:4b'          = @{ params_b = 4.0;  quant_bits = 4.5 }
    'deepseek-r1:7b'     = @{ params_b = 7.0;  quant_bits = 4.5 }
    'deepseek-r1:14b'    = @{ params_b = 14.0; quant_bits = 4.5 }
    'qwen3:14b'          = @{ params_b = 14.0; quant_bits = 4.5 }
    'qwen3:30b'          = @{ params_b = 30.0; quant_bits = 4.5 }
    'llama3:8b'          = @{ params_b = 8.0;  quant_bits = 4.5 }
    'gpt-oss:20b'        = @{ params_b = 20.0; quant_bits = 4.5 }
}

# ── Benchmark prompts (embedded data section) ────────────────────────────────

$DiagnosticPrompts = @(
    @{
        id       = 'diag-smart'
        category = 'diagnostic'
        prompt   = 'A Windows 11 workstation reports SMART status "Pred Fail" on a Samsung 990 Pro NVMe. The drive has 45,000 power-on hours and 98% wear leveling remaining. What is the most likely cause, and what immediate actions should be taken?'
    }
    @{
        id       = 'diag-bsod'
        category = 'diagnostic'
        prompt   = 'Event log shows WHEA_UNCORRECTABLE_ERROR (0x124) occurring every 3-4 days on a Ryzen 9 7950X system. Memory tests pass. What diagnostic steps should be performed to isolate the cause?'
    }
    @{
        id       = 'diag-usb'
        category = 'diagnostic'
        prompt   = 'USB 3.2 devices on a Thunderbolt 4 dock intermittently disconnect with Event ID 219 (driver failed to load). The dock firmware is current. List the three most likely root causes in order of probability.'
    }
    @{
        id       = 'diag-perf'
        category = 'diagnostic'
        prompt   = 'A workstation with dual NVIDIA GPUs (RTX 5060 Ti 16GB + RTX 2000 Ada 8GB) shows 30% lower LLM inference throughput than expected. nvidia-smi shows both GPUs at 0% utilization during inference. What configuration issues could cause this?'
    }
    @{
        id       = 'diag-network'
        category = 'diagnostic'
        prompt   = 'Thunderbolt peer networking between two Windows 11 machines shows 2.5 Gbps link speed instead of expected 40 Gbps. ARP tables are populated correctly. What are the most common causes of reduced Thunderbolt networking throughput?'
    }
)

$CodePrompts = @(
    @{
        id       = 'code-rust-error'
        category = 'code'
        prompt   = 'Write a Rust function that reads a JSON configuration file using serde, returning a Result with a custom error type that wraps both io::Error and serde_json::Error. Include proper error context with thiserror.'
    }
    @{
        id       = 'code-ps-pipeline'
        category = 'code'
        prompt   = 'Write a PowerShell function that takes pipeline input of file paths, filters to .rs files modified in the last 24 hours, and outputs a summary table with filename, line count, and last-modified timestamp.'
    }
    @{
        id       = 'code-ffi'
        category = 'code'
        prompt   = 'Explain the key safety considerations when designing a Rust FFI boundary for calling from C#/.NET via P/Invoke. Cover string marshaling, error handling, and memory ownership.'
    }
)

$RoutingPrompts = @(
    @{
        id       = 'route-tool-select'
        category = 'routing'
        prompt   = 'The user says: "Check if my NVMe drive is healthy." Which diagnostic tool should be invoked: Get-DiskHealth, Get-DeviceErrors, Get-SystemEvents, or Get-UsbStatus? Respond with just the tool name and a one-sentence justification.'
    }
    @{
        id       = 'route-no-tool'
        category = 'routing'
        prompt   = 'The user says: "What is the difference between AHCI and NVMe?" Should a diagnostic tool be invoked, or should this be answered directly? Respond with your decision and reasoning.'
    }
    @{
        id       = 'route-multi-tool'
        category = 'routing'
        prompt   = 'The user says: "My PC is running slow and USB devices keep disconnecting." Which diagnostic tools should be invoked and in what order? List each tool with the reason for inclusion.'
    }
)

# ── Suite definitions ─────────────────────────────────────────────────────────

function Get-SuiteConfig {
    param([string]$SuiteName)

    switch ($SuiteName) {
        'quick' {
            @{
                DefaultModels = @('qwen2.5-coder:3b')
                Prompts       = @($DiagnosticPrompts | Select-Object -First 3)
            }
        }
        'standard' {
            @{
                DefaultModels = @('qwen2.5-coder:3b', 'qwen2.5-coder:7b')
                Prompts       = @($DiagnosticPrompts | Select-Object -First 5) + @($CodePrompts | Select-Object -First 3)
            }
        }
        'full' {
            @{
                DefaultModels = $null  # load from config
                Prompts       = @($DiagnosticPrompts) + @($CodePrompts) + @($RoutingPrompts)
            }
        }
    }
}

# ── Helper: compute roofline ceiling ──────────────────────────────────────────

function Get-RooflineCeiling {
    param(
        [double]$ParamsB,
        [double]$QuantBits,
        [double]$BandwidthGbps
    )

    $modelSizeGb = $ParamsB * $QuantBits / 8.0
    if ($modelSizeGb -le 0) { return 0.0 }
    return [math]::Round($BandwidthGbps / $modelSizeGb, 1)
}

# ── Helper: calculate statistics ──────────────────────────────────────────────

function Get-Statistics {
    param([double[]]$Values)

    if ($Values.Count -eq 0) {
        return @{ Mean = 0.0; Median = 0.0; StdDev = 0.0; Min = 0.0; Max = 0.0 }
    }

    $sorted = @($Values | Sort-Object)
    $count = $sorted.Count
    $sum = ($sorted | Measure-Object -Sum).Sum
    $mean = $sum / $count

    $median = if ($count % 2 -eq 0) {
        ($sorted[($count / 2) - 1] + $sorted[$count / 2]) / 2.0
    } else {
        $sorted[[math]::Floor($count / 2)]
    }

    $variance = 0.0
    foreach ($v in $sorted) {
        $variance += ($v - $mean) * ($v - $mean)
    }
    $stddev = if ($count -gt 1) { [math]::Sqrt($variance / ($count - 1)) } else { 0.0 }

    return @{
        Mean   = [math]::Round($mean, 1)
        Median = [math]::Round($median, 1)
        StdDev = [math]::Round($stddev, 1)
        Min    = [math]::Round($sorted[0], 1)
        Max    = [math]::Round($sorted[-1], 1)
    }
}

# ── Helper: detect GPU info via nvidia-smi ────────────────────────────────────

function Get-GpuInfo {
    $gpuName = $DefaultGpuName
    $gpuFreeMb = 0
    $gpuTotalMb = 0
    $gpuBandwidthGbps = $DefaultGpuBandwidthGbps

    try {
        $smiName = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($smiName) {
            $gpuName = ($smiName | Select-Object -First 1).Trim()
        }

        $smiMem = nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>$null
        if ($smiMem) {
            $parts = ($smiMem | Select-Object -First 1).Trim() -split ',\s*'
            if ($parts.Count -ge 2) {
                $gpuFreeMb = [int]$parts[0]
                $gpuTotalMb = [int]$parts[1]
            }
        }
    } catch {
        Write-Verbose "nvidia-smi query failed: $($_.Exception.Message)"
    }

    return @{
        Name          = $gpuName
        FreeMb        = $gpuFreeMb
        TotalMb       = $gpuTotalMb
        BandwidthGbps = $gpuBandwidthGbps
    }
}

# ── Helper: test Ollama connectivity ──────────────────────────────────────────

function Test-OllamaConnection {
    param([string]$Url)

    try {
        $response = Invoke-RestMethod -Uri "$Url/api/tags" -Method Get -TimeoutSec 5 -ErrorAction Stop
        $modelNames = @($response.models | ForEach-Object { $_.name })
        return @{
            Available = $true
            Models    = $modelNames
        }
    } catch {
        return @{
            Available = $false
            Models    = @()
        }
    }
}

# ── Helper: run single Ollama generate call ───────────────────────────────────

function Invoke-OllamaGenerate {
    param(
        [string]$Url,
        [string]$Model,
        [string]$Prompt,
        [int]$MaxTok,
        [double]$Temp,
        [int]$Ctx
    )

    $body = @{
        model   = $Model
        prompt  = $Prompt
        stream  = $false
        options = @{
            temperature = $Temp
            num_predict = $MaxTok
            num_ctx     = $Ctx
            seed        = 42
        }
    } | ConvertTo-Json -Depth 3

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-RestMethod -Uri "$Url/api/generate" `
            -Method Post `
            -Body $body `
            -ContentType 'application/json' `
            -TimeoutSec 180 `
            -ErrorAction Stop
        $stopwatch.Stop()

        # Ollama returns eval_duration in nanoseconds
        $evalDurationNs = [double]$response.eval_duration
        $evalCount = [int]$response.eval_count
        $promptEvalDurationNs = [double]$response.prompt_eval_duration
        $promptEvalCount = [int]$response.prompt_eval_count
        $totalDurationNs = [double]$response.total_duration

        $evalTokS = if ($evalDurationNs -gt 0) {
            $evalCount / ($evalDurationNs / 1e9)
        } else { 0.0 }

        $promptTokS = if ($promptEvalDurationNs -gt 0) {
            $promptEvalCount / ($promptEvalDurationNs / 1e9)
        } else { 0.0 }

        return @{
            Success            = $true
            EvalTokPerSec      = [math]::Round($evalTokS, 1)
            PromptTokPerSec    = [math]::Round($promptTokS, 1)
            EvalCount          = $evalCount
            PromptEvalCount    = $promptEvalCount
            TotalDurationMs    = [math]::Round($totalDurationNs / 1e6, 0)
            EvalDurationMs     = [math]::Round($evalDurationNs / 1e6, 0)
            PromptDurationMs   = [math]::Round($promptEvalDurationNs / 1e6, 0)
            WallClockMs        = $stopwatch.ElapsedMilliseconds
            ResponseLength     = ([string]$response.response).Length
            Error              = $null
        }
    } catch {
        $stopwatch.Stop()
        return @{
            Success            = $false
            EvalTokPerSec      = 0.0
            PromptTokPerSec    = 0.0
            EvalCount          = 0
            PromptEvalCount    = 0
            TotalDurationMs    = $stopwatch.ElapsedMilliseconds
            EvalDurationMs     = 0
            PromptDurationMs   = 0
            WallClockMs        = $stopwatch.ElapsedMilliseconds
            ResponseLength     = 0
            Error              = $_.Exception.Message
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

$startTime = [datetime]::UtcNow
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# Collect git hash
$gitHash = 'unknown'
try {
    $gitHash = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
    if (-not $gitHash) { $gitHash = 'unknown' }
} catch { }

Write-Host ''
Write-Host "  === LLM Performance Benchmark ($Suite suite) ===" -ForegroundColor Cyan
Write-Host "  Date: $([datetime]::UtcNow.ToString('o'))  Git: $gitHash" -ForegroundColor DarkGray

# ── 1. Preflight GPU check ───────────────────────────────────────────────────

$gpuInfo = Get-GpuInfo

# Try the full preflight if the module is available
$preflightResult = $null
$gpuModulePath = Join-Path $RepoRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'
if (Test-Path $gpuModulePath -ErrorAction SilentlyContinue) {
    try {
        Import-Module $gpuModulePath -Force -ErrorAction Stop
        if (Get-Command Test-PcaiGpuReadiness -ErrorAction SilentlyContinue) {
            $preflightResult = Test-PcaiGpuReadiness -RequiredMB 2000
        }
    } catch {
        Write-Verbose "GPU module import failed: $($_.Exception.Message)"
    }
}

if ($preflightResult) {
    Write-Host "  GPU: $($gpuInfo.Name) ($($gpuInfo.FreeMb) MB free)" -ForegroundColor DarkGray
    if ($preflightResult.Verdict -eq 'fail') {
        Write-Error "GPU preflight failed: $($preflightResult.Reason)"
        return
    } elseif ($preflightResult.Verdict -eq 'warn') {
        Write-Warning "GPU preflight warning: $($preflightResult.Reason)"
    }
} else {
    Write-Host "  GPU: $($gpuInfo.Name) ($($gpuInfo.FreeMb) MB free) [preflight skipped]" -ForegroundColor DarkGray
}

# ── 2. Check Ollama availability ─────────────────────────────────────────────

$ollamaStatus = Test-OllamaConnection -Url $OllamaUrl
if (-not $ollamaStatus.Available) {
    Write-Error "Ollama not running at $OllamaUrl. Start Ollama and retry."
    return
}

Write-Host "  Ollama: $OllamaUrl ($($ollamaStatus.Models.Count) models available)" -ForegroundColor DarkGray

# ── 3. Resolve models and prompts for the suite ──────────────────────────────

$suiteConfig = Get-SuiteConfig -SuiteName $Suite

# Use -Models override if provided, otherwise suite defaults
$targetModels = if ($Models -and $Models.Count -gt 0) {
    @($Models)
} elseif ($suiteConfig.DefaultModels) {
    @($suiteConfig.DefaultModels)
} else {
    # Full suite: load candidates from config
    $configPath = Join-Path $RepoRoot 'Config\pcai-ollama-benchmark.json'
    if (Test-Path $configPath) {
        $benchConfig = Get-Content $configPath -Raw | ConvertFrom-Json -Depth 10
        @($benchConfig.candidateModels | Where-Object {
            -not $_.PSObject.Properties['provider'] -or $_.provider -eq 'ollama'
        } | ForEach-Object { $_.name })
    } else {
        Write-Warning "Config not found at $configPath; falling back to defaults."
        @('qwen2.5-coder:3b', 'qwen2.5-coder:7b')
    }
}

$prompts = @($suiteConfig.Prompts)

# Filter to models actually available in Ollama
$availableModels = @($targetModels | Where-Object { $_ -in $ollamaStatus.Models })
$missingModels = @($targetModels | Where-Object { $_ -notin $ollamaStatus.Models })

if ($missingModels.Count -gt 0) {
    Write-Warning "Models not available in Ollama (skipped): $($missingModels -join ', ')"
}

if ($availableModels.Count -eq 0) {
    Write-Error "No requested models are available in Ollama. Available: $($ollamaStatus.Models -join ', ')"
    return
}

Write-Host "  Models: $($availableModels -join ', ')" -ForegroundColor DarkGray
Write-Host "  Prompts: $($prompts.Count) | Iterations: $Iterations | MaxTokens: $MaxTokens" -ForegroundColor DarkGray
Write-Host ''

# ── 4. Run benchmarks ────────────────────────────────────────────────────────

$allResults = [System.Collections.Generic.List[hashtable]]::new()
$modelSummaries = [System.Collections.Generic.List[hashtable]]::new()

foreach ($model in $availableModels) {
    Write-Host "  Benchmarking: $model" -ForegroundColor Yellow

    $modelIterationToks = [System.Collections.Generic.List[double]]::new()
    $modelPromptToks = [System.Collections.Generic.List[double]]::new()
    $promptResults = [System.Collections.Generic.List[hashtable]]::new()

    foreach ($p in $prompts) {
        $iterToks = [System.Collections.Generic.List[double]]::new()
        $iterPromptToks = [System.Collections.Generic.List[double]]::new()

        for ($i = 1; $i -le $Iterations; $i++) {
            $result = Invoke-OllamaGenerate `
                -Url $OllamaUrl `
                -Model $model `
                -Prompt $p.prompt `
                -MaxTok $MaxTokens `
                -Temp $Temperature `
                -Ctx $NumCtx

            if ($result.Success -and $result.EvalTokPerSec -gt 0) {
                $iterToks.Add($result.EvalTokPerSec)
                $iterPromptToks.Add($result.PromptTokPerSec)
                $modelIterationToks.Add($result.EvalTokPerSec)
                $modelPromptToks.Add($result.PromptTokPerSec)
            } else {
                Write-Verbose "  [$model] $($p.id) iter $i failed: $($result.Error)"
            }

            $allResults.Add(@{
                model             = $model
                prompt_id         = $p.id
                category          = $p.category
                iteration         = $i
                eval_tok_s        = $result.EvalTokPerSec
                prompt_tok_s      = $result.PromptTokPerSec
                eval_count        = $result.EvalCount
                prompt_eval_count = $result.PromptEvalCount
                total_duration_ms = $result.TotalDurationMs
                eval_duration_ms  = $result.EvalDurationMs
                wall_clock_ms     = $result.WallClockMs
                response_length   = $result.ResponseLength
                success           = $result.Success
                error             = $result.Error
            })
        }

        if ($iterToks.Count -gt 0) {
            $stats = Get-Statistics -Values @($iterToks)
            $promptResults.Add(@{
                prompt_id = $p.id
                category  = $p.category
                stats     = $stats
            })

            $statusChar = '.'
            Write-Host "    $($p.id): $($stats.Mean) tok/s (sd=$($stats.StdDev)) $statusChar" -ForegroundColor Gray
        } else {
            Write-Host "    $($p.id): FAILED" -ForegroundColor Red
        }
    }

    # Model-level summary
    if ($modelIterationToks.Count -gt 0) {
        $evalStats = Get-Statistics -Values @($modelIterationToks)
        $promptStats = Get-Statistics -Values @($modelPromptToks)

        $meta = $ModelParams[$model]
        $paramsB = if ($meta) { $meta.params_b } else { 0.0 }
        $quantBits = if ($meta) { $meta.quant_bits } else { 4.5 }
        $ceiling = if ($paramsB -gt 0) {
            Get-RooflineCeiling -ParamsB $paramsB -QuantBits $quantBits -BandwidthGbps $gpuInfo.BandwidthGbps
        } else { 0.0 }
        $efficiency = if ($ceiling -gt 0) {
            [math]::Round(($evalStats.Mean / $ceiling) * 100.0, 1)
        } else { 0.0 }

        $modelSummaries.Add(@{
            model            = $model
            eval_tok_s       = $evalStats
            prompt_tok_s     = $promptStats
            params_b         = $paramsB
            quant_bits       = $quantBits
            ceiling          = $ceiling
            efficiency_pct   = $efficiency
            prompt_results   = @($promptResults)
            total_iterations = $modelIterationToks.Count
        })

        Write-Host "    => Mean: $($evalStats.Mean) tok/s | Median: $($evalStats.Median) | StdDev: $($evalStats.StdDev)" -ForegroundColor Green
    } else {
        Write-Host "    => All iterations failed for $model" -ForegroundColor Red
    }

    Write-Host ''
}

if ($modelSummaries.Count -eq 0) {
    Write-Error 'No successful benchmark results. Check Ollama availability and model status.'
    return
}

# ── 5. Regression check (if -CompareBaseline) ────────────────────────────────

$regressionResults = [System.Collections.Generic.List[hashtable]]::new()
$regressionCount = 0

if ($CompareBaseline) {
    if (-not (Test-Path $BaselinePath)) {
        Write-Warning "No baseline found at $BaselinePath. Run with -SaveBaseline first."
    } else {
        $baseline = Get-Content $BaselinePath -Raw | ConvertFrom-Json -Depth 10
        Write-Host '  Regression Check (vs baseline)' -ForegroundColor Cyan
        Write-Host "  Baseline: $($baseline.timestamp) | Git: $($baseline.git_hash)" -ForegroundColor DarkGray

        foreach ($summary in $modelSummaries) {
            $modelName = $summary.model
            $currentToks = $summary.eval_tok_s.Mean
            $baseEntry = $baseline.models.PSObject.Properties[$modelName]

            if ($baseEntry) {
                $baselineToks = [double]$baseEntry.Value.toks
                $delta = ($currentToks - $baselineToks) / $baselineToks
                $isRegression = $delta -lt (-$RegressionThreshold)
                $status = if ($isRegression) { 'FAIL' } else { 'PASS' }
                if ($isRegression) { $regressionCount++ }

                $regressionResults.Add(@{
                    model        = $modelName
                    baseline     = $baselineToks
                    current      = $currentToks
                    delta_pct    = [math]::Round($delta * 100.0, 1)
                    status       = $status
                })

                $color = if ($isRegression) { 'Red' } else { 'Green' }
                Write-Host "    $modelName`: $currentToks tok/s vs $baselineToks baseline ($('{0:+0.0;-0.0;0.0}' -f ($delta * 100.0))%) [$status]" -ForegroundColor $color
            } else {
                $regressionResults.Add(@{
                    model        = $modelName
                    baseline     = $null
                    current      = $currentToks
                    delta_pct    = $null
                    status       = 'NEW'
                })
                Write-Host "    $modelName`: $currentToks tok/s [NEW - no baseline]" -ForegroundColor DarkGray
            }
        }

        Write-Host ''
    }
}

# ── 6. Save baseline (if -SaveBaseline) ──────────────────────────────────────

if ($SaveBaseline) {
    $baselineDir = Split-Path $BaselinePath -Parent
    if (-not (Test-Path $baselineDir)) {
        New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
    }

    $modelsSection = @{}
    foreach ($summary in $modelSummaries) {
        $modelsSection[$summary.model] = @{
            toks                = $summary.eval_tok_s.Mean
            params_b            = $summary.params_b
            quant_bits          = $summary.quant_bits
            theoretical_ceiling = $summary.ceiling
            efficiency_pct      = $summary.efficiency_pct
        }
    }

    $baselineData = [ordered]@{
        timestamp                  = [datetime]::UtcNow.ToString('o')
        git_hash                   = $gitHash
        gpu                        = $gpuInfo.Name
        gpu_memory_bandwidth_gbps  = $gpuInfo.BandwidthGbps
        suite                      = $Suite
        iterations                 = $Iterations
        models                     = $modelsSection
    }

    $baselineData | ConvertTo-Json -Depth 5 | Set-Content -Path $BaselinePath -Encoding UTF8
    Write-Host "  Baseline saved: $BaselinePath" -ForegroundColor Green
    Write-Host "  Models: $($modelsSection.Count) | GPU: $($gpuInfo.Name)" -ForegroundColor DarkGray
    Write-Host ''
}

# ── 7. Write JSON report to Reports/llm-perf/<timestamp>/ ────────────────────

$reportDir = if ($ReportPath) {
    Split-Path $ReportPath -Parent
} else {
    Join-Path $RepoRoot "Reports\llm-perf\$timestamp"
}

if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}

$jsonReportPath = if ($ReportPath) { $ReportPath } else { Join-Path $reportDir 'benchmark.json' }

$fullReport = [ordered]@{
    timestamp          = [datetime]::UtcNow.ToString('o')
    git_hash           = $gitHash
    suite              = $Suite
    iterations         = $Iterations
    max_tokens         = $MaxTokens
    temperature        = $Temperature
    num_ctx            = $NumCtx
    gpu                = [ordered]@{
        name              = $gpuInfo.Name
        free_mb           = $gpuInfo.FreeMb
        total_mb          = $gpuInfo.TotalMb
        bandwidth_gbps    = $gpuInfo.BandwidthGbps
    }
    models_benchmarked = $availableModels
    models_missing     = $missingModels
    model_summaries    = @($modelSummaries)
    all_iterations     = @($allResults)
    regression         = if ($CompareBaseline) {
        [ordered]@{
            threshold   = $RegressionThreshold
            count       = $regressionCount
            results     = @($regressionResults)
        }
    } else { $null }
    duration_seconds   = [math]::Round(([datetime]::UtcNow - $startTime).TotalSeconds, 1)
}

$fullReport | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonReportPath -Encoding UTF8
Write-Verbose "Full JSON report: $jsonReportPath"

# ── 8. Format and display output ─────────────────────────────────────────────

$rows = [System.Collections.Generic.List[pscustomobject]]::new()

foreach ($summary in $modelSummaries) {
    $baselineToks = 'N/A'
    $deltaPct = 'N/A'
    $status = '-'

    if ($CompareBaseline -and $regressionResults.Count -gt 0) {
        $regEntry = $regressionResults | Where-Object { $_.model -eq $summary.model } | Select-Object -First 1
        if ($regEntry) {
            $baselineToks = if ($null -ne $regEntry.baseline) { $regEntry.baseline } else { 'N/A' }
            $deltaPct = if ($null -ne $regEntry.delta_pct) { ('{0:+0.0;-0.0;0.0}%' -f $regEntry.delta_pct) } else { 'N/A' }
            $status = $regEntry.status
        }
    }

    $row = [pscustomobject]@{
        Model    = $summary.model
        'tok/s'  = $summary.eval_tok_s.Mean
        sigma    = $summary.eval_tok_s.StdDev
    }

    if ($IncludeRoofline) {
        $row | Add-Member -NotePropertyName 'Ceiling' -NotePropertyValue $summary.ceiling
        $row | Add-Member -NotePropertyName 'Effcy%' -NotePropertyValue ('{0:F1}%' -f $summary.efficiency_pct)
    }

    if ($CompareBaseline) {
        $row | Add-Member -NotePropertyName 'Baseline' -NotePropertyValue $baselineToks
        $row | Add-Member -NotePropertyName 'Delta%' -NotePropertyValue $deltaPct
        $row | Add-Member -NotePropertyName 'Status' -NotePropertyValue $status
    }

    $rows.Add($row)
}

# Calculate summary stats
$avgEfficiency = if ($IncludeRoofline -and $modelSummaries.Count -gt 0) {
    $effs = @($modelSummaries | Where-Object { $_.efficiency_pct -gt 0 } | ForEach-Object { $_.efficiency_pct })
    if ($effs.Count -gt 0) {
        [math]::Round(($effs | Measure-Object -Average).Average, 1)
    } else { 0.0 }
} else { 0.0 }

$totalDuration = [math]::Round(([datetime]::UtcNow - $startTime).TotalSeconds, 1)

switch ($OutputFormat) {
    'Json' {
        $rows | ConvertTo-Json -Depth 4
    }
    'Markdown' {
        Write-Output ''
        Write-Output "## LLM Performance Benchmark ($Suite suite)"
        Write-Output ''
        Write-Output "- **GPU:** $($gpuInfo.Name) ($($gpuInfo.FreeMb) MB free)"
        Write-Output "- **Date:** $([datetime]::UtcNow.ToString('o'))"
        Write-Output "- **Git:** $gitHash"
        Write-Output "- **Iterations:** $Iterations | **MaxTokens:** $MaxTokens"
        Write-Output ''

        # Build dynamic header
        $headerCols = @('Model', 'tok/s', 'sigma')
        if ($IncludeRoofline) { $headerCols += @('Ceiling', 'Effcy%') }
        if ($CompareBaseline) { $headerCols += @('Baseline', 'Delta%', 'Status') }
        $header = '| ' + ($headerCols -join ' | ') + ' |'
        $sep = '| ' + (($headerCols | ForEach-Object { '---' }) -join ' | ') + ' |'

        Write-Output $header
        Write-Output $sep

        foreach ($row in $rows) {
            $cols = @($row.Model, $row.'tok/s', $row.sigma)
            if ($IncludeRoofline) { $cols += @($row.Ceiling, $row.'Effcy%') }
            if ($CompareBaseline) { $cols += @($row.Baseline, $row.'Delta%', $row.Status) }
            Write-Output ('| ' + ($cols -join ' | ') + ' |')
        }

        Write-Output ''
        $summaryLine = "**Summary:** $($modelSummaries.Count) model(s), $regressionCount regression(s)"
        if ($IncludeRoofline) { $summaryLine += ", avg efficiency ${avgEfficiency}%" }
        $summaryLine += ", ${totalDuration}s total"
        Write-Output $summaryLine
    }
    default {
        # Table output
        Write-Host ''
        Write-Host "  === LLM Performance Benchmark ($Suite suite) ===" -ForegroundColor Cyan
        Write-Host "  GPU: $($gpuInfo.Name) ($($gpuInfo.FreeMb) MB free)" -ForegroundColor DarkGray
        Write-Host "  Date: $([datetime]::UtcNow.ToString('o'))" -ForegroundColor DarkGray
        Write-Host "  Git: $gitHash" -ForegroundColor DarkGray
        Write-Host ''

        $rows | Format-Table -AutoSize

        $summaryParts = @(
            "$($modelSummaries.Count) model(s)"
            "$regressionCount regression(s)"
        )
        if ($IncludeRoofline) { $summaryParts += "avg efficiency ${avgEfficiency}%" }
        $summaryParts += "${totalDuration}s total"

        Write-Host "  Summary: $($summaryParts -join ', ')" -ForegroundColor DarkGray
    }
}

Write-Host ''
Write-Host "  Report: $jsonReportPath" -ForegroundColor DarkGray
Write-Host ''

# ── Return structured output for pipeline consumption ─────────────────────────

[pscustomobject]@{
    Suite          = $Suite
    Timestamp      = [datetime]::UtcNow.ToString('o')
    GitHash        = $gitHash
    Gpu            = $gpuInfo.Name
    ModelCount     = $modelSummaries.Count
    Regressions    = $regressionCount
    AvgEfficiency  = $avgEfficiency
    DurationSec    = $totalDuration
    ReportPath     = $jsonReportPath
    Summaries      = @($modelSummaries)
}
