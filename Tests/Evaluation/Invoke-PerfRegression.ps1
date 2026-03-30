#Requires -Version 7.0
<#
.SYNOPSIS
    Performance regression detector — compare LLM inference tok/s against saved baselines.

.DESCRIPTION
    Reads a baseline file containing model-to-tok/s mappings, runs or loads the
    current benchmark results, and reports any regressions that exceed a
    configurable threshold.  Optionally includes roofline efficiency analysis
    (actual tok/s vs theoretical memory-bandwidth ceiling).

    Designed to integrate with CI gates: use -FailOnRegression to exit 1 when
    any model's throughput drops beyond the threshold.

    Two operating modes:
      1. Provide -ResultsPath pointing to an existing benchmark JSON
         (format matching Reports/llm-benchmark-*.json).
      2. Omit -ResultsPath to run the benchmark sweep automatically via
         Invoke-OllamaBenchmarkSweep.ps1.

.PARAMETER BaselinePath
    Path to the performance baseline JSON file.  Relative paths are resolved
    from the repository root.  Default: .pcai/benchmarks/perf-baseline.json

.PARAMETER ResultsPath
    Path to benchmark results JSON.  When omitted, the script runs the Ollama
    benchmark sweep and uses the latest results.

.PARAMETER Threshold
    Maximum allowed regression as a decimal fraction (0.10 = 10%).
    A model whose tok/s drops more than this fraction below baseline is
    flagged as a regression.  Default: 0.10

.PARAMETER SaveBaseline
    Save the current results as the new baseline and exit.  Captures git hash,
    GPU info, and roofline ceilings alongside raw tok/s.

.PARAMETER FailOnRegression
    Exit with code 1 if any model regresses beyond the threshold.
    Intended for CI gate usage.

.PARAMETER OutputFormat
    Output format for the regression report.
    Table: Console-friendly table (default).
    Json: Machine-readable JSON array.
    Markdown: GitHub-flavoured Markdown table.

.PARAMETER IncludeRoofline
    Add theoretical ceiling and bandwidth efficiency percentage to the
    output.  Uses the roofline model from pcai_core_lib (GPU memory
    bandwidth / model size = max decode tok/s).

.EXAMPLE
    .\Tests\Evaluation\Invoke-PerfRegression.ps1 -ResultsPath Reports\llm-benchmark-20260328.json
    Compare the benchmark results against the saved baseline.

.EXAMPLE
    .\Tests\Evaluation\Invoke-PerfRegression.ps1 -SaveBaseline -ResultsPath Reports\llm-benchmark-20260328.json
    Save the benchmark results as the new performance baseline.

.EXAMPLE
    .\Tests\Evaluation\Invoke-PerfRegression.ps1 -FailOnRegression -Threshold 0.15 -IncludeRoofline
    Run the benchmark, compare with 15% threshold, include roofline, exit 1 on regression.

.EXAMPLE
    .\Tests\Evaluation\Invoke-PerfRegression.ps1 -OutputFormat Markdown -IncludeRoofline
    Produce a Markdown regression report with roofline efficiency data.
#>
[CmdletBinding()]
param(
    [string]$BaselinePath = '.pcai/benchmarks/perf-baseline.json',

    [string]$ResultsPath,

    [ValidateRange(0.01, 1.0)]
    [double]$Threshold = 0.10,

    [switch]$SaveBaseline,

    [switch]$FailOnRegression,

    [ValidateSet('Table', 'Json', 'Markdown')]
    [string]$OutputFormat = 'Table',

    [switch]$IncludeRoofline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve repo root ─────────────────────────────────────────────────────────

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')
$RepoRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot

# ── Resolve baseline path ─────────────────────────────────────────────────────

$AbsBaseline = if ([System.IO.Path]::IsPathRooted($BaselinePath)) {
    $BaselinePath
} else {
    Join-Path $RepoRoot $BaselinePath
}

# ── Known model metadata (params_b, default quant_bits) ──────────────────────
# Used when the benchmark JSON does not carry parameter counts.
# Quant bits default to 4.5 (Q4_K_M) which is the Ollama default for most
# models.  Override per-model in the baseline JSON when needed.

$KnownModels = @{
    'qwen2.5-coder:3b'  = @{ params_b = 3.0;  quant_bits = 4.5 }
    'qwen2.5-coder:7b'  = @{ params_b = 7.0;  quant_bits = 4.5 }
    'qwen2.5-coder:14b' = @{ params_b = 14.0; quant_bits = 4.5 }
    'gemma3:4b'          = @{ params_b = 4.0;  quant_bits = 4.5 }
    'deepseek-r1:7b'     = @{ params_b = 7.0;  quant_bits = 4.5 }
    'deepseek-r1:14b'    = @{ params_b = 14.0; quant_bits = 4.5 }
    'qwen3:14b'          = @{ params_b = 14.0; quant_bits = 4.5 }
    'qwen3:30b'          = @{ params_b = 30.0; quant_bits = 4.5 }
    'llama3:8b'          = @{ params_b = 8.0;  quant_bits = 4.5 }
}

# ── GPU specs for roofline (RTX 5060 Ti default) ─────────────────────────────
# These are used when pcai-perf roofline is unavailable.  The script attempts
# the native CLI first and falls back to this table.

$DefaultGpuBandwidthGbps = 448.0   # RTX 5060 Ti
$DefaultGpuName = 'NVIDIA GeForce RTX 5060 Ti'

# ── Helper: compute roofline ceiling ──────────────────────────────────────────

function Get-RooflineCeiling {
    <#
    .SYNOPSIS
        Compute theoretical decode tok/s ceiling from memory bandwidth.
    .DESCRIPTION
        ceiling = memory_bandwidth_gbps / (params_b * quant_bits / 8)
    #>
    param(
        [double]$ParamsB,
        [double]$QuantBits,
        [double]$BandwidthGbps
    )

    $modelSizeGb = $ParamsB * $QuantBits / 8.0
    if ($modelSizeGb -le 0) { return 0.0 }
    return [math]::Round($BandwidthGbps / $modelSizeGb, 1)
}

# ── Helper: try pcai-perf roofline CLI ────────────────────────────────────────

function Get-NativeRoofline {
    <#
    .SYNOPSIS
        Attempt to get roofline analysis from pcai-perf CLI.
    .DESCRIPTION
        Calls pcai-perf roofline --model-params X --quant-bits Y --actual-toks Z
        and returns the parsed JSON analysis.  Returns $null on failure.
    #>
    param(
        [double]$ParamsB,
        [double]$QuantBits,
        [double]$ActualToks
    )

    $perfExe = $null
    $candidates = @(
        (Join-Path $RepoRoot 'Native\pcai_core\target\release\pcai-perf.exe'),
        (Join-Path $env:USERPROFILE '.local\bin\pcai-perf.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $perfExe = $candidate
            break
        }
    }
    if (-not $perfExe) { return $null }

    try {
        $cliArgs = @('roofline', '--model-params', $ParamsB.ToString(), '--quant-bits', $QuantBits.ToString())
        if ($ActualToks -gt 0) {
            $cliArgs += @('--actual-toks', $ActualToks.ToString())
        }
        $rawOutput = & $perfExe @cliArgs 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }

        $jsonLine = $null
        foreach ($line in @($rawOutput)) {
            $trimmed = "$line".Trim()
            if ($trimmed.StartsWith('[') -or $trimmed.StartsWith('{')) {
                $jsonLine = $trimmed
                break
            }
        }
        if ($jsonLine) {
            $parsed = $jsonLine | ConvertFrom-Json
            # CLI returns an array of analyses (one per GPU); take the first.
            if ($parsed -is [System.Array] -and $parsed.Count -gt 0) {
                return $parsed[0]
            }
            return $parsed
        }
    } catch {
        Write-Verbose "pcai-perf roofline failed: $($_.Exception.Message)"
    }

    return $null
}

# ── Helper: get GPU bandwidth (prefer native, fall back to default) ───────────

function Get-GpuBandwidthGbps {
    <#
    .SYNOPSIS
        Determine GPU memory bandwidth for roofline calculations.
    #>

    # Try pcai-perf roofline with a dummy model to get GPU specs
    $analysis = Get-NativeRoofline -ParamsB 7.0 -QuantBits 4.5 -ActualToks 0
    if ($analysis -and $analysis.PSObject.Properties['gpu']) {
        Write-Verbose "GPU detected via pcai-perf: $($analysis.gpu)"
        # Reverse-calculate bandwidth from the analysis fields
        if ($analysis.theoretical_max_toks -gt 0 -and $analysis.model_size_gb -gt 0) {
            return [math]::Round($analysis.theoretical_max_toks * $analysis.model_size_gb, 1)
        }
    }

    Write-Verbose "Using default GPU bandwidth: $DefaultGpuBandwidthGbps GB/s ($DefaultGpuName)"
    return $DefaultGpuBandwidthGbps
}

# ── Load or run benchmark results ─────────────────────────────────────────────

function Get-BenchmarkResults {
    <#
    .SYNOPSIS
        Load benchmark results from JSON or run the sweep.
    .DESCRIPTION
        Returns a hashtable of model_name -> average tok/s (from medium+long
        prompts, excluding short prompts which produce unreliable measurements).
    #>
    param([string]$Path)

    $benchmarkData = $null

    if ($Path) {
        $absPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
        if (-not (Test-Path $absPath)) {
            Write-Error "Benchmark results file not found: $absPath"
            exit 1
        }
        $benchmarkData = Get-Content $absPath -Raw | ConvertFrom-Json -Depth 10
    } else {
        # Run the benchmark sweep
        $sweepScript = Join-Path $PSScriptRoot 'Invoke-OllamaBenchmarkSweep.ps1'
        if (-not (Test-Path $sweepScript)) {
            Write-Error "Benchmark sweep script not found: $sweepScript. Provide -ResultsPath instead."
            exit 1
        }

        Write-Host '  Running Ollama benchmark sweep...' -ForegroundColor Cyan
        $sweepResult = & $sweepScript -Quick
        if ($sweepResult -and $sweepResult.SummaryPath -and (Test-Path $sweepResult.SummaryPath)) {
            $benchmarkData = Get-Content $sweepResult.SummaryPath -Raw | ConvertFrom-Json -Depth 10
        } else {
            Write-Error 'Benchmark sweep did not produce usable results.'
            exit 1
        }
    }

    # Parse results into model -> average tok/s
    # Expected format: { results: [ { model, prompt, tok_s, ... }, ... ] }
    $modelToks = @{}

    $resultItems = $null
    if ($benchmarkData.PSObject.Properties['results']) {
        $resultItems = @($benchmarkData.results)
    } elseif ($benchmarkData -is [System.Array]) {
        $resultItems = @($benchmarkData)
    } else {
        Write-Error 'Unrecognized benchmark results format. Expected .results array or top-level array.'
        exit 1
    }

    foreach ($entry in $resultItems) {
        $modelName = [string]$entry.model
        $prompt = [string]$entry.prompt
        $tokS = [double]$entry.tok_s

        # Skip short prompts (1-3 tokens produce unreliable tok/s measurements)
        if ($prompt -eq 'short') { continue }

        if (-not $modelToks.ContainsKey($modelName)) {
            $modelToks[$modelName] = [System.Collections.Generic.List[double]]::new()
        }
        $modelToks[$modelName].Add($tokS)
    }

    # Average the tok/s per model
    $averaged = @{}
    foreach ($kvp in $modelToks.GetEnumerator()) {
        $values = @($kvp.Value)
        if ($values.Count -gt 0) {
            $avg = ($values | Measure-Object -Average).Average
            $averaged[$kvp.Key] = [math]::Round($avg, 1)
        }
    }

    return $averaged
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Host "`n  PC-AI Performance Regression Detector" -ForegroundColor Cyan
Write-Host "  Threshold: $([math]::Round($Threshold * 100, 0))% | Format: $OutputFormat`n" -ForegroundColor DarkGray

# Load current results
$currentResults = Get-BenchmarkResults -Path $ResultsPath
if ($currentResults.Count -eq 0) {
    Write-Warning 'No benchmark results found.'
    exit 0
}

Write-Host "  Models measured: $($currentResults.Count)" -ForegroundColor DarkGray

# Resolve GPU bandwidth for roofline
$gpuBandwidth = if ($IncludeRoofline -or $SaveBaseline) {
    Get-GpuBandwidthGbps
} else {
    $DefaultGpuBandwidthGbps
}

# ── Save baseline mode ───────────────────────────────────────────────────────

if ($SaveBaseline) {
    $baselineDir = Split-Path $AbsBaseline -Parent
    if (-not (Test-Path $baselineDir)) {
        New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
    }

    # Collect git hash
    $gitHash = 'unknown'
    try {
        $gitHash = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
        if (-not $gitHash) { $gitHash = 'unknown' }
    } catch { }

    # Detect GPU name
    $gpuName = $DefaultGpuName
    try {
        $smiOutput = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($smiOutput) {
            $gpuName = ($smiOutput | Select-Object -First 1).Trim()
        }
    } catch { }

    # Build models section
    $modelsSection = @{}
    foreach ($kvp in $currentResults.GetEnumerator()) {
        $modelName = $kvp.Key
        $toks = $kvp.Value

        $meta = $KnownModels[$modelName]
        $paramsB = if ($meta) { $meta.params_b } else { 0.0 }
        $quantBits = if ($meta) { $meta.quant_bits } else { 4.5 }

        $ceiling = if ($paramsB -gt 0) {
            Get-RooflineCeiling -ParamsB $paramsB -QuantBits $quantBits -BandwidthGbps $gpuBandwidth
        } else { 0.0 }

        $efficiency = if ($ceiling -gt 0) {
            [math]::Round(($toks / $ceiling) * 100.0, 1)
        } else { 0.0 }

        $modelsSection[$modelName] = @{
            toks                = $toks
            params_b            = $paramsB
            quant_bits          = $quantBits
            theoretical_ceiling = $ceiling
            efficiency_pct      = $efficiency
        }
    }

    $baseline = @{
        timestamp                = [datetime]::UtcNow.ToString('o')
        git_hash                 = $gitHash
        gpu                      = $gpuName
        gpu_memory_bandwidth_gbps = $gpuBandwidth
        models                   = $modelsSection
    }

    $baseline | ConvertTo-Json -Depth 5 | Set-Content -Path $AbsBaseline -Encoding UTF8
    Write-Host "  Baseline saved: $AbsBaseline" -ForegroundColor Green
    Write-Host "  Models: $($modelsSection.Count) | GPU: $gpuName" -ForegroundColor DarkGray
    exit 0
}

# ── Load baseline ─────────────────────────────────────────────────────────────

if (-not (Test-Path $AbsBaseline)) {
    Write-Warning "No baseline found at $AbsBaseline. Run with -SaveBaseline first to establish a performance baseline."
    # Still output current results without comparison
    $report = [System.Collections.Generic.List[pscustomobject]]::new()
    foreach ($kvp in $currentResults.GetEnumerator()) {
        $row = [pscustomobject]@{
            Model      = $kvp.Key
            Baseline   = 'N/A'
            Current    = $kvp.Value
            DeltaPct   = 'no baseline'
            Status     = 'NEW'
        }
        if ($IncludeRoofline) {
            $meta = $KnownModels[$kvp.Key]
            $paramsB = if ($meta) { $meta.params_b } else { 0.0 }
            $quantBits = if ($meta) { $meta.quant_bits } else { 4.5 }
            $ceiling = if ($paramsB -gt 0) { Get-RooflineCeiling -ParamsB $paramsB -QuantBits $quantBits -BandwidthGbps $gpuBandwidth } else { 0.0 }
            $eff = if ($ceiling -gt 0) { '{0:F1}%' -f (($kvp.Value / $ceiling) * 100.0) } else { 'N/A' }
            $row | Add-Member -NotePropertyName Ceiling -NotePropertyValue $ceiling
            $row | Add-Member -NotePropertyName Efficiency -NotePropertyValue $eff
        }
        $report.Add($row)
    }
    switch ($OutputFormat) {
        'Json'     { $report | ConvertTo-Json -Depth 4 }
        'Markdown' {
            $header = '| Model | Baseline | Current | Delta% | Status |'
            $sep    = '|-------|----------|---------|--------|--------|'
            if ($IncludeRoofline) {
                $header = '| Model | Baseline | Current | Delta% | Ceiling | Efficiency | Status |'
                $sep    = '|-------|----------|---------|--------|---------|------------|--------|'
            }
            $header; $sep
            foreach ($row in $report) {
                if ($IncludeRoofline) {
                    "| $($row.Model) | $($row.Baseline) | $($row.Current) | $($row.DeltaPct) | $($row.Ceiling) | $($row.Efficiency) | $($row.Status) |"
                } else {
                    "| $($row.Model) | $($row.Baseline) | $($row.Current) | $($row.DeltaPct) | $($row.Status) |"
                }
            }
        }
        default { $report | Format-Table -AutoSize }
    }
    exit 0
}

$baselineData = Get-Content $AbsBaseline -Raw | ConvertFrom-Json -Depth 10
$baselineModels = $baselineData.models

Write-Host "  Baseline: $AbsBaseline" -ForegroundColor DarkGray
Write-Host "  Baseline date: $($baselineData.timestamp) | Git: $($baselineData.git_hash)" -ForegroundColor DarkGray

# Use baseline GPU bandwidth if available
if ($baselineData.PSObject.Properties['gpu_memory_bandwidth_gbps']) {
    $gpuBandwidth = [double]$baselineData.gpu_memory_bandwidth_gbps
}

# ── Compare results against baseline ─────────────────────────────────────────

$regressionCount = 0
$report = [System.Collections.Generic.List[pscustomobject]]::new()

# Collect all model names (union of baseline + current)
$allModels = @($baselineModels.PSObject.Properties.Name) + @($currentResults.Keys) | Sort-Object -Unique

foreach ($modelName in $allModels) {
    $baseEntry = $baselineModels.PSObject.Properties[$modelName]
    $currentToks = if ($currentResults.ContainsKey($modelName)) { $currentResults[$modelName] } else { $null }

    $baselineToks = $null
    $baseCeiling = $null
    $baseParamsB = 0.0
    $baseQuantBits = 4.5

    if ($baseEntry) {
        $baselineToks = [double]$baseEntry.Value.toks
        if ($baseEntry.Value.PSObject.Properties['theoretical_ceiling']) {
            $baseCeiling = [double]$baseEntry.Value.theoretical_ceiling
        }
        if ($baseEntry.Value.PSObject.Properties['params_b']) {
            $baseParamsB = [double]$baseEntry.Value.params_b
        }
        if ($baseEntry.Value.PSObject.Properties['quant_bits']) {
            $baseQuantBits = [double]$baseEntry.Value.quant_bits
        }
    }

    # Fall back to known models table for parameter info
    if ($baseParamsB -eq 0.0 -and $KnownModels.ContainsKey($modelName)) {
        $baseParamsB = $KnownModels[$modelName].params_b
        $baseQuantBits = $KnownModels[$modelName].quant_bits
    }

    # Calculate delta
    $deltaPct = $null
    $regression = $false
    $status = 'NEW'

    if ($null -ne $baselineToks -and $null -ne $currentToks) {
        $deltaPct = ($currentToks - $baselineToks) / $baselineToks
        $regression = $deltaPct -lt (-$Threshold)
        $status = if ($regression) { 'FAIL' } else { 'PASS' }
        if ($regression) { $regressionCount++ }
    } elseif ($null -eq $currentToks) {
        $status = 'MISSING'
    }

    # Build row
    $row = [pscustomobject]@{
        Model    = $modelName
        Baseline = if ($null -ne $baselineToks) { $baselineToks } else { 'N/A' }
        Current  = if ($null -ne $currentToks) { $currentToks } else { 'N/A' }
        DeltaPct = if ($null -ne $deltaPct) { ('{0:+0.0;-0.0;0.0}%' -f ($deltaPct * 100.0)) } else { 'no baseline' }
        Status   = $status
    }

    if ($IncludeRoofline) {
        $ceiling = $baseCeiling
        if (-not $ceiling -and $baseParamsB -gt 0) {
            $ceiling = Get-RooflineCeiling -ParamsB $baseParamsB -QuantBits $baseQuantBits -BandwidthGbps $gpuBandwidth
        }

        $efficiency = if ($ceiling -and $ceiling -gt 0 -and $null -ne $currentToks) {
            '{0:F1}%' -f (($currentToks / $ceiling) * 100.0)
        } else { 'N/A' }

        $row | Add-Member -NotePropertyName Ceiling -NotePropertyValue $(if ($ceiling) { $ceiling } else { 'N/A' })
        $row | Add-Member -NotePropertyName Efficiency -NotePropertyValue $efficiency
    }

    $report.Add($row)
}

# ── Render output ─────────────────────────────────────────────────────────────

switch ($OutputFormat) {
    'Json' {
        $report | ConvertTo-Json -Depth 4
    }
    'Markdown' {
        $header = '| Model | Baseline | Current | Delta% | Status |'
        $sep    = '|-------|----------|---------|--------|--------|'
        if ($IncludeRoofline) {
            $header = '| Model | Baseline | Current | Delta% | Ceiling | Efficiency | Status |'
            $sep    = '|-------|----------|---------|--------|---------|------------|--------|'
        }
        $header; $sep
        foreach ($row in $report) {
            if ($IncludeRoofline) {
                "| $($row.Model) | $($row.Baseline) | $($row.Current) | $($row.DeltaPct) | $($row.Ceiling) | $($row.Efficiency) | $($row.Status) |"
            } else {
                "| $($row.Model) | $($row.Baseline) | $($row.Current) | $($row.DeltaPct) | $($row.Status) |"
            }
        }
    }
    default {
        $report | Format-Table -AutoSize
    }
}

# ── Final verdict ─────────────────────────────────────────────────────────────

$passCount = @($report | Where-Object { $_.Status -eq 'PASS' }).Count
$failCount = $regressionCount
$newCount = @($report | Where-Object { $_.Status -eq 'NEW' }).Count
$missingCount = @($report | Where-Object { $_.Status -eq 'MISSING' }).Count

Write-Host ''
if ($regressionCount -gt 0) {
    Write-Host "  REGRESSION: $regressionCount model(s) declined beyond $([math]::Round($Threshold * 100, 0))% threshold." -ForegroundColor Red
    Write-Host "  ($passCount passed, $failCount failed, $newCount new, $missingCount missing)" -ForegroundColor DarkGray
    if ($FailOnRegression) { exit 1 }
} else {
    Write-Host "  All models within regression threshold ($([math]::Round($Threshold * 100, 0))%)." -ForegroundColor Green
    Write-Host "  ($passCount passed, $newCount new, $missingCount missing)" -ForegroundColor DarkGray
}
