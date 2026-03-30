#Requires -Version 7.0
<#
.SYNOPSIS
    Profiling-enabled test runner that measures PC-AI operation performance
    with statistical rigor.

.DESCRIPTION
    Runs key PC-AI operations with N iterations (default 10) for statistical
    significance.  Captures per-operation timing (mean, median, stddev, p95,
    min, max) and memory deltas where possible.  Outputs structured JSON
    results and identifies bottlenecks by ranking operations slowest to fastest.

    Categories profiled:
      - GPU Operations       (FFI + CLI paths)
      - Native FFI Operations (pcai-perf processes, disk, hash)
      - Module Loading        (Import-Module warm/cold)
      - GGUF Parsing          (preflight with model files)

.PARAMETER Iterations
    Number of timed iterations per operation (default 10).  The warmup
    iteration is always run first and excluded from statistics.

.PARAMETER Suite
    Preset iteration counts.
      quick    = 3 iterations  (fast smoke-test)
      standard = 10 iterations (default)
      full     = 25 iterations (high-confidence)

.PARAMETER OutputFormat
    How to render the console report.
      Table    = fixed-width aligned table (default)
      Json     = raw JSON to stdout
      Markdown = markdown-formatted table

.PARAMETER ReportPath
    Directory for the JSON report file.  Defaults to
    Reports/profiling/<timestamp>/profile.json under the repo root.

.PARAMETER CompareBaseline
    Load baseline from .pcai/benchmarks/profile-baseline.json and flag
    any operation whose mean regressed more than 15%.

.PARAMETER SaveBaseline
    After profiling, save results as the new baseline at
    .pcai/benchmarks/profile-baseline.json.

.EXAMPLE
    .\Invoke-ProfileTests.ps1
    Runs the standard suite (10 iterations) and prints a table.

.EXAMPLE
    .\Invoke-ProfileTests.ps1 -Suite full -OutputFormat Markdown -SaveBaseline
    High-confidence run with markdown output; saves as new baseline.

.EXAMPLE
    .\Invoke-ProfileTests.ps1 -Suite quick -CompareBaseline
    Quick smoke-test that flags regressions against the saved baseline.
#>
[CmdletBinding()]
param(
    [int]$Iterations = 10,

    [ValidateSet('quick', 'standard', 'full')]
    [string]$Suite = 'standard',

    [ValidateSet('Table', 'Json', 'Markdown')]
    [string]$OutputFormat = 'Table',

    [string]$ReportPath,

    [switch]$CompareBaseline,

    [switch]$SaveBaseline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve repo root ────────────────────────────────────────────────────────
. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')
$RepoRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot

# ── Suite presets override $Iterations ────────────────────────────────────────
switch ($Suite) {
    'quick'    { $Iterations = 3  }
    'standard' { $Iterations = 10 }
    'full'     { $Iterations = 25 }
}

# ── Constants ─────────────────────────────────────────────────────────────────
$RegressionThresholdPct = 15
$PcaiPerfExe = $null
$PcaiPerfCandidates = @(
    'T:\RustCache\cargo-target\release\pcai-perf.exe',
    (Join-Path $RepoRoot 'Native\pcai_core\target\release\pcai-perf.exe'),
    (Join-Path $env:USERPROFILE '.local\bin\pcai-perf.exe')
)
foreach ($candidate in $PcaiPerfCandidates) {
    if (Test-Path $candidate -ErrorAction SilentlyContinue) {
        $PcaiPerfExe = $candidate
        break
    }
}

$Model1BPath = Join-Path $RepoRoot 'Models\Janus-Pro-1B\janus-pro-1b-llama-q4_k_m.gguf'
$Model7BPath = Join-Path $RepoRoot 'Models\Janus-Pro-7B\janus-pro-7b-llama-q4_k_m.gguf'

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# ── Statistics helpers ────────────────────────────────────────────────────────

function Get-Stats {
    <#
    .SYNOPSIS
        Compute descriptive statistics for an array of numeric values.
    #>
    param([Parameter(Mandatory)][double[]]$Values)

    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if ($n -eq 0) {
        return @{ Mean = 0; Median = 0; StdDev = 0; P95 = 0; Min = 0; Max = 0; N = 0 }
    }

    $mean = ($sorted | Measure-Object -Average).Average
    $median = if ($n % 2 -eq 1) {
        $sorted[([math]::Floor($n / 2))]
    } else {
        ($sorted[$n / 2 - 1] + $sorted[$n / 2]) / 2.0
    }

    $variance = 0.0
    foreach ($v in $sorted) {
        $variance += ($v - $mean) * ($v - $mean)
    }
    $stddev = if ($n -gt 1) { [math]::Sqrt($variance / ($n - 1)) } else { 0.0 }

    $p95Index = [math]::Min([math]::Floor($n * 0.95), $n - 1)

    return @{
        Mean   = [math]::Round($mean, 4)
        Median = [math]::Round($median, 4)
        StdDev = [math]::Round($stddev, 4)
        P95    = [math]::Round($sorted[$p95Index], 4)
        Min    = [math]::Round($sorted[0], 4)
        Max    = [math]::Round($sorted[-1], 4)
        N      = $n
    }
}

function Measure-Operation {
    <#
    .SYNOPSIS
        Run a scriptblock N times with a warmup pass, returning timing stats
        and memory deltas.  Uses Stopwatch for high-resolution timing.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$ScriptBlock,
        [int]$Count = 10
    )

    $timingsMs = [System.Collections.Generic.List[double]]::new()
    $memoryDeltas = [System.Collections.Generic.List[long]]::new()

    # ── Warmup (discarded) ────────────────────────────────────────────────
    try {
        & $ScriptBlock | Out-Null
    }
    catch {
        Write-Warning "  [SKIP] $Name - warmup failed: $($_.Exception.Message)"
        return $null
    }

    # ── Timed iterations ──────────────────────────────────────────────────
    for ($i = 0; $i -lt $Count; $i++) {
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
        [GC]::Collect()

        $memBefore = [System.GC]::GetTotalMemory($false)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        try {
            & $ScriptBlock | Out-Null
        }
        catch {
            Write-Warning "  [FAIL] $Name iteration $($i + 1): $($_.Exception.Message)"
            continue
        }

        $sw.Stop()
        $memAfter = [System.GC]::GetTotalMemory($false)

        $timingsMs.Add($sw.Elapsed.TotalMilliseconds)
        $memoryDeltas.Add($memAfter - $memBefore)
    }

    if ($timingsMs.Count -eq 0) {
        Write-Warning "  [SKIP] $Name - all iterations failed."
        return $null
    }

    $timeStats = Get-Stats -Values $timingsMs.ToArray()
    $memStats  = Get-Stats -Values ($memoryDeltas | ForEach-Object { [double]$_ })

    return [PSCustomObject]@{
        Name           = $Name
        TimeMs         = $timeStats
        MemoryDeltaBytes = $memStats
        RawTimingsMs   = $timingsMs.ToArray()
    }
}

function Format-Ms {
    <#
    .SYNOPSIS  Format a millisecond value for display.
    #>
    param([double]$Value)
    if ($Value -ge 1000) {
        return '{0:N2}s' -f ($Value / 1000)
    }
    elseif ($Value -ge 1) {
        return '{0:N1}ms' -f $Value
    }
    else {
        return '{0:N2}ms' -f $Value
    }
}

function Format-Bytes {
    param([double]$Value)
    $abs = [math]::Abs($Value)
    $sign = if ($Value -lt 0) { '-' } else { '' }
    if ($abs -ge 1MB) { return '{0}{1:N1}MB' -f $sign, ($abs / 1MB) }
    if ($abs -ge 1KB) { return '{0}{1:N1}KB' -f $sign, ($abs / 1KB) }
    return '{0}{1:N0}B' -f $sign, $abs
}

# ── GPU detection (for header) ────────────────────────────────────────────────
function Get-GpuLabel {
    try {
        $smi = & nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $smi) {
            return ($smi | Select-Object -First 1).Trim()
        }
    }
    catch {}
    return 'Unknown GPU'
}

# ── Run all profile operations ────────────────────────────────────────────────

$allResults = [System.Collections.Generic.List[PSCustomObject]]::new()
$categoryMap = [ordered]@{}   # category -> list of result objects

function Add-ProfileResult {
    param(
        [string]$Category,
        [PSCustomObject]$Result
    )
    if ($null -eq $Result) { return }
    $allResults.Add($Result)
    if (-not $categoryMap.Contains($Category)) {
        $categoryMap[$Category] = [System.Collections.Generic.List[PSCustomObject]]::new()
    }
    $categoryMap[$Category].Add($Result)
}

Write-Host ''
Write-Host "=== PC-AI Performance Profiler ===" -ForegroundColor Cyan
Write-Host "    Iterations : $Iterations"
Write-Host "    Suite      : $Suite"
Write-Host "    pcai-perf  : $(if ($PcaiPerfExe) { $PcaiPerfExe } else { '(not found)' })"
Write-Host ''

# ────────────────────────────────────────────────────────────────────────────
# Category: GPU Operations
# ────────────────────────────────────────────────────────────────────────────
Write-Host 'Category: GPU Operations' -ForegroundColor Yellow

$gpuModulePath = Join-Path $RepoRoot 'Modules\PC-AI.Gpu'
$gpuModuleAvailable = Test-Path (Join-Path $gpuModulePath 'PC-AI.Gpu.psd1') -ErrorAction SilentlyContinue

if ($gpuModuleAvailable) {
    # Pre-load the module once so we measure operation cost, not module-load cost.
    try {
        Import-Module $gpuModulePath -Force -ErrorAction Stop
        $gpuModuleLoaded = $true
    }
    catch {
        Write-Warning "  Could not load PC-AI.Gpu: $($_.Exception.Message)"
        $gpuModuleLoaded = $false
    }
}
else {
    Write-Warning '  PC-AI.Gpu module not found -- skipping GPU operations.'
    $gpuModuleLoaded = $false
}

if ($gpuModuleLoaded) {
    Write-Host '  Test-PcaiGpuReadiness (inventory)...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'Test-PcaiGpuReadiness' -Count $Iterations -ScriptBlock {
            Test-PcaiGpuReadiness
        }
    )

    Write-Host '  Test-PcaiGpuReadiness -RequiredMB 4000...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'Test-PcaiGpuReadiness -4GB' -Count $Iterations -ScriptBlock {
            Test-PcaiGpuReadiness -RequiredMB 4000
        }
    )

    Write-Host '  Get-NvidiaGpuInventory...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'Get-NvidiaGpuInventory' -Count $Iterations -ScriptBlock {
            Get-NvidiaGpuInventory
        }
    )

    if (Get-Command Get-NvidiaGpuUtilization -ErrorAction SilentlyContinue) {
        Write-Host '  Get-NvidiaGpuUtilization -Index 0...'
        Add-ProfileResult -Category 'GPU Operations' -Result (
            Measure-Operation -Name 'Get-NvidiaGpuUtilization -Index 0' -Count $Iterations -ScriptBlock {
                Get-NvidiaGpuUtilization -Index 0
            }
        )
    }
    else {
        Write-Warning '  Get-NvidiaGpuUtilization not available -- skipped.'
    }
}

if ($PcaiPerfExe) {
    Write-Host '  pcai-perf preflight (CLI)...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'pcai-perf preflight (CLI)' -Count $Iterations -ScriptBlock {
            & $PcaiPerfExe preflight 2>$null
        }.GetNewClosure()
    )

    Write-Host '  pcai-perf roofline (specs)...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'pcai-perf roofline (specs)' -Count $Iterations -ScriptBlock {
            & $PcaiPerfExe roofline 2>$null
        }.GetNewClosure()
    )

    Write-Host '  pcai-perf roofline --model-params 7 --quant-bits 4.5...'
    Add-ProfileResult -Category 'GPU Operations' -Result (
        Measure-Operation -Name 'pcai-perf roofline (7B/Q4.5)' -Count $Iterations -ScriptBlock {
            & $PcaiPerfExe roofline --model-params 7 --quant-bits 4.5 2>$null
        }.GetNewClosure()
    )
}
else {
    Write-Warning '  pcai-perf.exe not found -- skipping CLI GPU operations.'
}

# GC between categories
[GC]::Collect()
[GC]::WaitForPendingFinalizers()
[GC]::Collect()

# ────────────────────────────────────────────────────────────────────────────
# Category: Native FFI Operations
# ────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host 'Category: Native FFI Operations' -ForegroundColor Yellow

if ($PcaiPerfExe) {
    Write-Host '  pcai-perf processes --top 10...'
    Add-ProfileResult -Category 'Native FFI Operations' -Result (
        Measure-Operation -Name 'pcai-perf processes --top 10' -Count $Iterations -ScriptBlock {
            & $PcaiPerfExe processes --top 10 2>$null
        }.GetNewClosure()
    )

    Write-Host '  pcai-perf disk --path C:\codedev\PC_AI --top 5...'
    $diskPath = $RepoRoot
    Add-ProfileResult -Category 'Native FFI Operations' -Result (
        Measure-Operation -Name 'pcai-perf disk --top 5' -Count $Iterations -ScriptBlock {
            & $PcaiPerfExe disk --path $diskPath --top 5 2>$null
        }.GetNewClosure()
    )

    # Find a small file for hash-list (use CLAUDE.md as a representative small file)
    $hashTarget = Join-Path $RepoRoot 'CLAUDE.md'
    if (Test-Path $hashTarget) {
        Write-Host '  pcai-perf hash-list (CLAUDE.md)...'
        Add-ProfileResult -Category 'Native FFI Operations' -Result (
            Measure-Operation -Name 'pcai-perf hash-list (small file)' -Count $Iterations -ScriptBlock {
                & $PcaiPerfExe hash-list $hashTarget 2>$null
            }.GetNewClosure()
        )
    }
}
else {
    Write-Warning '  pcai-perf.exe not found -- skipping Native FFI operations.'
}

[GC]::Collect()
[GC]::WaitForPendingFinalizers()
[GC]::Collect()

# ────────────────────────────────────────────────────────────────────────────
# Category: Module Loading
# ────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host 'Category: Module Loading' -ForegroundColor Yellow

$modulesToProfile = @(
    @{ Name = 'PC-AI.Gpu';          Path = Join-Path $RepoRoot 'Modules\PC-AI.Gpu' },
    @{ Name = 'PC-AI.Hardware';      Path = Join-Path $RepoRoot 'Modules\PC-AI.Hardware' },
    @{ Name = 'PC-AI.Acceleration';  Path = Join-Path $RepoRoot 'Modules\PC-AI.Acceleration' },
    @{ Name = 'PC-AI.Evaluation';    Path = Join-Path $RepoRoot 'Modules\PC-AI.Evaluation' }
)

foreach ($mod in $modulesToProfile) {
    $modManifest = Join-Path $mod.Path "$($mod.Name).psd1"
    if (-not (Test-Path $modManifest -ErrorAction SilentlyContinue)) {
        Write-Warning "  $($mod.Name) manifest not found at $modManifest -- skipped."
        continue
    }

    Write-Host "  Import-Module $($mod.Name)..."
    $modPath = $mod.Path
    Add-ProfileResult -Category 'Module Loading' -Result (
        Measure-Operation -Name "Import-Module $($mod.Name)" -Count $Iterations -ScriptBlock {
            # Remove then re-import to measure cold-load each iteration.
            Remove-Module $modPath -Force -ErrorAction SilentlyContinue 2>$null
            Import-Module $modPath -Force -ErrorAction Stop
        }.GetNewClosure()
    )
}

[GC]::Collect()
[GC]::WaitForPendingFinalizers()
[GC]::Collect()

# ────────────────────────────────────────────────────────────────────────────
# Category: GGUF Parsing
# ────────────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host 'Category: GGUF Parsing' -ForegroundColor Yellow

if ($PcaiPerfExe) {
    if (Test-Path $Model1BPath -ErrorAction SilentlyContinue) {
        Write-Host '  pcai-perf preflight --model Janus-Pro-1B...'
        Add-ProfileResult -Category 'GGUF Parsing' -Result (
            Measure-Operation -Name 'pcai-perf preflight 1B' -Count $Iterations -ScriptBlock {
                & $PcaiPerfExe preflight --model $Model1BPath 2>$null
            }.GetNewClosure()
        )
    }
    else {
        Write-Warning "  Model not found: $Model1BPath -- skipped."
    }

    if (Test-Path $Model7BPath -ErrorAction SilentlyContinue) {
        Write-Host '  pcai-perf preflight --model Janus-Pro-7B...'
        Add-ProfileResult -Category 'GGUF Parsing' -Result (
            Measure-Operation -Name 'pcai-perf preflight 7B' -Count $Iterations -ScriptBlock {
                & $PcaiPerfExe preflight --model $Model7BPath 2>$null
            }.GetNewClosure()
        )
    }
    else {
        Write-Warning "  Model not found: $Model7BPath -- skipped."
    }
}
else {
    Write-Warning '  pcai-perf.exe not found -- skipping GGUF parsing.'
}

# ────────────────────────────────────────────────────────────────────────────
# Assemble report
# ────────────────────────────────────────────────────────────────────────────
Write-Host ''

$gpuLabel = Get-GpuLabel

# Build structured JSON result
$reportData = [ordered]@{
    Timestamp   = (Get-Date -Format 'o')
    Suite       = $Suite
    Iterations  = $Iterations
    Gpu         = $gpuLabel
    Hostname    = $env:COMPUTERNAME
    Categories  = [ordered]@{}
    Bottlenecks = @()
}

foreach ($cat in $categoryMap.Keys) {
    $catResults = @()
    foreach ($r in $categoryMap[$cat]) {
        $catResults += [ordered]@{
            Name     = $r.Name
            TimeMs   = [ordered]@{
                Mean   = $r.TimeMs.Mean
                Median = $r.TimeMs.Median
                StdDev = $r.TimeMs.StdDev
                P95    = $r.TimeMs.P95
                Min    = $r.TimeMs.Min
                Max    = $r.TimeMs.Max
                N      = $r.TimeMs.N
            }
            MemoryDeltaBytes = [ordered]@{
                Mean   = $r.MemoryDeltaBytes.Mean
                Median = $r.MemoryDeltaBytes.Median
                StdDev = $r.MemoryDeltaBytes.StdDev
                P95    = $r.MemoryDeltaBytes.P95
                Min    = $r.MemoryDeltaBytes.Min
                Max    = $r.MemoryDeltaBytes.Max
            }
        }
    }
    $reportData.Categories[$cat] = $catResults
}

# ── Bottleneck ranking (top 5, slowest first) ────────────────────────────────
$ranked = $allResults | Sort-Object { $_.TimeMs.Mean } -Descending
$topN = [math]::Min(5, $ranked.Count)
$bottlenecks = @()
for ($i = 0; $i -lt $topN; $i++) {
    $bottlenecks += [ordered]@{
        Rank   = $i + 1
        Name   = $ranked[$i].Name
        MeanMs = $ranked[$i].TimeMs.Mean
    }
}
$reportData.Bottlenecks = $bottlenecks

# ── Baseline comparison ──────────────────────────────────────────────────────
$baselinePath = Join-Path $RepoRoot '.pcai\benchmarks\profile-baseline.json'
$regressions = @()

if ($CompareBaseline -and (Test-Path $baselinePath)) {
    Write-Host 'Comparing against baseline...' -ForegroundColor Cyan
    $baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json

    foreach ($r in $allResults) {
        # Search baseline categories for a matching operation name.
        $baselineMean = $null
        foreach ($bCat in $baseline.Categories.PSObject.Properties) {
            foreach ($bOp in $bCat.Value) {
                if ($bOp.Name -eq $r.Name) {
                    $baselineMean = $bOp.TimeMs.Mean
                    break
                }
            }
            if ($null -ne $baselineMean) { break }
        }

        if ($null -ne $baselineMean -and $baselineMean -gt 0) {
            $pctChange = (($r.TimeMs.Mean - $baselineMean) / $baselineMean) * 100
            if ($pctChange -gt $RegressionThresholdPct) {
                $regressions += [PSCustomObject]@{
                    Name         = $r.Name
                    BaselineMs   = $baselineMean
                    CurrentMs    = $r.TimeMs.Mean
                    PctChange    = [math]::Round($pctChange, 1)
                }
            }
        }
    }

    $reportData['BaselineComparison'] = [ordered]@{
        BaselineFile = $baselinePath
        Regressions  = @($regressions | ForEach-Object {
            [ordered]@{
                Name       = $_.Name
                BaselineMs = $_.BaselineMs
                CurrentMs  = $_.CurrentMs
                PctChange  = $_.PctChange
            }
        })
    }
}
elseif ($CompareBaseline) {
    Write-Warning "Baseline file not found at $baselinePath -- skipping comparison."
}

# ── Output rendering ─────────────────────────────────────────────────────────

function Write-TableReport {
    param($Data, $AllResults, $CategoryMap, $GpuLabel, $Iterations, $Regressions)

    Write-Host ''
    Write-Host "=== PC-AI Performance Profile ($Iterations iterations) ===" -ForegroundColor Cyan
    Write-Host "GPU: $GpuLabel"
    Write-Host ''

    $header = '{0,-40} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f 'Operation', 'Mean', 'Median', 'StdDev', 'P95', 'Min', 'Max'
    $separator = '-' * $header.Length

    foreach ($cat in $CategoryMap.Keys) {
        Write-Host "Category: $cat" -ForegroundColor Yellow
        Write-Host $header -ForegroundColor DarkGray
        Write-Host $separator -ForegroundColor DarkGray

        foreach ($r in $CategoryMap[$cat]) {
            $line = '{0,-40} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f `
                $r.Name,
                (Format-Ms $r.TimeMs.Mean),
                (Format-Ms $r.TimeMs.Median),
                (Format-Ms $r.TimeMs.StdDev),
                (Format-Ms $r.TimeMs.P95),
                (Format-Ms $r.TimeMs.Min),
                (Format-Ms $r.TimeMs.Max)
            Write-Host $line
        }
        Write-Host ''
    }

    # Bottleneck ranking
    $topN = [math]::Min(5, $AllResults.Count)
    if ($topN -gt 0) {
        Write-Host "Top $topN Bottlenecks:" -ForegroundColor Magenta
        $sorted = $AllResults | Sort-Object { $_.TimeMs.Mean } -Descending
        for ($i = 0; $i -lt $topN; $i++) {
            $entry = $sorted[$i]
            $memNote = ''
            if ($entry.MemoryDeltaBytes.Mean -ne 0) {
                $memNote = ", mem delta $(Format-Bytes $entry.MemoryDeltaBytes.Mean)"
            }
            Write-Host ("  {0}. {1} -- {2}{3}" -f ($i + 1), $entry.Name, (Format-Ms $entry.TimeMs.Mean), $memNote)
        }
        Write-Host ''
    }

    # Regressions
    if ($Regressions -and $Regressions.Count -gt 0) {
        Write-Host "REGRESSIONS (>$RegressionThresholdPct% vs baseline):" -ForegroundColor Red
        foreach ($reg in $Regressions) {
            Write-Host ("  {0}: {1} -> {2} (+{3}%)" -f $reg.Name, (Format-Ms $reg.BaselineMs), (Format-Ms $reg.CurrentMs), $reg.PctChange) -ForegroundColor Red
        }
        Write-Host ''
    }
}

function Write-MarkdownReport {
    param($Data, $AllResults, $CategoryMap, $GpuLabel, $Iterations, $Regressions)

    Write-Host ''
    Write-Host "## PC-AI Performance Profile ($Iterations iterations)"
    Write-Host ''
    Write-Host "**GPU:** $GpuLabel"
    Write-Host ''

    foreach ($cat in $CategoryMap.Keys) {
        Write-Host "### $cat"
        Write-Host ''
        Write-Host '| Operation | Mean | Median | StdDev | P95 | Min | Max |'
        Write-Host '|-----------|------|--------|--------|-----|-----|-----|'

        foreach ($r in $CategoryMap[$cat]) {
            Write-Host ('| {0} | {1} | {2} | {3} | {4} | {5} | {6} |' -f `
                $r.Name,
                (Format-Ms $r.TimeMs.Mean),
                (Format-Ms $r.TimeMs.Median),
                (Format-Ms $r.TimeMs.StdDev),
                (Format-Ms $r.TimeMs.P95),
                (Format-Ms $r.TimeMs.Min),
                (Format-Ms $r.TimeMs.Max))
        }
        Write-Host ''
    }

    if ($AllResults.Count -gt 0) {
        Write-Host '### Bottlenecks (slowest first)'
        Write-Host ''
        $sorted = $AllResults | Sort-Object { $_.TimeMs.Mean } -Descending
        $topN = [math]::Min(5, $sorted.Count)
        for ($i = 0; $i -lt $topN; $i++) {
            Write-Host ("1. **{0}** -- {1}" -f $sorted[$i].Name, (Format-Ms $sorted[$i].TimeMs.Mean))
        }
        Write-Host ''
    }

    if ($Regressions -and $Regressions.Count -gt 0) {
        Write-Host '### Regressions'
        Write-Host ''
        foreach ($reg in $Regressions) {
            Write-Host ("- **{0}**: {1} -> {2} (+{3}%)" -f $reg.Name, (Format-Ms $reg.BaselineMs), (Format-Ms $reg.CurrentMs), $reg.PctChange)
        }
        Write-Host ''
    }
}

# Render to console
switch ($OutputFormat) {
    'Table' {
        Write-TableReport -Data $reportData -AllResults $allResults -CategoryMap $categoryMap `
            -GpuLabel $gpuLabel -Iterations $Iterations -Regressions $regressions
    }
    'Json' {
        $reportData | ConvertTo-Json -Depth 10
    }
    'Markdown' {
        Write-MarkdownReport -Data $reportData -AllResults $allResults -CategoryMap $categoryMap `
            -GpuLabel $gpuLabel -Iterations $Iterations -Regressions $regressions
    }
}

# ── Write JSON report to disk ────────────────────────────────────────────────
if (-not $ReportPath) {
    $ReportPath = Join-Path $RepoRoot "Reports\profiling\$Timestamp"
}

if (-not (Test-Path $ReportPath)) {
    New-Item -Path $ReportPath -ItemType Directory -Force | Out-Null
}

$jsonReportFile = Join-Path $ReportPath 'profile.json'
$reportData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonReportFile -Encoding UTF8
Write-Host "Report written to: $jsonReportFile" -ForegroundColor Green

# ── Save baseline ─────────────────────────────────────────────────────────────
if ($SaveBaseline) {
    $baselineDir = Split-Path -Parent $baselinePath
    if (-not (Test-Path $baselineDir)) {
        New-Item -Path $baselineDir -ItemType Directory -Force | Out-Null
    }
    $reportData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
    Write-Host "Baseline saved to: $baselinePath" -ForegroundColor Green
}

# ── Summary line ──────────────────────────────────────────────────────────────
$totalOps = $allResults.Count
$totalTime = ($allResults | ForEach-Object { $_.TimeMs.Mean } | Measure-Object -Sum).Sum
Write-Host ''
Write-Host "Profiled $totalOps operations in $(Format-Ms $totalTime) total mean time." -ForegroundColor Cyan

if ($regressions -and $regressions.Count -gt 0) {
    Write-Host "$($regressions.Count) regression(s) detected!" -ForegroundColor Red
    exit 1
}
else {
    if ($CompareBaseline -and (Test-Path $baselinePath)) {
        Write-Host 'No regressions detected vs baseline.' -ForegroundColor Green
    }
    exit 0
}
