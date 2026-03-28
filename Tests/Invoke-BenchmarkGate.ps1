#Requires -Version 7.0
<#
.SYNOPSIS
    CI benchmark regression detector. Compares the quick benchmark suite against
    a saved baseline and exits 1 when regressions exceed the threshold.

.EXAMPLE
    pwsh Tests\Invoke-BenchmarkGate.ps1 -SaveBaseline
    pwsh Tests\Invoke-BenchmarkGate.ps1 -FailOnRegression -Format Markdown
#>
[CmdletBinding()]
param(
    [switch]$SaveBaseline,
    [int]$Threshold = 15,
    [ValidateSet('Table', 'Json', 'Markdown')][string]$Format = 'Table',
    [string]$BaselinePath = '.pcai/benchmarks/baseline.json',
    [string]$Suite = 'quick',
    [switch]$FailOnRegression
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Helpers ──────────────────────────────────────────────────────────────────

function Get-RepoRoot {
    $here = $PSScriptRoot
    while ($here -and -not (Test-Path (Join-Path $here '.git'))) {
        $here = Split-Path $here -Parent
    }
    if (-not $here) { throw 'Cannot locate repository root (.git not found).' }
    $here
}

function Get-GitMeta {
    [hashtable]$m = @{ Hash = 'unknown'; Branch = 'unknown' }
    try { $m.Hash = (git rev-parse --short HEAD 2>$null).Trim() } catch {}
    try { $m.Branch = (git rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
    $m
}

function Invoke-BenchmarkRunner ([string]$RepoRoot, [string]$Suite) {
    $runner = Join-Path $RepoRoot 'Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1'
    if (-not (Test-Path $runner)) {
        Write-Warning "Benchmark runner not found at '$runner' — using latest existing report."
        return
    }
    try   { & pwsh -NoProfile -File $runner -Suite $Suite }
    catch { Write-Warning "Benchmark runner failed: $_ — using latest existing report." }
}

function Read-LatestReport ([string]$RepoRoot) {
    $root = Join-Path $RepoRoot 'Reports\tooling-benchmarks'
    if (-not (Test-Path $root)) { return $null }
    $f = Get-ChildItem -LiteralPath $root -Filter 'tooling-benchmark-report.json' `
        -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $f) { return $null }
    Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json
}

# Extract best (native-preferred) MeanMs per CaseId → hashtable
function ConvertTo-CaseMap ([object]$Report) {
    [hashtable]$map = @{}
    foreach ($row in $Report.Results) {
        $id = $row.CaseId
        if (-not $map.ContainsKey($id) -or $row.Backend -eq 'native') {
            $map[$id] = @{ mean_ms = [double]$row.MeanMs; median_ms = [double]$row.MedianMs; iterations = [int]$row.Iterations }
        }
    }
    $map
}

# ── Main ─────────────────────────────────────────────────────────────────────

$repoRoot = Get-RepoRoot
$absBaseline = if ([IO.Path]::IsPathRooted($BaselinePath)) { $BaselinePath } `
               else { Join-Path $repoRoot $BaselinePath }

# ── Save-baseline mode ───────────────────────────────────────────────────────
if ($SaveBaseline) {
    Invoke-BenchmarkRunner -RepoRoot $repoRoot -Suite $Suite
    $report = Read-LatestReport -RepoRoot $repoRoot
    if (-not $report) { Write-Error 'No benchmark report found.'; exit 1 }

    $meta = Get-GitMeta
    $dir  = Split-Path $absBaseline -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    [ordered]@{
        created_at = (Get-Date -Format 'o')
        git_hash   = $meta.Hash
        git_branch = $meta.Branch
        suite      = $Suite
        cases      = ConvertTo-CaseMap $report
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $absBaseline -Encoding UTF8

    $count = (ConvertTo-CaseMap $report).Count
    Write-Host "Baseline saved: $absBaseline  ($count cases, git $($meta.Hash))"
    exit 0
}

# ── Comparison mode ──────────────────────────────────────────────────────────
if (-not (Test-Path $absBaseline)) {
    Write-Warning "No baseline at '$absBaseline'. Run with -SaveBaseline first. Skipping gate."
    exit 0
}

$baselineData = Get-Content -LiteralPath $absBaseline -Raw | ConvertFrom-Json -AsHashtable
Invoke-BenchmarkRunner -RepoRoot $repoRoot -Suite $Suite
$report = Read-LatestReport -RepoRoot $repoRoot
if (-not $report) { Write-Warning 'No report after run. Skipping gate.'; exit 0 }

$currentMap   = ConvertTo-CaseMap $report
$bCases       = $baselineData['cases']
$baselineHash = $baselineData['git_hash']

[Collections.Generic.List[hashtable]]$rows = @()

foreach ($id in $bCases.Keys) {
    if ($currentMap.ContainsKey($id)) {
        $delta  = (($currentMap[$id].mean_ms - $bCases[$id].mean_ms) / $bCases[$id].mean_ms) * 100.0
        $status = if ($delta -gt $Threshold) { 'regressed' } elseif ($delta -lt -5.0) { 'improved' } else { 'unchanged' }
        $rows.Add(@{ CaseId = $id; BaselineMs = $bCases[$id].mean_ms; CurrentMs = $currentMap[$id].mean_ms; DeltaPct = $delta; Status = $status })
    } else {
        $rows.Add(@{ CaseId = $id; BaselineMs = $bCases[$id].mean_ms; CurrentMs = $null; DeltaPct = $null; Status = 'missing' })
    }
}
foreach ($id in $currentMap.Keys) {
    if (-not $bCases.ContainsKey($id)) {
        $rows.Add(@{ CaseId = $id; BaselineMs = $null; CurrentMs = $currentMap[$id].mean_ms; DeltaPct = $null; Status = 'new' })
    }
}

$regressionCount = @($rows | Where-Object { $_.Status -eq 'regressed' }).Count

# ── Output ───────────────────────────────────────────────────────────────────
switch ($Format) {
    'Json' {
        [ordered]@{ baseline_hash = $baselineHash; suite = $Suite; threshold_pct = $Threshold
                    regression_count = $regressionCount; cases = $rows } | ConvertTo-Json -Depth 5
    }
    'Markdown' {
        "## Benchmark Gate: ``$Suite`` vs baseline (``$baselineHash``)"
        ''
        '| Case | Baseline | Current | Delta | Status |'
        '|------|----------|---------|-------|--------|'
        foreach ($r in $rows | Sort-Object CaseId) {
            $s = switch ($r.Status) { 'regressed' { ':red_circle: **REGRESSED**' } 'improved' { ':green_circle: improved' }
                                      'new' { ':blue_circle: new' } 'missing' { ':warning: MISSING' } default { 'unchanged' } }
            $b = if ($r.BaselineMs) { '{0:F1}ms' -f $r.BaselineMs } else { 'n/a' }
            $c = if ($r.CurrentMs)  { '{0:F1}ms' -f $r.CurrentMs }  else { 'n/a' }
            $d = if ($null -ne $r.DeltaPct) { '{0:+0.0;-0.0}%' -f $r.DeltaPct } else { 'n/a' }
            "| ``$($r.CaseId)`` | $b | $c | $d | $s |"
        }
        ''
        "> **Overall:** $regressionCount regression(s) | threshold: $Threshold%"
    }
    default {
        $header = "Benchmark Gate: $Suite suite vs baseline ($baselineHash)"
        Write-Output $header
        Write-Output ('-' * $header.Length)
        Write-Output ('{0,-28} {1,10} {2,10} {3,9}  {4}' -f 'Case', 'Baseline', 'Current', 'Delta', 'Status')
        Write-Output ('-' * ($header.Length + 4))
        foreach ($r in $rows | Sort-Object CaseId) {
            $s = switch ($r.Status) { 'regressed' { 'REGRESSED' } 'improved' { 'improved' }
                                      'new' { 'new' } 'missing' { 'MISSING' } default { 'unchanged' } }
            $b = if ($r.BaselineMs) { '{0:F1}ms' -f $r.BaselineMs } else { 'n/a' }
            $c = if ($r.CurrentMs)  { '{0:F1}ms' -f $r.CurrentMs }  else { 'n/a' }
            $d = if ($null -ne $r.DeltaPct) { '{0:+0.0;-0.0}%' -f $r.DeltaPct } else { 'n/a' }
            Write-Output ('{0,-28} {1,10} {2,10} {3,9}  {4}' -f $r.CaseId, $b, $c, $d, $s)
        }
        Write-Output ''
        Write-Output "Overall: $regressionCount regression(s) (threshold: $Threshold%)"
    }
}

if ($FailOnRegression -and $regressionCount -gt 0) { exit 1 }
exit 0
