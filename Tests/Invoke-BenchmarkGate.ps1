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
    [switch]$FailOnRegression,
    [string]$RepoRoot,
    [string]$RunnerPath,
    [string]$ReportPath,
    [switch]$SkipRunner,
    [switch]$AllowStaleReport
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

function Get-GitMeta ([string]$TargetRoot) {
    [hashtable]$m = @{ Hash = 'unknown'; Branch = 'unknown' }
    $rootsToTry = @($TargetRoot, $PSScriptRoot) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($candidate in $rootsToTry) {
        try {
            $hashOut = git -C $candidate rev-parse --short HEAD 2>$null
            if ($LASTEXITCODE -eq 0 -and $hashOut) {
                $m.Hash = ([string]$hashOut).Trim()
                $branchOut = git -C $candidate rev-parse --abbrev-ref HEAD 2>$null
                if ($LASTEXITCODE -eq 0 -and $branchOut) {
                    $m.Branch = ([string]$branchOut).Trim()
                }
                break
            }
        } catch {}
    }
    $m
}

function Invoke-BenchmarkRunner ([string]$TargetRoot, [string]$TargetSuite, [string]$ExplicitRunnerPath) {
    $runner = if ($ExplicitRunnerPath) {
        if ([IO.Path]::IsPathRooted($ExplicitRunnerPath)) { $ExplicitRunnerPath }
        else { Join-Path $TargetRoot $ExplicitRunnerPath }
    } else {
        Join-Path $TargetRoot 'Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1'
    }

    if (-not (Test-Path -LiteralPath $runner)) {
        Write-Error "Benchmark runner not found at '$runner'." -ErrorAction Continue
        exit 1
    }

    & pwsh -NoProfile -File $runner -Suite $TargetSuite
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error "Benchmark runner failed with exit code $exitCode." -ErrorAction Continue
        exit 1
    }
}

function Read-LatestReport ([string]$TargetRoot, [string]$ExplicitReportPath, [nullable[datetime]]$MinWriteTimeUtc) {
    $reportFile = $null
    if ($ExplicitReportPath) {
        $resolvedReport = if ([IO.Path]::IsPathRooted($ExplicitReportPath)) { $ExplicitReportPath }
                          else { Join-Path $TargetRoot $ExplicitReportPath }
        if (-not (Test-Path -LiteralPath $resolvedReport)) {
            return $null
        }
        $reportFile = Get-Item -LiteralPath $resolvedReport
    } else {
        $root = Join-Path $TargetRoot 'Reports\tooling-benchmarks'
        if (-not (Test-Path -LiteralPath $root)) { return $null }
        $reportFile = Get-ChildItem -LiteralPath $root -Filter 'tooling-benchmark-report.json' `
            -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
    }

    if (-not $reportFile) { return $null }

    if ($null -ne $MinWriteTimeUtc -and $reportFile.LastWriteTimeUtc -lt $MinWriteTimeUtc) {
        Write-Error "Stale benchmark report detected at '$($reportFile.FullName)' (last modified $($reportFile.LastWriteTimeUtc.ToString('o')), expected >= $($MinWriteTimeUtc.Value.ToString('o')))." -ErrorAction Continue
        exit 1
    }

    Get-Content -LiteralPath $reportFile.FullName -Raw | ConvertFrom-Json
}

function Assert-ReportFreshness ([object]$Report, [hashtable]$GitMeta, [switch]$AllowStale) {
    if ($Report.PSObject.Properties['EquivalenceVerified'] -and -not [bool]$Report.EquivalenceVerified) {
        Write-Error 'Benchmark report failed equivalence verification (EquivalenceVerified = false).' -ErrorAction Continue
        exit 1
    }

    if ($AllowStale) {
        return
    }

    $reportHash = $null
    if ($Report.PSObject.Properties['GitHead']) {
        $reportHash = [string]$Report.GitHead
    } elseif ($Report.PSObject.Properties['git_hash']) {
        $reportHash = [string]$Report.git_hash
    }

    if ([string]::IsNullOrWhiteSpace($reportHash) -or $reportHash -eq 'unknown') {
        Write-Error "Benchmark report is missing a valid GitHead stamp (found '$reportHash'). Pass -AllowStaleReport to override." -ErrorAction Continue
        exit 1
    }

    if ($GitMeta.Hash -ne 'unknown' -and $reportHash -ne $GitMeta.Hash) {
        Write-Error "Benchmark report GitHead '$reportHash' does not match current HEAD '$($GitMeta.Hash)'. Pass -AllowStaleReport to override." -ErrorAction Continue
        exit 1
    }
}

# Extract best (native-preferred) MeanMs per CaseId → hashtable
function ConvertTo-CaseMap ([object]$Report) {
    [hashtable]$map = @{}
    foreach ($row in @($Report.Results)) {
        $id = $row.CaseId
        if (-not $map.ContainsKey($id) -or $row.Backend -eq 'native') {
            $map[$id] = @{ mean_ms = [double]$row.MeanMs; median_ms = [double]$row.MedianMs; iterations = [int]$row.Iterations }
        }
    }
    $map
}

# ── Main ─────────────────────────────────────────────────────────────────────

$repoRoot = if ($RepoRoot) { $RepoRoot } else { Get-RepoRoot }
$absBaseline = if ([IO.Path]::IsPathRooted($BaselinePath)) { $BaselinePath } `
               else { Join-Path $repoRoot $BaselinePath }
$meta = Get-GitMeta -TargetRoot $repoRoot

# ── Save-baseline mode ───────────────────────────────────────────────────────
if ($SaveBaseline) {
    $minWriteTimeUtc = $null
    if (-not $SkipRunner) {
        $minWriteTimeUtc = [DateTime]::UtcNow.AddSeconds(-1)
        Invoke-BenchmarkRunner -TargetRoot $repoRoot -TargetSuite $Suite -ExplicitRunnerPath $RunnerPath
    }
    if ($AllowStaleReport) { $minWriteTimeUtc = $null }

    $report = Read-LatestReport -TargetRoot $repoRoot -ExplicitReportPath $ReportPath -MinWriteTimeUtc $minWriteTimeUtc
    if (-not $report) {
        Write-Error 'No benchmark report found.' -ErrorAction Continue
        exit 1
    }
    Assert-ReportFreshness -Report $report -GitMeta $meta -AllowStale:$AllowStaleReport

    $dir = Split-Path $absBaseline -Parent
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

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
if (-not (Test-Path -LiteralPath $absBaseline)) {
    Write-Error "Baseline file not found at '$absBaseline'. Run with -SaveBaseline first." -ErrorAction Continue
    exit 1
}

$baselineData = Get-Content -LiteralPath $absBaseline -Raw | ConvertFrom-Json -AsHashtable

$minWriteTimeUtc = $null
if (-not $SkipRunner) {
    $minWriteTimeUtc = [DateTime]::UtcNow.AddSeconds(-1)
    Invoke-BenchmarkRunner -TargetRoot $repoRoot -TargetSuite $Suite -ExplicitRunnerPath $RunnerPath
}
if ($AllowStaleReport) { $minWriteTimeUtc = $null }

$report = Read-LatestReport -TargetRoot $repoRoot -ExplicitReportPath $ReportPath -MinWriteTimeUtc $minWriteTimeUtc
if (-not $report) {
    Write-Error 'No benchmark report found after runner execution.' -ErrorAction Continue
    exit 1
}
Assert-ReportFreshness -Report $report -GitMeta $meta -AllowStale:$AllowStaleReport

$currentMap   = ConvertTo-CaseMap $report
$bCases       = $baselineData['cases']
$baselineHash = $baselineData['git_hash']

if (-not $bCases -or $bCases.Count -eq 0) {
    Write-Error "Baseline at '$absBaseline' contains no benchmark cases." -ErrorAction Continue
    exit 1
}

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
$missingCount    = @($rows | Where-Object { $_.Status -eq 'missing' }).Count

# ── Output ───────────────────────────────────────────────────────────────────
switch ($Format) {
    'Json' {
        [ordered]@{ baseline_hash = $baselineHash; suite = $Suite; threshold_pct = $Threshold
                    regression_count = $regressionCount; missing_count = $missingCount; cases = $rows } | ConvertTo-Json -Depth 5
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
        "> **Overall:** $regressionCount regression(s), $missingCount missing | threshold: $Threshold%"
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
        Write-Output "Overall: $regressionCount regression(s), $missingCount missing (threshold: $Threshold%)"
    }
}

if ($missingCount -gt 0) {
    Write-Error "Benchmark gate failed: $missingCount baseline case(s) missing from current report." -ErrorAction Continue
    exit 1
}
if ($FailOnRegression -and $regressionCount -gt 0) { exit 1 }
exit 0

