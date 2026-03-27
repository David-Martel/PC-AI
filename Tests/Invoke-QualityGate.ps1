#Requires -Version 7.0
<#
.SYNOPSIS
    A/B code quality comparison between two git refs for PC_AI.
.DESCRIPTION
    Measures Rust Clippy warnings, Rust test counts, Pester unit test counts, and
    PSScriptAnalyzer violations on HEAD first, then checks out the base ref, measures
    again, restores HEAD, and reports deltas.  Dirty working trees are allowed (warned).
.PARAMETER BaseRef
    Baseline git ref (default: origin/main).
.PARAMETER HeadRef
    Head git ref representing the changes (default: HEAD).
.PARAMETER Format
    Output format: Table (default), Json, or Markdown.
.PARAMETER FailOnRegression
    Exit 1 if any metric regresses.
.PARAMETER ResultsDir
    Directory for result JSON files (relative to repo root).
.EXAMPLE
    .\Tests\Invoke-QualityGate.ps1 -Format Markdown -FailOnRegression
#>
param(
    [string]$BaseRef = 'origin/main',
    [string]$HeadRef = 'HEAD',
    [ValidateSet('Table', 'Json', 'Markdown')][string]$Format = 'Table',
    [switch]$FailOnRegression,
    [string]$ResultsDir = 'Tests/Results'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot    = git -C $PSScriptRoot rev-parse --show-toplevel
$CargoRoot   = Join-Path $RepoRoot 'Native/pcai_core'
$ModulesPath = Join-Path $RepoRoot 'Modules'
$PesterPath  = Join-Path $RepoRoot 'Tests/Unit'
$ResultsDirAbs = if ([System.IO.Path]::IsPathRooted($ResultsDir)) { $ResultsDir } else { Join-Path $RepoRoot $ResultsDir }

# --- metric helpers -----------------------------------------------------------

function Measure-ClippyWarnings {
    Push-Location $CargoRoot
    try {
        $out = & cargo clippy --workspace --all-targets --no-deps --message-format=json `
            -- -D warnings -A clippy::type_complexity 2>&1
        ($out | Where-Object { $_ -match '"level":"warning"' }).Count
    } finally { Pop-Location }
}

function Measure-RustTests {
    Push-Location $CargoRoot
    try {
        $out  = & cargo test --workspace --no-default-features --features server,ffi 2>&1
        $line = $out | Where-Object { $_ -match '^test result:' } | Select-Object -Last 1
        if ($line -match 'test result:.*?(\d+) passed.*?(\d+) failed') {
            return [pscustomobject]@{ Passed = [int]$Matches[1]; Failed = [int]$Matches[2] }
        }
    } finally { Pop-Location }
    [pscustomobject]@{ Passed = 0; Failed = 0 }
}

function Measure-PesterTests {
    if (-not (Get-Module -ListAvailable Pester | Where-Object Version -ge '5.0') -or -not (Test-Path $PesterPath)) {
        return [pscustomobject]@{ Passed = 0; Failed = 0 }
    }
    $cfg = New-PesterConfiguration
    $cfg.Run.Path        = $PesterPath
    $cfg.Filter.Tag      = @('Portable')
    $cfg.Output.Verbosity = 'None'
    $cfg.Run.PassThru    = $true
    $r = Invoke-Pester -Configuration $cfg
    [pscustomobject]@{ Passed = $r.PassedCount; Failed = $r.FailedCount }
}

function Measure-ScriptAnalyzer {
    if (-not (Get-Module -ListAvailable PSScriptAnalyzer) -or -not (Test-Path $ModulesPath)) { return 0 }
    (Invoke-ScriptAnalyzer -Path $ModulesPath -Recurse -Severity Warning, Error).Count
}

function Invoke-Metrics {
    param([string]$Label)
    Write-Host "  Collecting $Label ..." -ForegroundColor Gray
    $c = Measure-ClippyWarnings
    $r = Measure-RustTests
    $p = Measure-PesterTests
    $a = Measure-ScriptAnalyzer
    [pscustomobject]@{
        ClippyWarnings       = $c
        RustTestsPassed      = $r.Passed
        RustTestsFailed      = $r.Failed
        PesterPassed         = $p.Passed
        PesterFailed         = $p.Failed
        PSAnalyzerViolations = $a
    }
}

function Get-DeltaStatus ([string]$Metric, [int]$Delta) {
    if ($Delta -eq 0) { return 'unchanged' }
    $lowerBetter = 'ClippyWarnings','RustTestsFailed','PesterFailed','PSAnalyzerViolations'
    if ($Metric -in $lowerBetter) { return if ($Delta -lt 0) { 'improved' } else { 'regressed' } }
    return if ($Delta -gt 0) { 'improved' } else { 'regressed' }
}

# --- pre-flight ---------------------------------------------------------------

$currentHead = git -C $RepoRoot rev-parse HEAD 2>&1
if ((git -C $RepoRoot status --porcelain 2>&1) -ne '') {
    Write-Warning "Working tree has uncommitted changes — head metrics include them."
}

# --- collect metrics: HEAD first, then base, restore -------------------------

Write-Host "=== Quality Gate: $HeadRef vs $BaseRef ===" -ForegroundColor Cyan
$headMetrics = Invoke-Metrics -Label "HEAD ($HeadRef)"

Write-Host "  Checking out $BaseRef ..." -ForegroundColor Gray
git -C $RepoRoot checkout $BaseRef --quiet
try {
    $baseMetrics = Invoke-Metrics -Label "BASE ($BaseRef)"
} finally {
    Write-Host "  Restoring HEAD ..." -ForegroundColor Gray
    git -C $RepoRoot checkout $currentHead --quiet
}

# --- compute deltas -----------------------------------------------------------

$metricDefs = @(
    @{ Key='ClippyWarnings';       Label='Clippy warnings'             }
    @{ Key='RustTestsPassed';      Label='Rust tests passing'          }
    @{ Key='RustTestsFailed';      Label='Rust tests failing'          }
    @{ Key='PesterPassed';         Label='Pester tests passing'        }
    @{ Key='PesterFailed';         Label='Pester tests failing'        }
    @{ Key='PSAnalyzerViolations'; Label='PSScriptAnalyzer violations' }
)

$deltas      = [ordered]@{}
$regressions = [System.Collections.Generic.List[string]]::new()

foreach ($m in $metricDefs) {
    $base   = [int]$baseMetrics.($m.Key)
    $head   = [int]$headMetrics.($m.Key)
    $delta  = $head - $base
    $status = Get-DeltaStatus -Metric $m.Key -Delta $delta
    $deltas[$m.Key] = [pscustomobject]@{ Label=$m.Label; Base=$base; Head=$head; Delta=$delta; Status=$status }
    if ($status -eq 'regressed') { $regressions.Add($m.Key) }
}

$overall = if ($regressions.Count -gt 0) { 'REGRESSED' }
           elseif (($deltas.Values | Where-Object Status -eq 'improved').Count -gt 0) { 'IMPROVED' }
           else { 'UNCHANGED' }

# --- persist results ----------------------------------------------------------

$null = New-Item -ItemType Directory -Path $ResultsDirAbs -Force
$outFile = Join-Path $ResultsDirAbs "quality-gate-$(Get-Date -Format 'yyyyMMdd_HHmmss').json"

$jsonDeltas = [ordered]@{}
foreach ($kv in $deltas.GetEnumerator()) {
    $jsonDeltas[$kv.Key] = [ordered]@{ base=$kv.Value.Base; head=$kv.Value.Head; delta=$kv.Value.Delta; status=$kv.Value.Status }
}
[ordered]@{
    base_ref    = $BaseRef
    head_ref    = $HeadRef
    timestamp   = (Get-Date -Format 'o')
    deltas      = $jsonDeltas
    overall     = $overall
    regressions = $regressions.ToArray()
} | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding utf8
Write-Host "Results saved: $outFile" -ForegroundColor Gray

# --- render output ------------------------------------------------------------

$statusColor = @{ improved='Green'; unchanged='Gray'; regressed='Red' }

switch ($Format) {
    'Json' { Get-Content $outFile }
    'Markdown' {
        "## Quality Gate: $HeadRef vs $BaseRef`n"
        "| Metric | Base | Head | Delta | Status |"
        "|--------|------|------|-------|--------|"
        foreach ($kv in $deltas.GetEnumerator()) {
            $d = $kv.Value
            "| $($d.Label) | $($d.Base) | $($d.Head) | $(if ($d.Delta -gt 0){'+'}else{''})$($d.Delta) | $($d.Status) |"
        }
        ""
        "**Overall: $overall** ($($regressions.Count) regression(s))"
    }
    default {
        Write-Host ""
        Write-Host ("{0,-35} {1,6} {2,6} {3,7}  {4}" -f 'Metric','Base','Head','Delta','Status') -ForegroundColor Cyan
        Write-Host ('-' * 65) -ForegroundColor DarkGray
        foreach ($kv in $deltas.GetEnumerator()) {
            $d = $kv.Value
            Write-Host ("{0,-35} {1,6} {2,6} {3,7}  {4}" -f $d.Label,$d.Base,$d.Head,"$(if($d.Delta -gt 0){'+'}else{''})$($d.Delta)",$d.Status) `
                -ForegroundColor $statusColor[$d.Status]
        }
        Write-Host ('-' * 65) -ForegroundColor DarkGray
        $oc = if ($overall -eq 'REGRESSED'){'Red'} elseif ($overall -eq 'IMPROVED'){'Green'} else {'Gray'}
        Write-Host "Overall: $overall  ($($regressions.Count) regression(s))" -ForegroundColor $oc
        Write-Host ""
    }
}

if ($FailOnRegression -and $regressions.Count -gt 0) {
    Write-Error "Quality gate FAILED: $($regressions.Count) regression(s): $($regressions -join ', ')"
    exit 1
}
