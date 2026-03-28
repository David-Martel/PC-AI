#Requires -Version 7.0
<#
.SYNOPSIS
    LLM evaluation regression runner — compare dataset metrics against saved baselines.
.EXAMPLE
    .\Tests\Invoke-EvalRegression.ps1 -Dataset Diagnostic,Safety -FailOnRegression
.EXAMPLE
    .\Tests\Invoke-EvalRegression.ps1 -SaveBaseline
#>
[CmdletBinding()]
param(
    [ValidateSet('All', 'Diagnostic', 'Safety', 'Performance', 'Routing', 'CodeQuality')]
    [string[]]$Dataset = 'All',

    [string]$Backend = 'ollama',

    [string]$Model,

    [switch]$SaveBaseline,

    [string]$BaselinePath = '.pcai/evaluation/baseline.json',

    [ValidateRange(1, 100)]
    [int]$Threshold = 10,

    [ValidateSet('Table', 'Json', 'Markdown')]
    [string]$Format = 'Table',

    [switch]$FailOnRegression,

    [switch]$SkipIfOffline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'Helpers\Resolve-TestRepoRoot.ps1')
$RepoRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot

# --- Resolve paths -----------------------------------------------------------
$LlmConfigPath = Join-Path $RepoRoot 'Config\llm-config.json'
$DatasetDir    = Join-Path $RepoRoot 'Modules\PC-AI.Evaluation\Datasets'
$AbsBaseline   = if ([System.IO.Path]::IsPathRooted($BaselinePath)) {
    $BaselinePath
} else {
    Join-Path $RepoRoot $BaselinePath
}

# --- Resolve backend URL and default model from llm-config.json --------------
$llmConfig  = Get-Content $LlmConfigPath -Raw | ConvertFrom-Json
$providerCfg = $llmConfig.providers.$Backend
if (-not $providerCfg) {
    Write-Error "Backend '$Backend' not found in $LlmConfigPath. Available: $($llmConfig.providers.PSObject.Properties.Name -join ', ')"
    exit 1
}
$BackendBaseUrl   = $providerCfg.baseUrl
$healthUrl        = "$BackendBaseUrl/api/tags"          # Ollama; fallback below
if ($Backend -ne 'ollama') { $healthUrl = "$BackendBaseUrl/health" }
if (-not $Model) { $Model = $providerCfg.defaultModel }

# --- Health check ------------------------------------------------------------
$backendOnline = $false
try {
    $null = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
    $backendOnline = $true
} catch { }

if (-not $backendOnline) {
    $msg = "LLM backend '$Backend' is offline ($healthUrl). Start it before running evaluations."
    if ($SkipIfOffline) {
        Write-Warning $msg
        exit 0
    }
    Write-Error $msg
    exit 1
}

# --- Expand dataset list -----------------------------------------------------
$allNames = @('Diagnostic','Safety','Performance','Routing','CodeQuality')
$targets  = if ($Dataset -contains 'All') { $allNames } else { $Dataset }

# --- Helper: send one prompt, measure latency --------------------------------
function Invoke-LlmPrompt {
    param([string]$Prompt, [int]$TimeoutSec = 60)

    $body = @{
        model  = $Model
        prompt = $Prompt
        stream = $false
    } | ConvertTo-Json

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$BackendBaseUrl/api/generate" `
                    -Method Post -Body $body -ContentType 'application/json' `
                    -TimeoutSec $TimeoutSec -ErrorAction Stop
        $sw.Stop()
        return [pscustomobject]@{ Response = $resp.response; LatencyMs = $sw.ElapsedMilliseconds; Error = $null }
    } catch {
        $sw.Stop()
        return [pscustomobject]@{ Response = ''; LatencyMs = $sw.ElapsedMilliseconds; Error = $_.Exception.Message }
    }
}

# --- Helper: compute similarity score (keyword overlap 0-1) ------------------
function Get-KeywordScore {
    param([string]$Got, [string]$Expected)
    if ([string]::IsNullOrWhiteSpace($Expected)) { return 1.0 }
    $expWords = ($Expected.ToLower() -split '\W+') | Where-Object { $_.Length -gt 3 } | Sort-Object -Unique
    if ($expWords.Count -eq 0) { return 1.0 }
    $gotLower = $Got.ToLower()
    $hits = ($expWords | Where-Object { $gotLower -contains $_ -or $gotLower -match [regex]::Escape($_) }).Count
    return [math]::Round($hits / $expWords.Count, 4)
}

# --- Run each dataset --------------------------------------------------------
$results = [System.Collections.Generic.List[pscustomobject]]::new()

foreach ($name in $targets) {
    $jsonFile = Join-Path $DatasetDir "pcai-$($name.ToLower())-eval.json"
    if (-not (Test-Path $jsonFile)) {
        Write-Warning "Dataset file not found, skipping: $jsonFile"
        continue
    }

    $cases       = Get-Content $jsonFile -Raw | ConvertFrom-Json
    $passed      = 0
    $totalScore  = 0.0
    $totalLatMs  = 0L
    $caseCount   = $cases.Count

    Write-Host "  Evaluating $name ($caseCount cases)..." -ForegroundColor Cyan

    foreach ($case in $cases) {
        $r      = Invoke-LlmPrompt -Prompt $case.prompt
        $score  = if ($r.Error) { 0.0 } else { Get-KeywordScore -Got $r.Response -Expected $case.expected }
        $isPass = (-not $r.Error) -and ($score -ge 0.5)

        if ($isPass) { $passed++ }
        $totalScore += $score
        $totalLatMs += $r.LatencyMs
    }

    $results.Add([pscustomobject]@{
        dataset            = $name.ToLower()
        cases_total        = $caseCount
        cases_passed       = $passed
        pass_rate          = if ($caseCount -gt 0) { [math]::Round($passed / $caseCount, 4) } else { 0.0 }
        average_latency_ms = if ($caseCount -gt 0) { [math]::Round($totalLatMs / $caseCount) } else { 0 }
        average_score      = if ($caseCount -gt 0) { [math]::Round($totalScore / $caseCount, 4) } else { 0.0 }
        backend            = $Backend
        model              = $Model
    })
}

if ($results.Count -eq 0) {
    Write-Warning 'No datasets were evaluated.'
    exit 0
}

# --- Save baseline if requested ----------------------------------------------
if ($SaveBaseline) {
    $baselineDir = Split-Path $AbsBaseline -Parent
    if (-not (Test-Path $baselineDir)) { New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null }
    @{
        saved_at = [datetime]::UtcNow.ToString('o')
        backend  = $Backend
        model    = $Model
        datasets = $results
    } | ConvertTo-Json -Depth 5 | Set-Content -Path $AbsBaseline -Encoding UTF8
    Write-Host "Baseline saved: $AbsBaseline" -ForegroundColor Green
    exit 0
}

# --- Compare against baseline ------------------------------------------------
$regressionCount = 0
$report = [System.Collections.Generic.List[pscustomobject]]::new()

$baseline = $null
if (Test-Path $AbsBaseline) {
    $baseline = (Get-Content $AbsBaseline -Raw | ConvertFrom-Json).datasets
}

foreach ($r in $results) {
    $base = $baseline | Where-Object { $_.dataset -eq $r.dataset } | Select-Object -First 1
    $prDelta = $null; $asDelta = $null; $regression = $false

    if ($base) {
        $prDelta    = [math]::Round(($r.pass_rate - $base.pass_rate) * 100, 1)
        $asDelta    = [math]::Round(($r.average_score - $base.average_score) * 100, 1)
        $regression = ($prDelta -lt -$Threshold) -or ($asDelta -lt -$Threshold)
        if ($regression) { $regressionCount++ }
    }

    $report.Add([pscustomobject]@{
        Dataset          = $r.dataset
        Cases            = "$($r.cases_passed)/$($r.cases_total)"
        PassRate         = '{0:P1}' -f $r.pass_rate
        AvgScore         = '{0:F3}' -f $r.average_score
        AvgLatencyMs     = $r.average_latency_ms
        PassRateDelta    = if ($null -ne $prDelta) { ('{0:+0.0;-0.0;0.0}' -f $prDelta) + '%' } else { 'no baseline' }
        AvgScoreDelta    = if ($null -ne $asDelta) { ('{0:+0.0;-0.0;0.0}' -f $asDelta) + '%' } else { 'no baseline' }
        Status           = if ($regression) { 'REGRESSION' } elseif ($null -eq $base) { 'NEW' } else { 'OK' }
    })
}

# --- Render output -----------------------------------------------------------
switch ($Format) {
    'Json'     { $report | ConvertTo-Json -Depth 4 }
    'Markdown' {
        '| Dataset | Cases | Pass Rate | Avg Score | Lat ms | PR Delta | Score Delta | Status |'
        '|---------|-------|-----------|-----------|--------|----------|-------------|--------|'
        foreach ($row in $report) {
            "| $($row.Dataset) | $($row.Cases) | $($row.PassRate) | $($row.AvgScore) | $($row.AvgLatencyMs) | $($row.PassRateDelta) | $($row.AvgScoreDelta) | $($row.Status) |"
        }
    }
    default    { $report | Format-Table -AutoSize }
}

# --- Final verdict -----------------------------------------------------------
if ($regressionCount -gt 0) {
    Write-Host "`nREGRESSION: $regressionCount dataset(s) declined beyond ${Threshold}% threshold." -ForegroundColor Red
    if ($FailOnRegression) { exit 1 }
} else {
    Write-Host "`nAll datasets within regression threshold ($Threshold%)." -ForegroundColor Green
}
