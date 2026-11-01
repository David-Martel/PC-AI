#Requires -Version 7.0
<#
.SYNOPSIS
Runs fresh, commit-labelled benchmarks and rejects missing or incompatible evidence.
#>
[CmdletBinding()]
param(
    [switch]$SaveBaseline,
    [ValidateRange(0,1000)][int]$Threshold = 15,
    [ValidateSet('Table','Json','Markdown')][string]$Format = 'Table',
    [string]$BaselinePath = '.pcai/benchmarks/baseline.json',
    [string]$Suite = 'quick',
    [switch]$FailOnRegression
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-CaseMap($Report) {
    $map = @{}
    foreach ($row in @($Report.Results)) {
        if ([string]::IsNullOrWhiteSpace($row.CaseId) -or [string]::IsNullOrWhiteSpace($row.Backend)) { throw 'Missing case/backend identity' }
        $key = "$($row.CaseId)/$($row.Backend)"
        if ($map.ContainsKey($key)) { throw "Duplicate measurement: $key" }
        foreach ($metric in @('MeanMs','MedianMs')) {
            $value = [double]$row.$metric
            if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) { throw "Invalid $metric for $key" }
        }
        if ([int]$row.Iterations -le 0) { throw "No successful iterations for $key" }
        $map[$key] = @{ mean_ms=[double]$row.MeanMs; median_ms=[double]$row.MedianMs; iterations=[int]$row.Iterations }
    }
    if ($map.Count -eq 0) { throw 'No benchmark measurements' }
    return $map
}

$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $repoRoot '.git'))) { throw 'Cannot locate repository root' }
$head = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify source commit' }
$absBaseline = if ([IO.Path]::IsPathRooted($BaselinePath)) { $BaselinePath } else { Join-Path $repoRoot $BaselinePath }
$baseline = $null
if (-not $SaveBaseline) {
    if (-not (Test-Path -LiteralPath $absBaseline -PathType Leaf)) { throw 'No baseline; explicitly create and review one with -SaveBaseline' }
    $baseline = Get-Content -LiteralPath $absBaseline -Raw | ConvertFrom-Json -AsHashtable
    if ($baseline.schema_version -ne 2) { throw 'Legacy baseline lacks evidence identity; regenerate explicitly' }
}
$config = Get-Content (Join-Path $repoRoot 'Config/pcai-tooling-benchmarks.json') -Raw | ConvertFrom-Json -AsHashtable
if (-not $config.suites.ContainsKey($Suite)) { throw "Unknown suite: $Suite" }
$required = @($config.suites[$Suite] | Sort-Object -Unique)
if ($required.Count -eq 0) { throw 'Empty required case inventory' }
$runId = [guid]::NewGuid().ToString('N')
$outputRoot = Join-Path $repoRoot 'Reports/benchmark-admission'
$runner = Join-Path $repoRoot 'Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw 'Benchmark runner missing' }
$runnerHash = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash
if (-not $SaveBaseline -and (-not $baseline.ContainsKey('runner_sha256') -or $baseline.runner_sha256 -cne $runnerHash)) { throw 'Baseline benchmark harness differs or is unbound; explicitly review and regenerate with -SaveBaseline' }
& pwsh -NoLogo -NoProfile -File $runner -Suite $Suite -OutputRoot $outputRoot -RunId $runId | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Benchmark runner failed with exit $LASTEXITCODE" }
$reportPath = Join-Path (Join-Path $outputRoot $runId) 'tooling-benchmark-report.json'
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw 'This run produced no report' }
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
$configHash = (Get-FileHash -LiteralPath (Join-Path $repoRoot 'Config/pcai-tooling-benchmarks.json') -Algorithm SHA256).Hash
if ($report.ConfigSha256 -cne $configHash -or $report.RunnerSha256 -cne $runnerHash -or $runnerHash -cne (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash) { throw 'Report input fingerprint mismatch' }
if ($report.SchemaVersion -ne 2 -or $report.RunId -cne $runId -or $report.GitHead -cne $head -or $report.Suite -cne $Suite) { throw 'Report source/run/suite identity mismatch' }
if ($report.MeasurementsComplete -isnot [bool] -or -not $report.MeasurementsComplete) { throw 'Incomplete measurement report' }
if ($report.Machine -cne [Environment]::MachineName -or $report.Architecture -cne [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString() -or $report.HostOS -cne [Runtime.InteropServices.RuntimeInformation]::OSDescription) { throw 'Report host identity mismatch' }
$selected = @($report.SelectedCases | Sort-Object -Unique)
$actual = @($report.Results | ForEach-Object { $_.CaseId } | Sort-Object -Unique)
if (($selected -join '|') -cne ($required -join '|') -or ($actual -join '|') -cne ($required -join '|')) { throw 'Required benchmark case inventory changed or is missing' }
$current = ConvertTo-CaseMap $report
$hostIdentity = @{ machine=$report.Machine; os=$report.HostOS; architecture=$report.Architecture }
if ($SaveBaseline) {
    $parent = Split-Path $absBaseline -Parent
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    @{ schema_version=2; git_hash=$head; run_id=$runId; suite=$Suite; config_sha256=$configHash; runner_sha256=$report.RunnerSha256; host_identity=$hostIdentity; cases=$current } |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $absBaseline -Encoding utf8
    Write-Output "Baseline saved from verified run $runId"
    exit 0
}
if ($baseline.suite -cne $Suite -or $baseline.config_sha256 -cne $configHash) { throw 'Baseline suite or configuration differs' }
# The same case names/configuration can describe a different measurement body.
# Never compare a changed harness or an older baseline that did not bind it.
if (-not $baseline.ContainsKey('runner_sha256') -or $baseline.runner_sha256 -cne $report.RunnerSha256) { throw 'Baseline benchmark harness differs or is unbound; explicitly review and regenerate with -SaveBaseline' }
foreach ($key in @('machine','os','architecture')) {
    if ($baseline.host_identity[$key] -cne $hostIdentity[$key]) { throw "Baseline host differs: $key" }
}
if (($baseline.cases.Keys | Sort-Object) -join '|' -cne (($current.Keys | Sort-Object) -join '|')) { throw 'Baseline case/backend inventory differs; no fallback comparison permitted' }
$rows = foreach ($key in $baseline.cases.Keys) {
    $before = [double]$baseline.cases[$key].mean_ms
    if ([double]::IsNaN($before) -or [double]::IsInfinity($before) -or $before -le 0) { throw "Invalid baseline: $key" }
    $delta = (($current[$key].mean_ms - $before) / $before) * 100
    [pscustomobject]@{ Case=$key; BaselineMs=$before; CurrentMs=$current[$key].mean_ms; DeltaPct=$delta; Regressed=($delta -gt $Threshold) }
}
$regressions = @($rows | Where-Object Regressed).Count
if ($Format -eq 'Json') { @{run_id=$runId; git_hash=$head; regression_count=$regressions; cases=@($rows)} | ConvertTo-Json -Depth 6 }
elseif ($Format -eq 'Markdown') {
    '| Case/backend | Baseline ms | Current ms | Delta % |'; '|---|---:|---:|---:|'
    foreach ($row in $rows) { "| $($row.Case) | $($row.BaselineMs) | $($row.CurrentMs) | $($row.DeltaPct) |" }
} else { $rows | Format-Table | Out-Host }
if ($FailOnRegression -and $regressions -gt 0) { exit 1 }
exit 0
