[CmdletBinding()]
param(
    [int]$LookbackMinutes = 120,
    [string]$ProfilePath = (Join-Path $PSScriptRoot '..' 'Config' 'process-lasso.ai-dev-workstation.json'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..' 'Reports' 'process-lasso')
)

$ErrorActionPreference = 'Stop'

$modulePath = Join-Path $PSScriptRoot '..' 'Release' 'PowerShell' 'PC-AI' 'Modules' 'PC-AI.Acceleration' 'PC-AI.Acceleration.psd1'
Import-Module $modulePath -Force

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputDir = Join-Path $OutputRoot $timestamp
$null = New-Item -ItemType Directory -Path $outputDir -Force

$snapshot = Get-ProcessLassoSnapshot -LookbackMinutes $LookbackMinutes
$overlayPath = Join-Path $outputDir 'process-lasso.overlay.ini'
$overlay = New-ProcessLassoOverlay -ProfilePath $ProfilePath -OutputPath $overlayPath -PassThru
$profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json -Depth 10

$counterSamples = Get-Counter '\Memory\% Committed Bytes In Use','\Memory\Pages/sec','\Processor Information(_Total)\% Processor Utility'
$counterMap = @{}
foreach ($sample in $counterSamples.CounterSamples) {
    $counterMap[$sample.Path] = [math]::Round($sample.CookedValue, 2)
}

$topMemory = Get-Process |
    Sort-Object PrivateMemorySize64 -Descending |
    Select-Object -First 15 ProcessName, Id, PriorityClass,
        @{ Name = 'PrivateMB'; Expression = { [math]::Round($_.PrivateMemorySize64 / 1MB, 1) } },
        @{ Name = 'WorkingSetMB'; Expression = { [math]::Round($_.WorkingSet64 / 1MB, 1) } }

$report = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    profile_path = (Resolve-Path $ProfilePath).Path
    counters     = $counterMap
    snapshot     = $snapshot
    top_memory   = $topMemory
}

$reportJsonPath = Join-Path $outputDir 'process-lasso-analysis.json'
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportJsonPath -Encoding UTF8

$recommendations = [System.Collections.Generic.List[string]]::new()
if (($counterMap.Values | Select-Object -First 1) -ne $null) {
    $commitPath = ($counterMap.Keys | Where-Object { $_ -like '*% committed bytes in use' } | Select-Object -First 1)
    $pagesPath = ($counterMap.Keys | Where-Object { $_ -like '*pages/sec' } | Select-Object -First 1)
    if ($commitPath -and $counterMap[$commitPath] -ge $profile.analysis.counterThresholds.commitUsageWarningPercent) {
        $recommendations.Add("Commit usage is above the profile warning threshold. Review background AI, Docker, and WSL memory pressure first.")
    }
    if ($pagesPath -and $counterMap[$pagesPath] -ge $profile.analysis.counterThresholds.pagesPerSecondWarning) {
        $recommendations.Add("Paging is above the profile warning threshold. Avoid broad ProBalance exclusions and keep background LLM workloads contained.")
    }
}
if ($snapshot.log_summary.efficiency_mode_events -gt $profile.analysis.ruleNoiseThresholds.maxRepeatedEventsPerHour) {
    $recommendations.Add("Efficiency Mode OFF events are excessively noisy. Consider narrowing the disable list or suppressing this log category during normal operation.")
}

$markdown = @"
# Process Lasso Analysis

- Generated: $($report.generated_at)
- Profile: $($report.profile_path)
- Output: $outputDir

## Current State

- StartWithPowerPlan: $($snapshot.summary.start_with_power_plan)
- TargetPowerPlan: $($snapshot.summary.target_power_plan)
- OocExclusions: $([string]::Join(', ', @($snapshot.summary.ooc_exclusions)))
- SmartTrimExclusions: $([string]::Join(', ', @($snapshot.summary.smart_trim_exclusions)))
- EfficiencyModeOff: $([string]::Join(', ', @($snapshot.summary.efficiency_mode_off)))
- LogEfficiencyMode: $($snapshot.summary.log_efficiency_mode)
- LogCPUSets: $($snapshot.summary.log_cpu_sets)

## Recent Log Activity

- Total events: $($snapshot.log_summary.total_events)
- Efficiency Mode events: $($snapshot.log_summary.efficiency_mode_events)
- CPU Set events: $($snapshot.log_summary.cpu_set_events)
- SmartTrim events: $($snapshot.log_summary.smart_trim_events)
- Power profile events: $($snapshot.log_summary.power_profile_events)

## Recommended Focus

$(([string]::Join([Environment]::NewLine, ($recommendations | ForEach-Object { "- $_" }))))
"@

$markdownPath = Join-Path $outputDir 'process-lasso-analysis.md'
Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding UTF8

Write-Host "Wrote Process Lasso analysis to $outputDir" -ForegroundColor Green
