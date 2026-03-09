<#
.SYNOPSIS
    Tests the optimizer cmdlets (PowerShell fallback path).
#>

# Source the functions directly
. "$PSScriptRoot/../Modules/PC-AI.Performance/Public/Get-PcaiMemoryPressure.ps1"
. "$PSScriptRoot/../Modules/PC-AI.Performance/Public/Get-PcaiProcessCategories.ps1"
. "$PSScriptRoot/../Modules/PC-AI.Performance/Public/Get-PcaiOptimizationPlan.ps1"

Write-Host '=== Memory Pressure Analysis ===' -ForegroundColor Cyan
$pressure = Get-PcaiMemoryPressure -Verbose
$pressure | Format-List

Write-Host '=== Process Categories ===' -ForegroundColor Cyan
$categories = Get-PcaiProcessCategories
$categories | Format-Table Category, ProcessCount, WorkingSetMB, PrivateMB, HandleCount, TopProcess -AutoSize

Write-Host '=== Optimization Recommendations ===' -ForegroundColor Cyan
$plan = Get-PcaiOptimizationPlan
foreach ($rec in $plan) {
    $color = switch ($rec.Priority) {
        1 { 'Red' }
        2 { 'Yellow' }
        3 { 'DarkYellow' }
        default { 'Gray' }
    }
    Write-Host ("  P{0} [{1,-20}] Saves ~{2,6} MB | {3}" -f $rec.Priority, $rec.Category, $rec.EstimatedSavingsMB, $rec.Description) -ForegroundColor $color
    if ($rec.SafeToAuto) {
        Write-Host "     ^ Safe to automate: $($rec.Action)" -ForegroundColor Green
    }
}

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$totalSavings = ($plan | Measure-Object -Property EstimatedSavingsMB -Sum).Sum
$criticalCount = ($plan | Where-Object { $_.Priority -eq 1 } | Measure-Object).Count
$autoCount = ($plan | Where-Object { $_.SafeToAuto } | Measure-Object).Count
Write-Host "Total estimated reclaimable: ~$([math]::Round($totalSavings / 1024, 1)) GB"
Write-Host "Critical issues: $criticalCount"
Write-Host "Auto-fixable issues: $autoCount"
Write-Host "Pressure level: $($pressure.PressureLevel) (Available: $($pressure.AvailableMB) MB)"

# Save combined report
$report = @{
    Timestamp     = (Get-Date -Format 'o')
    Pressure      = $pressure
    Categories    = $categories
    Recommendations = $plan
    Summary       = @{
        TotalEstimatedSavingsMB = $totalSavings
        CriticalIssueCount      = $criticalCount
        AutoFixableCount        = $autoCount
    }
}
$outPath = Join-Path (Join-Path $PSScriptRoot '..') 'Reports' | Join-Path -ChildPath 'optimization-report.json'
$report | ConvertTo-Json -Depth 5 | Set-Content $outPath -Encoding UTF8
Write-Host "`nFull report saved to: $outPath" -ForegroundColor Green
