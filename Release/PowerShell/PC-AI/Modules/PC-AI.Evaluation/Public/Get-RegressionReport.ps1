function Get-RegressionReport {
    <#
    .SYNOPSIS
        Generates a detailed regression report comparing multiple baselines
    #>
    [CmdletBinding()]
    param(
        [EvaluationSuite]$Suite = $script:CurrentSuite,

        [string[]]$BaselineNames
    )

    if (-not $BaselineNames) {
        # Get all baselines
        $baselineDir = $script:EvaluationConfig.BaselinePath
        if (Test-Path $baselineDir) {
            $BaselineNames = Get-ChildItem $baselineDir -Filter "*.json" | ForEach-Object { $_.BaseName }
        }
    }

    $reports = @()
    foreach ($name in $BaselineNames) {
        $report = Test-ForRegression -BaselineName $name -Suite $Suite
        if ($report) {
            $reports += $report
        }
    }

    return @{
        Timestamp = [datetime]::UtcNow
        BaselinesCompared = $reports.Count
        Reports = $reports
        Summary = @{
            TotalRegressions = ($reports | ForEach-Object { $_.Regressions.Count } | Measure-Object -Sum).Sum
            TotalImprovements = ($reports | ForEach-Object { $_.Improvements.Count } | Measure-Object -Sum).Sum
        }
    }
}
