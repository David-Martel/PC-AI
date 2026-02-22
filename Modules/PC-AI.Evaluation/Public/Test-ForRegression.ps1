function Test-ForRegression {
    <#
    .SYNOPSIS
        Tests current performance against baseline for regressions

    .PARAMETER BaselineName
        Name of baseline to compare against

    .PARAMETER Suite
        Current evaluation suite with results

    .PARAMETER Threshold
        Regression threshold (default 0.05 = 5% degradation)
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$BaselineName,

        [Parameter(Mandatory)]
        [EvaluationSuite]$Suite,

        [double]$Threshold = 0.05
    )

    # Load baseline
    $baselinePath = Join-Path $script:EvaluationConfig.BaselinePath "$BaselineName.json"
    if (-not (Test-Path $baselinePath)) {
        Write-Error "Baseline not found: $BaselineName"
        return $null
    }

    $baseline = Get-Content $baselinePath | ConvertFrom-Json -AsHashtable

    # Compare metrics
    $currentMetrics = Get-EvaluationResults -Suite $Suite -Format metrics
    $baselineMetrics = $baseline.Metrics

    $regressions = @()
    $improvements = @()

    foreach ($metricName in $currentMetrics.Keys) {
        $current = $currentMetrics[$metricName].Mean
        $base = $baselineMetrics[$metricName]

        if ($null -eq $base) { continue }
        $baseMean = if ($base -is [hashtable]) { $base.Mean } else { $base }

        $change = ($current - $baseMean) / [math]::Abs($baseMean)

        # For metrics where lower is better (latency, memory, toxicity)
        $lowerIsBetter = $metricName -in @('latency', 'memory', 'toxicity')

        $isRegression = if ($lowerIsBetter) {
            $change -gt $Threshold  # Increase is bad
        } else {
            $change -lt -$Threshold  # Decrease is bad
        }

        $isImprovement = if ($lowerIsBetter) {
            $change -lt -$Threshold
        } else {
            $change -gt $Threshold
        }

        if ($isRegression) {
            $regressions += @{
                Metric = $metricName
                Baseline = $baseMean
                Current = $current
                Change = [math]::Round($change * 100, 2)
            }
        } elseif ($isImprovement) {
            $improvements += @{
                Metric = $metricName
                Baseline = $baseMean
                Current = $current
                Change = [math]::Round($change * 100, 2)
            }
        }
    }

    $result = @{
        BaselineName = $BaselineName
        BaselineDate = $baseline.Timestamp
        HasRegressions = $regressions.Count -gt 0
        Regressions = $regressions
        Improvements = $improvements
        Threshold = "$([math]::Round($Threshold * 100, 1))%"
    }

    # Display results
    if ($regressions.Count -gt 0) {
        Write-Host "`nREGRESSIONS DETECTED!" -ForegroundColor Red
        foreach ($reg in $regressions) {
            Write-Host "  $($reg.Metric): $($reg.Baseline) -> $($reg.Current) ($($reg.Change)%)" -ForegroundColor Red
        }
    } else {
        Write-Host "`nNo regressions detected" -ForegroundColor Green
    }

    if ($improvements.Count -gt 0) {
        Write-Host "`nImprovements:" -ForegroundColor Green
        foreach ($imp in $improvements) {
            Write-Host "  $($imp.Metric): $($imp.Baseline) -> $($imp.Current) ($($imp.Change)%)" -ForegroundColor Green
        }
    }

    return $result
}
