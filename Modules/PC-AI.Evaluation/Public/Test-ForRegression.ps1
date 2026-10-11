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

    if (-not [double]::IsFinite($Threshold) -or $Threshold -lt 0.0) {
        throw 'Regression threshold must be finite and nonnegative.'
    }
    if ([string]::IsNullOrWhiteSpace($BaselineName) -or $BaselineName.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
        $BaselineName -ne $BaselineName.TrimEnd(' ','.') -or $BaselineName -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
        throw 'Baseline name must be an ordinary filename leaf.'
    }
    if ($Suite.Results.Count -eq 0 -or $Suite.Metrics.Count -eq 0 -or
        ($Suite.TestCases.Count -gt 0 -and $Suite.Results.Count -ne $Suite.TestCases.Count)) {
        throw 'Regression comparison requires nonempty completed results.'
    }
    foreach ($result in $Suite.Results) {
        if ($result.Status -notin @('pass','fail') -or -not [double]::IsFinite($result.OverallScore)) {
            throw 'Regression comparison cannot contain failed execution or nonfinite scores.'
        }
        foreach ($metric in $Suite.Metrics) {
            $value=$result.Metrics[$metric.Name]
            if ($null -eq $value -or $value -is [bool] -or $value -isnot [ValueType] -or -not [double]::IsFinite([double]$value)) {
                throw 'Regression comparison requires finite numeric observations.'
            }
        }
    }
    # Load baseline
    $baselinePath = Join-Path $script:EvaluationConfig.BaselinePath "$BaselineName.json"
    if (-not (Test-Path -LiteralPath $baselinePath)) {
        Write-Error "Baseline not found: $BaselineName"
        return $null
    }

    $baseline = Get-Content -LiteralPath $baselinePath | ConvertFrom-Json -AsHashtable

    # Compare metrics
    $currentMetrics = Get-EvaluationResults -Suite $Suite -Format metrics
    $baselineMetrics = $baseline.Metrics
    if ($baselineMetrics -isnot [Collections.IDictionary] -or $currentMetrics.Count -eq 0) {
        throw 'Regression comparison requires metric distributions.'
    }

    $regressions = @()
    $improvements = @()

    foreach ($metricName in $currentMetrics.Keys) {
        $current = $currentMetrics[$metricName].Mean
        $base = $baselineMetrics[$metricName]

        if ($null -eq $base) { throw "Baseline lacks current metric $metricName." }
        $baseMean = if ($base -is [hashtable]) { $base.Mean } else { $base }
        if ($null -eq $baseMean -or $baseMean -is [bool] -or $baseMean -isnot [ValueType] -or
            -not [double]::IsFinite([double]$baseMean) -or [double]$baseMean -eq 0.0 -or -not [double]::IsFinite([double]$current)) {
            throw 'Relative regression requires finite means and a nonzero baseline.'
        }

        $change = ($current - $baseMean) / [math]::Abs($baseMean)
        if (-not [double]::IsFinite([double]$change) -or -not [double]::IsFinite([double]($change*100.0))) {
            throw 'Relative regression exceeds the finite numerical domain.'
        }

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
