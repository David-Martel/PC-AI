function Get-ABTestAnalysis {
    <#
    .SYNOPSIS
        Performs statistical analysis on A/B test results

    .PARAMETER TestName
        Name of the A/B test

    .PARAMETER Alpha
        Significance level (default 0.05)
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$TestName,

        [double]$Alpha = 0.05
    )

    if (-not $script:ABTests.ContainsKey($TestName)) {
        Write-Error "A/B test not found: $TestName"
        return
    }

    $test = $script:ABTests[$TestName]

    $aScores = [double[]]$test.VariantAScores.ToArray()
    $bScores = [double[]]$test.VariantBScores.ToArray()

    if ($aScores.Count -lt 2 -or $bScores.Count -lt 2) {
        return @{
            Error = "Insufficient samples (need at least 2 per variant)"
            VariantASamples = $aScores.Count
            VariantBSamples = $bScores.Count
        }
    }

    # Calculate statistics
    $aMean = ($aScores | Measure-Object -Average).Average
    $bMean = ($bScores | Measure-Object -Average).Average
    $aStd = Get-StandardDeviation $aScores
    $bStd = Get-StandardDeviation $bScores

    # Welch's t-test
    $tStat = ($bMean - $aMean) / [math]::Sqrt(($aStd * $aStd / $aScores.Count) + ($bStd * $bStd / $bScores.Count))

    # Degrees of freedom (Welch-Satterthwaite)
    $num = [math]::Pow(($aStd * $aStd / $aScores.Count) + ($bStd * $bStd / $bScores.Count), 2)
    $denom = ([math]::Pow($aStd, 4) / ([math]::Pow($aScores.Count, 2) * ($aScores.Count - 1))) +
             ([math]::Pow($bStd, 4) / ([math]::Pow($bScores.Count, 2) * ($bScores.Count - 1)))
    $df = $num / $denom

    # Approximate p-value using normal distribution for large samples
    $pValue = 2 * (1 - [math]::Min(1, [math]::Abs($tStat) / 2))

    # Effect size (Cohen's d)
    $pooledStd = [math]::Sqrt(($aStd * $aStd + $bStd * $bStd) / 2)
    $cohensD = if ($pooledStd -gt 0) { ($bMean - $aMean) / $pooledStd } else { 0 }

    $effectSize = switch ([math]::Abs($cohensD)) {
        { $_ -lt 0.2 } { "negligible" }
        { $_ -lt 0.5 } { "small" }
        { $_ -lt 0.8 } { "medium" }
        default { "large" }
    }

    $analysis = @{
        TestName = $TestName
        VariantA = @{
            Name = $test.VariantAName
            Samples = $aScores.Count
            Mean = [math]::Round($aMean, 4)
            StdDev = [math]::Round($aStd, 4)
        }
        VariantB = @{
            Name = $test.VariantBName
            Samples = $bScores.Count
            Mean = [math]::Round($bMean, 4)
            StdDev = [math]::Round($bStd, 4)
        }
        Difference = [math]::Round($bMean - $aMean, 4)
        RelativeImprovement = if ($aMean -ne 0) { [math]::Round(($bMean - $aMean) / $aMean * 100, 2) } else { 0 }
        TStatistic = [math]::Round($tStat, 4)
        DegreesOfFreedom = [math]::Round($df, 2)
        PValue = [math]::Round($pValue, 4)
        StatisticallySignificant = $pValue -lt $Alpha
        CohensD = [math]::Round($cohensD, 4)
        EffectSize = $effectSize
        Winner = if ($pValue -lt $Alpha) {
            if ($bMean -gt $aMean) { $test.VariantBName } else { $test.VariantAName }
        } else { "inconclusive" }
        Alpha = $Alpha
    }

    # Display results
    Write-Host "`nA/B Test Analysis: $TestName" -ForegroundColor Cyan
    Write-Host "  $($test.VariantAName): mean=$($analysis.VariantA.Mean), n=$($aScores.Count)"
    Write-Host "  $($test.VariantBName): mean=$($analysis.VariantB.Mean), n=$($bScores.Count)"
    Write-Host "  Difference: $($analysis.Difference) ($($analysis.RelativeImprovement)%)"
    Write-Host "  p-value: $($analysis.PValue) ($(if ($analysis.StatisticallySignificant) { 'significant' } else { 'not significant' }))"
    Write-Host "  Effect size: $effectSize (d=$($analysis.CohensD))"
    Write-Host "  Winner: $($analysis.Winner)" -ForegroundColor $(if ($analysis.Winner -eq 'inconclusive') { 'Yellow' } else { 'Green' })

    return $analysis
}
