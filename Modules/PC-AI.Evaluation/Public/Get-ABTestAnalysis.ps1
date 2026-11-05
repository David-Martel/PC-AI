function Get-ABTestAnalysis {
    <#
    .SYNOPSIS
        Performs statistical analysis on A/B test results

    .DESCRIPTION
        Uses a two-sided Welch test for independent finite sample groups.
        Degenerate variance and unrepresentable summaries return an Error result.
        Effect size retains the existing equal-weight RMS standard-deviation
        convention; it is not a sample-count-weighted pooled estimator.

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

    if (-not [double]::IsFinite($Alpha) -or $Alpha -le 0.0 -or $Alpha -ge 1.0) {
        throw 'Alpha must be finite and strictly between zero and one.'
    }
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

    # A common scale avoids overflowing the variance for finite large observations.
    $scale = 0.0
    foreach ($score in @($aScores) + @($bScores)) {
        if (-not [double]::IsFinite($score)) { return @{ Error='All observations must be finite.' } }
        $scale = [Math]::Max($scale, [Math]::Abs($score))
    }
    if ($scale -eq 0.0) { return @{ Error='Welch inference requires nonzero sample variance.' } }
    $normalizedA = [double[]]@($aScores | ForEach-Object { $_/$scale })
    $normalizedB = [double[]]@($bScores | ForEach-Object { $_/$scale })
    $normalizedMeanA = ($normalizedA | Measure-Object -Average).Average
    $normalizedMeanB = ($normalizedB | Measure-Object -Average).Average
    $normalizedStdA = Get-StandardDeviation $normalizedA
    $normalizedStdB = Get-StandardDeviation $normalizedB
    $varianceA = $normalizedStdA*$normalizedStdA/$aScores.Count
    $varianceB = $normalizedStdB*$normalizedStdB/$bScores.Count
    $variance = $varianceA+$varianceB
    if ($variance -le 0.0) { return @{ Error='Welch inference requires nonzero sample variance.' } }
    $aMean = $normalizedMeanA*$scale
    $bMean = $normalizedMeanB*$scale
    $aStd = $normalizedStdA*$scale
    $bStd = $normalizedStdB*$scale

    # Welch's t-test
    $tStat = ($normalizedMeanB-$normalizedMeanA)/[Math]::Sqrt($variance)

    # Degrees of freedom (Welch-Satterthwaite)
    $shareA = $varianceA/$variance
    $shareB = $varianceB/$variance
    $df = 1.0/($shareA*$shareA/($aScores.Count-1) + $shareB*$shareB/($bScores.Count-1))

    $pValue = Get-StudentTProbability -Statistic $tStat -DegreesOfFreedom $df

    # Effect size (Cohen's d)
    $pooledStd = [Math]::Sqrt(($normalizedStdA*$normalizedStdA + $normalizedStdB*$normalizedStdB)/2.0)
    $cohensD = ($normalizedMeanB-$normalizedMeanA)/$pooledStd
    $relativeImprovement = if ($aMean -ne 0.0) { ($bMean-$aMean)/$aMean*100.0 } else { 0.0 }
    if (-not [double]::IsFinite($aStd) -or -not [double]::IsFinite($bStd) -or -not [double]::IsFinite($bMean-$aMean) -or -not [double]::IsFinite($relativeImprovement)) {
        return @{ Error='Reported summary exceeds the finite numerical domain.' }
    }

    $effectSize = switch ([math]::Abs($cohensD)) {
        { $_ -lt 0.2 } { "negligible"; break }
        { $_ -lt 0.5 } { "small"; break }
        { $_ -lt 0.8 } { "medium"; break }
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
        RelativeImprovement = [math]::Round($relativeImprovement, 2)
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
