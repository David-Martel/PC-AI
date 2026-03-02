function Get-StandardDeviation {
    param([double[]]$Values)

    if ($Values.Count -lt 2) { return 0 }

    $mean = ($Values | Measure-Object -Average).Average
    $sumSquares = ($Values | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum

    return [math]::Sqrt($sumSquares / ($Values.Count - 1))
}
