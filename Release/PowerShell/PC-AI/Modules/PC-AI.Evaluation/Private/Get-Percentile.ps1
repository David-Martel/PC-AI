function Get-Percentile {
    param([double[]]$Values, [int]$Percentile)

    $sorted = $Values | Sort-Object
    $index = [math]::Ceiling($Percentile / 100 * $sorted.Count) - 1
    $index = [math]::Max(0, [math]::Min($index, $sorted.Count - 1))

    return $sorted[$index]
}
