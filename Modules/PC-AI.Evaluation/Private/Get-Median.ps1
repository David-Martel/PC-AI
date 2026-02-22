function Get-Median {
    param([double[]]$Values)

    $sorted = $Values | Sort-Object
    $count = $sorted.Count

    if ($count % 2 -eq 0) {
        return ($sorted[$count/2 - 1] + $sorted[$count/2]) / 2
    } else {
        return $sorted[[math]::Floor($count/2)]
    }
}
