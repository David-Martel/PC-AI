function Get-PcCommandMapCacheKey {
    param(
        [Parameter(Mandatory)]
        [string]$ModulesRoot,
        [Parameter(Mandatory)]
        [System.IO.FileInfo[]]$Manifests
    )

    if (-not $Manifests -or $Manifests.Count -eq 0) {
        return "$ModulesRoot|0|0"
    }

    $latest = ($Manifests | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc.Ticks
    return "$ModulesRoot|$($Manifests.Count)|$latest"
}
