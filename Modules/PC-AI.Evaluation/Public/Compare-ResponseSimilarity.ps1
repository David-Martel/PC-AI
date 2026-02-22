function Compare-ResponseSimilarity {
    <#
    .SYNOPSIS
        Compares semantic similarity between response and expected output

    .DESCRIPTION
        Uses word overlap and embedding-based similarity when available
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Response,

        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Expected
    )

    # Handle empty strings
    if ([string]::IsNullOrWhiteSpace($Response) -and [string]::IsNullOrWhiteSpace($Expected)) {
        return 1.0  # Both empty = identical
    }
    if ([string]::IsNullOrWhiteSpace($Response) -or [string]::IsNullOrWhiteSpace($Expected)) {
        return 0.0  # One empty = no similarity
    }

    # Normalize text
    $respNorm = $Response.ToLower() -replace '[^\w\s]', '' -replace '\s+', ' '
    $expNorm = $Expected.ToLower() -replace '[^\w\s]', '' -replace '\s+', ' '

    # Word overlap (Jaccard similarity)
    $respWords = $respNorm -split '\s+' | Where-Object { $_.Length -gt 2 }
    $expWords = $expNorm -split '\s+' | Where-Object { $_.Length -gt 2 }

    $intersection = ($respWords | Where-Object { $_ -in $expWords }).Count
    $union = ($respWords + $expWords | Select-Object -Unique).Count

    $jaccard = if ($union -gt 0) { $intersection / $union } else { 0 }

    # N-gram overlap (bigrams)
    $respBigrams = Get-NGrams -Text $respNorm -N 2
    $expBigrams = Get-NGrams -Text $expNorm -N 2

    $bigramIntersection = ($respBigrams | Where-Object { $_ -in $expBigrams }).Count
    $bigramUnion = ($respBigrams + $expBigrams | Select-Object -Unique).Count

    $bigramSimilarity = if ($bigramUnion -gt 0) { $bigramIntersection / $bigramUnion } else { 0 }

    # Combined score (weighted average)
    $similarity = ($jaccard * 0.4) + ($bigramSimilarity * 0.6)

    return [math]::Round($similarity, 4)
}
