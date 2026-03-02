function Measure-Coherence {
    <#
    .SYNOPSIS
        Measures logical coherence of response
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [AllowEmptyString()]
        [string]$Response
    )

    if (-not $Response) {
        return 0
    }

    $score = 1.0

    # Check for abrupt endings
    if ($Response -notmatch '[.!?]$') {
        $score -= 0.1
    }

    # Check for repeated phrases
    $sentences = $Response -split '[.!?]' | Where-Object { $_.Trim().Length -gt 10 }
    if ($sentences.Count -gt 1) {
        $uniqueRatio = ($sentences | Select-Object -Unique).Count / $sentences.Count
        if ($uniqueRatio -lt 0.8) {
            $score -= (1 - $uniqueRatio) * 0.3
        }
    }

    # Check for sentence length variance (good writing has varied sentence lengths)
    $lengths = $sentences | ForEach-Object { ($_ -split '\s+').Count }
    if ($lengths.Count -gt 2) {
        $variance = Get-StandardDeviation $lengths
        if ($variance -lt 2) {
            $score -= 0.1  # Too uniform
        }
    }

    return [math]::Round([math]::Max(0, $score), 4)
}
