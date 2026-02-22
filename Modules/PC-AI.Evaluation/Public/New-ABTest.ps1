function New-ABTest {
    <#
    .SYNOPSIS
        Creates a new A/B test for comparing inference variants

    .PARAMETER Name
        Name of the A/B test

    .PARAMETER VariantAName
        Name for variant A (e.g., "llamacpp")

    .PARAMETER VariantBName
        Name for variant B (e.g., "mistralrs")
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [string]$VariantAName = "A",

        [string]$VariantBName = "B"
    )

    $test = @{
        Name = $Name
        VariantAName = $VariantAName
        VariantBName = $VariantBName
        VariantAScores = [List[double]]::new()
        VariantBScores = [List[double]]::new()
        StartTime = [datetime]::UtcNow
    }

    $script:ABTests[$Name] = $test

    Write-Host "A/B Test created: $Name ($VariantAName vs $VariantBName)" -ForegroundColor Cyan

    return $test
}
