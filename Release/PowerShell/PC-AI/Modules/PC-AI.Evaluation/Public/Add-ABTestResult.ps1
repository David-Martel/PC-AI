function Add-ABTestResult {
    <#
    .SYNOPSIS
        Adds a result to an A/B test

    .PARAMETER TestName
        Name of the A/B test

    .PARAMETER Variant
        Which variant: "A" or "B"

    .PARAMETER Score
        Score for this result
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$TestName,

        [Parameter(Mandatory)]
        [ValidateSet("A", "B")]
        [string]$Variant,

        [Parameter(Mandatory)]
        [double]$Score
    )

    if (-not $script:ABTests.ContainsKey($TestName)) {
        Write-Error "A/B test not found: $TestName"
        return
    }

    $test = $script:ABTests[$TestName]

    if ($Variant -eq "A") {
        $test.VariantAScores.Add($Score)
    } else {
        $test.VariantBScores.Add($Score)
    }
}
