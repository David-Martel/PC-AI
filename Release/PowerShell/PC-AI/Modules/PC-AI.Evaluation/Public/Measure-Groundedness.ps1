function Measure-Groundedness {
    <#
    .SYNOPSIS
        Measures how well the response is grounded in provided context
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Response,

        [Parameter(Mandatory)]
        [string]$Context
    )

    # Extract key phrases from context
    $contextWords = $Context.ToLower() -replace '[^\w\s]', '' -split '\s+' | Where-Object { $_.Length -gt 3 }
    $responseWords = $Response.ToLower() -replace '[^\w\s]', '' -split '\s+' | Where-Object { $_.Length -gt 3 }

    # Check what percentage of response words appear in context
    $grounded = ($responseWords | Where-Object { $_ -in $contextWords }).Count
    $total = $responseWords.Count

    $groundednessScore = if ($total -gt 0) { $grounded / $total } else { 0 }

    # Penalize if response contains many words not in context
    $ungrounded = $total - $grounded
    $penalty = [math]::Min(0.3, $ungrounded / 100)

    return [math]::Round([math]::Max(0, $groundednessScore - $penalty), 4)
}
