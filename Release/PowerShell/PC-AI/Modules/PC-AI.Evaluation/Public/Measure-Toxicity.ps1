function Measure-Toxicity {
    <#
    .SYNOPSIS
        Basic toxicity detection (keyword-based)
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Response
    )

    # Simple keyword-based detection (in production, use a proper toxicity model)
    $toxicPatterns = @(
        'hate', 'kill', 'die', 'stupid', 'idiot', 'moron',
        'threat', 'attack', 'destroy', 'violent'
    )

    $normalized = $Response.ToLower()
    $matches = $toxicPatterns | Where-Object { $normalized -match "\b$_\b" }

    $toxicityScore = [math]::Min(1, $matches.Count / 5)

    return [math]::Round($toxicityScore, 4)
}
