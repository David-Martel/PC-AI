#Requires -Version 5.1
<#
.SYNOPSIS
    Updates the LLM provider fallback order via PcaiServiceHost.
#>
function Set-LLMProviderOrder {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string[]]$Order
    )

    $orderValue = $Order -join ','
    $result = Invoke-PcaiServiceHost -ServerArgs @('provider','set-order', $orderValue)
    if (-not $result.Success) {
        throw "Failed to update provider order: $($result.Output)"
    }

    Write-Host "Provider order updated: $orderValue" -ForegroundColor Green
    return $result
}
