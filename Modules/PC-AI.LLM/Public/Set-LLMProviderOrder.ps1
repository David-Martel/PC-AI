#Requires -PSEdition Core
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

    $script:ModuleConfig.ProviderOrder = @($Order)

    $configPath = if ($script:ModuleConfig.ProjectConfigPath) { $script:ModuleConfig.ProjectConfigPath } else { $script:ModuleConfig.ConfigPath }
    if (-not (Test-Path $configPath)) {
        throw "Config file not found: $configPath"
    }

    $config = Get-Content -Path $configPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    if ($config.PSObject.Properties['fallbackOrder']) {
        $config.fallbackOrder = @($Order)
    } else {
        $config | Add-Member -MemberType NoteProperty -Name fallbackOrder -Value @($Order) -Force
    }

    [System.IO.File]::WriteAllText($configPath, ($config | ConvertTo-Json -Depth 20), [System.Text.Encoding]::UTF8)

    Write-Host "Provider order updated: $($Order -join ',')" -ForegroundColor Green
    return [PSCustomObject]@{
        Success = $true
        Order = @($Order)
        ConfigPath = $configPath
    }
}
