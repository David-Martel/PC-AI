#Requires -PSEdition Core
<#
.SYNOPSIS
    Updates the LLM provider fallback order in the canonical JSON config.
.DESCRIPTION
    Validates every provider before writing. Supported providers are ollama,
    pcai-inference, vllm and lmstudio. The legacy aliases pcai-native (ollama)
    and functiongemma (vllm) remain supported. Other configuration is preserved.
    Publication currently requires Windows. Other platforms reject writes before
    mutation pending qualified atomic exchange and metadata preservation.
#>
function Set-LLMProviderOrder {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string[]]$Order
    )

    Assert-LLMProviderOrder -Order $Order

    $configPath = if ($script:ModuleConfig.ProjectConfigPath) { $script:ModuleConfig.ProjectConfigPath } else { $script:ModuleConfig.ConfigPath }
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Config file not found: $configPath"
    }

    $snapshot = Read-LLMConfigSnapshot -Path $configPath
    $config = $snapshot.Configuration
    if ($config.PSObject.Properties['fallbackOrder']) {
        $config.fallbackOrder = @($Order)
    } else {
        $config | Add-Member -MemberType NoteProperty -Name fallbackOrder -Value @($Order) -Force
    }

    if (-not $PSCmdlet.ShouldProcess($configPath, "Set LLM provider order to $($Order -join ',')")) {
        return
    }

    Save-LLMConfigAtomically -Configuration $config -Snapshot $snapshot
    $script:ModuleConfig.ProviderOrder = @($Order)

    Write-Host "Provider order updated: $($Order -join ',')" -ForegroundColor Green
    return [PSCustomObject]@{
        Success = $true
        Order = @($Order)
        ConfigPath = $configPath
    }
}
