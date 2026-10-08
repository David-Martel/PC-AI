#Requires -PSEdition Core

function Assert-LLMProviderOrder {
    <#
    .SYNOPSIS
        Rejects unsupported providers before configuration persistence.
    .DESCRIPTION
        Accepts the chat backends and the legacy aliases resolved by
        Invoke-LLMChatWithFallback. An empty order or unknown entry is an error.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [ValidateSet('ollama', 'pcai-inference', 'vllm', 'lmstudio', 'pcai-native', 'functiongemma')]
        [string[]]$Order
    )
}
