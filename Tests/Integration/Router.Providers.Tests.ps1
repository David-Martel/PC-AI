<#
.SYNOPSIS
    Integration tests for routed chat provider selection (Ollama/vLLM).
#>

BeforeAll {
    $script:PcaiRoot = Join-Path $PSScriptRoot '..\..'
    $script:ModulePath = Join-Path $script:PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $script:ModulePath -Force -ErrorAction Stop
}

Describe "Invoke-LLMChatRouted provider selection" -Tag 'Integration', 'Router', 'Provider' {
    BeforeEach {
        Mock Invoke-FunctionGemmaReAct {
            [PSCustomObject]@{ ToolCalls = @(); ToolResults = @() }
        } -ModuleName PC-AI.LLM
    }

    It "Should route to Ollama when specified" {
        Mock Get-CachedProviderHealth { $true } -ModuleName PC-AI.LLM
        Mock Invoke-OllamaChat {
            [PSCustomObject]@{ message = @{ content = 'ollama response' } }
        } -ModuleName PC-AI.LLM

        $result = Invoke-LLMChatRouted -Message "Hi" -Mode chat -Provider ollama
        $result.Provider | Should -Be 'ollama'
        $result.Response | Should -Be 'ollama response'
    }

    It "Should route to vLLM when specified" {
        Mock Get-CachedProviderHealth { $true } -ModuleName PC-AI.LLM
        Mock Invoke-OpenAIChat {
            [PSCustomObject]@{ message = @{ content = 'vllm response' } }
        } -ModuleName PC-AI.LLM

        $result = Invoke-LLMChatRouted -Message "Hi" -Mode chat -Provider vllm
        $result.Provider | Should -Be 'vllm'
        $result.Response | Should -Be 'vllm response'
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
