# Tests/Unit/LLM-Fallback.Tests.ps1
BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
}

Describe 'Invoke-LLMChatWithFallback' -Tag 'Unit', 'LLM', 'Fallback', 'Portable' {
    It 'Falls back to the next provider when the first is unhealthy' {
        Mock Get-CachedProviderHealth {
            param([string]$Provider, [int]$TimeoutSeconds, [string]$ApiUrl)
            if ($Provider -eq 'pcai-inference') { return $false }
            return $true
        } -ModuleName PC-AI.LLM

        Mock Invoke-OpenAIChat {
            return [PSCustomObject]@{
                message = [PSCustomObject]@{ content = 'ok' }
                provider = 'openai'
            }
        } -ModuleName PC-AI.LLM

        InModuleScope PC-AI.LLM {
            $script:ModuleConfig.ProviderOrder = @('pcai-inference', 'vllm')
            $script:ModuleConfig.VLLMApiUrl = 'http://test'
            $script:ModuleConfig.VLLMModel = 'test-model'

            $result = Invoke-LLMChatWithFallback -Messages @(@{ role = 'user'; content = 'hi' }) -Model 'test-model' -Provider 'auto'
            $result.Provider | Should -Be 'vllm'
            $result.message.content | Should -Be 'ok'
        }

        Should -Invoke Invoke-OpenAIChat -ModuleName PC-AI.LLM -ParameterFilter {
            $ApiUrl -eq 'http://test'
        }
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
