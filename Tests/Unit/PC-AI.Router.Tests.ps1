<#
.SYNOPSIS
    Unit tests for routed chat and JSON enforcement.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
}

Describe "Invoke-LLMChatRouted" -Tag 'Unit', 'LLM', 'Router', 'Portable' {
    BeforeEach {
        Mock Invoke-FunctionGemmaReAct {
            [PSCustomObject]@{
                ToolCalls = @()
                ToolResults = @()
            }
        } -ModuleName PC-AI.LLM
    }

    It "Should parse JSON in diagnose mode" {
        Mock Invoke-LLMChatWithFallback {
            [PSCustomObject]@{
                Provider = 'ollama'
                message = @{ content = '{"diagnosis_version":"2.0.0","timestamp":"2026-01-27T00:00:00Z","model_id":"test","environment":{"os_version":"Windows","pcai_tooling":"Test"},"summary":[],"findings":[],"recommendations":[],"what_is_missing":[]}' }
            }
        } -ModuleName PC-AI.LLM

        $result = Invoke-LLMChatRouted -Message "test" -Mode diagnose
        $result.JsonValid | Should -Be $true
        $result.ResponseJson.diagnosis_version | Should -Be '2.0.0'
    }

    It "Should throw when diagnose mode JSON is invalid" {
        Mock Invoke-LLMChatWithFallback {
            [PSCustomObject]@{
                Provider = 'ollama'
                message = @{ content = 'not json' }
            }
        } -ModuleName PC-AI.LLM

        { Invoke-LLMChatRouted -Message "test" -Mode diagnose -ErrorAction Stop } | Should -Throw
    }

    It "Should not enforce JSON in chat mode" {
        Mock Invoke-LLMChatWithFallback {
            [PSCustomObject]@{
                Provider = 'ollama'
                message = @{ content = 'plain text response' }
            }
        } -ModuleName PC-AI.LLM

        $result = Invoke-LLMChatRouted -Message "test" -Mode chat
        $result.Response | Should -Be 'plain text response'
        $result.JsonValid | Should -Be $false
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
