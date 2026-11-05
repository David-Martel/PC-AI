<#
.SYNOPSIS
    Unit tests for routed chat and JSON enforcement.
#>

Describe "Invoke-LLMChatRouted" -Tag 'Unit', 'LLM', 'Router', 'Portable' {
    BeforeAll {
        $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
        Import-Module $ModulePath -Force -ErrorAction Stop
    }
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
    AfterAll {
        Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
    }
}

Describe 'Actual routed diagnosis JSON contract' -Tag 'Unit', 'LLM', 'Router', 'Portable', 'DiagnoseJsonContract' {
    BeforeAll {
        $sourceRoot = Join-Path $PSScriptRoot '../..'
        # Full maintained router/parser bodies; only external collaborators are inert.
        # No aggregate import, provider, health probe, native load or tool execution.
        $script:JsonContractModule = New-Module -Name PcaiRouterJsonContractFixture -ArgumentList $sourceRoot -ScriptBlock {
            param($SourceRoot)
            $script:ModuleConfig = @{ DefaultModel = 'fixture'; DefaultTimeout = 120; RouterApiUrl = 'https://fixture.invalid'; RouterModel = 'fixture' }
            function Get-EnrichedSystemPrompt { param($Mode) 'inert fixture prompt' }
            function Resolve-PcaiEndpoint { param($ApiUrl, $ProviderName) $ApiUrl }
            function Invoke-LLMChatWithFallback {
                param($Messages, $Model, $Temperature, $TimeoutSeconds, $Provider)
                [pscustomobject]@{ Provider = 'fixture'; message = @{ content = $script:FixtureContent } }
            }
            . (Join-Path $SourceRoot 'Modules/PC-AI.LLM/Private/ConvertFrom-LLMJson.ps1')
            . (Join-Path $SourceRoot 'Modules/PC-AI.LLM/Public/Invoke-LLMChatRouted.ps1')
        }
    }

    # Protects: enforced diagnosis never reports an absent response as valid JSON.
    # Detects: parser's permissive empty-input return becoming JsonValid=true.
    # Needs: real router/parser with only final-provider/prompt/endpoint boundaries inert.
    # Breadcrumb: source-bound empty-response reproduction in the fleet signal audit.
    It 'rejects <Label> diagnosis output when JSON is enforced' -ForEach @(
        @{ Label = 'empty'; Content = '' }, @{ Label = 'whitespace'; Content = " `t`r`n" }
    ) {
        & $script:JsonContractModule { param($Value) $script:FixtureContent = $Value } $Content
        { & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode diagnose -BypassRouter -ToolsPath 'inert' } } |
            Should -Throw '*Diagnose mode requires valid JSON output*'
    }

    It 'marks <Label> output invalid without throwing when enforcement is disabled' -ForEach @(
        @{ Label = 'empty'; Content = '' }, @{ Label = 'whitespace'; Content = '   ' }
    ) {
        & $script:JsonContractModule { param($Value) $script:FixtureContent = $Value } $Content
        $result = & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode diagnose -BypassRouter -ToolsPath 'inert' -EnforceJson:$false }
        $result.JsonValid | Should -BeFalse
        $result.JsonError | Should -Not -BeNullOrEmpty
        $result.ResponseJson | Should -BeNullOrEmpty
    }

    It 'continues parsing populated JSON through the real parser' {
        & $script:JsonContractModule { $script:FixtureContent = '{"diagnosis_version":"2.0.0","findings":[],"recommendations":[]}' }
        $result = & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode diagnose -BypassRouter -ToolsPath 'inert' }
        $result.JsonValid | Should -BeTrue
        $result.ResponseJson.diagnosis_version | Should -BeExactly '2.0.0'
        $result.Provider | Should -BeExactly 'fixture'
    }

    It 'continues rejecting nonempty malformed JSON' {
        & $script:JsonContractModule { $script:FixtureContent = 'not json' }
        { & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode diagnose -BypassRouter -ToolsPath 'inert' } } |
            Should -Throw '*Diagnose mode requires valid JSON output*'
    }

    It 'preserves literal JSON null semantics' {
        & $script:JsonContractModule { $script:FixtureContent = 'null' }
        $result = & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode diagnose -BypassRouter -ToolsPath 'inert' }
        $result.JsonValid | Should -BeTrue
        $result.ResponseJson | Should -BeNullOrEmpty
    }

    It 'preserves ordinary empty chat output' {
        & $script:JsonContractModule { $script:FixtureContent = '' }
        $result = & $script:JsonContractModule { Invoke-LLMChatRouted -Message 'fixture' -Mode chat -BypassRouter -ToolsPath 'inert' }
        $result.Response | Should -BeExactly ''
        $result.JsonValid | Should -BeFalse
        $result.JsonError | Should -BeNullOrEmpty
    }
}
