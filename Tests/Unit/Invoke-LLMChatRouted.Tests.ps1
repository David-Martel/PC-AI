#Requires -Modules @{ ModuleName='Pester'; ModuleVersion='5.0.0' }

BeforeAll {
    $ModulePath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Modules\PC-AI.LLM\PC-AI.LLM.psm1'
    Import-Module $ModulePath -Force
}

Describe 'Invoke-LLMChatRouted - Graceful Degradation' -Tag 'Unit', 'Portable' {
    BeforeAll {
        # Mock dependencies
        Mock -ModuleName PC-AI.LLM Get-EnrichedSystemPrompt {
            return "Test system prompt"
        }

        Mock -ModuleName PC-AI.LLM Invoke-LLMChatWithFallback {
            return [PSCustomObject]@{
                message = @{
                    content = "Test response from LLM"
                }
                Provider = 'ollama'
            }
        }

        # Default successful router mock
        Mock -ModuleName PC-AI.LLM Invoke-FunctionGemmaReAct {
            return [PSCustomObject]@{
                ToolCalls = @(
                    @{
                        tool = 'Get-SystemInfo'
                        arguments = @{ detailed = $true }
                    }
                )
                ToolResults = @(
                    @{
                        tool = 'Get-SystemInfo'
                        result = 'System info output'
                    }
                )
            }
        }

        Mock -ModuleName PC-AI.LLM Get-CachedProviderHealth {
            param([string]$Provider, [int]$TimeoutSeconds, [string]$ApiUrl)
            return $true
        }
    }

    Context 'BypassRouter Parameter' {
        It 'Should skip router call when BypassRouter is specified' {
            # Arrange
            Mock -ModuleName PC-AI.LLM Invoke-FunctionGemmaReAct {
                throw "Router should not be called when BypassRouter is used"
            }

            # Act
            $result = Invoke-LLMChatRouted -Message "Test message" -BypassRouter

            # Assert
            $result | Should -Not -BeNullOrEmpty
            $result.DegradedMode | Should -BeTrue
            Should -Invoke -ModuleName PC-AI.LLM -CommandName Invoke-FunctionGemmaReAct -Times 0
        }

        It 'Should set DegradedMode to true when BypassRouter is used' {
            # Arrange
            Mock -ModuleName PC-AI.LLM Invoke-FunctionGemmaReAct {
                throw "Router should not be called"
            }

            # Act
            $result = Invoke-LLMChatRouted -Message "Test message" -BypassRouter

            # Assert
            $result.DegradedMode | Should -BeTrue
            $result.RouterAvailable | Should -BeFalse
        }
    }

    Context 'RouterAvailable Field' {
        It 'Should set RouterAvailable to true when router succeeds' {
            # Act
            $result = Invoke-LLMChatRouted -Message "Test message"

            # Assert
            $result.RouterAvailable | Should -BeTrue
            $result.DegradedMode | Should -BeFalse
        }
    }

    Context 'Router Failure Handling' {
        It 'Should gracefully fall back when router fails' {
            # Arrange
            Mock -ModuleName PC-AI.LLM Invoke-FunctionGemmaReAct {
                throw "Router service unavailable"
            }

            # Act
            $result = Invoke-LLMChatRouted -Message "Test message"

            # Assert
            $result | Should -Not -BeNullOrEmpty
            $result.Response | Should -Be "Test response from LLM"
            $result.RouterAvailable | Should -BeFalse
            $result.DegradedMode | Should -BeTrue
            $result.ToolCalls | Should -BeNullOrEmpty
            $result.ToolResults | Should -BeNullOrEmpty
        }

        It 'Should log warning when router fails' {
            # Arrange
            Mock -ModuleName PC-AI.LLM Invoke-FunctionGemmaReAct {
                throw "Router service unavailable"
            }
            Mock -ModuleName PC-AI.LLM Write-Warning

            # Act
            $result = Invoke-LLMChatRouted -Message "Test message"

            # Assert
            Should -Invoke -ModuleName PC-AI.LLM -CommandName Write-Warning -Times 1
        }
    }

    Context 'Response Object Fields' {
        It 'Should include RouterAvailable and DegradedMode in response' {
            # Act
            $result = Invoke-LLMChatRouted -Message "Test message"

            # Assert
            $result.PSObject.Properties.Name | Should -Contain 'RouterAvailable'
            $result.PSObject.Properties.Name | Should -Contain 'DegradedMode'
        }

        It 'Should preserve all existing fields' {
            # Act
            $result = Invoke-LLMChatRouted -Message "Test message"

            # Assert
            $result.PSObject.Properties.Name | Should -Contain 'Mode'
            $result.PSObject.Properties.Name | Should -Contain 'Prompt'
            $result.PSObject.Properties.Name | Should -Contain 'ToolCalls'
            $result.PSObject.Properties.Name | Should -Contain 'ToolResults'
            $result.PSObject.Properties.Name | Should -Contain 'Response'
            $result.PSObject.Properties.Name | Should -Contain 'ResponseJson'
            $result.PSObject.Properties.Name | Should -Contain 'JsonValid'
            $result.PSObject.Properties.Name | Should -Contain 'JsonError'
            $result.PSObject.Properties.Name | Should -Contain 'Provider'
            $result.PSObject.Properties.Name | Should -Contain 'Model'
            $result.PSObject.Properties.Name | Should -Contain 'RouterModel'
            $result.PSObject.Properties.Name | Should -Contain 'RouterBaseUrl'
            $result.PSObject.Properties.Name | Should -Contain 'Timestamp'
        }
    }
}
