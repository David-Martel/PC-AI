<#
.SYNOPSIS
    Robustness tests for PC-AI.LLM module.
#>

BeforeAll {
    $PcaiRoot = Join-Path $PSScriptRoot '..\..'
    Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1') -Force -ErrorAction Stop
}

Describe "PC-AI Robustness" -Tag 'Unit', 'Robustness', 'Portable' {

    It "Should enforce content size limit (5MB)" {
        InModuleScope PC-AI.LLM {
            $largeString = "A" * (5MB + 1)
            { ConvertFrom-LLMJson -Content $largeString -Strict } | Should -Throw "Content exceeded 5MB limit"
        }
    }

    It "Should handle empty or whitespace content gracefully" {
        InModuleScope PC-AI.LLM {
            ConvertFrom-LLMJson -Content "" | Should -BeNull
            ConvertFrom-LLMJson -Content "   " | Should -BeNull
        }
    }

    It "Should fallback to raw content on parse error (non-strict)" {
        InModuleScope PC-AI.LLM {
            $invalidJson = "{ 'invalid': json "
            $result = ConvertFrom-LLMJson -Content $invalidJson
            $result | Should -Be $invalidJson
        }
    }

    It "Should truncate large tool results" {
        InModuleScope PC-AI.LLM {
            Mock Invoke-FunctionGemmaChat {
                return [PSCustomObject]@{
                    choices = @(
                        [PSCustomObject]@{
                            message = [PSCustomObject]@{
                                tool_calls = @(
                                    [PSCustomObject]@{
                                        id = "call_1"
                                        function = [PSCustomObject]@{
                                            name = "pcai_get_system_info"
                                            arguments = "{}"
                                        }
                                    }
                                )
                            }
                        }
                    )
                }
            }

            Mock Invoke-ToolByName {
                return "X" * 10000 # 10KB
            }

            # Must pass -ExecuteTools for truncation logic to run
            $result = Invoke-FunctionGemmaReAct -Prompt "test" -MaxToolCalls 1 -ResultLimit 1024 -ExecuteTools -SkipHealthCheck

            $result.ToolResults[0].result.Length | Should -BeLessThan 2000 # 1024 + truncation message
            $result.ToolResults[0].result | Should -Match "TRUNCATED"
        }
    }

    It "Should handle variables for prompt assembly" {
        $type = [Type]::GetType('PcaiNative.PcaiCore', $false)
        if ($null -eq $type) { Set-ItResult -Skipped -Because "PcaiCore not found" ; return }

        $template = "Hello {{Name}}"
        $vars = @{ Name = "Robustness" }

        # Use Try-Catch to avoid cryptic Pester binding errors if .NET fails
        $result = $null
        try {
            $result = $type::AssemblePrompt($template, $vars)
        } catch {
            Write-Host "AssemblePrompt failure: $_"
            $_.Exception | Select-Object * | Out-String | Write-Host
            throw "AssemblePrompt failed with: $($_.Exception.Message)"
        }

        if ($null -eq $result) { Set-ItResult -Skipped -Because "Native core unavailable" ; return }
        $result | Should -Match "Hello Robustness"
    }
}
