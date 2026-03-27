<#
.SYNOPSIS
    Unit tests for Invoke-FunctionGemmaReAct.
#>

BeforeAll {
    $PcaiRoot = Join-Path $PSScriptRoot '..\..'

    # Import modules
    Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1') -Force -ErrorAction Stop
    Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1') -Force -ErrorAction Stop

    # Mock dependencies
    $script:ToolsPath = Join-Path $PcaiRoot 'Config\pcai-tools.json'
}

Describe "Invoke-FunctionGemmaReAct" -Tag 'Unit', 'LLM', 'ReAct' {
    BeforeEach {
        Mock Invoke-FunctionGemmaChat {
            return [PSCustomObject]@{
                choices = @(
                    [PSCustomObject]@{
                        message = [PSCustomObject]@{
                            content = "Final Answer: Done"
                        }
                    }
                )
            }
        } -ModuleName PC-AI.LLM
    }

    It "Should handle empty choices from LLM gracefully (Fix for crash)" {
        Mock Invoke-FunctionGemmaChat {
            return [PSCustomObject]@{ choices = @() }
        } -ModuleName PC-AI.LLM

        $result = Invoke-FunctionGemmaReAct -Prompt "test" -BaseUrl "http://test" -Model "test" -SkipHealthCheck
        $result.FinalAnswer | Should -Match "Error: LLM returned no response or choices."
    }

    It "Should limit maximum tool calls" {
        Mock Invoke-FunctionGemmaChat {
            return [PSCustomObject]@{
                choices = @(
                    [PSCustomObject]@{
                        message = [PSCustomObject]@{
                            tool_calls = @(
                                [PSCustomObject]@{ id = "call_1"; function = [PSCustomObject]@{ name = "pcai_get_docker_status"; arguments = "{}" } },
                                [PSCustomObject]@{ id = "call_2"; function = [PSCustomObject]@{ name = "pcai_get_docker_status"; arguments = "{}" } },
                                [PSCustomObject]@{ id = "call_3"; function = [PSCustomObject]@{ name = "pcai_get_docker_status"; arguments = "{}" } }
                            )
                        }
                    }
                )
            }
        } -ModuleName PC-AI.LLM

        Mock Get-DockerStatus { return @{ Running = $true } } -ModuleName PC-AI.LLM

        $maxCalls = 2
        $result = Invoke-FunctionGemmaReAct -Prompt "test" -BaseUrl "http://test" -Model "test" -MaxToolCalls $maxCalls -ExecuteTools -SkipHealthCheck

        $result.ToolResults.Count | Should -Be $maxCalls
    }

    It "Should parse tool arguments correctly (JSON string or object)" {
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
                                        arguments = '{"Category": "CPU"}'
                                    }
                                }
                            )
                        }
                    }
                )
            }
        } -ModuleName PC-AI.LLM

        Mock Get-SystemInfoTool { return "CPU Info" } -ModuleName PC-AI.LLM

        $result = Invoke-FunctionGemmaReAct -Prompt "test" -BaseUrl "http://test" -Model "test" -ExecuteTools -SkipHealthCheck

        $result.ToolResults[0].arguments.Category | Should -Be "CPU"
    }

    It "Should handle tool execution errors" {
        Mock Invoke-FunctionGemmaChat {
            return [PSCustomObject]@{
                choices = @(
                    [PSCustomObject]@{
                        message = [PSCustomObject]@{
                            tool_calls = @(
                                [PSCustomObject]@{
                                    id = "call_1"
                                    function = [PSCustomObject]@{
                                        name = "pcai_get_docker_status"
                                        arguments = "{}"
                                    }
                                }
                            )
                        }
                    }
                )
            }
        } -ModuleName PC-AI.LLM

        Mock Get-DockerStatus { throw "Docker failed" } -ModuleName PC-AI.LLM

        $result = Invoke-FunctionGemmaReAct -Prompt "test" -BaseUrl "http://test" -Model "test" -ExecuteTools -SkipHealthCheck

        $result.ToolResults[0].result | Should -Match "Error executing tool"
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
