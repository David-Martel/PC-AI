<#
.SYNOPSIS
    Unit tests for PC-AI.LLM module

.DESCRIPTION
    Tests Ollama connectivity, LLM chat functionality, and PC diagnosis using local LLMs
#>

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop
}

Describe 'Get-LLMStatus' -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    Context 'When pcai-inference is running and accessible' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @(
                    [PSCustomObject]@{ Name = 'pcai-inference' }
                    [PSCustomObject]@{ Name = 'llama3.2:latest' }
                )
            } -ModuleName PC-AI.LLM
        }

        It 'Should detect pcai-inference is running' {
            $result = Get-LLMStatus -TestConnection
            $result | Should -Not -BeNullOrEmpty
            $result.PcaiInference.ApiConnected | Should -Be $true
        }

        It 'Should check default endpoint' {
            Get-LLMStatus -TestConnection

            Should -Invoke Test-PcaiInferenceConnection -ModuleName PC-AI.LLM -Times 1
        }
    }

    Context 'When pcai-inference is not running' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $false } -ModuleName PC-AI.LLM
        }

        It 'Should detect pcai-inference is not available' {
            $result = Get-LLMStatus
            $result.PcaiInference.ApiConnected | Should -Be $false
            $result.Recommendations | Should -Not -BeNullOrEmpty
        }
    }

    Context 'When checking available models' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @(
                    [PSCustomObject]@{ Name = 'llama3.2:latest' }
                )
            } -ModuleName PC-AI.LLM
        }

        It 'Should list available models' {
            $result = Get-LLMStatus -TestConnection
            $result.PcaiInference.Models | Should -Not -BeNullOrEmpty
            $result.PcaiInference.Models.Count | Should -BeGreaterThan 0
            $result.PcaiInference.Models[0].Name | Should -Be 'llama3.2:latest'
        }
    }
}

Describe 'Send-OllamaRequest' -Tag 'Unit', 'LLM', 'Slow', 'Portable' {
    Context 'When sending a successful request' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    choices = @(@{ text = 'OK' })
                    usage   = @{ prompt_tokens = 5; completion_tokens = 5; total_tokens = 10 }
                }
            } -ModuleName PC-AI.LLM
        }

        It 'Should send prompt to Ollama' {
            $result = Send-OllamaRequest -Prompt 'Analyze this system' -Model 'llama3.2:latest'
            $result | Should -Not -BeNullOrEmpty
        }

        It 'Should include model in request' {
            Send-OllamaRequest -Prompt 'Test' -Model 'llama3.2:latest'

            Should -Invoke Invoke-OllamaGenerate -ModuleName PC-AI.LLM -ParameterFilter {
                $Model -eq 'llama3.2:latest' -and $Prompt -eq 'Test'
            }
        }

        It 'Should include prompt in request' {
            Send-OllamaRequest -Prompt 'Test prompt' -Model 'llama3.2:latest'

            Should -Invoke Invoke-OllamaGenerate -ModuleName PC-AI.LLM -ParameterFilter {
                $Prompt -eq 'Test prompt'
            }
        }

        It 'Should return response text' {
            $result = Send-OllamaRequest -Prompt 'Test' -Model 'llama3.2:latest'
            $result.Response | Should -Not -BeNullOrEmpty
        }
    }

    Context 'When model is not available' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'nonexistent:latest'
                    created = 123
                    choices = @(@{ text = 'OK' })
                }
            } -ModuleName PC-AI.LLM
            Mock Write-Warning {} -ModuleName PC-AI.LLM
        }

        It 'Should warn when model is not in model list' {
            Send-OllamaRequest -Prompt 'Test' -Model 'nonexistent:latest'
            Should -Invoke Write-Warning -ModuleName PC-AI.LLM -Times 1
        }
    }

    Context 'When using system message' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    choices = @(@{ text = 'OK' })
                }
            } -ModuleName PC-AI.LLM
        }

        It 'Should include system message' {
            Send-OllamaRequest -Prompt 'Test' -Model 'llama3.2:latest' -System 'You are a PC diagnostics expert'

            Should -Invoke Invoke-OllamaGenerate -ModuleName PC-AI.LLM -ParameterFilter {
                $System -eq 'You are a PC diagnostics expert'
            }
        }
    }

    Context 'When setting temperature' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    choices = @(@{ text = 'OK' })
                }
            } -ModuleName PC-AI.LLM
        }

        It 'Should include temperature parameter' {
            Send-OllamaRequest -Prompt 'Test' -Model 'llama3.2:latest' -Temperature 0.7

            Should -Invoke Invoke-OllamaGenerate -ModuleName PC-AI.LLM -ParameterFilter {
                $Temperature -eq 0.7
            }
        }
    }

    Context 'When Ollama is not responding' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $false } -ModuleName PC-AI.LLM
        }

        It 'Should handle timeout errors' {
            { Send-OllamaRequest -Prompt 'Test' -Model 'llama3.2:latest' -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe 'Invoke-LLMChat' -Tag 'Unit', 'LLM', 'Slow', 'Portable' {
    Context 'When starting an interactive chat' {
        BeforeAll {
            Mock Invoke-LLMChatWithFallback {
                return [PSCustomObject]@{
                    message  = @{ content = 'Hello! How can I help you?' }
                    Provider = 'pcai-inference'
                }
            } -ModuleName PC-AI.LLM
            Mock Read-Host { 'exit' } -ModuleName PC-AI.LLM
        }

        It 'Should start chat session' {
            { Invoke-LLMChat -Interactive -Model 'llama3.2:latest' } | Should -Not -Throw
        }

        It 'Should use specified model' {
            Invoke-LLMChat -Interactive -Model 'llama3.2:latest'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -ParameterFilter {
                $Model -eq 'llama3.2:latest'
            }
        }
    }

    Context 'When using system prompt' {
        BeforeAll {
            Mock Invoke-LLMChatWithFallback {
                return [PSCustomObject]@{
                    message  = @{ content = "I'm here to help!" }
                    Provider = 'pcai-inference'
                }
            } -ModuleName PC-AI.LLM
            Mock Read-Host { 'exit' } -ModuleName PC-AI.LLM
        }

        It 'Should apply system prompt' {
            Invoke-LLMChat -Interactive -Model 'llama3.2:latest' -System 'You are helpful'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -ParameterFilter {
                $Messages[0].role -eq 'system' -and $Messages[0].content -match 'You are helpful'
            }
        }
    }

    Context 'When maintaining conversation context' {
        BeforeAll {
            Mock Invoke-LLMChatWithFallback {
                return [PSCustomObject]@{
                    message  = @{ content = 'Response' }
                    Provider = 'pcai-inference'
                }
            } -ModuleName PC-AI.LLM
            $script:callCount = 0
            Mock Read-Host {
                $script:callCount++
                if ($script:callCount -gt 2) { 'exit' } else { "test message $script:callCount" }
            } -ModuleName PC-AI.LLM
        }

        It 'Should maintain conversation context' {
            Invoke-LLMChat -Interactive -Model 'llama3.2:latest'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -Times 2
        }
    }
}

Describe 'Invoke-PCDiagnosis' -Tag 'Unit', 'LLM', 'Integration', 'Portable' {
    Context 'When analyzing diagnostic data' {
        BeforeAll {
            Mock Get-Content {
                param($Path)
                if ($Path -match 'DIAGNOSE') {
                    'You are a PC diagnostics expert.'
                } else {
                    @'
=== Device Errors ===
Device: USB Mass Storage Device
Error Code: 43

=== Disk Health ===
Samsung SSD 980 PRO: OK
WDC HDD: Pred Fail
'@
                }
            } -ModuleName PC-AI.LLM

            Mock Invoke-LLMChatWithFallback {
                return [PSCustomObject]@{
                    message  = @{ content = '{"diagnosis_version":"2.0.0","timestamp":"2026-01-27T00:00:00Z","model_id":"qwen2.5-coder:7b","environment":{"os_version":"Windows","pcai_tooling":"Test"},"summary":["USB device error code 43 found."],"findings":[],"recommendations":[],"what_is_missing":[]}' }
                    Provider = 'pcai-inference'
                }
            } -ModuleName PC-AI.LLM

            Mock Test-Path { $true } -ModuleName PC-AI.LLM
        }

        It 'Should read diagnostic report' {
            Invoke-PCDiagnosis -DiagnosticReportPath 'TestDrive:\report.txt'

            Should -Invoke Get-Content -ModuleName PC-AI.LLM -ParameterFilter {
                $Path -match 'report\.txt'
            }
        }

        It 'Should send report to LLM via chat endpoint' {
            Invoke-PCDiagnosis -DiagnosticReportPath 'TestDrive:\report.txt'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -Times 1
        }

        It 'Should include DIAGNOSE.md prompt in system message' {
            Invoke-PCDiagnosis -DiagnosticReportPath 'TestDrive:\report.txt'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -ParameterFilter {
                $Messages[0].role -eq 'system'
            }
        }

        It 'Should use specified model' {
            Invoke-PCDiagnosis -DiagnosticReportPath 'TestDrive:\report.txt' -Model 'qwen2.5:7b'

            Should -Invoke Invoke-LLMChatWithFallback -ModuleName PC-AI.LLM -ParameterFilter {
                $Model -eq 'qwen2.5:7b'
            }
        }
    }

    Context 'When report file does not exist' {
        BeforeAll {
            Mock Test-Path { $false } -ModuleName PC-AI.LLM
        }

        It 'Should handle missing report file' {
            { Invoke-PCDiagnosis -DiagnosticReportPath 'TestDrive:\nonexistent.txt' -ErrorAction Stop } | Should -Throw
        }
    }

    Context 'When saving analysis output' {
        BeforeAll {
            Mock Get-Content {
                param($Path)
                if ($Path -match 'DIAGNOSE') {
                    'You are a PC diagnostics expert.'
                } else {
                    'Diagnostic data'
                }
            } -ModuleName PC-AI.LLM

            Mock Invoke-LLMChatWithFallback {
                return [PSCustomObject]@{
                    message  = @{ content = '{"diagnosis_version":"2.0.0","timestamp":"2026-01-27T00:00:00Z","model_id":"qwen2.5-coder:7b","environment":{"os_version":"Windows","pcai_tooling":"Test"},"summary":["Analysis results"],"findings":[],"recommendations":[],"what_is_missing":[]}' }
                    Provider = 'pcai-inference'
                }
            } -ModuleName PC-AI.LLM

            Mock Test-Path { $true } -ModuleName PC-AI.LLM
        }

        It 'Should save analysis to file' {
            # Use Join-Path with TestDrive to get proper path
            $reportPath = Join-Path $TestDrive 'report.txt'
            $analysisPath = Join-Path $TestDrive 'analysis.txt'

            $result = Invoke-PCDiagnosis -DiagnosticReportPath $reportPath -OutputPath $analysisPath -SaveReport

            # Verify the result contains the saved path
            $result.ReportSavedTo | Should -Be $analysisPath
            $result.Analysis | Should -Match '"diagnosis_version":"2.0.0"'
            $result.JsonValid | Should -Be $true
        }
    }
}

Describe 'Set-LLMConfig' -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    Context 'When configuring LLM settings' {
        BeforeAll {
            Mock Test-Path { $false } -ModuleName PC-AI.LLM -ParameterFilter { $Path -match 'config' }
            Mock Test-PcaiInferenceConnection { $false } -ModuleName PC-AI.LLM
        }

        It 'Should save configuration and return config object' {
            $result = Set-LLMConfig -OllamaApiUrl 'http://mock-server:11434' -DefaultModel 'llama3.2:latest'

            # Verify the configuration object was returned with correct values
            $result | Should -Not -BeNullOrEmpty
            $result.OllamaApiUrl | Should -Be 'http://mock-server:11434'
            $result.DefaultModel | Should -Be 'llama3.2:latest'
            $result.PSObject.Properties.Name | Should -Contain 'ConfigPath'
            $result.PSObject.Properties.Name | Should -Contain 'LastUpdated'
        }

        It 'Should save JSON configuration with proper structure' {
            $result = Set-LLMConfig -OllamaApiUrl 'http://custom:11434' -DefaultModel 'llama3.2:latest'

            # Verify the returned configuration has proper structure
            $result | Should -Not -BeNullOrEmpty
            $result.OllamaApiUrl | Should -Be 'http://custom:11434'
            $result.DefaultModel | Should -Be 'llama3.2:latest'
            $result.PSObject.Properties.Name | Should -Contain 'DefaultTimeout'
        }

        It 'Should update default model in config' {
            $result = Set-LLMConfig -DefaultModel 'qwen2.5:7b'

            $result.DefaultModel | Should -Be 'qwen2.5:7b'
        }
    }

    Context 'When showing current configuration' {
        It 'Should return current config with ShowConfig' {
            $result = Set-LLMConfig -ShowConfig

            $result | Should -Not -BeNullOrEmpty
            $result.PSObject.Properties.Name | Should -Contain 'OllamaApiUrl'
            $result.PSObject.Properties.Name | Should -Contain 'DefaultModel'
            $result.PSObject.Properties.Name | Should -Contain 'ConfigPath'
        }
    }

    Context 'When resetting configuration' {
        It 'Should reset config to defaults' {
            $result = Set-LLMConfig -Reset

            # Verify the configuration was reset to production defaults.
            # Values come from Config/llm-config.json captured in $script:ModuleDefaults at module load:
            #   DefaultModel   = 'qwen2.5-coder:3b'  (replaces legacy 'pcai-inference' placeholder)
            #   OllamaApiUrl   = 'http://127.0.0.1:11434'  (standard Ollama port)
            #   DefaultTimeout = 180 s  (llm-config.json timeout_ms: 180000)
            $result | Should -Not -BeNullOrEmpty
            $result.DefaultModel | Should -Be 'qwen2.5-coder:3b'
            $result.OllamaApiUrl | Should -Be 'http://127.0.0.1:11434'
            $result.DefaultTimeout | Should -Be 180
        }
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
