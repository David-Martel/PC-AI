<#
.SYNOPSIS
    Unit tests for PC-AI.LLM module

.DESCRIPTION
    Tests Ollama connectivity, LLM chat functionality, and PC diagnosis using local LLMs
#>

BeforeAll {
    $script:RepositoryConfigPath = Join-Path $PSScriptRoot '../../Config/llm-config.json'
    $script:RepositoryConfigHash = (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash
    $script:RepositoryConfigJson = Get-Content -LiteralPath $script:RepositoryConfigPath -Raw
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    $script:OriginalModuleConfig = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
    $script:OriginalModuleDefaults = InModuleScope PC-AI.LLM { $script:ModuleDefaults.Clone() }

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop
}

Describe 'Get-LLMStatus' -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    Context 'When pcai-inference is running and accessible' {
        BeforeAll {
            Mock Test-PcaiInferenceConnection { $true } -ModuleName PC-AI.LLM
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $false } -ModuleName PC-AI.LLM
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    # Must match the shape Invoke-OllamaNativeChat actually returns.
                    # Send-OllamaRequest reads $response.message.content; the previous
                    # OpenAI-completions shape (choices[].text) is never produced by
                    # this code path, so .Response came back $null.
                    message = [PSCustomObject]@{ content = 'OK' }
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'nonexistent:latest'
                    created = 123
                    # Must match the shape Invoke-OllamaNativeChat actually returns.
                    # Send-OllamaRequest reads $response.message.content; the previous
                    # OpenAI-completions shape (choices[].text) is never produced by
                    # this code path, so .Response came back $null.
                    message = [PSCustomObject]@{ content = 'OK' }
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    # Must match the shape Invoke-OllamaNativeChat actually returns.
                    # Send-OllamaRequest reads $response.message.content; the previous
                    # OpenAI-completions shape (choices[].text) is never produced by
                    # this code path, so .Response came back $null.
                    message = [PSCustomObject]@{ content = 'OK' }
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $true } -ModuleName PC-AI.LLM
            Mock Get-OllamaModels {
                @([PSCustomObject]@{ Name = 'llama3.2:latest' })
            } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaGenerate {
                return [PSCustomObject]@{
                    model   = 'llama3.2:latest'
                    created = 123
                    # Must match the shape Invoke-OllamaNativeChat actually returns.
                    # Send-OllamaRequest reads $response.message.content; the previous
                    # OpenAI-completions shape (choices[].text) is never produced by
                    # this code path, so .Response came back $null.
                    message = [PSCustomObject]@{ content = 'OK' }
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
            # Send-OllamaRequest gates on Test-OllamaConnection, NOT on
            # Test-PcaiInferenceConnection. Both exist in LLM-Helpers.ps1, and only the
            # latter was mocked, so the real connection probe ran and threw "Cannot
            # reach the native Ollama runner". Mock both so the intent -- connection
            # reachable or not -- actually reaches the code under test.
            Mock Test-OllamaConnection { $false } -ModuleName PC-AI.LLM
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

Describe 'Set-LLMConfig' -Tag 'Unit', 'LLM', 'Fast', 'Windows' {
    # Isolation belongs to the Describe, so even a Reset-only filtered run is safe.
    BeforeEach {
        $script:LlmConfigTempPath = Join-Path $TestDrive 'llm-config.json'
        $script:RepositoryConfigJson | Set-Content -LiteralPath $script:LlmConfigTempPath -Encoding utf8NoBOM
        InModuleScope PC-AI.LLM -Parameters @{ TempPath = $script:LlmConfigTempPath; Original = $script:OriginalModuleConfig } {
            $script:ModuleConfig = $Original.Clone()
            $script:ModuleConfig.ConfigPath = $TempPath
            $script:ModuleConfig.ProjectConfigPath = $TempPath
            $script:ModuleConfig.ProviderOrder = @('ollama', 'pcai-inference')
        }
        Mock Test-PcaiInferenceConnection { $false } -ModuleName PC-AI.LLM
        Mock Test-OllamaConnection { $false } -ModuleName PC-AI.LLM
    }

    AfterEach {
        (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash | Should -Be $script:RepositoryConfigHash
    }

    Context 'When configuring LLM settings' {
        It 'publishes a new BOM-free config when the selected destination is absent' {
            $newPath = Join-Path $TestDrive 'new-config-parent/llm-config.json'
            InModuleScope PC-AI.LLM -Parameters @{ Path = $newPath } {
                $script:ModuleConfig.ProjectConfigPath = $Path
                $script:ModuleConfig.ConfigPath = $Path
            }
            Set-LLMConfig -DefaultTimeout 15 -ErrorAction Stop | Out-Null
            (Get-Content $newPath -Raw | ConvertFrom-Json).ollama.timeout_ms | Should -Be 15000
            [IO.File]::ReadAllBytes($newPath)[0] | Should -Be 123
            InModuleScope PC-AI.LLM { $script:ModuleConfig.DefaultTimeout } | Should -Be 15
        }
        It 'Should save configuration and return config object' {
            $result = Set-LLMConfig -OllamaApiUrl 'http://mock-server:11434' -DefaultModel 'llama3.2:latest'
            $result.OllamaApiUrl | Should -Be 'http://mock-server:11434'
            $result.DefaultModel | Should -Be 'llama3.2:latest'
            $result.ConfigPath | Should -Be $script:LlmConfigTempPath
            $result.PSObject.Properties.Name | Should -Contain 'LastUpdated'
        }

        It 'Should persist settings without losing unrelated machine and model tuning' {
            Set-LLMConfig -OllamaApiUrl 'http://custom:11434' -DefaultModel 'llama3.2:latest' | Out-Null
            $saved = Get-Content -LiteralPath $script:LlmConfigTempPath -Raw | ConvertFrom-Json
            $original = $script:RepositoryConfigJson | ConvertFrom-Json
            $saved.providers.ollama.baseUrl | Should -Be 'http://custom:11434'
            $saved.ollama.model | Should -Be 'llama3.2:latest'
            foreach ($name in @('tool_model', 'summary_model', 'toolInvokerPath', 'num_gpu', 'num_thread', 'num_ctx', 'adaptive_ctx_max')) {
                $saved.ollama.$name | Should -Be $original.ollama.$name
            }
            $saved.fallbackOrder -join ',' | Should -Be 'ollama,pcai-inference'
            [System.IO.File]::ReadAllBytes($script:LlmConfigTempPath)[0] | Should -Be 123
        }

        It 'Should update default model in config' {
            $result = Set-LLMConfig -DefaultModel 'qwen2.5:7b'
            $result.DefaultModel | Should -Be 'qwen2.5:7b'
            (Get-Content -LiteralPath $script:LlmConfigTempPath -Raw | ConvertFrom-Json).ollama.model | Should -Be 'qwen2.5:7b'
        }
    }

    Context 'When showing current configuration' {
        It 'Should return current config with ShowConfig without writing' {
            $before = (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash
            $result = Set-LLMConfig -ShowConfig
            $result.ConfigPath | Should -Be $script:LlmConfigTempPath
            $result.PSObject.Properties.Name | Should -Contain 'DefaultModel'
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -Be $before
        }
    }

    Context 'When resetting configuration' {
        It 'Should reset config to the machine startup defaults in the isolated file' {
            Set-LLMProviderOrder -Order @('lmstudio', 'ollama') | Out-Null
            Set-LLMConfig -DefaultModel 'temporary-test-model' -DefaultTimeout 15 | Out-Null
            $result = Set-LLMConfig -Reset
            $result.DefaultModel | Should -Be $script:OriginalModuleDefaults.DefaultModel
            $result.OllamaApiUrl | Should -Be $script:OriginalModuleDefaults.OllamaApiUrl
            $result.DefaultTimeout | Should -Be $script:OriginalModuleDefaults.DefaultTimeout
            $result.ConfigPath | Should -Be $script:LlmConfigTempPath
            $saved = Get-Content -LiteralPath $script:LlmConfigTempPath -Raw | ConvertFrom-Json
            $saved.ollama.model | Should -Be $script:OriginalModuleDefaults.DefaultModel
            $saved.fallbackOrder -join ',' | Should -Be 'lmstudio,ollama'
            $original = $script:RepositoryConfigJson | ConvertFrom-Json
            foreach ($name in @('tool_model', 'summary_model', 'num_gpu', 'num_ctx', 'adaptive_ctx_max')) {
                $saved.ollama.$name | Should -Be $original.ollama.$name
            }
        }
    }

    Context 'When inherited provider order is invalid' {
        BeforeEach {
            InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder = @('ollama', 'invalid') }
            $script:BeforeConfigHash = (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash
        }

        It 'rejects a settings update before disk or memory changes' {
            { Set-LLMConfig -DefaultTimeout 15 -ErrorAction Stop } | Should -Throw '*invalid*'
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -Be $script:BeforeConfigHash
            InModuleScope PC-AI.LLM { $script:ModuleConfig.DefaultTimeout } | Should -Be $script:OriginalModuleConfig.DefaultTimeout
        }

        It 'rejects Reset before disk or memory changes' {
            InModuleScope PC-AI.LLM { $script:ModuleConfig.DefaultTimeout = 15 }
            { Set-LLMConfig -Reset -ErrorAction Stop } | Should -Throw '*invalid*'
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -Be $script:BeforeConfigHash
            InModuleScope PC-AI.LLM { $script:ModuleConfig.DefaultTimeout } | Should -Be 15
        }

        It 'allows read-only inspection of invalid configuration' {
            Set-LLMConfig -ShowConfig | Out-Null
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -Be $script:BeforeConfigHash
        }

        It 'does not create a missing destination or parent for invalid provider order' {
            $missingPath = Join-Path $TestDrive 'absent/llm-config.json'
            InModuleScope PC-AI.LLM -Parameters @{ Path = $missingPath } {
                $script:ModuleConfig.ConfigPath = $Path
                $script:ModuleConfig.ProjectConfigPath = $Path
            }
            { Set-LLMConfig -DefaultTimeout 15 -ErrorAction Stop } | Should -Throw '*invalid*'
            Test-Path -LiteralPath (Split-Path -Parent $missingPath) | Should -BeFalse
        }
    }

    Context 'When unrelated configuration has nested JSON' {
        It 'preserves a deeply nested machine extension after <Operation>' -TestCases @(
            @{ Operation = 'update' }, @{ Operation = 'Reset' }
        ) {
            param($Operation)
            $deep = ('{"nested":' * 24) + '"machine-extension-leaf"' + ('}' * 24)
            ('{"fallbackOrder":["ollama"],"providers":{},"machineSetting":' + $deep + '}') |
                Set-Content -LiteralPath $script:LlmConfigTempPath -Encoding utf8NoBOM
            if ($Operation -eq 'Reset') { Set-LLMConfig -Reset -ErrorAction Stop | Out-Null }
            else { Set-LLMConfig -DefaultTimeout 15 -ErrorAction Stop | Out-Null }
            $saved = Get-Content -LiteralPath $script:LlmConfigTempPath -Raw | ConvertFrom-Json
            $value = $saved.machineSetting
            for ($depth = 0; $depth -lt 24; $depth++) { $value = $value.nested }
            $value | Should -BeExactly 'machine-extension-leaf'
        }

        It 'rejects over-limit JSON before disk or memory changes after <Operation>' -TestCases @(
            @{ Operation = 'update' }, @{ Operation = 'Reset' }
        ) {
            param($Operation)
            # Raw JSON deliberately exceeds ConvertTo-Json's maximum depth100;
            # generating this fixture with that serializer would corrupt it first.
            $deep = ('{"nested":' * 110) + '"must-remain-unmodified"' + ('}' * 110)
            ('{"fallbackOrder":["ollama"],"providers":{},"machineSetting":' + $deep + '}') |
                Set-Content -LiteralPath $script:LlmConfigTempPath -Encoding utf8NoBOM
            $beforeHash = (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash
            $beforeMemory = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
            if ($Operation -eq 'Reset') { { Set-LLMConfig -Reset -ErrorAction Stop } | Should -Throw }
            else { { Set-LLMConfig -DefaultTimeout 15 -ErrorAction Stop } | Should -Throw }
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -BeExactly $beforeHash
            $afterMemory = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
            foreach ($key in $beforeMemory.Keys) {
                ($afterMemory[$key] | ConvertTo-Json -Depth 100 -Compress) | Should -BeExactly ($beforeMemory[$key] | ConvertTo-Json -Depth 100 -Compress)
            }
        }
    }

    Context 'When the real config writer fails' {
        It 'restores every prior memory setting after a failed <Operation>' -TestCases @(
            @{ Operation = 'update' }
            @{ Operation = 'Reset' }
        ) {
            param($Operation)
            # Reset has something to change, and update touches several keys.
            InModuleScope PC-AI.LLM {
                $script:ModuleConfig.DefaultTimeout = 19
                $script:ModuleConfig.DefaultModel = 'existing-machine-model'
                $script:ModuleConfig.OllamaApiUrl = 'http://existing-machine:11434'
                $script:ModuleConfig.PcaiInferenceApiUrl = 'http://existing-machine:18080'
            }
            $beforeMemory = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
            $beforeHash = (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash
            # Read and stage real bytes first, then deny sharing at the final
            # publication boundary. This exercises the real File.Replace call.
            Mock Get-LLMConfigCurrentHash {
                param($Path)
                $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
                # Permit retained original reads while denying replacement/delete.
                $script:ConfigWriteFixtureHandle = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
                return $hash
            } -ModuleName PC-AI.LLM -ParameterFilter { [IO.Path]::GetFileName($Path) -like 'llm-config*.json' }
            try {
                if ($Operation -eq 'Reset') {
                    $failure = { Set-LLMConfig -Reset -ErrorAction Stop } | Should -Throw -PassThru
                } else {
                    $failure = { Set-LLMConfig -DefaultTimeout 15 -DefaultModel 'new-model' -OllamaApiUrl 'http://new-machine:11434' -ErrorAction Stop } | Should -Throw -PassThru
                }
                $failure.Exception.GetBaseException() | Should -BeOfType ([System.IO.IOException])
                $failure.Exception.Message | Should -Match 'Replace'
                $afterMemory = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
                foreach ($key in $beforeMemory.Keys) {
                    ($afterMemory[$key] | ConvertTo-Json -Depth 20 -Compress) | Should -Be ($beforeMemory[$key] | ConvertTo-Json -Depth 20 -Compress)
                }
            } finally {
                if ($script:ConfigWriteFixtureHandle) { $script:ConfigWriteFixtureHandle.Dispose(); $script:ConfigWriteFixtureHandle = $null }
            }
            (Get-FileHash -LiteralPath $script:LlmConfigTempPath).Hash | Should -Be $beforeHash
        }
    }
}

AfterAll {
    if ($script:OriginalModuleConfig -and (Get-Module PC-AI.LLM)) {
        InModuleScope PC-AI.LLM -Parameters @{ Original = $script:OriginalModuleConfig } {
            $script:ModuleConfig = $Original
        }
    }
    (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash | Should -Be $script:RepositoryConfigHash
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
