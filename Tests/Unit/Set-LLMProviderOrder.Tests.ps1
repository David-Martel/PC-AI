<#
.SYNOPSIS
    Unit tests for Set-LLMProviderOrder function.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    $script:OriginalModuleConfig = InModuleScope PC-AI.LLM {
        @{
            ConfigPath        = $script:ModuleConfig.ConfigPath
            ProjectConfigPath = $script:ModuleConfig.ProjectConfigPath
            ProviderOrder     = @($script:ModuleConfig.ProviderOrder)
        }
    }
}

AfterAll {
    $original = $script:OriginalModuleConfig
    InModuleScope PC-AI.LLM -Parameters @{ Original = $original } {
        $script:ModuleConfig.ConfigPath = $Original.ConfigPath
        $script:ModuleConfig.ProjectConfigPath = $Original.ProjectConfigPath
        $script:ModuleConfig.ProviderOrder = @($Original.ProviderOrder)
    }

    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}

Describe "Set-LLMProviderOrder" -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    Context "When updating provider order" {
        BeforeEach {
            $script:ConfigPath = Join-Path $TestDrive 'llm-config.json'
            @'
{
  "fallbackOrder": [
    "ollama"
  ],
  "providers": {}
}
'@ | Set-Content -Path $script:ConfigPath -Encoding UTF8

            $configPath = $script:ConfigPath
            InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $configPath } {
                $script:ModuleConfig.ConfigPath = $ConfigPath
                $script:ModuleConfig.ProjectConfigPath = $ConfigPath
            }
        }

        It "updates the in-memory provider order and JSON fallback order" {
            $order = @('pcai-inference', 'ollama')
            $result = Set-LLMProviderOrder -Order $order
            $config = Get-Content -Path $script:ConfigPath -Raw | ConvertFrom-Json

            $result.Success | Should -BeTrue
            $result.Order -join ',' | Should -Be 'pcai-inference,ollama'
            $config.fallbackOrder -join ',' | Should -Be 'pcai-inference,ollama'

            InModuleScope PC-AI.LLM {
                $script:ModuleConfig.ProviderOrder -join ','
            } | Should -Be 'pcai-inference,ollama'
        }

        It "returns the config path that was updated" {
            $result = Set-LLMProviderOrder -Order @('pcai-inference')

            $result.Success | Should -BeTrue
            $result.ConfigPath | Should -Be $script:ConfigPath
        }

        It "adds fallbackOrder when the config does not already contain it" {
            '{"providers":{}}' | Set-Content -Path $script:ConfigPath -Encoding UTF8

            Set-LLMProviderOrder -Order @('vllm', 'ollama') | Out-Null
            $config = Get-Content -Path $script:ConfigPath -Raw | ConvertFrom-Json

            $config.fallbackOrder -join ',' | Should -Be 'vllm,ollama'
        }
    }

    Context "When the config file is missing" {
        It "throws a clear config-not-found error" {
            $missingPath = Join-Path $TestDrive 'missing-llm-config.json'
            InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $missingPath } {
                $script:ModuleConfig.ConfigPath = $ConfigPath
                $script:ModuleConfig.ProjectConfigPath = $ConfigPath
            }

            { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw "Config file not found: $missingPath"
        }
    }

    Context "When order parameter is invalid" {
        It "throws validation error for null order" {
            { Set-LLMProviderOrder -Order $null -ErrorAction Stop } | Should -Throw
        }
    }
}
