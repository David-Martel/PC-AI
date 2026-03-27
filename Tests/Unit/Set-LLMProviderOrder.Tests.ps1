<#
.SYNOPSIS
    Unit tests for Set-LLMProviderOrder function.
#>

BeforeAll {
    $VirtualizationModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Virtualization\PC-AI.Virtualization.psm1'
    Import-Module $VirtualizationModulePath -Force -ErrorAction Stop

    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
}

Describe "Set-LLMProviderOrder" -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    Context "When updating provider order" {
        BeforeEach {
            Mock Invoke-PcaiServiceHost {
                [PSCustomObject]@{
                    Success = $true
                    Output  = "Order updated"
                }
            } -ModuleName PC-AI.LLM
        }

        It "calls Invoke-PcaiServiceHost with ServerArgs" {
            $order = @('pcai-inference', 'ollama')
            Set-LLMProviderOrder -Order $order | Out-Null

            Should -Invoke Invoke-PcaiServiceHost -ModuleName PC-AI.LLM -ParameterFilter {
                $ServerArgs -join ',' -eq 'provider,set-order,pcai-inference,ollama'
            }
        }

        It "returns successful result" {
            $result = Set-LLMProviderOrder -Order @('pcai-inference')
            $result.Success | Should -BeTrue
        }
    }

    Context "When service host returns failure" {
        BeforeEach {
            Mock Invoke-PcaiServiceHost {
                [PSCustomObject]@{
                    Success = $false
                    Output  = "Error: Invalid provider"
                }
            } -ModuleName PC-AI.LLM
        }

        It "throws with service host error message" {
            { Set-LLMProviderOrder -Order @('invalid') -ErrorAction Stop } | Should -Throw "Failed to update provider order: Error: Invalid provider"
        }
    }

    Context "When order parameter is invalid" {
        It "throws validation error for null order" {
            { Set-LLMProviderOrder -Order $null -ErrorAction Stop } | Should -Throw
        }
    }
}
