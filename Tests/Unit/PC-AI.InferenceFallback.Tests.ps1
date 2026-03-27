#Requires -Modules @{ ModuleName='Pester'; ModuleVersion='5.0.0' }

<#
.SYNOPSIS
    Unit tests for native inference fallback behavior in PC-AI.ps1
#>

Describe 'Initialize-InferenceBackend fallback behavior' -Tag 'Unit', 'Inference', 'Fallback', 'Portable' {
    BeforeAll {
        $ScriptPath = Join-Path $PSScriptRoot '..\..\PC-AI.ps1'
        . $ScriptPath -Command 'help' | Out-Null

        $script:PcaiInferenceModulePath = Join-Path $script:ModulesPath 'PcaiInference.psm1'
    }

    BeforeEach {
        $script:InferenceMode = 'http'
        $script:NativeInferenceReady = $false
    }

    Context 'When HTTP backend is requested' {
        It 'Should stay in HTTP mode and skip native initialization' {
            Mock Import-Module { throw 'Import-Module should not be called' }

            $result = Initialize-InferenceBackend -Backend 'http' -ModelPath $null -GpuLayers -1

            $result | Should -BeTrue
            $script:InferenceMode | Should -Be 'http'
            $script:NativeInferenceReady | Should -BeFalse
            Should -Not -Invoke Import-Module
        }
    }

    Context 'When native module is missing' {
        It 'Should fall back to HTTP' {
            Mock Test-Path { return $false } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { throw 'Import-Module should not be called' }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath $null -GpuLayers -1

            $result | Should -BeFalse
            $script:InferenceMode | Should -Be 'http'
            Should -Not -Invoke Import-Module
        }
    }

    Context 'When native DLL is missing' {
        It 'Should fall back to HTTP without initializing the backend' {
            Mock Test-Path { return $true } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { }
            Mock Get-PcaiInferenceStatus { [PSCustomObject]@{ DllExists = $false } }
            Mock Initialize-PcaiInference { }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath $null -GpuLayers -1

            $result | Should -BeFalse
            $script:InferenceMode | Should -Be 'http'
            Should -Not -Invoke Initialize-PcaiInference
        }
    }

    Context 'When backend initialization fails' {
        It 'Should fall back to HTTP' {
            Mock Test-Path { return $true } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { }
            Mock Get-PcaiInferenceStatus { [PSCustomObject]@{ DllExists = $true } }
            Mock Initialize-PcaiInference { @{ Success = $false } }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath $null -GpuLayers -1

            $result | Should -BeFalse
            $script:InferenceMode | Should -Be 'http'
            Should -Invoke Initialize-PcaiInference -Times 1
        }
    }

    Context 'When model load fails' {
        It 'Should fall back to HTTP and close native backend' {
            Mock Test-Path { return $true } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { }
            Mock Get-PcaiInferenceStatus { [PSCustomObject]@{ DllExists = $true } }
            Mock Initialize-PcaiInference { @{ Success = $true } }
            Mock Import-PcaiModel { @{ Success = $false } }
            Mock Close-PcaiInference { }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath 'C:\\models\\test.gguf' -GpuLayers 0

            $result | Should -BeFalse
            $script:InferenceMode | Should -Be 'http'
            Should -Invoke Close-PcaiInference -Times 1
        }
    }

    Context 'When model load succeeds' {
        It 'Should switch to native mode' {
            Mock Test-Path { return $true } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { }
            Mock Get-PcaiInferenceStatus { [PSCustomObject]@{ DllExists = $true } }
            Mock Initialize-PcaiInference { @{ Success = $true } }
            Mock Import-PcaiModel { @{ Success = $true } }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath 'C:\\models\\test.gguf' -GpuLayers 0

            $result | Should -BeTrue
            $script:InferenceMode | Should -Be 'native'
            $script:NativeInferenceReady | Should -BeTrue
        }
    }

    Context 'When no model is provided' {
        It 'Should keep HTTP mode until a model is loaded' {
            Mock Test-Path { return $true } -ParameterFilter { $Path -eq $script:PcaiInferenceModulePath }
            Mock Import-Module { }
            Mock Get-PcaiInferenceStatus { [PSCustomObject]@{ DllExists = $true } }
            Mock Initialize-PcaiInference { @{ Success = $true } }
            Mock Import-PcaiModel { }

            $result = Initialize-InferenceBackend -Backend 'mistralrs' -ModelPath $null -GpuLayers -1

            $result | Should -BeFalse
            $script:InferenceMode | Should -Be 'http'
            Should -Not -Invoke Import-PcaiModel
        }
    }
}
