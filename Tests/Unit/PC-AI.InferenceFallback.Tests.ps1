#Requires -Modules @{ ModuleName='Pester'; ModuleVersion='5.0.0' }

<#
.SYNOPSIS
    Unit tests for native inference fallback behavior in PC-AI.ps1
#>

Describe 'Initialize-InferenceBackend fallback behavior' -Tag 'Unit', 'Inference', 'Fallback', 'Portable' {
    BeforeAll {
        $ScriptPath = Join-Path $PSScriptRoot '..\..\PC-AI.ps1'

        # Dot-sourcing the CLI defines its whole surface into this session and
        # never took it back out. Among those definitions is Get-ProcessPerformance,
        # which then SHADOWS the PC-AI.Performance module's function -- so later
        # suites calling it reach a different function entirely and their
        # `-ModuleName PC-AI.Performance` mocks never apply. Running this file
        # immediately before PC-AI.Performance.Tests.ps1 fails 25 of its 26 tests;
        # in the full suite the surviving symptom was
        # "Should handle process retrieval errors". Snapshot first so AfterAll can
        # put the session back.
        $script:FunctionsBeforeDotSource = @(Get-ChildItem 'Function:\' | Select-Object -ExpandProperty Name)
        $script:ModulesBeforeDotSource = @(Get-Module | Select-Object -ExpandProperty Name)

        . $ScriptPath -Command 'help' | Out-Null

        $script:PcaiInferenceModulePath = Join-Path $script:ModulesPath 'PcaiInference.psm1'
    }

    AfterAll {
        # Remove only what the dot-source added, so nothing this file loaded can
        # shadow a module function for the suites that run after it.
        Get-ChildItem 'Function:\' |
            Where-Object { $_.Name -notin $script:FunctionsBeforeDotSource } |
            ForEach-Object { Remove-Item "Function:\$($_.Name)" -Force -ErrorAction SilentlyContinue }

        Get-Module |
            Where-Object { $_.Name -notin $script:ModulesBeforeDotSource } |
            ForEach-Object { Remove-Module $_.Name -Force -ErrorAction SilentlyContinue }
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
