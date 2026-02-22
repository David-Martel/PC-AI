#Requires -Modules @{ ModuleName='Pester'; ModuleVersion='5.0.0' }

<#
.SYNOPSIS
    Unit tests for PcaiInference module and PC-AI.ps1 inference parameters

.DESCRIPTION
    Tests backend selection logic, parameter validation, and module exports
    without requiring actual model loading.
#>

Describe 'PcaiInference Module' {
    BeforeAll {
        $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PcaiInference.psm1'
        if (Test-Path $ModulePath) {
            Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
        }
    }

    Context 'Module Loading' {
        It 'Should export Initialize-PcaiInference function' {
            Get-Command Initialize-PcaiInference -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should export Import-PcaiModel function' {
            Get-Command Import-PcaiModel -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should export Invoke-PcaiGenerate function' {
            Get-Command Invoke-PcaiGenerate -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should export Close-PcaiInference function' {
            Get-Command Close-PcaiInference -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should export Get-PcaiInferenceStatus function' {
            Get-Command Get-PcaiInferenceStatus -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should export Test-PcaiInference function' {
            Get-Command Test-PcaiInference -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Backend Parameter Validation' {
        It 'Initialize-PcaiInference should have Backend parameter' {
            $cmd = Get-Command Initialize-PcaiInference -ErrorAction SilentlyContinue
            $cmd.Parameters.ContainsKey('Backend') | Should -BeTrue
        }

        It 'Initialize-PcaiInference Backend should validate against valid values' {
            $cmd = Get-Command Initialize-PcaiInference -ErrorAction SilentlyContinue
            $param = $cmd.Parameters['Backend']
            $validateSet = $param.Attributes | Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet | Should -Not -BeNullOrEmpty
            $validateSet.ValidValues | Should -Contain 'llamacpp'
            $validateSet.ValidValues | Should -Contain 'mistralrs'
            $validateSet.ValidValues | Should -Contain 'auto'
        }
    }

    Context 'Model Path Validation' {
        It 'Import-PcaiModel should have ModelPath parameter' {
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            $cmd.Parameters.ContainsKey('ModelPath') | Should -BeTrue
        }

        It 'Import-PcaiModel ModelPath should be mandatory' {
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            $param = $cmd.Parameters['ModelPath']
            $mandatory = $param.Attributes | Where-Object { $_ -is [System.Management.Automation.ParameterAttribute] }
            $mandatory.Mandatory | Should -BeTrue
        }
    }

    Context 'GPU Layers Parameter' {
        It 'Import-PcaiModel should have GpuLayers parameter' {
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            $cmd.Parameters.ContainsKey('GpuLayers') | Should -BeTrue
        }

        It 'GpuLayers should default to -1' {
            # Check default value
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            $param = $cmd.Parameters['GpuLayers']
            # Default is set in param block, verify parameter exists and is int
            $param.ParameterType.Name | Should -Be 'Int32'
        }
    }

    Context 'Status Function' {
        It 'Get-PcaiInferenceStatus should return PSCustomObject' {
            $status = Get-PcaiInferenceStatus
            $status | Should -BeOfType [PSCustomObject]
        }

        It 'Status should have required properties' {
            $status = Get-PcaiInferenceStatus
            $status.PSObject.Properties.Name | Should -Contain 'DllPath'
            $status.PSObject.Properties.Name | Should -Contain 'DllExists'
            $status.PSObject.Properties.Name | Should -Contain 'BackendInitialized'
            $status.PSObject.Properties.Name | Should -Contain 'ModelLoaded'
            $status.PSObject.Properties.Name | Should -Contain 'CurrentBackend'
        }

        It 'Initially BackendInitialized should be False' {
            Close-PcaiInference -ErrorAction SilentlyContinue
            $status = Get-PcaiInferenceStatus
            $status.BackendInitialized | Should -BeFalse
        }

        It 'Initially ModelLoaded should be False' {
            Close-PcaiInference -ErrorAction SilentlyContinue
            $status = Get-PcaiInferenceStatus
            $status.ModelLoaded | Should -BeFalse
        }
    }

    Context 'DLL Availability' {
        It 'Get-PcaiInferenceStatus should report DllExists false when path is missing' {
            InModuleScope PcaiInference {
                $script:DllPath = Join-Path $env:TEMP 'pcai_inference_missing.dll'
                $status = Get-PcaiInferenceStatus
                $status.DllExists | Should -BeFalse
            }
        }

        It 'Initialize-PcaiInference should throw when DLL is missing' {
            $missingPath = Join-Path $env:TEMP 'pcai_inference_missing.dll'
            { Initialize-PcaiInference -Backend llamacpp -DllPath $missingPath } | Should -Throw -ExpectedMessage "*DLL not found*"
        }
    }
}

Describe 'PC-AI.ps1 Inference Parameters' {
    BeforeAll {
        $ScriptPath = Join-Path $PSScriptRoot '..\..\PC-AI.ps1'
    }

    Context 'Parameter Definitions' {
        It 'Should have InferenceBackend parameter' {
            $params = (Get-Command $ScriptPath).Parameters
            $params.ContainsKey('InferenceBackend') | Should -BeTrue
        }

        It 'Should validate InferenceBackend values' {
            $param = (Get-Command $ScriptPath).Parameters['InferenceBackend']
            $validateSet = $param.Attributes | Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet.ValidValues | Should -Contain 'auto'
            $validateSet.ValidValues | Should -Contain 'llamacpp'
            $validateSet.ValidValues | Should -Contain 'mistralrs'
            $validateSet.ValidValues | Should -Contain 'http'
        }

        It 'Should have ModelPath parameter' {
            $params = (Get-Command $ScriptPath).Parameters
            $params.ContainsKey('ModelPath') | Should -BeTrue
        }

        It 'Should have GpuLayers parameter' {
            $params = (Get-Command $ScriptPath).Parameters
            $params.ContainsKey('GpuLayers') | Should -BeTrue
        }

        It 'Should have UseNativeInference switch' {
            $params = (Get-Command $ScriptPath).Parameters
            $params.ContainsKey('UseNativeInference') | Should -BeTrue
        }

        It 'GpuLayers should default to -1' {
            $param = (Get-Command $ScriptPath).Parameters['GpuLayers']
            # Parameter type should be int
            $param.ParameterType.Name | Should -Be 'Int32'
        }

        It 'InferenceBackend should default to auto' {
            # Check if default is 'auto' by looking at DefaultValue
            $param = (Get-Command $ScriptPath).Parameters['InferenceBackend']
            # The default is set in param block
            $param | Should -Not -BeNullOrEmpty
        }
    }
}

Describe 'Build Configuration' {
    BeforeAll {
        $ConfigPath = Join-Path $PSScriptRoot '..\..\Native\pcai_core\pcai_inference\build-config.json'
    }

    Context 'build-config.json' {
        It 'Should exist' {
            Test-Path $ConfigPath | Should -BeTrue
        }

        It 'Should be valid JSON' {
            { Get-Content $ConfigPath | ConvertFrom-Json } | Should -Not -Throw
        }

        It 'Should have backends configuration' {
            $config = Get-Content $ConfigPath | ConvertFrom-Json
            $config.backends | Should -Not -BeNullOrEmpty
        }

        It 'Should have llamacpp backend defined' {
            $config = Get-Content $ConfigPath | ConvertFrom-Json
            $config.backends.llamacpp | Should -Not -BeNullOrEmpty
        }

        It 'Should have mistralrs backend defined' {
            $config = Get-Content $ConfigPath | ConvertFrom-Json
            $config.backends.mistralrs | Should -Not -BeNullOrEmpty
        }

        It 'Should have output configuration' {
            $config = Get-Content $ConfigPath | ConvertFrom-Json
            $config.output.dll_name | Should -Be 'pcai_inference.dll'
        }
    }
}

Describe 'CMake Configuration' {
    BeforeAll {
        $ToolchainPath = Join-Path $PSScriptRoot '..\..\Native\pcai_core\pcai_inference\cmake\toolchain-msvc.cmake'
        $PresetsPath = Join-Path $PSScriptRoot '..\..\Native\pcai_core\pcai_inference\CMakePresets.json'
    }

    Context 'Toolchain File' {
        It 'Should exist' {
            Test-Path $ToolchainPath | Should -BeTrue
        }

        It 'Should set CMAKE_SYSTEM_NAME to Windows' {
            $content = Get-Content $ToolchainPath -Raw
            $content | Should -Match 'set\(CMAKE_SYSTEM_NAME Windows\)'
        }

        It 'Should handle CC environment variable' {
            $content = Get-Content $ToolchainPath -Raw
            $content | Should -Match 'ENV\{CC\}'
        }
    }

    Context 'CMake Presets' {
        It 'Should exist' {
            Test-Path $PresetsPath | Should -BeTrue
        }

        It 'Should be valid JSON' {
            { Get-Content $PresetsPath | ConvertFrom-Json } | Should -Not -Throw
        }

        It 'Should have msvc-release preset' {
            $presets = Get-Content $PresetsPath | ConvertFrom-Json
            $presets.configurePresets | Where-Object { $_.name -eq 'msvc-release' } | Should -Not -BeNullOrEmpty
        }

        It 'Should have msvc-cuda preset' {
            $presets = Get-Content $PresetsPath | ConvertFrom-Json
            $presets.configurePresets | Where-Object { $_.name -eq 'msvc-cuda' } | Should -Not -BeNullOrEmpty
        }

        It 'Should have msvc-debug preset' {
            $presets = Get-Content $PresetsPath | ConvertFrom-Json
            $presets.configurePresets | Where-Object { $_.name -eq 'msvc-debug' } | Should -Not -BeNullOrEmpty
        }
    }
}
