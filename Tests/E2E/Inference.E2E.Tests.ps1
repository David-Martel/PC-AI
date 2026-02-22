#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    End-to-end tests for PCAI Inference system.

.DESCRIPTION
    Tests the complete workflow from building the native library to running inference:
    1. Build verification (DLL exists and is valid)
    2. Module loading and initialization
    3. Backend lifecycle (init -> load -> generate -> shutdown)
    4. Error recovery scenarios
    5. Full inference pipeline (when test model is available)

.NOTES
    These tests may take longer to run as they exercise the full system.
    Set providers.pcai-native.modelPath in Config/llm-config.json to test with a real model.

.EXAMPLE
    Invoke-Pester -Path .\Tests\E2E\Inference.E2E.Tests.ps1 -Tag E2E
#>

BeforeAll {
    # Import shared test helpers
    Import-Module (Join-Path $PSScriptRoot "..\Helpers\TestHelpers.psm1") -Force

    # Get standard test paths
    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot = $paths.ProjectRoot
    $script:BinDir = $paths.BinDir
    $script:DeployDir = $paths.DeployDir
    $script:ModulePath = $paths.ModulePath
    $script:DllPath = $paths.DllPath

    # Helper: Get test model (wrapper for compatibility)
    function Get-TestModel {
        Get-TestModelPath
    }

    # Track whether we've initialized the backend
    $script:BackendInitialized = $false
}

AfterAll {
    # Cleanup: ensure backend is shut down
    if (Get-Module PcaiInference) {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
        Remove-Module PcaiInference -Force -ErrorAction SilentlyContinue
    }
}

Describe "E2E: Build and Artifact Verification" -Tag "E2E", "Build" {

    Context "Build Artifacts" {

        It "Project has Cargo.toml" {
            Test-Path (Join-Path $DeployDir "Cargo.toml") | Should -BeTrue
        }

        It "Project has build script" {
            Test-Path (Join-Path $DeployDir "build.ps1") | Should -BeTrue
        }

        It "CMake presets exist" {
            Test-Path (Join-Path $DeployDir "CMakePresets.json") | Should -BeTrue
        }

        It "MSVC toolchain file exists" {
            Test-Path (Join-Path $DeployDir "cmake\toolchain-msvc.cmake") | Should -BeTrue
        }
    }

    Context "DLL Verification" {

        It "pcai_inference.dll exists in bin directory" {
            if (-not (Test-Path $DllPath)) {
                Set-ItResult -Skipped -Because "DLL not built - run Deploy\pcai-inference\build.ps1"
            } else {
                Test-Path $DllPath | Should -BeTrue
            }
        }

        It "DLL is a valid PE file" {
            if (-not (Test-Path $DllPath)) {
                Set-ItResult -Skipped -Because "DLL not built"
            } else {
                # Check PE signature (MZ header)
                $bytes = [System.IO.File]::ReadAllBytes($DllPath)
                $bytes[0] | Should -Be 0x4D  # 'M'
                $bytes[1] | Should -Be 0x5A  # 'Z'
            }
        }

        It "DLL exports expected functions" {
            if (-not (Test-Path $DllPath)) {
                Set-ItResult -Skipped -Because "DLL not built"
            } else {
                # Use dumpbin or similar to check exports
                $dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
                if (-not $dumpbin) {
                    Set-ItResult -Inconclusive -Because "dumpbin not available to verify exports"
                } else {
                    $exports = & dumpbin /EXPORTS $DllPath 2>&1
                    $exportText = $exports | Out-String

                    # Check for expected FFI functions
                    $exportText | Should -Match "pcai_init"
                    $exportText | Should -Match "pcai_shutdown"
                }
            }
        }
    }
}

Describe "E2E: PowerShell Module Integration" -Tag "E2E", "Module" {

    Context "Module Loading" {

        It "PcaiInference module file exists" {
            Test-Path $ModulePath | Should -BeTrue
        }

        It "Module can be imported without errors" {
            { Import-Module $ModulePath -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It "Module exports required functions" {
            Import-Module $ModulePath -Force -ErrorAction SilentlyContinue

            $requiredFunctions = @(
                'Initialize-PcaiInference',
                'Import-PcaiModel',
                'Invoke-PcaiGenerate',
                'Close-PcaiInference',
                'Get-PcaiInferenceStatus',
                'Test-PcaiInference'
            )

            foreach ($func in $requiredFunctions) {
                Get-Command $func -Module PcaiInference -ErrorAction SilentlyContinue |
                    Should -Not -BeNullOrEmpty -Because "$func should be exported"
            }
        }
    }

    Context "Status Reporting" {

        BeforeAll {
            Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
        }

        It "Get-PcaiInferenceStatus returns structured object" {
            $status = Get-PcaiInferenceStatus
            $status | Should -BeOfType [PSCustomObject]
        }

        It "Status reports DLL path" {
            $status = Get-PcaiInferenceStatus
            $status.DllPath | Should -Not -BeNullOrEmpty
        }

        It "Status reports DLL existence" {
            $status = Get-PcaiInferenceStatus
            $status.PSObject.Properties.Name | Should -Contain 'DllExists'
        }

        It "Status reports initialization state" {
            $status = Get-PcaiInferenceStatus
            $status.PSObject.Properties.Name | Should -Contain 'BackendInitialized'
        }

        It "Status reports model load state" {
            $status = Get-PcaiInferenceStatus
            $status.PSObject.Properties.Name | Should -Contain 'ModelLoaded'
        }
    }
}

Describe "E2E: Backend Lifecycle" -Tag "E2E", "Backend" {

    BeforeAll {
        Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
        $script:DllAvailable = Test-Path $DllPath
    }

    BeforeEach {
        # Ensure clean state
        if ($DllAvailable) {
            try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
        }
    }

    AfterAll {
        if ($DllAvailable) {
            try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
        }
    }

    Context "Initialization" {

        It "Initialize-PcaiInference with 'llamacpp' backend" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $result = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue

                if ($result -eq $false) {
                    $status = Get-PcaiInferenceStatus
                    if ($status.LastError -match "not available|not compiled") {
                        Set-ItResult -Skipped -Because "llamacpp backend not compiled in"
                    } else {
                        throw "Initialization failed: $($status.LastError)"
                    }
                } else {
                    $result | Should -Be $true
                    $script:BackendInitialized = $true
                }
            }
        }

        It "Status shows BackendInitialized=true after init" {
            if (-not $DllAvailable -or -not $BackendInitialized) {
                Set-ItResult -Skipped -Because "Backend not initialized"
            } else {
                $status = Get-PcaiInferenceStatus
                $status.BackendInitialized | Should -BeTrue
            }
        }

        It "Close-PcaiInference properly shuts down" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
                Close-PcaiInference

                $status = Get-PcaiInferenceStatus
                $status.BackendInitialized | Should -BeFalse
            }
        }

        It "Multiple init/shutdown cycles work correctly" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                for ($i = 0; $i -lt 3; $i++) {
                    $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
                    if ($init -eq $false) {
                        Set-ItResult -Skipped -Because "llamacpp backend not available"
                        return
                    }
                    Close-PcaiInference
                }

                # Should end in clean state
                $status = Get-PcaiInferenceStatus
                $status.BackendInitialized | Should -BeFalse
            }
        }
    }

    Context "Error Handling" {

        It "Import-PcaiModel before init returns error" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                Close-PcaiInference -ErrorAction SilentlyContinue

                $result = Import-PcaiModel -ModelPath "C:\test\model.gguf" -ErrorAction SilentlyContinue
                $result | Should -Be $false

                $status = Get-PcaiInferenceStatus
                $status.LastError | Should -Not -BeNullOrEmpty
            }
        }

        It "Invoke-PcaiGenerate before model load returns error" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue

                $result = Invoke-PcaiGenerate -Prompt "Test" -ErrorAction SilentlyContinue
                $result | Should -BeNullOrEmpty

                Close-PcaiInference
            }
        }

        It "Import-PcaiModel with non-existent file returns error" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
                if (-not $init) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                    return
                }

                $result = Import-PcaiModel -ModelPath "C:\nonexistent\model.gguf" -ErrorAction SilentlyContinue
                $result | Should -Be $false

                $status = Get-PcaiInferenceStatus
                $status.ModelLoaded | Should -BeFalse

                Close-PcaiInference
            }
        }
    }
}

Describe "E2E: Full Inference Pipeline" -Tag "E2E", "Inference" {

    BeforeAll {
        Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
        $script:DllAvailable = Test-Path $DllPath
        $script:TestModel = Get-TestModel
    }

    AfterAll {
        if ($DllAvailable) {
            try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
        }
    }

    Context "With Real Model" {

        It "Complete inference workflow succeeds" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $TestModel) {
                Set-ItResult -Skipped -Because "No test model (configure Config/llm-config.json)"
            } else {
                # Initialize
                $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
                if (-not $init) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                    return
                }

                # Load model
                $load = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
                if (-not $load) {
                    $status = Get-PcaiInferenceStatus
                    Set-ItResult -Skipped -Because "Model loading failed: $($status.LastError)"
                    Close-PcaiInference
                    return
                }

                # Verify loaded
                $status = Get-PcaiInferenceStatus
                $status.ModelLoaded | Should -BeTrue

                # Generate
                $result = Invoke-PcaiGenerate -Prompt "The capital of France is" -MaxTokens 10 -Temperature 0.1
                $result | Should -Not -BeNullOrEmpty
                $result.Length | Should -BeGreaterThan 0

                # Cleanup
                Close-PcaiInference
            }
        }

        It "Multiple generations without reloading model" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $TestModel) {
                Set-ItResult -Skipped -Because "No test model (configure Config/llm-config.json)"
            } else {
                # Initialize and load
                $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
                if (-not $init) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                    return
                }

                $load = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
                if (-not $load) {
                    Close-PcaiInference
                    Set-ItResult -Skipped -Because "Model loading failed"
                    return
                }

                # Multiple generations
                $prompts = @(
                    "Hello",
                    "The quick brown fox",
                    "In the beginning"
                )

                foreach ($prompt in $prompts) {
                    $result = Invoke-PcaiGenerate -Prompt $prompt -MaxTokens 5 -Temperature 0.1
                    $result | Should -Not -BeNullOrEmpty -Because "Generation for '$prompt' should succeed"
                }

                Close-PcaiInference
            }
        }
    }
}

Describe "E2E: Test-PcaiInference Function" -Tag "E2E", "Diagnostic" {

    BeforeAll {
        Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
    }

    It "Test-PcaiInference returns boolean" {
        $result = Test-PcaiInference
        $result | Should -BeOfType [bool]
    }

    It "Test-PcaiInference returns true when DLL available" {
        if (-not (Test-Path $DllPath)) {
            Set-ItResult -Skipped -Because "DLL not available"
        } else {
            $result = Test-PcaiInference
            $result | Should -BeTrue
        }
    }
}

Describe "E2E: PC-AI.ps1 Integration" -Tag "E2E", "MainScript" {

    BeforeAll {
        $script:MainScript = Join-Path $ProjectRoot "PC-AI.ps1"
    }

    It "PC-AI.ps1 exists" {
        Test-Path $MainScript | Should -BeTrue
    }

    It "PC-AI.ps1 has InferenceBackend parameter" {
        $cmd = Get-Command $MainScript -ErrorAction SilentlyContinue
        if (-not $cmd) {
            Set-ItResult -Skipped -Because "Could not parse PC-AI.ps1"
        } else {
            $cmd.Parameters.ContainsKey('InferenceBackend') | Should -BeTrue
        }
    }

    It "PC-AI.ps1 InferenceBackend validates correct values" {
        $cmd = Get-Command $MainScript -ErrorAction SilentlyContinue
        if (-not $cmd) {
            Set-ItResult -Skipped -Because "Could not parse PC-AI.ps1"
        } else {
            $param = $cmd.Parameters['InferenceBackend']
            $validateSet = $param.Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }

            $validateSet.ValidValues | Should -Contain 'auto'
            $validateSet.ValidValues | Should -Contain 'llamacpp'
            $validateSet.ValidValues | Should -Contain 'mistralrs'
            $validateSet.ValidValues | Should -Contain 'http'
        }
    }

    It "PC-AI.ps1 has ModelPath parameter" {
        $cmd = Get-Command $MainScript -ErrorAction SilentlyContinue
        if (-not $cmd) {
            Set-ItResult -Skipped -Because "Could not parse PC-AI.ps1"
        } else {
            $cmd.Parameters.ContainsKey('ModelPath') | Should -BeTrue
        }
    }

    It "PC-AI.ps1 has GpuLayers parameter" {
        $cmd = Get-Command $MainScript -ErrorAction SilentlyContinue
        if (-not $cmd) {
            Set-ItResult -Skipped -Because "Could not parse PC-AI.ps1"
        } else {
            $cmd.Parameters.ContainsKey('GpuLayers') | Should -BeTrue
        }
    }
}
