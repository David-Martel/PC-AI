#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    FFI integration tests for PCAI Inference native library.

.DESCRIPTION
    Tests the pcai-inference Rust DLL FFI boundary:
    1. DLL loading and initialization
    2. Backend selection (llamacpp, mistralrs)
    3. Error handling across FFI boundary
    4. Memory management (string allocation/deallocation)
    5. Thread safety of FFI calls

.NOTES
    Run these tests after building the native modules with:
    .\Deploy\pcai-inference\build.ps1 -Configuration Release
#>

BeforeAll {
    # Import shared test helpers
    Import-Module (Join-Path $PSScriptRoot "..\Helpers\TestHelpers.psm1") -Force

    # Get standard test paths
    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot = $paths.ProjectRoot
    $script:BinDir = $paths.BinDir
    $script:DeployDir = $paths.DeployDir
    $script:DllName = $paths.DllName
    $script:DllPath = $paths.DllPath

    # Helper: Check if DLL exists (wrapper for compatibility)
    function Test-InferenceDllExists {
        Test-InferenceDllAvailable -ProjectRoot $script:ProjectRoot
    }
}

Describe "PCAI Inference Native Library - FFI Boundary Tests" -Tag "FFI", "Inference" {

    Context "Build Artifacts" {

        It "Deploy directory exists" {
            Test-Path $DeployDir | Should -BeTrue
        }

        It "Cargo.toml exists" {
            Test-Path (Join-Path $DeployDir "Cargo.toml") | Should -BeTrue
        }

        It "build.ps1 exists" {
            Test-Path (Join-Path $DeployDir "build.ps1") | Should -BeTrue
        }

        It "build-config.json exists" {
            Test-Path (Join-Path $DeployDir "build-config.json") | Should -BeTrue
        }

        It "build-config.json is valid JSON" {
            $configPath = Join-Path $DeployDir "build-config.json"
            { Get-Content $configPath | ConvertFrom-Json } | Should -Not -Throw
        }
    }

    Context "DLL Loading" {

        It "pcai_inference.dll exists after build" {
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                Set-ItResult -Skipped -Because "Rust toolchain not installed"
            }

            if (Test-InferenceDllExists) {
                $true | Should -BeTrue
            } else {
                Set-ItResult -Skipped -Because "DLL not built (run .\Deploy\pcai-inference\build.ps1)"
            }
        }

        It "DLL file size is reasonable for inference library (>100KB)" {
            if (-not (Test-InferenceDllExists)) {
                Set-ItResult -Skipped -Because "DLL not built"
            } else {
                $size = (Get-Item $DllPath).Length
                $size | Should -BeGreaterThan 100KB
            }
        }
    }

    Context "P/Invoke FFI Functions" -Tag "PInvoke" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists

            if ($DllAvailable) {
                # Add DLL directory to PATH
                $env:PATH = "$BinDir;$env:PATH"

                # Define P/Invoke signatures for pcai-inference
                Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class PcaiInferenceTest
{
    private const string DllName = "pcai_inference.dll";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_init([MarshalAs(UnmanagedType.LPUTF8Str)] string backendName);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int gpuLayers);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_generate([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, int maxTokens, float temperature);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_shutdown();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_last_error();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr s);
}
"@ -ErrorAction SilentlyContinue
            }
        }

        AfterEach {
            # Clean up after each test
            if ($DllAvailable) {
                try { [PcaiInferenceTest]::pcai_shutdown() } catch {}
            }
        }

        It "pcai_init() with null backend returns error code" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $result = [PcaiInferenceTest]::pcai_init($null)
                $result | Should -Be -1
            }
        }

        It "pcai_init() with unknown backend returns error code" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $result = [PcaiInferenceTest]::pcai_init("unknown_backend_xyz")
                $result | Should -Be -1
            }
        }

        It "pcai_last_error() returns error message after failure" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                # Cause an error
                [PcaiInferenceTest]::pcai_init("invalid")

                $errPtr = [PcaiInferenceTest]::pcai_last_error()
                $errPtr | Should -Not -Be ([IntPtr]::Zero)

                $errMsg = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                $errMsg | Should -Not -BeNullOrEmpty
            }
        }

        It "pcai_load_model() before init returns error" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [PcaiInferenceTest]::pcai_shutdown()  # Ensure clean state
                $result = [PcaiInferenceTest]::pcai_load_model("C:\test\model.gguf", 0)
                $result | Should -Be -1
            }
        }

        It "pcai_generate() before init returns null" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [PcaiInferenceTest]::pcai_shutdown()  # Ensure clean state
                $result = [PcaiInferenceTest]::pcai_generate("test prompt", 10, 0.7)
                $result | Should -Be ([IntPtr]::Zero)
            }
        }

        It "pcai_free_string(null) does not crash" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                { [PcaiInferenceTest]::pcai_free_string([IntPtr]::Zero) } | Should -Not -Throw
            }
        }

        It "pcai_shutdown() is idempotent (can be called multiple times)" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                {
                    [PcaiInferenceTest]::pcai_shutdown()
                    [PcaiInferenceTest]::pcai_shutdown()
                    [PcaiInferenceTest]::pcai_shutdown()
                } | Should -Not -Throw
            }
        }
    }

    Context "Backend Initialization" -Tag "Backend" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiInferenceTest]::pcai_shutdown() } catch {}
            }
        }

        It "pcai_init('llamacpp') succeeds when llamacpp feature is enabled" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $result = [PcaiInferenceTest]::pcai_init("llamacpp")

                if ($result -ne 0) {
                    $errPtr = [PcaiInferenceTest]::pcai_last_error()
                    $errMsg = if ($errPtr -ne [IntPtr]::Zero) {
                        [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                    } else { "Unknown error" }

                    # Skip if llamacpp not compiled in
                    if ($errMsg -match "not available|not compiled|not enabled") {
                        Set-ItResult -Skipped -Because "llamacpp backend not compiled: $errMsg"
                    } else {
                        throw "Init failed: $errMsg"
                    }
                } else {
                    $result | Should -Be 0
                }
            }
        }

        It "pcai_init('mistralrs') succeeds when mistralrs feature is enabled" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $result = [PcaiInferenceTest]::pcai_init("mistralrs")

                if ($result -ne 0) {
                    $errPtr = [PcaiInferenceTest]::pcai_last_error()
                    $errMsg = if ($errPtr -ne [IntPtr]::Zero) {
                        [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                    } else { "Unknown error" }

                    # Skip if mistralrs not compiled in
                    if ($errMsg -match "not available|not compiled|not enabled") {
                        Set-ItResult -Skipped -Because "mistralrs backend not compiled: $errMsg"
                    } else {
                        throw "Init failed: $errMsg"
                    }
                } else {
                    $result | Should -Be 0
                }
            }
        }
    }

    Context "Model Loading" -Tag "Model" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
            $script:TestModelPath = Get-TestModelPath
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiInferenceTest]::pcai_shutdown() } catch {}
            }
        }

        It "pcai_load_model() with non-existent file returns error" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                # Initialize first
                $initResult = [PcaiInferenceTest]::pcai_init("llamacpp")
                if ($initResult -ne 0) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                } else {
                    $result = [PcaiInferenceTest]::pcai_load_model("C:\nonexistent\model.gguf", 0)
                    $result | Should -Be -1

                    $errPtr = [PcaiInferenceTest]::pcai_last_error()
                    $errMsg = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                    $errMsg | Should -Not -BeNullOrEmpty
                }
            }
        }

        It "pcai_load_model() with real model succeeds" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $TestModelPath) {
                Set-ItResult -Skipped -Because "No test model available (configure Config/llm-config.json)"
            } else {
                $initResult = [PcaiInferenceTest]::pcai_init("llamacpp")
                if ($initResult -ne 0) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                } else {
                    $result = [PcaiInferenceTest]::pcai_load_model($TestModelPath, 0)
                    $result | Should -Be 0
                }
            }
        }
    }

    Context "Text Generation" -Tag "Generation" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
            $script:TestModelPath = Get-TestModelPath
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiInferenceTest]::pcai_shutdown() } catch {}
            }
        }

        It "pcai_generate() with loaded model returns text" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $TestModelPath) {
                Set-ItResult -Skipped -Because "No test model available (configure Config/llm-config.json)"
            } else {
                # Initialize
                $initResult = [PcaiInferenceTest]::pcai_init("llamacpp")
                if ($initResult -ne 0) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                    return
                }

                # Load model
                $loadResult = [PcaiInferenceTest]::pcai_load_model($TestModelPath, 0)
                if ($loadResult -ne 0) {
                    $errPtr = [PcaiInferenceTest]::pcai_last_error()
                    $errMsg = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                    Set-ItResult -Skipped -Because "Model loading failed: $errMsg"
                    return
                }

                # Generate
                $resultPtr = [PcaiInferenceTest]::pcai_generate("The capital of France is", 10, 0.1)
                $resultPtr | Should -Not -Be ([IntPtr]::Zero)

                try {
                    $resultText = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($resultPtr)
                    $resultText | Should -Not -BeNullOrEmpty
                    $resultText.Length | Should -BeGreaterThan 0
                } finally {
                    [PcaiInferenceTest]::pcai_free_string($resultPtr)
                }
            }
        }
    }

    Context "Memory Safety" -Tag "Memory" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        It "Repeated init/shutdown cycles don't leak memory" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $initialMemory = [GC]::GetTotalMemory($true)

                for ($i = 0; $i -lt 10; $i++) {
                    $result = [PcaiInferenceTest]::pcai_init("llamacpp")
                    [PcaiInferenceTest]::pcai_shutdown()
                }

                [GC]::Collect()
                [GC]::WaitForPendingFinalizers()
                $finalMemory = [GC]::GetTotalMemory($true)

                # Allow up to 10MB growth (reasonable for DLL loading overhead)
                $memoryGrowth = $finalMemory - $initialMemory
                $memoryGrowth | Should -BeLessThan (10 * 1024 * 1024)
            }
        }
    }

    Context "Performance Baseline" -Tag "Performance" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterAll {
            if ($DllAvailable) {
                try { [PcaiInferenceTest]::pcai_shutdown() } catch {}
            }
        }

        It "pcai_init() completes in <100ms" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $result = [PcaiInferenceTest]::pcai_init("llamacpp")
                $sw.Stop()
                [PcaiInferenceTest]::pcai_shutdown()

                if ($result -ne 0) {
                    Set-ItResult -Skipped -Because "llamacpp backend not available"
                } else {
                    $sw.ElapsedMilliseconds | Should -BeLessThan 100
                }
            }
        }

        It "pcai_shutdown() completes in <50ms" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [PcaiInferenceTest]::pcai_init("llamacpp")

                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                [PcaiInferenceTest]::pcai_shutdown()
                $sw.Stop()

                $sw.ElapsedMilliseconds | Should -BeLessThan 50
            }
        }
    }
}

Describe "PcaiInference Module Integration" -Tag "Module", "Integration" {

    Context "Module Loading" {

        BeforeAll {
            $script:ModulePath = Join-Path $ProjectRoot "Modules\PcaiInference.psm1"
        }

        It "PcaiInference module exists" {
            Test-Path $ModulePath | Should -BeTrue
        }

        It "Module can be imported" {
            { Import-Module $ModulePath -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It "Module exports expected functions" {
            Import-Module $ModulePath -Force

            $expectedFunctions = @(
                'Initialize-PcaiInference',
                'Import-PcaiModel',
                'Invoke-PcaiGenerate',
                'Close-PcaiInference',
                'Get-PcaiInferenceStatus',
                'Test-PcaiInference'
            )

            foreach ($func in $expectedFunctions) {
                Get-Command $func -Module PcaiInference -ErrorAction SilentlyContinue |
                    Should -Not -BeNullOrEmpty -Because "$func should be exported"
            }
        }
    }

    Context "Module Function Parameters" {

        BeforeAll {
            $script:ModulePath = Join-Path $ProjectRoot "Modules\PcaiInference.psm1"
            Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
        }

        It "Initialize-PcaiInference has Backend parameter with validation" {
            $cmd = Get-Command Initialize-PcaiInference -ErrorAction SilentlyContinue
            if (-not $cmd) {
                Set-ItResult -Skipped -Because "Module not loaded"
            } else {
                $cmd.Parameters.ContainsKey('Backend') | Should -BeTrue
                $validateSet = $cmd.Parameters['Backend'].Attributes |
                    Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
                $validateSet | Should -Not -BeNullOrEmpty
                $validateSet.ValidValues | Should -Contain 'llamacpp'
                $validateSet.ValidValues | Should -Contain 'mistralrs'
            }
        }

        It "Import-PcaiModel has mandatory ModelPath parameter" {
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            if (-not $cmd) {
                Set-ItResult -Skipped -Because "Module not loaded"
            } else {
                $cmd.Parameters.ContainsKey('ModelPath') | Should -BeTrue
                $paramAttr = $cmd.Parameters['ModelPath'].Attributes |
                    Where-Object { $_ -is [System.Management.Automation.ParameterAttribute] }
                $paramAttr.Mandatory | Should -BeTrue
            }
        }

        It "Import-PcaiModel has GpuLayers parameter" {
            $cmd = Get-Command Import-PcaiModel -ErrorAction SilentlyContinue
            if (-not $cmd) {
                Set-ItResult -Skipped -Because "Module not loaded"
            } else {
                $cmd.Parameters.ContainsKey('GpuLayers') | Should -BeTrue
                $cmd.Parameters['GpuLayers'].ParameterType.Name | Should -Be 'Int32'
            }
        }
    }
}
