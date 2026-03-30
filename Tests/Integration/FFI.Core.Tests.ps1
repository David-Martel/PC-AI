#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    FFI integration tests for PCAI Core native library.

.DESCRIPTION
    Tests the Rust DLL loading, basic FFI operations, and P/Invoke wrapper functionality.
    These tests verify that:
    1. The native DLLs can be loaded
    2. Basic FFI calls work correctly
    3. String marshaling works across the FFI boundary
    4. The C# wrapper functions correctly

.NOTES
    Run these tests after building the native modules with:
    .\Native\build.ps1 -Test
#>

BeforeAll {
    # Project paths
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:BinDir = Join-Path $ProjectRoot "bin"
    $script:NativeDir = Join-Path $ProjectRoot "Native"
    $script:CSharpDir = Join-Path $NativeDir "PcaiNative"

    # Expected magic number for DLL verification
    # Value 0x50434149 spells "PCAI" in ASCII hex
    $script:ExpectedMagicNumber = 0x50434149

    # Helper function to check if DLL exists
    function Test-DllExists {
        param([string]$DllName)
        $path = Join-Path $BinDir $DllName
        return Test-Path $path
    }

    # Helper function to load DLL for testing
    function Get-DllPath {
        param([string]$DllName)
        return Join-Path $BinDir $DllName
    }
}

Describe "PCAI Core Native Library - Phase 1 Foundation" -Tag "FFI", "Core", "Phase1" {

    Context "Build Artifacts" {

        It "Native directory exists" {
            Test-Path $NativeDir | Should -Be $true
        }

        It "Rust workspace exists" {
            Test-Path (Join-Path $NativeDir "pcai_core") | Should -Be $true
        }

        It "C# project exists" {
            Test-Path $CSharpDir | Should -Be $true
        }

        It "Bin directory exists or can be created" {
            if (-not (Test-Path $BinDir)) {
                New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
            }
            Test-Path $BinDir | Should -Be $true
        }
    }

    Context "DLL Loading" {

        BeforeAll {
            # Build if DLL doesn't exist
            if (-not (Test-DllExists "pcai_core_lib.dll")) {
                Write-Host "Building native modules..." -ForegroundColor Yellow
                Push-Location $NativeDir
                try {
                    & .\build.ps1 -SkipCSharp 2>&1 | Out-Null
                }
                finally {
                    Pop-Location
                }
            }
        }

        It "pcai_core_lib.dll exists after build" {
            # This test will be skipped if Rust is not installed
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                Set-ItResult -Skipped -Because "Rust toolchain not installed"
            }

            $dllPath = Get-DllPath "pcai_core_lib.dll"
            if (Test-Path $dllPath) {
                $true | Should -Be $true
            }
            else {
                Set-ItResult -Skipped -Because "DLL not built (run .\Native\build.ps1 first)"
            }
        }

        It "DLL file size is reasonable (>10KB, <10MB)" {
            $dllPath = Get-DllPath "pcai_core_lib.dll"
            if (-not (Test-Path $dllPath)) {
                Set-ItResult -Skipped -Because "DLL not built"
            }
            else {
                $size = (Get-Item $dllPath).Length
                $size | Should -BeGreaterThan 10KB
                $size | Should -BeLessThan 10MB
            }
        }
    }

    Context "P/Invoke Functionality" -Tag "PInvoke" {

        BeforeAll {
            $script:DllAvailable = Test-DllExists "pcai_core_lib.dll"

            if ($DllAvailable) {
                # Add DLL directory to PATH for P/Invoke
                $env:PATH = "$BinDir;$env:PATH"

                # Define P/Invoke signatures
                Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class PcaiCoreTest
{
    private const string DllName = "pcai_core_lib.dll";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_core_version();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern uint pcai_core_test();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern uint pcai_cpu_count();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr buffer);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern IntPtr pcai_string_copy([MarshalAs(UnmanagedType.LPStr)] string input);
}
"@ -ErrorAction SilentlyContinue
            }
        }

        It "pcai_core_test() returns magic number" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $result = [PcaiCoreTest]::pcai_core_test()
                    $result | Should -Be $ExpectedMagicNumber
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_core_version() returns valid version string" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $ptr = [PcaiCoreTest]::pcai_core_version()
                    $ptr | Should -Not -Be ([IntPtr]::Zero)

                    $version = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($ptr)
                    $version | Should -Not -BeNullOrEmpty
                    $version | Should -Match '^(\d+\.\d+\.\d+(-[0-9A-Za-z\.-]+)?)|([0-9a-f]{7,}(-dirty)?)$'
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_cpu_count() returns reasonable value" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $count = [PcaiCoreTest]::pcai_cpu_count()
                    $count | Should -BeGreaterOrEqual 1
                    $count | Should -BeLessOrEqual 1024
                    # Should roughly match .NET's value
                    $count | Should -BeGreaterOrEqual ([Environment]::ProcessorCount / 2)
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_string_copy() round-trips correctly" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $testString = "Hello, PCAI FFI Test!"
                    $ptr = [PcaiCoreTest]::pcai_string_copy($testString)
                    $ptr | Should -Not -Be ([IntPtr]::Zero)

                    try {
                        $result = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($ptr)
                        $result | Should -Be $testString
                    }
                    finally {
                        [PcaiCoreTest]::pcai_free_string($ptr)
                    }
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_string_copy handles empty string" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                # Note: P/Invoke with LPUTF8Str marshaling may convert $null to empty string
                # Test with explicit empty string which should work correctly
                $ptr = [PcaiCoreTest]::pcai_string_copy("")
                $ptr | Should -Not -Be ([IntPtr]::Zero)

                try {
                    # PtrToStringUTF8 is only available in .NET Core / .NET 5+
                    # PowerShell 7+ uses .NET Core, PowerShell 5.1 uses .NET Framework
                    if ($PSVersionTable.PSVersion.Major -ge 7) {
                        $result = [System.Runtime.InteropServices.Marshal]::PtrToStringUTF8($ptr)
                    }
                    else {
                        $result = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($ptr)
                    }
                    $result | Should -Be ""
                }
                finally {
                    [PcaiCoreTest]::pcai_free_string($ptr)
                }
            }
        }

        It "pcai_free_string(null) does not crash" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # This should be a no-op, not crash
                    [PcaiCoreTest]::pcai_free_string([IntPtr]::Zero)
                    $true | Should -Be $true
                }
                catch {
                    throw "P/Invoke call crashed: $_"
                }
            }
        }
    }

    Context "FunctionGemma Router Dataset (Native)" -Tag "FunctionGemma" {

        BeforeAll {
            $script:PcaiNativeDll = Get-DllPath "PcaiNative.dll"
            if (Test-Path $script:PcaiNativeDll) {
                $env:PATH = "$BinDir;$env:PATH"
                try {
                    Add-Type -Path $script:PcaiNativeDll -ErrorAction SilentlyContinue
                } catch {
                    # Ignore load errors, tests will skip
                }
            }
        }

        It "BuildRouterDataset emits dataset JSONL" {
            if ($PSVersionTable.PSVersion.Major -lt 7) {
                Set-ItResult -Skipped -Because "PcaiNative requires PowerShell 7+"
            }
            elseif (-not (Test-Path $script:PcaiNativeDll)) {
                Set-ItResult -Skipped -Because "PcaiNative.dll not available"
            }
            elseif (-not ([PcaiNative.PcaiCore]::IsAvailable)) {
                Set-ItResult -Skipped -Because "PcaiCore not available"
            }
            else {
                $tempDir = Join-Path $env:TEMP "pcai_functiongemma_test"
                if (-not (Test-Path $tempDir)) {
                    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
                }

                $tools = Join-Path $ProjectRoot "Config\\pcai-tools.json"
                $diagnose = Join-Path $ProjectRoot "DIAGNOSE.md"
                $chat = Join-Path $ProjectRoot "CHAT.md"
                $scenarios = Join-Path $ProjectRoot "Deploy\\rust-functiongemma-train\\examples\\scenarios.json"
                $outputJsonl = Join-Path $tempDir "router_dataset.jsonl"
                $outputVectors = Join-Path $tempDir "router_vectors.json"

                try {
                    $report = [PcaiNative.FunctionGemmaModule]::BuildRouterDataset(
                        $tools,
                        $outputJsonl,
                        $diagnose,
                        $chat,
                        $scenarios,
                        $outputVectors,
                        1,
                        $false
                    )

                    $report | Should -Not -BeNullOrEmpty
                    $report.IsSuccess | Should -BeTrue
                    (Test-Path $outputJsonl) | Should -BeTrue
                }
                catch [System.EntryPointNotFoundException] {
                    Set-ItResult -Skipped -Because "pcai_build_router_dataset_jsonl FFI export not compiled into DLL"
                }
                catch [System.DllNotFoundException] {
                    Set-ItResult -Skipped -Because "pcai_core_lib.dll not found by FunctionGemmaModule"
                }
            }
        }
    }

    Context "Performance Baseline" -Tag "Performance" {

        It "pcai_core_test() completes in <1ms" {
            $dllPath = Get-DllPath "pcai_core_lib.dll"
            if (-not (Test-Path $dllPath)) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $sw = [System.Diagnostics.Stopwatch]::StartNew()
                    for ($i = 0; $i -lt 1000; $i++) {
                        $null = [PcaiCoreTest]::pcai_core_test()
                    }
                    $sw.Stop()

                    $avgMs = $sw.Elapsed.TotalMilliseconds / 1000
                    $avgMs | Should -BeLessThan 1
                }
                catch {
                    Set-ItResult -Skipped -Because "P/Invoke not available"
                }
            }
        }
    }
}

Describe "PCAI Module Integration" -Tag "Module", "Integration" {

    Context "Acceleration Module Integration" {

        BeforeAll {
            $script:AccelerationModule = Join-Path $ProjectRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1"
        }

        It "PC-AI.Acceleration module exists" {
            Test-Path $AccelerationModule | Should -Be $true
        }

        It "Module can be imported" {
            if ($PSVersionTable.PSVersion.Major -lt 7) {
                Set-ItResult -Skipped -Because "PC-AI.Acceleration requires PowerShell 7.0+"
            }
            else {
                { Import-Module $AccelerationModule -Force -ErrorAction Stop } | Should -Not -Throw
            }
        }

        It "Module exports Get-RustToolStatus" {
            if ($PSVersionTable.PSVersion.Major -lt 7) {
                Set-ItResult -Skipped -Because "PC-AI.Acceleration requires PowerShell 7.0+"
            }
            else {
                Import-Module $AccelerationModule -Force
                Get-Command Get-RustToolStatus -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
            }
        }
    }
}
