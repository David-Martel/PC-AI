#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    FFI stress/torture tests for pcai_inference DLL.

.DESCRIPTION
    Subjects the C#/Rust FFI boundary to extreme conditions:
    1. Rapid init/shutdown cycling (100+ rounds)
    2. Concurrent P/Invoke calls from multiple threads
    3. Error path hammering (repeated invalid inputs)
    4. String allocation/deallocation stress
    5. Memory stability under sustained load
    6. Mixed valid/invalid operation interleaving
    7. Null pointer resilience

.NOTES
    Run after building the native modules. These tests do NOT require
    a model to be loaded - they test the FFI boundary itself.
#>

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot "..\Helpers\TestHelpers.psm1") -Force

    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot = $paths.ProjectRoot
    $script:BinDir = $paths.BinDir
    $script:DllPath = $paths.DllPath

    function Test-InferenceDllExists {
        Test-InferenceDllAvailable -ProjectRoot $script:ProjectRoot
    }

    $script:DllAvailable = Test-InferenceDllExists

    if ($DllAvailable) {
        $env:PATH = "$BinDir;$env:PATH"

        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

public static class PcaiStressTest
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
    public static extern int pcai_last_error_code();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr s);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_is_initialized();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_is_model_loaded();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_get_backend_name();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_version();

    /// <summary>
    /// Run init/shutdown cycles on multiple threads simultaneously.
    /// Returns the number of exceptions caught.
    /// </summary>
    public static int ConcurrentInitShutdown(int threadCount, int cyclesPerThread)
    {
        int exceptionCount = 0;
        var tasks = new Task[threadCount];
        var barrier = new Barrier(threadCount);

        for (int t = 0; t < threadCount; t++)
        {
            tasks[t] = Task.Run(() =>
            {
                barrier.SignalAndWait();  // Synchronize start
                for (int i = 0; i < cyclesPerThread; i++)
                {
                    try
                    {
                        pcai_init("invalid_backend_" + Thread.CurrentThread.ManagedThreadId);
                        pcai_shutdown();
                    }
                    catch (Exception)
                    {
                        Interlocked.Increment(ref exceptionCount);
                    }
                }
            });
        }

        Task.WaitAll(tasks);
        return exceptionCount;
    }

    /// <summary>
    /// Hammer error paths from multiple threads.
    /// Returns total exceptions caught (should be 0).
    /// </summary>
    public static int ConcurrentErrorPaths(int threadCount, int opsPerThread)
    {
        int exceptionCount = 0;
        var tasks = new Task[threadCount];
        var barrier = new Barrier(threadCount);

        for (int t = 0; t < threadCount; t++)
        {
            int threadId = t;
            tasks[t] = Task.Run(() =>
            {
                barrier.SignalAndWait();
                for (int i = 0; i < opsPerThread; i++)
                {
                    try
                    {
                        // Mix of error-inducing operations
                        switch (i % 7)
                        {
                            case 0:
                                pcai_init(null);
                                break;
                            case 1:
                                pcai_init("bogus_" + threadId + "_" + i);
                                break;
                            case 2:
                                pcai_load_model("C:\\nonexistent\\model_" + i + ".gguf", 0);
                                break;
                            case 3:
                                pcai_generate("test prompt " + i, 10, 0.5f);
                                break;
                            case 4:
                                pcai_free_string(IntPtr.Zero);
                                break;
                            case 5:
                                pcai_last_error();
                                break;
                            case 6:
                                pcai_shutdown();
                                break;
                        }
                    }
                    catch (Exception)
                    {
                        Interlocked.Increment(ref exceptionCount);
                    }
                }
            });
        }

        Task.WaitAll(tasks);
        return exceptionCount;
    }

    /// <summary>
    /// Stress the string allocation/deallocation path.
    /// Returns count of non-null error strings retrieved.
    /// </summary>
    public static int StringAllocStress(int iterations)
    {
        int nonNullCount = 0;
        for (int i = 0; i < iterations; i++)
        {
            // Trigger an error to create an error string
            pcai_init("invalid_" + i);

            // Read the error string (each call returns same pointer, no free needed for last_error)
            IntPtr errPtr = pcai_last_error();
            if (errPtr != IntPtr.Zero)
            {
                string msg = Marshal.PtrToStringAnsi(errPtr);
                if (!string.IsNullOrEmpty(msg))
                    nonNullCount++;
            }

            // Generate returns null when not initialized (no string to free)
            IntPtr genPtr = pcai_generate("test", 1, 0.1f);
            if (genPtr != IntPtr.Zero)
            {
                pcai_free_string(genPtr);
            }

            // Free null is safe
            pcai_free_string(IntPtr.Zero);
        }
        return nonNullCount;
    }

    /// <summary>
    /// Check if pcai_is_initialized entry point exists.
    /// </summary>
    public static bool HasIsInitialized()
    {
        try { pcai_is_initialized(); return true; }
        catch (EntryPointNotFoundException) { return false; }
    }

    /// <summary>
    /// Rapid state queries from multiple threads.
    /// Uses only functions known to exist in the DLL.
    /// Returns exception count (should be 0).
    /// </summary>
    public static int ConcurrentStateQueries(int threadCount, int queriesPerThread)
    {
        int exceptionCount = 0;
        bool hasIsInit = HasIsInitialized();
        var tasks = new Task[threadCount];
        var barrier = new Barrier(threadCount);

        for (int t = 0; t < threadCount; t++)
        {
            tasks[t] = Task.Run(() =>
            {
                barrier.SignalAndWait();
                for (int i = 0; i < queriesPerThread; i++)
                {
                    try
                    {
                        pcai_last_error_code();

                        IntPtr verPtr = pcai_version();
                        if (verPtr != IntPtr.Zero)
                            Marshal.PtrToStringAnsi(verPtr);

                        if (hasIsInit)
                        {
                            pcai_is_initialized();
                            pcai_is_model_loaded();

                            IntPtr namePtr = pcai_get_backend_name();
                            if (namePtr != IntPtr.Zero)
                                Marshal.PtrToStringAnsi(namePtr);
                        }
                    }
                    catch (EntryPointNotFoundException)
                    {
                        // Skip - function not in this DLL build
                    }
                    catch (Exception)
                    {
                        Interlocked.Increment(ref exceptionCount);
                    }
                }
            });
        }

        Task.WaitAll(tasks);
        return exceptionCount;
    }
}
"@ -ErrorAction SilentlyContinue
    }
}

Describe "PCAI Inference FFI - Stress Tests" -Tag "FFI", "Stress" {

    Context "Rapid Init/Shutdown Cycling" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        It "Survives 100 rapid init/shutdown cycles without crash" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $cycles = 100
                {
                    for ($i = 0; $i -lt $cycles; $i++) {
                        [PcaiStressTest]::pcai_init("invalid_backend_$i")
                        [PcaiStressTest]::pcai_shutdown()
                    }
                } | Should -Not -Throw
            }
        }

        It "Survives 500 shutdown-only calls (idempotency stress)" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                {
                    for ($i = 0; $i -lt 500; $i++) {
                        [PcaiStressTest]::pcai_shutdown()
                    }
                } | Should -Not -Throw
            }
        }

        It "Init/shutdown cycling has bounded memory growth (<20MB over 200 cycles)" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [GC]::Collect()
                [GC]::WaitForPendingFinalizers()
                $initialMemory = [GC]::GetTotalMemory($true)

                for ($i = 0; $i -lt 200; $i++) {
                    [PcaiStressTest]::pcai_init("invalid_$i")
                    [PcaiStressTest]::pcai_shutdown()
                }

                [GC]::Collect()
                [GC]::WaitForPendingFinalizers()
                $finalMemory = [GC]::GetTotalMemory($true)

                $growthMB = ($finalMemory - $initialMemory) / 1MB
                Write-Host "  Memory growth over 200 cycles: $([Math]::Round($growthMB, 2)) MB"
                $growthMB | Should -BeLessThan 20 -Because "init/shutdown should not leak memory"
            }
        }
    }

    Context "Concurrent Thread Safety" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiStressTest]::pcai_shutdown() } catch {}
            }
        }

        It "4 threads x 50 init/shutdown cycles = no unmanaged exceptions" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $exceptions = [PcaiStressTest]::ConcurrentInitShutdown(4, 50)
                $exceptions | Should -Be 0 -Because "concurrent init/shutdown must not throw unmanaged exceptions"
            }
        }

        It "8 threads x 100 mixed error operations = no crashes" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $exceptions = [PcaiStressTest]::ConcurrentErrorPaths(8, 100)
                $exceptions | Should -Be 0 -Because "concurrent error paths must not crash"
            }
        }

        It "8 threads x 200 state queries = no crashes" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $exceptions = [PcaiStressTest]::ConcurrentStateQueries(8, 200)
                $exceptions | Should -Be 0 -Because "concurrent state queries must be thread-safe"
            }
        }
    }

    Context "Error Path Hammering" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiStressTest]::pcai_shutdown() } catch {}
            }
        }

        It "1000 null backend init attempts = no crash" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                {
                    for ($i = 0; $i -lt 1000; $i++) {
                        [PcaiStressTest]::pcai_init($null)
                    }
                } | Should -Not -Throw
            }
        }

        It "1000 load_model before init = consistent error code" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [PcaiStressTest]::pcai_shutdown()
                $codes = @()
                for ($i = 0; $i -lt 1000; $i++) {
                    $codes += [PcaiStressTest]::pcai_load_model("C:\fake\model_$i.gguf", 0)
                }

                # All should be the same negative error code
                $uniqueCodes = $codes | Sort-Object -Unique
                $uniqueCodes.Count | Should -Be 1 -Because "error code should be consistent"
                $uniqueCodes[0] | Should -BeLessThan 0 -Because "should return error"
            }
        }

        It "1000 generate before init = all null pointers" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                [PcaiStressTest]::pcai_shutdown()
                $allNull = $true
                for ($i = 0; $i -lt 1000; $i++) {
                    $ptr = [PcaiStressTest]::pcai_generate("prompt $i", 10, 0.5)
                    if ($ptr -ne [IntPtr]::Zero) {
                        $allNull = $false
                        [PcaiStressTest]::pcai_free_string($ptr)
                    }
                }
                $allNull | Should -BeTrue -Because "generate before init should always return null"
            }
        }

        It "Error code accessor is stable under repeated calls" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                # Trigger an error
                [PcaiStressTest]::pcai_init("bogus")

                # Read error code 500 times - should be consistent
                $codes = @()
                for ($i = 0; $i -lt 500; $i++) {
                    $codes += [PcaiStressTest]::pcai_last_error_code()
                }

                $uniqueCodes = $codes | Sort-Object -Unique
                $uniqueCodes.Count | Should -Be 1 -Because "error code should not change between reads"
            }
        }
    }

    Context "String Allocation Stress" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiStressTest]::pcai_shutdown() } catch {}
            }
        }

        It "500 error string retrievals = all valid" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $validCount = [PcaiStressTest]::StringAllocStress(500)
                $validCount | Should -BeGreaterThan 400 -Because "most error strings should be retrievable"
                Write-Host "  Valid error strings retrieved: $validCount / 500"
            }
        }

        It "1000 pcai_free_string(null) calls = no crash" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                {
                    for ($i = 0; $i -lt 1000; $i++) {
                        [PcaiStressTest]::pcai_free_string([IntPtr]::Zero)
                    }
                } | Should -Not -Throw
            }
        }

        It "Error message content is valid UTF-8 after 200 error cycles" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $invalidMessages = 0
                for ($i = 0; $i -lt 200; $i++) {
                    [PcaiStressTest]::pcai_init("bad_backend_$i")

                    $errPtr = [PcaiStressTest]::pcai_last_error()
                    if ($errPtr -ne [IntPtr]::Zero) {
                        $msg = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($errPtr)
                        if ([string]::IsNullOrEmpty($msg) -or $msg.Contains([char]0xFFFD)) {
                            $invalidMessages++
                        }
                    }
                }

                $invalidMessages | Should -Be 0 -Because "all error messages should be valid strings"
            }
        }
    }

    Context "State Machine Integrity" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
            $script:HasStateApi = $false
            if ($DllAvailable) {
                $script:HasStateApi = [PcaiStressTest]::HasIsInitialized()
            }
        }

        AfterEach {
            if ($DllAvailable) {
                try { [PcaiStressTest]::pcai_shutdown() } catch {}
            }
        }

        It "is_initialized returns 0 after shutdown" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $HasStateApi) {
                Set-ItResult -Skipped -Because "pcai_is_initialized not exported in this DLL build"
            } else {
                [PcaiStressTest]::pcai_shutdown()
                $result = [PcaiStressTest]::pcai_is_initialized()
                $result | Should -Be 0
            }
        }

        It "is_model_loaded returns 0 when no model loaded" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $HasStateApi) {
                Set-ItResult -Skipped -Because "pcai_is_model_loaded not exported in this DLL build"
            } else {
                [PcaiStressTest]::pcai_shutdown()
                $result = [PcaiStressTest]::pcai_is_model_loaded()
                $result | Should -Be 0
            }
        }

        It "State flags are consistent through 100 init/shutdown transitions" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $HasStateApi) {
                Set-ItResult -Skipped -Because "State API not exported in this DLL build"
            } else {
                for ($i = 0; $i -lt 100; $i++) {
                    [PcaiStressTest]::pcai_shutdown()
                    [PcaiStressTest]::pcai_is_initialized() | Should -Be 0

                    [PcaiStressTest]::pcai_init("invalid_$i")
                }

                [PcaiStressTest]::pcai_shutdown()
                [PcaiStressTest]::pcai_is_initialized() | Should -Be 0
            }
        }

        It "pcai_version returns non-null string" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $verPtr = [PcaiStressTest]::pcai_version()
                $verPtr | Should -Not -Be ([IntPtr]::Zero)
                $version = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($verPtr)
                $version | Should -Not -BeNullOrEmpty
                Write-Host "  DLL version: $version"
            }
        }

        It "Backend name is null/empty when not initialized" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } elseif (-not $HasStateApi) {
                Set-ItResult -Skipped -Because "pcai_get_backend_name not exported in this DLL build"
            } else {
                [PcaiStressTest]::pcai_shutdown()
                $namePtr = [PcaiStressTest]::pcai_get_backend_name()
                if ($namePtr -ne [IntPtr]::Zero) {
                    $name = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($namePtr)
                    ($null -eq $name -or $name -eq "" -or $name -eq "none") | Should -BeTrue
                }
            }
        }
    }

    Context "Performance Under Stress" {

        BeforeAll {
            $script:DllAvailable = Test-InferenceDllExists
        }

        AfterAll {
            if ($DllAvailable) {
                try { [PcaiStressTest]::pcai_shutdown() } catch {}
            }
        }

        It "100 error cycles complete in <500ms" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $sw = [System.Diagnostics.Stopwatch]::StartNew()

                for ($i = 0; $i -lt 100; $i++) {
                    [PcaiStressTest]::pcai_init("invalid_$i")
                    [PcaiStressTest]::pcai_last_error()
                    [PcaiStressTest]::pcai_last_error_code()
                    [PcaiStressTest]::pcai_shutdown()
                }

                $sw.Stop()
                Write-Host "  100 error cycles: $($sw.ElapsedMilliseconds)ms ($([Math]::Round($sw.ElapsedMilliseconds / 100, 2))ms/cycle)"
                $sw.ElapsedMilliseconds | Should -BeLessThan 500 -Because "error paths should be fast"
            }
        }

        It "1000 state queries complete in <200ms" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $hasStateApi = [PcaiStressTest]::HasIsInitialized()
                $sw = [System.Diagnostics.Stopwatch]::StartNew()

                for ($i = 0; $i -lt 1000; $i++) {
                    [PcaiStressTest]::pcai_last_error_code()
                    if ($hasStateApi) {
                        [PcaiStressTest]::pcai_is_initialized()
                        [PcaiStressTest]::pcai_is_model_loaded()
                    }
                }

                $sw.Stop()
                $queryCount = if ($hasStateApi) { 3000 } else { 1000 }
                Write-Host "  $queryCount state queries: $($sw.ElapsedMilliseconds)ms"
                $sw.ElapsedMilliseconds | Should -BeLessThan 200 -Because "state queries should be sub-microsecond"
            }
        }

        It "Concurrent 8-thread stress test completes in <5s" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            } else {
                $sw = [System.Diagnostics.Stopwatch]::StartNew()

                $exceptions = [PcaiStressTest]::ConcurrentErrorPaths(8, 200)

                $sw.Stop()
                Write-Host "  8 threads x 200 ops: $($sw.ElapsedMilliseconds)ms, $exceptions exceptions"
                $sw.ElapsedMilliseconds | Should -BeLessThan 5000
                $exceptions | Should -Be 0
            }
        }
    }
}
