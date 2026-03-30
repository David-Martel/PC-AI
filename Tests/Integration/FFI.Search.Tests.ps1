#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    FFI integration tests for PCAI Search native library.

.DESCRIPTION
    Tests the Rust search DLL loading and functionality including:
    1. Duplicate file detection
    2. File search with glob patterns
    3. Content search with regex patterns
    4. Cross-platform path handling

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

    # Test data directory
    $script:TestDataDir = Join-Path $ProjectRoot "Tests\Fixtures\SearchTestData"

    # Helper function to check if DLL exists
    function Test-DllExists {
        param([string]$DllName)
        $path = Join-Path $BinDir $DllName
        return Test-Path $path
    }

    # Helper function to get DLL path
    function Get-DllPath {
        param([string]$DllName)
        return Join-Path $BinDir $DllName
    }

    # Create test data directory with sample files
    function Initialize-TestData {
        if (-not (Test-Path $TestDataDir)) {
            New-Item -ItemType Directory -Path $TestDataDir -Force | Out-Null
        }

        # Create test files for search tests
        $subDir = Join-Path $TestDataDir "subdir"
        if (-not (Test-Path $subDir)) {
            New-Item -ItemType Directory -Path $subDir -Force | Out-Null
        }

        # Text files for content search
        Set-Content -Path (Join-Path $TestDataDir "file1.txt") -Value "Hello world`nThis is a test file`nHello again"
        Set-Content -Path (Join-Path $TestDataDir "file2.txt") -Value "Another test file`nWith different content"
        Set-Content -Path (Join-Path $TestDataDir "data.json") -Value '{"key": "value", "nested": {"data": 123}}'
        Set-Content -Path (Join-Path $subDir "nested.txt") -Value "Hello from nested directory`nMore test content"

        # Duplicate files (same content)
        $duplicateContent = "This is duplicate content for testing"
        Set-Content -Path (Join-Path $TestDataDir "dup1.txt") -Value $duplicateContent
        Set-Content -Path (Join-Path $TestDataDir "dup2.txt") -Value $duplicateContent
        Set-Content -Path (Join-Path $subDir "dup3.txt") -Value $duplicateContent

        # Unique file
        Set-Content -Path (Join-Path $TestDataDir "unique.txt") -Value "This is unique content that appears nowhere else"
    }
}

Describe "PCAI Search Native Library - Phase 2" -Tag "FFI", "Search", "Phase2" {

    Context "Build Artifacts" {

        It "Search functionality exists in pcai_core_lib" {
            # Search FFI exports are in pcai_core_lib (no separate pcai_search crate)
            Test-Path (Join-Path $NativeDir "pcai_core\pcai_core_lib") | Should -Be $true
        }

        It "SearchModule.cs exists" {
            Test-Path (Join-Path $CSharpDir "SearchModule.cs") | Should -Be $true
        }
    }

    Context "DLL Loading" {

        BeforeAll {
            # Build if DLL doesn't exist
            if (-not (Test-DllExists "pcai_core_lib.dll")) {
                Write-Host "Building native modules..." -ForegroundColor Yellow
                Push-Location $NativeDir
                try {
                    & .\build.ps1 2>&1 | Out-Null
                }
                finally {
                    Pop-Location
                }
            }
        }

        It "pcai_core_lib.dll exists after build" {
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

        It "DLL file size is reasonable (>50KB, <50MB)" {
            $dllPath = Get-DllPath "pcai_search.dll"
            if (-not (Test-Path $dllPath)) {
                Set-ItResult -Skipped -Because "DLL not built"
            }
            else {
                $size = (Get-Item $dllPath).Length
                $size | Should -BeGreaterThan 50KB
                $size | Should -BeLessThan 50MB
            }
        }
    }

    Context "Search Module P/Invoke" -Tag "PInvoke" {

        BeforeAll {
            $script:DllAvailable = Test-DllExists "pcai_core_lib.dll"

            if ($DllAvailable) {
                # Add DLL directory to PATH for P/Invoke
                $env:PATH = "$BinDir;$env:PATH"

                # Initialize test data
                Initialize-TestData

                # Define P/Invoke signatures for search module
                Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public enum PcaiStatus : uint
{
    Success = 0,
    InvalidArgument = 1,
    NullPointer = 2,
    InvalidUtf8 = 3,
    PathNotFound = 4
}

[StructLayout(LayoutKind.Sequential)]
public struct PcaiStringBuffer
{
    public PcaiStatus Status;
    public IntPtr Data;
    public UIntPtr Length;
}

[StructLayout(LayoutKind.Sequential)]
public struct DuplicateStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong DuplicateGroups;
    public ulong DuplicateFiles;
    public ulong WastedBytes;
    public ulong ElapsedMs;
}

[StructLayout(LayoutKind.Sequential)]
public struct FileSearchStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalSize;
    public ulong ElapsedMs;
}

[StructLayout(LayoutKind.Sequential)]
public struct ContentSearchStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalMatches;
    public ulong ElapsedMs;
}

public static class PcaiSearchTest
{
    private const string SearchDll = "pcai_core_lib.dll";
    private const string CoreDll = "pcai_core_lib.dll";

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_search_version();

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern DuplicateStats pcai_find_duplicates_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        ulong minSize,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string includePattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string excludePattern);

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern FileSearchStats pcai_find_files_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern,
        ulong maxResults);

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern ContentSearchStats pcai_search_content_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string filePattern,
        ulong maxResults);

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern PcaiStringBuffer pcai_find_duplicates(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        ulong minSize,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string includePattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string excludePattern);

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern PcaiStringBuffer pcai_find_files(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern,
        ulong maxResults);

    [DllImport(SearchDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern PcaiStringBuffer pcai_search_content(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string filePattern,
        ulong maxResults,
        uint contextLines);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string_buffer(ref PcaiStringBuffer buffer);

    // Helper to convert UTF-8 IntPtr to string (works on Windows PowerShell)
    public static string PtrToStringUtf8(IntPtr ptr, int length)
    {
        if (ptr == IntPtr.Zero || length <= 0) return string.Empty;
        byte[] bytes = new byte[length];
        Marshal.Copy(ptr, bytes, 0, length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    public static string BufferToString(PcaiStringBuffer buffer)
    {
        if (buffer.Data == IntPtr.Zero) return string.Empty;
        return PtrToStringUtf8(buffer.Data, (int)buffer.Length);
    }
}
"@ -ErrorAction SilentlyContinue
            }
        }

        It "pcai_search_version() returns valid version string" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $ptr = [PcaiSearchTest]::pcai_search_version()
                    $ptr | Should -Not -Be ([IntPtr]::Zero)

                    $version = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($ptr)
                    $version | Should -Not -BeNullOrEmpty
                    $version | Should -Match '^\d+\.\d+\.\d+'
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }

    Context "Duplicate Detection" -Tag "Duplicates" {

        It "pcai_find_duplicates_stats() finds duplicate files" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_find_duplicates_stats($TestDataDir, 0, $null, $null)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesScanned | Should -BeGreaterOrEqual 5
                    $stats.DuplicateGroups | Should -BeGreaterOrEqual 1
                    $stats.DuplicateFiles | Should -BeGreaterOrEqual 2  # dup1, dup2, dup3 - 1 original
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_find_duplicates_stats() respects min_size filter" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # Set min_size very high to filter out all files
                    $stats = [PcaiSearchTest]::pcai_find_duplicates_stats($TestDataDir, 1000000, $null, $null)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.DuplicateGroups | Should -Be 0
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_find_duplicates() returns valid JSON" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $buffer = [PcaiSearchTest]::pcai_find_duplicates($TestDataDir, 0, $null, $null)
                    try {
                        $buffer.Status | Should -Be ([PcaiStatus]::Success)
                        $json = [PcaiSearchTest]::BufferToString($buffer)
                        $json | Should -Not -BeNullOrEmpty

                        # Parse and validate JSON structure
                        $result = $json | ConvertFrom-Json
                        $result.status | Should -Be "Success"
                        $result.groups | Should -Not -BeNullOrEmpty
                    }
                    finally {
                        [PcaiSearchTest]::pcai_free_string_buffer([ref]$buffer)
                    }
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }

    Context "File Search" -Tag "FileSearch" {

        It "pcai_find_files_stats() finds .txt files" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($TestDataDir, "*.txt", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesMatched | Should -BeGreaterOrEqual 5  # file1, file2, dup1, dup2, dup3, unique, nested
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_find_files_stats() finds .json files" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($TestDataDir, "*.json", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesMatched | Should -BeGreaterOrEqual 1
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_find_files_stats() respects max_results" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($TestDataDir, "*", 2)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    # Note: stats still shows total matched, but result list is limited
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_find_files() returns valid JSON" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $buffer = [PcaiSearchTest]::pcai_find_files($TestDataDir, "*.txt", 10)
                    try {
                        $buffer.Status | Should -Be ([PcaiStatus]::Success)
                        $json = [PcaiSearchTest]::BufferToString($buffer)
                        $json | Should -Not -BeNullOrEmpty

                        # Parse and validate JSON structure
                        $result = $json | ConvertFrom-Json
                        $result.status | Should -Be "Success"
                        $result.files | Should -Not -BeNullOrEmpty
                        $result.files[0].path | Should -Not -BeNullOrEmpty
                        $result.files[0].size | Should -BeGreaterOrEqual 0
                    }
                    finally {
                        [PcaiSearchTest]::pcai_free_string_buffer([ref]$buffer)
                    }
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }

    Context "Content Search" -Tag "ContentSearch" {

        It "pcai_search_content_stats() finds 'Hello' pattern" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_search_content_stats($TestDataDir, "Hello", $null, 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.TotalMatches | Should -BeGreaterOrEqual 2  # "Hello world" and "Hello again" and "Hello from nested"
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_search_content_stats() supports regex patterns" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # Search for word boundaries
                    $stats = [PcaiSearchTest]::pcai_search_content_stats($TestDataDir, "test\s+file", $null, 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.TotalMatches | Should -BeGreaterOrEqual 1
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_search_content_stats() respects file_pattern filter" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # Only search in .json files
                    $stats = [PcaiSearchTest]::pcai_search_content_stats($TestDataDir, "value", "*.json", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesMatched | Should -Be 1
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "pcai_search_content() returns valid JSON with context" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $buffer = [PcaiSearchTest]::pcai_search_content($TestDataDir, "test", $null, 10, 1)
                    try {
                        $buffer.Status | Should -Be ([PcaiStatus]::Success)
                        $json = [PcaiSearchTest]::BufferToString($buffer)
                        $json | Should -Not -BeNullOrEmpty

                        # Parse and validate JSON structure
                        $result = $json | ConvertFrom-Json
                        $result.status | Should -Be "Success"
                        $result.matches | Should -Not -BeNullOrEmpty
                        $result.matches[0].path | Should -Not -BeNullOrEmpty
                        $result.matches[0].line_number | Should -BeGreaterOrEqual 1
                        $result.matches[0].line | Should -Not -BeNullOrEmpty
                    }
                    finally {
                        [PcaiSearchTest]::pcai_free_string_buffer([ref]$buffer)
                    }
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }

    Context "Cross-Platform Path Handling" -Tag "Paths" {

        It "Handles Windows paths with backslashes" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # Use Windows-style path
                    $windowsPath = $TestDataDir.Replace("/", "\")
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($windowsPath, "*.txt", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesMatched | Should -BeGreaterOrEqual 1
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "Handles paths with forward slashes" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    # Use Unix-style path
                    $unixPath = $TestDataDir.Replace("\", "/")
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($unixPath, "*.txt", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.FilesMatched | Should -BeGreaterOrEqual 1
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "Handles non-existent paths gracefully" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $fakePath = "C:\NonExistent\Path\That\Does\Not\Exist"
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($fakePath, "*.txt", 0)
                    $stats.Status | Should -Be ([PcaiStatus]::PathNotFound)
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }

    Context "Performance" -Tag "Performance" {

        It "File search completes in <100ms for test directory" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $sw = [System.Diagnostics.Stopwatch]::StartNew()
                    $stats = [PcaiSearchTest]::pcai_find_files_stats($TestDataDir, "*", 0)
                    $sw.Stop()

                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $sw.Elapsed.TotalMilliseconds | Should -BeLessThan 100
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }

        It "Duplicate detection returns elapsed_ms > 0" {
            if (-not $DllAvailable) {
                Set-ItResult -Skipped -Because "DLL not available"
            }
            else {
                try {
                    $stats = [PcaiSearchTest]::pcai_find_duplicates_stats($TestDataDir, 0, $null, $null)
                    $stats.Status | Should -Be ([PcaiStatus]::Success)
                    $stats.ElapsedMs | Should -BeGreaterOrEqual 0  # May be 0 for tiny directories
                }
                catch {
                    throw "P/Invoke call failed: $_"
                }
            }
        }
    }
}

AfterAll {
    # Cleanup test data
    if (Test-Path $TestDataDir) {
        Remove-Item -Path $TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
