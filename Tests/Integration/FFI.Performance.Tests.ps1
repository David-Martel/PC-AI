#Requires -Version 7.0
#Requires -Modules Pester

<#
.SYNOPSIS
    FFI Integration tests for pcai_performance.dll

.DESCRIPTION
    Tests the performance module for disk usage analysis, process monitoring,
    and memory statistics functionality via C# P/Invoke wrapper.
#>

BeforeDiscovery {
    # Load the C# assembly at discovery time for Skip conditions
    $BinDir = Join-Path $PSScriptRoot "..\..\bin"
    $PcaiNativeDll = Join-Path $BinDir "PcaiNative.dll"

    $script:PerformanceAvailable = $false
    # Presence of the DLL, kept separate from whether it loads. Presence decides
    # whether this suite applies at all; loading is what the DLL Loading tests
    # assert. Collapsing the two would make those tests unable to fail.
    $script:PerformanceDllPresent = Test-Path $PcaiNativeDll
    if ($script:PerformanceDllPresent) {
        try {
            Add-Type -Path $PcaiNativeDll -ErrorAction Stop
        }
        catch [System.Reflection.ReflectionTypeLoadException] {
            # Already loaded, ignore
        }
        catch {
            # Ignore other errors
        }

        try {
            $version = [PcaiNative.PerformanceModule]::GetVersion()
            $script:PerformanceAvailable = $version -gt 0
        }
        catch {
            # Module not available
        }
    }
}

BeforeAll {
    # Load the C# assembly
    $BinDir = Join-Path $PSScriptRoot "..\..\bin"
    $PcaiNativeDll = Join-Path $BinDir "PcaiNative.dll"
    $PerformanceDll = Join-Path $BinDir "pcai_core_lib.dll"

    # A missing DLL is not a test failure. This file also runs in `powershell-test`,
    # which builds nothing, so the DLL is legitimately absent there. Throwing in
    # BeforeAll fails every It in the file -- including the ~21 that already carry
    # -Skip: and would have skipped cleanly. Leave the flag false and let the
    # per-It skips do the job they were already written to do.
    if (-not (Test-Path $PcaiNativeDll) -or -not (Test-Path $PerformanceDll)) {
        $script:PerformanceAvailable = $false
        return
    }

    # Load the assembly
    try {
        Add-Type -Path $PcaiNativeDll -ErrorAction Stop
    }
    catch [System.Reflection.ReflectionTypeLoadException] {
        # Already loaded, ignore
    }

    # Set availability flag for tests
    $script:PerformanceAvailable = $false
    try {
        $version = [PcaiNative.PerformanceModule]::GetVersion()
        $script:PerformanceAvailable = $version -gt 0
    }
    catch {
        # Module not available
    }
}

Describe "Performance Module - DLL Loading" -Tag "FFI", "Performance", "Unit" -Skip:(-not $script:PerformanceDllPresent) {
    It "Should have pcai_core_lib.dll in bin directory" {
        $dll = Join-Path $PSScriptRoot "..\..\bin\pcai_core_lib.dll"
        $dll | Should -Exist
    }

    It "Should load without errors" {
        $script:PerformanceAvailable | Should -BeTrue
    }

    It "Should return valid version number" {
        $version = [PcaiNative.PerformanceModule]::GetVersion()
        $version | Should -BeGreaterThan 0
        # Version 1.0.0 = 0x010000
        $version | Should -Be 0x010000
    }

    It "Should pass magic number test" {
        $result = [PcaiNative.PerformanceModule]::Test()
        $result | Should -BeTrue
    }
}

Describe "Performance Module - Disk Usage" -Tag "FFI", "Performance", "DiskUsage" {
    BeforeAll {
        # Create a temp directory with some files for testing
        $script:TestDir = Join-Path $env:TEMP "PcaiPerfTest_$(Get-Random)"
        New-Item -ItemType Directory -Path $script:TestDir -Force | Out-Null

        # Create subdirectories with files
        $subDir1 = Join-Path $script:TestDir "subdir1"
        $subDir2 = Join-Path $script:TestDir "subdir2"
        New-Item -ItemType Directory -Path $subDir1, $subDir2 -Force | Out-Null

        # Create test files
        Set-Content -Path (Join-Path $subDir1 "file1.txt") -Value ("A" * 1000)
        Set-Content -Path (Join-Path $subDir1 "file2.txt") -Value ("B" * 2000)
        Set-Content -Path (Join-Path $subDir2 "file3.txt") -Value ("C" * 3000)
    }

    AfterAll {
        if (Test-Path $script:TestDir) {
            Remove-Item -Path $script:TestDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Should return disk usage statistics" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetDiskUsage($script:TestDir, 10)
        $stats.IsSuccess | Should -BeTrue
        $stats.TotalFiles | Should -BeGreaterOrEqual 3
        $stats.TotalSizeBytes | Should -BeGreaterThan 0
        $stats.TotalDirs | Should -BeGreaterOrEqual 2
    }

    It "Should return disk usage as JSON" -Skip:(-not $script:PerformanceAvailable) {
        $json = [PcaiNative.PerformanceModule]::GetDiskUsageJson($script:TestDir, 10)
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.total_files | Should -BeGreaterOrEqual 3
        $data.top_entries | Should -Not -BeNullOrEmpty
    }

    It "Should handle non-existent path" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetDiskUsage("C:\NonExistent\Path\XYZ123", 10)
        $stats.IsSuccess | Should -BeFalse
        $stats.Status | Should -Be ([PcaiNative.PcaiStatus]::IoError)
    }
}

Describe "Performance Module - Process Stats" -Tag "FFI", "Performance", "Process" {
    It "Should return process statistics" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetProcessStats()
        $stats.IsSuccess | Should -BeTrue
        $stats.TotalProcesses | Should -BeGreaterThan 0
        $stats.SystemMemoryTotalBytes | Should -BeGreaterThan 0
    }

    It "Should return memory usage percentage" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetProcessStats()
        $memPercent = $stats.MemoryUsagePercent
        $memPercent | Should -BeGreaterThan 0
        $memPercent | Should -BeLessThan 100
    }

    It "Should return top processes as JSON sorted by memory" -Skip:(-not $script:PerformanceAvailable) {
        $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson(10, "memory")
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.processes | Should -Not -BeNullOrEmpty
        $data.processes.Count | Should -BeLessOrEqual 10
        $data.sort_by | Should -Be "memory"
    }

    It "Should return top processes as JSON sorted by CPU" -Skip:(-not $script:PerformanceAvailable) {
        $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson(5, "cpu")
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.sort_by | Should -Be "cpu"
    }

    It "Should include process details in JSON" -Skip:(-not $script:PerformanceAvailable) {
        $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson(5, "memory")
        $data = $json | ConvertFrom-Json

        $proc = $data.processes[0]
        $proc.pid | Should -BeGreaterThan 0
        $proc.name | Should -Not -BeNullOrEmpty
        $proc.memory_bytes | Should -BeGreaterOrEqual 0
        $proc.memory_formatted | Should -Not -BeNullOrEmpty
    }
}

Describe "Performance Module - Memory Stats" -Tag "FFI", "Performance", "Memory" {
    It "Should return memory statistics" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetMemoryStats()
        $stats.IsSuccess | Should -BeTrue
        $stats.TotalMemoryBytes | Should -BeGreaterThan 0
        $stats.UsedMemoryBytes | Should -BeGreaterThan 0
        $stats.AvailableMemoryBytes | Should -BeGreaterThan 0
    }

    It "Should return valid memory usage percentage" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetMemoryStats()
        $memPercent = $stats.MemoryUsagePercent
        $memPercent | Should -BeGreaterThan 0
        $memPercent | Should -BeLessThan 100
    }

    It "Should report swap information" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetMemoryStats()
        # Swap might be 0 if not configured, but total should be >= 0
        $stats.TotalSwapBytes | Should -BeGreaterOrEqual 0
    }

    It "Should return memory stats as JSON" -Skip:(-not $script:PerformanceAvailable) {
        $json = [PcaiNative.PerformanceModule]::GetMemoryStatsJson()
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.total_memory_bytes | Should -BeGreaterThan 0
        $data.used_memory_bytes | Should -BeGreaterThan 0
        $data.memory_usage_percent | Should -BeGreaterThan 0
    }

    It "Should calculate memory usage correctly" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetMemoryStats()

        # UsedMemory + AvailableMemory should be close to TotalMemory
        $calculatedTotal = $stats.UsedMemoryBytes + $stats.AvailableMemoryBytes
        # Allow some variance for cache/buffers
        $ratio = [Math]::Abs($calculatedTotal - $stats.TotalMemoryBytes) / $stats.TotalMemoryBytes
        $ratio | Should -BeLessThan 0.1  # Within 10%
    }
}

Describe "Performance Module - Utility Functions" -Tag "FFI", "Performance", "Utility" -Skip:(-not $script:PerformanceDllPresent) {
    It "Should format bytes correctly - Bytes" {
        $result = [PcaiNative.PerformanceModule]::FormatBytes(512)
        $result | Should -Be "512 B"
    }

    It "Should format bytes correctly - Kilobytes" {
        $result = [PcaiNative.PerformanceModule]::FormatBytes(1024)
        $result | Should -Be "1.00 KB"
    }

    It "Should format bytes correctly - Megabytes" {
        $result = [PcaiNative.PerformanceModule]::FormatBytes(1048576)
        $result | Should -Be "1.00 MB"
    }

    It "Should format bytes correctly - Gigabytes" {
        $result = [PcaiNative.PerformanceModule]::FormatBytes(1073741824)
        $result | Should -Be "1.00 GB"
    }

    It "Should format bytes correctly - Terabytes" {
        $result = [PcaiNative.PerformanceModule]::FormatBytes(1099511627776)
        $result | Should -Be "1.00 TB"
    }
}

Describe "Performance Module - Error Handling" -Tag "FFI", "Performance", "ErrorHandling" {
    It "Should handle null path gracefully for disk usage" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetDiskUsage($null, 10)
        $stats.IsSuccess | Should -BeFalse
    }

    It "Should return valid status codes" -Skip:(-not $script:PerformanceAvailable) {
        $stats = [PcaiNative.PerformanceModule]::GetDiskUsage("C:\NonExistent\Path", 10)
        [Enum]::IsDefined([PcaiNative.PcaiStatus], $stats.Status) | Should -BeTrue
    }
}

Describe "Performance Module - Integration with Core" -Tag "FFI", "Performance", "Integration" {
    It "Should work with core module loaded" -Skip:(-not $script:PerformanceAvailable) {
        # Verify core is also available
        $coreAvailable = [PcaiNative.PcaiCore]::IsAvailable
        $coreAvailable | Should -BeTrue

        # Performance module should work when core is available
        $stats = [PcaiNative.PerformanceModule]::GetProcessStats()
        $stats.IsSuccess | Should -BeTrue
    }
}
