#Requires -Version 7.0
#Requires -Modules Pester

<#
.SYNOPSIS
    FFI Integration tests for pcai_system.dll

.DESCRIPTION
    Tests the system module for PATH analysis and log searching
    functionality via C# P/Invoke wrapper.
#>

BeforeDiscovery {
    # Load the C# assembly at discovery time for Skip conditions
    $BinDir = Join-Path $PSScriptRoot "..\..\bin"
    $PcaiNativeDll = Join-Path $BinDir "PcaiNative.dll"

    $script:SystemAvailable = $false
    # Presence of the DLL, kept separate from whether it loads. Presence decides
    # whether this suite applies at all; loading is what the DLL Loading tests
    # assert. Collapsing the two would make those tests unable to fail.
    $script:SystemDllPresent = Test-Path $PcaiNativeDll
    if ($script:SystemDllPresent) {
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
            $version = [PcaiNative.SystemModule]::GetVersion()
            $script:SystemAvailable = $version -gt 0
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
    $SystemDll = Join-Path $BinDir "pcai_core_lib.dll"

    # A missing DLL is not a test failure. This file also runs in `powershell-test`,
    # which builds nothing, so the DLL is legitimately absent there. Throwing in
    # BeforeAll fails every It in the file -- including the ~20 that already carry
    # -Skip: and would have skipped cleanly. Leave the flag false and let the
    # per-It skips do the job they were already written to do.
    if (-not (Test-Path $PcaiNativeDll) -or -not (Test-Path $SystemDll)) {
        $script:SystemAvailable = $false
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
    $script:SystemAvailable = $false
    try {
        $version = [PcaiNative.SystemModule]::GetVersion()
        $script:SystemAvailable = $version -gt 0
    }
    catch {
        # Module not available
    }
}

Describe "System Module - DLL Loading" -Tag "FFI", "System", "Unit" -Skip:(-not $script:SystemDllPresent) {
    It "Should have pcai_core_lib.dll in bin directory" {
        $dll = Join-Path $PSScriptRoot "..\..\bin\pcai_core_lib.dll"
        $dll | Should -Exist
    }

    It "Should load without errors" {
        $script:SystemAvailable | Should -BeTrue
    }

    It "Should return valid version number" {
        $version = [PcaiNative.SystemModule]::GetVersion()
        $version | Should -BeGreaterThan 0
        # Version 1.0.0 = 0x010000
        $version | Should -Be 0x010000
    }

    It "Should pass magic number test" {
        $result = [PcaiNative.SystemModule]::Test()
        $result | Should -BeTrue
    }
}

Describe "System Module - PATH Analysis" -Tag "FFI", "System", "PATH" {
    It "Should return PATH analysis statistics" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::AnalyzePath()
        $stats.IsSuccess | Should -BeTrue
        $stats.TotalEntries | Should -BeGreaterThan 0
        $stats.UniqueEntries | Should -BeGreaterThan 0
    }

    It "Should detect unique entries count" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::AnalyzePath()
        # UniqueEntries should be <= TotalEntries
        $stats.UniqueEntries | Should -BeLessOrEqual $stats.TotalEntries
    }

    It "Should return PATH analysis as JSON" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::AnalyzePathJson()
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.total_entries | Should -BeGreaterThan 0
        $data.health_status | Should -Not -BeNullOrEmpty
    }

    It "Should include recommendations in JSON" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::AnalyzePathJson()
        $data = $json | ConvertFrom-Json

        # recommendations should be present (may be empty array, single item, or array)
        # PowerShell converts single-element arrays to scalars
        $null -ne $data.PSObject.Properties['recommendations'] | Should -BeTrue
    }

    It "Should include issues array in JSON" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::AnalyzePathJson()
        $data = $json | ConvertFrom-Json

        # issues should be present (may be empty array, single item, or array)
        # PowerShell converts single-element arrays to scalars
        $null -ne $data.PSObject.Properties['issues'] | Should -BeTrue
    }

    It "Should report health status" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::AnalyzePath()
        $healthStatus = $stats.HealthStatus
        $healthStatus | Should -Match "^(Healthy|MinorIssues|NeedsAttention)$"
    }
}

Describe "System Module - Log Search" -Tag "FFI", "System", "LogSearch" {
    BeforeAll {
        # Create a temp directory with test log files
        $script:TestDir = Join-Path $env:TEMP "PcaiSystemTest_$(Get-Random)"
        New-Item -ItemType Directory -Path $script:TestDir -Force | Out-Null

        # Create test log files
        @"
2024-01-15 10:00:00 INFO Application started
2024-01-15 10:00:01 DEBUG Loading configuration
2024-01-15 10:00:02 ERROR Failed to connect to database
2024-01-15 10:00:03 INFO Retrying connection
2024-01-15 10:00:04 ERROR Connection timeout
2024-01-15 10:00:05 INFO Connection established
"@ | Set-Content -Path (Join-Path $script:TestDir "app.log")

        @"
2024-01-15 11:00:00 INFO Service started
2024-01-15 11:00:01 WARNING Disk space low
2024-01-15 11:00:02 ERROR Out of memory
"@ | Set-Content -Path (Join-Path $script:TestDir "service.log")

        # Create a non-log file (should not be searched by default)
        "This is not a log file" | Set-Content -Path (Join-Path $script:TestDir "readme.txt")
    }

    AfterAll {
        if (Test-Path $script:TestDir) {
            Remove-Item -Path $script:TestDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "Should return log search statistics" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::SearchLogs($script:TestDir, "ERROR", "*.log", $false, 2, 100)
        $stats.IsSuccess | Should -BeTrue
        $stats.FilesSearched | Should -BeGreaterOrEqual 2
        $stats.TotalMatches | Should -BeGreaterThan 0
    }

    It "Should find ERROR entries in log files" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::SearchLogsJson($script:TestDir, "ERROR", "*.log", $false, 2, 100)
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Success"
        $data.total_matches | Should -Be 3  # 2 in app.log + 1 in service.log
    }

    It "Should include context lines" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::SearchLogsJson($script:TestDir, "ERROR", "*.log", $false, 2, 100)
        $data = $json | ConvertFrom-Json

        # First result should have context
        $firstResult = $data.results[0]
        $firstMatch = $firstResult.matches[0]

        # Should have context lines
        ($firstMatch.context_before.Count + $firstMatch.context_after.Count) | Should -BeGreaterThan 0
    }

    It "Should respect case sensitivity" -Skip:(-not $script:SystemAvailable) {
        # Case-insensitive search
        $insensitive = [PcaiNative.SystemModule]::SearchLogsJson($script:TestDir, "error", "*.log", $false, 0, 100)
        $insensitiveData = $insensitive | ConvertFrom-Json

        # Case-sensitive search (should find 0 as "error" != "ERROR")
        $sensitive = [PcaiNative.SystemModule]::SearchLogsJson($script:TestDir, "error", "*.log", $true, 0, 100)
        $sensitiveData = $sensitive | ConvertFrom-Json

        $insensitiveData.total_matches | Should -BeGreaterThan $sensitiveData.total_matches
    }

    It "Should handle non-existent path" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::SearchLogs("C:\NonExistent\Path\XYZ123", "ERROR", "*.log", $false, 2, 100)
        $stats.IsSuccess | Should -BeFalse
    }

    It "Should return JSON even for non-existent path" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::SearchLogsJson("C:\NonExistent\Path\XYZ123", "ERROR", "*.log", $false, 2, 100)
        $json | Should -Not -BeNullOrEmpty

        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Error"
    }

    It "Should respect max matches limit" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::SearchLogsJson($script:TestDir, ".", "*.log", $false, 0, 2)
        $data = $json | ConvertFrom-Json

        $data.total_matches | Should -BeLessOrEqual 2
    }
}

Describe "System Module - Utility Functions" -Tag "FFI", "System", "Utility" -Skip:(-not $script:SystemDllPresent) {
    It "Should format bytes correctly - Bytes" {
        $result = [PcaiNative.SystemModule]::FormatBytes(512)
        $result | Should -Be "512 B"
    }

    It "Should format bytes correctly - Kilobytes" {
        $result = [PcaiNative.SystemModule]::FormatBytes(1024)
        $result | Should -Be "1.00 KB"
    }

    It "Should format bytes correctly - Megabytes" {
        $result = [PcaiNative.SystemModule]::FormatBytes(1048576)
        $result | Should -Be "1.00 MB"
    }

    It "Should format bytes correctly - Gigabytes" {
        $result = [PcaiNative.SystemModule]::FormatBytes(1073741824)
        $result | Should -Be "1.00 GB"
    }

    It "Should format bytes correctly - Terabytes" {
        $result = [PcaiNative.SystemModule]::FormatBytes(1099511627776)
        $result | Should -Be "1.00 TB"
    }
}

Describe "System Module - Error Handling" -Tag "FFI", "System", "ErrorHandling" {
    It "Should handle empty pattern gracefully" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::SearchLogs("C:\Windows", "", "*.log", $false, 2, 100)
        $stats.IsSuccess | Should -BeFalse
    }

    It "Should handle invalid regex gracefully" -Skip:(-not $script:SystemAvailable) {
        $json = [PcaiNative.SystemModule]::SearchLogsJson("C:\Windows", "[invalid(regex", "*.log", $false, 2, 100)
        $data = $json | ConvertFrom-Json
        $data.status | Should -Be "Error"
    }

    It "Should return valid status codes" -Skip:(-not $script:SystemAvailable) {
        $stats = [PcaiNative.SystemModule]::SearchLogs("C:\NonExistent\Path", "test", "*.log", $false, 2, 100)
        [Enum]::IsDefined([PcaiNative.PcaiStatus], $stats.Status) | Should -BeTrue
    }
}

Describe "System Module - Integration with Core" -Tag "FFI", "System", "Integration" {
    It "Should work with core module loaded" -Skip:(-not $script:SystemAvailable) {
        # Verify core is also available
        $coreAvailable = [PcaiNative.PcaiCore]::IsAvailable
        $coreAvailable | Should -BeTrue

        # System module should work when core is available
        $stats = [PcaiNative.SystemModule]::AnalyzePath()
        $stats.IsSuccess | Should -BeTrue
    }

    It "Should coexist with other modules" -Skip:(-not $script:SystemAvailable) {
        # Verify all modules can be used together
        $systemVersion = [PcaiNative.SystemModule]::GetVersion()
        $searchAvailable = [PcaiNative.PcaiSearch]::IsAvailable
        $perfVersion = [PcaiNative.PerformanceModule]::GetVersion()

        $systemVersion | Should -BeGreaterThan 0
        $searchAvailable | Should -BeTrue
        $perfVersion | Should -BeGreaterThan 0
    }
}
