<#
.SYNOPSIS
    Unit tests for PC-AI.Performance module

.DESCRIPTION
    Tests disk space monitoring, process performance tracking, system resource watching, and disk optimization
#>

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Performance\PC-AI.Performance.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop
}

Describe "Get-DiskSpace" -Tag 'Unit', 'Performance', 'Fast', 'Portable' {
    Context "When checking disk space with healthy drives" {
        BeforeAll {
            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'C:'
                        DriveType = 3  # Fixed
                        Size = 500GB
                        FreeSpace = 350GB
                        VolumeName = 'System'
                        FileSystem = 'NTFS'
                    }
                    [PSCustomObject]@{
                        DeviceID = 'D:'
                        DriveType = 3  # Fixed
                        Size = 2000GB
                        FreeSpace = 1500GB
                        VolumeName = 'Data'
                        FileSystem = 'NTFS'
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should return disk space information" {
            $result = Get-DiskSpace
            $result | Should -Not -BeNullOrEmpty
            $result.Count | Should -BeGreaterThan 0
        }

        It "Should calculate total size correctly" {
            $result = Get-DiskSpace | Where-Object { $_.DriveLetter -eq 'C' }
            $result.TotalSize | Should -BeGreaterThan 0
        }

        It "Should calculate percentage free" {
            $result = Get-DiskSpace | Where-Object { $_.DriveLetter -eq 'C' }
            $result.FreePercent | Should -BeGreaterThan 0
            $result.FreePercent | Should -BeLessOrEqual 100
        }

        It "Should filter by drive type" {
            $result = Get-DiskSpace
            $result | ForEach-Object {
                $_.DriveType | Should -Be 'Fixed'
            }
        }
    }

    Context "When disk space is low" {
        BeforeAll {
            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'C:'
                        DriveType = 3  # Fixed
                        Size = 500GB
                        FreeSpace = 40GB
                        VolumeName = 'System'
                        FileSystem = 'NTFS'
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should detect low disk space" {
            $result = Get-DiskSpace
            $result.FreePercent | Should -BeLessThan 15
        }
    }

    Context "When disk space is critical" {
        BeforeAll {
            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'C:'
                        DriveType = 3  # Fixed
                        Size = 500GB
                        FreeSpace = 15GB
                        VolumeName = 'System'
                        FileSystem = 'NTFS'
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should detect critical disk space" {
            $result = Get-DiskSpace
            $result.FreePercent | Should -BeLessThan 5
        }
    }

    Context "When filtering by drive letter" {
        BeforeAll {
            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'C:'
                        DriveType = 3  # Fixed
                        Size = 500GB
                        FreeSpace = 350GB
                        VolumeName = 'System'
                        FileSystem = 'NTFS'
                    }
                    [PSCustomObject]@{
                        DeviceID = 'D:'
                        DriveType = 3  # Fixed
                        Size = 2000GB
                        FreeSpace = 1500GB
                        VolumeName = 'Data'
                        FileSystem = 'NTFS'
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should filter by specific drive" {
            $result = Get-DiskSpace -DriveLetter 'C'
            $result.DriveLetter | Should -Be 'C'
        }
    }
}

Describe "Get-ProcessPerformance" -Tag 'Unit', 'Performance', 'Fast', 'Portable' {
    Context "When monitoring process performance" {
        BeforeAll {
            Mock Get-Process {
                [PSCustomObject]@{
                    ProcessName = "chrome"
                    Id = 1234
                    CPU = 45.5
                    TotalProcessorTime = [TimeSpan]::FromSeconds(45.5)
                    StartTime = (Get-Date).AddMinutes(-10)
                    WorkingSet64 = 500MB
                    Threads = @(1..15)
                    HandleCount = 1500
                    Path = "C:\Program Files\Chrome\chrome.exe"
                    MainModule = $null
                    PriorityClass = 'Normal'
                }
                [PSCustomObject]@{
                    ProcessName = "Code"
                    Id = 5678
                    CPU = 12.3
                    TotalProcessorTime = [TimeSpan]::FromSeconds(12.3)
                    StartTime = (Get-Date).AddMinutes(-5)
                    WorkingSet64 = 300MB
                    Threads = @(1..10)
                    HandleCount = 800
                    Path = "C:\Program Files\VS Code\Code.exe"
                    MainModule = $null
                    PriorityClass = 'Normal'
                }
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory = 16GB
                        NumberOfLogicalProcessors = 8
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should return process information with Both structure" {
            $result = Get-ProcessPerformance
            $result | Should -Not -BeNullOrEmpty
            $result.TopByCPU | Should -Not -BeNullOrEmpty
            $result.TopByMemory | Should -Not -BeNullOrEmpty
            $result.Summary | Should -Not -BeNullOrEmpty
        }

        It "Should include CPU usage in TopByCPU" {
            $result = Get-ProcessPerformance
            $result.TopByCPU[0].CpuPercent | Should -BeGreaterOrEqual 0
        }

        It "Should include memory usage in TopByMemory" {
            $result = Get-ProcessPerformance
            $result.TopByMemory[0].MemoryMB | Should -BeGreaterThan 0
        }

        It "Should sort by CPU usage in TopByCPU" {
            $result = Get-ProcessPerformance
            $result.TopByCPU[0].CpuPercent | Should -BeGreaterOrEqual $result.TopByCPU[1].CpuPercent
        }
    }

    Context "When sorting by CPU only" {
        BeforeAll {
            Mock Get-Process {
                [PSCustomObject]@{
                    ProcessName = "chrome"
                    Id = 1234
                    CPU = 45.5
                    TotalProcessorTime = [TimeSpan]::FromSeconds(45.5)
                    StartTime = (Get-Date).AddMinutes(-10)
                    WorkingSet64 = 500MB
                    Threads = @(1..15)
                    HandleCount = 1500
                    Path = "C:\Program Files\Chrome\chrome.exe"
                    MainModule = $null
                    PriorityClass = 'Normal'
                }
                [PSCustomObject]@{
                    ProcessName = "Code"
                    Id = 5678
                    CPU = 12.3
                    TotalProcessorTime = [TimeSpan]::FromSeconds(12.3)
                    StartTime = (Get-Date).AddMinutes(-5)
                    WorkingSet64 = 300MB
                    Threads = @(1..10)
                    HandleCount = 800
                    Path = "C:\Program Files\VS Code\Code.exe"
                    MainModule = $null
                    PriorityClass = 'Normal'
                }
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory = 16GB
                        NumberOfLogicalProcessors = 8
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should return flat array when SortBy is CPU" {
            $result = Get-ProcessPerformance -SortBy CPU
            $result | Should -BeOfType [System.Object]
            $result[0].ProcessName | Should -Be "chrome"
        }
    }

    Context "When limiting top processes" {
        BeforeAll {
            Mock Get-Process {
                1..10 | ForEach-Object {
                    [PSCustomObject]@{
                        ProcessName = "Process$_"
                        Id = $_
                        CPU = (100 - $_ * 5)
                        TotalProcessorTime = [TimeSpan]::FromSeconds((100 - $_ * 5))
                        StartTime = (Get-Date).AddMinutes(-10)
                        WorkingSet64 = (500MB - $_ * 10MB)
                        Threads = @(1..5)
                        HandleCount = 500
                        Path = "C:\Windows\System32\Process$_.exe"
                        MainModule = $null
                        PriorityClass = 'Normal'
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory = 16GB
                        NumberOfLogicalProcessors = 8
                    }
                }
            } -ModuleName PC-AI.Performance
        }

        It "Should limit results to top N processes when using CPU sort" {
            $result = Get-ProcessPerformance -Top 5 -SortBy CPU
            $result.Count | Should -Be 5
        }
    }

    Context "When Get-Process fails" {
        BeforeAll {
            Mock Get-Process { throw "Process not found" } -ModuleName PC-AI.Performance
        }

        It "Should handle process retrieval errors" {
            { Get-ProcessPerformance -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe "Watch-SystemResources" -Tag 'Unit', 'Performance', 'Slow', 'Portable' {
    Context "When monitoring system resources" {
        BeforeAll {
            # Mock Get-Counter for CPU counter
            Mock Get-Counter {
                param($Counter)

                $samples = @()

                # CPU counter
                if ($Counter -like '*Processor*') {
                    $samples += [PSCustomObject]@{
                        Path = '\Processor(_Total)\% Processor Time'
                        CookedValue = 35.5
                    }
                }

                # Network counters
                if ($Counter -like '*Network Interface*') {
                    $samples += [PSCustomObject]@{
                        Path = '\Network Interface(Ethernet)\Bytes Received/sec'
                        InstanceName = 'Ethernet'
                        CookedValue = 1048576
                    }
                    $samples += [PSCustomObject]@{
                        Path = '\Network Interface(Ethernet)\Bytes Sent/sec'
                        InstanceName = 'Ethernet'
                        CookedValue = 524288
                    }
                }

                [PSCustomObject]@{
                    CounterSamples = $samples
                }
            } -ModuleName PC-AI.Performance

            # Mock Get-CimInstance for memory and system info
            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory = 16GB
                        NumberOfLogicalProcessors = 8
                    }
                }
                elseif ($ClassName -eq 'Win32_OperatingSystem') {
                    [PSCustomObject]@{
                        TotalVisibleMemorySize = 16GB / 1KB
                        FreePhysicalMemory = 8GB / 1KB
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-DiskIOCounters {
                [PSCustomObject]@{
                    ReadBytesPerSec = 10MB
                    WriteBytesPerSec = 5MB
                }
            } -ModuleName PC-AI.Performance

            Mock Start-Sleep {} -ModuleName PC-AI.Performance
        }

        It "Should collect CPU metrics" {
            Watch-SystemResources -RefreshInterval 1 -Duration 1 -OutputMode Object

            Should -Invoke Get-Counter -ModuleName PC-AI.Performance -ParameterFilter {
                $Counter -like '*Processor*'
            }
        }

        It "Should collect memory metrics via CIM" {
            Watch-SystemResources -RefreshInterval 1 -Duration 1 -OutputMode Object

            Should -Invoke Get-CimInstance -ModuleName PC-AI.Performance -ParameterFilter {
                $ClassName -eq 'Win32_OperatingSystem'
            }
        }

        It "Should support custom duration" {
            $result = Watch-SystemResources -RefreshInterval 1 -Duration 2 -OutputMode Object

            $result.Count | Should -BeGreaterOrEqual 1
        }

        It "Should support custom refresh interval" {
            Watch-SystemResources -RefreshInterval 5 -Duration 1 -OutputMode Object

            Should -Invoke Start-Sleep -ModuleName PC-AI.Performance -ParameterFilter {
                $Seconds -eq 5
            }
        }
    }

    Context "When performance counters are unavailable" {
        BeforeAll {
            Mock Get-Counter { throw "Performance counter not found" } -ModuleName PC-AI.Performance
        }

        It "Should handle counter errors" {
            { Watch-SystemResources -Iterations 1 -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe "Optimize-Disks" -Tag 'Unit', 'Performance', 'Slow', 'RequiresAdmin', 'Portable' {
    Context "When optimizing SSD drives" {
        BeforeAll {
            Mock Test-IsAdmin { $true } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName, $Filter)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'C:'
                        DriveType = 3  # Fixed disk
                        Size = 500GB
                        FreeSpace = 350GB
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-DriveMediaType {
                param($DriveLetter)
                'SSD'
            } -ModuleName PC-AI.Performance

            Mock Get-Volume {
                [PSCustomObject]@{
                    DriveLetter = 'C'
                    FileSystem = 'NTFS'
                    DriveType = 'Fixed'
                }
            } -ModuleName PC-AI.Performance

            Mock Optimize-Volume {} -ModuleName PC-AI.Performance
        }

        It "Should run TRIM on SSD" {
            Optimize-Disks -DriveLetter 'C' -Force

            Should -Invoke Optimize-Volume -ModuleName PC-AI.Performance -ParameterFilter {
                $ReTrim -eq $true
            }
        }

        It "Should not defragment SSD" {
            Optimize-Disks -DriveLetter 'C' -Force

            Should -Invoke Optimize-Volume -ModuleName PC-AI.Performance -Times 0 -ParameterFilter {
                $Defrag -eq $true
            }
        }
    }

    Context "When optimizing HDD drives" {
        BeforeAll {
            Mock Test-IsAdmin { $true } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName, $Filter)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    [PSCustomObject]@{
                        DeviceID = 'D:'
                        DriveType = 3  # Fixed disk
                        Size = 2000GB
                        FreeSpace = 1500GB
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-DriveMediaType {
                param($DriveLetter)
                'HDD'
            } -ModuleName PC-AI.Performance

            Mock Get-Volume {
                [PSCustomObject]@{
                    DriveLetter = 'D'
                    FileSystem = 'NTFS'
                    DriveType = 'Fixed'
                }
            } -ModuleName PC-AI.Performance

            Mock Optimize-Volume {} -ModuleName PC-AI.Performance
        }

        It "Should defragment HDD" {
            Optimize-Disks -DriveLetter 'D' -Force

            Should -Invoke Optimize-Volume -ModuleName PC-AI.Performance -ParameterFilter {
                $Defrag -eq $true
            }
        }

        It "Should not TRIM HDD" {
            Optimize-Disks -DriveLetter 'D' -Force

            Should -Invoke Optimize-Volume -ModuleName PC-AI.Performance -Times 0 -ParameterFilter {
                $ReTrim -eq $true
            }
        }
    }

    Context "When not running as Administrator" {
        BeforeAll {
            Mock Test-IsAdmin { $false } -ModuleName PC-AI.Performance
        }

        It "Should require Administrator privileges" {
            { Optimize-Disks -DriveLetter 'C' -ErrorAction Stop } | Should -Throw
        }
    }

    Context "When AllDrives switch is used" {
        BeforeAll {
            Mock Test-IsAdmin { $true } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName, $Filter)
                if ($ClassName -eq 'Win32_LogicalDisk') {
                    @(
                        [PSCustomObject]@{
                            DeviceID = 'C:'
                            DriveType = 3  # Fixed disk
                            Size = 500GB
                            FreeSpace = 350GB
                        }
                        [PSCustomObject]@{
                            DeviceID = 'D:'
                            DriveType = 3  # Fixed disk
                            Size = 2000GB
                            FreeSpace = 1500GB
                        }
                    )
                }
            } -ModuleName PC-AI.Performance

            Mock Get-DriveMediaType {
                param($DriveLetter)
                'SSD'
            } -ModuleName PC-AI.Performance

            Mock Get-Volume {
                @(
                    [PSCustomObject]@{ DriveLetter = 'C'; FileSystem = 'NTFS'; DriveType = 'Fixed' }
                    [PSCustomObject]@{ DriveLetter = 'D'; FileSystem = 'NTFS'; DriveType = 'Fixed' }
                )
            } -ModuleName PC-AI.Performance

            Mock Optimize-Volume {} -ModuleName PC-AI.Performance
        }

        It "Should optimize all fixed drives" {
            Optimize-Disks -Force

            Should -Invoke Optimize-Volume -ModuleName PC-AI.Performance -Times 2
        }
    }
}

AfterAll {
    Remove-Module PC-AI.Performance -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
