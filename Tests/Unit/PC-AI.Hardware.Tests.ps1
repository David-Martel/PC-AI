<#
.SYNOPSIS
    Unit tests for PC-AI.Hardware module

.DESCRIPTION
    Tests device error detection, disk health, USB status, network adapters, and system events
#>

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Hardware\PC-AI.Hardware.psd1'
    # Evict any copy already loaded by an earlier suite before importing.
    # `Import-Module -Force` re-imports, but it does NOT remove a copy that
    # was loaded from a different path, so two modules of the same name can
    # coexist. Pester then refuses to mock into either -- "Multiple script or
    # manifest modules named 'X' are currently loaded" -- and every mocked
    # call falls through to the real cmdlet. That is why these files pass in
    # isolation and fail in a full run.
    Get-Module 'PC-AI.Hardware' -All | Remove-Module -Force -ErrorAction SilentlyContinue
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop
}

Describe "Get-DeviceErrors" -Tag 'Unit', 'Hardware', 'Fast', 'Portable' {
    Context "When devices have errors" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockDevicesWithErrors } -ModuleName PC-AI.Hardware
        }

        It "Should return devices with non-zero error codes" {
            $result = Get-DeviceErrors
            $result | Should -Not -BeNullOrEmpty
            $result.Count | Should -BeGreaterThan 0
        }

        It "Should include device name and error code" {
            $result = Get-DeviceErrors
            $result[0].Name | Should -Not -BeNullOrEmpty
            $result[0].ErrorCode | Should -BeGreaterThan 0
        }

        It "Should include error description" {
            $result = Get-DeviceErrors
            $result[0].PSObject.Properties.Name | Should -Contain 'ErrorDescription'
        }

        It "Should filter out devices with error code 0" {
            $result = Get-DeviceErrors
            $result | Where-Object { $_.ErrorCode -eq 0 } | Should -BeNullOrEmpty
        }
    }

    Context "When no devices have errors" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockDevicesHealthy } -ModuleName PC-AI.Hardware
        }

        It "Should return empty result" {
            $result = Get-DeviceErrors
            $result | Where-Object { $_.ErrorCode -ne 0 } | Should -BeNullOrEmpty
        }
    }

    Context "When Get-CimInstance fails" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { throw "Access denied" } -ModuleName PC-AI.Hardware
        }

        It "Should handle errors gracefully" {
            { Get-DeviceErrors -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe "Get-DiskHealth" -Tag 'Unit', 'Hardware', 'Fast', 'Portable' {
    Context "When all disks are healthy" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            # Suppress native DLL path so CIM fallback is always exercised
            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-CimInstance {
                @(
                    [PSCustomObject]@{
                        Model = "Samsung SSD 980 PRO 1TB"
                        Status = "OK"
                        MediaType = "Fixed hard disk media"
                        InterfaceType = "NVMe"
                        Size = 1000GB * 1GB
                        Partitions = 3
                        Index = 0
                        DeviceID = "\\.\PHYSICALDRIVE0"
                        SerialNumber = "S5GXNX0T123456"
                    }
                    [PSCustomObject]@{
                        Model = "WDC WD40EZRZ-00GXCB0"
                        Status = "OK"
                        MediaType = "Fixed hard disk media"
                        InterfaceType = "SATA"
                        Size = 4000GB * 1GB
                        Partitions = 1
                        Index = 1
                        DeviceID = "\\.\PHYSICALDRIVE1"
                        SerialNumber = "WD-WCC7K1234567"
                    }
                )
            } -ModuleName PC-AI.Hardware
        }

        It "Should return disk status information" {
            $result = Get-DiskHealth
            $result | Should -Not -BeNullOrEmpty
            $result.Count | Should -BeGreaterThan 0
        }

        It "Should include disk properties" {
            $result = Get-DiskHealth
            $result[0].PSObject.Properties.Name | Should -Contain 'Model'
            $result[0].PSObject.Properties.Name | Should -Contain 'Status'
            $result[0].PSObject.Properties.Name | Should -Contain 'Severity'
        }

        It "Should not contain 'Bad' status" {
            $result = Get-DiskHealth
            $result | Where-Object { $_.Status -match 'Bad' } | Should -BeNullOrEmpty
        }

        It "Should not contain 'Pred Fail' status" {
            $result = Get-DiskHealth
            $result | Where-Object { $_.Status -match 'Pred Fail' } | Should -BeNullOrEmpty
        }
    }

    Context "When disks have warnings" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-CimInstance {
                @(
                    [PSCustomObject]@{
                        Model = "WDC WD40EZRZ-00GXCB0"
                        Status = "Pred Fail"
                        MediaType = "Fixed hard disk media"
                        InterfaceType = "SATA"
                        Size = 4000GB * 1GB
                        Partitions = 1
                        Index = 0
                        DeviceID = "\\.\PHYSICALDRIVE0"
                        SerialNumber = "WD-WCC7K1234567"
                    }
                )
            } -ModuleName PC-AI.Hardware
        }

        It "Should contain 'Pred Fail' status" {
            $result = Get-DiskHealth
            $result | Where-Object { $_.Status -match 'Pred Fail' } | Should -Not -BeNullOrEmpty
        }

        It "Should have Critical severity" {
            $result = Get-DiskHealth
            $result[0].Severity | Should -Be 'Critical'
        }
    }

    Context "When disks have failures" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-CimInstance {
                @(
                    [PSCustomObject]@{
                        Model = "WDC WD40EZRZ-00GXCB0"
                        Status = "Bad"
                        MediaType = "Fixed hard disk media"
                        InterfaceType = "SATA"
                        Size = 4000GB * 1GB
                        Partitions = 1
                        Index = 0
                        DeviceID = "\\.\PHYSICALDRIVE0"
                        SerialNumber = "WD-WCC7K1234567"
                    }
                )
            } -ModuleName PC-AI.Hardware
        }

        It "Should detect failed disk status" {
            $result = Get-DiskHealth
            $result | Where-Object { $_.Status -eq 'Bad' } | Should -Not -BeNullOrEmpty
        }
    }

    Context "When Get-CimInstance fails" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-CimInstance { throw "Access denied" } -ModuleName PC-AI.Hardware
        }

        It "Should handle errors gracefully" {
            { Get-DiskHealth -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe "Get-UsbStatus" -Tag 'Unit', 'Hardware', 'Fast', 'Portable' {
    Context "When USB devices are present" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance {
                @(
                    New-MockPnPEntity -Name "USB Mass Storage Device" -DeviceID "USB\VID_0781&PID_5567" `
                        -Status "OK" -ConfigManagerErrorCode 0
                    New-MockPnPEntity -Name "USB Root Hub (USB 3.0)" -DeviceID "USB\ROOT_HUB30\1234" `
                        -Status "OK" -ConfigManagerErrorCode 0
                )
            } -ModuleName PC-AI.Hardware
        }

        It "Should return USB devices" {
            $result = Get-UsbStatus
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should filter USB devices by DeviceID pattern" {
            $result = Get-UsbStatus
            $result.DeviceID | Should -Match '^USB\\'
        }
    }

    Context "When USB controllers have errors" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance {
                @(
                    New-MockPnPEntity -Name "Intel(R) USB 3.0 eXtensible Host Controller" `
                        -DeviceID "PCI\VEN_8086&DEV_9CB1" `
                        -Status "Error" -ConfigManagerErrorCode 10
                )
            } -ModuleName PC-AI.Hardware
        }

        It "Should detect controller errors" {
            $result = Get-UsbStatus
            $result | Where-Object { $_.ErrorCode -ne 0 } | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Get-NetworkAdapters" -Tag 'Unit', 'Hardware', 'Fast', 'Portable' {
    Context "When physical adapters are present" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockNetworkAdapters } -ModuleName PC-AI.Hardware
        }

        It "Should return network adapters" {
            $result = Get-NetworkAdapters
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should filter physical adapters" {
            $result = Get-NetworkAdapters
            $result.PhysicalAdapter | Should -Be $true
        }

        It "Should include MAC address" {
            $result = Get-NetworkAdapters
            $result[0].MACAddress | Should -Not -BeNullOrEmpty
        }
    }

    Context "When IncludeVirtual is specified" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockNetworkAdapters -IncludeVirtual } -ModuleName PC-AI.Hardware
        }

        It "Should return both physical and virtual adapters" {
            $result = Get-NetworkAdapters -IncludeVirtual
            $result | Should -Not -BeNullOrEmpty
            $result.Count | Should -BeGreaterThan 1
        }
    }

    Context "When no adapters are found" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { @() } -ModuleName PC-AI.Hardware
        }

        It "Should return empty result" {
            $result = Get-NetworkAdapters
            $result | Should -BeNullOrEmpty
        }
    }
}

Describe "Get-SystemEvents" -Tag 'Unit', 'Hardware', 'Slow', 'Portable' {
    Context "When disk and USB errors exist" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-WinEvent { Get-MockDiskUsbEvents -ErrorType Mixed } -ModuleName PC-AI.Hardware
        }

        It "Should return system events" {
            $result = Get-SystemEvents -Days 3
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should include disk errors" {
            $result = Get-SystemEvents -Days 3
            $result | Where-Object { $_.ProviderName -eq 'disk' } | Should -Not -BeNullOrEmpty
        }

        It "Should include USB errors" {
            $result = Get-SystemEvents -Days 3
            $result | Where-Object { $_.ProviderName -match 'USB' } | Should -Not -BeNullOrEmpty
        }

        It "Should include error level" {
            $result = Get-SystemEvents -Days 3
            $result[0].PSObject.Properties.Name | Should -Contain 'Level'
        }

        It "Should include timestamp" {
            $result = Get-SystemEvents -Days 3
            $result[0].TimeCreated | Should -BeOfType [datetime]
        }
    }

    Context "When no errors exist" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-WinEvent { Get-MockDiskUsbEvents -ErrorType None } -ModuleName PC-AI.Hardware
        }

        It "Should return empty result" {
            $result = Get-SystemEvents -Days 3
            $result | Should -BeNullOrEmpty
        }
    }

    Context "When event log is not accessible" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-WinEvent { throw "Access denied" } -ModuleName PC-AI.Hardware
        }

        It "Should handle errors gracefully" {
            { Get-SystemEvents -Days 3 -ErrorAction Stop } | Should -Throw
        }
    }
}

Describe "New-DiagnosticReport" -Tag 'Unit', 'Hardware', 'Integration', 'Portable' {
    BeforeAll {
        # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
        # probe first and only fall back to CIM/Get-WinEvent when it returns
        # nothing. Whether the native path succeeds depends on what other suites
        # happened to load into the session, so mocking only Get-CimInstance made
        # these tests pass alone and fail in a full run -- the mock was bypassed
        # entirely and real device data came through. Force the fallback path.
        Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
        Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

        # Mock all dependent functions
        Mock Get-DeviceErrors { Get-MockDevicesWithErrors } -ModuleName PC-AI.Hardware
        Mock Get-DiskHealth {
            @(
                [PSCustomObject]@{
                    Model = "Samsung SSD 980 PRO 1TB"
                    Status = "OK"
                    MediaType = "Fixed hard disk media"
                    InterfaceType = "NVMe"
                    SizeGB = 1000
                    Partitions = 3
                    Severity = "OK"
                    DeviceID = "\\.\PHYSICALDRIVE0"
                    SerialNumber = "S5GXNX0T123456"
                }
            )
        } -ModuleName PC-AI.Hardware
        Mock Get-SystemEvents { Get-MockDiskUsbEvents -ErrorType Mixed } -ModuleName PC-AI.Hardware
        Mock Get-UsbStatus { @(New-MockPnPEntity -Name "USB Device" -DeviceID "USB\VID_1234") } -ModuleName PC-AI.Hardware
        Mock Get-NetworkAdapters { Get-MockNetworkAdapters } -ModuleName PC-AI.Hardware
    }

    Context "When generating a full report" {
        It "Should create a diagnostic report without errors" {
            { New-DiagnosticReport -OutputPath "$TestDrive\report.txt" } | Should -Not -Throw
        }

        It "Should call all diagnostic functions" {
            New-DiagnosticReport -OutputPath "$TestDrive\report.txt"

            Should -Invoke Get-DeviceErrors -ModuleName PC-AI.Hardware -Times 1
            Should -Invoke Get-DiskHealth -ModuleName PC-AI.Hardware -Times 1
            Should -Invoke Get-SystemEvents -ModuleName PC-AI.Hardware -Times 1
            Should -Invoke Get-UsbStatus -ModuleName PC-AI.Hardware -Times 1
            Should -Invoke Get-NetworkAdapters -ModuleName PC-AI.Hardware -Times 1
        }

        It "Should write output to file" {
            $reportPath = "$TestDrive\report.txt"
            New-DiagnosticReport -OutputPath $reportPath

            # Verify file was created and has content
            Test-Path $reportPath | Should -Be $true
            (Get-Content $reportPath -Raw).Length | Should -BeGreaterThan 0
        }
    }

    Context "When a section fails" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent when it returns
            # nothing. Whether the native path succeeds depends on what other suites
            # happened to load into the session, so mocking only Get-CimInstance made
            # these tests pass alone and fail in a full run -- the mock was bypassed
            # entirely and real device data came through. Force the fallback path.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-DiskHealth { throw "SMART data unavailable" } -ModuleName PC-AI.Hardware
        }

        It "Should continue with other sections" {
            { New-DiagnosticReport -OutputPath "$TestDrive\report.txt" -ErrorAction SilentlyContinue } | Should -Not -Throw
        }
    }
}

AfterAll {
    Remove-Module PC-AI.Hardware -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
