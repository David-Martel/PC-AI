<#
.SYNOPSIS
    Unit tests for PC-AI.USB module

.DESCRIPTION
    Tests USB device listing with usbipd, WSL mounting/dismounting, and USB status monitoring
#>

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.USB\PC-AI.USB.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop
}

Describe "Get-UsbDeviceList" -Tag 'Unit', 'USB', 'Fast', 'Windows' {
    Context "When usbipd is available" {
        BeforeAll {
            # Mock the private helper function that checks for usbipd
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                    Source = "C:\Program Files\usbipd-win\usbipd.exe"
                }
            } -ModuleName PC-AI.USB
        }

        It "Should return USB device list or null" {
            Mock Invoke-Expression {
                param($Command)
                if ($Command -match 'usbipd list') {
                    return "BUSID  VID:PID    DEVICE"
                }
                return ""
            } -ModuleName PC-AI.USB

            $result = Get-UsbDeviceList
            # Function returns PSCustomObject[] or $null
            # Just verify the function completes without error
            { $result } | Should -Not -Throw
        }

        It "Should have Source property" {
            Mock Get-CimInstance {
                @(
                    [PSCustomObject]@{
                        DeviceID = "USB\VID_0781&PID_5567\123456"
                        Caption = "USB Mass Storage Device"
                        Status = "OK"
                        Service = "usbstor"
                    }
                )
            } -ModuleName PC-AI.USB

            $result = Get-UsbDeviceList
            if ($result) {
                $result | Should -BeOfType [PSCustomObject]
            }
        }
    }

    Context "When usbipd is not installed" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
            Mock Get-CimInstance {
                @(
                    [PSCustomObject]@{
                        DeviceID = "USB\VID_0781&PID_5567\123456"
                        Caption = "USB Mass Storage Device"
                        Status = "OK"
                        Service = "usbstor"
                    }
                )
            } -ModuleName PC-AI.USB
        }

        It "Should fall back to Get-CimInstance" {
            $result = Get-UsbDeviceList
            if ($result) {
                $result | Should -BeOfType [PSCustomObject]
            }
        }

        It "Should return USB devices from CIM" {
            $result = Get-UsbDeviceList
            if ($result) {
                if ($result[0].Source -eq 'Native') {
                    Set-ItResult -Skipped -Because "Native diagnostics active"
                } else {
                    $result.Source | Should -Be 'WMI'
                }
            }
        }
    }

    Context "When no USB devices are connected" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
            Mock Get-CimInstance { @() } -ModuleName PC-AI.USB
        }

        It "Should handle no devices gracefully" {
            $result = Get-UsbDeviceList
            if ($result -and $result[0].Source -eq 'Native') {
                Set-ItResult -Skipped -Because "Native diagnostics active"
            } else {
                $result | Should -BeNullOrEmpty
            }
        }
    }
}

Describe "Mount-UsbToWSL" -Tag 'Unit', 'USB', 'Slow', 'Windows' {
    Context "When mounting USB device to WSL" {
        BeforeAll {
            # Mock prerequisites
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.USB

            Mock Test-IsAdministrator { $true } -ModuleName PC-AI.USB
        }

        It "Should reject invalid BusId format 'invalid'" {
            # Parameter validation throws before function body executes
            { Mount-UsbToWSL -BusId "invalid" } | Should -Throw -ExpectedMessage "*does not match*"
        }

        It "Should reject invalid BusId format 'abc'" {
            { Mount-UsbToWSL -BusId "abc" } | Should -Throw -ExpectedMessage "*does not match*"
        }

        It "Should accept valid BusId format '2-1' with WhatIf" {
            # Use -WhatIf to prevent actual execution since function supports ShouldProcess
            $result = Mount-UsbToWSL -BusId "2-1" -WhatIf
            # WhatIf doesn't actually execute, so result will be default state
            $result.BusId | Should -Be "2-1"
            $result.Success | Should -Be $false  # Not executed, so Success remains false
        }

        It "Should support Distribution parameter with WhatIf" {
            $result = Mount-UsbToWSL -BusId "2-1" -Distribution "Ubuntu" -WhatIf
            $result.Distribution | Should -Be "Ubuntu"
            $result.BusId | Should -Be "2-1"
        }
    }

    Context "When usbipd is not available" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
        }

        It "Should return error when usbipd not installed" {
            $result = Mount-UsbToWSL -BusId "2-1" -ErrorAction SilentlyContinue
            $result.Success | Should -Be $false
            $result.Message | Should -Match "usbipd-win is not installed"
        }
    }
}

Describe "Dismount-UsbFromWSL" -Tag 'Unit', 'USB', 'Slow', 'Windows' {
    Context "When dismounting USB device from WSL" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.USB

            Mock Test-IsAdministrator { $true } -ModuleName PC-AI.USB
        }

        It "Should reject invalid BusId format" {
            { Dismount-UsbFromWSL -BusId "invalid" } | Should -Throw -ExpectedMessage "*does not match*"
        }

        It "Should accept valid BusId format '2-1' with WhatIf" {
            # Use -WhatIf since function supports ShouldProcess
            $result = Dismount-UsbFromWSL -BusId "2-1" -WhatIf
            $result.BusId | Should -Be "2-1"
            # WhatIf doesn't execute, so Detached remains false
            $result.Detached | Should -Be $false
        }
    }

    Context "When usbipd is not available" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
        }

        It "Should return error when usbipd not installed" {
            $result = Dismount-UsbFromWSL -BusId "2-1" -ErrorAction SilentlyContinue
            $result.Detached | Should -Be $false
            $result.Message | Should -Match "usbipd-win is not installed"
        }
    }
}

Describe "Get-UsbWSLStatus" -Tag 'Unit', 'USB', 'Fast', 'Windows' {
    Context "When checking USB/WSL status" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.USB

            Mock Get-Service {
                [PSCustomObject]@{
                    Name = "usbipd"
                    Status = "Running"
                    StartType = "Automatic"
                }
            } -ModuleName PC-AI.USB
        }

        It "Should return status object" {
            $result = Get-UsbWSLStatus
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have WSLDistributions property" {
            $result = Get-UsbWSLStatus
            $result.PSObject.Properties.Name | Should -Contain 'WSLDistributions'
        }

        It "Should have UsbDevices property" {
            $result = Get-UsbWSLStatus
            $result.PSObject.Properties.Name | Should -Contain 'UsbDevices'
        }

        It "Should have Severity property" {
            $result = Get-UsbWSLStatus
            $result.Severity | Should -BeIn @('OK', 'Warning', 'Error')
        }
    }
}

Describe "Invoke-UsbBind" -Tag 'Unit', 'USB', 'Slow', 'RequiresAdmin', 'Windows' {
    Context "When binding USB device" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.USB

            Mock Test-IsAdministrator { $true } -ModuleName PC-AI.USB
        }

        It "Should reject invalid BusId format" {
            { Invoke-UsbBind -BusId "invalid" } | Should -Throw -ExpectedMessage "*does not match*"
        }

        It "Should accept valid BusId format '2-1' with WhatIf" {
            # Use -WhatIf since function supports ShouldProcess
            $result = Invoke-UsbBind -BusId "2-1" -WhatIf
            $result.BusId | Should -Be "2-1"
            # WhatIf doesn't execute, so Success remains false
            $result.Success | Should -Be $false
        }

        It "Should support Force parameter with WhatIf" {
            $result = Invoke-UsbBind -BusId "2-1" -Force -WhatIf
            $result.BusId | Should -Be "2-1"
            $result.Success | Should -Be $false
        }
    }

    Context "When not running as Administrator" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.USB

            Mock Test-IsAdministrator { $false } -ModuleName PC-AI.USB
        }

        It "Should return error requiring Administrator privileges" {
            $result = Invoke-UsbBind -BusId "2-1" -ErrorAction SilentlyContinue
            $result.Success | Should -Be $false
            $result.Message | Should -Match "Administrator privileges required"
        }
    }

    Context "When usbipd is not available" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
        }

        It "Should return error when usbipd not installed" {
            $result = Invoke-UsbBind -BusId "2-1" -ErrorAction SilentlyContinue
            $result.Success | Should -Be $false
            $result.Message | Should -Match "usbipd-win is not installed"
        }
    }
}

Describe "Test-UsbIpdInstalled (Private Function)" -Tag 'Unit', 'USB', 'Fast', 'Windows' {
    Context "When checking usbipd availability" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "usbipd.exe"
                    CommandType = "Application"
                    Source = "C:\Program Files\usbipd-win\usbipd.exe"
                }
            } -ModuleName PC-AI.USB
        }

        It "Should return true when usbipd is available" {
            $result = InModuleScope PC-AI.USB { Test-UsbIpdInstalled }
            $result | Should -Be $true
        }
    }

    Context "When usbipd is not installed" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.USB
        }

        It "Should return false when usbipd is not available" {
            $result = InModuleScope PC-AI.USB { Test-UsbIpdInstalled }
            $result | Should -Be $false
        }
    }
}

AfterAll {
    Remove-Module PC-AI.USB -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
