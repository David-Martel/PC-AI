<#
.SYNOPSIS
    Unit tests for PC-AI.Virtualization module

.DESCRIPTION
    Tests WSL status, Hyper-V detection, Docker diagnostics, and virtualization optimization
#>

if (-not (Get-Variable -Name IsAdmin -Scope Script -ErrorAction SilentlyContinue)) {
    $script:IsAdmin = $false
}

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop

    # Helper function to check if running as Administrator
    function Test-IsAdmin {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    $script:IsAdmin = Test-IsAdmin
}

Describe "Get-WSLStatus" -Tag 'Unit', 'Virtualization', 'Fast', 'Portable' {
    Context "When WSL is installed and running" {
        BeforeAll {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "wsl.exe"
                    CommandType = "Application"
                    Source = "C:\Windows\System32\wsl.exe"
                }
            } -ModuleName PC-AI.Virtualization
        }

        It "Should return a PSCustomObject" {
            $result = Get-WSLStatus
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have Installed property" {
            $result = Get-WSLStatus
            $result.PSObject.Properties.Name | Should -Contain 'Installed'
        }

        It "Should have Distributions property" {
            $result = Get-WSLStatus
            $result.PSObject.Properties.Name | Should -Contain 'Distributions'
        }

        It "Should have Severity property" {
            $result = Get-WSLStatus
            $result.Severity | Should -BeIn @('OK', 'Warning', 'Error')
        }

        It "Should have WSLConfigPath property" {
            $result = Get-WSLStatus
            $result.PSObject.Properties.Name | Should -Contain 'WSLConfigPath'
        }
    }

    Context "When WSL is not installed" {
        BeforeAll {
            Mock Get-Command { $null } -ModuleName PC-AI.Virtualization
        }

        It "Should return result with Installed = false" {
            $result = Get-WSLStatus
            $result.Installed | Should -Be $false
        }
    }
}

Describe "Get-HyperVStatus" -Tag 'Unit', 'Virtualization', 'Fast', 'Portable' {
    Context "When Hyper-V is enabled" {
        BeforeAll {
            Mock Get-WindowsOptionalFeature {
                [PSCustomObject]@{
                    FeatureName = "Microsoft-Hyper-V-All"
                    State = "Enabled"
                }
            } -ModuleName PC-AI.Virtualization

            Mock Get-Service {
                @(
                    [PSCustomObject]@{
                        Name = "vmcompute"
                        Status = "Running"
                        StartType = "Automatic"
                    },
                    [PSCustomObject]@{
                        Name = "vmms"
                        Status = "Running"
                        StartType = "Automatic"
                    }
                )
            } -ModuleName PC-AI.Virtualization
        }

        It "Should return a PSCustomObject" {
            $result = Get-HyperVStatus
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have Enabled property" {
            $result = Get-HyperVStatus
            $result.PSObject.Properties.Name | Should -Contain 'Enabled'
        }

        It "Should have Services property" {
            $result = Get-HyperVStatus
            $result.PSObject.Properties.Name | Should -Contain 'Services'
        }

        It "Should have Severity property" {
            $result = Get-HyperVStatus
            $result.Severity | Should -BeIn @('OK', 'Warning', 'Error', 'Unknown')
        }
    }

    Context "When Hyper-V is disabled" {
        BeforeAll {
            Mock Get-WindowsOptionalFeature {
                [PSCustomObject]@{
                    FeatureName = "Microsoft-Hyper-V-All"
                    State = "Disabled"
                }
            } -ModuleName PC-AI.Virtualization
        }

        It "Should return result with Enabled = false" {
            $result = Get-HyperVStatus
            $result.Enabled | Should -Be $false
        }
    }

    Context "When running on Windows Home (no Hyper-V feature)" {
        BeforeAll {
            Mock Get-WindowsOptionalFeature { $null } -ModuleName PC-AI.Virtualization
        }

        It "Should return result with Installed = false" {
            $result = Get-HyperVStatus
            $result.Installed | Should -Be $false
        }
    }
}

Describe "Get-DockerStatus" -Tag 'Unit', 'Virtualization', 'Fast', 'Portable' {
    Context "When Docker Desktop is running" {
        BeforeAll {
            Mock Get-Process {
                [PSCustomObject]@{
                    Name = "Docker Desktop"
                    Id = 1234
                    CPU = 2.5
                    WorkingSet = 500MB
                }
            } -ModuleName PC-AI.Virtualization

            Mock Get-Command {
                [PSCustomObject]@{
                    Name = "docker.exe"
                    CommandType = "Application"
                }
            } -ModuleName PC-AI.Virtualization

            Mock Test-Path { $true } -ModuleName PC-AI.Virtualization
        }

        It "Should return a PSCustomObject" {
            $result = Get-DockerStatus
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have Running property" {
            $result = Get-DockerStatus
            $result.PSObject.Properties.Name | Should -Contain 'Running'
        }

        It "Should have Installed property" {
            $result = Get-DockerStatus
            $result.PSObject.Properties.Name | Should -Contain 'Installed'
        }

        It "Should have Severity property" {
            $result = Get-DockerStatus
            $result.Severity | Should -BeIn @('OK', 'Warning', 'Error', 'Unknown')
        }
    }

    Context "When Docker Desktop is not installed" {
        BeforeAll {
            Mock Get-Process { $null } -ModuleName PC-AI.Virtualization
            Mock Get-Command { $null } -ModuleName PC-AI.Virtualization
            Mock Test-Path { $false } -ModuleName PC-AI.Virtualization
        }

        It "Should return result with Installed = false" {
            $result = Get-DockerStatus
            $result.Installed | Should -Be $false
        }
    }
}

Describe "Optimize-WSLConfig" -Tag 'Unit', 'Virtualization', 'Slow', 'Portable' {
    BeforeAll {
        # Mock CIM query for system info
        Mock Get-CimInstance {
            [PSCustomObject]@{
                TotalVisibleMemorySize = 33554432  # 32 GB in KB
            }
        } -ModuleName PC-AI.Virtualization
    }

    Context "When running in DryRun mode" {
        It "Should return result without modifying files" {
            $result = Optimize-WSLConfig -DryRun
            $result | Should -BeOfType [PSCustomObject]
            $result.Applied | Should -Be $false
        }

        It "Should have Memory property" {
            $result = Optimize-WSLConfig -DryRun
            $result.PSObject.Properties.Name | Should -Contain 'Memory'
        }

        It "Should have Processors property" {
            $result = Optimize-WSLConfig -DryRun
            $result.PSObject.Properties.Name | Should -Contain 'Processors'
        }

        It "Should have Config property with content" {
            $result = Optimize-WSLConfig -DryRun
            $result.Config | Should -Not -BeNullOrEmpty
        }
    }

    Context "When specifying custom values" {
        It "Should accept custom Memory value" {
            $result = Optimize-WSLConfig -Memory "16GB" -DryRun
            $result.Memory | Should -Be "16GB"
        }

        It "Should accept custom Processors value" {
            $result = Optimize-WSLConfig -Processors 8 -DryRun
            $result.Processors | Should -Be 8
        }
    }

    Context "When .wslconfig does not exist" {
        BeforeAll {
            Mock Test-Path { $false } -ModuleName PC-AI.Virtualization -ParameterFilter { $Path -match "\.wslconfig$" }
            Mock Out-File {} -ModuleName PC-AI.Virtualization
        }

        It "Should create new config with -WhatIf" {
            { Optimize-WSLConfig -WhatIf } | Should -Not -Throw
        }
    }
}

Describe "Set-WSLDefenderExclusions" -Tag 'Unit', 'Virtualization', 'Slow', 'RequiresAdmin', 'Portable' {
    BeforeAll {
        Mock Add-MpPreference {} -ModuleName PC-AI.Virtualization
    }

    Context "When running with -WhatIf" {
        It "Should not actually add exclusions" -Skip:(-not $script:IsAdmin) {
            { Set-WSLDefenderExclusions -WhatIf } | Should -Not -Throw
        }
    }

    Context "When Add-MpPreference succeeds" {
        BeforeAll {
            Mock Add-MpPreference {} -ModuleName PC-AI.Virtualization
        }

        It "Should return PSCustomObject with exclusion lists" -Skip:(-not $script:IsAdmin) {
            $result = Set-WSLDefenderExclusions -WhatIf
            $result | Should -BeOfType [PSCustomObject]
            $result.PSObject.Properties.Name | Should -Contain 'PathExclusions'
            $result.PSObject.Properties.Name | Should -Contain 'ProcessExclusions'
        }
    }

    Context "When Add-MpPreference fails (not admin)" {
        BeforeAll {
            Mock Add-MpPreference { throw "Access denied" } -ModuleName PC-AI.Virtualization
        }

        It "Should capture errors in result" -Skip:(-not $script:IsAdmin) {
            $result = Set-WSLDefenderExclusions
            # Errors are captured, not thrown
            $result.Errors.Count | Should -BeGreaterThan 0
        }
    }
}

Describe "Repair-WSLNetworking" -Tag 'Unit', 'Virtualization', 'Slow', 'RequiresAdmin', 'Portable' {
    BeforeAll {
        Mock Get-VMSwitch {
            [PSCustomObject]@{
                Name = "WSL"
                SwitchType = "Internal"
            }
        } -ModuleName PC-AI.Virtualization

        Mock Get-NetAdapter {
            [PSCustomObject]@{
                Name = "vEthernet (WSL)"
                Status = "Up"
            }
        } -ModuleName PC-AI.Virtualization

        Mock Remove-NetIPAddress {} -ModuleName PC-AI.Virtualization
        Mock Remove-NetRoute {} -ModuleName PC-AI.Virtualization
        Mock New-NetIPAddress {} -ModuleName PC-AI.Virtualization
        Mock Get-NetNat { $null } -ModuleName PC-AI.Virtualization
        Mock New-NetNat {} -ModuleName PC-AI.Virtualization
        Mock Start-Sleep {} -ModuleName PC-AI.Virtualization
    }

    Context "When running with -WhatIf" {
        It "Should not make changes" -Skip:(-not $script:IsAdmin) {
            { Repair-WSLNetworking -WhatIf -RestartWSL:$false } | Should -Not -Throw
        }
    }

    Context "When repairing WSL networking" {
        BeforeAll {
            Mock Get-VMSwitch { $null } -ModuleName PC-AI.Virtualization
            Mock New-VMSwitch {} -ModuleName PC-AI.Virtualization
        }

        It "Should return PSCustomObject" -Skip:(-not $script:IsAdmin) {
            $result = Repair-WSLNetworking -WhatIf -RestartWSL:$false
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have VirtualSwitchFixed property" -Skip:(-not $script:IsAdmin) {
            $result = Repair-WSLNetworking -WhatIf -RestartWSL:$false
            $result.PSObject.Properties.Name | Should -Contain 'VirtualSwitchFixed'
        }

        It "Should have NetworkStackReset property" -Skip:(-not $script:IsAdmin) {
            $result = Repair-WSLNetworking -WhatIf -RestartWSL:$false
            $result.PSObject.Properties.Name | Should -Contain 'NetworkStackReset'
        }

        It "Should have Errors property" -Skip:(-not $script:IsAdmin) {
            $result = Repair-WSLNetworking -WhatIf -RestartWSL:$false
            $result.PSObject.Properties.Name | Should -Contain 'Errors'
        }
    }
}

Describe "Backup-WSLConfig" -Tag 'Unit', 'Virtualization', 'Fast', 'Portable' {
    Context "When .wslconfig exists" {
        BeforeAll {
            Mock Test-Path { $true } -ModuleName PC-AI.Virtualization
            Mock Copy-Item {} -ModuleName PC-AI.Virtualization
            Mock Get-Date { [datetime]"2024-01-15 10:30:00" } -ModuleName PC-AI.Virtualization
        }

        It "Should return backup path" {
            $result = Backup-WSLConfig
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should call Copy-Item" {
            Backup-WSLConfig
            Should -Invoke Copy-Item -ModuleName PC-AI.Virtualization -Times 1
        }
    }

    Context "When .wslconfig does not exist" {
        BeforeAll {
            Mock Test-Path { $false } -ModuleName PC-AI.Virtualization
            Mock Copy-Item {} -ModuleName PC-AI.Virtualization
        }

        It "Should return null" {
            $result = Backup-WSLConfig
            $result | Should -BeNullOrEmpty
        }

        It "Should not call Copy-Item" {
            Backup-WSLConfig
            Should -Invoke Copy-Item -ModuleName PC-AI.Virtualization -Times 0
        }
    }

    Context "When custom path is specified" {
        BeforeAll {
            Mock Test-Path { $true } -ModuleName PC-AI.Virtualization
            Mock Copy-Item {} -ModuleName PC-AI.Virtualization
        }

        It "Should use custom path" {
            $customPath = "C:\Backups\wslconfig.bak"
            $result = Backup-WSLConfig -Path $customPath
            $result | Should -Be $customPath
        }
    }
}

AfterAll {
    Remove-Module PC-AI.Virtualization -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
