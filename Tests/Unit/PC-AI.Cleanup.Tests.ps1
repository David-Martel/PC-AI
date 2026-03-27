<#
.SYNOPSIS
    Unit tests for PC-AI.Cleanup module

.DESCRIPTION
    Tests PATH cleanup, duplicate file detection, and temporary file removal
#>

BeforeAll {
    $script:OriginalDisableNative = $env:PCAI_DISABLE_NATIVE
    $env:PCAI_DISABLE_NATIVE = '1'

    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Cleanup\PC-AI.Cleanup.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop

    # Mock common helper functions used across all tests
    Mock Write-CleanupLog {} -ModuleName PC-AI.Cleanup
    Mock Format-FileSize {
        param($Bytes)
        return "$Bytes Bytes"
    } -ModuleName PC-AI.Cleanup
}

Describe "Get-PathDuplicates" -Tag 'Unit', 'Cleanup', 'Fast', 'Portable' {
    Context "When PATH has duplicate entries" {
        # Note: This test uses real PATH data since Get-PathDuplicates
        # calls [Environment]::GetEnvironmentVariable() which cannot be easily mocked

        It "Should detect duplicate entries" {
            $result = Get-PathDuplicates -Target Both
            $result | Should -Not -BeNullOrEmpty
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should return PSCustomObject with Summary" {
            $result = Get-PathDuplicates -Target Both
            $result.Summary | Should -Not -BeNullOrEmpty
        }

        It "Should have HealthStatus property" {
            $result = Get-PathDuplicates -Target Both
            $result.Summary.HealthStatus | Should -BeIn @('Healthy', 'Needs Attention', 'Needs Cleanup')
        }
    }

    Context "When checking User PATH" {
        It "Should support User target" {
            $result = Get-PathDuplicates -Target User
            $result | Should -Not -BeNullOrEmpty
            $result.UserPath | Should -Not -BeNullOrEmpty
        }
    }

    Context "When checking Machine PATH" {
        It "Should support Machine target" {
            $result = Get-PathDuplicates -Target Machine
            $result | Should -Not -BeNullOrEmpty
            $result.MachinePath | Should -Not -BeNullOrEmpty
        }
    }

    Context "When checking Both PATHs" {
        It "Should return both User and Machine results" {
            $result = Get-PathDuplicates -Target Both
            $result.UserPath | Should -Not -BeNullOrEmpty
            $result.MachinePath | Should -Not -BeNullOrEmpty
        }

        It "Should detect cross-duplicates" {
            $result = Get-PathDuplicates -Target Both
            $result.PSObject.Properties.Name | Should -Contain 'CrossDuplicates'
        }
    }
}

Describe "Repair-MachinePath" -Tag 'Unit', 'Cleanup', 'Slow', 'RequiresAdmin', 'Portable' {
    BeforeAll {
        # Mock helper functions used by the module
        Mock Backup-EnvironmentVariable { "C:\backup\path.bak" } -ModuleName PC-AI.Cleanup
        Mock Test-IsAdministrator { $true } -ModuleName PC-AI.Cleanup
    }

    Context "When removing duplicate PATH entries" {
        It "Should return a result object" {
            $result = Repair-MachinePath -Target User -WhatIf
            $result | Should -Not -BeNullOrEmpty
            $result | Should -BeOfType [PSCustomObject]
        }

        It "Should have Success property" {
            $result = Repair-MachinePath -Target User -WhatIf
            $result.PSObject.Properties.Name | Should -Contain 'Success'
        }

        It "Should track duplicates removed count" {
            $result = Repair-MachinePath -Target User -WhatIf
            $result.PSObject.Properties.Name | Should -Contain 'DuplicatesRemoved'
        }
    }

    Context "When removing non-existent PATH entries" {
        It "Should support RemoveNonExistent parameter" {
            $result = Repair-MachinePath -Target User -RemoveNonExistent -WhatIf
            $result | Should -Not -BeNullOrEmpty
            $result.PSObject.Properties.Name | Should -Contain 'NonExistentRemoved'
        }

        It "Should track non-existent entries removed" {
            $result = Repair-MachinePath -Target User -RemoveNonExistent -WhatIf
            $result.NonExistentRemoved | Should -BeOfType [int]
        }
    }

    Context "When WhatIf is specified" {
        It "Should not modify PATH with WhatIf" {
            $result = Repair-MachinePath -Target User -WhatIf
            # WhatIf should return the result without making changes
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should return changes list even with WhatIf" {
            $result = Repair-MachinePath -Target User -WhatIf
            $result.PSObject.Properties.Name | Should -Contain 'Changes'
        }
    }

    Context "When targeting Machine PATH" {
        BeforeAll {
            Mock Test-IsAdministrator { $false } -ModuleName PC-AI.Cleanup
        }

        It "Should require Administrator privileges for Machine PATH" {
            $result = Repair-MachinePath -Target Machine -ErrorAction SilentlyContinue
            $result.Success | Should -Be $false
            $result.Warnings | Should -Not -BeNullOrEmpty
        }
    }
}

Describe "Find-DuplicateFiles" -Tag 'Unit', 'Cleanup', 'Slow', 'Portable' {
    Context "When scanning for duplicate files" {
        BeforeAll {
            Mock Get-ChildItem {
                @(
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file1.txt"
                        Name = "file1.txt"
                        DirectoryName = "C:\Temp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-1)
                        CreationTime = (Get-Date).AddDays(-2)
                    }
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file2.txt"
                        Name = "file2.txt"
                        DirectoryName = "C:\Temp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-1)
                        CreationTime = (Get-Date).AddDays(-2)
                    }
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file3.txt"
                        Name = "file3.txt"
                        DirectoryName = "C:\Temp"
                        Length = 2048
                        LastWriteTime = (Get-Date).AddDays(-1)
                        CreationTime = (Get-Date).AddDays(-2)
                    }
                )
            } -ModuleName PC-AI.Cleanup

            # Mock Get-FileHashSafe instead of Get-FileHash (actual implementation uses this)
            Mock Get-FileHashSafe {
                param($Path, $Algorithm)
                if ($Path -match "file[12]\.txt") {
                    return "ABC123"
                } else {
                    return "DEF456"
                }
            } -ModuleName PC-AI.Cleanup
        }

        It "Should detect duplicate files by hash" {
            $result = Find-DuplicateFiles -Path "C:\Temp"
            $result.DuplicateGroups | Where-Object { $_.Hash -eq "ABC123" } | Should -Not -BeNullOrEmpty
        }

        It "Should group files by size first" {
            Find-DuplicateFiles -Path "C:\Temp"

            # Should only hash files with matching sizes (file1.txt and file2.txt both 1024 bytes)
            Should -Invoke Get-FileHashSafe -ModuleName PC-AI.Cleanup -Times 2
        }

        It "Should return result object with DuplicateGroups" {
            $result = Find-DuplicateFiles -Path "C:\Temp"
            $result.PSObject.Properties.Name | Should -Contain 'DuplicateGroups'
            $result.DuplicateGroups[0].Files[0].PSObject.Properties.Name | Should -Contain 'FullName'
        }
    }

    Context "When no duplicates exist" {
        BeforeAll {
            Mock Get-ChildItem {
                @(
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file1.txt"
                        Name = "file1.txt"
                        DirectoryName = "C:\Temp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-1)
                        CreationTime = (Get-Date).AddDays(-2)
                    }
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file2.txt"
                        Name = "file2.txt"
                        DirectoryName = "C:\Temp"
                        Length = 2048
                        LastWriteTime = (Get-Date).AddDays(-1)
                        CreationTime = (Get-Date).AddDays(-2)
                    }
                )
            } -ModuleName PC-AI.Cleanup

            Mock Get-FileHashSafe {
                param($Path, $Algorithm)
                return [guid]::NewGuid().ToString()
            } -ModuleName PC-AI.Cleanup
        }

        It "Should return result with empty DuplicateGroups" {
            $result = Find-DuplicateFiles -Path "C:\Temp"
            $result.DuplicateGroups | Should -BeNullOrEmpty
        }
    }

    Context "When path does not exist" {
        BeforeAll {
            Mock Get-ChildItem { throw "Path not found" } -ModuleName PC-AI.Cleanup
        }

        It "Should handle invalid path" {
            { Find-DuplicateFiles -Path "C:\NonExistent" -ErrorAction Stop } | Should -Throw
        }
    }

    Context "When filtering by extension" {
        BeforeAll {
            Mock Get-ChildItem {
                param($Path, $Include, $Recurse, $File, $ErrorAction)
                if ($Include -contains "*.txt") {
                    @(
                        [PSCustomObject]@{
                            FullName = "C:\Temp\file1.txt"
                            Name = "file1.txt"
                            DirectoryName = "C:\Temp"
                            Length = 1024
                            LastWriteTime = (Get-Date).AddDays(-1)
                            CreationTime = (Get-Date).AddDays(-2)
                        }
                    )
                }
            } -ModuleName PC-AI.Cleanup
        }

        It "Should filter by extension using Include parameter" {
            Find-DuplicateFiles -Path "C:\Temp" -Include "*.txt"

            Should -Invoke Get-ChildItem -ModuleName PC-AI.Cleanup -ParameterFilter {
                $Include -contains "*.txt"
            }
        }
    }
}

Describe "Clear-TempFiles" -Tag 'Unit', 'Cleanup', 'Slow', 'Portable' {
    BeforeAll {
        # Mock helper functions used by Clear-TempFiles
        Mock Test-IsAdministrator { $false } -ModuleName PC-AI.Cleanup
        Mock Test-Path { $true } -ModuleName PC-AI.Cleanup
        Mock Get-TempPaths {
            @(
                [PSCustomObject]@{
                    Name = 'User Temp'
                    Path = 'C:\Users\TestUser\AppData\Local\Temp'
                    RequiresAdmin = $false
                }
            )
        } -ModuleName PC-AI.Cleanup
        # Don't mock Measure-Object - let it work naturally with mocked files
    }

    Context "When clearing Windows temp files" {
        BeforeAll {
            # Override Get-ChildItem mock to return files from the path Get-TempPaths returns
            Mock Get-ChildItem {
                param($Path, $File, $Recurse, $ErrorAction, $Filter)
                # Return files for our test temp path
                @(
                    [PSCustomObject]@{
                        FullName = "C:\Users\TestUser\AppData\Local\Temp\file1.tmp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-10)
                    }
                    [PSCustomObject]@{
                        FullName = "C:\Users\TestUser\AppData\Local\Temp\file2.log"
                        Length = 2048
                        LastWriteTime = (Get-Date).AddDays(-5)
                    }
                )
            } -ModuleName PC-AI.Cleanup

            Mock Remove-Item {} -ModuleName PC-AI.Cleanup
        }

        It "Should remove old temp files" {
            Clear-TempFiles -OlderThanDays 7 -Force

            Should -Invoke Remove-Item -ModuleName PC-AI.Cleanup -Times 1
        }

        It "Should skip recent files" {
            Clear-TempFiles -OlderThanDays 7 -Force

            Should -Invoke Remove-Item -ModuleName PC-AI.Cleanup -Times 0 -ParameterFilter {
                $Path -match "file2\.log"
            }
        }

        It "Should clean multiple temp locations" {
            Clear-TempFiles -Force

            # Verify Get-ChildItem was called at least once
            Should -Invoke Get-ChildItem -ModuleName PC-AI.Cleanup -Times 1 -Exactly:$false
        }
    }

    Context "When WhatIf is specified" {
        BeforeAll {
            Mock Get-ChildItem {
                @(
                    [PSCustomObject]@{
                        FullName = "C:\Temp\file1.tmp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-10)
                    }
                )
            } -ModuleName PC-AI.Cleanup

            Mock Remove-Item {} -ModuleName PC-AI.Cleanup
        }

        It "Should not delete files with WhatIf" {
            Clear-TempFiles -OlderThanDays 7 -WhatIf

            Should -Invoke Remove-Item -ModuleName PC-AI.Cleanup -Times 0
        }
    }

    Context "When files are in use" {
        BeforeAll {
            Mock Get-ChildItem {
                @(
                    [PSCustomObject]@{
                        FullName = "C:\Temp\locked.tmp"
                        Length = 1024
                        LastWriteTime = (Get-Date).AddDays(-10)
                    }
                )
            } -ModuleName PC-AI.Cleanup

            Mock Remove-Item { throw "File is in use" } -ModuleName PC-AI.Cleanup
        }

        It "Should handle locked files gracefully" {
            { Clear-TempFiles -OlderThanDays 7 -Force -ErrorAction SilentlyContinue } | Should -Not -Throw
        }
    }
}

AfterAll {
    if ($null -eq $script:OriginalDisableNative) {
        Remove-Item Env:PCAI_DISABLE_NATIVE -ErrorAction SilentlyContinue
    } else {
        $env:PCAI_DISABLE_NATIVE = $script:OriginalDisableNative
    }
    Remove-Module PC-AI.Cleanup -Force -ErrorAction SilentlyContinue
    Remove-Module MockData -Force -ErrorAction SilentlyContinue
}
