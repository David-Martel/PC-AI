#Requires -Modules Pester

BeforeAll {
    $modulePath = Split-Path -Parent $PSScriptRoot
    Import-Module (Join-Path $modulePath 'PC-AI.Acceleration.psd1') -Force
}

Describe 'Find-DuplicatesFast' {
    It 'Uses the native duplicate path when available for SHA256 requests' {
        Mock -CommandName Get-Command -ModuleName 'PC-AI.Acceleration' -MockWith {
            [pscustomobject]@{ Name = 'Invoke-PcaiNativeDuplicates' }
        } -ParameterFilter { $Name -eq 'Invoke-PcaiNativeDuplicates' }
        Mock -CommandName Test-PcaiNativeAvailable -ModuleName 'PC-AI.Acceleration' -MockWith { $true }
        Mock -CommandName Invoke-PcaiNativeDuplicates -ModuleName 'PC-AI.Acceleration' -MockWith {
            [pscustomobject]@{
                IsSuccess = $true
                FilesScanned = 3
                DuplicateFiles = 1
                WastedBytes = 128
                ElapsedMs = 5
                Groups = @(
                    [pscustomobject]@{
                        Hash = 'abc'
                        Size = 128
                        WastedBytes = 128
                        Paths = @('C:\temp\a.txt', 'C:\temp\b.txt')
                    }
                )
            }
        }

        $result = @(Find-DuplicatesFast -Path $env:TEMP -Recurse)
        $result.Count | Should -Be 1
        $result[0].Provider | Should -Be 'PcaiNative'
        $result[0].Duplicates.Count | Should -Be 1
        Should -Invoke Invoke-PcaiNativeDuplicates -ModuleName 'PC-AI.Acceleration' -Times 1
    }

    It 'Falls back when the native path is disabled' {
        Mock -CommandName Invoke-PowerShellDuplicateScan -ModuleName 'PC-AI.Acceleration' -MockWith {
            @([pscustomobject]@{ Hash = 'fallback'; WastedBytes = 0 })
        }

        $result = @(Find-DuplicatesFast -Path $env:TEMP -DisableNative)
        $result.Count | Should -Be 1
        $result[0].Hash | Should -Be 'fallback'
        Should -Invoke Invoke-PowerShellDuplicateScan -ModuleName 'PC-AI.Acceleration' -Times 1
    }
}
