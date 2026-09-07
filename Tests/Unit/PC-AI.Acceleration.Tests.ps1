<#
.SYNOPSIS
    Unit tests for Get-UnifiedHardwareReportJson.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
    # Evict any copy already loaded by an earlier suite before importing.
    # `Import-Module -Force` re-imports, but it does NOT remove a copy that
    # was loaded from a different path, so two modules of the same name can
    # coexist. Pester then refuses to mock into either -- "Multiple script or
    # manifest modules named 'X' are currently loaded" -- and every mocked
    # call falls through to the real cmdlet. That is why these files pass in
    # isolation and fail in a full run.
    Get-Module 'PC-AI.Acceleration' -All | Remove-Module -Force -ErrorAction SilentlyContinue
    Import-Module $ModulePath -Force -ErrorAction Stop
}

Describe "Get-UnifiedHardwareReportJson" -Tag 'Unit', 'Acceleration', 'Fast', 'Portable' {
    Context "When native diagnostics are unavailable" {
        BeforeEach {
            Mock Test-PcaiNativeAvailable { $false } -ModuleName PC-AI.Acceleration
        }

        It "returns fallback object with error metadata" {
            $result = Get-UnifiedHardwareReportJson -Verbosity Normal
            $result.Error | Should -Be "Native diagnostics unavailable"
            $result.Timestamp | Should -Not -BeNullOrEmpty
        }
    }

    Context "When native diagnostics are available" {
        BeforeEach {
            Mock Test-PcaiNativeAvailable { $true } -ModuleName PC-AI.Acceleration
            Mock Invoke-PcaiNativeUnifiedHardwareReport {
                '{"system":{"hostname":"pc-ai-dev"}}'
            } -ModuleName PC-AI.Acceleration
            Mock Invoke-PcaiNativeEstimateTokens { [uint64]42 } -ModuleName PC-AI.Acceleration
        }

        It "returns parsed report and token estimate" {
            $result = Get-UnifiedHardwareReportJson -Verbosity Full

            $result.system.hostname | Should -Be 'pc-ai-dev'
            $result.TokenEstimate | Should -Be 42

            Should -Invoke Invoke-PcaiNativeUnifiedHardwareReport -ModuleName PC-AI.Acceleration -ParameterFilter {
                $Verbosity -eq 'Full'
            }
            Should -Invoke Invoke-PcaiNativeEstimateTokens -ModuleName PC-AI.Acceleration -Times 1
        }
    }

    Context "When native report generation throws" {
        BeforeEach {
            Mock Test-PcaiNativeAvailable { $true } -ModuleName PC-AI.Acceleration
            Mock Invoke-PcaiNativeUnifiedHardwareReport { throw "native failure" } -ModuleName PC-AI.Acceleration
            Mock Write-Error {} -ModuleName PC-AI.Acceleration
        }

        It "returns null and reports an error" {
            $result = Get-UnifiedHardwareReportJson -Verbosity Normal -ErrorAction SilentlyContinue
            $result | Should -BeNullOrEmpty
            Should -Invoke Write-Error -ModuleName PC-AI.Acceleration -Times 1
        }
    }
}
