#Requires -Version 5.1
#Requires -Modules Pester

Describe "PC-AI USB High-Fidelity Diagnostics (Phase 6)" {
    BeforeAll {
        $helperDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'Helpers'
        $resolveHelper = Join-Path $helperDir 'Resolve-TestRepoRoot.ps1'
        if (Test-Path $resolveHelper) {
            . $resolveHelper
        } else {
            throw "Cannot find test helper: $resolveHelper"
        }
        $PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
        Import-Module (Join-Path $PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1") -Force
        Import-Module (Join-Path $PcaiRoot "Modules\PC-AI.USB\PC-AI.USB.psd1") -Force
    }

    Context "Native Device Enumeration (Mocked for testing enrichment)" {
        It "Should retrieve USB devices and enrich with native core" {
            # Mock must target the module that calls the function internally.
            # Also mock Test-PcaiNativeAvailable inside PC-AI.USB to ensure the native path is taken.
            Mock Get-PcaiNativeUsbDiagnostics {
                return '[{"name":"Broken Mock Device","hardware_id":"USB\\VID_1234&PID_5678\\GARY_BROKEN","status":"Error","config_error_code":43,"error_summary":"CM_PROB_DEVICE_REPORTED_FAILURE","help_url":"https://example.com"}]'
            } -ModuleName 'PC-AI.USB'
            Mock Test-PcaiNativeAvailable { return $true } -ModuleName 'PC-AI.USB'

            $results = Get-UsbDeviceList
            $broken = $results | Where-Object { $_.DeviceID -match 'GARY_BROKEN' }

            $broken | Should -Not -BeNullOrEmpty
            $broken.NativeStatus.Code | Should -Be 43
        }
    }

    Context "Error Code Mapping" {
        It "Should map Code 43 (Mocked or Real)" {
            if (Test-PcaiNativeAvailable) {
                $prob = [PcaiNative.PcaiCore]::GetUsbProblemInfo(43)
                $prob.Description | Should -Be "CM_PROB_DEVICE_REPORTED_FAILURE"
                $prob.Summary | Should -Match "stopped|problems|failed"
            }
        }
    }

    Context "Fallback Mechanism" {
        It "Should still work if native core is missing (Mocked absence)" {
            Mock Test-PcaiNativeAvailable { return $false } -ModuleName 'PC-AI.USB'

            $results = Get-UsbDeviceList
            $results | Should -Not -BeNullOrEmpty
            $results | Where-Object { $_.Source -eq 'WMI' } | Should -Not -BeNullOrEmpty
        }
    }
}
