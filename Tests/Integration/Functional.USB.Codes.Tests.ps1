#Requires -Version 5.1
#Requires -Modules Pester

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

Describe "PC-AI USB High-Fidelity Diagnostics (Phase 6)" {
    BeforeAll {
        $PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
        Import-Module (Join-Path $PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1") -Force
        Import-Module (Join-Path $PcaiRoot "Modules\PC-AI.USB\PC-AI.USB.psd1") -Force
    }

    Context "Native Device Enumeration (Mocked for testing enrichment)" {
        # Global mock to ensure it's hit
        Mock Get-PcaiNativeUsbDiagnostics {
            return '[{"name":"Broken Mock Device","hardware_id":"USB\\VID_1234&PID_5678\\GARY_BROKEN","status":"Error","config_error_code":43,"error_summary":"CM_PROB_DEVICE_REPORTED_FAILURE","help_url":"https://example.com"}]'
        }

        It "Should retrieve USB devices and enrich with native core" {
            $results = Get-UsbDeviceList
            $broken = $results | Where-Object { $_.DeviceID -match 'GARY_BROKEN' }

            # If still null, let's try calling the helper directly to see if the mock is hit
            if ($null -eq $broken) {
                $raw = Get-PcaiNativeUsbDiagnostics
                Write-Warning "RAW MOCK OUTPUT: $raw"
            }

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
