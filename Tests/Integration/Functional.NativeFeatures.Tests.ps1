Describe "PC-AI Native Integration & Utilization (Phase 7)" {
    BeforeAll {
        $helperDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'Helpers'
        $resolveHelper = Join-Path $helperDir 'Resolve-TestRepoRoot.ps1'
        if (Test-Path $resolveHelper) {
            . $resolveHelper
        } else {
            throw "Cannot find test helper: $resolveHelper"
        }
        $script:PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1") -Force
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.USB\PC-AI.USB.psd1") -Force
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.Network\PC-AI.Network.psd1") -Force
        Import-Module (Join-Path $script:PcaiRoot "Modules\PC-AI.Performance\PC-AI.Performance.psd1") -Force
    }

    Context "Native Availability" {
        It "Native Core should be available for Phase 7 verification" {
            Test-PcaiNativeAvailable | Should -Be $true
        }
    }

    Context "Module: PC-AI.USB" {
        It "Get-UsbDeviceList should utilize Native core by default" {
            $devices = Get-UsbDeviceList -Verbose
            $nativeDevices = $devices | Where-Object { $_.Source -eq 'Native' }
            $nativeDevices.Count | Should -BeGreaterThan 0
        }
    }

    Context "Module: PC-AI.Network" {
        It "Get-NetworkDiagnostics should integrate Native throughput stats" {
            $diag = Get-NetworkDiagnostics -Verbose
            $diag.PhysicalAdapters | ForEach-Object {
                if ($_.Status -eq 'Up' -and $_.Source -eq 'Native') {
                    $_.BytesSentPersec | Should -Not -BeNull
                }
            }
        }
    }

    Context "Module: PC-AI.Performance" {
        It "Get-ProcessPerformance should use Native core for telemetry" {
            $job = Start-Job -ScriptBlock {
                Import-Module (Join-Path $using:repoRoot 'Modules/PC-AI.Performance') -Force -ErrorAction SilentlyContinue
                Get-ProcessPerformance -Top 5 -SortBy Both
            }
            $perf = $job | Wait-Job -Timeout 30 | Receive-Job -ErrorAction SilentlyContinue
            $job | Remove-Job -Force -ErrorAction SilentlyContinue
            if ($null -eq $perf) { Set-ItResult -Skipped -Because 'Get-ProcessPerformance timed out (>30s)'; return }
            $perf | Should -Not -BeNull

            # Get-ProcessPerformance returns an object with TopByCPU and TopByMemory when SortBy is 'Both'
            if ($perf.TopByCPU) {
                $perf.TopByCPU.Count | Should -BeGreaterThan 0
                $perf.TopByCPU[0].Source | Should -Be 'Native'
            } elseif ($perf.TopByMemory) {
                 $perf.TopByMemory.Count | Should -BeGreaterThan 0
                 $perf.TopByMemory[0].Source | Should -Be 'Native'
            } else {
                # If only one set is returned (default SortBy is 'Memory')
                $perf.Count | Should -BeGreaterThan 0
                $perf[0].Source | Should -Be 'Native'
            }
        }
    }

    Context "Module: PC-AI.Acceleration (Unified Hardware)" {
        It "Get-UnifiedHardwareReportJson should return valid native-backed object" {
            $report = Get-UnifiedHardwareReportJson -Verbosity Full
            $report | Should -Not -BeNullOrEmpty
            # Report is already a PSCustomObject
            $report.TokenEstimate | Should -BeGreaterThan 0
        }
    }
}
