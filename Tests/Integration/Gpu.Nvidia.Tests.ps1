<#
.SYNOPSIS
    Hardware integration tests for PC-AI.Gpu module.

.DESCRIPTION
    Verifies NVIDIA GPU discovery, software status, utilization, and compatibility
    on actual hardware. These tests require a physical NVIDIA GPU and driver.
#>

BeforeAll {
    $script:ModulesPath = Join-Path $PSScriptRoot '..\..\Modules'
    $manifestPath = Join-Path $script:ModulesPath "PC-AI.Gpu\PC-AI.Gpu.psd1"

    if (-not (Test-Path $manifestPath)) {
        throw "PC-AI.Gpu module not found at $manifestPath"
    }

    Import-Module $manifestPath -Force -ErrorAction Stop
}

Describe "NVIDIA GPU Hardware Integration" -Tag 'Integration', 'Gpu', 'Nvidia', 'RequiresHardware' {
    Context "Get-NvidiaGpuInventory" {
        It "Should return at least one NVIDIA GPU" {
            $inventory = Get-NvidiaGpuInventory
            $inventory | Should -Not -BeNullOrEmpty
            $inventory.Count | Should -BeGreaterOrEqual 1
        }

        It "Should have expected properties on inventory objects" {
            $inventory = Get-NvidiaGpuInventory | Select-Object -First 1
            $inventory.Name | Should -Not -BeNullOrEmpty
            $inventory.DriverVersion | Should -Not -BeNullOrEmpty
            $inventory.Source | Should -Match 'nvml-ffi|nvidia-smi|cim'
        }
    }

    Context "Get-NvidiaSoftwareStatus" {
        It "Should return software status for NVIDIA components" {
            $status = Get-NvidiaSoftwareStatus
            $status | Should -Not -BeNullOrEmpty
            $status.ComponentId | Should -Contain 'gpu-driver'
        }

        It "Should have valid status values" {
            $status = Get-NvidiaSoftwareStatus | Select-Object -First 1
            $status.Status | Should -Match 'Current|Outdated|NotInstalled|Unknown'
        }
    }

    Context "Get-NvidiaGpuUtilization" {
        It "Should return real-time utilization metrics" {
            $inventory = Get-NvidiaGpuInventory
            $util = Get-NvidiaGpuUtilization

            if ($inventory[0].Source -eq 'nvidia-smi') {
                $util | Should -Not -BeNullOrEmpty
                $util[0].GpuUtilization | Should -Not -BeNull
                $util[0].Timestamp | Should -Not -BeNull
            }
            else {
                # If only CIM is available, utilization returns @()
                $util | Should -BeEmpty
            }
        }
    }

    Context "Get-NvidiaCompatibilityMatrix" {
        It "Should build a compatibility matrix" {
            $matrix = Get-NvidiaCompatibilityMatrix
            $matrix | Should -Not -BeNullOrEmpty
            $matrix.GpuName | Should -Not -BeNullOrEmpty
            $matrix.Status | Should -Not -BeNullOrEmpty
        }

        It "Should identify blockers if they exist" {
            $matrix = Get-NvidiaCompatibilityMatrix
            $matrix[0].IsBlocker | Should -Not -BeNull
        }
    }
}

AfterAll {
    Remove-Module PC-AI.Gpu -ErrorAction SilentlyContinue
}
