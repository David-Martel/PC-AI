#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Unit tests for Get-ProcessPerformance -ExcludeIdle parameter.

.DESCRIPTION
    Verifies that the -ExcludeIdle switch correctly filters out the System Idle
    Process while leaving all other processes intact.  All Get-Process and
    Get-CimInstance calls are mocked so no live system access is needed.

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Unit\Get-ProcessPerformance.ExcludeIdle.Tests.ps1 -Tag Unit,Performance,ExcludeIdle
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Performance\PC-AI.Performance.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
}

AfterAll {
    Remove-Module 'PC-AI.Performance' -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

function New-MockProcess {
    param([string]$Name, [int]$Id, [double]$Cpu = 0, [long]$MemoryBytes = 100MB)
    [PSCustomObject]@{
        ProcessName             = $Name
        Id                      = $Id
        CPU                     = $Cpu
        TotalProcessorTime      = [TimeSpan]::FromSeconds($Cpu)
        StartTime               = (Get-Date).AddMinutes(-10)
        WorkingSet64            = $MemoryBytes
        Threads                 = @(1)
        HandleCount             = 100
        Path                    = if ($Name -eq 'Idle') { $null } else { "C:\fake\$Name.exe" }
        MainModule              = $null
        PriorityClass           = 'Normal'
    }
}

Describe 'Get-ProcessPerformance -ExcludeIdle' -Tag 'Unit', 'Performance', 'ExcludeIdle', 'Portable' {

    # ------------------------------------------------------------------
    # Context: process list contains the Idle process
    # ------------------------------------------------------------------
    Context 'When process list contains the Idle process' {

        BeforeEach {
            Mock Get-Process {
                New-MockProcess -Name 'Idle'   -Id 0   -Cpu 0
                New-MockProcess -Name 'chrome' -Id 100 -Cpu 45.5 -MemoryBytes 500MB
                New-MockProcess -Name 'Code'   -Id 200 -Cpu 12.3 -MemoryBytes 300MB
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory        = 16GB
                        NumberOfLogicalProcessors  = 8
                    }
                }
            } -ModuleName PC-AI.Performance

            # Owner lookup — keep it simple
            Mock Get-ProcessOwner { return 'SYSTEM' } -ModuleName PC-AI.Performance
        }

        It 'Excludes the Idle process when -ExcludeIdle is specified (SortBy CPU)' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $processNames = $result | Select-Object -ExpandProperty ProcessName
            $processNames | Should -Not -Contain 'Idle'
        }

        It 'Includes the Idle process when -ExcludeIdle is NOT specified (SortBy CPU)' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10
            $processNames = $result | Select-Object -ExpandProperty ProcessName
            $processNames | Should -Contain 'Idle'
        }

        It 'Excludes the Idle process when using SortBy Memory and -ExcludeIdle' {
            $result = Get-ProcessPerformance -SortBy Memory -Top 10 -ExcludeIdle
            $processNames = $result | Select-Object -ExpandProperty ProcessName
            $processNames | Should -Not -Contain 'Idle'
        }

        It 'Excludes the Idle process from TopByCPU when SortBy Both and -ExcludeIdle' {
            $result = Get-ProcessPerformance -SortBy Both -Top 10 -ExcludeIdle
            $cpuNames = $result.TopByCPU | Select-Object -ExpandProperty ProcessName
            $cpuNames | Should -Not -Contain 'Idle'
        }

        It 'Excludes the Idle process from TopByMemory when SortBy Both and -ExcludeIdle' {
            $result = Get-ProcessPerformance -SortBy Both -Top 10 -ExcludeIdle
            $memNames = $result.TopByMemory | Select-Object -ExpandProperty ProcessName
            $memNames | Should -Not -Contain 'Idle'
        }

        It 'Returns exactly 2 processes when Idle is excluded from a 3-process list' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $result.Count | Should -Be 2
        }

        It 'Returns all 3 processes when Idle is NOT excluded' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10
            $result.Count | Should -Be 3
        }

        It 'Preserves non-Idle processes in sorted order after exclusion' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $result[0].ProcessName | Should -Be 'chrome'
            $result[1].ProcessName | Should -Be 'Code'
        }
    }

    # ------------------------------------------------------------------
    # Context: process list does NOT contain the Idle process
    # ------------------------------------------------------------------
    Context 'When process list does not contain the Idle process' {

        BeforeEach {
            Mock Get-Process {
                New-MockProcess -Name 'notepad' -Id 300 -Cpu 1.0 -MemoryBytes 50MB
                New-MockProcess -Name 'explorer' -Id 400 -Cpu 5.0 -MemoryBytes 200MB
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory        = 16GB
                        NumberOfLogicalProcessors  = 8
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-ProcessOwner { return 'User' } -ModuleName PC-AI.Performance
        }

        It 'Returns all processes unaffected when -ExcludeIdle is set but no Idle process exists' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $result.Count | Should -Be 2
        }

        It 'Produces identical results with and without -ExcludeIdle when no Idle is present' {
            $withExclude    = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $withoutExclude = Get-ProcessPerformance -SortBy CPU -Top 10
            $withExclude.Count | Should -Be $withoutExclude.Count
        }
    }

    # ------------------------------------------------------------------
    # Context: process list contains ONLY the Idle process
    # ------------------------------------------------------------------
    Context 'When the only process is the Idle process' {

        BeforeEach {
            Mock Get-Process {
                New-MockProcess -Name 'Idle' -Id 0 -Cpu 0
            } -ModuleName PC-AI.Performance

            Mock Get-CimInstance {
                param($ClassName)
                if ($ClassName -eq 'Win32_ComputerSystem') {
                    [PSCustomObject]@{
                        TotalPhysicalMemory        = 16GB
                        NumberOfLogicalProcessors  = 8
                    }
                }
            } -ModuleName PC-AI.Performance

            Mock Get-ProcessOwner { return 'SYSTEM' } -ModuleName PC-AI.Performance
        }

        It 'Returns empty result set when all processes are excluded' {
            $result = Get-ProcessPerformance -SortBy CPU -Top 10 -ExcludeIdle
            $result | Should -BeNullOrEmpty
        }
    }
}
