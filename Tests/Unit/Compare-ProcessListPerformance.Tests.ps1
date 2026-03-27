#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Unit tests for Compare-ProcessListPerformance (via Compare-ToolPerformance).

.DESCRIPTION
    Tests the Compare-ProcessListPerformance internal function exposed through
    Compare-ToolPerformance -Test ProcessList in the PC-AI.Acceleration module.
    All external tool and timing calls are mocked so no Rust binaries are required.

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Unit\Compare-ProcessListPerformance.Tests.ps1 -Tag Unit,Acceleration,Benchmark
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
}

AfterAll {
    Remove-Module 'PC-AI.Acceleration' -Force -ErrorAction SilentlyContinue
}

Describe 'Compare-ProcessListPerformance' -Tag 'Unit', 'Acceleration', 'Benchmark' {

    Context 'When procs (Rust tool) is not available' {

        BeforeEach {
            Mock Get-RustToolPath { return $null } -ModuleName PC-AI.Acceleration
        }

        It 'Returns null when procs is missing' {
            $result = Compare-ToolPerformance -Test ProcessList
            $result | Should -BeNullOrEmpty
        }

        It 'Emits a warning about the missing tool' {
            Mock Write-Warning {} -ModuleName PC-AI.Acceleration
            Compare-ToolPerformance -Test ProcessList | Out-Null
            Should -Invoke Write-Warning -ModuleName PC-AI.Acceleration -Times 1 -ParameterFilter {
                $Message -match 'procs'
            }
        }
    }

    Context 'When procs is available and both runs succeed' {

        BeforeEach {
            # Stub the path so the existence check passes
            Mock Get-RustToolPath {
                param($ToolName)
                if ($ToolName -eq 'procs') { return 'C:\fake\procs.exe' }
                return $null
            } -ModuleName PC-AI.Acceleration

            # Return consistent timing objects for both timed runs
            Mock Measure-CommandPerformance {
                param($Name)
                [PSCustomObject]@{
                    Name       = $Name
                    Mean       = if ($Name -eq 'procs') { 12.5 } else { 230.0 }
                    Min        = 10.0
                    Max        = 15.0
                    StdDev     = 1.2
                    Median     = 12.0
                    Iterations = 3
                    Warmup     = 1
                    Unit       = 'ms'
                    Tool       = 'Measure-Command'
                }
            } -ModuleName PC-AI.Acceleration
        }

        It 'Returns a PSCustomObject' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result | Should -BeOfType [PSCustomObject]
        }

        It 'Returns exactly one result for the ProcessList test' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            @($result).Count | Should -Be 1
        }

        It 'Has Test property set to ProcessList' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result.Test | Should -Be 'ProcessList'
        }

        It 'Has RustTool property set to procs' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result.RustTool | Should -Be 'procs'
        }

        It 'Has positive RustMs value' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result.RustMs | Should -BeGreaterThan 0
        }

        It 'Has positive PowerShellMs value' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result.PowerShellMs | Should -BeGreaterThan 0
        }

        It 'Calculates Speedup as PowerShellMs divided by RustMs' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $expected = [Math]::Round(230.0 / 12.5, 2)
            $result.Speedup | Should -Be $expected
        }

        It 'Reflects the requested Iterations count' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 3
            $result.Iterations | Should -Be 3
        }

        It 'Calls Measure-CommandPerformance twice — once per timing side' {
            Compare-ToolPerformance -Test ProcessList -Iterations 3 | Out-Null
            Should -Invoke Measure-CommandPerformance -ModuleName PC-AI.Acceleration -Times 2
        }
    }

    Context 'When RustMs equals zero (avoids divide-by-zero in Speedup)' {

        BeforeEach {
            Mock Get-RustToolPath {
                param($ToolName)
                if ($ToolName -eq 'procs') { return 'C:\fake\procs.exe' }
                return $null
            } -ModuleName PC-AI.Acceleration

            Mock Measure-CommandPerformance {
                param($Name)
                [PSCustomObject]@{
                    Name       = $Name
                    Mean       = 0.0
                    Min        = 0.0
                    Max        = 0.0
                    StdDev     = 0.0
                    Median     = 0.0
                    Iterations = 1
                    Warmup     = 0
                    Unit       = 'ms'
                    Tool       = 'Measure-Command'
                }
            } -ModuleName PC-AI.Acceleration
        }

        It 'Returns a result object even when timing is zero' {
            $result = Compare-ToolPerformance -Test ProcessList -Iterations 1
            $result | Should -Not -BeNullOrEmpty
        }
    }

    Context 'When Compare-ToolPerformance is called with -Test All' {

        BeforeEach {
            # Make every tool unavailable except procs to isolate the ProcessList path
            Mock Get-RustToolPath {
                param($ToolName)
                switch ($ToolName) {
                    'procs'     { return 'C:\fake\procs.exe' }
                    default     { return $null }
                }
            } -ModuleName PC-AI.Acceleration

            Mock Measure-CommandPerformance {
                param($Name)
                [PSCustomObject]@{
                    Name = $Name; Mean = 50.0; Min = 40.0; Max = 60.0
                    StdDev = 5.0; Median = 50.0; Iterations = 2; Warmup = 1; Unit = 'ms'; Tool = 'Measure-Command'
                }
            } -ModuleName PC-AI.Acceleration
        }

        It 'Includes a ProcessList entry in All results' {
            $results = Compare-ToolPerformance -Test All -Iterations 2
            $processList = $results | Where-Object { $_.Test -eq 'ProcessList' }
            $processList | Should -Not -BeNullOrEmpty
        }
    }
}
