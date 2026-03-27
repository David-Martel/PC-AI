#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:BenchmarkScript = Join-Path $script:ProjectRoot 'Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1'
}

Describe "Invoke-PcaiToolingBenchmarks" -Tag 'Unit', 'Benchmarks', 'Acceleration', 'Portable' {
    It "uses the repo-imported acceleration module instead of a shadowed session function" {
        function global:Measure-CommandPerformance {
            param()

            [PSCustomObject]@{
                Name                   = 'shadowed'
                Command                = 'shadowed'
                Mean                   = 1
                StdDev                 = 0
                Min                    = 1
                Max                    = 1
                Median                 = 1
                Iterations             = 1
                Warmup                 = 0
                Unit                   = 'ms'
                Tool                   = 'shadowed'
                WorkingSetDeltaMeanBytes = $null
                WorkingSetDeltaMaxBytes  = $null
                PrivateMemoryDeltaMeanBytes = $null
                PrivateMemoryDeltaMaxBytes  = $null
                ManagedMemoryDeltaMeanBytes = $null
                ManagedMemoryDeltaMaxBytes  = $null
                ManagedAllocatedMeanBytes = $null
                ManagedAllocatedMaxBytes  = $null
            }
        }

        try {
            $result = & $script:BenchmarkScript -CaseId 'runtime-config' -SkipCapabilities -PassThru
            $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $row = @($report.Results | Where-Object CaseId -eq 'runtime-config' | Select-Object -First 1)[0]

            $row | Should -Not -BeNullOrEmpty
            $row.Tool | Should -Be 'Measure-Command'
            $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
            $row.ManagedMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
        }
        finally {
            Remove-Item -Path Function:\global:Measure-CommandPerformance -ErrorAction SilentlyContinue
        }
    }

    It "records memory metrics for the direct Rust probe case" {
        $result = & $script:BenchmarkScript -CaseId 'direct-core-probe' -SkipCapabilities -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'direct-core-probe' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.MeanMs | Should -BeGreaterThan 0
        $row.WorkingSetDeltaMeanBytes | Should -Not -BeNullOrEmpty
        $row.PrivateMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
        $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
    }

    It "emits content-search rows for all expected backends with memory metrics" {
        $result = & $script:BenchmarkScript -CaseId 'content-search' -SkipCapabilities -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $rows = @($report.Results | Where-Object CaseId -eq 'content-search')

        @($rows | Select-Object -ExpandProperty Backend | Sort-Object) | Should -Be @('accelerated', 'native', 'powershell')
        foreach ($row in $rows) {
            $row.WorkingSetDeltaMeanBytes | Should -Not -BeNullOrEmpty
            $row.PrivateMemoryDeltaMeanBytes | Should -Not -BeNullOrEmpty
            $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
        }
    }

    It "emits a shared-cache benchmark row with memory metrics" {
        $result = & $script:BenchmarkScript -CaseId 'shared-cache-hit' -SkipCapabilities -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'shared-cache-hit' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
        $row.ManagedAllocatedMeanBytes | Should -Not -BeNullOrEmpty
    }

    It "emits an external-cache-status benchmark row" {
        $result = & $script:BenchmarkScript -CaseId 'external-cache-status' -SkipCapabilities -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'external-cache-status' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
    }

    It "emits an acceleration-stack benchmark row" {
        $result = & $script:BenchmarkScript -CaseId 'acceleration-stack' -SkipCapabilities -PassThru
        $report = Get-Content -Path $result.JsonReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row = @($report.Results | Where-Object CaseId -eq 'acceleration-stack' | Select-Object -First 1)[0]

        $row | Should -Not -BeNullOrEmpty
        $row.Backend | Should -Be 'powershell'
        $row.MeanMs | Should -BeGreaterThan 0
    }
}
