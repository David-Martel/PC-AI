#Requires -Version 7.0

BeforeAll {
    $script:EvaluationModule = Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force -PassThru |
        Where-Object Name -eq 'PC-AI.Evaluation'
}

AfterAll {
    Remove-Module PC-AI.Evaluation -Force -ErrorAction SilentlyContinue
}

Describe 'Evaluation baselines preserve the metric distribution consumed by regression checks' {
    BeforeEach {
        & $script:EvaluationModule {
            param($directory)
            $script:EvaluationConfig.BaselinePath = $directory
            $script:Baselines = @{}
        } (Join-Path $TestDrive 'baselines')
        Mock Invoke-EvaluationSuite -ModuleName PC-AI.Evaluation {
            param($Suite)
            $Suite.Results.Clear()
            foreach ($values in @(@(100.0, 10.0), @(200.0, 20.0))) {
                $result = [Activator]::CreateInstance($Suite.Results.GetType().GetGenericArguments()[0])
                $result.TestCaseId = 'case-' + $Suite.Results.Count
                $result.Status = 'pass'
                $result.OverallScore = 0.9
                $result.Duration = [TimeSpan]::FromMilliseconds($values[0])
                $result.Metrics = @{ latency=$values[0]; throughput=$values[1] }
                $Suite.Results.Add($result)
            }
            $Suite.GetSummary()
        }
    }

    It 'persists actual metric means and preserves the completed-run summary separately' {
        $suite = New-EvaluationSuite -Name 'baseline-fixture' -Metrics @('latency', 'throughput')
        $baseline = New-BaselineSnapshot -Name 'controlled-reference' -Suite $suite
        $saved = Get-Content (Join-Path $TestDrive 'baselines/controlled-reference.json') -Raw | ConvertFrom-Json -AsHashtable
        $saved.Metrics.Keys | Should -Contain 'latency'
        $saved.Metrics.latency.Mean | Should -Be 150
        $saved.Metrics.latency.Min | Should -Be 100
        $saved.Metrics.latency.Max | Should -Be 200
        $saved.Metrics.throughput.Mean | Should -Be 15
        $saved.Summary.TotalTests | Should -Be 2
        $baseline.TestCount | Should -Be 2
        Should -Invoke Invoke-EvaluationSuite -ModuleName PC-AI.Evaluation -Times 1 -Exactly
    }

    It 'detects both latency degradation and throughput loss against the persisted baseline' {
        $suite = New-EvaluationSuite -Name 'regression-fixture' -Metrics @('latency', 'throughput')
        New-BaselineSnapshot -Name 'controlled-reference' -Suite $suite | Out-Null
        & $script:EvaluationModule {
            param($Suite)
            foreach ($result in $Suite.Results) {
                $result.Metrics.latency *= 1.2
                $result.Metrics.throughput *= 0.8
            }
        } $suite
        $comparison = Test-ForRegression -BaselineName 'controlled-reference' -Suite $suite
        $comparison.HasRegressions | Should -BeTrue
        $comparison.Regressions.Count | Should -Be 2
        ($comparison.Regressions | Where-Object Metric -eq latency).Change | Should -Be 20
        ($comparison.Regressions | Where-Object Metric -eq throughput).Change | Should -Be -20
        $comparison.Improvements.Count | Should -Be 0
    }
}
