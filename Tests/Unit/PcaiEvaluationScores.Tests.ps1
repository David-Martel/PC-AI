#Requires -Version 7.0

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force
}

AfterAll { Remove-Module PC-AI.Evaluation -Force -ErrorAction SilentlyContinue }

Describe 'Fractional evaluation scores retain their numerical precision' {
    It 'retains a fractional normalized <Metric> score' -ForEach @(
        @{ Metric='latency'; Value=1250.0; Expected=0.75 },
        @{ Metric='throughput'; Value=25.0; Expected=0.25 },
        @{ Metric='memory'; Value=250.0; Expected=0.75 },
        @{ Metric='accuracy'; Value=0.6; Expected=0.6 },
        @{ Metric='toxicity'; Value=0.2; Expected=0.8 },
        @{ Metric='latency'; Value=6000.0; Expected=0.0 },
        @{ Metric='throughput'; Value=150.0; Expected=1.0 },
        @{ Metric='accuracy'; Value=-0.2; Expected=0.0 }
    ) {
        $suite = New-EvaluationSuite -Name 'normalized-score' -Metrics @($Metric)
        $result = [Activator]::CreateInstance($suite.Results.GetType().GetGenericArguments()[0])
        $result.Metrics = @{ $Metric=$Value }
        $score = & (Get-Module PC-AI.Evaluation) {
            param($result, $metrics)
            Calculate-OverallScore -Result $result -Metrics $metrics
        } $result $suite.Metrics
        $score | Should -Be $Expected
    }

    It 'combines fractional metrics using the configured weights' {
        $suite = New-EvaluationSuite -Name 'weighted-score' -Metrics @('latency', 'throughput')
        $suite.Metrics[0].Weight = 2.0
        $suite.Metrics[1].Weight = 1.0
        $result = [Activator]::CreateInstance($suite.Results.GetType().GetGenericArguments()[0])
        $result.Metrics = @{ latency=1250.0; throughput=25.0 }
        $score = & (Get-Module PC-AI.Evaluation) {
            param($result, $metrics)
            Calculate-OverallScore -Result $result -Metrics $metrics
        } $result $suite.Metrics
        $score | Should -Be 0.5833
    }

    It 'retains the intermediate keyword toxicity score instead of rounding it to zero' {
        Measure-Toxicity -Response 'I hate that.' | Should -Be 0.2
        Measure-Toxicity -Response 'Normal diagnostic output.' | Should -Be 0.0
    }

    It 'retains the grounded word fraction and its unmatched-word penalty' {
        Measure-Groundedness -Response 'alpha delta gamma' -Context 'alpha beta gamma' | Should -Be 0.6567
    }

    It 'retains coherence penalties for repeated sentences' {
        Measure-Coherence -Response 'Error detected. Error detected. Error detected. Error detected. Error detected.' | Should -Be 0.72
        Measure-Coherence -Response '' | Should -Be 0.0
    }
}
