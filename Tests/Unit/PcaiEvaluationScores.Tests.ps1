#Requires -Version 7.0

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force
}

Describe 'Metric definitions execute their actual calculators' -Tag 'Unit', 'Fast', 'Portable', 'Evaluation' {
    # These are consumer inputs and independent expected values, not mocked
    # calculators. Keep units, missing metadata and zero-duration boundaries.
    It 'calculates <Metric> for <Case>' -ForEach @(
        @{ Metric='latency'; Case='milliseconds'; Seconds=1.25; Response=''; Metadata=@{}; Expected=1250.0 }
        @{ Metric='throughput'; Case='four words in two seconds'; Seconds=2.0; Response='one two three four'; Metadata=@{}; Expected=2.0 }
        @{ Metric='throughput'; Case='zero duration'; Seconds=0.0; Response='one two'; Metadata=@{}; Expected=0.0 }
        @{ Metric='memory'; Case='recorded megabytes'; Seconds=0.0; Response=''; Metadata=@{memory_mb=42.5}; Expected=42.5 }
        @{ Metric='memory'; Case='missing megabytes'; Seconds=0.0; Response=''; Metadata=@{}; Expected=0.0 }
        @{ Metric='accuracy'; Case='fractional observation'; Seconds=0.0; Response=''; Metadata=@{accuracy=0.375}; Expected=0.375 }
        @{ Metric='accuracy'; Case='missing observation'; Seconds=0.0; Response=''; Metadata=@{}; Expected=0.0 }
        @{ Metric='similarity'; Case='no reference supplied'; Seconds=0.0; Response='alpha'; Metadata=@{}; Expected=1.0 }
        @{ Metric='similarity'; Case='identical reference'; Seconds=0.0; Response='alpha beta'; Metadata=@{expected='alpha beta'}; Expected=1.0 }
        @{ Metric='groundedness'; Case='no context supplied'; Seconds=0.0; Response='alpha'; Metadata=@{}; Expected=1.0 }
        @{ Metric='groundedness'; Case='partial overlap'; Seconds=0.0; Response='alpha delta gamma'; Metadata=@{context='alpha beta gamma'}; Expected=0.6567 }
        @{ Metric='coherence'; Case='empty response'; Seconds=0.0; Response=''; Metadata=@{}; Expected=0.0 }
        @{ Metric='coherence'; Case='repeated sentences'; Seconds=0.0; Response='Error detected. Error detected. Error detected. Error detected. Error detected.'; Metadata=@{}; Expected=0.72 }
        @{ Metric='toxicity'; Case='one keyword'; Seconds=0.0; Response='I hate that.'; Metadata=@{}; Expected=0.2 }
        @{ Metric='toxicity'; Case='ordinary diagnostic'; Seconds=0.0; Response='Normal diagnostic output.'; Metadata=@{}; Expected=0.0 }
    ) {
        $actual = & (Get-Module PC-AI.Evaluation) {
            param($name, $seconds, $response, $metadata)
            $definition = Get-MetricDefinition -Name $name
            $result = [pscustomobject]@{
                Duration = [timespan]::FromSeconds($seconds)
                Response = $response
                Metrics = $metadata
            }
            [pscustomobject]@{ Name=$definition.Name; Value=(& $definition.Calculator $result) }
        } $Metric $Seconds $Response $Metadata
        $actual.Name | Should -BeExactly $Metric
        $actual.Value | Should -Be $Expected
    }

    It 'leaves unsupported metrics unavailable' {
        & (Get-Module PC-AI.Evaluation) { Get-MetricDefinition -Name 'unsupported-metric' } |
            Should -BeNullOrEmpty
    }
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
