#Requires -Version 7.0
#Requires -Modules Pester

<#
.SYNOPSIS
    Pester tests for PC-AI.Evaluation module

.DESCRIPTION
    Unit and integration tests for the LLM evaluation framework
#>

BeforeAll {
    $projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    Import-Module (Join-Path $projectRoot "Modules\PC-AI.Evaluation\PC-AI.Evaluation.psd1") -Force
}

Describe "PC-AI.Evaluation Module" {
    Context "Module Loading" {
        It "Should import the module successfully" {
            Get-Module PC-AI.Evaluation | Should -Not -BeNullOrEmpty
        }

        It "Should export expected functions" {
            $exportedFunctions = (Get-Module PC-AI.Evaluation).ExportedFunctions.Keys

            $expectedFunctions = @(
                'New-EvaluationSuite',
                'Invoke-EvaluationSuite',
                'Get-EvaluationResults',
                'Measure-InferenceLatency',
                'Invoke-LLMJudge',
                'Compare-ResponsePair',
                'New-BaselineSnapshot',
                'Test-ForRegression',
                'New-ABTest',
                'Add-ABTestResult',
                'Get-ABTestAnalysis',
                'Get-EvaluationDataset'
            )

            foreach ($func in $expectedFunctions) {
                $exportedFunctions | Should -Contain $func
            }
        }
    }

    Context "Evaluation Suite Creation" {
        It "Should create an evaluation suite with default metrics" {
            $suite = New-EvaluationSuite -Name "TestSuite" -Description "Test"

            $suite | Should -Not -BeNullOrEmpty
            $suite.Name | Should -Be "TestSuite"
            $suite.Metrics.Count | Should -BeGreaterThan 0
        }

        It "Should create suite with specified metrics" {
            $suite = New-EvaluationSuite -Name "Custom" -Metrics @('latency', 'accuracy')

            $suite.Metrics | Where-Object { $_.Name -eq 'latency' } | Should -Not -BeNullOrEmpty
        }

        It "Should include default metrics when requested" {
            $suite = New-EvaluationSuite -Name "WithDefaults" -Metrics @('similarity') -IncludeDefaultMetrics

            $metricNames = $suite.Metrics | ForEach-Object { $_.Name }
            $metricNames | Should -Contain 'latency'
            $metricNames | Should -Contain 'throughput'
            $metricNames | Should -Contain 'similarity'
        }
    }

    Context "Test Case Management" {
        It "Should create a test case with required fields" {
            $tc = New-EvaluationTestCase -Id "test-001" -Prompt "Hello world" -Category "general"

            $tc | Should -Not -BeNullOrEmpty
            $tc.Id | Should -Be "test-001"
            $tc.Prompt | Should -Be "Hello world"
            $tc.Category | Should -Be "general"
        }

        It "Should create test case with optional fields" {
            $tc = New-EvaluationTestCase -Id "test-002" `
                -Prompt "What is 2+2?" `
                -ExpectedOutput "4" `
                -Context @{ topic = "math" } `
                -Tags @('math', 'simple')

            $tc.ExpectedOutput | Should -Be "4"
            $tc.Context.topic | Should -Be "math"
            $tc.Tags | Should -Contain 'math'
        }
    }

    Context "Built-in Datasets" {
        It "Should load diagnostic dataset" {
            $dataset = Get-EvaluationDataset -Name 'diagnostic'

            $dataset | Should -Not -BeNullOrEmpty
            $dataset.Count | Should -BeGreaterThan 0
            $dataset[0].Id | Should -Not -BeNullOrEmpty
            $dataset[0].Prompt | Should -Not -BeNullOrEmpty
        }

        It "Should load general dataset" {
            $dataset = Get-EvaluationDataset -Name 'general'

            $dataset | Should -Not -BeNullOrEmpty
            $dataset | Where-Object { $_.Category -eq 'factual' } | Should -Not -BeNullOrEmpty
        }

        It "Should load safety dataset" {
            $dataset = Get-EvaluationDataset -Name 'safety'

            $dataset | Should -Not -BeNullOrEmpty
            $dataset | Where-Object { $_.Category -eq 'refusal' } | Should -Not -BeNullOrEmpty
        }

        It "Should load custom JSON dataset" {
            $datasetPath = Join-Path $projectRoot "Modules\PC-AI.Evaluation\Datasets\pcai-diagnostic-eval.json"

            if (Test-Path $datasetPath) {
                $dataset = Get-EvaluationDataset -Name $datasetPath
                $dataset | Should -Not -BeNullOrEmpty
                $dataset.Count | Should -BeGreaterThan 0
            }
        }
    }

    Context "Similarity Metrics" {
        It "Should calculate similarity between identical strings" {
            $sim = Compare-ResponseSimilarity -Response "Hello world" -Expected "Hello world"

            $sim | Should -BeGreaterThan 0.9
        }

        It "Should calculate lower similarity for different strings" {
            $sim = Compare-ResponseSimilarity -Response "The quick brown fox" -Expected "A lazy dog sleeps"

            $sim | Should -BeLessThan 0.5
        }

        It "Should handle empty strings" {
            $sim = Compare-ResponseSimilarity -Response "" -Expected ""

            $sim | Should -BeOfType [double]
        }

        It "Should calculate reasonable similarity for paraphrased content" {
            $response = "Paris is the capital city of France"
            $expected = "The capital of France is Paris"

            $sim = Compare-ResponseSimilarity -Response $response -Expected $expected

            $sim | Should -BeGreaterThan 0.3
        }
    }

    Context "Coherence Measurement" {
        It "Should give high coherence to well-formed text" {
            $text = "The system diagnostic shows all hardware is functioning properly. No errors were detected. All services are running normally."

            $coherence = Measure-Coherence -Response $text

            $coherence | Should -BeGreaterThan 0.7
        }

        It "Should give lower coherence to repeated text" {
            $text = "Error detected. Error detected. Error detected. Error detected. Error detected."

            $coherence = Measure-Coherence -Response $text

            $coherence | Should -BeGreaterOrEqual 0
            $coherence | Should -BeLessOrEqual 1.0
            $control = Measure-Coherence -Response 'The system diagnostic shows all hardware is functioning properly. No errors were detected. All services are running normally.'
            $coherence | Should -BeLessThan $control
        }
    }

    Context "Toxicity Detection" {
        It "Should detect low toxicity in normal text" {
            $text = "The weather is nice today and I hope you have a great day."

            $toxicity = Measure-Toxicity -Response $text

            $toxicity | Should -BeLessThan 0.2
        }

        It "Should detect higher toxicity in concerning text" {
            # Use words that are in our keyword list
            $text = "I want to kill this process and destroy the attack vector."

            $toxicity = Measure-Toxicity -Response $text

            $toxicity | Should -BeGreaterThan 0
        }
    }

    Context "Diagnostic Quality Evaluation" {
        It "Should validate proper JSON diagnostic output" {
            $output = @{
                findings = @("Disk health is good", "Network connected")
                recommendations = @("No action required")
                priority = "low"
            } | ConvertTo-Json

            $result = Measure-DiagnosticQuality -DiagnosticOutput $output -DiagnosticInput "Check system"

            $result.valid_json | Should -BeTrue
            $result.has_findings | Should -BeTrue
            $result.has_recommendations | Should -BeTrue
            $result.score | Should -BeGreaterThan 0.5
        }

        It "Should flag missing sections" {
            $output = @{
                findings = @("Issue found")
                # Missing recommendations
            } | ConvertTo-Json

            $result = Measure-DiagnosticQuality -DiagnosticOutput $output -DiagnosticInput "Check disk"

            $result.has_recommendations | Should -BeFalse
            $result.issues | Should -Contain "Missing recommendations section"
        }

        It "Should require safety warnings for critical issues" {
            $input = "SMART shows disk failure with bad sectors"
            $output = @{
                findings = @("Disk is failing")
                recommendations = @("Replace disk")
                priority = "critical"
            } | ConvertTo-Json

            $result = Measure-DiagnosticQuality -DiagnosticOutput $output -DiagnosticInput $input

            $result.safety_warnings_present | Should -BeFalse
            $result.issues | Should -Contain "Missing safety warnings for critical issues"
        }
    }

    Context "A/B Testing" {
        It "Should create an A/B test" {
            $test = New-ABTest -Name "test-ab" -VariantAName "ModelA" -VariantBName "ModelB"

            $test | Should -Not -BeNullOrEmpty
            $test.Name | Should -Be "test-ab"
            $test.VariantAName | Should -Be "ModelA"
            $test.VariantBName | Should -Be "ModelB"
        }

        It "Should add results to A/B test" {
            $test = New-ABTest -Name "test-ab-results"

            Add-ABTestResult -TestName "test-ab-results" -Variant "A" -Score 0.8
            Add-ABTestResult -TestName "test-ab-results" -Variant "A" -Score 0.85
            Add-ABTestResult -TestName "test-ab-results" -Variant "B" -Score 0.9
            Add-ABTestResult -TestName "test-ab-results" -Variant "B" -Score 0.92

            # Scores should be added (internal state)
            { Get-ABTestAnalysis -TestName "test-ab-results" } | Should -Not -Throw
        }

        It "Should analyze A/B test with sufficient samples" {
            $test = New-ABTest -Name "test-analysis"

            # Add 10 samples per variant
            1..10 | ForEach-Object { Add-ABTestResult -TestName "test-analysis" -Variant "A" -Score (0.7 + (Get-Random -Minimum 0 -Maximum 10) / 100) }
            1..10 | ForEach-Object { Add-ABTestResult -TestName "test-analysis" -Variant "B" -Score (0.8 + (Get-Random -Minimum 0 -Maximum 10) / 100) }

            $analysis = Get-ABTestAnalysis -TestName "test-analysis"

            $analysis | Should -Not -BeNullOrEmpty
            $analysis.VariantA.Samples | Should -Be 10
            $analysis.VariantB.Samples | Should -Be 10
            $analysis.CohensD | Should -Not -BeNullOrEmpty
            $analysis.EffectSize | Should -BeIn @('negligible', 'small', 'medium', 'large')
        }
    }

    Context "Statistical Helpers" {
        It "Should calculate standard deviation" {
            $values = @(2, 4, 4, 4, 5, 5, 7, 9)

            # Using internal helper - may need adjustment based on module structure
            $stdDev = & (Get-Module PC-AI.Evaluation) { Get-StandardDeviation $args[0] } $values

            # Expected std dev is approximately 2
            $stdDev | Should -BeGreaterThan 1.5
            $stdDev | Should -BeLessThan 2.5
        }

        It "Should calculate median" {
            $odd = @(1, 3, 5, 7, 9)
            $even = @(1, 2, 3, 4)

            $medianOdd = & (Get-Module PC-AI.Evaluation) { Get-Median $args[0] } $odd
            $medianEven = & (Get-Module PC-AI.Evaluation) { Get-Median $args[0] } $even

            $medianOdd | Should -Be 5
            $medianEven | Should -Be 2.5
        }

        It "Should calculate percentile" {
            $values = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

            $p50 = & (Get-Module PC-AI.Evaluation) { Get-Percentile $args[0] $args[1] } $values 50
            $p90 = & (Get-Module PC-AI.Evaluation) { Get-Percentile $args[0] $args[1] } $values 90

            $p50 | Should -Be 5
            $p90 | Should -Be 9
        }
    }
}

Describe 'Evaluation result aggregation' -Tag 'Unit', 'Portable' {
    It 'returns an empty summary and no fabricated metrics when no cases completed' {
        $suite = New-EvaluationSuite -Name 'empty-results' -Metrics @('latency', 'throughput')
        $summary = Get-EvaluationResults -Suite $suite -Format summary
        $summary.TotalTests | Should -Be 0
        $summary.PassRate | Should -Be 0
        $summary.AverageScore | Should -Be 0
        (Get-EvaluationResults -Suite $suite -Format metrics).Count | Should -Be 0
    }

    It 'aggregates mixed outcomes, retains zero metric values and returns only failed cases' {
        $suite = New-EvaluationSuite -Name 'mixed-results' -Metrics @('latency')
        $resultType = $suite.Results.GetType().GetGenericArguments()[0]
        foreach ($case in @(
            @{ Id='successful'; Status='pass'; Value=100.0 },
            @{ Id='failed'; Status='fail'; Value=0.0 },
            @{ Id='errored'; Status='error'; Value=200.0 }
        )) {
            $result = [Activator]::CreateInstance($resultType)
            $result.TestCaseId = $case.Id
            $result.Status = $case.Status
            $result.Response = 'x' * 120
            $result.Duration = [TimeSpan]::FromMilliseconds($case.Value)
            $result.Metrics = @{ latency=$case.Value }
            $suite.Results.Add($result)
        }
        $summary = Get-EvaluationResults -Suite $suite -Format summary
        $summary.TotalTests | Should -Be 3
        $summary.Passed | Should -Be 1
        $summary.Failed | Should -Be 1
        $summary.Errors | Should -Be 1
        $summary.PassRate | Should -Be 33.33
        $summary.AverageLatency | Should -Be 100
        $metrics = Get-EvaluationResults -Suite $suite -Format metrics
        $metrics.latency.Mean | Should -Be 100
        $metrics.latency.Min | Should -Be 0
        $metrics.latency.Max | Should -Be 200
        $failures = @(Get-EvaluationResults -Suite $suite -Format failures)
        $failures.Count | Should -Be 2
        $failures.TestId | Should -Not -Contain 'successful'
        @(Get-EvaluationResults -Suite $suite -Format detailed).Response | Should -Be @((('x' * 100) + '...'), (('x' * 100) + '...'), (('x' * 100) + '...'))
    }
}
