function Get-MetricDefinition {
    [CmdletBinding()]
    param([string]$Name)

    switch ($Name) {
        'latency' {
            [EvaluationMetric]::new('latency', 'Response generation time in milliseconds', {
                param($result)
                return $result.Duration.TotalMilliseconds
            })
        }
        'throughput' {
            [EvaluationMetric]::new('throughput', 'Tokens per second', {
                param($result)
                $tokens = ($result.Response -split '\s+').Count
                $seconds = $result.Duration.TotalSeconds
                if ($seconds -gt 0) {
                    return [math]::Round($tokens / $seconds, 2)
                }
                return 0
            })
        }
        'memory' {
            [EvaluationMetric]::new('memory', 'Memory usage in MB', {
                param($result)
                return $result.Metrics['memory_mb'] ?? 0
            })
        }
        'similarity' {
            [EvaluationMetric]::new('similarity', 'Semantic similarity to expected output (0-1)', {
                param($result)
                if (-not $result.Metrics['expected']) { return 1.0 }
                return Compare-ResponseSimilarity -Response $result.Response -Expected $result.Metrics['expected']
            })
        }
        'groundedness' {
            [EvaluationMetric]::new('groundedness', 'Response grounded in provided context (0-1)', {
                param($result)
                if (-not $result.Metrics['context']) { return 1.0 }
                return Measure-Groundedness -Response $result.Response -Context $result.Metrics['context']
            })
        }
        'accuracy' {
            [EvaluationMetric]::new('accuracy', 'Factual accuracy score (0-1)', {
                param($result)
                return $result.Metrics['accuracy'] ?? 0
            })
        }
        'coherence' {
            [EvaluationMetric]::new('coherence', 'Logical coherence score (0-1)', {
                param($result)
                return Measure-Coherence -Response $result.Response
            })
        }
        'toxicity' {
            [EvaluationMetric]::new('toxicity', 'Toxicity score (0=safe, 1=toxic)', {
                param($result)
                return Measure-Toxicity -Response $result.Response
            })
        }
        default { $null }
    }
}
