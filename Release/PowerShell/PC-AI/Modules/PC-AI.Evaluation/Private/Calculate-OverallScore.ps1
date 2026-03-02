function Calculate-OverallScore {
    [CmdletBinding()]
    param(
        [EvaluationResult]$Result,
        [EvaluationMetric[]]$Metrics
    )

    $totalWeight = 0
    $weightedSum = 0

    foreach ($metric in $Metrics) {
        $value = $Result.Metrics[$metric.Name]
        if ($null -eq $value) { continue }

        # Normalize metric value to 0-1 range
        $normalized = switch ($metric.Name) {
            'latency' {
                # Lower is better, normalize: 0-5000ms -> 1-0
                [math]::Max(0, 1 - ($value / 5000))
            }
            'throughput' {
                # Higher is better, normalize: 0-100 tps -> 0-1
                [math]::Min(1, $value / 100)
            }
            'memory' {
                # Lower is better, normalize: 0-1000MB -> 1-0
                [math]::Max(0, 1 - ($value / 1000))
            }
            'toxicity' {
                # Lower is better (inverted)
                1 - $value
            }
            default {
                # Assume 0-1 range for other metrics
                [math]::Max(0, [math]::Min(1, $value))
            }
        }

        $weightedSum += $normalized * $metric.Weight
        $totalWeight += $metric.Weight
    }

    if ($totalWeight -gt 0) {
        return [math]::Round($weightedSum / $totalWeight, 4)
    }
    return 0
}
