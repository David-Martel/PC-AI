function New-EvaluationSuite {
    <#
    .SYNOPSIS
        Creates a new evaluation suite for testing LLM inference

    .PARAMETER Name
        Name of the evaluation suite

    .PARAMETER Description
        Description of what this suite tests

    .PARAMETER Metrics
        Array of metric names to include: latency, throughput, memory, similarity, groundedness

    .PARAMETER IncludeDefaultMetrics
        Include default performance metrics (latency, throughput, memory)

    .EXAMPLE
        $suite = New-EvaluationSuite -Name "DiagnosticQuality" -Metrics @('latency', 'similarity', 'groundedness')
    #>
    [CmdletBinding()]
    [OutputType([EvaluationSuite])]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [string]$Description = "",

        [ValidateSet('latency', 'throughput', 'memory', 'similarity', 'groundedness', 'accuracy', 'coherence', 'toxicity')]
        [string[]]$Metrics = @('latency', 'similarity'),

        [switch]$IncludeDefaultMetrics
    )

    $suite = [EvaluationSuite]::new()
    $suite.Name = $Name
    $suite.Description = $Description

    # Add requested metrics
    $metricsToAdd = if ($IncludeDefaultMetrics) {
        $Metrics + $script:EvaluationConfig.DefaultMetrics | Select-Object -Unique
    } else { $Metrics }

    foreach ($metricName in $metricsToAdd) {
        $metric = Get-MetricDefinition -Name $metricName
        if ($metric) {
            $suite.AddMetric($metric)
        }
    }

    $script:CurrentSuite = $suite
    return $suite
}
