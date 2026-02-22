function Get-EvaluationResults {
    <#
    .SYNOPSIS
        Gets results from the current or specified evaluation suite
    #>
    [CmdletBinding()]
    param(
        [EvaluationSuite]$Suite = $script:CurrentSuite,

        [ValidateSet('summary', 'detailed', 'metrics', 'failures')]
        [string]$Format = 'summary'
    )

    if (-not $Suite) {
        Write-Error "No evaluation suite available. Run New-EvaluationSuite first."
        return
    }

    switch ($Format) {
        'summary' {
            return $Suite.GetSummary()
        }
        'detailed' {
            return $Suite.Results | ForEach-Object {
                [PSCustomObject]@{
                    TestId = $_.TestCaseId
                    Status = $_.Status
                    Score = $_.OverallScore
                    Latency = "$([math]::Round($_.Duration.TotalMilliseconds, 2))ms"
                    Response = $_.Response.Substring(0, [math]::Min(100, $_.Response.Length)) + "..."
                    Metrics = $_.Metrics
                }
            }
        }
        'metrics' {
            $aggregated = @{}
            foreach ($metric in $Suite.Metrics) {
                $values = $Suite.Results | ForEach-Object { $_.Metrics[$metric.Name] } | Where-Object { $null -ne $_ }
                if ($values.Count -gt 0) {
                    $aggregated[$metric.Name] = @{
                        Mean = [math]::Round(($values | Measure-Object -Average).Average, 4)
                        Min = [math]::Round(($values | Measure-Object -Minimum).Minimum, 4)
                        Max = [math]::Round(($values | Measure-Object -Maximum).Maximum, 4)
                        StdDev = [math]::Round((Get-StandardDeviation $values), 4)
                    }
                }
            }
            return $aggregated
        }
        'failures' {
            return $Suite.Results | Where-Object { $_.Status -ne 'pass' } | ForEach-Object {
                [PSCustomObject]@{
                    TestId = $_.TestCaseId
                    Status = $_.Status
                    Score = $_.OverallScore
                    Error = $_.ErrorMessage
                    Response = $_.Response
                }
            }
        }
    }
}
