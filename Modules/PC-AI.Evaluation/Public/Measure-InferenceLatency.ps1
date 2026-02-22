function Measure-InferenceLatency {
    <#
    .SYNOPSIS
        Measures inference latency over multiple runs

    .PARAMETER Prompt
        Test prompt to use

    .PARAMETER Iterations
        Number of iterations for measurement

    .PARAMETER WarmupRuns
        Number of warmup runs before measurement

    .EXAMPLE
        $latency = Measure-InferenceLatency -Prompt "Hello world" -Iterations 10
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Prompt,

        [int]$Iterations = 10,

        [int]$WarmupRuns = 2,

        [int]$MaxTokens = 128
    )

    $latencies = [List[double]]::new()

    # Warmup
    for ($i = 0; $i -lt $WarmupRuns; $i++) {
        try {
            $null = Invoke-PcaiGenerate -Prompt $Prompt -MaxTokens $MaxTokens
        } catch {
            Write-Warning "Warmup run $i failed: $_"
        }
    }

    # Measure
    for ($i = 0; $i -lt $Iterations; $i++) {
        $sw = [Stopwatch]::StartNew()
        try {
            $null = Invoke-PcaiGenerate -Prompt $Prompt -MaxTokens $MaxTokens
            $sw.Stop()
            $latencies.Add($sw.Elapsed.TotalMilliseconds)
        } catch {
            Write-Warning "Iteration $i failed: $_"
        }
    }

    if ($latencies.Count -eq 0) {
        return @{ Error = "All iterations failed" }
    }

    return @{
        Mean = [math]::Round(($latencies | Measure-Object -Average).Average, 2)
        Median = [math]::Round((Get-Median $latencies), 2)
        Min = [math]::Round(($latencies | Measure-Object -Minimum).Minimum, 2)
        Max = [math]::Round(($latencies | Measure-Object -Maximum).Maximum, 2)
        StdDev = [math]::Round((Get-StandardDeviation $latencies), 2)
        P95 = [math]::Round((Get-Percentile $latencies 95), 2)
        P99 = [math]::Round((Get-Percentile $latencies 99), 2)
        Samples = $latencies.Count
    }
}
