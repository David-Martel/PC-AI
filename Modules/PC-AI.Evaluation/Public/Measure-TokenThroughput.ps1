function Measure-TokenThroughput {
    <#
    .SYNOPSIS
        Measures token generation throughput

    .PARAMETER Prompt
        Test prompt to use

    .PARAMETER TargetTokens
        Target number of tokens to generate

    .PARAMETER Iterations
        Number of iterations for measurement
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Prompt,

        [int]$TargetTokens = 256,

        [int]$Iterations = 5
    )

    $throughputs = [List[double]]::new()

    for ($i = 0; $i -lt $Iterations; $i++) {
        $sw = [Stopwatch]::StartNew()
        try {
            $response = Invoke-PcaiGenerate -Prompt $Prompt -MaxTokens $TargetTokens
            $sw.Stop()

            # Estimate token count (rough: ~4 chars per token)
            $estimatedTokens = [math]::Ceiling($response.Length / 4)
            $tokensPerSecond = $estimatedTokens / $sw.Elapsed.TotalSeconds

            $throughputs.Add($tokensPerSecond)
        } catch {
            Write-Warning "Iteration $i failed: $_"
        }
    }

    if ($throughputs.Count -eq 0) {
        return @{ Error = "All iterations failed" }
    }

    return @{
        Mean = [math]::Round(($throughputs | Measure-Object -Average).Average, 2)
        Min = [math]::Round(($throughputs | Measure-Object -Minimum).Minimum, 2)
        Max = [math]::Round(($throughputs | Measure-Object -Maximum).Maximum, 2)
        Unit = "tokens/second"
        Samples = $throughputs.Count
    }
}
