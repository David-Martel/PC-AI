#Requires -Version 7.0
<#
.SYNOPSIS
    Benchmarks command performance using hyperfine or native measurement

.DESCRIPTION
    Uses hyperfine (Rust benchmarking tool) when available for accurate
    statistical benchmarking. Falls back to Measure-Command with multiple
    iterations for timing.

.PARAMETER Command
    Command string or scriptblock to benchmark

.PARAMETER Iterations
    Number of iterations (default: 10)

.PARAMETER Warmup
    Number of warmup runs before measurement (default: 3)

.PARAMETER Name
    Name/label for the benchmark

.PARAMETER Shell
    Shell to use for command execution

.EXAMPLE
    Measure-CommandPerformance -Command "Get-Process" -Iterations 20
    Benchmarks Get-Process with 20 iterations

.EXAMPLE
    Measure-CommandPerformance -Command { Get-ChildItem C:\ -Recurse -Depth 2 } -Warmup 5
    Benchmarks with 5 warmup runs

.OUTPUTS
    PSCustomObject with benchmark statistics
#>
function Measure-CommandPerformance {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [object]$Command,

        [Parameter()]
        [int]$Iterations = 10,

        [Parameter()]
        [int]$Warmup = 3,

        [Parameter()]
        [string]$Name = 'Benchmark',

        [Parameter()]
        [ValidateSet('pwsh', 'powershell', 'cmd', 'bash')]
        [string]$Shell = 'pwsh'
    )

    $hyperfine = Get-RustToolPath -ToolName 'hyperfine'
    $useHyperfine = $null -ne $hyperfine -and (Test-Path $hyperfine) -and ($Command -is [string])

    if ($useHyperfine) {
        return Measure-WithHyperfine -Command $Command -Iterations $Iterations -Warmup $Warmup -Name $Name -Shell $Shell -HyperfinePath $hyperfine
    }
    else {
        return Measure-WithNative -Command $Command -Iterations $Iterations -Warmup $Warmup -Name $Name
    }
}

function Measure-WithHyperfine {
    [CmdletBinding()]
    param(
        [string]$Command,
        [int]$Iterations,
        [int]$Warmup,
        [string]$Name,
        [string]$Shell,
        [string]$HyperfinePath
    )

    $args = @(
        '--runs', $Iterations.ToString(),
        '--warmup', $Warmup.ToString(),
        '--export-json', '-',
        '--shell', $Shell
    )

    $args += $Command

    try {
        $output = & $HyperfinePath @args 2>&1

        # Parse JSON output
        $jsonStart = $output.IndexOf('{')
        if ($jsonStart -ge 0) {
            $jsonStr = ($output[$jsonStart..($output.Count - 1)]) -join ''
            $json = $jsonStr | ConvertFrom-Json

            $result = $json.results[0]

            return [PSCustomObject]@{
                Name       = $Name
                Command    = $Command
                Mean       = [Math]::Round($result.mean * 1000, 2)  # Convert to ms
                StdDev     = [Math]::Round($result.stddev * 1000, 2)
                Min        = [Math]::Round($result.min * 1000, 2)
                Max        = [Math]::Round($result.max * 1000, 2)
                Median     = [Math]::Round($result.median * 1000, 2)
                Iterations = $Iterations
                Warmup     = $Warmup
                Unit       = 'ms'
                Tool       = 'hyperfine'
            }
        }
    }
    catch {
        Write-Warning "hyperfine failed: $_"
    }

    # Fallback to native
    return Measure-WithNative -Command ([scriptblock]::Create($Command)) -Iterations $Iterations -Warmup $Warmup -Name $Name
}

function Measure-WithNative {
    [CmdletBinding()]
    param(
        [object]$Command,
        [int]$Iterations,
        [int]$Warmup,
        [string]$Name
    )

    $scriptBlock = if ($Command -is [scriptblock]) {
        $Command
    }
    else {
        [scriptblock]::Create($Command.ToString())
    }

    # Warmup runs
    for ($i = 0; $i -lt $Warmup; $i++) {
        try {
            $null = & $scriptBlock
        }
        catch {
            # Ignore warmup errors
        }
    }

    # Measurement runs
    $times = [System.Collections.Generic.List[double]]::new()

    for ($i = 0; $i -lt $Iterations; $i++) {
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $null = & $scriptBlock
            $sw.Stop()
            $times.Add($sw.Elapsed.TotalMilliseconds)
        }
        catch {
            Write-Warning "Iteration $i failed: $_"
        }
    }

    if ($times.Count -eq 0) {
        Write-Error "All iterations failed"
        return $null
    }

    $sortedTimes = $times | Sort-Object
    $mean = ($times | Measure-Object -Average).Average
    $sum = ($times | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Sum).Sum
    $stdDev = [Math]::Sqrt($sum / $times.Count)

    $medianIndex = [Math]::Floor($sortedTimes.Count / 2)
    $median = if ($sortedTimes.Count % 2 -eq 0) {
        ($sortedTimes[$medianIndex - 1] + $sortedTimes[$medianIndex]) / 2
    }
    else {
        $sortedTimes[$medianIndex]
    }

    return [PSCustomObject]@{
        Name       = $Name
        Command    = $Command.ToString()
        Mean       = [Math]::Round($mean, 2)
        StdDev     = [Math]::Round($stdDev, 2)
        Min        = [Math]::Round($sortedTimes[0], 2)
        Max        = [Math]::Round($sortedTimes[-1], 2)
        Median     = [Math]::Round($median, 2)
        Iterations = $times.Count
        Warmup     = $Warmup
        Unit       = 'ms'
        Tool       = 'Measure-Command'
    }
}
