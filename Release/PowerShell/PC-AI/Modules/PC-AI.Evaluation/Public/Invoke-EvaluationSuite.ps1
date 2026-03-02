function Invoke-EvaluationSuite {
    <#
    .SYNOPSIS
        Runs the evaluation suite against specified backend

    .PARAMETER Suite
        The evaluation suite to run

    .PARAMETER Backend
        Inference backend: 'llamacpp', 'mistralrs', 'http', 'ollama'

    .PARAMETER ModelPath
        Path to model file (for native backends)

    .PARAMETER BaseUrl
        API base URL (for HTTP backends)

    .PARAMETER Parallel
        Run test cases in parallel

    .PARAMETER RunLabel
        Label for the evaluation run (used in output folder naming)

    .PARAMETER OutputRoot
        Root folder for evaluation run outputs (defaults to .pcai/evaluation/runs)

    .PARAMETER ProgressMode
        Progress output mode: auto, stream, bar, silent

    .PARAMETER EmitStructuredMessages
        Emit JSON event lines to the pipeline for LLM-friendly parsing

    .PARAMETER HeartbeatSeconds
        Interval for heartbeat status events

    .PARAMETER RequestTimeoutSec
        Timeout for HTTP requests per test case

    .PARAMETER StopSignalPath
        Stop signal file path; if present, evaluation will stop gracefully

    .PARAMETER RunContext
        Pre-created run context (output paths + identifiers)

    .EXAMPLE
        $results = Invoke-EvaluationSuite -Suite $suite -Backend 'llamacpp' -ModelPath "C:\models\llama-3.2-1b.gguf"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [EvaluationSuite]$Suite,

        [Parameter(Mandatory)]
        [ValidateSet('llamacpp', 'mistralrs', 'llamacpp-bin', 'mistralrs-bin', 'http', 'ollama')]
        [string]$Backend,

        [string]$ModelPath,

        [string]$BaseUrl = "http://127.0.0.1:8080",

        [int]$GpuLayers = -1,

        [switch]$Parallel,

        [int]$MaxTokens = 512,

        [float]$Temperature = 0.7,

        [string]$RunLabel,

        [string]$OutputRoot,

        [ValidateSet('auto', 'stream', 'bar', 'silent')]
        [string]$ProgressMode = 'auto',

        [switch]$EmitStructuredMessages,

        [int]$HeartbeatSeconds = 15,

        [int]$RequestTimeoutSec = 120,

        [string]$StopSignalPath,

        [pscustomobject]$RunContext
    )

    Write-Host "Starting evaluation suite: $($Suite.Name)" -ForegroundColor Cyan
    Write-Host "Backend: $Backend | Test cases: $($Suite.TestCases.Count)" -ForegroundColor Gray
    Initialize-EvaluationPaths

    if (-not $RunContext) {
        $RunContext = New-PcaiEvaluationRunContext -RunLabel $RunLabel -OutputRoot $OutputRoot -SuiteName $Suite.Name -Backend $Backend
    }

    $script:EvaluationConfig.ProgressMode = $ProgressMode
    $script:EvaluationConfig.EmitStructuredMessages = [bool]$EmitStructuredMessages
    $script:EvaluationConfig.HeartbeatSeconds = $HeartbeatSeconds
    $script:EvaluationConfig.ProgressLogPath = $RunContext.ProgressLogPath
    $script:EvaluationConfig.EventsLogPath = $RunContext.EventsLogPath
    $script:EvaluationConfig.StopSignalPath = if ($StopSignalPath) { $StopSignalPath } else { $RunContext.StopSignalPath }

    $script:EvaluationRunState = [ordered]@{
        RunId = $RunContext.RunId
        RunDir = $RunContext.RunDir
        Suite = $Suite.Name
        Backend = $Backend
        StartTimeUtc = (Get-Date).ToUniversalTime().ToString('o')
        TotalTests = $Suite.TestCases.Count
        Completed = 0
        Cancelled = $false
        LastProgressUtc = $null
    }
    if ($Backend -eq 'ollama' -and $BaseUrl -eq 'http://127.0.0.1:8080') {
        $BaseUrl = 'http://127.0.0.1:11434'
    }

    $summary = $null

    try {
        Write-EvaluationEvent -Type 'start' -Message "Evaluation started: $($Suite.Name)" -Data @{
            backend = $Backend
            runId = $RunContext.RunId
            runDir = $RunContext.RunDir
            stopSignal = $script:EvaluationConfig.StopSignalPath
            totalTests = $Suite.TestCases.Count
        }
        # Initialize backend
        $backendReady = Initialize-EvaluationBackend -Backend $Backend -ModelPath $ModelPath -BaseUrl $BaseUrl -GpuLayers $GpuLayers

        if (-not $backendReady) {
            Write-Warning "Failed to initialize backend: $Backend"
            Write-EvaluationEvent -Type 'backend_error' -Message "Backend initialization failed: $Backend" -Data @{
                backend = $Backend
            } -Level 'warn'
            return $null
        }

        $startTime = Get-Date
        $lastHeartbeat = Get-Date

        # Run test cases
        $testCases = $Suite.TestCases
        $progress = 0

        foreach ($testCase in $testCases) {
            $progress++
            if (Test-EvaluationStopSignal) {
                $script:EvaluationRunState.Cancelled = $true
                Write-EvaluationEvent -Type 'cancel' -Message "Stop signal detected. Cancelling evaluation run." -Data @{
                    runId = $RunContext.RunId
                    completed = $progress - 1
                } -Level 'warn'
                break
            }

            Write-EvaluationEvent -Type 'test_start' -Message "Running test case $($testCase.Id)" -Data @{
                index = $progress
                total = $testCases.Count
                testCaseId = $testCase.Id
            }

            $result = Invoke-SingleTestCase -TestCase $testCase -Backend $Backend -MaxTokens $MaxTokens -Temperature $Temperature -RequestTimeoutSec $RequestTimeoutSec

            # Calculate metrics
            foreach ($metric in $Suite.Metrics) {
                try {
                    $metricValue = & $metric.Calculator $result
                    $result.Metrics[$metric.Name] = $metricValue
                } catch {
                    Write-Warning "Failed to calculate metric $($metric.Name): $_"
                    Write-EvaluationEvent -Type 'metric_error' -Message "Metric calculation failed: $($metric.Name)" -Data @{
                        testCaseId = $testCase.Id
                        error = $_.Exception.Message
                    } -Level 'warn'
                    $result.Metrics[$metric.Name] = $null
                }
            }

            # Calculate overall score (weighted average of normalized metrics)
            $result.OverallScore = Calculate-OverallScore -Result $result -Metrics $Suite.Metrics

            # Determine pass/fail
            $result.Status = if ($result.ErrorMessage) { 'error' }
                             elseif ($result.OverallScore -ge 0.7) { 'pass' }
                             else { 'fail' }

            $Suite.Results.Add($result)

            $script:EvaluationRunState.Completed = $progress
            Write-EvaluationProgress -Completed $progress -Total $testCases.Count -TestCaseId $testCase.Id -Elapsed ((Get-Date) - $startTime)
            Write-EvaluationEvent -Type 'test_complete' -Message "Completed test case $($testCase.Id)" -Data @{
                index = $progress
                total = $testCases.Count
                status = $result.Status
                duration_ms = [math]::Round($result.Duration.TotalMilliseconds, 2)
            }

            if ($HeartbeatSeconds -gt 0 -and ((Get-Date) - $lastHeartbeat).TotalSeconds -ge $HeartbeatSeconds) {
                $lastHeartbeat = Get-Date
                Write-EvaluationEvent -Type 'heartbeat' -Message "Evaluation heartbeat" -Data @{
                    completed = $progress
                    total = $testCases.Count
                }
            }
        }

        if ($script:EvaluationConfig.ProgressMode -in @('auto', 'bar')) {
            Write-Progress -Activity "Running Evaluation" -Completed
        }

        $endTime = Get-Date
        $totalDuration = $endTime - $startTime

        # Generate summary
        $summary = $Suite.GetSummary()
        $summary.TotalDuration = $totalDuration
        $summary.Backend = $Backend
        $summary.Model = $ModelPath ?? $BaseUrl
        $summary.Cancelled = $script:EvaluationRunState.Cancelled

        if ($RunContext -and $RunContext.SummaryPath) {
            $summary | ConvertTo-Json -Depth 6 | Set-Content -Path $RunContext.SummaryPath
        }

        Write-Host "`nEvaluation Complete" -ForegroundColor Green
        Write-Host "  Pass Rate: $($summary.PassRate)%" -ForegroundColor $(if ($summary.PassRate -ge 80) { 'Green' } elseif ($summary.PassRate -ge 60) { 'Yellow' } else { 'Red' })
        Write-Host "  Average Score: $($summary.AverageScore)" -ForegroundColor Gray
        Write-Host "  Average Latency: $([math]::Round($summary.AverageLatency, 2))ms" -ForegroundColor Gray
        Write-Host "  Total Duration: $($totalDuration.ToString('mm\:ss\.fff'))" -ForegroundColor Gray
        Write-Host "  Run Dir: $($RunContext.RunDir)" -ForegroundColor DarkGray
        if ($summary.Cancelled) {
            Write-Host "  Status: CANCELLED" -ForegroundColor Yellow
        }

        Write-EvaluationEvent -Type 'complete' -Message "Evaluation complete" -Data @{
            runId = $RunContext.RunId
            totalDurationSec = [math]::Round($totalDuration.TotalSeconds, 2)
            passRate = $summary.PassRate
            cancelled = $summary.Cancelled
        }

        return $summary
    } finally {
        Stop-EvaluationBackend -Backend $Backend
    }
}
