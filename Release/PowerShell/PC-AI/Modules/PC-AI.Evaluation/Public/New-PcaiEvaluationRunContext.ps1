function New-PcaiEvaluationRunContext {
    [CmdletBinding()]
    param(
        [string]$RunLabel,
        [string]$OutputRoot,
        [string]$SuiteName,
        [string]$Backend
    )

    Initialize-EvaluationPaths

    $root = if ($OutputRoot) { $OutputRoot } else { $script:EvaluationConfig.RunRoot }
    if (-not (Test-Path $root)) {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
    }

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $safeLabel = if ($RunLabel) { ($RunLabel -replace '[^a-zA-Z0-9_.-]', '-') } else { $null }
    $runId = if ($safeLabel) { "$timestamp-$safeLabel" } else { $timestamp }
    $runDir = Join-Path $root $runId

    if (-not (Test-Path $runDir)) {
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    }

    return [pscustomobject]@{
        RunId = $runId
        RunDir = $runDir
        SuiteName = $SuiteName
        Backend = $Backend
        ProgressLogPath = Join-Path $runDir 'progress.log'
        EventsLogPath = Join-Path $runDir 'events.jsonl'
        SummaryPath = Join-Path $runDir 'summary.json'
        StopSignalPath = Join-Path $runDir 'stop.signal'
        CreatedUtc = (Get-Date).ToUniversalTime().ToString('o')
    }
}
