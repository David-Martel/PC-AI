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

    $safeLabel = if ($RunLabel) { ($RunLabel -replace '[^a-zA-Z0-9_.-]', '-') } else { 'evaluation' }
    $runId = "$safeLabel-$([guid]::NewGuid().ToString('N'))"
    $runDir = Join-Path $root $runId

    # Reserve a new directory; a collision must never reuse another run's files.
    New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

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
