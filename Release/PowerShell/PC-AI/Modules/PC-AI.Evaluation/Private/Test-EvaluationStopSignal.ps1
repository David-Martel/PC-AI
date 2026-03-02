function Test-EvaluationStopSignal {
    if ($script:EvaluationConfig.StopSignalPath -and (Test-Path $script:EvaluationConfig.StopSignalPath)) {
        return $true
    }
    return $false
}
