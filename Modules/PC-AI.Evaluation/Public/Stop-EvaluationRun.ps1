function Stop-EvaluationRun {
    [CmdletBinding()]
    param(
        [string]$StopSignalPath
    )

    $path = if ($StopSignalPath) { $StopSignalPath } else { $script:EvaluationConfig.StopSignalPath }
    if (-not $path) {
        Write-Warning "No stop signal path configured."
        return $false
    }

    $dir = Split-Path -Parent $path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    New-Item -ItemType File -Path $path -Force | Out-Null
    Write-EvaluationEvent -Type 'cancel_requested' -Message "Stop signal created: $path" -Data @{
        stopSignal = $path
    } -Level 'warn'
    return $true
}
