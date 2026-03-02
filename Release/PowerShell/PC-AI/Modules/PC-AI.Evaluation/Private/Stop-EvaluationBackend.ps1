function Stop-EvaluationBackend {
    [CmdletBinding()]
    param(
        [string]$Backend
    )

    if ($script:CompiledServerProcess) {
        try {
            if (-not $script:CompiledServerProcess.HasExited) {
                $script:CompiledServerProcess.CloseMainWindow() | Out-Null
                Start-Sleep -Seconds 2
                if (-not $script:CompiledServerProcess.HasExited) {
                    Stop-Process -Id $script:CompiledServerProcess.Id -Force -ErrorAction SilentlyContinue
                }
            }
        } catch { }
    }

    if ($script:CompiledServerConfigPath -and (Test-Path $script:CompiledServerConfigPath)) {
        Remove-Item $script:CompiledServerConfigPath -Force -ErrorAction SilentlyContinue
    }

    $script:CompiledServerProcess = $null
    $script:CompiledServerConfigPath = $null
    $script:CompiledServerBaseUrl = $null
    $script:CompiledServerBackend = $null

    if ($Backend -in @('llamacpp', 'mistralrs')) {
        try {
            if (Get-Command Close-PcaiInference -ErrorAction SilentlyContinue) {
                Close-PcaiInference -ErrorAction SilentlyContinue
            } elseif (Get-Command Stop-PcaiInference -ErrorAction SilentlyContinue) {
                Stop-PcaiInference -ErrorAction SilentlyContinue
            }
        } catch { }
    }

    Write-EvaluationEvent -Type 'backend_stop' -Message "Backend stopped: $Backend" -Data @{
        backend = $Backend
    }
}
