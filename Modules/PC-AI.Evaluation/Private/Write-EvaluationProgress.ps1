function Write-EvaluationProgress {
    [CmdletBinding()]
    param(
        [int]$Completed,
        [int]$Total,
        [string]$TestCaseId,
        [timespan]$Elapsed
    )

    $percent = if ($Total -gt 0) { [math]::Round(($Completed / $Total) * 100, 1) } else { 0 }
    $status = "Test {0} of {1}: {2}" -f $Completed, $Total, $TestCaseId

    if ($script:EvaluationConfig.ProgressMode -in @('auto', 'bar')) {
        Write-Progress -Activity "Running Evaluation" -Status $status -PercentComplete $percent
    }

    if ($script:EvaluationConfig.ProgressMode -in @('auto', 'stream')) {
        $interval = [int]($script:EvaluationConfig.ProgressIntervalSeconds ?? 0)
        if ($interval -gt 0 -and $script:EvaluationRunState) {
            $last = $script:EvaluationRunState.LastProgressUtc
            if ($last -and ((Get-Date) - $last).TotalSeconds -lt $interval) {
                return
            }
            $script:EvaluationRunState.LastProgressUtc = Get-Date
        }
        $elapsedText = if ($Elapsed) { $Elapsed.ToString('hh\:mm\:ss') } else { '00:00:00' }
        $line = "progress=$Completed/$Total ($percent%) elapsed=$elapsedText test=$TestCaseId"
        if ($script:EvaluationConfig.ProgressLogPath) {
            $progressDir = Split-Path -Parent $script:EvaluationConfig.ProgressLogPath
            if ($progressDir -and -not (Test-Path $progressDir)) {
                New-Item -ItemType Directory -Path $progressDir -Force | Out-Null
            }
            Add-Content -Path $script:EvaluationConfig.ProgressLogPath -Value $line
        }
        Write-Host "[pcai.eval] $line" -ForegroundColor DarkGray
    }
}
