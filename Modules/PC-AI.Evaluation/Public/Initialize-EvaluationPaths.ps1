function Initialize-EvaluationPaths {
    $artifactsRoot = Get-PcaiArtifactsRoot
    $evalRoot = Join-Path $artifactsRoot 'evaluation'
    $runRoot = Join-Path $evalRoot 'runs'
    $resultsRoot = Join-Path $evalRoot 'results'

    foreach ($path in @($evalRoot, $runRoot, $resultsRoot)) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }
    }

    $script:EvaluationConfig.ArtifactsRoot = $artifactsRoot
    $script:EvaluationConfig.EvaluationRoot = $evalRoot
    $script:EvaluationConfig.RunRoot = $runRoot
    $script:EvaluationConfig.ResultsPath = $resultsRoot
}
