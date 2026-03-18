function Get-SafetyTestCases {
    $datasetPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Datasets' 'pcai-safety-eval.json'
    return Import-EvaluationDataset -Path $datasetPath
}
