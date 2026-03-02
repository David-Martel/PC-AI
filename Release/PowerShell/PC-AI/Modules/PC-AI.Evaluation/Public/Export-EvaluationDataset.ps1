function Export-EvaluationDataset {
    [CmdletBinding()]
    param(
        [EvaluationTestCase[]]$TestCases,
        [string]$Path
    )

    $data = $TestCases | ForEach-Object {
        @{
            id = $_.Id
            category = $_.Category
            prompt = $_.Prompt
            expected = $_.ExpectedOutput
            context = $_.Context
            tags = $_.Tags
        }
    }

    $data | ConvertTo-Json -Depth 5 | Set-Content -Path $Path
    Write-Host "Dataset exported: $Path" -ForegroundColor Green
}
