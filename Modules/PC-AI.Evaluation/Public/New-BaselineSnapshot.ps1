function New-BaselineSnapshot {
    <#
    .SYNOPSIS
        Creates a baseline snapshot of current model performance

    .PARAMETER Name
        Name for this baseline

    .PARAMETER Suite
        Evaluation suite to use

    .PARAMETER Backend
        Inference backend

    .PARAMETER ModelPath
        Path to model file
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [EvaluationSuite]$Suite,

        [string]$Backend = 'llamacpp',

        [string]$ModelPath
    )

    Write-Host "Creating baseline snapshot: $Name" -ForegroundColor Cyan

    # Run evaluation
    $results = Invoke-EvaluationSuite -Suite $Suite -Backend $Backend -ModelPath $ModelPath

    # Create baseline object
    $baseline = @{
        Name = $Name
        Timestamp = [datetime]::UtcNow.ToString('o')
        Backend = $Backend
        Model = $ModelPath
        Metrics = $results
        TestCount = $Suite.Results.Count
        DetailedResults = $Suite.Results | ForEach-Object {
            @{
                TestId = $_.TestCaseId
                Score = $_.OverallScore
                Metrics = $_.Metrics
            }
        }
    }

    # Save baseline
    $baselinePath = Join-Path $script:EvaluationConfig.BaselinePath "$Name.json"
    $baselineDir = Split-Path $baselinePath -Parent
    if (-not (Test-Path $baselineDir)) {
        New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
    }

    $baseline | ConvertTo-Json -Depth 10 | Set-Content -Path $baselinePath

    $script:Baselines[$Name] = $baseline

    Write-Host "Baseline saved: $baselinePath" -ForegroundColor Green

    return $baseline
}
