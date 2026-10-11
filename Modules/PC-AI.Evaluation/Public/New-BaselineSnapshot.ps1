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

    if ([string]::IsNullOrWhiteSpace($Name) -or $Name.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
        $Name -ne $Name.TrimEnd(' ','.') -or $Name -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
        throw 'Baseline name must be an ordinary filename leaf.'
    }
    Write-Host "Creating baseline snapshot: $Name" -ForegroundColor Cyan

    # Run evaluation
    $summary = Invoke-EvaluationSuite -Suite $Suite -Backend $Backend -ModelPath $ModelPath

    # A reference may have low scores, but must contain completed finite evidence.
    if ($summary -isnot [Collections.IDictionary] -or $Suite.Results.Count -eq 0 -or $Suite.Metrics.Count -eq 0 -or
        $summary.TotalTests -ne $Suite.Results.Count -or $summary.Cancelled -or
        ($Suite.TestCases.Count -gt 0 -and $Suite.Results.Count -ne $Suite.TestCases.Count)) {
        throw 'Baseline requires a nonempty completed evaluation.'
    }
    foreach ($result in $Suite.Results) {
        if ($result.Status -notin @('pass','fail') -or -not [double]::IsFinite($result.OverallScore)) {
            throw 'Baseline cannot contain failed execution or nonfinite scores.'
        }
        foreach ($metric in $Suite.Metrics) {
            $value=$result.Metrics[$metric.Name]
            if ($null -eq $value -or $value -is [bool] -or $value -isnot [ValueType] -or -not [double]::IsFinite([double]$value)) {
                throw "Baseline requires finite numeric observations for $($metric.Name)."
            }
        }
    }
    $metrics=Get-EvaluationResults -Suite $Suite -Format metrics
    foreach ($metric in $Suite.Metrics) {
        if (-not $metrics.ContainsKey($metric.Name) -or -not [double]::IsFinite([double]$metrics[$metric.Name].Mean)) {
            throw 'Baseline metric aggregation is incomplete or nonfinite.'
        }
    }

    # Create baseline object
    $baseline = @{
        Name = $Name
        Timestamp = [datetime]::UtcNow.ToString('o')
        Backend = $Backend
        Model = $ModelPath
        Metrics = $metrics
        Summary = $summary
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
    if (-not (Test-Path -LiteralPath $baselineDir)) {
        New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
    }

    $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath

    $script:Baselines[$Name] = $baseline

    Write-Host "Baseline saved: $baselinePath" -ForegroundColor Green

    return $baseline
}
