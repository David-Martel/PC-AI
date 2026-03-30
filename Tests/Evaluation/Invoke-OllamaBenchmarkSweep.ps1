#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ConfigPath,
    [string]$OutputRoot,
    [string[]]$Models,
    [string[]]$Profiles,
    [switch]$Quick,
    [switch]$SkipToolEval,
    [switch]$SkipInferenceEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot 'Config\pcai-ollama-benchmark.json'
}
$ConfigPath = (Resolve-Path $ConfigPath).Path
$configDir = Split-Path -Parent $ConfigPath

function Resolve-BenchmarkPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return [System.IO.Path]::GetFullPath((Join-Path $configDir $Path))
}

function Get-ObjectValue {
    param(
        $Object,
        [string]$Name
    )

    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) {
        return $Object[$Name]
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($property) {
        return $property.Value
    }

    return $null
}

function Get-ModelSlug {
    param([string]$Value)
    return (($Value -replace '[^a-zA-Z0-9]+', '-') -replace '(^-+|-+$)', '').ToLowerInvariant()
}

function Get-CandidatePaths {
    param([string]$HuggingFaceId)

    $escaped = $HuggingFaceId -replace '/', '--'
    return @(
        (Join-Path $HOME ".cache\huggingface\hub\models--$escaped"),
        (Join-Path $HOME ".cache\huggingface\hub\models--$escaped*"),
        (Join-Path $HOME ".lmstudio\models\$escaped"),
        (Join-Path $HOME ".lmstudio\models\*$escaped*"),
        (Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--$escaped"),
        (Join-Path $env:USERPROFILE ".lmstudio\models\$escaped")
    ) | Select-Object -Unique
}

function Test-ExpectedArgumentsMatch {
    param(
        $Expected,
        $Actual
    )

    if ($null -eq $Expected) { return $true }
    if ($null -eq $Actual) { return $false }

    foreach ($prop in $Expected.PSObject.Properties) {
        $actualProp = $Actual.PSObject.Properties[$prop.Name]
        if (-not $actualProp) { return $false }
        if ([string]$actualProp.Value -ne [string]$prop.Value) { return $false }
    }

    return $true
}

function Get-InferenceSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return $null }
    $payload = Get-Content -Raw $Path | ConvertFrom-Json -Depth 20
    foreach ($entry in @($payload.Results)) {
        if ($entry.Key -eq 'ollama') {
            return $entry.Value
        }
    }
    return $null
}

$benchmarkConfig = Get-Content -Raw $ConfigPath | ConvertFrom-Json -Depth 20 -AsHashtable
if (-not $OutputRoot) {
    $OutputRoot = Resolve-BenchmarkPath -Path (Get-ObjectValue -Object $benchmarkConfig -Name 'outputRoot')
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runRoot = Join-Path $OutputRoot $timestamp
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

Import-Module (Join-Path $repoRoot 'Modules\PC-AI.Evaluation\PC-AI.Evaluation.psd1') -Force
Import-Module (Join-Path $repoRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1') -Force

$status = Get-LLMStatus
$availableOllamaModels = @($status.Ollama.AvailableModels | ForEach-Object { $_.Name })

#region Preflight GPU Readiness Check

$script:PreflightResult = $null
if (Get-Command Test-PcaiGpuReadiness -ErrorAction SilentlyContinue) {
    $script:PreflightResult = Test-PcaiGpuReadiness -RequiredMB 2000  # 2GB minimum for any inference
    Write-Host "[Preflight] GPU: $($script:PreflightResult.Verdict) — $($script:PreflightResult.Reason)"
    if ($script:PreflightResult.Verdict -eq 'fail') {
        Write-Warning "Preflight VRAM check failed. Inference may OOM."
        if ($script:PreflightResult.Gpus) {
            foreach ($gpu in $script:PreflightResult.Gpus) {
                Write-Warning "  GPU$($gpu.index) ($($gpu.name)): $($gpu.free_mb)MB free / $($gpu.total_mb)MB total"
                foreach ($proc in $gpu.processes | Select-Object -First 3) {
                    Write-Warning "    $($proc.name) ($($proc.pid)): $($proc.used_mb)MB"
                }
            }
        }
    }

    # Write preflight result to run output
    $preflightPath = Join-Path $runRoot 'preflight.json'
    $preflightPayload = [ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString('o')
        verdict   = $script:PreflightResult.Verdict
        reason    = $script:PreflightResult.Reason
        gpus      = $script:PreflightResult.Gpus
    }
    $preflightPayload | ConvertTo-Json -Depth 5 | Set-Content -Path $preflightPath -Encoding UTF8
}

#endregion

$candidateModels = @($benchmarkConfig['candidateModels'])
if ($Models -and $Models.Count -gt 0) {
    $candidateModels = @($candidateModels | Where-Object { (Get-ObjectValue -Object $_ -Name 'name') -in $Models })
}

$profileEntries = @($benchmarkConfig['profiles'])
if ($Profiles -and $Profiles.Count -gt 0) {
    $profileEntries = @($profileEntries | Where-Object { (Get-ObjectValue -Object $_ -Name 'name') -in $Profiles })
}
if ($Quick) {
    $profileEntries = @($profileEntries | Where-Object { (Get-ObjectValue -Object $_ -Name 'name') -in @('fast', 'balanced') })
}

$availability = New-Object System.Collections.Generic.List[object]
$benchmarkRows = New-Object System.Collections.Generic.List[object]
$toolRows = New-Object System.Collections.Generic.List[object]

$toolEvalConfig = Get-ObjectValue -Object $benchmarkConfig -Name 'toolEvaluation'
$scenarioPath = Resolve-BenchmarkPath -Path (Get-ObjectValue -Object $toolEvalConfig -Name 'scenarioPath')
$scenarioItems = @()
if (-not $SkipToolEval -and (Get-ObjectValue -Object $toolEvalConfig -Name 'enabled') -and (Test-Path $scenarioPath)) {
    $scenarioItems = @((Get-Content -Raw $scenarioPath | ConvertFrom-Json -Depth 20).scenarios)
    if ($Quick) {
        $scenarioItems = @($scenarioItems | Select-Object -First ([Math]::Min(4, $scenarioItems.Count)))
    } elseif ((Get-ObjectValue -Object $toolEvalConfig -Name 'maxScenarios') -gt 0) {
        $scenarioItems = @($scenarioItems | Select-Object -First (Get-ObjectValue -Object $toolEvalConfig -Name 'maxScenarios'))
    }
}

foreach ($candidate in $candidateModels) {
    $candidateName = [string](Get-ObjectValue -Object $candidate -Name 'name')
    $candidateRole = [string](Get-ObjectValue -Object $candidate -Name 'role')
    $provider = if (Get-ObjectValue -Object $candidate -Name 'provider') { [string](Get-ObjectValue -Object $candidate -Name 'provider') } else { 'ollama' }
    $isAvailable = $false
    $availabilityHints = @()

    if ($provider -eq 'ollama') {
        $isAvailable = $availableOllamaModels -contains $candidateName
    } elseif ($provider -eq 'huggingface') {
        foreach ($pathHint in Get-CandidatePaths -HuggingFaceId $candidateName) {
            $matches = @(Get-ChildItem -Path $pathHint -ErrorAction SilentlyContinue)
            if ($matches.Count -gt 0) {
                $isAvailable = $true
                $availabilityHints += @($matches | Select-Object -ExpandProperty FullName)
            }
        }
    }

    $availability.Add([pscustomobject]@{
        Name = $candidateName
        Provider = $provider
        Role = $candidateRole
        Available = $isAvailable
        Hints = ($availabilityHints -join '; ')
    })

    if (-not $isAvailable -or $provider -ne 'ollama') {
        continue
    }

    foreach ($profile in $profileEntries) {
        $profileName = [string](Get-ObjectValue -Object $profile -Name 'name')
        $modelSlug = Get-ModelSlug -Value $candidateName
        $profileSlug = Get-ModelSlug -Value $profileName
        $runPrefix = "$modelSlug-$profileSlug"

        $datasetSummaries = @()
        if (-not $SkipInferenceEval) {
            foreach ($dataset in @($benchmarkConfig['qualityDatasets'])) {
                $datasetName = [string](Get-ObjectValue -Object $dataset -Name 'name')
                $testCases = @(Get-EvaluationDataset -Name $datasetName)
                $maxCases = [int](Get-ObjectValue -Object $dataset -Name 'maxTestCases')
                if ($maxCases -gt 0) {
                    $testCases = @($testCases | Select-Object -First $maxCases)
                }
                $caseResults = @()
                foreach ($testCase in $testCases) {
                    $requestParams = @{
                        Prompt = [string]$testCase.Prompt
                        Model = $candidateName
                        MaxTokens = [int](Get-ObjectValue -Object $profile -Name 'maxTokens')
                        Temperature = [double](Get-ObjectValue -Object $profile -Name 'temperature')
                        TimeoutSeconds = [int]$benchmarkConfig['requestTimeoutSec']
                        MaxRetries = 1
                        NumCtx = [int](Get-ObjectValue -Object $profile -Name 'numCtx')
                        NumThread = [int](Get-ObjectValue -Object $profile -Name 'numThread')
                        TopP = [double](Get-ObjectValue -Object $profile -Name 'topP')
                        TopK = [int](Get-ObjectValue -Object $profile -Name 'topK')
                        RepeatLastN = [int](Get-ObjectValue -Object $profile -Name 'repeatLastN')
                        RepeatPenalty = [double](Get-ObjectValue -Object $profile -Name 'repeatPenalty')
                        TfsZ = [double](Get-ObjectValue -Object $profile -Name 'tfsZ')
                        Seed = [int](Get-ObjectValue -Object $profile -Name 'seed')
                    }

                    $response = Send-OllamaRequest @requestParams
                    $responseText = [string]$response.Response
                    $similarity = if ($testCase.ExpectedOutput) {
                        [double](Compare-ResponseSimilarity -Response $responseText -Expected $testCase.ExpectedOutput)
                    } else {
                        $null
                    }
                    $coherence = [double](Measure-Coherence -Response $responseText)
                    $qualityScore = if ($null -ne $similarity) {
                        [math]::Round((($similarity + $coherence) / 2.0), 4)
                    } else {
                        [math]::Round($coherence, 4)
                    }

                    $caseResults += [pscustomobject]@{
                        QualityScore = $qualityScore
                        LatencyMs = [double]$response.RequestDurationSeconds * 1000.0
                        Passed = ($qualityScore -ge 0.7)
                    }
                }

                $datasetSummary = if ($caseResults.Count -gt 0) {
                    [pscustomobject]@{
                        AverageScore = [math]::Round((($caseResults | Measure-Object -Property QualityScore -Average).Average), 4)
                        PassRate = [math]::Round((((@($caseResults | Where-Object { $_.Passed }).Count) / $caseResults.Count) * 100.0), 2)
                        AverageLatency = [math]::Round((($caseResults | Measure-Object -Property LatencyMs -Average).Average), 2)
                    }
                } else {
                    $null
                }
                if ($datasetSummary) {
                    $datasetSummaries += [pscustomobject]@{
                        Dataset = $datasetName
                        AverageScore = [double]$datasetSummary.AverageScore
                        PassRate = [double]$datasetSummary.PassRate
                        AverageLatency = [double]$datasetSummary.AverageLatency
                    }
                }
            }
        }

        $toolScenarioResults = @()
        if (-not $SkipToolEval -and $scenarioItems.Count -gt 0) {
            foreach ($scenario in $scenarioItems) {
                $toolParams = @{
                    Prompt = [string]$scenario.user_content
                    Model = $candidateName
                    System = [string](Get-ObjectValue -Object $toolEvalConfig -Name 'systemPrompt')
                    Temperature = [double](Get-ObjectValue -Object $profile -Name 'temperature')
                    MaxTokens = [int](Get-ObjectValue -Object $toolEvalConfig -Name 'maxTokens')
                    TimeoutSeconds = [int]$benchmarkConfig['requestTimeoutSec']
                    MaxRetries = 1
                    EnableTools = $true
                    NumCtx = [int](Get-ObjectValue -Object $profile -Name 'numCtx')
                    NumThread = [int](Get-ObjectValue -Object $profile -Name 'numThread')
                    TopP = [double](Get-ObjectValue -Object $profile -Name 'topP')
                    TopK = [int](Get-ObjectValue -Object $profile -Name 'topK')
                    RepeatLastN = [int](Get-ObjectValue -Object $profile -Name 'repeatLastN')
                    RepeatPenalty = [double](Get-ObjectValue -Object $profile -Name 'repeatPenalty')
                    TfsZ = [double](Get-ObjectValue -Object $profile -Name 'tfsZ')
                    Seed = [int](Get-ObjectValue -Object $profile -Name 'seed')
                }

                $response = Send-OllamaRequest @toolParams
                $executedTool = $null
                if ($response.ExecutedTools -and $response.ExecutedTools.Count -gt 0) {
                    $executedTool = $response.ExecutedTools[0]
                }

                $expectedTool = if ($scenario.PSObject.Properties['tool_name']) { [string]$scenario.tool_name } else { '' }
                $expectsNoTool = ($scenario.PSObject.Properties['assistant_content'] -and [string]$scenario.assistant_content -eq 'NO_TOOL')
                $toolNameMatch = if ($expectsNoTool) {
                    -not $executedTool
                } else {
                    $executedTool -and $executedTool.name -eq $expectedTool
                }
                $argMatch = if ($expectsNoTool) {
                    $toolNameMatch
                } elseif ($executedTool) {
                    Test-ExpectedArgumentsMatch -Expected $scenario.tool_arguments -Actual $executedTool.arguments
                } else {
                    $false
                }

                $toolScenarioResults += [pscustomobject]@{
                    Prompt = [string]$scenario.user_content
                    ExpectedTool = if ($expectsNoTool) { 'NO_TOOL' } else { $expectedTool }
                    ActualTool = if ($executedTool) { [string]$executedTool.name } else { 'NO_TOOL' }
                    ToolNameMatch = [bool]$toolNameMatch
                    ArgumentMatch = [bool]$argMatch
                    DurationSeconds = [double]$response.RequestDurationSeconds
                }
            }
        }

        $avgQuality = if ($datasetSummaries.Count -gt 0) { [math]::Round((($datasetSummaries | Measure-Object -Property AverageScore -Average).Average), 4) } else { $null }
        $avgPassRate = if ($datasetSummaries.Count -gt 0) { [math]::Round((($datasetSummaries | Measure-Object -Property PassRate -Average).Average), 2) } else { $null }
        $avgLatency = if ($datasetSummaries.Count -gt 0) { [math]::Round((($datasetSummaries | Measure-Object -Property AverageLatency -Average).Average), 2) } else { $null }
        $toolNameAccuracy = if ($toolScenarioResults.Count -gt 0) { [math]::Round((($toolScenarioResults | Measure-Object -Property { if ($_.ToolNameMatch) { 1 } else { 0 } } -Average).Average), 4) } else { $null }
        $argAccuracy = if ($toolScenarioResults.Count -gt 0) { [math]::Round((($toolScenarioResults | Measure-Object -Property { if ($_.ArgumentMatch) { 1 } else { 0 } } -Average).Average), 4) } else { $null }
        $avgToolDuration = if ($toolScenarioResults.Count -gt 0) { [math]::Round((($toolScenarioResults | Measure-Object -Property DurationSeconds -Average).Average), 2) } else { $null }

        $row = [pscustomobject]@{
            Model = $candidateName
            Role = $candidateRole
            Profile = $profileName
            AverageQuality = $avgQuality
            AveragePassRate = $avgPassRate
            AverageLatencyMs = $avgLatency
            ToolNameAccuracy = $toolNameAccuracy
            ArgumentAccuracy = $argAccuracy
            ToolDurationSeconds = $avgToolDuration
        }

        $benchmarkRows.Add($row)
        foreach ($toolRow in $toolScenarioResults) {
            $toolRows.Add([pscustomobject]@{
                Model = $candidateName
                Profile = $profileName
                Prompt = $toolRow.Prompt
                ExpectedTool = $toolRow.ExpectedTool
                ActualTool = $toolRow.ActualTool
                ToolNameMatch = $toolRow.ToolNameMatch
                ArgumentMatch = $toolRow.ArgumentMatch
                DurationSeconds = $toolRow.DurationSeconds
            })
        }
    }
}

$rows = @($benchmarkRows.ToArray())
if ($rows.Count -gt 0) {
    $qualityRows = @($rows | Where-Object { $null -ne $_.AverageQuality })
    $latencyRows = @($rows | Where-Object { $null -ne $_.AverageLatencyMs })
    $qualityMin = if ($qualityRows.Count -gt 0) { ($qualityRows | Measure-Object -Property AverageQuality -Minimum).Minimum } else { $null }
    $qualityMax = if ($qualityRows.Count -gt 0) { ($qualityRows | Measure-Object -Property AverageQuality -Maximum).Maximum } else { $null }
    $latencyMin = if ($latencyRows.Count -gt 0) { ($latencyRows | Measure-Object -Property AverageLatencyMs -Minimum).Minimum } else { $null }
    $latencyMax = if ($latencyRows.Count -gt 0) { ($latencyRows | Measure-Object -Property AverageLatencyMs -Maximum).Maximum } else { $null }

    foreach ($row in $rows) {
        $qualityNorm = if ($null -eq $row.AverageQuality -or $null -eq $qualityMin -or $null -eq $qualityMax -or $qualityMax -le $qualityMin) { 0.0 } else { ($row.AverageQuality - $qualityMin) / ($qualityMax - $qualityMin) }
        $latencyNorm = if ($null -eq $row.AverageLatencyMs -or $null -eq $latencyMin -or $null -eq $latencyMax -or $latencyMax -le $latencyMin) { 0.0 } else { 1.0 - (($row.AverageLatencyMs - $latencyMin) / ($latencyMax - $latencyMin)) }
        $toolNorm = if ($null -eq $row.ToolNameAccuracy) { 0.0 } else { [double]$row.ToolNameAccuracy }
        $row | Add-Member -NotePropertyName CompositeScore -NotePropertyValue ([math]::Round((0.45 * $qualityNorm) + (0.30 * $latencyNorm) + (0.25 * $toolNorm), 4)) -Force
    }
}

$rows = @($rows | Sort-Object CompositeScore -Descending)

$availabilityPath = Join-Path $runRoot 'availability.json'
$summaryPath = Join-Path $runRoot 'summary.json'
$toolDetailsPath = Join-Path $runRoot 'tool-eval.json'
$reportPath = Join-Path $runRoot 'report.md'

@($availability.ToArray()) | ConvertTo-Json -Depth 8 | Set-Content -Path $availabilityPath -Encoding UTF8
@($rows) | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8
@($toolRows.ToArray()) | ConvertTo-Json -Depth 8 | Set-Content -Path $toolDetailsPath -Encoding UTF8

$report = New-Object System.Text.StringBuilder
[void]$report.AppendLine('# PC-AI Ollama Benchmark Sweep')
[void]$report.AppendLine()
[void]$report.AppendLine("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
[void]$report.AppendLine()
[void]$report.AppendLine('## Availability')
[void]$report.AppendLine()
[void]$report.AppendLine('| Model | Provider | Role | Available | Hints |')
[void]$report.AppendLine('| --- | --- | --- | --- | --- |')
foreach ($entry in @($availability.ToArray())) {
    [void]$report.AppendLine("| $($entry.Name) | $($entry.Provider) | $($entry.Role) | $($entry.Available) | $($entry.Hints -replace '\|','/') |")
}
[void]$report.AppendLine()
[void]$report.AppendLine('## Ranked Results')
[void]$report.AppendLine()
[void]$report.AppendLine('| Model | Profile | Quality | Pass Rate | Latency (ms) | Tool Accuracy | Arg Accuracy | Composite |')
[void]$report.AppendLine('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
foreach ($row in $rows) {
    [void]$report.AppendLine("| $($row.Model) | $($row.Profile) | $($row.AverageQuality) | $($row.AveragePassRate) | $($row.AverageLatencyMs) | $($row.ToolNameAccuracy) | $($row.ArgumentAccuracy) | $($row.CompositeScore) |")
}
Set-Content -Path $reportPath -Value $report.ToString() -Encoding UTF8

[pscustomobject]@{
    RunRoot = $runRoot
    AvailabilityPath = $availabilityPath
    SummaryPath = $summaryPath
    ToolDetailsPath = $toolDetailsPath
    ReportPath = $reportPath
    TopRecommendation = if ($rows.Count -gt 0) { $rows[0] } else { $null }
    Rows = @($rows)
}
