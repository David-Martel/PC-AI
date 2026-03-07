function Invoke-SingleTestCase {
    [CmdletBinding()]
    param(
        [EvaluationTestCase]$TestCase,
        [string]$Backend,
        [string]$Model,
        [int]$MaxTokens = 512,
        [float]$Temperature = 0.7,
        [int]$NumCtx,
        [int]$NumThread,
        [double]$TopP,
        [int]$TopK,
        [int]$RepeatLastN,
        [double]$RepeatPenalty,
        [double]$TfsZ,
        [int]$Seed,
        [int]$RequestTimeoutSec = 120
    )

    $result = [EvaluationResult]::new()
    $result.TestCaseId = $TestCase.Id
    $result.Backend = $Backend
    $result.Timestamp = [datetime]::UtcNow
    $result.Prompt = $TestCase.Prompt

    # Store expected output and context for metric calculation
    $result.Metrics['expected'] = $TestCase.ExpectedOutput
    $result.Metrics['context'] = $TestCase.Context['context'] ?? $null

    # Measure memory before
    $memBefore = [System.GC]::GetTotalMemory($false) / 1MB

    $stopwatch = [Stopwatch]::StartNew()

    try {
        switch ($Backend) {
            { $_ -in 'llamacpp', 'mistralrs' } {
                $result.Response = Invoke-PcaiGenerate -Prompt $TestCase.Prompt -MaxTokens $MaxTokens -Temperature $Temperature
            }
            { $_ -in 'http', 'llamacpp-bin', 'mistralrs-bin' } {
                $body = @{
                    prompt = $TestCase.Prompt
                    max_tokens = $MaxTokens
                    temperature = $Temperature
                } | ConvertTo-Json

                $response = Invoke-RestMethod -Uri "$script:EvaluationConfig.HttpBaseUrl/v1/completions" `
                    -Method Post -Body $body -ContentType 'application/json' -TimeoutSec $RequestTimeoutSec
                $text = $response.choices[0].text
                if (-not $text -and $response.choices[0].message) {
                    $text = $response.choices[0].message.content
                }

                if (-not $text) {
                    $chatBody = @{
                        messages = @(
                            @{
                                role = 'user'
                                content = $TestCase.Prompt
                            }
                        )
                        max_tokens = $MaxTokens
                        temperature = $Temperature
                    } | ConvertTo-Json -Depth 6

                    $chatResponse = Invoke-RestMethod -Uri "$script:EvaluationConfig.HttpBaseUrl/v1/chat/completions" `
                        -Method Post -Body $chatBody -ContentType 'application/json' -TimeoutSec $RequestTimeoutSec
                    $text = $chatResponse.choices[0].message.content
                }

                $result.Response = $text
            }
            'ollama' {
                $selectedModel = if ($Model) { $Model } else { ($script:EvaluationConfig.OllamaModel ?? 'llama3.2') }

                if (Get-Command Send-OllamaRequest -ErrorAction SilentlyContinue) {
                    $requestParams = @{
                        Prompt = $TestCase.Prompt
                        Model = $selectedModel
                        MaxTokens = $MaxTokens
                        Temperature = $Temperature
                        TimeoutSeconds = $RequestTimeoutSec
                        MaxRetries = 1
                    }

                    if ($PSBoundParameters.ContainsKey('NumCtx')) { $requestParams.NumCtx = $NumCtx }
                    if ($PSBoundParameters.ContainsKey('NumThread')) { $requestParams.NumThread = $NumThread }
                    if ($PSBoundParameters.ContainsKey('TopP')) { $requestParams.TopP = $TopP }
                    if ($PSBoundParameters.ContainsKey('TopK')) { $requestParams.TopK = $TopK }
                    if ($PSBoundParameters.ContainsKey('RepeatLastN')) { $requestParams.RepeatLastN = $RepeatLastN }
                    if ($PSBoundParameters.ContainsKey('RepeatPenalty')) { $requestParams.RepeatPenalty = $RepeatPenalty }
                    if ($PSBoundParameters.ContainsKey('TfsZ')) { $requestParams.TfsZ = $TfsZ }
                    if ($PSBoundParameters.ContainsKey('Seed')) { $requestParams.Seed = $Seed }

                    $response = Send-OllamaRequest @requestParams
                    $result.Response = $response.Response
                    $result.Model = if ($response.Model) { $response.Model } else { $selectedModel }
                }
                else {
                    $options = @{
                        num_predict = $MaxTokens
                        temperature = $Temperature
                    }
                    if ($PSBoundParameters.ContainsKey('NumCtx')) { $options.num_ctx = $NumCtx }
                    if ($PSBoundParameters.ContainsKey('TopP')) { $options.top_p = $TopP }
                    if ($PSBoundParameters.ContainsKey('TopK')) { $options.top_k = $TopK }
                    if ($PSBoundParameters.ContainsKey('RepeatLastN')) { $options.repeat_last_n = $RepeatLastN }
                    if ($PSBoundParameters.ContainsKey('RepeatPenalty')) { $options.repeat_penalty = $RepeatPenalty }
                    if ($PSBoundParameters.ContainsKey('TfsZ')) { $options.tfs_z = $TfsZ }
                    if ($PSBoundParameters.ContainsKey('Seed')) { $options.seed = $Seed }
                    if ($PSBoundParameters.ContainsKey('NumThread')) { $options.num_thread = $NumThread }

                    $body = @{
                        model = $selectedModel
                        prompt = $TestCase.Prompt
                        stream = $false
                        options = $options
                    } | ConvertTo-Json

                    $response = Invoke-RestMethod -Uri "$script:EvaluationConfig.OllamaBaseUrl/api/generate" `
                        -Method Post -Body $body -ContentType 'application/json' -TimeoutSec $RequestTimeoutSec
                    $result.Response = $response.response
                    $result.Model = $selectedModel
                }
            }
        }

        if (-not $result.Model) {
            $result.Model = if ($Model) { $Model } else { $Backend }
        }
    } catch {
        $result.ErrorMessage = $_.Exception.Message
        $result.Response = ""
    }

    $stopwatch.Stop()
    $result.Duration = $stopwatch.Elapsed

    # Measure memory after
    $memAfter = [System.GC]::GetTotalMemory($false) / 1MB
    $result.Metrics['memory_mb'] = [math]::Round($memAfter - $memBefore, 2)

    return $result
}
