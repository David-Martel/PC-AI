function Invoke-SingleTestCase {
    [CmdletBinding()]
    param(
        [EvaluationTestCase]$TestCase,
        [string]$Backend,
        [int]$MaxTokens = 512,
        [float]$Temperature = 0.7,
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
                $body = @{
                    model = $script:EvaluationConfig.OllamaModel ?? 'llama3.2'
                    prompt = $TestCase.Prompt
                    stream = $false
                    options = @{
                        num_predict = $MaxTokens
                        temperature = $Temperature
                    }
                } | ConvertTo-Json

                $response = Invoke-RestMethod -Uri "$script:EvaluationConfig.OllamaBaseUrl/api/generate" `
                    -Method Post -Body $body -ContentType 'application/json' -TimeoutSec $RequestTimeoutSec
                $result.Response = $response.response
            }
        }

        $result.Model = $Backend
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
