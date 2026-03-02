#Requires -Version 5.1

function Send-OllamaRequest {
    <#
    .SYNOPSIS
        Sends a request to pcai-inference for text generation

    .DESCRIPTION
        Core wrapper function for pcai-inference completions endpoint. Supports streaming responses,
        custom models, temperature control, and timeout handling with automatic retries.

    .PARAMETER Prompt
        The text prompt to send to the model

    .PARAMETER Model
        The model to use (default: qwen2.5-coder:7b)

    .PARAMETER System
        System prompt to guide model behavior

    .PARAMETER Temperature
        Controls randomness (0.0-2.0). Lower = more deterministic. Default: 0.7

    .PARAMETER MaxTokens
        Maximum tokens to generate (optional)

    .PARAMETER Stream
        Enable streaming response (output tokens as they are generated)

    .PARAMETER TimeoutSeconds
        Request timeout in seconds (default: 120)

    .PARAMETER MaxRetries
        Maximum number of retry attempts on failure (default: 3)

    .PARAMETER RetryDelaySeconds
        Delay between retries in seconds (default: 2)

    .EXAMPLE
        Send-OllamaRequest -Prompt "Explain what a GPU does"
        Sends a simple prompt with default settings

    .EXAMPLE
        Send-OllamaRequest -Prompt "Analyze this code" -Model "deepseek-r1:8b" -Temperature 0.3
        Uses a different model with lower temperature for more consistent output

    .EXAMPLE
        Send-OllamaRequest -Prompt "Write a story" -System "You are a creative writer" -Stream
        Uses system prompt and streaming output

    .OUTPUTS
        PSCustomObject with response text, metadata, and timing information
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipeline)]
        [ValidateNotNullOrEmpty()]
        [string]$Prompt,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [string]$System,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [ValidateRange(1, 100000)]
        [int]$MaxTokens,

        [Parameter()]
        [switch]$Stream,

        [Parameter()]
        [ValidateRange(1, 600)]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [ValidateRange(0, 10)]
        [int]$MaxRetries = 3,

        [Parameter()]
        [ValidateRange(1, 60)]
        [int]$RetryDelaySeconds = 2
    )

    begin {
        Write-Verbose "Initializing pcai-inference request..."

        # Verify pcai-inference connectivity
        if (-not (Test-PcaiInferenceConnection)) {
            throw "Cannot connect to pcai-inference at $($script:ModuleConfig.PcaiInferenceApiUrl). Ensure the server is running."
        }

        # Verify model exists (best effort)
        $availableModels = Get-OllamaModels
        $modelExists = $availableModels | Where-Object { $_.Name -eq $Model }
        if (-not $modelExists -and $availableModels.Count -gt 0) {
            Write-Warning "Model '$Model' not found. Available models: $($availableModels.Name -join ', ')"
        }
    }

    process {
        $startTime = Get-Date
        $attempt = 0
        $success = $false
        $response = $null
        $lastError = $null

        while (-not $success -and $attempt -lt $MaxRetries) {
            $attempt++

            try {
                Write-Verbose "Attempt $attempt of $MaxRetries - Sending request to model '$Model'"

                $params = @{
                    Prompt = $Prompt
                    Model = $Model
                    Temperature = $Temperature
                    Stream = $Stream.IsPresent
                    TimeoutSeconds = $TimeoutSeconds
                }

                if ($System) {
                    $params['System'] = $System
                }

                if ($PSBoundParameters.ContainsKey('MaxTokens')) {
                    $params['MaxTokens'] = $MaxTokens
                }

                if ($Stream) {
                    $response = Invoke-OpenAICompletionStream -Prompt $Prompt -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $script:ModuleConfig.PcaiInferenceApiUrl
                } else {
                    $response = Invoke-OllamaGenerate @params
                }

                $success = $true
                Write-Verbose "Request completed successfully"
            }
            catch {
                $lastError = $_
                Write-Warning "Request attempt $attempt failed: $($_.Exception.Message)"

                if ($attempt -lt $MaxRetries) {
                    Write-Verbose "Retrying in $RetryDelaySeconds seconds..."
                    Start-Sleep -Seconds $RetryDelaySeconds
                }
            }
        }

        if (-not $success) {
            throw "Failed to complete pcai-inference request after $MaxRetries attempts. Last error: $lastError"
        }

        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        # Format response
        if (-not $Stream) {
            $text = $null
            if ($response.choices -and $response.choices.Count -gt 0) {
                $text = $response.choices[0].text
            }

            $result = [PSCustomObject]@{
                Response = $text
                Model = $response.model
                CreatedAt = $response.created
                Usage = $response.usage
                RequestDurationSeconds = [math]::Round($duration, 2)
                Timestamp = $startTime
            }

            return $result
        } else {
            return $response
        }
    }

    end {
        Write-Verbose "pcai-inference request completed"
    }
}
