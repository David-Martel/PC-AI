#Requires -Version 5.1

function Invoke-LLMChat {
    <#
    .SYNOPSIS
        Interactive chat interface with LLM providers (pcai-inference, FunctionGemma, OpenAI-compatible)

    .DESCRIPTION
        Provides a unified chat interface supporting single-shot and interactive modes with automatic
        provider fallback, tool calling via ReAct pattern, streaming responses, and detailed metrics.
        Supports pcai-inference as the primary OpenAI-compatible endpoint with optional fallback providers.

    .PARAMETER Message
        The user message to send to the LLM. Can be piped in for single-shot mode.

    .PARAMETER Model
        The model to use for generation. Default is configured in module settings (qwen2.5-coder:7b).

    .PARAMETER System
        System prompt to guide the model's behavior and set context.

    .PARAMETER Temperature
        Controls randomness in generation (0.0-2.0). Lower values are more deterministic. Default: 0.7

    .PARAMETER MaxTokens
        Maximum number of tokens to generate. Optional, uses model default if not specified.

    .PARAMETER TimeoutSeconds
        Request timeout in seconds. Default is configured in module settings (120).

    .PARAMETER Interactive
        Starts an interactive chat session with conversation history. Type 'exit', 'quit', or 'q' to end.
        Type 'clear' to reset conversation history.

    .PARAMETER ToJson
        Extracts and returns JSON content from the LLM response using the ConvertFrom-LLMJson parser.

    .PARAMETER History
        Array of message objects (@{role='user/assistant'; content='text'}) to initialize conversation context.

    .PARAMETER Provider
        LLM provider to use: 'auto' (tries configured order), 'pcai-inference', 'vllm', or 'lmstudio'. Default: 'auto'

    .PARAMETER UseRouter
        Routes the request through FunctionGemma for tool call planning before sending to the main LLM.

    .PARAMETER RouterMode
        Routing mode when UseRouter is enabled: 'chat' (general) or 'diagnose' (diagnostic-specific). Default: 'chat'

    .PARAMETER Stream
        Enables streaming output for OpenAI-compatible providers (tokens displayed as they are generated).

    .PARAMETER ShowProgress
        Displays a progress bar during generation with elapsed time updates.

    .PARAMETER ShowMetrics
        Shows detailed performance metrics including prompt tokens, generation tokens, tokens/second, and KV cache usage.

    .PARAMETER ProgressIntervalSeconds
        Update interval for progress display in seconds (1-10). Default: 1

    .PARAMETER ResultLimit
        Maximum length in bytes for tool results before truncation. Prevents context window overflow. Default: 8192

    .EXAMPLE
        Invoke-LLMChat -Message "Explain how DNS works"
        Single-shot query with default model

    .EXAMPLE
        Invoke-LLMChat -Interactive -Model "deepseek-r1:8b" -ShowMetrics
        Start interactive session with specific model and metrics display

    .EXAMPLE
        "Analyze this error" | Invoke-LLMChat -System "You are a debugging expert" -ToJson
        Pipe input with system prompt and JSON extraction

    .EXAMPLE
        Invoke-LLMChat -Message "What USB devices have errors?" -UseRouter -RouterMode diagnose
        Use FunctionGemma to route and plan diagnostic tool calls

    .EXAMPLE
        Invoke-LLMChat -Message "Write a story" -Stream -Provider pcai-inference
        Stream tokens as they are generated using pcai-inference

    .OUTPUTS
        PSCustomObject containing Response, RawResponse, Model, History, TotalDuration, Metrics, and Timestamp
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Position = 0, ValueFromPipeline)]
        [string]$Message,

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
        [ValidateRange(1, 600)]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [switch]$Interactive,

        [Parameter()]
        [switch]$ToJson,

        [Parameter()]
        [array]$History = @(),

        [Parameter()]
        [ValidateSet('auto', 'pcai-inference', 'ollama', 'vllm', 'lmstudio')]
        [string]$Provider = 'auto',

        [Parameter()]
        [switch]$UseRouter,

        [Parameter()]
        [ValidateSet('chat', 'diagnose')]
        [string]$RouterMode = 'chat',

        [Parameter()]
        [switch]$Stream,

        [Parameter()]
        [switch]$ShowProgress,

        [Parameter()]
        [switch]$ShowMetrics,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$ProgressIntervalSeconds = 1,

        [Parameter()]
        [int]$ResultLimit = 8192
    )

    begin {
        $conversationHistory = [System.Collections.ArrayList]::new()
        if ($System) { [void]$conversationHistory.Add(@{ role = 'system'; content = $System }) }
        foreach ($msg in $History) { [void]$conversationHistory.Add($msg) }
    }

    process {
        if ($UseRouter -and -not $Interactive) {
            return Invoke-LLMChatRouted -Message $Message -Mode $RouterMode -Model $Model -Provider $Provider -TimeoutSeconds $TimeoutSeconds -Temperature $Temperature
        }

        if ($Interactive) {
            Write-Host "`nStarting interactive chat session with $Model" -ForegroundColor Cyan
            $continueChat = $true
            while ($continueChat) {
                Write-Host "`nYou: " -NoNewline -ForegroundColor Green
                $userInput = Read-Host

                switch ($userInput.ToLower().Trim()) {
                    { $_ -in @('exit', 'quit', 'q') } { $continueChat = $false; break }
                    'clear' {
                        $conversationHistory.Clear()
                        if ($System) { [void]$conversationHistory.Add(@{ role = 'system'; content = $System }) }
                        continue
                    }
                }

                if ([string]::IsNullOrWhiteSpace($userInput)) { continue }

                if ($UseRouter) {
                    $routed = Invoke-LLMChatRouted -Message $userInput -Mode $RouterMode -Model $Model -Provider $Provider -TimeoutSeconds $TimeoutSeconds -Temperature $Temperature
                    Write-Host "`nAssistant: $($routed.Response)" -ForegroundColor Blue
                    continue
                }

                [void]$conversationHistory.Add(@{ role = 'user'; content = $userInput })

                # ReAct Tool Loop
                $toolCallLimit = 3
                $toolCallCount = 0
                $processingResponse = $true

                while ($processingResponse -and $toolCallCount -lt $toolCallLimit) {
                    try {
                        $metricsBefore = $null
                        $metricsProvider = if ($Provider -eq 'auto') { $script:ModuleConfig.ProviderOrder[0] } else { $Provider }
                        if ($ShowMetrics -and $metricsProvider -eq 'vllm') {
                            $metricsBefore = Get-VLLMMetricsSnapshot -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel -TimeoutSeconds 3
                        }

                        $sw = [System.Diagnostics.Stopwatch]::StartNew()
                        $params = @{
                            Messages       = $conversationHistory.ToArray()
                            Model          = $Model
                            Temperature    = $Temperature
                            TimeoutSeconds = $TimeoutSeconds
                            Provider       = $Provider
                            ShowProgress   = $ShowProgress
                            ProgressIntervalSeconds = $ProgressIntervalSeconds
                        }
                        if ($PSBoundParameters.ContainsKey('MaxTokens')) { $params['MaxTokens'] = $MaxTokens }

                        $assistantMessage = $null
                        if ($Stream) {
                            $streamProvider = if ($Provider -eq 'auto') { $script:ModuleConfig.ProviderOrder[0] } else { $Provider }
                            $streamApiUrl = switch ($streamProvider) {
                                'vllm' { $script:ModuleConfig.VLLMApiUrl }
                                'lmstudio' { $script:ModuleConfig.LMStudioApiUrl }
                                default { $script:ModuleConfig.PcaiInferenceApiUrl }
                            }
                            $assistantMessage = Invoke-OpenAIChatStream -Messages $params.Messages -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $streamApiUrl
                            $response = [PSCustomObject]@{ Provider = $streamProvider; message = @{ content = $assistantMessage } }
                        } else {
                            $response = Invoke-LLMChatWithFallback @params
                            $assistantMessage = $response.message.content
                        }
                        $sw.Stop()
                        $metricsAfter = $null
                        if ($ShowMetrics -and $response.Provider -eq 'vllm') {
                            $metricsAfter = Get-VLLMMetricsSnapshot -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel -TimeoutSeconds 3
                        }
                        [void]$conversationHistory.Add(@{ role = 'assistant'; content = $assistantMessage })

                        # Parse for tool calls (Simplified with helper or standard pattern)
                        $toolPattern = '(?s)callTool\((?<name>[^,]+),\s*(?<args>.*?)\)'
                        if ($assistantMessage -match $toolPattern) {
                            $toolName = $Matches['name'].Trim()
                            $toolArgs = $Matches['args'].Trim()
                            $toolCallCount++

                            Write-Host "`n[Tool Call] Executing $toolName..." -ForegroundColor Yellow
                            $toolResult = 'Error: Tool failed'
                            try {
                                # Use standardized Invoke-ToolByName helper
                                $toolResult = Invoke-ToolByName -Name $toolName -Args $toolArgs -ModuleRoot $ModuleRoot

                                # Truncate if exceeds limit
                                if ($ResultLimit -gt 0 -and $toolResult -is [string] -and $toolResult.Length -gt $ResultLimit) {
                                    $truncatedSize = $toolResult.Length - $ResultLimit
                                    $toolResult = $toolResult.Substring(0, $ResultLimit) + "`n`n[TRUNCATED: $truncatedSize bytes removed for context window safety]"
                                }
                            } catch { $toolResult = "Error: $($_.Exception.Message)" }

                            [void]$conversationHistory.Add(@{ role = 'user'; content = "TOOL_RESULT($toolName): $toolResult" })
                        } else {
                            Write-Host "`nAssistant: $assistantMessage" -ForegroundColor Blue
                            if ($ShowMetrics -and $metricsBefore -and $metricsAfter) {
                                $promptDelta = $metricsAfter.PromptTokensTotal - $metricsBefore.PromptTokensTotal
                                $genDelta = $metricsAfter.GenerationTokensTotal - $metricsBefore.GenerationTokensTotal
                                $tps = 0
                                if ($sw.Elapsed.TotalSeconds -gt 0) {
                                    $tps = ($promptDelta + $genDelta) / $sw.Elapsed.TotalSeconds
                                }
                                Write-Host ("vLLM metrics: {0:N0} prompt, {1:N0} gen tokens, {2:N2} tok/s, KV {3:P0}" -f $promptDelta, $genDelta, $tps, $metricsAfter.KVCacheUsagePerc) -ForegroundColor DarkGray
                            }
                            $processingResponse = $false
                        }
                    } catch { $processingResponse = $false }
                }
            }
        } else {
            # Single-shot mode
            [void]$conversationHistory.Add(@{ role = 'user'; content = $Message })
            $metricsBefore = $null
            $metricsProvider = if ($Provider -eq 'auto') { $script:ModuleConfig.ProviderOrder[0] } else { $Provider }
            if ($ShowMetrics -and $metricsProvider -eq 'vllm') {
                $metricsBefore = Get-VLLMMetricsSnapshot -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel -TimeoutSeconds 3
            }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $params = @{
                Messages       = $conversationHistory.ToArray()
                Model          = $Model
                Temperature    = $Temperature
                TimeoutSeconds = $TimeoutSeconds
                Provider       = $Provider
                ShowProgress   = $ShowProgress
                ProgressIntervalSeconds = $ProgressIntervalSeconds
            }
            if ($PSBoundParameters.ContainsKey('MaxTokens')) { $params['MaxTokens'] = $MaxTokens }

            $assistantMessage = $null
            if ($Stream) {
                $streamProvider = if ($Provider -eq 'auto') { $script:ModuleConfig.ProviderOrder[0] } else { $Provider }
                $streamApiUrl = switch ($streamProvider) {
                    'vllm' { $script:ModuleConfig.VLLMApiUrl }
                    'lmstudio' { $script:ModuleConfig.LMStudioApiUrl }
                    default { $script:ModuleConfig.PcaiInferenceApiUrl }
                }
                $assistantMessage = Invoke-OpenAIChatStream -Messages $params.Messages -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $streamApiUrl
                $response = [PSCustomObject]@{ Provider = $streamProvider; message = @{ content = $assistantMessage } }
            } else {
                $response = Invoke-LLMChatWithFallback @params
                $assistantMessage = $response.message.content
            }
            $sw.Stop()
            [void]$conversationHistory.Add(@{ role = 'assistant'; content = $assistantMessage })

            $finalResponse = $assistantMessage
            if ($ToJson) {
                # Utilize Natively-accelerated JSON extractor
                $finalResponse = ConvertFrom-LLMJson -Content $assistantMessage
            }

            $metricsAfter = $null
            $metricsSummary = $null
            if ($ShowMetrics -and $response.Provider -eq 'vllm') {
                $metricsAfter = Get-VLLMMetricsSnapshot -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel -TimeoutSeconds 3
                if ($metricsBefore -and $metricsAfter) {
                    $promptDelta = $metricsAfter.PromptTokensTotal - $metricsBefore.PromptTokensTotal
                    $genDelta = $metricsAfter.GenerationTokensTotal - $metricsBefore.GenerationTokensTotal
                    $tps = 0
                    if ($sw.Elapsed.TotalSeconds -gt 0) {
                        $tps = ($promptDelta + $genDelta) / $sw.Elapsed.TotalSeconds
                    }
                    $metricsSummary = [PSCustomObject]@{
                        ElapsedSeconds   = [math]::Round($sw.Elapsed.TotalSeconds, 3)
                        PromptTokens     = [int]$promptDelta
                        GenerationTokens = [int]$genDelta
                        TokensPerSecond  = [math]::Round($tps, 2)
                        KVCacheUsagePerc = $metricsAfter.KVCacheUsagePerc
                    }
                }
            }

            return [PSCustomObject]@{
                Response      = $finalResponse
                RawResponse   = $assistantMessage
                Model         = $Model
                History       = $conversationHistory.ToArray()
                TotalDuration = $response.total_duration
                Metrics       = $metricsSummary
                Timestamp     = Get-Date
            }
        }
    }
}
