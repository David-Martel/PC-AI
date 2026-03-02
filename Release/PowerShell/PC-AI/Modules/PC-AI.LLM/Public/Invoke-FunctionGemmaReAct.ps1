#Requires -Version 5.1

function Invoke-FunctionGemmaChat {
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter(Mandatory)]
        [array]$Tools,

        [Parameter(Mandatory)]
        [string]$BaseUrl,

        [Parameter(Mandatory)]
        [string]$Model,

        [Parameter()]
        [int]$TimeoutSeconds = 120
    )

    $payload = @{
        model = $Model
        messages = $Messages
        tools = $Tools
        tool_choice = 'auto'
        temperature = 0.2
    }

    $uri = ("{0}/v1/chat/completions" -f $BaseUrl.TrimEnd('/'))
    $jsonBody = $payload | ConvertTo-Json -Depth 12
    $jsonBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)

    # Prefer native OpenAI client for lower latency
    if (([System.Management.Automation.PSTypeName]'PcaiNative.PcaiCore').Type -and [PcaiNative.PcaiCore]::IsAvailable) {
        $client = [PcaiNative.PcaiOpenAiClient]::new($BaseUrl)
        try {
            $raw = $client.ChatCompletionsRaw($jsonBody)
            return ($raw | ConvertFrom-Json -ErrorAction Stop)
        } catch {
            # fallback to REST below
        } finally {
            $client.Dispose()
        }
    }

    return Invoke-RestMethod -Method Post -Uri $uri -Body $jsonBytes -ContentType 'application/json; charset=utf-8' -TimeoutSec $TimeoutSeconds
}


function Invoke-FunctionGemmaReAct {
    <#
    .SYNOPSIS
    Uses FunctionGemma (OpenAI-compatible API) to plan tool calls and optionally executes them.

    .DESCRIPTION
        Sends a prompt + tool schema to FunctionGemma and returns tool calls. When -ExecuteTools
        is specified, executes mapped PC_AI tools and optionally returns a final response from
        FunctionGemma after tool results are provided.

    .PARAMETER Prompt
        The user request or prompt.

    .PARAMETER BaseUrl
        Base URL for the FunctionGemma OpenAI-compatible API.

    .PARAMETER Model
        Model name to use.

    .PARAMETER ToolsPath
        Path to the tools configuration JSON file.

    .PARAMETER ExecuteTools
        Whether to execute the recommended tools.

    .PARAMETER ReturnFinal
        Whether to return a final synthesized response after tool execution.

    .PARAMETER MaxToolCalls
        Maximum number of tool calls to process.

    .PARAMETER ResultLimit
        Maximum length in bytes for tool results before truncation. Prevents context window overflow. Default: 8192

    .PARAMETER TimeoutSeconds
        Timeout for API requests in seconds.

    .PARAMETER ShowProgress
        Whether to show progress bars during generation.

    .PARAMETER ShowMetrics
        Whether to show token usage metrics (prompt tokens, generation tokens, tokens/second, KV cache).

    .PARAMETER ProgressIntervalSeconds
        Interval for progress updates in seconds (1-10). Default: 1

    .PARAMETER SkipHealthCheck
        Skip cached router health check (useful for tests or custom endpoints).

    .EXAMPLE
        Invoke-FunctionGemmaReAct -Prompt "What is the status of my USB devices?" -ExecuteTools
        Routes the query to FunctionGemma for tool planning and executes recommended diagnostic tools

    .EXAMPLE
        Invoke-FunctionGemmaReAct -Prompt "Analyze disk health" -ExecuteTools -ReturnFinal -ShowMetrics
        Executes tools and returns a final synthesized response with performance metrics
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$Prompt,

        [Parameter()]
        [string]$BaseUrl = $script:ModuleConfig.RouterApiUrl,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.RouterModel,

        [Parameter()]
        [string]$ToolsPath = (Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))) 'Config\pcai-tools.json'),

        [Parameter()]
        [switch]$ExecuteTools,

        [Parameter()]
        [switch]$ReturnFinal,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$MaxToolCalls = 5,

        [Parameter()]
        [int]$ResultLimit = 8192,

        [Parameter()]
        [ValidateRange(5, 600)]
        [int]$TimeoutSeconds = ([math]::Max(120, $script:ModuleConfig.DefaultTimeout)),

        [Parameter()]
        [switch]$ShowProgress,

        [Parameter()]
        [switch]$ShowMetrics,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$ProgressIntervalSeconds = 1,

        [Parameter()]
        [switch]$SkipHealthCheck
    )

    if (-not (Test-Path $ToolsPath)) {
        throw "Tools config not found: $ToolsPath"
    }

    $tools = (Get-Content -Path $ToolsPath -Raw -Encoding UTF8 | ConvertFrom-Json).tools
    $BaseUrl = Resolve-PcaiEndpoint -ApiUrl $BaseUrl -ProviderName 'functiongemma'
    $ModuleRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

    if (-not $SkipHealthCheck) {
        $routerHealthy = Get-CachedProviderHealth -Provider 'functiongemma' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $BaseUrl
        if (-not $routerHealthy) {
            throw "FunctionGemma router not reachable at $BaseUrl"
        }
    }

    # Inject Native Context into the prompt
    if (([System.Management.Automation.PSTypeName]'PcaiNative.PcaiCore').Type -and [PcaiNative.PcaiCore]::IsAvailable) {
        $contextJson = [PcaiNative.PcaiCore]::QueryFullContextJson()
        if ($contextJson) {
            $Prompt = @"
[NATIVE_CONTEXT]
$contextJson

[USER_REQUEST]
$Prompt
"@
        }
    }

    $messages = @(@{ role = 'user'; content = $Prompt })
    $metricsBefore = $null
    if ($ShowMetrics) {
        $metricsBefore = Get-VLLMMetricsSnapshot -ApiUrl $BaseUrl -ModelName $Model -TimeoutSeconds 3
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-FunctionGemmaChat -Messages $messages -Tools $tools -BaseUrl $BaseUrl -Model $Model -TimeoutSeconds $TimeoutSeconds
    $sw.Stop()

    if (-not $response -or -not $response.choices -or $response.choices.Count -eq 0) {
        return [PSCustomObject]@{
            ToolCalls = @()
            ToolResults = @()
            FinalAnswer = "Error: LLM returned no response or choices."
        }
    }

    $choice = $response.choices[0]
    $message = $choice.message

    $toolCalls = @()
    if ($message -and $message.tool_calls) {
        $toolCalls = $message.tool_calls
    }

    $toolResults = @()
    if ($ExecuteTools -and $toolCalls.Count -gt 0) {
        $callCount = 0
        foreach ($call in $toolCalls) {
            if ($callCount -ge $MaxToolCalls) { break }
            $callCount++
            $name = $call.function.name
            $args = $call.function.arguments
            if ($args -is [string]) {
                try { $args = $args | ConvertFrom-Json -ErrorAction Stop } catch { $args = @{} }
            }
            if (-not $args) { $args = @{} }

            $result = Invoke-ToolByName -Name $name -Args $args -Tools $tools -ModuleRoot $ModuleRoot

            $resultString = if ($result -is [string]) { $result } else { $result | ConvertTo-Json -Depth 5 }

            # Truncate if exceeds limit
            if ($ResultLimit -gt 0 -and $resultString.Length -gt $ResultLimit) {
                $truncatedSize = $resultString.Length - $ResultLimit
                $resultString = $resultString.Substring(0, $ResultLimit) + "`n`n[TRUNCATED: $truncatedSize bytes removed for context window safety]"
                Write-Verbose "Truncated tool result by $truncatedSize bytes"
            }

            $toolResults += [PSCustomObject]@{
                name = $name
                arguments = $args
                result = $resultString
            }

            $messages += @{
                role = 'assistant'
                tool_calls = @($call)
            }
            $messages += @{
                role = 'tool'
                tool_call_id = $call.id
                content = [string]$resultString
            }
        }

        if ($ReturnFinal) {
            $final = Invoke-FunctionGemmaChat -Messages $messages -Tools $tools -BaseUrl $BaseUrl -Model $Model -TimeoutSeconds $TimeoutSeconds
            $message = $final.choices[0].message
        }
    }

    $metricsAfter = $null
    $metricsSummary = $null
    if ($ShowMetrics) {
        $metricsAfter = Get-VLLMMetricsSnapshot -ApiUrl $BaseUrl -ModelName $Model -TimeoutSeconds 3
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
        Prompt = $Prompt
        ToolCalls = $toolCalls
        ToolResults = $toolResults
        Response = $message.content
        RawResponse = $message
        Metrics = $metricsSummary
        Model = $Model
        BaseUrl = $BaseUrl
    }
}
