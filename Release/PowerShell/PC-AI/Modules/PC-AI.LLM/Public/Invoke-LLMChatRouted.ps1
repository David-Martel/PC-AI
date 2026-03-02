#Requires -Version 5.1
<#
.SYNOPSIS
    Routes a user request through FunctionGemma tool-calling, then returns a final LLM response.

.DESCRIPTION
    Uses FunctionGemma (OpenAI-compatible) to select and execute PC-AI tools, then
    forwards tool results to the primary LLM provider (pcai-inference/OpenAI-compatible) using
    DIAGNOSE.md or CHAT.md as the system prompt.

.PARAMETER Message
    User request or prompt.

.PARAMETER Mode
    Prompt mode: chat (CHAT.md) or diagnose (DIAGNOSE.md + DIAGNOSE_LOGIC.md).

.PARAMETER Model
    Final LLM model to use for narrative response.

.PARAMETER Provider
    Final LLM provider: auto|pcai-inference|vllm|lmstudio.

.PARAMETER RouterBaseUrl
    FunctionGemma base URL (OpenAI-compatible).

.PARAMETER RouterModel
    FunctionGemma model name.

.PARAMETER ToolsPath
    Path to pcai-tools.json tool schema.

.PARAMETER ExecuteTools
    Execute tool calls selected by FunctionGemma.

.PARAMETER MaxToolCalls
    Maximum number of tool calls to execute in a single router pass.

.PARAMETER TimeoutSeconds
    Timeout for router and final LLM call.

.PARAMETER Temperature
    Final LLM temperature.

.PARAMETER BypassRouter
    Skip the FunctionGemma router and proceed directly to final LLM.
#>
function Invoke-LLMChatRouted {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Message,

        [Parameter()]
        [ValidateSet('chat', 'diagnose')]
        [string]$Mode = 'chat',

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [ValidateSet('auto', 'pcai-inference', 'ollama', 'vllm', 'lmstudio')]
        [string]$Provider = 'auto',

        [Parameter()]
        [string]$RouterBaseUrl = $script:ModuleConfig.RouterApiUrl,

        [Parameter()]
        [string]$RouterModel = $script:ModuleConfig.RouterModel,

        [Parameter()]
        [string]$ToolsPath = (Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))) 'Config\pcai-tools.json'),

        [Parameter()]
        [switch]$ExecuteTools,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$MaxToolCalls = 3,

        [Parameter()]
        [ValidateRange(5, 900)]
        [int]$TimeoutSeconds = ([math]::Max(120, $script:ModuleConfig.DefaultTimeout)),

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.4,

        [Parameter()]
        [switch]$EnforceJson,

        [Parameter()]
        [switch]$BypassRouter
    )

    $systemPrompt = Get-EnrichedSystemPrompt -Mode $Mode

    if (-not $PSBoundParameters.ContainsKey('EnforceJson')) {
        $EnforceJson = ($Mode -eq 'diagnose')
    }

    $routerPrompt = @"
[MODE]
$Mode

[SYSTEM_PROMPT]
$systemPrompt

[USER_REQUEST]
$Message
"@

    $resolvedRouterUrl = Resolve-PcaiEndpoint -ApiUrl $RouterBaseUrl -ProviderName 'functiongemma'

    # Initialize router state tracking
    $routerAvailable = $false
    $degradedMode = $false
    $routerResult = $null

    # Try to use router unless explicitly bypassed
    if (-not $BypassRouter) {
        $routerHealthy = Get-CachedProviderHealth -Provider 'functiongemma' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $resolvedRouterUrl
        if ($routerHealthy) {
            try {
                $routerResult = Invoke-FunctionGemmaReAct `
                    -Prompt $routerPrompt `
                    -BaseUrl $resolvedRouterUrl `
                    -Model $RouterModel `
                    -ToolsPath $ToolsPath `
                    -ExecuteTools:$ExecuteTools `
                    -ReturnFinal:$false `
                    -MaxToolCalls $MaxToolCalls `
                    -TimeoutSeconds $TimeoutSeconds `
                    -SkipHealthCheck
                $routerAvailable = $true
            } catch {
                Write-Warning "FunctionGemma router unavailable, proceeding without tool routing: $_"
                $degradedMode = $true
                $routerResult = [PSCustomObject]@{
                    ToolCalls = @()
                    ToolResults = @()
                }
            }
        } else {
            Write-Warning "FunctionGemma router health check failed for $resolvedRouterUrl. Proceeding without tool routing."
            $degradedMode = $true
            $routerResult = [PSCustomObject]@{
                ToolCalls = @()
                ToolResults = @()
            }
        }
    } else {
        # Router explicitly bypassed
        $degradedMode = $true
        $routerResult = [PSCustomObject]@{
            ToolCalls = @()
            ToolResults = @()
        }
    }

    $toolResults = $routerResult.ToolResults
    $toolSummary = ''
    if ($toolResults -and $toolResults.Count -gt 0) {
        $toolSummary = ($toolResults | ConvertTo-Json -Depth 6)
    }

    $userContent = if ($toolSummary) {
@"
$Message

[TOOL_RESULTS]
$toolSummary
"@
    } else {
        $Message
    }

    $messages = @()
    if ($systemPrompt) {
        $messages += @{ role = 'system'; content = $systemPrompt }
    }
    $messages += @{ role = 'user'; content = $userContent }

    $finalResponse = Invoke-LLMChatWithFallback -Messages $messages -Model $Model -Temperature $Temperature -TimeoutSeconds $TimeoutSeconds -Provider $Provider

    $jsonValid = $false
    $jsonError = $null
    $responseJson = $null
    if ($Mode -eq 'diagnose') {
        try {
            $responseJson = ConvertFrom-LLMJson -Content $finalResponse.message.content -Strict
            $jsonValid = $true
        } catch {
            $jsonError = $_.Exception.Message
            if ($EnforceJson) {
                throw "Diagnose mode requires valid JSON output. $jsonError"
            }
        }
    }

    return [PSCustomObject]@{
        Mode = $Mode
        Prompt = $Message
        ToolCalls = $routerResult.ToolCalls
        ToolResults = $toolResults
        Response = $finalResponse.message.content
        ResponseJson = $responseJson
        JsonValid = $jsonValid
        JsonError = $jsonError
        Provider = $finalResponse.Provider
        Model = $Model
        RouterModel = $RouterModel
        RouterBaseUrl = $resolvedRouterUrl
        RouterAvailable = $routerAvailable
        DegradedMode = $degradedMode
        Timestamp = Get-Date
    }
}
