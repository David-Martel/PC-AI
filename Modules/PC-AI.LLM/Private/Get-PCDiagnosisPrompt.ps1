#Requires -PSEdition Core

function Get-PCDiagnosisPrompt {
    <#
    .SYNOPSIS
        Generates system and user prompts for PC diagnostic analysis.

    .DESCRIPTION
        Reads diagnostic info, optional logic files, and optionally routes
        the diagnostic request through FunctionGemma tools to produce final
        prompts for LLM analysis.

    .PARAMETER DiagnosticText
        The raw diagnostic text content.
    
    .PARAMETER UseRouter
        Switch to indicate if FunctionGemma routing should be used.
        
    .PARAMETER RouterBaseUrl
        Base URL for the FunctionGemma router API.
        
    .PARAMETER RouterModel
        Model name to use for FunctionGemma routing.
        
    .PARAMETER RouterToolsPath
        Path to the tools configuration JSON file for FunctionGemma routing.
        
    .PARAMETER RouterMaxCalls
        Maximum number of tool calls to process during routing.
        
    .PARAMETER RouterExecuteTools
        Switch indicating whether to execute tools recommended by FunctionGemma router.
        
    .PARAMETER TimeoutSeconds
        Timeout for operations like health checks and router execution.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$DiagnosticText,

        [Parameter()]
        [switch]$UseRouter,

        [Parameter()]
        [string]$RouterBaseUrl,

        [Parameter()]
        [string]$RouterModel,

        [Parameter()]
        [string]$RouterToolsPath,

        [Parameter()]
        [int]$RouterMaxCalls = 3,

        [Parameter()]
        [switch]$RouterExecuteTools,

        [Parameter()]
        [int]$TimeoutSeconds = 300
    )

    $projectRoot = $null
    if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
        $projectRoot = Resolve-PcaiPath -PathType 'Root'
    } else {
        $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    }

    $diagnosePromptPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE.md'
    $diagnoseLogicPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE_LOGIC.md'

    $diagnosePrompt = if (Test-Path $diagnosePromptPath) { Get-Content -Path $diagnosePromptPath -Raw -Encoding utf8 } else { "" }
    $diagnoseLogic = if (Test-Path $diagnoseLogicPath) { Get-Content -Path $diagnoseLogicPath -Raw -Encoding utf8 } else { "" }

    $systemPrompt = @"
$diagnosePrompt

## REASONING FRAMEWORK

$diagnoseLogic

## INSTRUCTIONS

You are analyzing a Windows PC hardware diagnostic report. Follow the reasoning framework above to:
1. Parse the diagnostic output into structured categories
2. Identify issues by severity
3. Provide actionable recommendations
"@

    $routerSummary = ''
    if ($UseRouter) {
        $resolvedRouterUrl = $null
        if (Get-Command Resolve-PcaiEndpoint -ErrorAction SilentlyContinue) {
            $resolvedRouterUrl = Resolve-PcaiEndpoint -ApiUrl $RouterBaseUrl -ProviderName 'functiongemma'
        } else {
            $resolvedRouterUrl = $RouterBaseUrl
        }

        $routerPrompt = "Analyze this report and call tools if needed: `n`n $DiagnosticText"
        $routerHealthy = $false
        if (Get-Command Get-CachedProviderHealth -ErrorAction SilentlyContinue) {
             $routerHealthy = Get-CachedProviderHealth -Provider 'functiongemma' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $resolvedRouterUrl
        }

        if ($routerHealthy) {
            try {
                $routerInvokeSplat = @{
                    Prompt          = $routerPrompt
                    BaseUrl         = $resolvedRouterUrl
                    Model           = $RouterModel
                    ExecuteTools    = [bool]$RouterExecuteTools
                    MaxToolCalls    = $RouterMaxCalls
                    TimeoutSeconds  = $TimeoutSeconds
                    SkipHealthCheck = $true
                }
                if (-not [string]::IsNullOrWhiteSpace($RouterToolsPath)) {
                    $routerInvokeSplat.ToolsPath = $RouterToolsPath
                }
                $routerResult = Invoke-FunctionGemmaReAct @routerInvokeSplat

                if ($routerResult -and $routerResult.ToolResults) {
                    $routerSummary = ($routerResult.ToolResults | ConvertTo-Json -Depth 6)
                }
            } catch {
                Write-Warning "FunctionGemma router unavailable, proceeding without tool routing: $_"
            }
        } else {
            Write-Warning "FunctionGemma router health check failed for $resolvedRouterUrl. Proceeding without tool routing."
        }
    }

    $userPrompt = "Analyze this PC hardware diagnostic report: `n`n $DiagnosticText"
    if ($routerSummary) {
        $userPrompt += "`n`n[TOOL_RESULTS]`n$routerSummary"
    }

    return [PSCustomObject]@{
        SystemPrompt = $systemPrompt
        UserPrompt = $userPrompt
    }
}
