#Requires -PSEdition Core

function Invoke-PCDiagnosis {
    <#
    .SYNOPSIS
        Analyzes PC diagnostic reports using LLM

    .DESCRIPTION
        Main diagnostic analysis function that loads DIAGNOSE.md as system prompt and
        DIAGNOSE_LOGIC.md as reasoning guide, then submits a hardware diagnostic report
        to pcai-inference for intelligent analysis and recommendations.

    .PARAMETER DiagnosticReportPath
        Path to the hardware diagnostic report text file

    .PARAMETER ReportText
        Direct diagnostic report text (alternative to file path)

    .PARAMETER Model
        The model to use for analysis (default: configured module default)

    .PARAMETER Temperature
        Controls analysis consistency (0.0-2.0). Lower = more deterministic. Default: 0.3

    .PARAMETER IncludeRawResponse
        Include the raw LLM response in output

    .PARAMETER SaveReport
        Save the analysis report to a file

    .PARAMETER OutputPath
        Path to save the analysis report (default: Desktop\PC-Diagnosis-Analysis.txt)

    .PARAMETER TimeoutSeconds
        Request timeout in seconds (30-1800). Default: 300

    .PARAMETER UseRouter
        Routes the diagnostic report through FunctionGemma for tool call planning before analysis

    .PARAMETER RouterBaseUrl
        Base URL for the FunctionGemma router API. Default from module configuration.

    .PARAMETER RouterModel
        Model name to use for FunctionGemma routing. Default from module configuration.

    .PARAMETER RouterToolsPath
        Path to the tools configuration JSON file for FunctionGemma routing

    .PARAMETER RouterMaxCalls
        Maximum number of tool calls to process during routing (1-10). Default: 3

    .PARAMETER RouterExecuteTools
        Whether to execute tools recommended by FunctionGemma router

    .PARAMETER EnforceJson
        Require valid JSON output from the diagnostic analysis. Throws error if JSON is invalid.

    .EXAMPLE
        Invoke-PCDiagnosis -DiagnosticReportPath "C:\Reports\Hardware-Diagnostics-Report.txt"
        Analyzes a diagnostic report with default settings

    .EXAMPLE
        Get-Content report.txt | Invoke-PCDiagnosis -ReportText {$_} -SaveReport -UseRouter -RouterExecuteTools
        Pipes report text, enables routing and tool execution, and saves the analysis
    #>
    [CmdletBinding(DefaultParameterSetName = 'FromFile')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, ParameterSetName = 'FromFile', Position = 0)]
        [ValidateScript({ Test-Path $_ -PathType Leaf })]
        [string]$DiagnosticReportPath,

        [Parameter(Mandatory, ParameterSetName = 'FromText', ValueFromPipeline)]
        [ValidateNotNullOrEmpty()]
        [string]$ReportText,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.3,

        [Parameter()]
        [switch]$IncludeRawResponse,

        [Parameter()]
        [switch]$SaveReport,

        [Parameter()]
        [string]$OutputPath = (Join-Path -Path ([Environment]::GetFolderPath('Desktop')) -ChildPath 'PC-Diagnosis-Analysis.txt'),

        [Parameter()]
        [ValidateRange(30, 1800)]
        [int]$TimeoutSeconds = 300,

        [Parameter()]
        [switch]$UseRouter,

        [Parameter()]
        [string]$RouterBaseUrl = $script:ModuleConfig.RouterApiUrl,

        [Parameter()]
        [string]$RouterModel = $script:ModuleConfig.RouterModel,

        [Parameter()]
        [string]$RouterToolsPath,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$RouterMaxCalls = 3,

        [Parameter()]
        [switch]$RouterExecuteTools,

        [Parameter()]
        [switch]$EnforceJson
    )

    begin {
        Write-Verbose "Initializing PC diagnosis analysis..."
        $projectRoot = $null
        if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
            $projectRoot = Resolve-PcaiPath -PathType 'Root'
        } else {
            $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
        }

        $diagnosePromptPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE.md'
        $diagnoseLogicPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE_LOGIC.md'

        $diagnosePrompt = Get-Content -Path $diagnosePromptPath -Raw -Encoding utf8
        $diagnoseLogic = Get-Content -Path $diagnoseLogicPath -Raw -Encoding utf8

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
    }

    process {
        $diagnosticText = if ($PSCmdlet.ParameterSetName -eq 'FromFile') {
            Get-Content -Path $DiagnosticReportPath -Raw -Encoding utf8
        } else {
            $ReportText
        }

        if ([string]::IsNullOrWhiteSpace($diagnosticText)) {
            throw "Diagnostic report is empty"
        }

        $routerSummary = ''
        $resolvedRouterUrl = Resolve-PcaiEndpoint -ApiUrl $RouterBaseUrl -ProviderName 'functiongemma'
        if ($UseRouter) {
            $routerPrompt = "Analyze this report and call tools if needed: `n`n $diagnosticText"
            $routerHealthy = Get-CachedProviderHealth -Provider 'functiongemma' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $resolvedRouterUrl
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

        $userPrompt = "Analyze this PC hardware diagnostic report: `n`n $diagnosticText"
        if ($routerSummary) {
            $userPrompt += "`n`n[TOOL_RESULTS]`n$routerSummary"
        }

        Write-Host "Analyzing diagnostic report with $Model..." -ForegroundColor Cyan
        $startTime = Get-Date

        try {
            $messages = @(
                @{ role = 'system'; content = $systemPrompt }
                @{ role = 'user'; content = $userPrompt }
            )

            $response = Invoke-LLMChatWithFallback -Messages $messages -Model $Model -Temperature $Temperature -TimeoutSeconds $TimeoutSeconds
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds

            $analysisText = $response.message.content

            if (-not $PSBoundParameters.ContainsKey('EnforceJson')) {
                $EnforceJson = $true
            }

            $analysisJson = $null
            $jsonValid = $false
            $jsonError = $null
            try {
                $analysisJson = ConvertFrom-LLMJson -Content $analysisText -Strict
                $jsonValid = $true
            } catch {
                $jsonError = $_.Exception.Message
                if ($EnforceJson) {
                    throw "Diagnose mode requires valid JSON output. $jsonError"
                }
            }

            $result = [PSCustomObject]@{
                Analysis = $analysisText
                AnalysisJson = $analysisJson
                JsonValid = $jsonValid
                JsonError = $jsonError
                Model = $Model
                AnalysisDurationSeconds = [math]::Round($duration, 2)
                Timestamp = $startTime
                ReportSavedTo = $null
            }

            if ($SaveReport) {
                Set-Content -Path $OutputPath -Value $analysisText -Encoding utf8
                Write-Host "Report saved to: $OutputPath" -ForegroundColor Cyan
                $result.ReportSavedTo = $OutputPath
            }

            Write-Host $analysisText
            return $result
        }
        catch {
            Write-Error "Analysis failed: $_"
            throw
        }
    }

    end {
        Write-Verbose "PC diagnosis analysis completed"
    }
}
