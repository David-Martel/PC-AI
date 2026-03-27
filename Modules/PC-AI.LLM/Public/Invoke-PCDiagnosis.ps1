#Requires -PSEdition Core

# Private helper: validate and parse JSON from LLM diagnosis response.
# Returns a hashtable with keys: AnalysisJson, JsonValid, JsonError.
function script:ConvertFrom-DiagnosisResponse {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Content,

        [Parameter()]
        [bool]$EnforceJson
    )

    $result = @{
        AnalysisJson = $null
        JsonValid    = $false
        JsonError    = $null
    }

    try {
        $result.AnalysisJson = ConvertFrom-LLMJson -Content $Content -Strict
        $result.JsonValid    = $true
    } catch {
        $result.JsonError = $_.Exception.Message
        if ($EnforceJson) {
            throw "Diagnose mode requires valid JSON output. $($result.JsonError)"
        }
    }

    return $result
}

# Private helper: persist analysis text to disk and emit a status message.
# Returns the resolved output path, or $null when -SaveReport is not requested.
function script:Save-DiagnosisReport {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Content,

        [Parameter(Mandatory)]
        [bool]$ShouldSave,

        [Parameter(Mandatory)]
        [string]$OutputPath
    )

    if (-not $ShouldSave) {
        return $null
    }

    Set-Content -Path $OutputPath -Value $Content -Encoding utf8
    Write-Host "Report saved to: $OutputPath" -ForegroundColor Cyan
    return $OutputPath
}

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

        $promptSplat = @{
            DiagnosticText   = $diagnosticText
            TimeoutSeconds   = $TimeoutSeconds
        }
        
        if ($UseRouter) {
            $promptSplat.UseRouter          = $true
            $promptSplat.RouterBaseUrl      = $RouterBaseUrl
            $promptSplat.RouterModel        = $RouterModel
            $promptSplat.RouterMaxCalls     = $RouterMaxCalls
            $promptSplat.RouterExecuteTools = [bool]$RouterExecuteTools
            
            if (-not [string]::IsNullOrWhiteSpace($RouterToolsPath)) {
                $promptSplat.RouterToolsPath = $RouterToolsPath
            }
        }

        $prompts = Get-PCDiagnosisPrompt @promptSplat

        Write-Host "Analyzing diagnostic report with $Model..." -ForegroundColor Cyan
        $startTime = Get-Date

        try {
            $messages = @(
                @{ role = 'system'; content = $prompts.SystemPrompt }
                @{ role = 'user'; content = $prompts.UserPrompt }
            )

            $response = Invoke-LLMChatWithFallback -Messages $messages -Model $Model -Temperature $Temperature -TimeoutSeconds $TimeoutSeconds
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds

            $analysisText = $response.message.content

            if (-not $PSBoundParameters.ContainsKey('EnforceJson')) {
                $EnforceJson = $true
            }

            $jsonResult = ConvertFrom-DiagnosisResponse -Content $analysisText -EnforceJson ([bool]$EnforceJson)

            $result = [PSCustomObject]@{
                Analysis                = $analysisText
                AnalysisJson            = $jsonResult.AnalysisJson
                JsonValid               = $jsonResult.JsonValid
                JsonError               = $jsonResult.JsonError
                Model                   = $Model
                AnalysisDurationSeconds = [math]::Round($duration, 2)
                Timestamp               = $startTime
                ReportSavedTo           = (Save-DiagnosisReport -Content $analysisText -ShouldSave ([bool]$SaveReport) -OutputPath $OutputPath)
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
