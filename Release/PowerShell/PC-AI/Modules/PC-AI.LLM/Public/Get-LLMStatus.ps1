#Requires -Version 5.1

function Get-LLMStatus {
    <#
    .SYNOPSIS
        Checks the status of pcai-inference LLM service and available models

    .DESCRIPTION
        Verifies pcai-inference API connectivity and lists available models.
        Optionally checks FunctionGemma router and other OpenAI-compatible providers.

    .PARAMETER IncludeLMStudio
        Include LM Studio status check in the output

    .PARAMETER IncludeVLLM
        Include vLLM (OpenAI-compatible) status check in the output

    .PARAMETER TestConnection
        Perform connectivity tests to the API endpoints

    .EXAMPLE
        Get-LLMStatus
        Returns basic pcai-inference status and model list

    .EXAMPLE
        Get-LLMStatus -IncludeLMStudio -TestConnection
        Returns comprehensive status including router and connectivity tests

    .OUTPUTS
        PSCustomObject with status information
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$IncludeLMStudio,

        [Parameter()]
        [switch]$IncludeVLLM,

        [Parameter()]
        [switch]$TestConnection
    )

    begin {
        Write-Verbose 'Checking LLM service status...'
    }

    process {
        $status = [PSCustomObject]@{
            Timestamp       = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            PcaiInference   = [PSCustomObject]@{
                ApiUrl       = $script:ModuleConfig.PcaiInferenceApiUrl
                ApiConnected = $false
                Models       = @()
                DefaultModel = $script:ModuleConfig.DefaultModel
            }
            Router          = [PSCustomObject]@{
                ApiUrl       = $script:ModuleConfig.RouterApiUrl
                ApiConnected = $false
                Model        = $script:ModuleConfig.RouterModel
            }
            Ollama          = $null
            VLLM            = $null
            LMStudio        = $null
            Recommendations = @()
            ActiveProvider  = $script:ModuleConfig.ProviderOrder[0]
            ActiveModel     = $script:ModuleConfig.DefaultModel
        }

        # Test pcai-inference connectivity
        $status.PcaiInference.ApiConnected = Test-PcaiInferenceConnection
        Write-Verbose "pcai-inference API connectivity: $($status.PcaiInference.ApiConnected)"

        # Get available models if API is connected
        if ($status.PcaiInference.ApiConnected) {
            $models = Get-OllamaModels
            $status.PcaiInference.Models = @($models)

            Write-Verbose "Found $($models.Count) pcai-inference models"

            # Check if default model exists
            $defaultModelExists = $models | Where-Object { $_.Name -eq $script:ModuleConfig.DefaultModel }
            if (-not $defaultModelExists -and $models.Count -gt 0) {
                $status.Recommendations += "Default model '$($script:ModuleConfig.DefaultModel)' not found. Available models: $($models.Name -join ', ')"
            }
        } else {
            $status.Recommendations += 'pcai-inference API is not accessible. Ensure the server is running.'
        }

        # Compatibility fields expected by older callers
        $status.PcaiInference | Add-Member -MemberType NoteProperty -Name Available -Value $status.PcaiInference.ApiConnected -Force
        $status.PcaiInference | Add-Member -MemberType NoteProperty -Name AvailableModels -Value $status.PcaiInference.Models -Force
        $status.PcaiInference | Add-Member -MemberType NoteProperty -Name ModelsLoaded -Value ($status.PcaiInference.Models | ForEach-Object { $_.Name }) -Force

        # Alias legacy Ollama status to pcai-inference for compatibility
        $status.Ollama = $status.PcaiInference

        # Router status (FunctionGemma)
        if ($status.Router.ApiUrl) {
            $status.Router.ApiConnected = Test-OpenAIConnection -ApiUrl $status.Router.ApiUrl
            Write-Verbose "Router API connectivity: $($status.Router.ApiConnected)"
        }

        # Check LM Studio if requested
        if ($IncludeLMStudio) {
            $status.LMStudio = [PSCustomObject]@{
                ApiUrl       = $script:ModuleConfig.LMStudioApiUrl
                ApiConnected = $false
            }

            if ($TestConnection -or $status.PcaiInference.ApiConnected -eq $false) {
                $status.LMStudio.ApiConnected = Test-LMStudioConnection
                Write-Verbose "LM Studio API connectivity: $($status.LMStudio.ApiConnected)"

                if ($status.LMStudio.ApiConnected -and -not $status.PcaiInference.ApiConnected) {
                    $status.Recommendations += 'LM Studio is available as a fallback option.'
                }
            }
        }

        # Check vLLM if requested
        if ($IncludeVLLM) {
            $status.VLLM = [PSCustomObject]@{
                ApiUrl       = $script:ModuleConfig.VLLMApiUrl
                ApiConnected = $false
                DefaultModel = $script:ModuleConfig.VLLMModel
                ModelInfo    = $null
                Metrics      = $null
            }

            if ($TestConnection -or $status.PcaiInference.ApiConnected -eq $false) {
                $status.VLLM.ApiConnected = Test-OpenAIConnection -ApiUrl $script:ModuleConfig.VLLMApiUrl
                Write-Verbose "vLLM API connectivity: $($status.VLLM.ApiConnected)"

                if ($status.VLLM.ApiConnected -and -not $status.PcaiInference.ApiConnected) {
                    $status.Recommendations += 'vLLM is available as a fallback option.'
                }

                if ($status.VLLM.ApiConnected) {
                    $status.VLLM.ModelInfo = Get-VLLMModelInfo -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel
                    $status.VLLM.Metrics = Get-VLLMMetricsSnapshot -ApiUrl $script:ModuleConfig.VLLMApiUrl -ModelName $script:ModuleConfig.VLLMModel
                }
            }
        }

        # Overall health check
        $modelCount = @($status.PcaiInference.Models).Count
        if ($status.PcaiInference.ApiConnected -and $modelCount -gt 0) {
            Write-Verbose 'LLM system is healthy and ready'
        } else {
            Write-Warning 'LLM system is not fully operational. Check recommendations.'
        }

        return $status
    }

    end {
        Write-Verbose 'LLM status check complete'
    }
}
