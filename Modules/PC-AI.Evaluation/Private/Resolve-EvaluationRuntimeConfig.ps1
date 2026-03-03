function Initialize-EvaluationRuntimeDefaults {
    [CmdletBinding()]
    param()

    if (Get-Command Get-PcaiRuntimeConfig -ErrorAction SilentlyContinue) {
        $runtimeConfig = Get-PcaiRuntimeConfig -ProjectRoot (Split-Path -Parent $PSScriptRoot)
        if ($runtimeConfig.PcaiInferenceUrl) {
            $script:EvaluationConfig.HttpBaseUrl = $runtimeConfig.PcaiInferenceUrl
        }
        if ($runtimeConfig.OllamaBaseUrl) {
            $script:EvaluationConfig.OllamaBaseUrl = $runtimeConfig.OllamaBaseUrl
        }
        if ($runtimeConfig.OllamaModel) {
            $script:EvaluationConfig.OllamaModel = $runtimeConfig.OllamaModel
        }
    }
}

function Resolve-EvaluationBackendBaseUrl {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Backend,
        [string]$BaseUrl
    )

    if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
        return $BaseUrl
    }

    if ($Backend -eq 'ollama') {
        return $script:EvaluationConfig.OllamaBaseUrl
    }

    return $script:EvaluationConfig.HttpBaseUrl
}
