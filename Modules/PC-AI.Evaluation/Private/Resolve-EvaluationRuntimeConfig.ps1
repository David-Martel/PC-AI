function Initialize-EvaluationRuntimeDefaults {
    [CmdletBinding()]
    param()

    # Query loaded functions without triggering module autoload for an absent optional helper.
    $resolver = Get-Item -LiteralPath 'Function:\Get-PcaiRuntimeConfig' -ErrorAction SilentlyContinue
    if ($resolver) {
        $runtimeConfig = & $resolver -ProjectRoot (Split-Path -Parent $PSScriptRoot)
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
