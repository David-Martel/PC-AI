function Get-PcaiRuntimeDefaults {
    [CmdletBinding()]
    param(
        [string]$ConfigPath
    )

    if (Get-Command Get-PcaiRuntimeConfig -ErrorAction SilentlyContinue) {
        $runtimeConfig = Get-PcaiRuntimeConfig -ProjectRoot (Split-Path -Parent $PSScriptRoot) -ConfigPath $ConfigPath
        return [pscustomobject]@{
            ProjectRoot = $runtimeConfig.ProjectRoot
            ConfigPath = $runtimeConfig.ConfigPath
            PcaiInferenceUrl = $runtimeConfig.PcaiInferenceUrl
            FunctionGemmaUrl = $runtimeConfig.FunctionGemmaUrl
            OllamaBaseUrl = $runtimeConfig.OllamaBaseUrl
            vLLMBaseUrl = $runtimeConfig.vLLMBaseUrl
            NativeDllSearchPaths = @($runtimeConfig.NativeDllSearchPaths)
        }
    }

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $projectRoot = Split-Path -Parent $moduleRoot
    $resolvedConfigPath = if ($ConfigPath) { $ConfigPath } else { Join-Path $projectRoot 'Config\llm-config.json' }
    return [pscustomobject]@{
        ProjectRoot = $projectRoot
        ConfigPath = $resolvedConfigPath
        PcaiInferenceUrl = 'http://127.0.0.1:8080'
        FunctionGemmaUrl = 'http://127.0.0.1:8000'
        OllamaBaseUrl = 'http://127.0.0.1:11434'
        vLLMBaseUrl = 'http://127.0.0.1:8001'
        NativeDllSearchPaths = @()
    }
}
