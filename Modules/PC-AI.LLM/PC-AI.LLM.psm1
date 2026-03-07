#Requires -PSEdition Core

<#
.SYNOPSIS
    PC-AI.LLM PowerShell Module Loader
.DESCRIPTION
    Loads the PC-AI.LLM module for pcai-inference integration with PC diagnostics
#>

# Get module paths
$ModuleRoot = $PSScriptRoot
$PrivatePath = Join-Path -Path $ModuleRoot -ChildPath 'Private'
$PublicPath = Join-Path -Path $ModuleRoot -ChildPath 'Public'
$sharedCacheHelper = Join-Path (Split-Path -Parent $ModuleRoot) 'PC-AI.Common\Public\Get-PcaiSharedCache.ps1'
$sharedRuntimeHelper = Join-Path (Split-Path -Parent $ModuleRoot) 'PC-AI.Common\Public\Get-PcaiRuntimeConfig.ps1'
if (Test-Path $sharedCacheHelper) {
    . $sharedCacheHelper
}
if (Test-Path $sharedRuntimeHelper) {
    . $sharedRuntimeHelper
}
$projectRoot = if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
    Resolve-PcaiRepoRoot -StartPath $ModuleRoot
} else {
    Split-Path -Parent (Split-Path -Parent $ModuleRoot)
}
$runtimeConfig = if (Get-Command Get-PcaiRuntimeConfig -ErrorAction SilentlyContinue) {
    Get-PcaiRuntimeConfig -ProjectRoot $projectRoot
} else {
    $null
}
$resolvedConfigPath = if ($runtimeConfig -and $runtimeConfig.ConfigPath) {
    $runtimeConfig.ConfigPath
} else {
    Join-Path -Path $projectRoot -ChildPath 'Config\llm-config.json'
}

# Module-level variables
$script:ModuleConfig = @{
    ProjectRoot          = $projectRoot
    ConfigPath           = $resolvedConfigPath
    ProjectConfigPath    = $resolvedConfigPath
    ToolsPath            = if ($runtimeConfig -and $runtimeConfig.ToolsPath) { $runtimeConfig.ToolsPath } else { Join-Path $projectRoot 'Config\pcai-tools.json' }

    # Legacy HTTP inference remains available as a fallback only.
    PcaiInferenceApiUrl  = if ($runtimeConfig -and $runtimeConfig.PcaiInferenceUrl) { $runtimeConfig.PcaiInferenceUrl } else { 'http://127.0.0.1:18080' }
    PcaiInferenceModel   = if ($runtimeConfig -and $runtimeConfig.PcaiInferenceModel) { $runtimeConfig.PcaiInferenceModel } else { 'llama.cpp' }
    PcaiInferenceTimeout = if ($runtimeConfig -and $runtimeConfig.PcaiInferenceTimeoutMs) { [math]::Ceiling([double]$runtimeConfig.PcaiInferenceTimeoutMs / 1000.0) } else { 120 }

    # Native Ollama runner configuration.
    OllamaPath           = ''
    OllamaApiUrl         = if ($runtimeConfig -and $runtimeConfig.OllamaBaseUrl) { $runtimeConfig.OllamaBaseUrl } else { 'http://127.0.0.1:11434' }
    OllamaToolModel      = if ($runtimeConfig -and $runtimeConfig.OllamaToolModel) { $runtimeConfig.OllamaToolModel } else { '' }
    OllamaSummaryModel   = if ($runtimeConfig -and $runtimeConfig.OllamaSummaryModel) { $runtimeConfig.OllamaSummaryModel } else { '' }
    OllamaTimeout        = if ($runtimeConfig -and $runtimeConfig.OllamaTimeoutMs) { [math]::Ceiling([double]$runtimeConfig.OllamaTimeoutMs / 1000.0) } else { 90 }
    OllamaCliSearchPaths = if ($runtimeConfig -and $runtimeConfig.OllamaCliSearchPaths) { @($runtimeConfig.OllamaCliSearchPaths) } else { @() }
    OllamaToolInvokerPath = if ($runtimeConfig -and $runtimeConfig.OllamaToolInvokerPath) { $runtimeConfig.OllamaToolInvokerPath } else { Join-Path $projectRoot 'Tools\Invoke-PcaiMappedTool.ps1' }

    # Rust FunctionGemma router
    RouterApiUrl         = if ($runtimeConfig -and $runtimeConfig.RouterBaseUrl) { $runtimeConfig.RouterBaseUrl } else { 'http://127.0.0.1:8000' }
    RouterModel          = if ($runtimeConfig -and $runtimeConfig.RouterModel) { $runtimeConfig.RouterModel } else { 'functiongemma-270m-it' }

    LMStudioApiUrl       = if ($runtimeConfig -and $runtimeConfig.LMStudioApiUrl) { $runtimeConfig.LMStudioApiUrl } else { 'http://127.0.0.1:1234' }
    VLLMApiUrl           = if ($runtimeConfig -and $runtimeConfig.vLLMBaseUrl) { $runtimeConfig.vLLMBaseUrl } elseif ($runtimeConfig -and $runtimeConfig.RouterBaseUrl) { $runtimeConfig.RouterBaseUrl } else { 'http://127.0.0.1:8000' }
    VLLMModel            = if ($runtimeConfig -and $runtimeConfig.vLLMModel) { $runtimeConfig.vLLMModel } elseif ($runtimeConfig -and $runtimeConfig.RouterModel) { $runtimeConfig.RouterModel } else { 'functiongemma-270m-it' }

    ProviderOrder        = if ($runtimeConfig -and $runtimeConfig.FallbackOrder -and $runtimeConfig.FallbackOrder.Count -gt 0) { @($runtimeConfig.FallbackOrder) } else { @('ollama', 'pcai-inference') }
    DefaultModel         = if ($runtimeConfig -and $runtimeConfig.OllamaModel) { $runtimeConfig.OllamaModel } else { 'qwen2.5-coder:3b' }
    DefaultTimeout       = if ($runtimeConfig -and $runtimeConfig.OllamaTimeoutMs) { [math]::Ceiling([double]$runtimeConfig.OllamaTimeoutMs / 1000.0) } else { 90 }
}

# Finally, check settings.json for direct overrides
$settingsPath = Join-Path -Path $projectRoot -ChildPath 'Config\settings.json'
if (Test-Path -Path $settingsPath) {
    try {
        $settingsStamp = if (Get-Command Get-PcaiDependencyStamp -ErrorAction SilentlyContinue) {
            Get-PcaiDependencyStamp -InputObject @($settingsPath)
        } else {
            $null
        }
        $settings = if (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
            Get-PcaiSharedCacheEntry -Namespace 'pcai-llm' -Key "settings::$settingsPath" -DependencyStamp $settingsStamp
        } else {
            $null
        }
        if (-not $settings) {
            $settings = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
            if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
                Set-PcaiSharedCacheEntry -Namespace 'pcai-llm' -Key "settings::$settingsPath" -Value $settings -DependencyStamp $settingsStamp | Out-Null
            }
        }
        if ($settings.llm) {
            if ($settings.llm.activeModel) { $script:ModuleConfig.DefaultModel = $settings.llm.activeModel }
            if ($settings.llm.toolModel) { $script:ModuleConfig.OllamaToolModel = $settings.llm.toolModel }
            if ($settings.llm.summaryModel) { $script:ModuleConfig.OllamaSummaryModel = $settings.llm.summaryModel }
            if ($settings.llm.routerUrl) {
                $script:ModuleConfig.RouterApiUrl = $settings.llm.routerUrl
                $script:ModuleConfig.VLLMApiUrl = $settings.llm.routerUrl
            }
            if ($settings.llm.routerModel) {
                $script:ModuleConfig.RouterModel = $settings.llm.routerModel
                $script:ModuleConfig.VLLMModel = $settings.llm.routerModel
            }
            if ($settings.llm.timeoutSeconds) { $script:ModuleConfig.DefaultTimeout = $settings.llm.timeoutSeconds }
            if ($settings.llm.activeProvider) {
                # Ensure active provider is at the front of the list
                $providers = @($settings.llm.activeProvider)
                if ($script:ModuleConfig.ProviderOrder) {
                    $providers += ($script:ModuleConfig.ProviderOrder | Where-Object { $_ -ne $settings.llm.activeProvider })
                }
                $script:ModuleConfig.ProviderOrder = $providers
            }
            Write-Verbose "Applied LLM overrides from settings.json"
        }
    } catch {
        Write-Warning "Failed to load overrides from settings.json: $_"
    }
}

# Capture startup defaults after project/settings configuration is applied.
# Set-LLMConfig -Reset uses this snapshot instead of hardcoded endpoints.
$script:ModuleDefaults = @{}
foreach ($key in @('PcaiInferenceApiUrl', 'OllamaApiUrl', 'LMStudioApiUrl', 'DefaultModel', 'DefaultTimeout')) {
    if ($script:ModuleConfig.ContainsKey($key)) {
        $script:ModuleDefaults[$key] = $script:ModuleConfig[$key]
    }
}

# Dot source private functions
if (Test-Path -Path $PrivatePath) {
    Get-ChildItem -Path $PrivatePath -Filter '*.ps1' -Recurse | ForEach-Object {
        try {
            . $_.FullName
            Write-Verbose "Loaded private function: $($_.Name)"
        } catch {
            Write-Error "Failed to load private function $($_.Name): $_"
        }
    }
}

# Dot source public functions
if (Test-Path -Path $PublicPath) {
    Get-ChildItem -Path $PublicPath -Filter '*.ps1' -Recurse | ForEach-Object {
        try {
            . $_.FullName
            Write-Verbose "Loaded public function: $($_.Name)"
        } catch {
            Write-Error "Failed to load public function $($_.Name): $_"
        }
    }
}

# Export public functions
Export-ModuleMember -Function @(
    'Get-LLMStatus'
    'Send-OllamaRequest'
    'Invoke-LLMChat'
    'Invoke-LLMChatRouted'
    'Invoke-LLMChatTui'
    'Invoke-FunctionGemmaReAct'
    'Invoke-FunctionGemmaDataset'
    'Invoke-FunctionGemmaTokenCache'
    'Invoke-FunctionGemmaTrain'
    'Invoke-FunctionGemmaEval'
    'Invoke-FunctionGemmaTests'
    'Invoke-PCDiagnosis'
    'Set-LLMConfig'
    'Set-LLMProviderOrder'
    'Invoke-SmartDiagnosis'
    'Invoke-NativeSearch'
    'Invoke-DocSearch'
    'Get-SystemInfoTool'
    'Invoke-LogSearch'
    'Resolve-PcaiEndpoint'
    'Test-OllamaConnection'
    'Test-OpenAIConnection'
)

Write-Verbose 'PC-AI.LLM module loaded successfully'
