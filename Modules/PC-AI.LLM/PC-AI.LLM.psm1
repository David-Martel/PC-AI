#Requires -Version 5.1

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

# Module-level variables
$script:ModuleConfig = @{
    # Primary Rust inference (pcai-inference)
    PcaiInferenceApiUrl = 'http://127.0.0.1:8080'
    DefaultModel        = 'pcai-inference'
    DefaultTimeout      = 120

    # Rust FunctionGemma router
    RouterApiUrl = 'http://127.0.0.1:8000'
    RouterModel  = 'functiongemma-270m-it'

    # Provider order (auto fallback)
    ProviderOrder = @('pcai-inference')

    # Legacy keys retained for compatibility (mapped to Rust backends)
    OllamaPath     = ''
    OllamaApiUrl   = 'http://127.0.0.1:8080'
    LMStudioApiUrl = ''
    VLLMApiUrl     = 'http://127.0.0.1:8000'
    VLLMModel      = 'functiongemma-270m-it'

    ConfigPath = Join-Path -Path $ModuleRoot -ChildPath 'llm-config.json'
}

# Load configuration if exists (module config)
if (Test-Path -Path $script:ModuleConfig.ConfigPath) {
    try {
        $savedConfig = Get-Content -Path $script:ModuleConfig.ConfigPath -Raw | ConvertFrom-Json
        foreach ($key in $savedConfig.PSObject.Properties.Name) {
            if ($script:ModuleConfig.ContainsKey($key)) {
                $script:ModuleConfig[$key] = $savedConfig.$key
            }
        }
        Write-Verbose "Loaded configuration from $($script:ModuleConfig.ConfigPath)"
    } catch {
        Write-Warning "Failed to load module configuration: $_"
    }
}

# Prefer project-level config if present (PC_AI\Config\llm-config.json)
$projectRoot = Split-Path -Parent (Split-Path -Parent $ModuleRoot)
$projectConfigPath = Join-Path -Path $projectRoot -ChildPath 'Config\llm-config.json'
if (Test-Path -Path $projectConfigPath) {
    try {
        $projectConfig = Get-Content -Path $projectConfigPath -Raw | ConvertFrom-Json

        # New Rust backends
        if ($projectConfig.providers.'pcai-inference'.baseUrl) {
            $script:ModuleConfig.PcaiInferenceApiUrl = $projectConfig.providers.'pcai-inference'.baseUrl
            $script:ModuleConfig.OllamaApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl
        }
        if ($projectConfig.providers.'pcai-inference'.defaultModel) {
            $script:ModuleConfig.DefaultModel = $projectConfig.providers.'pcai-inference'.defaultModel
        }
        if ($projectConfig.providers.'pcai-inference'.timeout) {
            $script:ModuleConfig.DefaultTimeout = [math]::Ceiling($projectConfig.providers.'pcai-inference'.timeout / 1000)
        }

        if ($projectConfig.providers.functiongemma.baseUrl) {
            $script:ModuleConfig.RouterApiUrl = $projectConfig.providers.functiongemma.baseUrl
            $script:ModuleConfig.VLLMApiUrl = $script:ModuleConfig.RouterApiUrl
        }
        if ($projectConfig.providers.functiongemma.defaultModel) {
            $script:ModuleConfig.RouterModel = $projectConfig.providers.functiongemma.defaultModel
            $script:ModuleConfig.VLLMModel = $script:ModuleConfig.RouterModel
        }

        if ($projectConfig.router.baseUrl) {
            $script:ModuleConfig.RouterApiUrl = $projectConfig.router.baseUrl
            $script:ModuleConfig.VLLMApiUrl = $script:ModuleConfig.RouterApiUrl
        }
        if ($projectConfig.router.model) {
            $script:ModuleConfig.RouterModel = $projectConfig.router.model
            $script:ModuleConfig.VLLMModel = $script:ModuleConfig.RouterModel
        }

        if ($projectConfig.fallbackOrder) {
            $script:ModuleConfig.ProviderOrder = @($projectConfig.fallbackOrder)
        }

        # Legacy provider support (if present)
        if ($projectConfig.providers.ollama.baseUrl) {
            $script:ModuleConfig.OllamaApiUrl = $projectConfig.providers.ollama.baseUrl
        }
        if ($projectConfig.providers.lmstudio.baseUrl) {
            $script:ModuleConfig.LMStudioApiUrl = $projectConfig.providers.lmstudio.baseUrl
        }
        if ($projectConfig.providers.vllm.baseUrl) {
            $script:ModuleConfig.VLLMApiUrl = $projectConfig.providers.vllm.baseUrl
        }
        if ($projectConfig.providers.vllm.defaultModel) {
            $script:ModuleConfig.VLLMModel = $projectConfig.providers.vllm.defaultModel
        }

        $script:ModuleConfig.ProjectConfigPath = $projectConfigPath
        Write-Verbose "Loaded project configuration from $projectConfigPath"
    } catch {
        Write-Warning "Failed to load project configuration: $_"
    }
}

# Finally, check settings.json for direct overrides
$settingsPath = Join-Path -Path $projectRoot -ChildPath 'Config\settings.json'
if (Test-Path -Path $settingsPath) {
    try {
        $settings = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        if ($settings.llm) {
            if ($settings.llm.activeModel) { $script:ModuleConfig.DefaultModel = $settings.llm.activeModel }
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
