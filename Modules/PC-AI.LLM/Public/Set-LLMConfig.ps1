#Requires -PSEdition Core

function Set-LLMConfig {
    <#
    .SYNOPSIS
        Configures LLM module settings

    .DESCRIPTION
        Sets and persists configuration for the PC-AI.LLM module including default model,
        API endpoints, timeouts, and other operational parameters.
        Windows publication preserves private original-byte custody. Other platforms
        reject writes until atomic exchange and metadata preservation are qualified;
        read-only inspection remains available.

    .PARAMETER DefaultModel
        Default model to use for LLM requests

    .PARAMETER PcaiInferenceApiUrl
        pcai-inference API endpoint URL

    .PARAMETER OllamaApiUrl
        Legacy alias for PcaiInferenceApiUrl (kept for compatibility)

    .PARAMETER LMStudioApiUrl
        LM Studio API endpoint URL

    .PARAMETER OllamaPath
        Legacy path to Ollama executable (kept for compatibility)

    .PARAMETER DefaultTimeout
        Default timeout in seconds for API requests

    .PARAMETER ShowConfig
        Display current configuration without making changes

    .PARAMETER Reset
        Restore model, endpoint and timeout values captured at module startup,
        including the machine's project config and settings.json overrides.
        Provider order and unrelated model/runtime tuning remain unchanged.

    .EXAMPLE
        Set-LLMConfig -DefaultModel "deepseek-r1:8b"
        Changes the default model

    .EXAMPLE
        Set-LLMConfig -DefaultTimeout 180
        Sets default timeout to 3 minutes

    .EXAMPLE
        Set-LLMConfig -ShowConfig
        Displays current configuration

    .EXAMPLE
        Set-LLMConfig -Reset
        Restores the configured machine model, endpoints and timeout.

    .OUTPUTS
        PSCustomObject with current configuration
    #>
    [CmdletBinding(DefaultParameterSetName = 'SetConfig')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(ParameterSetName = 'SetConfig')]
        [string]$DefaultModel,

        [Parameter(ParameterSetName = 'SetConfig')]
        [ValidatePattern('^https?://')]
        [string]$PcaiInferenceApiUrl,

        [Parameter(ParameterSetName = 'SetConfig')]
        [ValidatePattern('^https?://')]
        [string]$OllamaApiUrl,

        [Parameter(ParameterSetName = 'SetConfig')]
        [ValidatePattern('^https?://')]
        [string]$LMStudioApiUrl,

        [Parameter(ParameterSetName = 'SetConfig')]
        [ValidateScript({ Test-Path $_ -PathType Leaf })]
        [string]$OllamaPath,

        [Parameter(ParameterSetName = 'SetConfig')]
        [ValidateRange(1, 600)]
        [int]$DefaultTimeout,

        [Parameter(ParameterSetName = 'ShowConfig')]
        [switch]$ShowConfig,

        [Parameter(ParameterSetName = 'Reset')]
        [switch]$Reset
    )

    begin {
        Write-Verbose "Configuring LLM module settings..."

        $configPath = $script:ModuleConfig.ConfigPath
        $projectConfigPath = $script:ModuleConfig.ProjectConfigPath
        $targetConfigPath = if (-not [string]::IsNullOrWhiteSpace($projectConfigPath)) { $projectConfigPath } else { $configPath }

        # Reject inherited invalid fallback entries before changing memory or disk.
        # Read-only inspection remains available for diagnosing a broken config.
        if (-not $ShowConfig -and ($Reset -or @('DefaultModel', 'PcaiInferenceApiUrl', 'OllamaApiUrl', 'LMStudioApiUrl', 'OllamaPath', 'DefaultTimeout').Where({ $PSBoundParameters.ContainsKey($_) }).Count -gt 0)) {
            $providerOrder = if ($script:ModuleConfig.ProviderOrder -and $script:ModuleConfig.ProviderOrder.Count -gt 0) {
                @($script:ModuleConfig.ProviderOrder)
            } else {
                @('ollama', 'pcai-inference')
            }
            Assert-LLMProviderOrder -Order $providerOrder
        }

        # These setters replace values; a shallow snapshot preserves prior settings
        # when serialization or persistence fails, without changing machine defaults.
        $previousConfig = $script:ModuleConfig.Clone()
        $pendingConfig = $previousConfig.Clone()

        # Default configuration snapshot captured at module load time.
        if ($script:ModuleDefaults -and $script:ModuleDefaults.Count -gt 0) {
            $defaultConfig = @{}
            foreach ($key in $script:ModuleDefaults.Keys) {
                $defaultConfig[$key] = $script:ModuleDefaults[$key]
            }
        } else {
            $defaultConfig = @{
                PcaiInferenceApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl
                OllamaApiUrl = $script:ModuleConfig.OllamaApiUrl
                LMStudioApiUrl = $script:ModuleConfig.LMStudioApiUrl
                DefaultModel = $script:ModuleConfig.DefaultModel
                DefaultTimeout = $script:ModuleConfig.DefaultTimeout
            }
        }

        function Initialize-ConfigProperty {
            param(
                [Parameter(Mandatory)]
                [psobject]$Object,
                [Parameter(Mandatory)]
                [string]$Name,
                [Parameter(Mandatory)]
                [object]$DefaultValue
            )

            if (-not $Object.PSObject.Properties[$Name]) {
                $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $DefaultValue -Force
            }
            return $Object.PSObject.Properties[$Name].Value
        }

        function Set-ConfigValue {
            param(
                [Parameter(Mandatory)]
                [psobject]$Object,
                [Parameter(Mandatory)]
                [string]$Name,
                [Parameter(Mandatory)]
                [AllowNull()]
                [object]$Value
            )

            if ($Object.PSObject.Properties[$Name]) {
                $Object.PSObject.Properties[$Name].Value = $Value
            } else {
                $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $Value -Force
            }
        }

        function Save-CanonicalConfig {
            param(
                [Parameter(Mandatory)]
                [string]$TargetPath
            )

            if ([string]::IsNullOrWhiteSpace($TargetPath)) {
                throw 'Cannot persist LLM configuration because no target config path is defined.'
            }

            $fallbackOrder = if ($pendingConfig.ProviderOrder -and $pendingConfig.ProviderOrder.Count -gt 0) {
                @($pendingConfig.ProviderOrder)
            } else {
                @('ollama', 'pcai-inference')
            }
            Assert-LLMProviderOrder -Order $fallbackOrder

            $snapshot = Read-LLMConfigSnapshot -Path $TargetPath
            $projectConfig = $snapshot.Configuration

            $providers = Initialize-ConfigProperty -Object $projectConfig -Name 'providers' -DefaultValue ([PSCustomObject]@{})
            $pcaiProvider = Initialize-ConfigProperty -Object $providers -Name 'pcai-inference' -DefaultValue ([PSCustomObject]@{})
            $functionGemmaProvider = Initialize-ConfigProperty -Object $providers -Name 'functiongemma' -DefaultValue ([PSCustomObject]@{})
            $ollamaProvider = Initialize-ConfigProperty -Object $providers -Name 'ollama' -DefaultValue ([PSCustomObject]@{})
            $lmstudioProvider = Initialize-ConfigProperty -Object $providers -Name 'lmstudio' -DefaultValue ([PSCustomObject]@{})
            $vllmProvider = Initialize-ConfigProperty -Object $providers -Name 'vllm' -DefaultValue ([PSCustomObject]@{})

            $router = Initialize-ConfigProperty -Object $projectConfig -Name 'router' -DefaultValue ([PSCustomObject]@{})
            $ollamaRuntime = Initialize-ConfigProperty -Object $projectConfig -Name 'ollama' -DefaultValue ([PSCustomObject]@{})
            Set-ConfigValue -Object $pcaiProvider -Name 'baseUrl' -Value $pendingConfig.PcaiInferenceApiUrl
            Set-ConfigValue -Object $pcaiProvider -Name 'defaultModel' -Value $pendingConfig.DefaultModel
            Set-ConfigValue -Object $pcaiProvider -Name 'timeout' -Value ([int]($pendingConfig.DefaultTimeout * 1000))

            Set-ConfigValue -Object $functionGemmaProvider -Name 'baseUrl' -Value $pendingConfig.RouterApiUrl
            Set-ConfigValue -Object $functionGemmaProvider -Name 'defaultModel' -Value $pendingConfig.RouterModel

            Set-ConfigValue -Object $router -Name 'baseUrl' -Value $pendingConfig.RouterApiUrl
            Set-ConfigValue -Object $router -Name 'model' -Value $pendingConfig.RouterModel
            if (-not $router.toolsPath) {
                Set-ConfigValue -Object $router -Name 'toolsPath' -Value 'Config/pcai-tools.json'
            }

            Set-ConfigValue -Object $ollamaProvider -Name 'baseUrl' -Value $pendingConfig.OllamaApiUrl
            Set-ConfigValue -Object $ollamaProvider -Name 'defaultModel' -Value $pendingConfig.DefaultModel
            Set-ConfigValue -Object $ollamaProvider -Name 'timeout' -Value ([int]($pendingConfig.DefaultTimeout * 1000))
            Set-ConfigValue -Object $ollamaRuntime -Name 'base_url' -Value $pendingConfig.OllamaApiUrl
            Set-ConfigValue -Object $ollamaRuntime -Name 'model' -Value $pendingConfig.DefaultModel
            Set-ConfigValue -Object $ollamaRuntime -Name 'timeout_ms' -Value ([int]($pendingConfig.DefaultTimeout * 1000))
            if (-not $ollamaRuntime.toolInvokerPath) {
                Set-ConfigValue -Object $ollamaRuntime -Name 'toolInvokerPath' -Value 'Tools/Invoke-PcaiMappedTool.ps1'
            }
            Set-ConfigValue -Object $lmstudioProvider -Name 'baseUrl' -Value $pendingConfig.LMStudioApiUrl
            Set-ConfigValue -Object $vllmProvider -Name 'baseUrl' -Value $pendingConfig.VLLMApiUrl
            Set-ConfigValue -Object $vllmProvider -Name 'defaultModel' -Value $pendingConfig.VLLMModel

            Set-ConfigValue -Object $projectConfig -Name 'fallbackOrder' -Value $fallbackOrder

            Save-LLMConfigAtomically -Configuration $projectConfig -Snapshot $snapshot
        }
    }

    process {
        if ($Reset) {
            Write-Host "Resetting configuration to defaults..." -ForegroundColor Yellow

            # Reset to defaults
            foreach ($key in $defaultConfig.Keys) {
                $pendingConfig[$key] = $defaultConfig[$key]
            }

            # Persist to canonical config file
            try {
                Save-CanonicalConfig -TargetPath $targetConfigPath
                $script:ModuleConfig = $pendingConfig
                Write-Host "Configuration reset successfully" -ForegroundColor Green
            }
            catch {
                $script:ModuleConfig = $previousConfig
                Write-Error -Message "Failed to save configuration: $_" -Exception $_.Exception
            }
        }
        elseif ($ShowConfig) {
            # Display current configuration
            Write-Host "`nCurrent LLM Configuration:" -ForegroundColor Cyan
            Write-Host ("=" * 60) -ForegroundColor Gray

            foreach ($key in $script:ModuleConfig.Keys | Sort-Object) {
                $value = $script:ModuleConfig[$key]
                Write-Host "$($key.PadRight(20)): " -NoNewline -ForegroundColor Yellow
                Write-Host $value -ForegroundColor White
            }

            Write-Host ("=" * 60) -ForegroundColor Gray
        }
        else {
            # Update configuration
            $updated = $false

            if ($PSBoundParameters.ContainsKey('DefaultModel')) {
                # Verify model exists if pcai-inference is available
                if (Test-PcaiInferenceConnection) {
                    $availableModels = Get-OllamaModels
                    $modelExists = $availableModels | Where-Object { $_.Name -eq $DefaultModel }

                    if (-not $modelExists -and $availableModels.Count -gt 0) {
                        Write-Warning "Model '$DefaultModel' not found in pcai-inference. Available models: $($availableModels.Name -join ', ')"
                    }
                }

                $pendingConfig.DefaultModel = $DefaultModel
                Write-Host "Default model set to: $DefaultModel" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('PcaiInferenceApiUrl')) {
                $pendingConfig.PcaiInferenceApiUrl = $PcaiInferenceApiUrl
                $pendingConfig.OllamaApiUrl = $PcaiInferenceApiUrl
                Write-Host "pcai-inference API URL set to: $PcaiInferenceApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('OllamaApiUrl')) {
                $pendingConfig.OllamaApiUrl = $OllamaApiUrl
                $pendingConfig.PcaiInferenceApiUrl = $OllamaApiUrl
                Write-Host "pcai-inference API URL set to: $OllamaApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('LMStudioApiUrl')) {
                $pendingConfig.LMStudioApiUrl = $LMStudioApiUrl
                Write-Host "LM Studio API URL set to: $LMStudioApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('OllamaPath')) {
                $pendingConfig.OllamaPath = $OllamaPath
                Write-Host "Legacy Ollama path set to: $OllamaPath" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('DefaultTimeout')) {
                $pendingConfig.DefaultTimeout = $DefaultTimeout
                Write-Host "Default timeout set to: $DefaultTimeout seconds" -ForegroundColor Green
                $updated = $true
            }

            # Save configuration if anything was updated
            if ($updated) {
                try {
                    Save-CanonicalConfig -TargetPath $targetConfigPath
                    $script:ModuleConfig = $pendingConfig
                    Write-Verbose "Configuration saved to: $targetConfigPath"
                }
                catch {
                    $script:ModuleConfig = $previousConfig
                    Write-Error -Message "Failed to save configuration: $_" -Exception $_.Exception
                }
            }
            else {
                Write-Warning "No configuration changes specified. Use -ShowConfig to view current settings."
            }
        }

        # Return current configuration
        return [PSCustomObject]@{
            PcaiInferenceApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl
            OllamaApiUrl = $script:ModuleConfig.OllamaApiUrl
            OllamaPath = $script:ModuleConfig.OllamaPath
            LMStudioApiUrl = $script:ModuleConfig.LMStudioApiUrl
            DefaultModel = $script:ModuleConfig.DefaultModel
            DefaultTimeout = $script:ModuleConfig.DefaultTimeout
            ConfigPath = $targetConfigPath
            LastUpdated = Get-Date
        }
    }

    end {
        Write-Verbose "LLM configuration completed"
    }
}

