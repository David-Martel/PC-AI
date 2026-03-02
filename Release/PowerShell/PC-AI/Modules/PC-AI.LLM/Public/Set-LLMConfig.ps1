#Requires -Version 5.1

function Set-LLMConfig {
    <#
    .SYNOPSIS
        Configures LLM module settings

    .DESCRIPTION
        Sets and persists configuration for the PC-AI.LLM module including default model,
        API endpoints, timeouts, and other operational parameters.

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
        Reset configuration to default values

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
        Resets all settings to defaults

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
        $configPersistExclusions = @('ConfigPath', 'ProjectConfigPath')

        # Default configuration
        $defaultConfig = @{
            PcaiInferenceApiUrl = 'http://127.0.0.1:8080'
            OllamaApiUrl = 'http://127.0.0.1:8080'
            LMStudioApiUrl = 'http://localhost:1234'
            DefaultModel = 'pcai-inference'
            DefaultTimeout = 120
        }
    }

    process {
        if ($Reset) {
            Write-Host "Resetting configuration to defaults..." -ForegroundColor Yellow

            # Reset to defaults
            foreach ($key in $defaultConfig.Keys) {
                $script:ModuleConfig[$key] = $defaultConfig[$key]
            }

            # Save to file
            try {
                $persistConfig = @{}
                foreach ($key in $script:ModuleConfig.Keys) {
                    if ($key -in $configPersistExclusions) { continue }
                    $persistConfig[$key] = $script:ModuleConfig[$key]
                }
                $jsonContent = $persistConfig | ConvertTo-Json -Depth 10
                [System.IO.File]::WriteAllText($configPath, $jsonContent, [System.Text.Encoding]::UTF8)
                Write-Host "Configuration reset successfully" -ForegroundColor Green
            }
            catch {
                Write-Error "Failed to save configuration: $_"
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

                $script:ModuleConfig.DefaultModel = $DefaultModel
                Write-Host "Default model set to: $DefaultModel" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('PcaiInferenceApiUrl')) {
                $script:ModuleConfig.PcaiInferenceApiUrl = $PcaiInferenceApiUrl
                $script:ModuleConfig.OllamaApiUrl = $PcaiInferenceApiUrl
                Write-Host "pcai-inference API URL set to: $PcaiInferenceApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('OllamaApiUrl')) {
                $script:ModuleConfig.OllamaApiUrl = $OllamaApiUrl
                $script:ModuleConfig.PcaiInferenceApiUrl = $OllamaApiUrl
                Write-Host "pcai-inference API URL set to: $OllamaApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('LMStudioApiUrl')) {
                $script:ModuleConfig.LMStudioApiUrl = $LMStudioApiUrl
                Write-Host "LM Studio API URL set to: $LMStudioApiUrl" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('OllamaPath')) {
                $script:ModuleConfig.OllamaPath = $OllamaPath
                Write-Host "Legacy Ollama path set to: $OllamaPath" -ForegroundColor Green
                $updated = $true
            }

            if ($PSBoundParameters.ContainsKey('DefaultTimeout')) {
                $script:ModuleConfig.DefaultTimeout = $DefaultTimeout
                Write-Host "Default timeout set to: $DefaultTimeout seconds" -ForegroundColor Green
                $updated = $true
            }

            # Save configuration if anything was updated
            if ($updated) {
                try {
                    $persistConfig = @{}
                    foreach ($key in $script:ModuleConfig.Keys) {
                        if ($key -in $configPersistExclusions) { continue }
                        $persistConfig[$key] = $script:ModuleConfig[$key]
                    }
                    $jsonContent = $persistConfig | ConvertTo-Json -Depth 10
                    [System.IO.File]::WriteAllText($configPath, $jsonContent, [System.Text.Encoding]::UTF8)
                    Write-Verbose "Configuration saved to: $configPath"
                }
                catch {
                    Write-Error "Failed to save configuration: $_"
                }

                # Also update project-level config if available
                if ($projectConfigPath -and (Test-Path $projectConfigPath)) {
                    try {
                        $projectConfig = Get-Content -Path $projectConfigPath -Raw | ConvertFrom-Json
                        if ($PSBoundParameters.ContainsKey('DefaultModel')) {
                            if ($projectConfig.providers.'pcai-inference') {
                                $projectConfig.providers.'pcai-inference'.defaultModel = $DefaultModel
                            }
                        }
                        if ($PSBoundParameters.ContainsKey('PcaiInferenceApiUrl')) {
                            if ($projectConfig.providers.'pcai-inference') {
                                $projectConfig.providers.'pcai-inference'.baseUrl = $PcaiInferenceApiUrl
                            }
                        }
                        if ($PSBoundParameters.ContainsKey('OllamaApiUrl')) {
                            if ($projectConfig.providers.'pcai-inference') {
                                $projectConfig.providers.'pcai-inference'.baseUrl = $OllamaApiUrl
                            }
                        }
                        if ($PSBoundParameters.ContainsKey('LMStudioApiUrl')) {
                            $projectConfig.providers.lmstudio.baseUrl = $LMStudioApiUrl
                        }
                        if ($PSBoundParameters.ContainsKey('DefaultTimeout')) {
                            if ($projectConfig.providers.'pcai-inference') {
                                $projectConfig.providers.'pcai-inference'.timeout = ($DefaultTimeout * 1000)
                            }
                        }

                        $projectJson = $projectConfig | ConvertTo-Json -Depth 10
                        [System.IO.File]::WriteAllText($projectConfigPath, $projectJson, [System.Text.Encoding]::UTF8)
                        Write-Verbose "Project configuration saved to: $projectConfigPath"
                    }
                    catch {
                        Write-Warning "Failed to update project configuration: $_"
                    }
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
            ConfigPath = $configPath
            LastUpdated = Get-Date
        }
    }

    end {
        Write-Verbose "LLM configuration completed"
    }
}

