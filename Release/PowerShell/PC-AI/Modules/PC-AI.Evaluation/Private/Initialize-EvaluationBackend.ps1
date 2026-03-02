function Initialize-EvaluationBackend {
    [CmdletBinding()]
    param(
        [string]$Backend,
        [string]$ModelPath,
        [string]$BaseUrl,
        [int]$GpuLayers = -1
    )

    switch ($Backend) {
        { $_ -in 'llamacpp', 'mistralrs' } {
            # Native FFI backend
            try {
                if (-not (Get-Module -Name PcaiInference)) {
                    $projectRoot = Get-PcaiProjectRoot
                    $localModule = Join-Path $projectRoot 'Modules\PcaiInference.psd1'
                    if (Test-Path $localModule) {
                        Import-Module $localModule -Force -ErrorAction Stop
                    } else {
                        Import-Module PcaiInference -ErrorAction Stop
                    }
                }

                $initResult = Initialize-PcaiInference -Backend $Backend
                if (-not $initResult.Success) {
                    Write-Warning "Failed to initialize $Backend backend"
                    return $false
                }

                if ($ModelPath) {
                    $loadResult = Import-PcaiModel -ModelPath $ModelPath -GpuLayers $GpuLayers
                    if (-not $loadResult.Success) {
                        Write-Warning "Failed to load model: $ModelPath"
                        return $false
                    }
                }

                return $true
            } catch {
                Write-Warning "Backend initialization failed: $_"
                return $false
            }
        }
        { $_ -in 'llamacpp-bin', 'mistralrs-bin' } {
            if (-not $ModelPath) {
                Write-Warning "ModelPath is required for compiled backend: $Backend"
                return $false
            }

            $backendName = if ($Backend -eq 'llamacpp-bin') { 'llamacpp' } else { 'mistralrs' }
            $script:EvaluationConfig.HttpBaseUrl = $BaseUrl
            try {
                $device = $env:PCAI_MISTRAL_DEVICE
                if (-not $device) { $device = $env:PCAI_DEVICE }
                $null = Start-PcaiCompiledServer -Backend $backendName -ModelPath $ModelPath -BaseUrl $BaseUrl -GpuLayers $GpuLayers -Device $device
                return $true
            } catch {
                Write-Warning "Compiled backend initialization failed: $_"
                return $false
            }
        }
        'http' {
            # Test HTTP endpoint
            try {
                $script:EvaluationConfig.HttpBaseUrl = $BaseUrl
                $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
                return $true
            } catch {
                Write-Warning "HTTP backend not responding at $BaseUrl"
                return $false
            }
        }
        'ollama' {
            # Test Ollama endpoint
            try {
                $script:EvaluationConfig.OllamaBaseUrl = $BaseUrl
                $response = Invoke-RestMethod -Uri "$BaseUrl/api/tags" -Method Get -TimeoutSec 5 -ErrorAction Stop
                return $true
            } catch {
                Write-Warning "Ollama not responding at $BaseUrl"
                return $false
            }
        }
    }

    return $false
}
