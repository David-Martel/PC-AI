#Requires -Version 5.1

<#
.SYNOPSIS
    Provider health check caching to reduce latency on repeated LLM calls.
#>

# Module-scoped cache
$script:ProviderHealthCache = @{
    Results = @{}
    CacheTTLSeconds = 30
}

function Reset-ProviderHealthCache {
    [CmdletBinding()]
    param()
    $script:ProviderHealthCache.Results = @{}
    Write-Verbose "Provider health cache cleared"
}

function Set-ProviderHealthCache {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('pcai-inference', 'ollama', 'vllm', 'lmstudio', 'functiongemma')]
        [string]$Provider,

        [Parameter(Mandatory)]
        [bool]$IsHealthy,

        [Parameter()]
        [string]$Message
    )

    $script:ProviderHealthCache.Results[$Provider] = @{
        IsHealthy = $IsHealthy
        Message = $Message
        CachedAt = Get-Date
    }
    Write-Verbose "Cached health for ${Provider}: $IsHealthy"
}

function Get-ProviderHealthCache {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$Provider
    )

    if (-not $script:ProviderHealthCache.Results.ContainsKey($Provider)) {
        return $null
    }

    $cached = $script:ProviderHealthCache.Results[$Provider]
    $age = (Get-Date) - $cached.CachedAt

    if ($age.TotalSeconds -gt $script:ProviderHealthCache.CacheTTLSeconds) {
        Write-Verbose "Cache expired for ${Provider} (age: $($age.TotalSeconds)s)"
        $script:ProviderHealthCache.Results.Remove($Provider)
        return $null
    }

    Write-Verbose "Cache hit for ${Provider} (age: $($age.TotalSeconds)s)"
    return [PSCustomObject]@{
        Provider = $Provider
        IsHealthy = $cached.IsHealthy
        Message = $cached.Message
        CachedAt = $cached.CachedAt
        AgeSeconds = [int]$age.TotalSeconds
    }
}

function Get-CachedProviderHealth {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('pcai-inference', 'ollama', 'vllm', 'lmstudio', 'functiongemma')]
        [string]$Provider,

        [Parameter()]
        [int]$TimeoutSeconds = 5,

        [Parameter()]
        [string]$ApiUrl
    )

    # Check cache first
    $cached = Get-ProviderHealthCache -Provider $Provider
    if ($null -ne $cached) {
        Write-Verbose "Using cached health for $Provider"
        return $cached.IsHealthy
    }

    # Perform actual health check
    $isHealthy = $false
    $message = ''

    try {
        switch ($Provider) {
            'pcai-inference' {
                $targetUrl = if ($ApiUrl) { $ApiUrl } else { $script:ModuleConfig.PcaiInferenceApiUrl }
                $isHealthy = Test-PcaiInferenceConnection -ApiUrl $targetUrl -TimeoutSeconds $TimeoutSeconds
                $message = if ($isHealthy) { 'Connected' } else { 'Connection failed' }
            }
            'ollama' {
                $targetUrl = if ($ApiUrl) { $ApiUrl } else { $script:ModuleConfig.PcaiInferenceApiUrl }
                $isHealthy = Test-PcaiInferenceConnection -ApiUrl $targetUrl -TimeoutSeconds $TimeoutSeconds
                $message = if ($isHealthy) { 'Connected' } else { 'Connection failed' }
            }
            'vllm' {
                $targetUrl = if ($ApiUrl) { $ApiUrl } else { $script:ModuleConfig.VLLMApiUrl }
                $isHealthy = Test-OpenAIConnection -ApiUrl $targetUrl -TimeoutSeconds $TimeoutSeconds
                $message = if ($isHealthy) { 'Connected' } else { 'Connection failed' }
            }
            'lmstudio' {
                $targetUrl = if ($ApiUrl) { $ApiUrl } else { $script:ModuleConfig.LMStudioApiUrl }
                $isHealthy = Test-OpenAIConnection -ApiUrl $targetUrl -TimeoutSeconds $TimeoutSeconds
                $message = if ($isHealthy) { 'Connected' } else { 'Connection failed' }
            }
            'functiongemma' {
                $targetUrl = if ($ApiUrl) { $ApiUrl } else { $script:ModuleConfig.RouterApiUrl }
                if (-not $targetUrl) { $targetUrl = $script:ModuleConfig.VLLMApiUrl }
                $isHealthy = Test-OpenAIConnection -ApiUrl $targetUrl -TimeoutSeconds $TimeoutSeconds
                $message = if ($isHealthy) { 'Connected' } else { 'Connection failed' }
            }
        }
    } catch {
        $message = $_.Exception.Message
    }

    # Cache result
    Set-ProviderHealthCache -Provider $Provider -IsHealthy $isHealthy -Message $message

    return $isHealthy
}

function Set-ProviderHealthCacheTTL {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateRange(5, 300)]
        [int]$Seconds
    )

    $script:ProviderHealthCache.CacheTTLSeconds = $Seconds
    Write-Verbose "Cache TTL set to $Seconds seconds"
}
