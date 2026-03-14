#Requires -PSEdition Core

<#
.SYNOPSIS
    Private helper functions for LLM API operations
#>

function Resolve-PcaiEndpoint {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ApiUrl,

        [Parameter()]
        [string]$ProviderName,

        [Parameter()]
        [string]$ConfigPath
    )

    if (-not $ConfigPath) {
        if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
            $ConfigPath = Resolve-PcaiPath -PathType 'HVSockConfig'
        }
    }

    $resolveHvsockName = {
        param([string]$LookupName)
        if (-not $LookupName) { return $null }
        if (-not (Test-Path $ConfigPath)) { return $null }

        $lines = Get-Content $ConfigPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }
        foreach ($line in $lines) {
            $parts = $line.Split(':')
            if ($parts.Count -lt 4) { continue }
            $entryName = $parts[0]
            if ($entryName -ne $LookupName) { continue }

            $tcpHost = $parts[2]
            $tcpPort = $parts[3]
            if ($tcpHost -and $tcpPort) {
                return "http://$tcpHost`:$tcpPort"
            }
        }

        return $null
    }

    if ([string]::IsNullOrWhiteSpace($ApiUrl) -or $ApiUrl -eq 'auto') {
        $autoResolved = & $resolveHvsockName $ProviderName
        if ($autoResolved) { return $autoResolved }

        if ($ProviderName) {
            switch ($ProviderName) {
                'pcai-inference' { if ($script:ModuleConfig.PcaiInferenceApiUrl) { $ApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl } }
                'ollama' { if ($script:ModuleConfig.OllamaApiUrl) { $ApiUrl = $script:ModuleConfig.OllamaApiUrl } }
                'functiongemma' {
                    if ($script:ModuleConfig.RouterApiUrl) { $ApiUrl = $script:ModuleConfig.RouterApiUrl }
                    elseif ($script:ModuleConfig.VLLMApiUrl) { $ApiUrl = $script:ModuleConfig.VLLMApiUrl }
                }
                'vllm' { if ($script:ModuleConfig.VLLMApiUrl) { $ApiUrl = $script:ModuleConfig.VLLMApiUrl } }
                'lmstudio' { if ($script:ModuleConfig.LMStudioApiUrl) { $ApiUrl = $script:ModuleConfig.LMStudioApiUrl } }
            }
        }
    }

    if (-not $ApiUrl) { return $ApiUrl }

    $name = $null
    if ($ApiUrl -match '^(hvsock|vsock)://(?<name>.+)$') {
        $name = $Matches['name']
    }

    if (-not $name) {
        return $ApiUrl
    }

    $resolved = & $resolveHvsockName $name
    if ($resolved) { return $resolved }

    return $ApiUrl
}

function Resolve-OllamaNativeCliPath {
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $candidates = @()
    if ($script:ModuleConfig.OllamaCliSearchPaths) {
        $candidates += @($script:ModuleConfig.OllamaCliSearchPaths)
    }

    $candidates += @(
        (Join-Path $script:ModuleConfig.ProjectRoot 'Native\pcai_core\bin\pcai-ollama-rs.exe'),
        (Join-Path $script:ModuleConfig.ProjectRoot 'Native\pcai_core\target\release\pcai-ollama-rs.exe'),
        (Join-Path $script:ModuleConfig.ProjectRoot 'Native\pcai_core\pcai_ollama_rs\target\release\pcai-ollama-rs.exe')
    )

    $dependencyStamp = if (Get-Command Get-PcaiDependencyStamp -ErrorAction SilentlyContinue) {
        Get-PcaiDependencyStamp -InputObject @($candidates)
    } else {
        $null
    }
    if (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        $cachedCliPath = Get-PcaiSharedCacheEntry -Namespace 'pcai-llm' -Key "ollama-cli::$($script:ModuleConfig.ProjectRoot)" -DependencyStamp $dependencyStamp
        if ($cachedCliPath) {
            return [string]$cachedCliPath
        }
    }

    $resolved = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $resolved) {
        throw "pcai-ollama-rs.exe not found. Build Native\pcai_core\pcai_ollama_rs and ensure Config\llm-config.json ollama.cliSearchPaths points to the binary."
    }

    if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        Set-PcaiSharedCacheEntry -Namespace 'pcai-llm' -Key "ollama-cli::$($script:ModuleConfig.ProjectRoot)" -Value $resolved -DependencyStamp $dependencyStamp -TtlSeconds 300 | Out-Null
    }

    return $resolved
}

function Invoke-OllamaNativeCli {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $cliPath = Resolve-OllamaNativeCliPath
    $cliArgs = @()
    if ($Arguments.Count -gt 0) {
        $cliArgs += $Arguments[0]
        if ($script:ModuleConfig.ConfigPath) {
            $cliArgs += @('--config', $script:ModuleConfig.ConfigPath)
        }
        if ($Arguments.Count -gt 1) {
            $cliArgs += $Arguments[1..($Arguments.Count - 1)]
        }
    } elseif ($script:ModuleConfig.ConfigPath) {
        $cliArgs += @('--config', $script:ModuleConfig.ConfigPath)
    }

    $rawOutput = & $cliPath @cliArgs 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "pcai-ollama-rs failed: $($rawOutput.Trim())"
    }

    if ([string]::IsNullOrWhiteSpace($rawOutput)) {
        return $null
    }

    return $rawOutput | ConvertFrom-Json -Depth 20 -ErrorAction Stop
}

function Invoke-OllamaNativeChat {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [ValidateRange(1024, 1048576)]
        [int]$NumCtx,

        [Parameter()]
        [ValidateRange(1, 256)]
        [int]$NumThread,

        [Parameter()]
        [ValidateRange(0.0, 1.0)]
        [double]$TopP,

        [Parameter()]
        [ValidateRange(1, 1000)]
        [int]$TopK,

        [Parameter()]
        [ValidateRange(-1, 1048576)]
        [int]$RepeatLastN,

        [Parameter()]
        [ValidateRange(0.0, 5.0)]
        [double]$RepeatPenalty,

        [Parameter()]
        [ValidateRange(0.0, 5.0)]
        [double]$TfsZ,

        [Parameter()]
        [int]$Seed,

        [Parameter()]
        [switch]$EnableTools
    )

    $selectedModel = $Model
    if ($EnableTools -and
        $script:ModuleConfig.OllamaToolModel -and
        $selectedModel -eq $script:ModuleConfig.DefaultModel) {
        $selectedModel = $script:ModuleConfig.OllamaToolModel
    }

    $requestFile = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-ollama-request-{0}.json" -f ([System.Guid]::NewGuid().ToString('N')))
    try {
        $requestPayload = [ordered]@{
            model         = $selectedModel
            temperature   = $Temperature
            enableTools   = [bool]$EnableTools
            maxToolRounds = 4
            toolsPath     = $script:ModuleConfig.ToolsPath
            messages      = @()
        }
        if ($PSBoundParameters.ContainsKey('MaxTokens')) {
            $requestPayload.maxTokens = $MaxTokens
        }
        if ($PSBoundParameters.ContainsKey('NumCtx')) {
            $requestPayload.numCtx = $NumCtx
        }
        if ($PSBoundParameters.ContainsKey('NumThread')) {
            $requestPayload.numThread = $NumThread
        }
        if ($PSBoundParameters.ContainsKey('TopP')) {
            $requestPayload.topP = $TopP
        }
        if ($PSBoundParameters.ContainsKey('TopK')) {
            $requestPayload.topK = $TopK
        }
        if ($PSBoundParameters.ContainsKey('RepeatLastN')) {
            $requestPayload.repeatLastN = $RepeatLastN
        }
        if ($PSBoundParameters.ContainsKey('RepeatPenalty')) {
            $requestPayload.repeatPenalty = $RepeatPenalty
        }
        if ($PSBoundParameters.ContainsKey('TfsZ')) {
            $requestPayload.tfsZ = $TfsZ
        }
        if ($PSBoundParameters.ContainsKey('Seed')) {
            $requestPayload.seed = $Seed
        }

        foreach ($message in $Messages) {
            $toolCalls = @()
            if ($message.PSObject.Properties['tool_calls']) {
                $toolCalls = @($message.tool_calls)
            } elseif ($message.PSObject.Properties['toolCalls']) {
                $toolCalls = @($message.toolCalls)
            }

            $requestPayload.messages += [ordered]@{
                role      = [string]$message.role
                content   = [string]$message.content
                toolCalls = @($toolCalls | ForEach-Object {
                    if ($_ -is [string]) {
                        try { $_ | ConvertFrom-Json -Depth 20 } catch { $null }
                    } else {
                        $_
                    }
                } | Where-Object { $null -ne $_ })
            }
        }

        $requestPayload | ConvertTo-Json -Depth 20 | Set-Content -Path $requestFile -Encoding UTF8
        $result = Invoke-OllamaNativeCli -Arguments @('chat', '--request-file', $requestFile)
        if (-not $result.ok) {
            throw ($result.error ?? 'Unknown ollama native error')
        }

        return [PSCustomObject]@{
            message = [PSCustomObject]@{
                content = $result.content
            }
            Provider = 'ollama'
            model    = $result.model
            ToolCalls = @($result.toolCalls)
            ExecutedTools = @($result.executedTools)
            total_duration = if ($result.timing) { $result.timing.totalDurationNs } else { $null }
            raw = $result
        }
    } finally {
        if (Test-Path $requestFile) {
            Remove-Item -Path $requestFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-EnrichedSystemPrompt {
    <#
    .SYNOPSIS
        Loads a system prompt and injects live telemetry and context.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('chat', 'diagnose')]
        [string]$Mode,

        [Parameter()]
        [string]$ProjectRoot
    )

    if (-not $ProjectRoot) {
        if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
            $ProjectRoot = Resolve-PcaiPath -PathType 'Root'
        } elseif ($PSScriptRoot) {
            $ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
        } else {
            $ProjectRoot = (Get-Location).ProviderPath
        }
    }

    $promptPath = if ($Mode -eq 'chat') { Join-Path $ProjectRoot 'CHAT.md' } else { Join-Path $ProjectRoot 'DIAGNOSE.md' }
    if (-not (Test-Path $promptPath)) {
        Write-Warning "System prompt not found at $promptPath"
        return ''
    }

    $prompt = Get-Content -Path $promptPath -Raw -Encoding utf8

    # Inject logic if in diagnose mode
    if ($Mode -eq 'diagnose') {
        $logicPath = Join-Path $ProjectRoot 'DIAGNOSE_LOGIC.md'
        if (Test-Path $logicPath) {
            $logic = Get-Content -Path $logicPath -Raw -Encoding utf8
            $prompt = "$prompt`n`n## REASONING FRAMEWORK`n`n$logic"
        }
    }

    # Inject Telemetry using high-performance Native Dashboard call
    $telemetryBlock = ""
    try {
        if ([PcaiNative.PcaiCore]::IsAvailable) {
            $snapshot = [PcaiNative.PcaiCore]::GetDashboardSnapshotJson()
            if ($snapshot) {
                $summaryLines = $null
                try {
                    $snapshotObj = $snapshot | ConvertFrom-Json -ErrorAction Stop
                    $metrics = $snapshotObj.Metrics
                    if ($metrics) {
                        $cpu = $metrics.cpu_usage_perc
                        if ($null -eq $cpu -and $metrics.PSObject.Properties['CpuUsage']) { $cpu = $metrics.CpuUsage }
                        $memUsed = $metrics.memory_usage_bytes
                        if ($null -eq $memUsed -and $metrics.PSObject.Properties['MemoryUsage']) { $memUsed = $metrics.MemoryUsage }
                        $memTotal = $metrics.total_memory_bytes
                        if ($null -eq $memTotal -and $metrics.PSObject.Properties['TotalMemory']) { $memTotal = $metrics.TotalMemory }

                        $cpuLine = if ($null -ne $cpu) { "CPU Usage: $([math]::Round($cpu, 2))%" } else { "CPU Usage: n/a" }
                        $memLine = if ($null -ne $memUsed -and $null -ne $memTotal -and $memTotal -gt 0) {
                            $usedGb = [math]::Round($memUsed / 1GB, 2)
                            $totalGb = [math]::Round($memTotal / 1GB, 2)
                            "Memory: $usedGb GB / $totalGb GB"
                        } else {
                            "Memory: n/a"
                        }
                        $summaryLines = "$cpuLine`n$memLine"
                    }
                } catch {}

                if (-not $summaryLines) {
                    $summaryLines = "CPU Usage: n/a`nMemory: n/a"
                }

                # Add snapshot directly to prompt, with a short summary for tests/readability
                $telemetryBlock = "### LIVE SYSTEM TELEMETRY (NATIVE)`n$summaryLines`n$snapshot"
            }
        }
    } catch {}

    if (-not $telemetryBlock) {
        # Fallback to manual resource metrics if available
        try {
            if (Get-Command Get-PcaiResourceMetrics -ErrorAction SilentlyContinue) {
                $metrics = Get-PcaiResourceMetrics
                $telemetryBlock = @"
CPU Usage: $($metrics.CpuUsage)%
Memory Usage: $([math]::Round($metrics.MemoryUsage / 1GB, 2)) GB / $([math]::Round($metrics.TotalMemory / 1GB, 2)) GB
GPU Usage: $($metrics.GpuUsage)%
System Uptime: $([TimeSpan]::FromSeconds($metrics.Uptime).ToString("dd'd 'hh'h 'mm'm'"))
"@
            } else {
                # Basic telemetry fallback using built-in commands
                $cpuUsage = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
                $os = Get-CimInstance Win32_OperatingSystem
                $telemetryBlock = "CPU Usage: $cpuUsage%`nMemory: $([math]::Round($os.FreePhysicalMemory / 1MB, 2)) GB free / $([math]::Round($os.TotalVisibleMemorySize / 1MB, 2)) GB total"
            }
        } catch {}
    }

    if ($telemetryBlock) {
        if ($prompt -match '\[SYSTEM_RESOURCE_STATUS\]') {
            $prompt = $prompt -replace '\[SYSTEM_RESOURCE_STATUS\]', $telemetryBlock
        } else {
            $prompt += "`n`n[SYSTEM_RESOURCE_STATUS]`n$telemetryBlock"
        }
    }

    return $prompt
}

function Invoke-ToolByName {
    <#
    .SYNOPSIS
        Executes a PC-AI tool by name using the pcai-tools.json mapping.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter()]
        [object]$Args,

        [Parameter()]
        [array]$Tools = @(),

        [Parameter(Mandatory)]
        [string]$ModuleRoot
    )

    $argTable = @{}
    if ($Args -is [hashtable]) {
        $argTable = $Args
    } elseif ($Args -is [System.Collections.IDictionary]) {
        foreach ($key in $Args.Keys) { $argTable[$key] = $Args[$key] }
    } elseif ($Args -is [pscustomobject]) {
        foreach ($prop in $Args.PSObject.Properties) { $argTable[$prop.Name] = $prop.Value }
    }

    if ($Tools.Count -eq 0) {
        return "Unhandled tool: $Name (no tools provided)"
    }
    $toolDef = $Tools | Where-Object {
        if ($_.PSObject.Properties['function']) {
            return $_.function.name -eq $Name
        }
        return $_['function']['name'] -eq $Name
    }
    if (-not $toolDef -or -not $toolDef.pcai_mapping) {
        return "Unhandled tool: $Name (no mapping found)"
    }

    $mapping = $toolDef.pcai_mapping

    # Dynamic Module Loading
    if ($mapping.module) {
        if (-not (Get-Module -Name $mapping.module -ListAvailable)) {
            $modulePath = Join-Path $ModuleRoot "Modules\$($mapping.module)"
            Import-Module $modulePath -Force -ErrorAction SilentlyContinue
        }
    }

    # Parameter Binding
    if ($mapping.cmdlet -eq 'wsl') {
        $wslArgs = @()
        if ($mapping.args) { $wslArgs += $mapping.args }
        foreach ($key in $argTable.Keys) { $wslArgs += $argTable[$key] }
        & wsl @wslArgs | Out-Null
        return "WSL command executed: wsl $($wslArgs -join ' ')"
    }

    $params = @{}
    if ($mapping.params) {
        foreach ($pName in $mapping.params.psobject.Properties.Name) {
            $pValue = $mapping.params.$pName
            if ($pValue -is [string] -and $pValue -match '^\$') {
                $argKey = $pValue.TrimStart('$')
                if ($argTable.ContainsKey($argKey)) {
                    $params[$pName] = $argTable[$argKey]
                }
            } else {
                $params[$pName] = $pValue
            }
        }
    }

    # Execute Cmdlet
    try {
        $cmdResult = & $mapping.cmdlet @params
        if ($cmdResult -is [PSCustomObject] -or $cmdResult -is [hashtable]) {
            return ($cmdResult | ConvertTo-Json -Depth 6)
        }
        return [string]$cmdResult
    } catch {
        return "Error executing tool $Name ($($mapping.cmdlet)): $($_.Exception.Message)"
    }
}

function Test-PcaiInferenceConnection {
    <#
    .SYNOPSIS
        Tests connectivity to pcai-inference OpenAI-compatible API
    .OUTPUTS
        Boolean indicating connection status
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'pcai-inference'

    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/v1/models" -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        return $true
    }
    catch {
        Write-Verbose "pcai-inference connection test failed: $_"
        return $false
    }
}

function Test-OllamaConnection {
    <#
    .SYNOPSIS
        Tests connectivity to the primary LLM endpoint (pcai-inference)
    .OUTPUTS
        Boolean indicating connection status
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.OllamaApiUrl,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    try {
        $health = Invoke-OllamaNativeCli -Arguments @('health')
        return [bool]$health.ok
    }
    catch {
        Write-Verbose "ollama native health check failed: $_"
        return $false
    }
}

function Test-LMStudioConnection {
    <#
    .SYNOPSIS
        Tests connectivity to LM Studio API
    .OUTPUTS
        Boolean indicating connection status
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.LMStudioApiUrl,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'lmstudio'

    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/v1/models" -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        return $true
    }
    catch {
        Write-Verbose "LM Studio connection test failed: $_"
        return $false
    }
}

function Test-OpenAIConnection {
    <#
    .SYNOPSIS
        Tests connectivity to an OpenAI-compatible API (vLLM/LM Studio)
    .OUTPUTS
        Boolean indicating connection status
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$ApiUrl,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'vllm'

    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/v1/models" -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        return $true
    }
    catch {
        Write-Verbose "OpenAI-compatible connection test failed: $_"
        return $false
    }
}

function Get-VLLMModelInfo {
    <#
    .SYNOPSIS
        Retrieves model metadata from a vLLM OpenAI-compatible endpoint.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.VLLMApiUrl,

        [Parameter()]
        [string]$ModelName = $script:ModuleConfig.VLLMModel,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'vllm'

    try {
        $resp = Invoke-RestMethod -Uri "$ApiUrl/v1/models" -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        $model = $null
        if ($resp.data) {
            $model = $resp.data | Where-Object { $_.id -eq $ModelName } | Select-Object -First 1
            if (-not $model) { $model = $resp.data | Select-Object -First 1 }
        }

        if (-not $model) { return $null }

        return [PSCustomObject]@{
            Id          = $model.id
            MaxModelLen = $model.max_model_len
            Root        = $model.root
            OwnedBy     = $model.owned_by
        }
    }
    catch {
        Write-Verbose "Failed to get vLLM model info: $_"
        return $null
    }
}

function Get-VLLMMetricsSnapshot {
    <#
    .SYNOPSIS
        Fetches and parses vLLM /metrics for lightweight health/throughput data.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.VLLMApiUrl,

        [Parameter()]
        [string]$ModelName = $script:ModuleConfig.VLLMModel,

        [Parameter()]
        [int]$TimeoutSeconds = 5
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'vllm'

    try {
        $metricsText = Invoke-RestMethod -Uri "$ApiUrl/metrics" -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
    }
    catch {
        Write-Verbose "Failed to fetch vLLM metrics: $_"
        return $null
    }

    $values = @{
        PromptTokensTotal     = 0.0
        GenerationTokensTotal = 0.0
        RequestSuccessTotal   = 0.0
        NumRequestsRunning    = 0.0
        NumRequestsWaiting    = 0.0
        KVCacheUsagePerc      = 0.0
    }

    foreach ($line in ($metricsText -split "`n")) {
        if (-not $line -or $line.StartsWith('#')) { continue }
        $m = [regex]::Match($line, '^(?<metric>vllm:[^\\s{]+)(?<labels>{[^}]+})?\\s+(?<value>[-+0-9.eE]+)$')
        if (-not $m.Success) { continue }

        $metric = $m.Groups['metric'].Value
        $value = [double]$m.Groups['value'].Value
        $labels = $m.Groups['labels'].Value
        if ($labels) {
            $labels = $labels.Trim('{', '}')
            $modelMatch = [regex]::Match($labels, 'model_name=\"(?<name>[^\"]+)\"')
            if ($modelMatch.Success -and $ModelName -and $modelMatch.Groups['name'].Value -ne $ModelName) {
                continue
            }
        }

        switch ($metric) {
            'vllm:prompt_tokens_total' { $values.PromptTokensTotal = $value }
            'vllm:generation_tokens_total' { $values.GenerationTokensTotal = $value }
            'vllm:request_success_total' { $values.RequestSuccessTotal += $value }
            'vllm:num_requests_running' { $values.NumRequestsRunning = $value }
            'vllm:num_requests_waiting' { $values.NumRequestsWaiting = $value }
            'vllm:kv_cache_usage_perc' { $values.KVCacheUsagePerc = $value }
        }
    }

    return [PSCustomObject]@{
        CapturedAt            = Get-Date
        ModelName             = $ModelName
        ApiUrl                = $ApiUrl
        PromptTokensTotal     = [double]$values.PromptTokensTotal
        GenerationTokensTotal = [double]$values.GenerationTokensTotal
        RequestSuccessTotal   = [double]$values.RequestSuccessTotal
        NumRequestsRunning    = [double]$values.NumRequestsRunning
        NumRequestsWaiting    = [double]$values.NumRequestsWaiting
        KVCacheUsagePerc      = [double]$values.KVCacheUsagePerc
        TokensTotal           = [double]$values.PromptTokensTotal + [double]$values.GenerationTokensTotal
    }
}

function Invoke-OpenAIChatWithProgress {
    <#
    .SYNOPSIS
        Invokes OpenAI-compatible chat completion with a progress display and vLLM metrics.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter(Mandatory)]
        [string]$Model,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter(Mandatory)]
        [string]$ApiUrl,

        [Parameter()]
        [string]$ApiKey,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$ProgressIntervalSeconds = 1
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'vllm'

    $body = @{
        model = $Model
        messages = $Messages
        temperature = $Temperature
        stream = $false
    }
    if ($MaxTokens) { $body['max_tokens'] = $MaxTokens }

    $jsonBody = $body | ConvertTo-Json -Depth 10
    $headers = @{}
    if ($ApiKey) { $headers['Authorization'] = "Bearer $ApiKey" }

    $uri = "$ApiUrl/v1/chat/completions"
    $job = Start-Job -ScriptBlock {
        param($u, $body, $hdrs, $timeout)
        Invoke-RestMethod -Uri $u -Method Post -Body $body -Headers $hdrs -ContentType 'application/json' -TimeoutSec $timeout
    } -ArgumentList $uri, $jsonBody, $headers, $TimeoutSeconds

    $modelInfo = Get-VLLMModelInfo -ApiUrl $ApiUrl -ModelName $Model
    $start = Get-Date
    while ($job.State -eq 'Running') {
        $elapsed = (Get-Date) - $start
        $metrics = Get-VLLMMetricsSnapshot -ApiUrl $ApiUrl -ModelName $Model -TimeoutSeconds 2
        $kv = if ($metrics) { "{0:P0}" -f $metrics.KVCacheUsagePerc } else { 'n/a' }
        $running = if ($metrics) { $metrics.NumRequestsRunning } else { 'n/a' }
        $waiting = if ($metrics) { $metrics.NumRequestsWaiting } else { 'n/a' }
        $context = if ($modelInfo -and $modelInfo.MaxModelLen) { $modelInfo.MaxModelLen } else { 'n/a' }

        $status = "Elapsed {0}s | KV {1} | Running {2} | Waiting {3} | MaxCtx {4}" -f `
            [int]$elapsed.TotalSeconds, $kv, $running, $waiting, $context
        Write-Progress -Activity "vLLM request" -Status $status
        Start-Sleep -Seconds $ProgressIntervalSeconds
    }

    $response = Receive-Job $job -ErrorAction Stop
    Remove-Job $job
    Write-Progress -Activity "vLLM request" -Completed

    $content = $null
    if ($response.choices -and $response.choices.Count -gt 0) {
        $content = $response.choices[0].message.content
    }

    return [PSCustomObject]@{
        message = [PSCustomObject]@{ content = $content }
        provider = 'openai'
        raw = $response
    }
}

function Get-OllamaModels {
    <#
    .SYNOPSIS
        Retrieves list of available models from pcai-inference (OpenAI-compatible)
    .OUTPUTS
        Array of model objects
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param()

    try {
        $response = Invoke-OllamaNativeCli -Arguments @('models')
        return @($response.models | ForEach-Object {
            [PSCustomObject]@{
                Name = $_
                OwnedBy = 'ollama'
                Root = 'local'
            }
        })
    }
    catch {
        Write-Error "Failed to retrieve native Ollama models: $_"
        return @()
    }
}

function Invoke-OllamaGenerate {
    <#
    .SYNOPSIS
        Invokes pcai-inference completions endpoint (OpenAI-compatible)
    .OUTPUTS
        API response object
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$Prompt,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [string]$System,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$NumCtx,

        [Parameter()]
        [int]$NumThread,

        [Parameter()]
        [double]$TopP,

        [Parameter()]
        [int]$TopK,

        [Parameter()]
        [int]$RepeatLastN,

        [Parameter()]
        [double]$RepeatPenalty,

        [Parameter()]
        [double]$TfsZ,

        [Parameter()]
        [int]$Seed,

        [Parameter()]
        [bool]$Stream = $false,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [switch]$EnableTools
    )

    if ($Stream) {
        Write-Warning 'Native Ollama requests currently return the final response after completion; token streaming is not yet exposed through the PowerShell wrapper.'
    }

    $messages = @()
    if ($System) {
        $messages += @{ role = 'system'; content = $System }
    }
    $messages += @{ role = 'user'; content = $Prompt }

    return Invoke-OllamaNativeChat -Messages $messages -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -NumCtx $NumCtx -NumThread $NumThread -TopP $TopP -TopK $TopK -RepeatLastN $RepeatLastN -RepeatPenalty $RepeatPenalty -TfsZ $TfsZ -Seed $Seed -EnableTools:$EnableTools
}

function Invoke-OllamaChat {
    <#
    .SYNOPSIS
        Invokes pcai-inference chat completions endpoint (OpenAI-compatible)
    .OUTPUTS
        API response object
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [bool]$Stream = $false,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl
    )

    if ($Stream) {
        Write-Warning 'Native Ollama requests currently return the final response after completion; token streaming is not yet exposed through the PowerShell wrapper.'
    }

    return Invoke-OllamaNativeChat -Messages $Messages -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -EnableTools
}

function Invoke-OllamaChatStream {
    <#
    .SYNOPSIS
        Streams pcai-inference chat responses and returns the full content.
    .OUTPUTS
        String (full assistant message)
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [string]$ApiUrl = $script:ModuleConfig.PcaiInferenceApiUrl
    )

    return Invoke-OpenAIChatStream -Messages $Messages -Model $Model -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $ApiUrl
}

function Invoke-OpenAIChat {
    <#
    .SYNOPSIS
        Invokes an OpenAI-compatible chat completion endpoint (vLLM/LM Studio)
    .OUTPUTS
        PSCustomObject normalized to include message.content
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter(Mandatory)]
        [string]$Model,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter(Mandatory)]
        [string]$ApiUrl,

        [Parameter()]
        [string]$ApiKey
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'vllm'

    $body = @{
        model = $Model
        messages = $Messages
        temperature = $Temperature
        stream = $false
    }

    if ($MaxTokens) {
        $body['max_tokens'] = $MaxTokens
    }

    $jsonBody = $body | ConvertTo-Json -Depth 10

    $headers = @{}
    if ($ApiKey) {
        $headers['Authorization'] = "Bearer $ApiKey"
    }

    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/v1/chat/completions" `
            -Method Post `
            -Body $jsonBody `
            -Headers $headers `
            -ContentType 'application/json' `
            -TimeoutSec $TimeoutSeconds `
            -ErrorAction Stop

        $content = $null
        if ($response.choices -and $response.choices.Count -gt 0) {
            $content = $response.choices[0].message.content
        }

        return [PSCustomObject]@{
            message = [PSCustomObject]@{
                content = $content
            }
            provider = 'openai'
            raw = $response
        }
    }
    catch {
        Write-Error "OpenAI-compatible chat request failed: $_"
        throw
    }
}

function Invoke-OpenAIChatStream {
    <#
    .SYNOPSIS
        Streams OpenAI-compatible chat completions (SSE) and returns full content.
    .OUTPUTS
        String (full assistant message)
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter(Mandatory)]
        [string]$Model,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter(Mandatory)]
        [string]$ApiUrl
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'pcai-inference'

    $body = @{
        model = $Model
        messages = $Messages
        temperature = $Temperature
        stream = $true
    }

    if ($MaxTokens) {
        $body['max_tokens'] = $MaxTokens
    }

    $jsonBody = $body | ConvertTo-Json -Depth 10

    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    $content = [System.Net.Http.StringContent]::new($jsonBody, [System.Text.Encoding]::UTF8, 'application/json')
    $content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new('application/json')

    $sb = New-Object System.Text.StringBuilder
    try {
        $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$ApiUrl/v1/chat/completions")
        $request.Content = $content
        $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $response.EnsureSuccessStatusCode() | Out-Null
        $stream = $response.Content.ReadAsStreamAsync().Result
        $reader = New-Object System.IO.StreamReader($stream)
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            if (-not $line.StartsWith('data:')) { continue }

            $payload = $line.Substring(5).Trim()
            if ($payload -eq '[DONE]') { break }

            try {
                $chunk = $payload | ConvertFrom-Json -ErrorAction Stop
                $choice = $chunk.choices | Select-Object -First 1
                $delta = $choice.delta
                $text = $null
                if ($delta -and $delta.content) {
                    $text = $delta.content
                } elseif ($choice.text) {
                    $text = $choice.text
                }

                if ($text) {
                    Write-Host -NoNewline $text
                    $null = $sb.Append($text)
                }
            } catch { }
        }
        Write-Host ""
    } finally {
        $client.Dispose()
    }

    return $sb.ToString()
}

function Invoke-OpenAICompletionStream {
    <#
    .SYNOPSIS
        Streams OpenAI-compatible completions (SSE) and returns full text.
    .OUTPUTS
        String (full completion)
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$Prompt,

        [Parameter(Mandatory)]
        [string]$Model,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter(Mandatory)]
        [string]$ApiUrl
    )

    $ApiUrl = Resolve-PcaiEndpoint -ApiUrl $ApiUrl -ProviderName 'pcai-inference'

    $body = @{
        model = $Model
        prompt = $Prompt
        temperature = $Temperature
        stream = $true
    }

    if ($MaxTokens) {
        $body['max_tokens'] = $MaxTokens
    }

    $jsonBody = $body | ConvertTo-Json -Depth 10

    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    $content = [System.Net.Http.StringContent]::new($jsonBody, [System.Text.Encoding]::UTF8, 'application/json')
    $content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new('application/json')

    $sb = New-Object System.Text.StringBuilder
    try {
        $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$ApiUrl/v1/completions")
        $request.Content = $content
        $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $response.EnsureSuccessStatusCode() | Out-Null
        $stream = $response.Content.ReadAsStreamAsync().Result
        $reader = New-Object System.IO.StreamReader($stream)
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            if (-not $line.StartsWith('data:')) { continue }

            $payload = $line.Substring(5).Trim()
            if ($payload -eq '[DONE]') { break }

            try {
                $chunk = $payload | ConvertFrom-Json -ErrorAction Stop
                $choice = $chunk.choices | Select-Object -First 1
                $text = $choice.text
                if ($text) {
                    Write-Host -NoNewline $text
                    $null = $sb.Append($text)
                }
            } catch { }
        }
        Write-Host ""
    } finally {
        $client.Dispose()
    }

    return $sb.ToString()
}

function Invoke-LLMChatWithFallback {
    <#
    .SYNOPSIS
        Invokes LLM chat with provider fallback order.
    .OUTPUTS
        PSCustomObject normalized to include message.content and provider
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [array]$Messages,

        [Parameter()]
        [string]$Model,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,

        [Parameter()]
        [int]$MaxTokens,

        [Parameter()]
        [int]$TimeoutSeconds = $script:ModuleConfig.DefaultTimeout,

        [Parameter()]
        [ValidateSet('auto','pcai-inference','ollama','vllm','lmstudio')]
        [string]$Provider = 'auto',

        [Parameter()]
        [switch]$ShowProgress,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$ProgressIntervalSeconds = 1
    )

    $providers = if ($Provider -eq 'auto') {
        if ($script:ModuleConfig.ProviderOrder) { @($script:ModuleConfig.ProviderOrder) } else { @('pcai-inference') }
    } else {
        @($Provider)
    }
    $providers = @(
        $providers |
            ForEach-Object {
                switch ($_) {
                    'pcai-native' { 'ollama' }
                    'functiongemma' { 'vllm' }
                    default { $_ }
                }
            } |
            Where-Object { $_ -in @('pcai-inference', 'ollama', 'vllm', 'lmstudio') } |
            Select-Object -Unique
    )
    if (-not $providers -or $providers.Count -eq 0) {
        $providers = @('ollama', 'pcai-inference')
    }

    foreach ($p in $providers) {
        switch ($p) {
            'pcai-inference' {
                if (Get-CachedProviderHealth -Provider 'pcai-inference' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $script:ModuleConfig.PcaiInferenceApiUrl) {
                    $modelToUse = if ($Model) { $Model } else { $script:ModuleConfig.DefaultModel }
                    $resp = Invoke-OpenAIChat -Messages $Messages -Model $modelToUse -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $script:ModuleConfig.PcaiInferenceApiUrl
                    $resp | Add-Member -MemberType NoteProperty -Name Provider -Value 'pcai-inference' -Force
                    return $resp
                }
            }
            'ollama' {
                if (Get-CachedProviderHealth -Provider 'ollama' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $script:ModuleConfig.OllamaApiUrl) {
                    $modelToUse = if ($Model) { $Model } else { $script:ModuleConfig.DefaultModel }
                    $resp = Invoke-OllamaChat -Messages $Messages -Model $modelToUse -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds
                    $resp | Add-Member -MemberType NoteProperty -Name Provider -Value 'ollama' -Force
                    return $resp
                }
            }
            'vllm' {
                if (Get-CachedProviderHealth -Provider 'vllm' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $script:ModuleConfig.VLLMApiUrl) {
                    $modelToUse = if ($Model) { $Model } else { $script:ModuleConfig.VLLMModel }
                    if ($ShowProgress) {
                        $resp = Invoke-OpenAIChatWithProgress -Messages $Messages -Model $modelToUse -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $script:ModuleConfig.VLLMApiUrl -ProgressIntervalSeconds $ProgressIntervalSeconds
                    } else {
                        $resp = Invoke-OpenAIChat -Messages $Messages -Model $modelToUse -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $script:ModuleConfig.VLLMApiUrl
                    }
                    $resp | Add-Member -MemberType NoteProperty -Name Provider -Value 'vllm' -Force
                    return $resp
                }
            }
            'lmstudio' {
                if (Get-CachedProviderHealth -Provider 'lmstudio' -TimeoutSeconds ([math]::Min($TimeoutSeconds, 10)) -ApiUrl $script:ModuleConfig.LMStudioApiUrl) {
                    $modelToUse = if ($Model) { $Model } else { $script:ModuleConfig.DefaultModel }
                    $resp = Invoke-OpenAIChat -Messages $Messages -Model $modelToUse -Temperature $Temperature -MaxTokens $MaxTokens -TimeoutSeconds $TimeoutSeconds -ApiUrl $script:ModuleConfig.LMStudioApiUrl
                    $resp | Add-Member -MemberType NoteProperty -Name Provider -Value 'lmstudio' -Force
                    return $resp
                }
            }
        }
    }

    throw 'No LLM providers are reachable in the configured fallback order.'
}

function Format-TokenCount {
    <#
    .SYNOPSIS
        Formats byte size to human-readable format
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [long]$Bytes
    )

    if ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }
    elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    }
    elseif ($Bytes -ge 1KB) {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    }
    else {
        return "$Bytes bytes"
    }
}

function Get-ServiceStatus {
    <#
    .SYNOPSIS
        Gets the status of a Windows service by name
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$ServiceName
    )

    try {
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

        if ($service) {
            return [PSCustomObject]@{
                Name = $service.Name
                DisplayName = $service.DisplayName
                Status = $service.Status.ToString()
                StartType = $service.StartType.ToString()
                Running = ($service.Status -eq 'Running')
            }
        }
        else {
            return [PSCustomObject]@{
                Name = $ServiceName
                DisplayName = 'Not Found'
                Status = 'NotInstalled'
                StartType = 'Unknown'
                Running = $false
            }
        }
    }
    catch {
        Write-Warning "Failed to get service status for $ServiceName`: $($_.Exception.Message)"
        return [PSCustomObject]@{
            Name = $ServiceName
            DisplayName = 'Error'
            Status = 'Error'
            StartType = 'Unknown'
            Running = $false
        }
    }
}
