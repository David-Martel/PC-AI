#Requires -PSEdition Core

<#
.SYNOPSIS
    Starts the PC_AI inference service (Rust or legacy C# backend).

.DESCRIPTION
    Launches the pcai-inference Rust server or legacy C# service host.
    The Rust backend is preferred and provides dual-backend support
    (llamacpp and mistralrs) with GPU acceleration.

.PARAMETER Backend
    Backend to use: 'rust' (default), 'csharp' (legacy)

.PARAMETER NativeBackend
    Native Rust backend for HTTP server: 'llamacpp', 'mistralrs', or 'auto'

.PARAMETER Port
    HTTP server port. Default: 8080

.PARAMETER ModelPath
    Path to GGUF model file for native loading

.PARAMETER GpuLayers
    Number of layers to offload to GPU (-1 = auto)

.PARAMETER ServerArgs
    Additional arguments to pass to the server

.PARAMETER NoWait
    Start server in background without waiting

.EXAMPLE
    Invoke-PcaiServiceHost
    Starts the Rust inference server on default port

.EXAMPLE
    Invoke-PcaiServiceHost -ModelPath "C:\models\mistral-7b.gguf" -GpuLayers 35
    Starts server with specific model and GPU offloading

.EXAMPLE
    Invoke-PcaiServiceHost -Backend csharp -ServerArgs @('--mode', 'diagnostic')
    Starts legacy C# service host
#>
function Invoke-PcaiServiceHost {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('rust', 'csharp', 'auto')]
        [string]$Backend = 'auto',

        [Parameter()]
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$NativeBackend = 'auto',

        [Parameter()]
        [int]$Port = 8080,

        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1,

        [Parameter()]
        [string[]]$ServerArgs,

        [Parameter()]
        [switch]$NoWait,

        # Legacy parameter for backward compatibility
        [Parameter()]
        [string]$HostPath
    )

    $repoRoot = $null
    if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
        try {
            $repoRoot = Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
        } catch {}
    }
    if (-not $repoRoot) {
        try {
            $repoRoot = (Join-Path $PSScriptRoot '..\..\..' | Resolve-Path -ErrorAction Stop).Path
        } catch {}
    }

    if (-not $PSBoundParameters.ContainsKey('Port')) {
        try {
            if (Get-Command Get-PcaiRuntimeConfig -ErrorAction SilentlyContinue) {
                $runtimeConfig = Get-PcaiRuntimeConfig -ProjectRoot $repoRoot
                if ($runtimeConfig -and $runtimeConfig.PcaiInferenceUrl) {
                    $uri = [System.Uri]$runtimeConfig.PcaiInferenceUrl
                    if ($uri.Port -gt 0) {
                        $Port = $uri.Port
                    }
                }
            }
        } catch {
            Write-Verbose "Failed to resolve service port from runtime config: $($_.Exception.Message)"
        }
    }

    # Determine backend
    if ($Backend -eq 'auto') {
        # Check for Rust binary first
        $rustExePaths = @(
            'T:\RustCache\cargo-target\release\pcai-llamacpp.exe',
            'T:\RustCache\cargo-target\release\pcai-mistralrs.exe',
            $(if ($repoRoot) { Join-Path $repoRoot 'Native\pcai_core\pcai_inference\target\release\pcai-llamacpp.exe' }),
            $(if ($repoRoot) { Join-Path $repoRoot 'Native\pcai_core\pcai_inference\target\release\pcai-mistralrs.exe' }),
            $(if ($repoRoot) { Join-Path $repoRoot '.pcai\build\artifacts\pcai-llamacpp\pcai-llamacpp.exe' }),
            $(if ($repoRoot) { Join-Path $repoRoot '.pcai\build\artifacts\pcai-mistralrs\pcai-mistralrs.exe' })
        )

        $rustExe = $rustExePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($rustExe) {
            $Backend = 'rust'
        } else {
            $Backend = 'csharp'
        }
    }

    if ($Backend -eq 'rust') {
        return Start-RustInferenceServer -Port $Port -ModelPath $ModelPath -GpuLayers $GpuLayers -NativeBackend $NativeBackend -ServerArgs $ServerArgs -NoWait:$NoWait
    } else {
        return Start-CSharpServiceHost -HostPath $HostPath -ServerArgs $ServerArgs -NoWait:$NoWait
    }
}

function Start-RustInferenceServer {
    [CmdletBinding()]
    param(
        [int]$Port = 8080,
        [string]$ModelPath,
        [int]$GpuLayers = -1,
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$NativeBackend = 'auto',
        [string[]]$ServerArgs,
        [switch]$NoWait
    )

    function Resolve-ConfiguredModelPath {
        param([string]$ProjectRoot)

        if (-not $ProjectRoot) {
            return $null
        }

        $configPath = Join-Path $ProjectRoot 'Config\llm-config.json'
        if (-not (Test-Path $configPath)) {
            return $null
        }

        try {
            $config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
            $configuredPath = $config.providers.'pcai-native'.modelPath
            if (-not $configuredPath) {
                return $null
            }
            if ([System.IO.Path]::IsPathRooted([string]$configuredPath)) {
                return [string]$configuredPath
            }
            return (Join-Path $ProjectRoot ([string]$configuredPath))
        } catch {
            Write-Verbose "Failed to resolve configured model path from ${configPath}: $($_.Exception.Message)"
            return $null
        }
    }

    function Resolve-ProjectLlmConfig {
        param([string]$ProjectRoot)

        if (-not $ProjectRoot) {
            return $null
        }

        $configPath = Join-Path $ProjectRoot 'Config\llm-config.json'
        if (-not (Test-Path $configPath)) {
            return $null
        }

        try {
            return Get-Content -Raw -Path $configPath | ConvertFrom-Json
        } catch {
            Write-Verbose "Failed to parse ${configPath}: $($_.Exception.Message)"
            return $null
        }
    }

    # Find Rust executable
    $rootDir = $null
    if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
        try {
            $rootDir = Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
        } catch {}
    }
    if (-not $rootDir) {
        $rootDir = (Join-Path $PSScriptRoot '..\..\..' | Resolve-Path).Path
    }
    $candidateRoots = @(
        'T:\RustCache\cargo-target\release',
        (Join-Path $rootDir 'Native\pcai_core\pcai_inference\target\release'),
        (Join-Path $rootDir '.pcai\build\artifacts\pcai-llamacpp'),
        (Join-Path $rootDir '.pcai\build\artifacts\pcai-mistralrs')
    )

    $binaryNames = switch ($NativeBackend) {
        'llamacpp' { @('pcai-llamacpp.exe') }
        'mistralrs' { @('pcai-mistralrs.exe') }
        default { @('pcai-llamacpp.exe', 'pcai-mistralrs.exe') }
    }

    $rustExe = $null
    foreach ($root in $candidateRoots) {
        foreach ($name in $binaryNames) {
            $candidate = Join-Path $root $name
            if (Test-Path $candidate) {
                $rustExe = $candidate
                break
            }
        }
        if ($rustExe) { break }
    }

    if (-not $rustExe) {
        throw @"
pcai-llamacpp.exe or pcai-mistralrs.exe not found. Build with:

    .\Build.ps1 -Component inference

Or backend-specific:
    .\Build.ps1 -Component mistralrs
"@
    }

    if (-not $ModelPath) {
        $ModelPath = Resolve-ConfiguredModelPath -ProjectRoot $rootDir
    }

    if (-not $ModelPath) {
        throw "ModelPath is required for Rust inference server. Pass -ModelPath or set providers.pcai-native.modelPath in Config\\llm-config.json."
    }

    if (-not (Test-Path $ModelPath)) {
        throw "Configured model file not found: $ModelPath"
    }

    $selectedBackend = if ($NativeBackend -eq 'auto') {
        if ($rustExe -like '*pcai-mistralrs*') { 'mistralrs' } else { 'llamacpp' }
    } else {
        $NativeBackend
    }

    $runtimeDir = Join-Path $rootDir '.pcai\runtime\pcai-inference'
    if (-not (Test-Path $runtimeDir)) {
        New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    }

    $projectLlmConfig = Resolve-ProjectLlmConfig -ProjectRoot $rootDir
    $configuredGpuLayers = $null
    $configuredCtx = 4096
    $configuredMaxTokens = 512
    $configuredTemperature = 0.7
    $configuredTopP = 0.95
    $routerBaseUrl = $null
    $routerModel = 'functiongemma-270m-it'
    $routerToolsPath = Join-Path $rootDir 'Config\pcai-tools.json'
    $routerStrict = $false
    $routerForce = $false
    $routerDisable = $false
    $routerEnabled = $true
    $routerTemperature = 0.2

    if ($projectLlmConfig) {
        $nativeProvider = $projectLlmConfig.providers.'pcai-native'
        if ($nativeProvider) {
            if ($nativeProvider.PSObject.Properties.Name -contains 'gpuLayers') {
                $configuredGpuLayers = $nativeProvider.gpuLayers
            }
            if ($nativeProvider.PSObject.Properties.Name -contains 'defaultMaxTokens') {
                $configuredMaxTokens = $nativeProvider.defaultMaxTokens
            }
            if ($nativeProvider.PSObject.Properties.Name -contains 'defaultTemperature') {
                $configuredTemperature = $nativeProvider.defaultTemperature
            }
        }

        $llamaBackend = $projectLlmConfig.nativeInference.backends.llamacpp
        if ($llamaBackend -and $llamaBackend.settings) {
            if ($llamaBackend.settings.PSObject.Properties.Name -contains 'n_ctx' -and $llamaBackend.settings.n_ctx) {
                $configuredCtx = $llamaBackend.settings.n_ctx
            }
            if (($GpuLayers -lt 0) -and ($llamaBackend.settings.PSObject.Properties.Name -contains 'n_gpu_layers')) {
                $configuredGpuLayers = $llamaBackend.settings.n_gpu_layers
            }
            $samplers = @($llamaBackend.settings.samplers)
            foreach ($sampler in $samplers) {
                if ($sampler -match '^temperature:(?<value>[-+]?[0-9]*\.?[0-9]+)$') {
                    $configuredTemperature = [double]$Matches.value
                } elseif ($sampler -match '^topp:p=(?<value>[-+]?[0-9]*\.?[0-9]+)$') {
                    $configuredTopP = [double]$Matches.value
                }
            }
        }

        $routerConfig = $projectLlmConfig.router
        if ($routerConfig) {
            if ($routerConfig.PSObject.Properties.Name -contains 'baseUrl') {
                $routerBaseUrl = [string]$routerConfig.baseUrl
            }
            if ($routerConfig.PSObject.Properties.Name -contains 'model' -and $routerConfig.model) {
                $routerModel = [string]$routerConfig.model
            }
            if ($routerConfig.PSObject.Properties.Name -contains 'toolsPath' -and $routerConfig.toolsPath) {
                $routerToolsPath = if ([System.IO.Path]::IsPathRooted([string]$routerConfig.toolsPath)) {
                    [string]$routerConfig.toolsPath
                } else {
                    Join-Path $rootDir ([string]$routerConfig.toolsPath)
                }
            }
            if ($routerConfig.PSObject.Properties.Name -contains 'strict') { $routerStrict = [bool]$routerConfig.strict }
            if ($routerConfig.PSObject.Properties.Name -contains 'force') { $routerForce = [bool]$routerConfig.force }
            if ($routerConfig.PSObject.Properties.Name -contains 'disable') { $routerDisable = [bool]$routerConfig.disable }
            if ($routerConfig.PSObject.Properties.Name -contains 'enabled') { $routerEnabled = [bool]$routerConfig.enabled }
            if ($routerConfig.PSObject.Properties.Name -contains 'defaultTemperature' -and $routerConfig.defaultTemperature -ne $null) {
                $routerTemperature = [double]$routerConfig.defaultTemperature
            }
        }
    }

    if ($GpuLayers -lt 0 -and $configuredGpuLayers -ne $null) {
        $GpuLayers = [int]$configuredGpuLayers
    }

    $configPath = Join-Path $runtimeDir ("config-{0}.json" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $backendConfig = if ($selectedBackend -eq 'mistralrs') {
        @{ type = 'mistral_rs' }
    } else {
        @{ type = 'llama_cpp'; n_gpu_layers = if ($GpuLayers -ge 0) { $GpuLayers } else { $null }; n_ctx = $configuredCtx }
    }

    $config = @{
        backend = $backendConfig
        model   = @{
            path       = $ModelPath
            generation = @{
                max_tokens = $configuredMaxTokens
                temperature = $configuredTemperature
                top_p = $configuredTopP
            }
        }
        router  = @{
            enabled = $routerEnabled
            provider = 'functiongemma'
            base_url = if ($routerBaseUrl) { $routerBaseUrl } else { '' }
            model = $routerModel
            tools_path = $routerToolsPath
            strict = $routerStrict
            force = $routerForce
            disable = $routerDisable
            default_temperature = $routerTemperature
        }
        server  = @{
            host = '127.0.0.1'
            port = $Port
            cors = $true
        }
    }

    $config | ConvertTo-Json -Depth 6 | Set-Content -Path $configPath -Encoding UTF8

    Write-Verbose "Using config file: $configPath"
    if ($ServerArgs) {
        Write-Verbose "Ignoring ServerArgs for Rust inference server (config-driven)."
    }

    $workingDirectory = Split-Path -Path $rustExe -Parent

    if ($NoWait) {
        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $rustExe
        $startInfo.WorkingDirectory = $workingDirectory
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.Arguments = ('--config "{0}"' -f $configPath)

        $process = [System.Diagnostics.Process]::Start($startInfo)
        if (-not $process) {
            throw "Failed to start Rust inference server process: $rustExe"
        }
        return [PSCustomObject]@{
            Backend   = 'rust'
            ExePath   = $rustExe
            Args      = @('--config', $configPath)
            ProcessId = $process.Id
            Port      = $Port
            Success   = $true
            Message   = "Server started in background (PID: $($process.Id))"
        }
    } else {
        # Run and capture output
        Push-Location $workingDirectory
        try {
            $output = & $rustExe --config $configPath 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }

        return [PSCustomObject]@{
            Backend  = 'rust'
            ExePath  = $rustExe
            Args     = @('--config', $configPath)
            ExitCode = $exitCode
            Output   = ($output | Out-String).Trim()
            Success  = ($exitCode -eq 0)
        }
    }
}

function Start-CSharpServiceHost {
    [CmdletBinding()]
    param(
        [string]$HostPath,
        [string[]]$ServerArgs,
        [switch]$NoWait
    )

    if (-not $HostPath) {
        $repoRoot = $null
        if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
            try {
                $repoRoot = Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
            } catch {}
        }
        if (-not $repoRoot) {
            $repoRoot = (Join-Path $PSScriptRoot '..\..\..' | Resolve-Path).Path
        }
        $hostPathCandidates = @(
            (Join-Path $repoRoot 'Native\PcaiServiceHost\bin\Release\net8.0\PcaiServiceHost.dll'),
            (Join-Path $repoRoot '.pcai\build\artifacts\pcai-servicehost\PcaiServiceHost.dll')
        )
        $hostPathCandidates = $hostPathCandidates | Where-Object { $_ -and (Test-Path $_) }
        $HostPath = $hostPathCandidates | Select-Object -First 1
    }

    if (-not $HostPath) {
        throw "PcaiServiceHost DLL not found. Build with: .\\Build.ps1 -Component servicehost"
    }

    if (-not (Test-Path $HostPath)) {
        throw "PcaiServiceHost not found at $HostPath. Build with: .\\Build.ps1 -Component servicehost"
    }

    $dotnet = (Get-Command dotnet -ErrorAction SilentlyContinue)?.Source
    if (-not $dotnet) {
        throw 'dotnet not found in PATH.'
    }

    Write-Verbose "Starting: dotnet $HostPath $($ServerArgs -join ' ')"

    if ($NoWait) {
        $process = Start-Process -FilePath $dotnet -ArgumentList @($HostPath) + $ServerArgs -PassThru -WindowStyle Hidden
        return [PSCustomObject]@{
            Backend   = 'csharp'
            HostPath  = $HostPath
            Args      = $ServerArgs
            ProcessId = $process.Id
            Success   = $true
            Message   = "Service host started in background (PID: $($process.Id))"
        }
    } else {
        $output = & $dotnet $HostPath @ServerArgs 2>&1
        $exitCode = $LASTEXITCODE

        return [PSCustomObject]@{
            Backend  = 'csharp'
            HostPath = $HostPath
            Args     = $ServerArgs
            ExitCode = $exitCode
            Output   = ($output | Out-String).Trim()
            Success  = ($exitCode -eq 0)
        }
    }
}
