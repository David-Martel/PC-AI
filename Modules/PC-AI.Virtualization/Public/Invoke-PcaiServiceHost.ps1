#Requires -Version 5.1

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

    # Determine backend
    if ($Backend -eq 'auto') {
        # Check for Rust binary first
        $rustExePaths = @(
            'T:\RustCache\cargo-target\release\pcai-inference.exe',
            "$PSScriptRoot\..\..\..\Deploy\pcai-inference\target\release\pcai-inference.exe",
            "$env:USERPROFILE\PC_AI\Deploy\pcai-inference\target\release\pcai-inference.exe"
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

    # Find Rust executable
    $rootDir = (Join-Path $PSScriptRoot '..\..\..' | Resolve-Path).Path
    $candidateRoots = @(
        'T:\RustCache\cargo-target\release',
        (Join-Path $rootDir 'Native\pcai_core\pcai_inference\target\release'),
        (Join-Path $env:USERPROFILE 'PC_AI\Native\pcai_core\pcai_inference\target\release')
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

    cd Native\pcai_core\pcai_inference
    cargo build --release --features "llamacpp,server"

Or with mistralrs backend:
    cargo build --release --features "mistralrs-backend,server"
"@
    }

    if (-not $ModelPath) {
        throw "ModelPath is required for Rust inference server. Pass -ModelPath."
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

    $configPath = Join-Path $runtimeDir ("config-{0}.json" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $backendConfig = if ($selectedBackend -eq 'mistralrs') {
        @{ type = 'mistral_rs' }
    } else {
        @{ type = 'llama_cpp'; n_gpu_layers = if ($GpuLayers -ge 0) { $GpuLayers } else { $null }; n_ctx = 4096 }
    }

    $config = @{
        backend = $backendConfig
        model   = @{
            path       = $ModelPath
            generation = @{
                max_tokens = 512
                temperature = 0.7
                top_p = 0.95
            }
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

    if ($NoWait) {
        $process = Start-Process -FilePath $rustExe -ArgumentList @('--config', $configPath) -PassThru -WindowStyle Hidden
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
        $output = & $rustExe --config $configPath 2>&1
        $exitCode = $LASTEXITCODE

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
        $HostPath = 'C:\Users\david\PC_AI\Native\PcaiServiceHost\bin\Release\net8.0\PcaiServiceHost.dll'
    }

    if (-not (Test-Path $HostPath)) {
        throw "PcaiServiceHost not found at $HostPath. Build with: dotnet build -c Release"
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
