#Requires -Version 5.1

<#
.SYNOPSIS
    PcaiInference — PowerShell FFI wrapper for the pcai-inference Rust library.

.DESCRIPTION
    Provides synchronous and asynchronous LLM inference from PowerShell by calling
    into the Rust inference engine via PcaiNative.dll ([PcaiNative.InferenceModule]).
    Supports both llama.cpp and mistral.rs backends with optional CUDA GPU offload.

    Exported functions (sync):
      Initialize-PcaiInference  - Load PcaiNative.dll and initialize the selected
                                  inference backend (llamacpp | mistralrs | auto).
                                  Must be called before any generate/model operation.
      Import-PcaiModel          - Load a GGUF or SafeTensors model file into the
                                  active backend with optional GPU layer offloading.
      Invoke-PcaiInference      - Run synchronous text completion for a prompt.
      Invoke-PcaiGenerate       - Alias for Invoke-PcaiInference.
      Stop-PcaiInference        - Shut down the active backend and free resources.
      Close-PcaiInference       - Alias for Stop-PcaiInference.
      Get-PcaiInferenceStatus   - Return current state: DllExists, BackendInitialized,
                                  ModelLoaded, CurrentBackend, DllPath.
                                  Mapped by FunctionGemma router as pcai_native_inference_status.
      Test-PcaiInference        - Probe whether the DLL can initialize a backend
                                  (optionally loading a model); returns $true/$false.
      Test-PcaiDllVersion       - Return file version metadata for pcai_inference.dll.

    Exported functions (async):
      Invoke-PcaiGenerateAsync  - Submit a prompt for async generation. Returns the
                                  completed text by default; use -NoWait to get a
                                  request ID and poll separately.
      Get-PcaiAsyncResult       - Poll or block on a pending async request by ID.
      Stop-PcaiGeneration       - Cancel a pending or running async request.

    Native acceleration (PcaiNative.dll — InferenceModule):
      All inference functions require bin\PcaiNative.dll built from
      Native\PcaiNative\. DLL search order:
        1. Config\llm-config.json nativeInference.dllSearchPaths
        2. bin\pcai_inference.dll (project root)
        3. bin\Release\ / bin\Debug\
        4. .pcai\build\artifacts\pcai-llamacpp\ or pcai-mistralrs\
        5. Native\pcai_core\pcai_inference\target\release\
        6. %USERPROFILE%\.local\bin\
      PcaiNative.dll is loaded from bin\ alongside pcai_inference.dll.

    GPU offload:
      Pass -GpuLayers to Import-PcaiModel (-1 = full GPU, 0 = CPU only).
      GPU assignment: RTX 2000 Ada (8 GB, SM 89) for inference,
      RTX 5060 Ti (16 GB, SM 120) for training (set via Config\llm-config.json).

    Dependencies:
      - PowerShell 5.1 or later
      - bin\PcaiNative.dll (C# P/Invoke wrapper, Native\PcaiNative\)
      - pcai_inference.dll (Rust inference engine, Native\pcai_core\pcai_inference\)
      - CUDA Toolkit 13.x (optional, for GPU offload)
      - Windows 10/11 x64
#>

#region Module Variables
$script:ModulePath = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.ScriptBlock.File }
$script:BackendInitialized = $false
$script:ModelLoaded = $false
$script:CurrentBackend = $null
$script:DllPath = $null
$script:DllExists = $false
#endregion

#region Internal Logic
function Add-EnvPath {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path $Path)) { return }
    if ($env:PATH -notlike "*$Path*") {
        $env:PATH = "$Path;$env:PATH"
    }
}

function Resolve-PcaiInferenceDll {
    param([string]$OverridePath)

    if ($OverridePath) {
        if (Test-Path $OverridePath) {
            return (Resolve-Path $OverridePath).Path
        }
        return $null
    }

    $projectRoot = Split-Path $script:ModulePath -Parent
    $configPath = Join-Path $projectRoot 'Config\llm-config.json'
    $config = $null
    if (Test-Path $configPath) {
        try {
            $config = Get-Content $configPath -Raw | ConvertFrom-Json
        } catch {
            Write-Verbose "Failed to parse ${configPath}: $_"
        }
    }

    $candidates = @()
    if ($config -and $config.nativeInference -and $config.nativeInference.dllSearchPaths) {
        foreach ($path in $config.nativeInference.dllSearchPaths) {
            if (-not $path) { continue }
            if ([System.IO.Path]::IsPathRooted($path)) {
                $candidates += $path
            } else {
                $candidates += (Join-Path $projectRoot $path)
            }
        }
    }

    $candidates += @(
        (Join-Path $projectRoot 'bin\pcai_inference.dll'),
        (Join-Path $projectRoot 'bin\Release\pcai_inference.dll'),
        (Join-Path $projectRoot 'bin\Debug\pcai_inference.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-llamacpp\pcai_inference.dll'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-mistralrs\pcai_inference.dll'),
        (Join-Path $projectRoot 'Native\pcai_core\pcai_inference\target\release\pcai_inference.dll'),
        (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\bin\pcai_inference.dll')
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Initialize-PcaiFFI {
    param([string]$DllPath)

    $resolvedDll = Resolve-PcaiInferenceDll -OverridePath $DllPath
    $script:DllPath = $resolvedDll
    $script:DllExists = $null -ne $resolvedDll -and (Test-Path $resolvedDll)

    if ($script:DllExists) {
        Add-EnvPath (Split-Path $resolvedDll -Parent)
    }

    # Resolve project bin
    $projectRoot = Split-Path $script:ModulePath -Parent
    $projectBin = Join-Path $projectRoot 'bin'

    # Ensure PcaiNative.dll is loaded
    $nativeDll = Join-Path $projectBin 'PcaiNative.dll'
    if (Test-Path $nativeDll) {
        try {
            [void][Reflection.Assembly]::LoadFrom($nativeDll)
            return $true
        } catch {
            Write-Warning "Failed to load $($nativeDll): $($_)"
        }
    }
    return $false
}
#endregion

#region Public Functions

function Initialize-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$Backend = 'llamacpp',

        [Parameter()]
        [string]$DllPath
    )

    $backendName = if ($Backend -eq 'auto') { 'mistralrs' } else { $Backend }
    if ($backendName -eq 'llamacpp') {
        $backendName = 'llama_cpp'
    }

    if (-not (Initialize-PcaiFFI -DllPath $DllPath)) {
        throw 'PcaiNative.dll not found in bin. Please run build.ps1 first.'
    }

    if (-not $script:DllExists) {
        throw "DLL not found: pcai_inference.dll. Update Config/llm-config.json nativeInference.dllSearchPaths or build the native backend."
    }

    Write-Verbose "Initializing backend: $backendName"

    try {
        $result = [PcaiNative.InferenceModule]::pcai_init($backendName)
        if ($result -ne 0) {
            $error = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to initialize backend '$backendName': $error"
        }

        $script:BackendInitialized = $true
        $script:CurrentBackend = $backendName
        Write-Verbose "Backend initialized successfully: $backendName"
        return [PSCustomObject]@{
            Success = $true
            Backend = $backendName
            DllPath = $script:DllPath
        }
    } catch {
        $script:BackendInitialized = $false
        throw "Backend initialization failed: $_"
    }
}

function Import-PcaiModel {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1
    )

    if (-not $script:BackendInitialized) {
        throw 'Backend not initialized. Call Initialize-PcaiInference first.'
    }

    if (-not (Test-Path $ModelPath)) {
        throw "Model file not found: $ModelPath"
    }

    Write-Verbose "Loading model: $ModelPath"

    try {
        $result = [PcaiNative.InferenceModule]::pcai_load_model($ModelPath, $GpuLayers)
        if ($result -ne 0) {
            $error = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to load model: $error"
        }

        $script:ModelLoaded = $true
        Write-Verbose 'Model loaded successfully'
        return [PSCustomObject]@{
            Success   = $true
            ModelPath = $ModelPath
        }
    } catch {
        $script:ModelLoaded = $false
        throw "Model loading failed: $_"
    }
}

function Invoke-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiModel first.'
    }

    try {
        $result = [PcaiNative.InferenceModule]::Generate($Prompt, $MaxTokens, $Temperature)
        if ($null -eq $result) {
            $error = [PcaiNative.InferenceModule]::GetLastError()
            throw "Generation failed: $error"
        }
        return $result
    } catch {
        throw "Inference error: $_"
    }
}

function Invoke-PcaiGenerate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7
    )

    return Invoke-PcaiInference -Prompt $Prompt -MaxTokens $MaxTokens -Temperature $Temperature
}

function Invoke-PcaiGenerateAsync {
    <#
    .SYNOPSIS
        Starts an asynchronous inference request.
    .DESCRIPTION
        Submits a prompt for async generation. By default, polls until complete
        and returns the result. Use -NoWait to get the request ID immediately
        for manual polling with Get-PcaiAsyncResult.
    .PARAMETER Prompt
        The input prompt text.
    .PARAMETER MaxTokens
        Maximum tokens to generate.
    .PARAMETER Temperature
        Sampling temperature (0.0-2.0).
    .PARAMETER NoWait
        Return the request ID immediately without waiting for completion.
    .PARAMETER PollIntervalMs
        Milliseconds between poll attempts when waiting.
    .PARAMETER TimeoutSeconds
        Maximum seconds to wait before cancelling. 0 = no timeout.
    .EXAMPLE
        $result = Invoke-PcaiGenerateAsync -Prompt "Hello world"
    .EXAMPLE
        $id = Invoke-PcaiGenerateAsync -Prompt "Hello" -NoWait
        # ... do other work ...
        $result = Get-PcaiAsyncResult -RequestId $id -Wait
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7,

        [Parameter()]
        [switch]$NoWait,

        [Parameter()]
        [int]$PollIntervalMs = 50,

        [Parameter()]
        [int]$TimeoutSeconds = 0
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiModel first.'
    }

    try {
        $requestId = [PcaiNative.InferenceModule]::pcai_generate_async($Prompt, $MaxTokens, $Temperature)
        if ($requestId -lt 0) {
            $error = [PcaiNative.InferenceModule]::GetLastError()
            throw "Failed to start async generation: $error"
        }

        if ($NoWait) {
            return [PSCustomObject]@{
                RequestId = $requestId
                Status    = 'Pending'
            }
        }

        # Poll until complete
        $deadline = if ($TimeoutSeconds -gt 0) { (Get-Date).AddSeconds($TimeoutSeconds) } else { $null }

        while ($true) {
            $poll = [PcaiNative.InferenceModule]::PollResult($requestId)
            $status = $poll.Item1
            $text = $poll.Item2

            switch ($status) {
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Complete) {
                    return $text
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Failed) {
                    throw "Async generation failed: $text"
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled) {
                    Write-Warning 'Async request was cancelled.'
                    return $null
                }
                ([PcaiNative.InferenceModule+AsyncRequestStatus]::Unknown) {
                    throw "Unknown request ID: $requestId"
                }
                default {
                    if ($deadline -and (Get-Date) -gt $deadline) {
                        [PcaiNative.InferenceModule]::CancelRequest($requestId) | Out-Null
                        throw "Async generation timed out after $TimeoutSeconds seconds."
                    }
                    Start-Sleep -Milliseconds $PollIntervalMs
                }
            }
        }
    } catch {
        throw "Async inference error: $_"
    }
}

function Get-PcaiAsyncResult {
    <#
    .SYNOPSIS
        Gets the result of an async inference request.
    .PARAMETER RequestId
        The request ID returned by Invoke-PcaiGenerateAsync -NoWait.
    .PARAMETER Wait
        Block until the request completes.
    .PARAMETER PollIntervalMs
        Milliseconds between poll attempts when waiting.
    .EXAMPLE
        $result = Get-PcaiAsyncResult -RequestId $id -Wait
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [long]$RequestId,

        [Parameter()]
        [switch]$Wait,

        [Parameter()]
        [int]$PollIntervalMs = 50
    )

    do {
        $poll = [PcaiNative.InferenceModule]::PollResult($RequestId)
        $status = $poll.Item1
        $text = $poll.Item2

        $statusName = switch ($status) {
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Pending)   { 'Pending' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Running)   { 'Running' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Complete)  { 'Complete' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Failed)    { 'Failed' }
            ([PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled) { 'Cancelled' }
            default { 'Unknown' }
        }

        if ($status -in @(
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Complete,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Failed,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Cancelled,
            [PcaiNative.InferenceModule+AsyncRequestStatus]::Unknown
        )) {
            return [PSCustomObject]@{
                RequestId = $RequestId
                Status    = $statusName
                Text      = $text
            }
        }

        if (-not $Wait) {
            return [PSCustomObject]@{
                RequestId = $RequestId
                Status    = $statusName
                Text      = $null
            }
        }

        Start-Sleep -Milliseconds $PollIntervalMs
    } while ($Wait)
}

function Stop-PcaiGeneration {
    <#
    .SYNOPSIS
        Cancels a pending or running async inference request.
    .PARAMETER RequestId
        The request ID to cancel.
    .EXAMPLE
        Stop-PcaiGeneration -RequestId $id
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [long]$RequestId
    )

    $cancelled = [PcaiNative.InferenceModule]::CancelRequest($RequestId)
    return [PSCustomObject]@{
        RequestId = $RequestId
        Cancelled = $cancelled
    }
}

function Stop-PcaiInference {
    [CmdletBinding()]
    param()

    if ($script:BackendInitialized) {
        Write-Verbose 'Shutting down inference backend...'
        try {
            [PcaiNative.InferenceModule]::pcai_shutdown()
            $script:BackendInitialized = $false
            $script:ModelLoaded = $false
            $script:CurrentBackend = $null
        } catch {
            Write-Warning "Error during shutdown: $_"
        }
    }
}

function Close-PcaiInference {
    [CmdletBinding()]
    param()

    Stop-PcaiInference
}

function Get-PcaiInferenceStatus {
    [CmdletBinding()]
    param()

    if (-not $script:DllPath) {
        $script:DllPath = Resolve-PcaiInferenceDll
        $script:DllExists = $null -ne $script:DllPath -and (Test-Path $script:DllPath)
    }

    return [PSCustomObject]@{
        DllPath           = $script:DllPath
        DllExists         = $script:DllExists -and (Test-Path $script:DllPath)
        BackendInitialized = $script:BackendInitialized
        ModelLoaded        = $script:ModelLoaded
        CurrentBackend     = $script:CurrentBackend
    }
}

function Test-PcaiInference {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('auto', 'llamacpp', 'mistralrs')]
        [string]$Backend = 'auto',

        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1
    )

    try {
        $init = Initialize-PcaiInference -Backend $Backend
        if ($ModelPath) {
            $null = Import-PcaiModel -ModelPath $ModelPath -GpuLayers $GpuLayers
        }
        return $true
    } catch {
        Write-Verbose "Test-PcaiInference failed: $_"
        return $false
    } finally {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }
}

function Test-PcaiDllVersion {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$DllPath
    )

    $resolved = Resolve-PcaiInferenceDll -OverridePath $DllPath
    if (-not $resolved) {
        return [PSCustomObject]@{
            Success = $false
            Message = 'pcai_inference.dll not found'
        }
    }

    $info = Get-Item $resolved -ErrorAction SilentlyContinue
    return [PSCustomObject]@{
        Success        = $true
        DllPath        = $resolved
        FileVersion    = $info.VersionInfo.FileVersion
        ProductVersion = $info.VersionInfo.ProductVersion
    }
}

#endregion

#region Module Cleanup
if ($MyInvocation.MyCommand.ScriptBlock.Module) {
    $MyInvocation.MyCommand.ScriptBlock.Module.OnRemove = {
        if ($script:BackendInitialized) {
            [PcaiNative.InferenceModule]::pcai_shutdown()
        }
    }
}
#endregion

#region Module Exports
Export-ModuleMember -Function @(
    'Initialize-PcaiInference',
    'Import-PcaiModel',
    'Invoke-PcaiInference',
    'Invoke-PcaiGenerate',
    'Invoke-PcaiGenerateAsync',
    'Get-PcaiAsyncResult',
    'Stop-PcaiGeneration',
    'Stop-PcaiInference',
    'Close-PcaiInference',
    'Get-PcaiInferenceStatus',
    'Test-PcaiInference',
    'Test-PcaiDllVersion'
)
#endregion
