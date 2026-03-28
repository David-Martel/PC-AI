#Requires -Version 5.1

<#
.SYNOPSIS
    PcaiMedia — PowerShell FFI wrapper for the pcai-media Rust library.

.DESCRIPTION
    Provides image generation and image understanding (vision-language) from
    PowerShell by calling into the Rust media pipeline via PcaiNative.dll
    ([PcaiNative.MediaModule]).  Supports Janus-Pro models with optional
    CUDA GPU offload for text-to-image and image-to-text workflows.

    Exported functions:
      Initialize-PcaiMedia    - Load PcaiNative.dll and initialise the media
                                pipeline on the specified compute device.
                                Must be called before any model/generate operation.
      Import-PcaiMediaModel   - Load a Janus-Pro model (HF repo ID or local path)
                                with optional GPU layer offloading.
      New-PcaiImage           - Generate an image from a text prompt and save
                                as PNG.  Auto-generates output path when not
                                specified.
      Get-PcaiImageAnalysis   - Run image-to-text understanding: describe or
                                answer questions about an input image.
      Stop-PcaiMedia          - Shut down the media pipeline and free resources.
      Get-PcaiMediaStatus     - Return current state: Initialized, ModelLoaded,
                                CurrentModel.

    Native acceleration (PcaiNative.dll — MediaModule):
      All media functions require bin\PcaiNative.dll built from
      Native\PcaiNative\.  The DLL wraps pcai_media.dll (Rust) which provides
      the Janus-Pro inference pipeline with C ABI exports.

    GPU offload:
      Pass -Device cuda:auto, cuda:0, or cuda:1 to Initialize-PcaiMedia.
      Pass -GpuLayers to Import-PcaiMediaModel (-1 = full GPU, 0 = CPU only).

    Dependencies:
      - PowerShell 5.1 or later
      - bin\PcaiNative.dll (C# P/Invoke wrapper, Native\PcaiNative\)
      - pcai_media.dll (Rust media engine, Native\pcai_core\pcai_media\)
      - CUDA Toolkit 13.x (optional, for GPU offload)
      - Windows 10/11 x64
#>

#region Module Variables
$script:ModulePath = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.ScriptBlock.File }
$script:Initialized = $false
$script:ModelLoaded = $false
$script:CurrentModel = $null
$script:NextAsyncRequestId = 0
$script:AsyncImageRequests = [hashtable]::Synchronized(@{})
#endregion

#region Internal Logic
function Get-PcaiProjectRoot {
    return (Split-Path $script:ModulePath -Parent)
}

function New-PcaiAsyncRequestId {
    $script:NextAsyncRequestId += 1
    return $script:NextAsyncRequestId
}

function Get-PcaiImageAsyncRequest {
    param(
        [Parameter(Mandatory)]
        [int]$RequestId
    )

    if (-not $script:AsyncImageRequests.ContainsKey($RequestId)) {
        throw "Async image request not found: $RequestId"
    }

    return $script:AsyncImageRequests[$RequestId]
}

function Remove-PcaiImageAsyncRequest {
    param(
        [Parameter(Mandatory)]
        [int]$RequestId
    )

    if ($script:AsyncImageRequests.ContainsKey($RequestId)) {
        $entry = $script:AsyncImageRequests[$RequestId]
        try {
            if ($entry.CancellationTokenSource) {
                $entry.CancellationTokenSource.Dispose()
            }
        } catch {
            Write-Verbose "Failed to dispose async request token source ${RequestId}: $_"
        }
        $null = $script:AsyncImageRequests.Remove($RequestId)
    }
}

function Clear-PcaiImageAsyncRequests {
    foreach ($requestId in @($script:AsyncImageRequests.Keys)) {
        try {
            Remove-PcaiImageAsyncRequest -RequestId ([int]$requestId)
        } catch {
            Write-Verbose "Failed to clear async request ${requestId}: $_"
        }
    }
}

function Get-PcaiImageAsyncCompletionState {
    param(
        [Parameter(Mandatory)]
        $Task
    )

    $taskStatus = $Task.Status.ToString()

    if ($Task.IsCanceled) {
        return [PSCustomObject]@{
            TaskStatus = $taskStatus
            Status     = 'Cancelled'
            Result     = $null
            Error      = 'Cancelled'
        }
    }

    if ($Task.IsFaulted) {
        $message = $null
        if ($null -ne $Task.Exception) {
            $message = $Task.Exception.GetBaseException().Message
        }
        return [PSCustomObject]@{
            TaskStatus = $taskStatus
            Status     = 'Failed'
            Result     = $null
            Error      = $message
        }
    }

    if (-not $Task.IsCompleted) {
        return [PSCustomObject]@{
            TaskStatus = $taskStatus
            Status     = 'Running'
            Result     = $null
            Error      = $null
        }
    }

    $result = $Task.GetAwaiter().GetResult()
    if ($null -ne $result) {
        return [PSCustomObject]@{
            TaskStatus = $taskStatus
            Status     = 'Failed'
            Result     = $result
            Error      = $result
        }
    }

    return [PSCustomObject]@{
        TaskStatus = $taskStatus
        Status     = 'Completed'
        Result     = $null
        Error      = $null
    }
}

function Initialize-PcaiMediaFFI {
    <#
    .SYNOPSIS
        Loads PcaiNative.dll from the project bin directory.
    #>
    $projectRoot = Get-PcaiProjectRoot
    $projectBin = Join-Path $projectRoot 'bin'

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

function Initialize-PcaiMedia {
    <#
    .SYNOPSIS
        Initialise the pcai-media pipeline on the specified compute device.
    .DESCRIPTION
        Loads PcaiNative.dll and calls [PcaiNative.MediaModule]::pcai_media_init()
        to prepare the media pipeline.  Must be called before Import-PcaiMediaModel
        or any generation/understanding operation.
    .PARAMETER Device
        Target compute device.  Accepted values: cpu, cuda, cuda:auto, cuda:0, cuda:1.
        Defaults to cuda:auto.
    .EXAMPLE
        Initialize-PcaiMedia -Device cuda:auto
    .EXAMPLE
        Initialize-PcaiMedia -Device cpu
    #>
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [ValidateSet('cpu', 'cuda', 'cuda:auto', 'cuda:0', 'cuda:1')]
        [string]$Device = 'cuda:auto'
    )

    if (-not (Initialize-PcaiMediaFFI)) {
        throw 'PcaiNative.dll not found in bin. Please run Build.ps1 first.'
    }

    Write-Verbose "Initializing pcai-media on device: $Device"

    try {
        $result = [PcaiNative.MediaModule]::pcai_media_init($Device)
        if ($result -ne 0) {
            $errorMsg = [PcaiNative.MediaModule]::GetLastError()
            throw "Failed to initialize media pipeline on device '$Device': $errorMsg"
        }

        $script:Initialized = $true
        Write-Verbose "Media pipeline initialized successfully on device: $Device"
        return [PSCustomObject]@{
            Success = $true
            Device  = $Device
        }
    } catch {
        $script:Initialized = $false
        throw "Media pipeline initialization failed: $_"
    }
}

function Import-PcaiMediaModel {
    <#
    .SYNOPSIS
        Load a Janus-Pro model into the media pipeline.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::pcai_media_load_model() to load model
        weights.  The model path can be a HuggingFace Hub repo ID (e.g.
        "deepseek-ai/Janus-Pro-7B") or a local directory containing config.json,
        *.safetensors, and tokenizer.json.
    .PARAMETER ModelPath
        HuggingFace repo ID or absolute path to a local model directory.
    .PARAMETER GpuLayers
        Number of model layers to offload to GPU.  -1 = full GPU offload,
        0 = CPU only.  Defaults to -1.
    .EXAMPLE
        Import-PcaiMediaModel -ModelPath "deepseek-ai/Janus-Pro-7B"
    .EXAMPLE
        Import-PcaiMediaModel -ModelPath "C:\Models\Janus-Pro-7B" -GpuLayers 0
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$ModelPath,

        [Parameter()]
        [int]$GpuLayers = -1
    )

    if (-not $script:Initialized) {
        throw 'Media pipeline not initialized. Call Initialize-PcaiMedia first.'
    }

    # Validate ModelPath before passing to the native DLL.
    # The Rust DLL treats any value without a '/' separator as a HuggingFace repo ID and
    # prepends 'https://huggingface.co/', which breaks absolute local paths.
    if ([System.IO.Path]::IsPathRooted($ModelPath)) {
        # Caller supplied an absolute path — confirm it exists locally.
        if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
            throw "Model path '$ModelPath' is an absolute path but the directory does not exist on disk."
        }
        Write-Verbose "ModelPath is a local directory: $ModelPath"
    } elseif ($ModelPath -notmatch '/') {
        # Looks like neither a HuggingFace repo ID (requires 'owner/repo' format) nor a local path.
        throw "ModelPath '$ModelPath' is not a valid HuggingFace repo ID (expected 'owner/repo' format) and is not a rooted local path."
    }

    Write-Verbose "Loading media model: $ModelPath (GpuLayers=$GpuLayers)"

    try {
        $result = [PcaiNative.MediaModule]::pcai_media_load_model($ModelPath, $GpuLayers)
        if ($result -ne 0) {
            $errorMsg = [PcaiNative.MediaModule]::GetLastError()
            throw "Failed to load media model '$ModelPath': $errorMsg"
        }

        $script:ModelLoaded = $true
        $script:CurrentModel = $ModelPath
        Write-Verbose "Media model loaded successfully: $ModelPath"
        return [PSCustomObject]@{
            Success   = $true
            ModelPath = $ModelPath
            GpuLayers = $GpuLayers
        }
    } catch {
        $script:ModelLoaded = $false
        $script:CurrentModel = $null
        throw "Media model loading failed: $_"
    }
}

function New-PcaiImage {
    <#
    .SYNOPSIS
        Generate an image from a text prompt.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::GenerateImage() to produce a PNG image
        from the given text prompt using the loaded Janus-Pro model.  If
        -OutputPath is not specified, a timestamped file is created on the
        Desktop.
    .PARAMETER Prompt
        Text description of the image to generate.
    .PARAMETER OutputPath
        Absolute file path where the PNG will be saved.  When omitted an
        auto-generated path is used: Desktop\janus-yyyyMMdd-HHmmss.png.
    .PARAMETER CfgScale
        Classifier-Free Guidance scale.  Higher values produce images more
        faithful to the prompt.  Typical range 1.0 - 10.0.  Defaults to 5.0.
    .PARAMETER Temperature
        Sampling temperature.  1.0 = neutral, lower = sharper.  Defaults to 1.0.
    .EXAMPLE
        New-PcaiImage -Prompt "a glowing circuit board"
    .EXAMPLE
        New-PcaiImage -Prompt "sunset over mountains" -OutputPath "C:\Images\sunset.png" -CfgScale 7.0
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [string]$OutputPath,

        [Parameter()]
        [ValidateRange(0.1, 20.0)]
        [float]$CfgScale = 5.0,

        [Parameter()]
        [ValidateRange(0.01, 2.0)]
        [float]$Temperature = 1.0
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiMediaModel first.'
    }

    # Auto-generate output path if not provided
    if (-not $OutputPath) {
        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $OutputPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "janus-$timestamp.png"
    }

    # Ensure parent directory exists
    $parentDir = Split-Path $OutputPath -Parent
    if ($parentDir -and -not (Test-Path $parentDir)) {
        try {
            New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            Write-Verbose "Created output directory: $parentDir"
        } catch {
            throw "Failed to create output directory '$parentDir': $_"
        }
    }

    Write-Verbose "Generating image for prompt: '$Prompt' -> $OutputPath"

    try {
        $result = [PcaiNative.MediaModule]::GenerateImage($Prompt, $OutputPath, $CfgScale, $Temperature)
        if ($null -ne $result) {
            throw "Image generation failed: $result"
        }

        Write-Verbose "Image saved to: $OutputPath"
        return [PSCustomObject]@{
            Success    = $true
            Prompt     = $Prompt
            OutputPath = $OutputPath
            CfgScale   = $CfgScale
            Temperature = $Temperature
        }
    } catch {
        throw "Image generation error: $_"
    }
}

function Get-PcaiImageAnalysis {
    <#
    .SYNOPSIS
        Run image-to-text understanding on an input image.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::UnderstandImage() to analyse the given
        image with the loaded Janus-Pro model and return a text description or
        answer to a question about the image contents.
    .PARAMETER ImagePath
        Absolute file path to the image to analyse (PNG, JPEG, BMP, etc.).
    .PARAMETER Question
        The prompt or question to ask about the image.  Defaults to
        "Describe this image in detail."
    .PARAMETER MaxTokens
        Maximum number of tokens to generate in the response.  Defaults to 512.
    .PARAMETER Temperature
        Sampling temperature for text generation.  0.0 = greedy, 1.0 = creative.
        Defaults to 0.7.
    .EXAMPLE
        Get-PcaiImageAnalysis -ImagePath "C:\Photos\circuit.png"
    .EXAMPLE
        Get-PcaiImageAnalysis -ImagePath "C:\Photos\error.png" -Question "What error is shown in this screenshot?"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$ImagePath,

        [Parameter()]
        [string]$Question = 'Describe this image in detail.',

        [Parameter()]
        [uint32]$MaxTokens = 512,

        [Parameter()]
        [ValidateRange(0.0, 2.0)]
        [float]$Temperature = 0.7
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiMediaModel first.'
    }

    if (-not (Test-Path $ImagePath)) {
        throw "Image file not found: $ImagePath"
    }

    Write-Verbose "Analysing image: $ImagePath (Question: '$Question')"

    try {
        $result = [PcaiNative.MediaModule]::UnderstandImage($ImagePath, $Question, $MaxTokens, $Temperature)
        if ($null -eq $result) {
            $errorMsg = [PcaiNative.MediaModule]::GetLastError()
            throw "Image analysis failed: $errorMsg"
        }

        Write-Verbose 'Image analysis complete'
        return [PSCustomObject]@{
            Success     = $true
            ImagePath   = $ImagePath
            Question    = $Question
            Response    = $result
            MaxTokens   = $MaxTokens
            Temperature = $Temperature
        }
    } catch {
        throw "Image analysis error: $_"
    }
}

function Stop-PcaiMedia {
    <#
    .SYNOPSIS
        Shut down the media pipeline and free all resources.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::pcai_media_shutdown() to release the
        loaded model and pipeline state.  After this call, Import-PcaiMediaModel
        must be called again before generation or analysis can be performed.
    .EXAMPLE
        Stop-PcaiMedia
    #>
    [CmdletBinding()]
    param()

    if ($script:Initialized) {
        Write-Verbose 'Shutting down media pipeline...'
        try {
            Clear-PcaiImageAsyncRequests
            [PcaiNative.MediaModule]::pcai_media_shutdown()
            $script:Initialized = $false
            $script:ModelLoaded = $false
            $script:CurrentModel = $null
            Write-Verbose 'Media pipeline shut down successfully'
        } catch {
            Write-Warning "Error during media shutdown: $_"
        }
    }
}

function Get-PcaiMediaStatus {
    <#
    .SYNOPSIS
        Return the current state of the pcai-media pipeline.
    .DESCRIPTION
        Returns a PSCustomObject with Initialized, ModelLoaded, and CurrentModel
        properties reflecting the module-scoped state of the media pipeline.
    .EXAMPLE
        Get-PcaiMediaStatus
    #>
    [CmdletBinding()]
    param()

    return [PSCustomObject]@{
        Initialized  = $script:Initialized
        ModelLoaded  = $script:ModelLoaded
        CurrentModel = $script:CurrentModel
    }
}

function New-PcaiImageAsync {
    <#
    .SYNOPSIS
        Generate an image asynchronously from a text prompt.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::GenerateImageNativeAsync() to start async
        image generation.  Returns a request ID that can be polled or awaited.
    .PARAMETER Prompt
        Text description of the image to generate.
    .PARAMETER OutputPath
        Absolute file path where the PNG will be saved.
    .PARAMETER CfgScale
        Classifier-Free Guidance scale.  Defaults to 5.0.
    .PARAMETER Temperature
        Sampling temperature.  Defaults to 1.0.
    .PARAMETER PollIntervalMs
        Polling interval used by the native C# async wrapper. Defaults to 100ms.
    .EXAMPLE
        $id = New-PcaiImageAsync -Prompt "a sunset" -OutputPath "C:\out.png"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [string]$OutputPath,

        [Parameter()]
        [ValidateRange(0.1, 20.0)]
        [float]$CfgScale = 5.0,

        [Parameter()]
        [ValidateRange(0.01, 2.0)]
        [float]$Temperature = 1.0,

        [Parameter()]
        [ValidateRange(10, 60000)]
        [int]$PollIntervalMs = 100
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiMediaModel first.'
    }

    if (-not $OutputPath) {
        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $OutputPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "janus-$timestamp.png"
    }

    $parentDir = Split-Path $OutputPath -Parent
    if ($parentDir -and -not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    Write-Verbose "Starting async image generation: '$Prompt' -> $OutputPath"

    try {
        $requestId = New-PcaiAsyncRequestId
        $cancellationTokenSource = [System.Threading.CancellationTokenSource]::new()
        $task = [PcaiNative.MediaModule]::GenerateImageNativeAsync(
            $Prompt,
            $OutputPath,
            $CfgScale,
            $Temperature,
            $PollIntervalMs,
            $cancellationTokenSource.Token
        )

        $entry = [PSCustomObject]@{
            RequestId               = $requestId
            Prompt                  = $Prompt
            OutputPath              = $OutputPath
            Task                    = $task
            CancellationTokenSource = $cancellationTokenSource
            PollIntervalMs          = $PollIntervalMs
            StartedAt               = Get-Date
        }
        $script:AsyncImageRequests[$requestId] = $entry

        $state = Get-PcaiImageAsyncCompletionState -Task $task
        if ($state.Status -eq 'Failed') {
            Remove-PcaiImageAsyncRequest -RequestId $requestId
            $errorMsg = if ($state.Error) { $state.Error } else { [PcaiNative.MediaModule]::GetLastError() }
            throw "Async image generation failed to start: $errorMsg"
        }

        return [PSCustomObject]@{
            RequestId      = $requestId
            Prompt         = $Prompt
            OutputPath     = $OutputPath
            Status         = $state.Status
            TaskStatus     = $state.TaskStatus
            PollIntervalMs = $PollIntervalMs
            StartedAt      = $entry.StartedAt
        }
    } catch {
        throw "Async image generation error: $_"
    }
}

function Get-PcaiImageAsyncStatus {
    <#
    .SYNOPSIS
        Return the current status of an async image generation request.
    .DESCRIPTION
        Looks up a request created by New-PcaiImageAsync and reports the
        underlying C# task status together with the module-side state.
    .PARAMETER RequestId
        Identifier returned by New-PcaiImageAsync.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [int]$RequestId
    )

    $entry = Get-PcaiImageAsyncRequest -RequestId $RequestId
    $state = Get-PcaiImageAsyncCompletionState -Task $entry.Task
    return [PSCustomObject]@{
        RequestId      = $entry.RequestId
        Prompt         = $entry.Prompt
        OutputPath     = $entry.OutputPath
        Status         = $state.Status
        TaskStatus     = $state.TaskStatus
        Result         = $state.Result
        Error          = $state.Error
        PollIntervalMs = $entry.PollIntervalMs
        StartedAt      = $entry.StartedAt
    }
}

function Wait-PcaiImageAsync {
    <#
    .SYNOPSIS
        Wait for an async image generation request to finish.
    .DESCRIPTION
        Blocks until the request completes, then returns a success object or
        throws if the native async wrapper reported an error.
    .PARAMETER RequestId
        Identifier returned by New-PcaiImageAsync.
    .PARAMETER TimeoutSeconds
        Optional timeout for the wait operation. Defaults to no timeout.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [int]$RequestId,

        [Parameter()]
        [ValidateRange(0, 86400)]
        [int]$TimeoutSeconds = 0
    )

    $entry = Get-PcaiImageAsyncRequest -RequestId $RequestId
    $task = $entry.Task

    try {
        if ($TimeoutSeconds -gt 0) {
            if (-not $task.Wait([TimeSpan]::FromSeconds($TimeoutSeconds))) {
                throw "Timed out waiting for async image request $RequestId"
            }
        } else {
            $task.GetAwaiter().GetResult() | Out-Null
        }

        $state = Get-PcaiImageAsyncCompletionState -Task $task
        if ($state.Status -ne 'Completed') {
            $errorMsg = if ($state.Error) { $state.Error } else { 'Async image generation failed' }
            throw $errorMsg
        }

        return [PSCustomObject]@{
            Success        = $true
            RequestId      = $entry.RequestId
            Prompt         = $entry.Prompt
            OutputPath     = $entry.OutputPath
            Status         = $state.Status
            TaskStatus     = $state.TaskStatus
            PollIntervalMs = $entry.PollIntervalMs
            StartedAt      = $entry.StartedAt
        }
    } catch {
        throw "Async image generation wait failed: $_"
    } finally {
        Remove-PcaiImageAsyncRequest -RequestId $RequestId
    }
}

function Invoke-PcaiUpscale {
    <#
    .SYNOPSIS
        Upscale an image by 4x using RealESRGAN.
    .DESCRIPTION
        Calls [PcaiNative.MediaModule]::UpscaleImage() to perform 4x super-resolution
        upscaling on the given input image.  Requires pcai_media.dll built with the
        'upscale' feature and a RealESRGAN ONNX model.
    .PARAMETER InputPath
        Absolute file path to the image to upscale.
    .PARAMETER OutputPath
        Absolute file path where the upscaled image will be saved.
    .PARAMETER ModelPath
        Path to the RealESRGAN ONNX model file.  Defaults to
        Models/RealESRGAN/RealESRGAN_x4.onnx relative to the project root.
    .EXAMPLE
        Invoke-PcaiUpscale -InputPath "C:\Photos\small.png" -OutputPath "C:\Photos\small_4x.png"
    .EXAMPLE
        Invoke-PcaiUpscale -InputPath .\input.jpg -OutputPath .\output.png -ModelPath "C:\Models\esrgan.onnx"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$InputPath,

        [Parameter(Mandatory, Position = 1)]
        [string]$OutputPath,

        [Parameter()]
        [string]$ModelPath
    )

    if (-not $script:Initialized) {
        throw 'Media pipeline not initialized. Call Initialize-PcaiMedia first.'
    }

    if (-not (Test-Path $InputPath)) {
        throw "Input image not found: $InputPath"
    }

    if (-not $ModelPath) {
        $projectRoot = Get-PcaiProjectRoot
        $ModelPath = Join-Path $projectRoot 'Models\RealESRGAN\RealESRGAN_x4.onnx'
    }

    if (-not (Test-Path $ModelPath)) {
        throw "RealESRGAN ONNX model not found: $ModelPath"
    }

    $parentDir = Split-Path $OutputPath -Parent
    if ($parentDir -and -not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    Write-Verbose "Upscaling image: $InputPath -> $OutputPath (model: $ModelPath)"

    try {
        $nativeError = [PcaiNative.MediaModule]::UpscaleImage($ModelPath, $InputPath, $OutputPath)
        if ($null -ne $nativeError) {
            throw "Image upscaling failed: $nativeError"
        }

        Write-Verbose "Upscaled image saved to: $OutputPath"
        return [PSCustomObject]@{
            Success    = $true
            InputPath  = $InputPath
            OutputPath = $OutputPath
            ScaleFactor = 4
        }
    } catch {
        throw "Image upscaling error: $_"
    }
}

#endregion

#region Module Cleanup
if ($MyInvocation.MyCommand.ScriptBlock.Module) {
    $MyInvocation.MyCommand.ScriptBlock.Module.OnRemove = {
        Clear-PcaiImageAsyncRequests
        if ($script:Initialized) {
            try {
                [PcaiNative.MediaModule]::pcai_media_shutdown()
            } catch {
                # Suppress errors during module unload
            }
        }
    }
}
#endregion

#region Module Exports
Export-ModuleMember -Function @(
    'Initialize-PcaiMedia',
    'Import-PcaiMediaModel',
    'New-PcaiImage',
    'New-PcaiImageAsync',
    'Get-PcaiImageAsyncStatus',
    'Wait-PcaiImageAsync',
    'Get-PcaiImageAnalysis',
    'Invoke-PcaiUpscale',
    'Stop-PcaiMedia',
    'Get-PcaiMediaStatus'
)
#endregion
