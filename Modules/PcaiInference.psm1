#Requires -Version 5.1

<#
.SYNOPSIS
    PowerShell FFI wrapper for pcai-inference native Rust library using PcaiNative.dll
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
        (Join-Path $projectRoot 'Native\pcai_core\pcai_inference\target\release\pcai_inference.dll'),
        (Join-Path $projectRoot 'Deploy\pcai-inference\target\release\pcai_inference.dll'),
        'T:\RustCache\cargo-target\release\pcai_inference.dll'
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
    'Stop-PcaiInference',
    'Close-PcaiInference',
    'Get-PcaiInferenceStatus',
    'Test-PcaiInference',
    'Test-PcaiDllVersion'
)
#endregion
