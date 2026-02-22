#Requires -Version 7.0
<#
.SYNOPSIS
    Validates PC-AI.Evaluation module dependencies

.DESCRIPTION
    Checks for required components:
    - PcaiInference PowerShell module
    - pcai_inference.dll native library
    - Required .NET assemblies
#>

$ErrorActionPreference = 'Stop'

# Find project root
$moduleRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $moduleRoot
$configPath = Join-Path $projectRoot 'Config\llm-config.json'
$config = $null
if (Test-Path $configPath) {
    try {
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
    } catch {
        Write-Verbose "Failed to parse ${configPath}: $_"
    }
}

# DLL search paths
$dllSearchPaths = @()
if ($config -and $config.nativeInference -and $config.nativeInference.dllSearchPaths) {
    foreach ($path in $config.nativeInference.dllSearchPaths) {
        if (-not $path) { continue }
        if ([System.IO.Path]::IsPathRooted($path)) {
            $dllSearchPaths += $path
        } else {
            $dllSearchPaths += (Join-Path $projectRoot $path)
        }
    }
}

$userProfile = [Environment]::GetFolderPath('UserProfile')
$dllSearchPaths += @(
    (Join-Path $projectRoot 'bin\Release\pcai_inference.dll'),
    (Join-Path $projectRoot 'bin\Debug\pcai_inference.dll'),
    (Join-Path $userProfile '.local\bin\pcai_inference.dll')
) | Where-Object { $_ }

# Check for DLL
$dllFound = $false
foreach ($path in $dllSearchPaths) {
    if (Test-Path $path -ErrorAction SilentlyContinue) {
        $dllFound = $true
        $script:PcaiDllPath = $path
        break
    }
}

if (-not $dllFound) {
    $buildInstructions = @"

╔══════════════════════════════════════════════════════════════════╗
║  PC-AI.Evaluation requires pcai_inference.dll                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Build the native library first:                                  ║
║                                                                   ║
║    .\Build.ps1 -Component inference                               ║
║                                                                   ║
║  Or build a specific backend via Build.ps1:                       ║
║                                                                   ║
║    .\Build.ps1 -Component mistralrs                               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"@
    Write-Warning $buildInstructions

    # Don't throw - allow module to load for documentation/offline use
    # Actual usage will fail when trying to invoke native functions
}

# Check for compiled server binaries
$exeSearchDirs = @()
if ($config -and $config.evaluation -and $config.evaluation.binSearchPaths) {
    foreach ($path in $config.evaluation.binSearchPaths) {
        if (-not $path) { continue }
        if ([System.IO.Path]::IsPathRooted($path)) {
            $exeSearchDirs += $path
        } else {
            $exeSearchDirs += (Join-Path $projectRoot $path)
        }
    }
}

$exeSearchDirs += @(
    (Join-Path $userProfile '.local\bin'),
    'T:\RustCache\cargo-target\release'
) | Where-Object { $_ }

$llamacppExe = $null
$mistralrsExe = $null
foreach ($dir in $exeSearchDirs) {
    if (-not $llamacppExe) {
        $candidate = Join-Path $dir 'pcai-llamacpp.exe'
        if (Test-Path $candidate -ErrorAction SilentlyContinue) { $llamacppExe = $candidate }
    }
    if (-not $mistralrsExe) {
        $candidate = Join-Path $dir 'pcai-mistralrs.exe'
        if (Test-Path $candidate -ErrorAction SilentlyContinue) { $mistralrsExe = $candidate }
    }
}

# Check for PcaiInference module
$pcaiModulePath = Join-Path $projectRoot 'Modules\PcaiInference.psm1'
if (-not (Test-Path $pcaiModulePath)) {
    Write-Warning "PcaiInference.psm1 not found at: $pcaiModulePath"
}

# Export validation results
$script:DependencyStatus = @{
    DllAvailable = $dllFound
    DllPath = $script:PcaiDllPath
    ModuleAvailable = Test-Path $pcaiModulePath
    ModulePath = $pcaiModulePath
    CompiledBackends = @{
        LlamaCppExe  = $llamacppExe
        MistralRsExe = $mistralrsExe
    }
    ValidationTime = [datetime]::UtcNow
}
