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
$script:PcaiDllPath = $null

# Find project root
$moduleRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $moduleRoot
$configPath = Join-Path $projectRoot 'Config\llm-config.json'
$config = $null
if (Test-Path -LiteralPath $configPath -PathType Leaf) {
    try {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json -AsHashtable
        if ($config -isnot [System.Collections.IDictionary]) {
            throw 'Dependency configuration must be a JSON object.'
        }
    } catch {
        $config = $null
        Write-Warning "Failed to parse dependency configuration ${configPath} as a JSON object. Using default search paths."
    }
}

# Optional sections are absent in provider-only configurations. Invalid supplied
# paths must not coerce to strings or masquerade as available dependencies.
$getConfiguredPaths = {
    param([string]$Section, [string]$Property)
    if ($null -eq $config -or -not $config.Contains($Section) -or $null -eq $config[$Section]) { return }
    $settings = $config[$Section]
    if ($settings -isnot [System.Collections.IDictionary]) {
        Write-Warning "Dependency configuration section '$Section' must be an object. Using default search paths."
        return
    }
    if (-not $settings.Contains($Property) -or $null -eq $settings[$Property]) { return }
    foreach ($configuredPath in @($settings[$Property])) {
        if ($configuredPath -isnot [string] -or [string]::IsNullOrWhiteSpace($configuredPath)) {
            Write-Warning "Dependency configuration '$Section.$Property' contains an invalid path. Ignoring that entry."
            continue
        }
        if ([System.IO.Path]::IsPathRooted($configuredPath)) {
            $configuredPath
        } else {
            Join-Path $projectRoot $configuredPath
        }
    }
}

# DLL search paths
$dllSearchPaths = @(& $getConfiguredPaths 'nativeInference' 'dllSearchPaths')

$userProfile = [Environment]::GetFolderPath('UserProfile')
$dllSearchPaths += @(
    (Join-Path $projectRoot 'bin\Release\pcai_inference.dll'),
    (Join-Path $projectRoot 'bin\Debug\pcai_inference.dll'),
    (Join-Path $userProfile '.local\bin\pcai_inference.dll')
) | Where-Object { $_ }

# Check for DLL
$dllFound = $false
foreach ($path in $dllSearchPaths) {
    if (Test-Path -LiteralPath $path -PathType Leaf -ErrorAction SilentlyContinue) {
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
$exeSearchDirs = @(& $getConfiguredPaths 'evaluation' 'binSearchPaths')

$exeSearchDirs += @(
    (Join-Path $userProfile '.local\bin')
) | Where-Object { $_ }

$llamacppExe = $null
$mistralrsExe = $null
foreach ($dir in $exeSearchDirs) {
    if (-not $llamacppExe) {
        $candidate = Join-Path $dir 'pcai-llamacpp.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf -ErrorAction SilentlyContinue) { $llamacppExe = $candidate }
    }
    if (-not $mistralrsExe) {
        $candidate = Join-Path $dir 'pcai-mistralrs.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf -ErrorAction SilentlyContinue) { $mistralrsExe = $candidate }
    }
}

# Check for PcaiInference module
$pcaiModulePath = Join-Path $projectRoot 'Modules\PcaiInference.psm1'
if (-not (Test-Path -LiteralPath $pcaiModulePath -PathType Leaf)) {
    Write-Warning "PcaiInference.psm1 not found at: $pcaiModulePath"
}

# Export validation results
$script:DependencyStatus = @{
    DllAvailable = $dllFound
    DllPath = $script:PcaiDllPath
    ModuleAvailable = Test-Path -LiteralPath $pcaiModulePath -PathType Leaf
    ModulePath = $pcaiModulePath
    CompiledBackends = @{
        LlamaCppExe  = $llamacppExe
        MistralRsExe = $mistralrsExe
    }
    ValidationTime = [datetime]::UtcNow
}
