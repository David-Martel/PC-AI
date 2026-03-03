#Requires -Version 5.1

<#
.SYNOPSIS
    Shared test utilities for PCAI Inference tests.

.DESCRIPTION
    Common helper functions for integration, E2E, and functional tests:
    - Test model discovery (env var, LM Studio, Ollama)
    - DLL availability checks
    - Project path resolution
    - Test prerequisite validation

.NOTES
    Import this module in BeforeAll blocks:
    Import-Module (Join-Path $PSScriptRoot "..\Helpers\TestHelpers.psm1") -Force
#>

$repoRootResolver = Join-Path $PSScriptRoot 'Resolve-TestRepoRoot.ps1'
if (Test-Path $repoRootResolver) {
    . $repoRootResolver
}

function Get-ProjectRoot {
    <#
    .SYNOPSIS
        Get the project root directory.
    .DESCRIPTION
        Returns the absolute path to the PC_AI project root.
        Works from any test subdirectory by traversing upward.
    #>
    param(
        [Parameter(Mandatory = $false)]
        [string]$StartPath
    )

    if (-not $StartPath) {
        $StartPath = $PSScriptRoot
    }

    if (Get-Command Resolve-TestRepoRoot -ErrorAction SilentlyContinue) {
        return Resolve-TestRepoRoot -StartPath $StartPath
    }

    # Fallback when helper is unavailable.
    if (Test-Path $StartPath -PathType Leaf) {
        $StartPath = Split-Path -Parent $StartPath
    }
    return Split-Path -Parent (Split-Path -Parent $StartPath)
}

function Get-TestModelPath {
    <#
    .SYNOPSIS
        Find a test model for inference tests.
    .DESCRIPTION
        Searches for a .gguf model file in the following order:
        1. Config/llm-config.json (providers.pcai-native.modelPath)
        2. LM Studio cache directory
        3. Ollama cache directory
    .OUTPUTS
        String path to model file, or $null if not found.
    #>

    $projectRoot = Get-ProjectRoot -StartPath $PSScriptRoot
    if ($projectRoot) {
        $configPath = Join-Path $projectRoot 'Config\llm-config.json'
        if (Test-Path $configPath) {
            try {
                $config = Get-Content $configPath -Raw | ConvertFrom-Json
                $modelPath = $config.providers.'pcai-native'.modelPath
                if ($modelPath -and (Test-Path $modelPath)) {
                    Write-Verbose "Using model from Config/llm-config.json: $modelPath"
                    return $modelPath
                }
            } catch {
                Write-Verbose "Failed to parse llm-config.json: $_"
            }
        }
    }

    # Check LM Studio cache (Windows)
    $localAppData = [Environment]::GetFolderPath('LocalApplicationData')
    $lmStudioPath = Join-Path $localAppData "lm-studio\models"
    if (Test-Path $lmStudioPath) {
        $ggufFiles = Get-ChildItem -Path $lmStudioPath -Filter "*.gguf" -Recurse -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($ggufFiles) {
            Write-Verbose "Found model in LM Studio: $($ggufFiles.FullName)"
            return $ggufFiles.FullName
        }
    }

    # Check Ollama cache
    $userProfile = [Environment]::GetFolderPath('UserProfile')
    $ollamaPath = Join-Path $userProfile ".ollama\models\blobs"
    if (Test-Path $ollamaPath) {
        $ggufFiles = Get-ChildItem -Path $ollamaPath -Filter "*.gguf" -Recurse -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($ggufFiles) {
            Write-Verbose "Found model in Ollama: $($ggufFiles.FullName)"
            return $ggufFiles.FullName
        }
    }

    Write-Verbose "No test model found. Set providers.pcai-native.modelPath in Config/llm-config.json."
    return $null
}

function Test-InferenceDllAvailable {
    <#
    .SYNOPSIS
        Check if pcai_inference.dll exists.
    .DESCRIPTION
        Checks if the native inference DLL has been built and is available in bin/.
    .OUTPUTS
        Boolean indicating DLL availability.
    #>
    param(
        [Parameter(Mandatory = $false)]
        [string]$ProjectRoot
    )

    if (-not $ProjectRoot) {
        $ProjectRoot = Get-ProjectRoot -StartPath $PSScriptRoot
    }

    $dllPath = Resolve-InferenceDllPath -ProjectRoot $ProjectRoot
    return $null -ne $dllPath
}

function Get-InferenceDllPath {
    <#
    .SYNOPSIS
        Get the full path to pcai_inference.dll.
    .OUTPUTS
        String path to DLL.
    #>
    param(
        [Parameter(Mandatory = $false)]
        [string]$ProjectRoot
    )

    if (-not $ProjectRoot) {
        $ProjectRoot = Get-ProjectRoot -StartPath $PSScriptRoot
    }

    return Resolve-InferenceDllPath -ProjectRoot $ProjectRoot
}

function Resolve-InferenceDllPath {
    <#
    .SYNOPSIS
        Resolve the best available pcai_inference.dll path.
    #>
    param(
        [Parameter(Mandatory = $false)]
        [string]$ProjectRoot
    )

    if (-not $ProjectRoot) {
        $ProjectRoot = Get-ProjectRoot -StartPath $PSScriptRoot
    }

    $candidates = @(
        (Join-Path $ProjectRoot "bin\pcai_inference.dll"),
        (Join-Path $ProjectRoot "bin\Release\pcai_inference.dll"),
        (Join-Path $ProjectRoot "bin\Debug\pcai_inference.dll"),
        (Join-Path $env:USERPROFILE ".local\bin\pcai_inference.dll"),
        (Join-Path $env:CARGO_TARGET_DIR "release\pcai_inference.dll"),
        'T:\RustCache\cargo-target\release\pcai_inference.dll'
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Assert-TestPrerequisites {
    <#
    .SYNOPSIS
        Validate test prerequisites and skip test if not met.
    .DESCRIPTION
        Checks for required test conditions (DLL, model, backend).
        If prerequisites are not met, uses Set-ItResult to skip the test.
    .PARAMETER RequireModel
        Require a test model to be available.
    .PARAMETER RequireDll
        Require pcai_inference.dll to be built.
    .PARAMETER RequireBackend
        Require a specific backend to be available (tests initialization).
    .OUTPUTS
        Boolean - $true if all prerequisites met, $false if test should be skipped.
    #>
    param(
        [switch]$RequireModel,
        [switch]$RequireDll,
        [switch]$RequireBackend,
        [string]$ProjectRoot
    )

    if (-not $ProjectRoot) {
        $ProjectRoot = Get-ProjectRoot -StartPath $PSScriptRoot
    }

    if ($RequireDll) {
        $dllPath = Get-InferenceDllPath -ProjectRoot $ProjectRoot
        if (-not (Test-Path $dllPath)) {
            Set-ItResult -Skipped -Because "DLL not available (build with Native\pcai_core\pcai_inference\Invoke-PcaiBuild.ps1)"
            return $false
        }
    }

    if ($RequireModel) {
        $modelPath = Get-TestModelPath
        if (-not $modelPath) {
            Set-ItResult -Skipped -Because "No test model (configure Config/llm-config.json)"
            return $false
        }
    }

    if ($RequireBackend) {
        # Import module if not already loaded
        $modulePath = Join-Path $ProjectRoot "Modules\PcaiInference.psm1"
        if (Test-Path $modulePath) {
            Import-Module $modulePath -Force -ErrorAction SilentlyContinue
        }

        $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
        if (-not $init) {
            Set-ItResult -Skipped -Because "llamacpp backend not available"
            try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
            return $false
        }

        # Clean up after check
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    return $true
}

function Get-TestPaths {
    <#
    .SYNOPSIS
        Get all common test paths in a single object.
    .DESCRIPTION
        Returns a hashtable with all standard test paths:
        - ProjectRoot, BinDir, DeployDir, ModulePath, DllPath, etc.
    .OUTPUTS
        Hashtable with test paths.
    #>
    param(
        [Parameter(Mandatory = $false)]
        [string]$StartPath
    )

    # Use $PSScriptRoot if no StartPath provided
    if (-not $StartPath) {
        $StartPath = $PSScriptRoot
    }

    # Resolve relative paths to absolute
    if ($StartPath -and (Test-Path $StartPath)) {
        $StartPath = Resolve-Path $StartPath
    }

    $projectRoot = Get-ProjectRoot -StartPath $StartPath

    if (-not $projectRoot) {
        throw "Could not determine project root from path: $StartPath"
    }

    $binDir = Join-Path $projectRoot "bin"
    $deployDir = Join-Path $projectRoot "Native\pcai_core\pcai_inference"
    $modulePath = Join-Path $projectRoot "Modules\PcaiInference.psm1"
    $dllPath = Resolve-InferenceDllPath -ProjectRoot $projectRoot

    return @{
        ProjectRoot = $projectRoot
        BinDir      = $binDir
        DeployDir   = $deployDir
        ModulePath  = $modulePath
        DllPath     = $dllPath
        DllName     = "pcai_inference.dll"
    }
}

# Export functions
Export-ModuleMember -Function @(
    'Get-ProjectRoot',
    'Get-TestModelPath',
    'Test-InferenceDllAvailable',
    'Get-InferenceDllPath',
    'Assert-TestPrerequisites',
    'Get-TestPaths'
)
