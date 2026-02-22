#Requires -Version 5.1
<#
.SYNOPSIS
    Sets up the PC_AI development environment

.DESCRIPTION
    Installs required dependencies, configures pre-commit hooks, and validates the development environment

.PARAMETER SkipPreCommit
    Skip installation of pre-commit hooks

.PARAMETER SkipModules
    Skip installation of PowerShell modules

.EXAMPLE
    .\Setup-DevEnvironment.ps1
    Sets up the complete development environment

.EXAMPLE
    .\Setup-DevEnvironment.ps1 -SkipPreCommit
    Sets up environment without pre-commit hooks
#>

[CmdletBinding()]
param(
    [switch]$SkipPreCommit,
    [switch]$SkipModules
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "`n=== PC_AI Development Environment Setup ===" -ForegroundColor Cyan
Write-Host "This script will install required dependencies and configure your environment.`n" -ForegroundColor Gray

# Check PowerShell version
Write-Host "Checking PowerShell version..." -ForegroundColor Cyan
$psVersion = $PSVersionTable.PSVersion
Write-Host "  PowerShell $($psVersion.Major).$($psVersion.Minor)" -ForegroundColor Gray

if ($psVersion.Major -lt 5 -or ($psVersion.Major -eq 5 -and $psVersion.Minor -lt 1)) {
    Write-Error "PowerShell 5.1 or higher is required. Current version: $psVersion"
}
Write-Host "  ✓ PowerShell version is compatible" -ForegroundColor Green

# Check build/deploy toolchain prerequisites
Write-Host "`nChecking build toolchain..." -ForegroundColor Cyan
$toolchainWarnings = @()

$requiredTools = @(
    @{ Name = 'dotnet'; Hint = 'Install .NET SDK 8.0+ from https://dotnet.microsoft.com/download'; VersionCommand = { dotnet --version } },
    @{ Name = 'rustup'; Hint = 'Install rustup from https://rustup.rs/'; VersionCommand = { rustup --version } },
    @{ Name = 'cargo'; Hint = 'Install Rust toolchain via rustup'; VersionCommand = { cargo --version } },
    @{ Name = 'rustc'; Hint = 'Install Rust toolchain via rustup'; VersionCommand = { rustc --version } },
    @{ Name = 'cmake'; Hint = 'Install CMake 3.x (Visual Studio bundle or https://cmake.org/download/)'; VersionCommand = { cmake --version | Select-Object -First 1 } },
    @{ Name = 'ninja'; Hint = 'Install Ninja (e.g., choco install ninja -y)'; VersionCommand = { ninja --version } }
)

foreach ($tool in $requiredTools) {
    $cmd = Get-Command $tool.Name -ErrorAction SilentlyContinue
    if ($cmd) {
        $version = ''
        try {
            $version = (& $tool.VersionCommand 2>&1 | Select-Object -First 1).ToString()
        } catch {
            $version = '(version unavailable)'
        }
        Write-Host "  ✓ $($tool.Name): $version" -ForegroundColor Green
    } else {
        $msg = "$($tool.Name) not found. $($tool.Hint)"
        $toolchainWarnings += $msg
        Write-Warning "  $msg"
    }
}

# MSVC/C++ Build Tools check
$clCmd = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clCmd) {
    Write-Host "  ✓ MSVC compiler detected: $($clCmd.Source)" -ForegroundColor Green
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        try {
            $vsPath = & $vswhere -latest -property installationPath 2>$null
            if ($vsPath) {
                Write-Host "  [WARN] Visual Studio found at $vsPath, but cl.exe is not in PATH for this shell" -ForegroundColor Yellow
                Write-Host "    Use a Developer PowerShell, or run the build scripts that initialize VsDevShell." -ForegroundColor Yellow
            } else {
                $msg = 'MSVC Build Tools not detected by vswhere. Install Visual Studio 2022 C++ Build Tools + Windows SDK.'
                $toolchainWarnings += $msg
                Write-Warning "  $msg"
            }
        } catch {
            $msg = 'Failed to query Visual Studio Build Tools with vswhere.'
            $toolchainWarnings += $msg
            Write-Warning "  $msg"
        }
    } else {
        $msg = 'vswhere.exe not found. Install Visual Studio 2022 Build Tools (C++ workload + Windows SDK).'
        $toolchainWarnings += $msg
        Write-Warning "  $msg"
    }
}

# Optional CUDA check
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    try {
        $nvccVersion = (& nvcc --version 2>&1 | Select-Object -Last 1).ToString()
        Write-Host "  ✓ CUDA nvcc detected: $nvccVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ✓ CUDA nvcc detected" -ForegroundColor Green
    }
} else {
    Write-Host "  [WARN] CUDA toolkit not detected (optional unless using -EnableCuda builds)." -ForegroundColor Yellow
}

if ($toolchainWarnings.Count -gt 0) {
    Write-Warning "Toolchain checks completed with $($toolchainWarnings.Count) issue(s). See warnings above before running Build.ps1."
} else {
    Write-Host "  ✓ Build toolchain checks passed" -ForegroundColor Green
}

# Install PowerShell modules
if (-not $SkipModules) {
    Write-Host "`nInstalling required PowerShell modules..." -ForegroundColor Cyan

    $requiredModules = @(
        @{Name = 'PSScriptAnalyzer'; MinVersion = '1.21.0' },
        @{Name = 'Pester'; MinVersion = '5.6.1' },
        @{Name = 'PowerShellGet'; MinVersion = '2.2.5' }
    )

    foreach ($module in $requiredModules) {
        Write-Host "  Checking $($module.Name)..." -ForegroundColor Gray

        $installed = Get-Module -ListAvailable -Name $module.Name |
            Where-Object { $_.Version -ge [version]$module.MinVersion } |
            Sort-Object Version -Descending |
            Select-Object -First 1

        if ($installed) {
            Write-Host "    ✓ $($module.Name) v$($installed.Version) already installed" -ForegroundColor Green
        } else {
            Write-Host "    Installing $($module.Name)..." -ForegroundColor Yellow
            try {
                Install-Module -Name $module.Name -MinimumVersion $module.MinVersion -Scope CurrentUser -Force -AllowClobber
                Write-Host "    ✓ $($module.Name) installed successfully" -ForegroundColor Green
            } catch {
                Write-Warning "    Failed to install $($module.Name): $_"
            }
        }
    }
} else {
    Write-Host "`nSkipping PowerShell module installation" -ForegroundColor Yellow
}

# Check for Python and pre-commit
if (-not $SkipPreCommit) {
    Write-Host "`nChecking for Python..." -ForegroundColor Cyan

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonVersion = & python --version 2>&1
        Write-Host "  ✓ $pythonVersion found" -ForegroundColor Green

        Write-Host "`nChecking for pre-commit..." -ForegroundColor Cyan
        $preCommitCmd = Get-Command pre-commit -ErrorAction SilentlyContinue

        if (-not $preCommitCmd) {
            Write-Host "  Installing pre-commit..." -ForegroundColor Yellow
            try {
                & python -m pip install pre-commit --user --quiet
                Write-Host "  ✓ pre-commit installed successfully" -ForegroundColor Green
            } catch {
                Write-Warning "  Failed to install pre-commit: $_"
                Write-Host "  You can install it manually: pip install pre-commit" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ✓ pre-commit already installed" -ForegroundColor Green
        }

        # Install pre-commit hooks
        if (Test-Path .pre-commit-config.yaml) {
            Write-Host "`nInstalling pre-commit hooks..." -ForegroundColor Cyan
            try {
                & pre-commit install
                Write-Host "  ✓ Pre-commit hooks installed" -ForegroundColor Green
            } catch {
                Write-Warning "  Failed to install pre-commit hooks: $_"
            }
        }
    } else {
        Write-Host "  Python not found. Pre-commit hooks will not be available." -ForegroundColor Yellow
        Write-Host "  Install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nSkipping pre-commit setup" -ForegroundColor Yellow
}

# Validate module manifests
Write-Host "`nValidating module manifests..." -ForegroundColor Cyan
$modulesPath = Join-Path $PSScriptRoot "Modules"

if (Test-Path $modulesPath) {
    $manifestErrors = @()

    Get-ChildItem -Path $modulesPath -Directory | ForEach-Object {
        $manifestPath = Join-Path $_.FullName "$($_.Name).psd1"

        if (Test-Path $manifestPath) {
            Write-Host "  Validating $($_.Name)..." -ForegroundColor Gray
            try {
                $null = Test-ModuleManifest -Path $manifestPath -ErrorAction Stop
                Write-Host "    ✓ $($_.Name) manifest is valid" -ForegroundColor Green
            } catch {
                $manifestErrors += "$($_.Name): $_"
                Write-Host "    ✗ $($_.Name) manifest has errors" -ForegroundColor Red
            }
        }
    }

    if ($manifestErrors.Count -eq 0) {
        Write-Host "  ✓ All module manifests are valid" -ForegroundColor Green
    } else {
        Write-Warning "Some module manifests have errors:"
        $manifestErrors | ForEach-Object { Write-Warning "  $_" }
    }
} else {
    Write-Warning "Modules directory not found at: $modulesPath"
}

# Validate PSScriptAnalyzer settings
Write-Host "`nValidating PSScriptAnalyzer settings..." -ForegroundColor Cyan
$analyzerSettings = Join-Path $PSScriptRoot "PSScriptAnalyzerSettings.psd1"

if (Test-Path $analyzerSettings) {
    try {
        $settings = Import-PowerShellDataFile -Path $analyzerSettings
        Write-Host "  ✓ PSScriptAnalyzer settings are valid" -ForegroundColor Green

        # Run quick analysis
        Write-Host "`nRunning code quality check..." -ForegroundColor Cyan
        $issues = Invoke-ScriptAnalyzer -Path $PSScriptRoot -Settings $analyzerSettings -Recurse

        $errors = $issues | Where-Object { $_.Severity -eq 'Error' }
        $warnings = $issues | Where-Object { $_.Severity -eq 'Warning' }

        Write-Host "  Errors: $($errors.Count)" -ForegroundColor $(if ($errors.Count -eq 0) { 'Green' } else { 'Red' })
        Write-Host "  Warnings: $($warnings.Count)" -ForegroundColor $(if ($warnings.Count -eq 0) { 'Green' } else { 'Yellow' })

        if ($errors.Count -gt 0) {
            Write-Warning "Please fix PSScriptAnalyzer errors before committing"
        }
    } catch {
        Write-Warning "Failed to validate PSScriptAnalyzer settings: $_"
    }
} else {
    Write-Warning "PSScriptAnalyzer settings not found at: $analyzerSettings"
}

# Check Git configuration
Write-Host "`nChecking Git configuration..." -ForegroundColor Cyan
$gitCmd = Get-Command git -ErrorAction SilentlyContinue

if ($gitCmd) {
    $gitUser = git config user.name 2>$null
    $gitEmail = git config user.email 2>$null

    if ($gitUser -and $gitEmail) {
        Write-Host "  ✓ Git user configured: $gitUser <$gitEmail>" -ForegroundColor Green
    } else {
        Write-Host "  Git user not configured. Set up with:" -ForegroundColor Yellow
        Write-Host "    git config --global user.name ""Your Name""" -ForegroundColor Gray
        Write-Host "    git config --global user.email ""your.email@example.com""" -ForegroundColor Gray
    }

    # Check for uncommitted changes
    $status = git status --porcelain 2>$null
    if ($status) {
        Write-Host "  ⚠️  You have uncommitted changes" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Git not found. Install from https://git-scm.com/" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host "`nChecking directory structure..." -ForegroundColor Cyan
$requiredDirs = @('Tests', 'Reports', 'Config')

foreach ($dir in $requiredDirs) {
    $dirPath = Join-Path $PSScriptRoot $dir
    if (-not (Test-Path $dirPath)) {
        Write-Host "  Creating $dir directory..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        Write-Host "    ✓ $dir directory created" -ForegroundColor Green
    } else {
        Write-Host "  ✓ $dir directory exists" -ForegroundColor Green
    }
}

# Create .gitkeep in Reports directory
$reportsGitKeep = Join-Path $PSScriptRoot "Reports\.gitkeep"
if (-not (Test-Path $reportsGitKeep)) {
    New-Item -ItemType File -Path $reportsGitKeep -Force | Out-Null
}

# Summary
Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host @"

Next steps:
1. Import a module: Import-Module .\Modules\PC-AI.Hardware\PC-AI.Hardware.psd1
2. Run tests: .\Tests\.pester.ps1
3. Run diagnostics: .\PC-AI.ps1 diagnose hardware
4. Check code quality: Invoke-ScriptAnalyzer -Path . -Recurse -Settings PSScriptAnalyzerSettings.psd1

For more information, see CI-CD-GUIDE.md
"@ -ForegroundColor Gray

Write-Host "`n✓ Development environment is ready!" -ForegroundColor Green
