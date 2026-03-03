#Requires -Version 5.1
<#
.SYNOPSIS
    Builds and installs LLVM/Clang/LLD from source.
.DESCRIPTION
    Convenience script for building LLVM toolchain components from source using
    CMake + Ninja. Targets lld and clang by default, which provides lld-link.exe
    for fast Windows linking.

    Requires: Visual Studio 2022 Build Tools, CMake, Ninja, Git, Python 3.
.PARAMETER Version
    LLVM release tag to build (e.g., 'llvmorg-21.1.8'). Default: 'llvmorg-21.1.8'.
.PARAMETER InstallPrefix
    Installation directory. Default: 'C:\Program Files\LLVM'.
.PARAMETER SourceDir
    Directory for the LLVM source checkout. Default: 'T:\RustCache\llvm-src'.
.PARAMETER BuildDir
    Directory for the CMake build tree. Default: 'T:\RustCache\llvm-build'.
.PARAMETER Projects
    Semicolon-separated list of LLVM subprojects to build. Default: 'lld;clang'.
.PARAMETER Targets
    Semicolon-separated list of LLVM targets to enable. Default: 'X86;NVPTX'.
.PARAMETER Clean
    Remove build directory before configuring.
.EXAMPLE
    .\Install-LlvmFromSource.ps1
    # Builds LLVM 21.1.8 with lld+clang, installs to C:\Program Files\LLVM

    .\Install-LlvmFromSource.ps1 -Version 'llvmorg-22.0.0' -Projects 'lld;clang;compiler-rt'
    # Builds a newer version with additional project
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Version = 'llvmorg-21.1.8',
    [string]$InstallPrefix = 'C:\Program Files\LLVM',
    [string]$SourceDir = 'T:\RustCache\llvm-src',
    [string]$BuildDir = 'T:\RustCache\llvm-build',
    [string]$Projects = 'lld;clang',
    [string]$Targets = 'X86;NVPTX',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# Verify prerequisites
$requiredCmds = @('git', 'cmake', 'ninja', 'python')
foreach ($cmd in $requiredCmds) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "$cmd is required but not found in PATH."
        return
    }
}

# Step 1: Clone or update source
if (-not (Test-Path $SourceDir)) {
    Write-Host "[1/4] Shallow-cloning LLVM $Version..." -ForegroundColor Cyan
    if ($PSCmdlet.ShouldProcess($SourceDir, "git clone --depth 1 -b $Version")) {
        git clone --depth 1 -b $Version https://github.com/llvm/llvm-project.git $SourceDir
        if ($LASTEXITCODE -ne 0) { Write-Error "Git clone failed."; return }
    }
} else {
    Write-Host "[1/4] Source directory exists, checking tag..." -ForegroundColor Cyan
    Push-Location $SourceDir
    try {
        $currentTag = git describe --tags --exact-match 2>$null
        if ($currentTag -ne $Version) {
            Write-Host "  Fetching tag $Version..." -ForegroundColor Yellow
            if ($PSCmdlet.ShouldProcess($SourceDir, "git fetch origin tag $Version")) {
                git fetch --depth 1 origin tag $Version
                git checkout $Version
            }
        } else {
            Write-Host "  Already at $Version" -ForegroundColor Green
        }
    } finally {
        Pop-Location
    }
}

# Step 2: Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "[2/4] Cleaning build directory..." -ForegroundColor Cyan
    if ($PSCmdlet.ShouldProcess($BuildDir, 'Remove build directory')) {
        Remove-Item -Path $BuildDir -Recurse -Force
    }
}
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# Step 3: CMake configure
Write-Host "[3/4] Configuring CMake (Ninja, Release)..." -ForegroundColor Cyan
$cmakeArgs = @(
    '-S', (Join-Path $SourceDir 'llvm'),
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_INSTALL_PREFIX=$InstallPrefix",
    "-DLLVM_ENABLE_PROJECTS=$Projects",
    "-DLLVM_TARGETS_TO_BUILD=$Targets",
    '-DLLVM_ENABLE_ASSERTIONS=OFF',
    '-DLLVM_INCLUDE_TESTS=OFF',
    '-DLLVM_INCLUDE_BENCHMARKS=OFF',
    '-DLLVM_INCLUDE_EXAMPLES=OFF',
    '-DCLANG_INCLUDE_TESTS=OFF'
)

if ($PSCmdlet.ShouldProcess($BuildDir, 'cmake configure')) {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed."; return }
}

# Step 4: Build and install
Write-Host "[4/4] Building and installing (this may take 20-60 minutes)..." -ForegroundColor Cyan
if ($PSCmdlet.ShouldProcess($InstallPrefix, 'cmake --build + --install')) {
    $jobs = [Environment]::ProcessorCount
    & cmake --build $BuildDir --config Release -j $jobs
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed."; return }

    & cmake --install $BuildDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Install failed."; return }
}

# Verify installation
$lldLink = Join-Path $InstallPrefix 'bin\lld-link.exe'
if (Test-Path $lldLink) {
    Write-Host "`nInstallation complete!" -ForegroundColor Green
    $ver = & (Join-Path $InstallPrefix 'bin\clang.exe') --version 2>$null | Select-Object -First 1
    Write-Host "  Version: $ver"
    Write-Host "  lld-link: $lldLink"
    Write-Host "  Install:  $InstallPrefix"
} else {
    Write-Warning "Build completed but lld-link.exe not found at expected path."
}
