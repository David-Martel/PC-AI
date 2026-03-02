# Test CMake configuration directly
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$toolsDir = Join-Path $repoRoot 'Tools'
$cacheHelper = Join-Path $toolsDir 'Initialize-CacheEnvironment.ps1'
if (Test-Path $cacheHelper) {
    . $cacheHelper
    Initialize-CacheEnvironment -Quiet | Out-Null
}
$cmakeHelper = Join-Path $toolsDir 'Initialize-CmakeEnvironment.ps1'
if (Test-Path $cmakeHelper) {
    . $cmakeHelper
    Initialize-CmakeEnvironment -Quiet | Out-Null
}

# Setup VS environment
$devShell = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
& $devShell -SkipAutomaticLocation -HostArch amd64 -Arch amd64

# Clear polluted env vars
@('CL', '_CL_', 'LINK', '_LINK_') | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

# Add Windows SDK to PATH
$rcDir = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$env:PATH = "$rcDir;$env:PATH"

# Get compiler paths
$clPath = (Get-Command cl.exe).Source
$ninjaPath = "$env:USERPROFILE\.local\bin\ninja.exe"

# Source and build paths
$srcDir = "T:/RustCache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/llama-cpp-sys-2-0.1.132/llama.cpp"
$buildDir = "T:/RustCache/cargo-target/release/build/llama-cpp-sys-2-e20edaae6dc5d795/out/build"

# Clean build dir
Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

Write-Host "=== Running CMake Configure ===" -ForegroundColor Cyan
Write-Host "Source: $srcDir"
Write-Host "Build: $buildDir"
Write-Host "cl.exe: $clPath"
Write-Host "ninja: $ninjaPath"
Write-Host ""

# vcpkg toolchain for finding CURL
$vcpkgRoot = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\vcpkg"
$vcpkgToolchain = Join-Path $vcpkgRoot "scripts\buildsystems\vcpkg.cmake"

Write-Host "vcpkg toolchain: $vcpkgToolchain"

# Run cmake configure directly
& cmake -S $srcDir -B $buildDir `
    -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain" `
    "-DCMAKE_MAKE_PROGRAM=$ninjaPath" `
    "-DCMAKE_C_COMPILER=$clPath" `
    "-DCMAKE_CXX_COMPILER=$clPath" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DGGML_CUDA=ON" `
    "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" `
    "-DLLAMA_CURL=OFF" `
    "-DLLAMA_BUILD_TESTS=OFF" `
    "-DLLAMA_BUILD_EXAMPLES=OFF" `
    "-DLLAMA_BUILD_TOOLS=OFF"

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "=== CMake Configure SUCCESS ===" -ForegroundColor Green
    if (Test-Path "$buildDir/build.ninja") {
        Write-Host "build.ninja was created" -ForegroundColor Green
    } else {
        Write-Host "WARNING: build.ninja NOT created" -ForegroundColor Yellow
    }
} else {
    Write-Host "=== CMake Configure FAILED (exit code: $LASTEXITCODE) ===" -ForegroundColor Red
}
