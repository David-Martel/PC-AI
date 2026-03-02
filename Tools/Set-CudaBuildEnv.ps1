<#
.SYNOPSIS
    Sets CUDA build environment for Rust/Candle compilation.
.DESCRIPTION
    Prefers CUDA 13.1 by default for current pc-ai CUDA builds,
    auto-detects the newest MSVC x64 toolchain, and updates process-scoped
    environment variables for immediate build use.
#>
[CmdletBinding()]
param(
    [string]$CudaVersion = 'v13.1'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaVersion"
if (-not (Test-Path -LiteralPath "$cudaBase\bin\nvcc.exe")) {
    throw "CUDA $CudaVersion not found at $cudaBase"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe not found at $vswhere"
}

$installRoots = & $vswhere -all -products * -property installationPath
$msvcCandidates = @()
foreach ($installRoot in @($installRoots)) {
    if (-not $installRoot) { continue }
    $msvcRoot = Join-Path $installRoot 'VC\Tools\MSVC'
    if (-not (Test-Path -LiteralPath $msvcRoot)) { continue }

    $bins = Get-ChildItem -Path $msvcRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'bin\Hostx64\x64' } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ 'cl.exe') }
    if ($bins) { $msvcCandidates += $bins }
}

$msvcBin = $msvcCandidates |
    Sort-Object {
        if ($_ -match '\\MSVC\\(?<ver>[^\\]+)\\bin\\Hostx64\\x64$') {
            return [version]$Matches.ver
        }
        return [version]'0.0.0'
    } -Descending |
    Select-Object -First 1

if (-not $msvcBin) {
    throw 'Unable to locate MSVC cl.exe under Visual Studio installations.'
}

$env:CUDA_PATH = $cudaBase
$env:CUDA_HOME = $cudaBase
$env:NVCC_CCBIN = $msvcBin

foreach ($segment in @("$cudaBase\bin", "$cudaBase\nvvm\bin", $msvcBin)) {
    if ($env:PATH -notlike "*$segment*") {
        $env:PATH = "$segment;$env:PATH"
    }
}

Write-Host 'CUDA build environment set:' -ForegroundColor Green
Write-Host "  CUDA_PATH  = $env:CUDA_PATH"
Write-Host "  NVCC_CCBIN = $env:NVCC_CCBIN"
& nvcc --version | Select-Object -Last 2
