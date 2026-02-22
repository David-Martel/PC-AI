<#
.SYNOPSIS
    Sets CUDA 12.9 environment for Rust compilation (workaround for CUDA 13.1 bindgen_cuda panic).
.DESCRIPTION
    CUDA 13.1 causes bindgen_cuda::Builder::build_ptx to panic due to nvcc
    failing to preprocess host compiler properties with MSVC 19.44.
    CUDA 12.9 works correctly. This script sets the environment variables.
#>
param(
    [string]$CudaVersion = "v12.9"
)

$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaVersion"
$msvcBin = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

if (-not (Test-Path "$cudaBase\bin\nvcc.exe")) {
    Write-Error "CUDA $CudaVersion not found at $cudaBase"
    return
}

$env:CUDA_PATH = $cudaBase
$env:NVCC_CCBIN = $msvcBin
$env:PATH = "$cudaBase\bin;$msvcBin;$env:PATH"

Write-Host "CUDA build environment set:" -ForegroundColor Green
Write-Host "  CUDA_PATH  = $env:CUDA_PATH"
Write-Host "  NVCC_CCBIN = $env:NVCC_CCBIN"
& nvcc --version | Select-Object -Last 2
