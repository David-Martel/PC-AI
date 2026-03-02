<#
.SYNOPSIS
    Sets CUDA environment variables for Rust/Candle compilation.
.DESCRIPTION
    Configures CUDA_PATH plus PATH/include/lib entries. Defaults to process
    scope for safety; use -Scope Machine for persistent system updates.
#>
[CmdletBinding()]
param(
    [string]$CudaVersion = 'v13.1',
    [ValidateSet('Process', 'Machine')]
    [string]$Scope = 'Process'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaVersion"
if (-not (Test-Path -LiteralPath $cudaBase)) {
    throw "CUDA $CudaVersion not found at $cudaBase"
}

$setScope = if ($Scope -eq 'Machine') { 'Machine' } else { 'Process' }
$currentCudaPath = [Environment]::GetEnvironmentVariable('CUDA_PATH', $setScope)
if ($currentCudaPath -ne $cudaBase) {
    [Environment]::SetEnvironmentVariable('CUDA_PATH', $cudaBase, $setScope)
}

$pathValue = [Environment]::GetEnvironmentVariable('Path', $setScope)
if (-not $pathValue) { $pathValue = '' }

$requiredPathSegments = @(
    "$cudaBase\bin",
    "$cudaBase\nvvm\bin",
    "$cudaBase\libnvvp"
)

foreach ($segment in $requiredPathSegments) {
    if ((Test-Path -LiteralPath $segment) -and ($pathValue -notlike "*$segment*")) {
        $pathValue = if ([string]::IsNullOrWhiteSpace($pathValue)) { $segment } else { "$segment;$pathValue" }
    }
}

[Environment]::SetEnvironmentVariable('Path', $pathValue, $setScope)

# Mirror into process env for immediate use even when applying machine scope.
$env:CUDA_PATH = $cudaBase
$env:CUDA_HOME = $cudaBase
$env:PATH = $pathValue

if ($env:INCLUDE -notlike "*$cudaBase\\include*") {
    $env:INCLUDE = if ($env:INCLUDE) { "$cudaBase\include;$env:INCLUDE" } else { "$cudaBase\include" }
}
if ($env:LIB -notlike "*$cudaBase\\lib\\x64*") {
    $env:LIB = if ($env:LIB) { "$cudaBase\lib\x64;$env:LIB" } else { "$cudaBase\lib\x64" }
}

Write-Host "CUDA environment configured ($Scope scope):" -ForegroundColor Green
Write-Host "  CUDA_PATH: $env:CUDA_PATH"
Write-Host "  nvcc.exe exists: $(Test-Path -LiteralPath "$cudaBase\bin\nvcc.exe")"
Write-Host "  cicc.exe exists: $(Test-Path -LiteralPath "$cudaBase\nvvm\bin\cicc.exe")"
