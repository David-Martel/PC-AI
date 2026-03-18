#Requires -Version 7.0

<#
.SYNOPSIS
    PC-AI NVIDIA GPU Management Module

.DESCRIPTION
    Provides cmdlets for inventorying installed NVIDIA GPUs, querying the curated
    NVIDIA software registry, comparing installed component versions against known-latest,
    monitoring real-time GPU utilization, and preparing the CUDA/cuDNN/TensorRT
    environment for inference workloads.

    Exported functions:
      Get-NvidiaGpuInventory        - Enumerate all NVIDIA GPUs via nvidia-smi CSV output,
                                      with CIM fallback when nvidia-smi is unavailable.
      Get-NvidiaSoftwareRegistry    - Load and return nvidia-software-registry.json as a
                                      structured object. Accepts -ComponentId and -Category
                                      filters.
      Get-NvidiaSoftwareStatus      - Compare installed component versions (CUDA, cuDNN,
                                      TensorRT, Nsight) against registry entries; returns
                                      per-component status table.
      Get-NvidiaGpuUtilization      - Real-time GPU utilization snapshot via nvidia-smi CSV
                                      query: utilization, memory, temperature, power.
      Get-NvidiaCompatibilityMatrix - (Phase 2) Build compatibility matrix across CUDA,
                                      cuDNN, TensorRT, and driver versions.
      Initialize-NvidiaEnvironment  - (Phase 2) Configure CUDA_PATH, CUDNN_PATH,
                                      TENSORRT_PATH, and related environment variables.
      Install-NvidiaSoftware        - (Phase 3) Silent/interactive installer for NVIDIA
                                      components with rollback support.
      Update-NvidiaSoftwareRegistry - (Phase 3) Refresh nvidia-software-registry.json from
                                      a remote source or update individual component entries.

    Dependencies:
      - PowerShell 7.0 or later
      - Windows 10/11
      - nvidia-smi.exe on PATH or standard NVIDIA driver install location
      - nvidia-software-registry.json at Config\nvidia-software-registry.json
        relative to the PC_AI root
#>

$script:ModuleRoot = $PSScriptRoot

$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Get-NvidiaGpuInventory',
    'Get-NvidiaSoftwareRegistry',
    'Get-NvidiaSoftwareStatus',
    'Get-NvidiaGpuUtilization',
    'Get-NvidiaCompatibilityMatrix',
    'Initialize-NvidiaEnvironment',
    'Install-NvidiaSoftware',
    'Update-NvidiaSoftwareRegistry'
)
