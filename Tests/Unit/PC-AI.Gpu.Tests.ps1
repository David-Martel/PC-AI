#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psm1'
    $script:PsdPath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'

    $script:TempDir = Join-Path $env:TEMP "pcai_gpu_tests_$(New-Guid)"
    New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null

    $script:NvidiaSmiInventoryCsv = @'
0, GPU-abc-123, NVIDIA RTX 2000 Ada Generation Laptop GPU, 582.41, 8.9, 8188, 2048, 62, 45
1, GPU-def-456, NVIDIA GeForce RTX 5060 Ti, 582.41, 12.0, 16311, 512, 55, 12
'@

    $script:NvidiaSmiUtilizationCsv = @'
0, NVIDIA RTX 2000 Ada Generation Laptop GPU, 45, 2048, 8188, 62, 45.5, 30
1, NVIDIA GeForce RTX 5060 Ti, 12, 512, 16311, 55, 120.0, 25
'@

    $script:CudaRoot = Join-Path $script:TempDir 'CUDA\v13.2'
    New-Item -ItemType Directory -Path $script:CudaRoot -Force | Out-Null
    '{"cuda":{"name":"CUDA SDK","version":"13.2.0"}}' | Set-Content -Path (Join-Path $script:CudaRoot 'version.json') -Encoding UTF8

    $script:CudnnRoot = Join-Path $script:TempDir 'CUDNN\v9.8'
    New-Item -ItemType Directory -Path (Join-Path $script:CudnnRoot 'include') -Force | Out-Null
    @'
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 8
#define CUDNN_PATCHLEVEL 0
'@ | Set-Content -Path (Join-Path $script:CudnnRoot 'include\cudnn_version.h') -Encoding UTF8

    $script:TensorRtRoot = Join-Path $script:TempDir 'TensorRT'
    New-Item -ItemType Directory -Path (Join-Path $script:TensorRtRoot 'include') -Force | Out-Null
    @'
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 9
#define NV_TENSORRT_PATCH 0
'@ | Set-Content -Path (Join-Path $script:TensorRtRoot 'include\NvInferVersion.h') -Encoding UTF8

    $script:NsightBaseDir = Join-Path $script:TempDir 'NVIDIA Corporation'
    New-Item -ItemType Directory -Path (Join-Path $script:NsightBaseDir 'Nsight Systems 2025.1.2') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $script:NsightBaseDir 'Nsight Compute 2025.1.0') -Force | Out-Null

    $script:SoftwareRegistryPath = Join-Path $script:TempDir 'nvidia-software-registry.json'
    @'
{
  "version": "1.0.0-test",
  "lastUpdated": "2026-03-18T00:00:00Z",
  "categories": {
    "driver": { "displayName": "Drivers" },
    "runtime": { "displayName": "CUDA Runtime" },
    "library": { "displayName": "Libraries" },
    "tools": { "displayName": "Developer Tools" }
  },
  "components": [
    {
      "id": "gpu-driver",
      "name": "NVIDIA Display Driver",
      "category": "driver",
      "latestVersion": "582.41"
    },
    {
      "id": "cuda-toolkit",
      "name": "CUDA Toolkit",
      "category": "runtime",
      "latestVersion": "13.2.0"
    },
    {
      "id": "cudnn",
      "name": "cuDNN",
      "category": "library",
      "latestVersion": "9.8.0"
    },
    {
      "id": "tensorrt",
      "name": "TensorRT",
      "category": "library",
      "latestVersion": "10.9.0"
    },
    {
      "id": "nsight-systems",
      "name": "Nsight Systems",
      "category": "tools",
      "latestVersion": null
    }
  ]
}
'@ | Set-Content -Path $script:SoftwareRegistryPath -Encoding UTF8

    Import-Module $script:ModulePath -Force -ErrorAction Stop
}

AfterAll {
    Remove-Module 'PC-AI.Gpu' -Force -ErrorAction SilentlyContinue
    if (Test-Path $script:TempDir) {
        Remove-Item -Path $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Describe 'PC-AI.Gpu Module' -Tag 'Unit', 'Gpu', 'Fast' {
    Context 'Module Loading' {
        It 'imports without error' {
            { Import-Module $script:ModulePath -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It 'exports the supported public command surface' {
            $expected = @(
                'Get-NvidiaGpuInventory',
                'Get-NvidiaSoftwareRegistry',
                'Get-NvidiaSoftwareStatus',
                'Get-NvidiaGpuUtilization',
                'Get-NvidiaCompatibilityMatrix',
                'Initialize-NvidiaEnvironment',
                'Install-NvidiaSoftware',
                'Update-NvidiaSoftwareRegistry'
            )

            foreach ($name in $expected) {
                Get-Command -Module 'PC-AI.Gpu' -Name $name -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
            }
        }

        It 'keeps private helpers unexported' {
            $exported = (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys
            $exported | Should -Not -Contain 'Get-CudaVersionFromPath'
            $exported | Should -Not -Contain 'Get-CudnnVersionFromHeader'
            $exported | Should -Not -Contain 'Get-TensorRtVersionFromHeader'
            $exported | Should -Not -Contain 'Get-NsightVersions'
        }
    }

    Context 'Get-NvidiaGpuInventory' {
        BeforeAll {
            Mock -CommandName Resolve-PcaiCoreLibDll -ModuleName 'PC-AI.Gpu' -MockWith { return $null }
            Mock -CommandName Get-Command -ModuleName 'PC-AI.Gpu' -ParameterFilter { $Name -eq 'nvidia-smi.exe' } -MockWith {
                [pscustomobject]@{ Source = 'nvidia-smi.exe' }
            }
            Mock -CommandName 'nvidia-smi.exe' -ModuleName 'PC-AI.Gpu' -MockWith {
                $script:NvidiaSmiInventoryCsv
            }
        }

        It 'parses both GPUs from nvidia-smi output' {
            $result = @(Get-NvidiaGpuInventory)
            $result.Count | Should -Be 2
            $result[0].Index | Should -Be 0
            $result[0].UUID | Should -Be 'GPU-abc-123'
            $result[0].MemoryTotalMB | Should -Be 8188
            $result[1].Name | Should -Match 'RTX 5060 Ti'
            $result[1].ComputeCapability | Should -Be '12.0'
        }
    }

    Context 'Get-NvidiaSoftwareRegistry' {
        It 'loads the registry and filters by component id' {
            $registry = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cudnn'
            $registry.Version | Should -Be '1.0.0-test'
            $registry.Components.Count | Should -Be 1
            $registry.Components[0].id | Should -Be 'cudnn'
        }

        It 'filters by category' {
            $registry = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -Category 'library'
            $registry.Components.Count | Should -Be 2
        }
    }

    Context 'Get-NvidiaSoftwareStatus' {
        BeforeAll {
            Mock -CommandName Resolve-NvidiaInstallPath -ModuleName 'PC-AI.Gpu' -MockWith {
                @{
                    CUDA = $script:CudaRoot
                    cuDNN = $script:CudnnRoot
                    TensorRT = $script:TensorRtRoot
                }
            }
            Mock -CommandName Get-NsightVersions -ModuleName 'PC-AI.Gpu' -MockWith {
                @([pscustomobject]@{
                    Name = 'Nsight Systems 2025.1.2'
                    Product = 'NsightSystems'
                    Version = '2025.1.2'
                    Path = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.2'
                })
            }
            Mock -CommandName Get-NvidiaDriverVersion -ModuleName 'PC-AI.Gpu' -MockWith { '582.41' }
            Mock -CommandName Get-CudaVersionFromPath -ModuleName 'PC-AI.Gpu' -MockWith { '13.2.0' }
            Mock -CommandName Get-CudnnVersionFromHeader -ModuleName 'PC-AI.Gpu' -MockWith { '9.8.0' }
            Mock -CommandName Get-TensorRtVersionFromHeader -ModuleName 'PC-AI.Gpu' -MockWith { '10.9.0' }
            Mock -CommandName Get-ChildItem -ModuleName 'PC-AI.Gpu' -ParameterFilter { $Path -eq 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' } -MockWith {
                @(
                    [pscustomobject]@{ Name = 'v13.1' },
                    [pscustomobject]@{ Name = 'v13.2' }
                )
            }
        }

        It 'reports Current for the active CUDA install even with side-by-side versions present' {
            $result = @(Get-NvidiaSoftwareStatus -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit')
            $result[0].Status | Should -Be 'Current'
            $result[0].SideBySideCount | Should -Be 2
        }

        It 'reports Outdated when the detected CUDA version is older than the registry' {
            Mock -CommandName Get-CudaVersionFromPath -ModuleName 'PC-AI.Gpu' -MockWith { '12.6.0' }
            $result = @(Get-NvidiaSoftwareStatus -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit')
            $result[0].Status | Should -Be 'Outdated'
            $result[0].InstalledVersion | Should -Be '12.6.0'
        }

        It 'reports Unknown when a component has no latestVersion' {
            $result = @(Get-NvidiaSoftwareStatus -RegistryPath $script:SoftwareRegistryPath -ComponentId 'nsight-systems')
            $result[0].Status | Should -Be 'Unknown'
        }
    }

    Context 'Get-NvidiaGpuUtilization' {
        BeforeAll {
            Mock -CommandName Get-Command -ModuleName 'PC-AI.Gpu' -ParameterFilter { $Name -eq 'nvidia-smi.exe' } -MockWith {
                [pscustomobject]@{ Source = 'nvidia-smi.exe' }
            }
            Mock -CommandName 'nvidia-smi.exe' -ModuleName 'PC-AI.Gpu' -MockWith {
                $script:NvidiaSmiUtilizationCsv
            }
        }

        It 'parses utilization metrics with the current property names' {
            $result = @(Get-NvidiaGpuUtilization)
            $result.Count | Should -Be 2
            $result[0].GpuUtilization | Should -Be 45
            $result[0].MemoryUsedMB | Should -Be 2048
            $result[0].MemoryTotalMB | Should -Be 8188
            $result[0].Temperature | Should -Be 62
            [decimal]$result[0].PowerDraw | Should -Be 45.5
        }
    }

    Context 'Private helpers' {
        It 'parses the CUDA version from a toolkit root' {
            $cudaPath = $script:CudaRoot
            InModuleScope 'PC-AI.Gpu' -Parameters @{ CudaPath = $cudaPath } {
                param($CudaPath)
                Get-CudaVersionFromPath -CudaPath $CudaPath | Should -Be '13.2.0'
            }
        }

        It 'parses the cuDNN version from an install root' {
            $cudnnPath = $script:CudnnRoot
            InModuleScope 'PC-AI.Gpu' -Parameters @{ CudnnPath = $cudnnPath } {
                param($CudnnPath)
                Get-CudnnVersionFromHeader -CudnnPath $CudnnPath | Should -Be '9.8.0'
            }
        }

        It 'parses the TensorRT version from an install root' {
            $tensorRtPath = $script:TensorRtRoot
            InModuleScope 'PC-AI.Gpu' -Parameters @{ TensorRtPath = $tensorRtPath } {
                param($TensorRtPath)
                Get-TensorRtVersionFromHeader -TensorRtPath $TensorRtPath | Should -Be '10.9.0'
            }
        }

        It 'detects Nsight installs from a supplied search path' {
            $nsightBaseDir = $script:NsightBaseDir
            InModuleScope 'PC-AI.Gpu' -Parameters @{ NsightBaseDir = $nsightBaseDir } {
                param($NsightBaseDir)
                $entries = @(Get-NsightVersions -SearchPath $NsightBaseDir)
                $entries.Count | Should -Be 2
                ($entries | Where-Object Product -eq 'NsightSystems').Version | Should -Be '2025.1.2'
                ($entries | Where-Object Product -eq 'NsightCompute').Version | Should -Be '2025.1.0'
            }
        }
    }
}
