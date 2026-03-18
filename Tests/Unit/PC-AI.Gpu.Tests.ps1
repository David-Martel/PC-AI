#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Unit tests for the PC-AI.Gpu PowerShell module.

.DESCRIPTION
    Tests the PC-AI.Gpu module (Modules/PC-AI.Gpu/PC-AI.Gpu.psm1) without
    real hardware.  All nvidia-smi invocations, CIM queries, and file-system
    probes are either mocked or backed by temp files created in BeforeAll blocks.

    Test categories covered:
    - Module loading: import, export surface, private function hiding
    - Get-NvidiaGpuInventory: nvidia-smi CSV parsing (multi-GPU), CIM fallback,
      empty output handling
    - Get-NvidiaSoftwareRegistry: loading from file, -ComponentId filter,
      -Category filter, missing file error
    - Get-NvidiaSoftwareStatus: status classification (Current, Outdated,
      NotInstalled, MultipleVersions)
    - Get-NvidiaGpuUtilization: nvidia-smi utilization CSV parsing
    - Private: Get-CudaVersionFromPath, Get-CudnnVersionFromHeader,
      Get-TensorRtVersionFromHeader, Get-NsightVersions

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Unit\PC-AI.Gpu.Tests.ps1 -Tag Unit,Gpu
#>

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ModulePath  = Join-Path $ProjectRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psm1'
    $script:PsdPath     = Join-Path $ProjectRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'

    # ---- Temp directory for all file-based fixtures ----
    $script:TempDir = Join-Path $env:TEMP "pcai_gpu_tests_$(New-Guid)"
    New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null

    # ---- nvidia-smi GPU inventory CSV (two-GPU setup: RTX 2000 Ada + RTX 5060 Ti) ----
    $script:NvidiaSmiInventoryCsv = @'
0, GPU-abc-123, NVIDIA RTX 2000 Ada Generation Laptop GPU, 8188, 8.9, 582.41
1, GPU-def-456, NVIDIA GeForce RTX 5060 Ti, 16311, 12.0, 582.41
'@

    # ---- nvidia-smi utilization CSV ----
    $script:NvidiaSmiUtilizationCsv = @'
0, NVIDIA RTX 2000 Ada Generation Laptop GPU, 45, 2048, 8188, 62, 45.5, 30
1, NVIDIA GeForce RTX 5060 Ti, 12, 512, 16311, 55, 120.0, 25
'@

    # ---- CUDA version.json ----
    $script:CudaVersionJson = '{"cuda": {"name": "CUDA SDK", "version": "13.2.0"}}'
    $script:CudaVersionJsonPath = Join-Path $script:TempDir 'cuda_version.json'
    $script:CudaVersionJson | Set-Content -Path $script:CudaVersionJsonPath -Encoding UTF8

    # ---- cudnn_version.h ----
    $script:CudnnHeader = @'
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 8
#define CUDNN_PATCHLEVEL 0
'@
    $script:CudnnHeaderPath = Join-Path $script:TempDir 'cudnn_version.h'
    $script:CudnnHeader | Set-Content -Path $script:CudnnHeaderPath -Encoding UTF8

    # ---- NvInferVersion.h ----
    $script:TensorRtHeader = @'
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 9
#define NV_TENSORRT_PATCH 0
'@
    $script:TensorRtHeaderPath = Join-Path $script:TempDir 'NvInferVersion.h'
    $script:TensorRtHeader | Set-Content -Path $script:TensorRtHeaderPath -Encoding UTF8

    # ---- Software registry JSON fixture ----
    $script:MockSoftwareRegistryJson = @'
{
  "version": "1.0.0-test",
  "lastUpdated": "2026-03-14T00:00:00Z",
  "categories": {
    "runtime": { "displayName": "CUDA Runtime", "icon": "cuda" },
    "library":  { "displayName": "Deep Learning Libraries", "icon": "library" },
    "tools":    { "displayName": "Developer Tools", "icon": "tools" }
  },
  "components": [
    {
      "id":            "cuda-toolkit",
      "name":          "CUDA Toolkit",
      "category":      "runtime",
      "latestVersion": "13.2.0",
      "versionComparable": true,
      "notes":         "Primary CUDA runtime"
    },
    {
      "id":            "cudnn",
      "name":          "cuDNN",
      "category":      "library",
      "latestVersion": "9.8.0",
      "versionComparable": true,
      "notes":         "Deep learning primitives"
    },
    {
      "id":            "tensorrt",
      "name":          "TensorRT",
      "category":      "library",
      "latestVersion": "10.9.0",
      "versionComparable": true,
      "notes":         "Inference optimizer"
    },
    {
      "id":            "nsight-systems",
      "name":          "Nsight Systems",
      "category":      "tools",
      "latestVersion": null,
      "versionComparable": false,
      "notes":         "Profiling tool, no forced update"
    }
  ]
}
'@
    $script:SoftwareRegistryPath = Join-Path $script:TempDir 'nvidia-software-registry.json'
    $script:MockSoftwareRegistryJson | Set-Content -Path $script:SoftwareRegistryPath -Encoding UTF8

    # ---- Import the module under test (if it exists) ----
    if (Test-Path $script:ModulePath) {
        Import-Module $script:ModulePath -Force -ErrorAction Stop
    }
}

AfterAll {
    Remove-Module 'PC-AI.Gpu' -Force -ErrorAction SilentlyContinue
    if ($script:TempDir -and (Test-Path $script:TempDir)) {
        Remove-Item -Path $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ==============================================================================
# Module Loading
# ==============================================================================

Describe 'PC-AI.Gpu Module' -Tag 'Unit', 'Gpu', 'Fast' {

    Context 'Module Loading' {

        It 'Module file exists at expected path' {
            Test-Path $script:ModulePath | Should -BeTrue
        }

        It 'Module manifest (.psd1) exists' {
            Test-Path $script:PsdPath | Should -BeTrue
        }

        It 'Module imports without error' {
            { Import-Module $script:ModulePath -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It 'Module exports Get-NvidiaGpuInventory' {
            Get-Command -Module 'PC-AI.Gpu' -Name 'Get-NvidiaGpuInventory' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'Module exports Get-NvidiaSoftwareRegistry' {
            Get-Command -Module 'PC-AI.Gpu' -Name 'Get-NvidiaSoftwareRegistry' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'Module exports Get-NvidiaSoftwareStatus' {
            Get-Command -Module 'PC-AI.Gpu' -Name 'Get-NvidiaSoftwareStatus' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'Module exports Get-NvidiaGpuUtilization' {
            Get-Command -Module 'PC-AI.Gpu' -Name 'Get-NvidiaGpuUtilization' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'Does not export Get-CudaVersionFromPath (private)' {
            (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys |
                Should -Not -Contain 'Get-CudaVersionFromPath'
        }

        It 'Does not export Get-CudnnVersionFromHeader (private)' {
            (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys |
                Should -Not -Contain 'Get-CudnnVersionFromHeader'
        }

        It 'Does not export Get-TensorRtVersionFromHeader (private)' {
            (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys |
                Should -Not -Contain 'Get-TensorRtVersionFromHeader'
        }

        It 'Does not export Get-NsightVersions (private)' {
            (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys |
                Should -Not -Contain 'Get-NsightVersions'
        }
    }

    # ===========================================================================
    # Get-NvidiaGpuInventory
    # ===========================================================================

    Context 'Get-NvidiaGpuInventory' {

        Context 'Parsing nvidia-smi CSV output (two GPUs)' {

            BeforeAll {
                Mock -CommandName 'nvidia-smi' -ModuleName 'PC-AI.Gpu' -MockWith {
                    $script:NvidiaSmiInventoryCsv
                }
            }

            It 'Returns two GPU objects when nvidia-smi reports two GPUs' {
                $result = @(Get-NvidiaGpuInventory)
                $result.Count | Should -Be 2
            }

            It 'First GPU index is 0 (RTX 2000 Ada)' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].Index | Should -Be 0
            }

            It 'First GPU name matches RTX 2000 Ada' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].Name | Should -Match 'RTX 2000 Ada'
            }

            It 'First GPU VRAM is 8188 MiB' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].VramMiB | Should -Be 8188
            }

            It 'First GPU compute capability is 8.9' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].ComputeCapability | Should -Be '8.9'
            }

            It 'First GPU driver version is 582.41' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].DriverVersion | Should -Be '582.41'
            }

            It 'Second GPU index is 1 (RTX 5060 Ti)' {
                $result = @(Get-NvidiaGpuInventory)
                $result[1].Index | Should -Be 1
            }

            It 'Second GPU name matches RTX 5060 Ti' {
                $result = @(Get-NvidiaGpuInventory)
                $result[1].Name | Should -Match 'RTX 5060 Ti'
            }

            It 'Second GPU VRAM is 16311 MiB' {
                $result = @(Get-NvidiaGpuInventory)
                $result[1].VramMiB | Should -Be 16311
            }

            It 'Second GPU compute capability is 12.0 (Blackwell)' {
                $result = @(Get-NvidiaGpuInventory)
                $result[1].ComputeCapability | Should -Be '12.0'
            }

            It 'GPU UUID is parsed from nvidia-smi output' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].Uuid | Should -Not -BeNullOrEmpty
            }
        }

        Context 'CIM fallback when nvidia-smi is unavailable' {

            BeforeAll {
                Mock -CommandName 'nvidia-smi' -ModuleName 'PC-AI.Gpu' -MockWith {
                    throw 'nvidia-smi not found'
                }

                Mock -CommandName 'Get-CimInstance' -ModuleName 'PC-AI.Gpu' -MockWith {
                    @(
                        [PSCustomObject]@{
                            Name          = 'NVIDIA GeForce RTX 5060 Ti'
                            DriverVersion = '32.0.15.8241'
                            AdapterRAM    = 17103675392
                            PNPDeviceID   = 'PCI\VEN_10DE&DEV_2C11'
                            Status        = 'OK'
                        }
                    )
                }
            }

            It 'Returns at least one GPU via CIM fallback' {
                $result = @(Get-NvidiaGpuInventory)
                $result.Count | Should -BeGreaterOrEqual 1
            }

            It 'CIM fallback result has a Name property' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].Name | Should -Not -BeNullOrEmpty
            }

            It 'CIM fallback result has a Source of CIM' {
                $result = @(Get-NvidiaGpuInventory)
                $result[0].Source | Should -Be 'CIM'
            }
        }

        Context 'Empty nvidia-smi output' {

            BeforeAll {
                Mock -CommandName 'nvidia-smi' -ModuleName 'PC-AI.Gpu' -MockWith {
                    return ''
                }

                Mock -CommandName 'Get-CimInstance' -ModuleName 'PC-AI.Gpu' -MockWith {
                    return @()
                }
            }

            It 'Returns an empty collection when both sources report nothing' {
                $result = @(Get-NvidiaGpuInventory)
                $result.Count | Should -Be 0
            }
        }
    }

    # ===========================================================================
    # Get-NvidiaSoftwareRegistry
    # ===========================================================================

    Context 'Get-NvidiaSoftwareRegistry' {

        Context 'Loading from explicit path' {

            It 'Returns a registry object with version and components' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath
                $reg | Should -Not -BeNullOrEmpty
                $reg.Version | Should -Be '1.0.0-test'
                $reg.Components.Count | Should -Be 4
            }

            It 'Includes categories' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath
                $reg.Categories.runtime.displayName | Should -Be 'CUDA Runtime'
            }

            It 'Contains cuda-toolkit component' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath
                $match = $reg.Components | Where-Object { $_.id -eq 'cuda-toolkit' }
                $match | Should -Not -BeNullOrEmpty
            }
        }

        Context 'Filtering by -ComponentId' {

            It 'Returns only the matching component' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cudnn'
                $reg.Components.Count | Should -Be 1
                $reg.Components[0].id | Should -Be 'cudnn'
            }

            It 'Returns empty component list for non-existent id' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'does-not-exist'
                $reg.Components.Count | Should -Be 0
            }
        }

        Context 'Filtering by -Category' {

            It 'Returns only library-category components' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -Category 'library'
                $reg.Components.Count | Should -Be 2
                $reg.Components | ForEach-Object { $_.category | Should -Be 'library' }
            }

            It 'Returns only tools-category components' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -Category 'tools'
                $reg.Components.Count | Should -Be 1
                $reg.Components[0].id | Should -Be 'nsight-systems'
            }

            It 'Returns only runtime-category components' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -Category 'runtime'
                $reg.Components.Count | Should -Be 1
                $reg.Components[0].id | Should -Be 'cuda-toolkit'
            }
        }

        Context 'Error handling' {

            It 'Returns null or empty for a missing registry file' {
                $result = Get-NvidiaSoftwareRegistry -RegistryPath 'C:\nonexistent\nvidia-registry.json' -ErrorAction SilentlyContinue
                $result | Should -BeNullOrEmpty
            }
        }
    }

    # ===========================================================================
    # Get-NvidiaSoftwareStatus
    # ===========================================================================

    Context 'Get-NvidiaSoftwareStatus' {

        Context 'Current status when installed version matches latest' {

            BeforeAll {
                Mock -CommandName 'Get-CudaVersionFromPath' -ModuleName 'PC-AI.Gpu' -MockWith {
                    '13.2.0'
                }
            }

            It 'Reports Current for cuda-toolkit when installed equals latest' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'cuda-toolkit' }
                $match | Should -Not -BeNullOrEmpty
                $match.Status | Should -Be 'Current'
            }
        }

        Context 'Outdated status when installed version is older' {

            BeforeAll {
                Mock -CommandName 'Get-CudaVersionFromPath' -ModuleName 'PC-AI.Gpu' -MockWith {
                    '12.6.0'
                }
            }

            It 'Reports Outdated for cuda-toolkit when installed is older than latest' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'cuda-toolkit' }
                $match.Status | Should -Be 'Outdated'
            }

            It 'Outdated result includes InstalledVersion' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'cuda-toolkit' }
                $match.InstalledVersion | Should -Be '12.6.0'
            }

            It 'Outdated result includes LatestVersion' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'cuda-toolkit' }
                $match.LatestVersion | Should -Be '13.2.0'
            }
        }

        Context 'NotInstalled status when detection returns empty' {

            BeforeAll {
                Mock -CommandName 'Get-CudaVersionFromPath' -ModuleName 'PC-AI.Gpu' -MockWith {
                    return $null
                }
            }

            It 'Reports NotInstalled for cuda-toolkit when not detected' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'cuda-toolkit'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'cuda-toolkit' }
                $match.Status | Should -Be 'NotInstalled'
            }
        }

        Context 'NoUpdate status for components with null latestVersion' {

            It 'Reports NoUpdate for nsight-systems (latestVersion is null)' {
                $reg = Get-NvidiaSoftwareRegistry -RegistryPath $script:SoftwareRegistryPath -ComponentId 'nsight-systems'
                $result = @(Get-NvidiaSoftwareStatus -Registry $reg)
                $match = $result | Where-Object { $_.ComponentId -eq 'nsight-systems' }
                $match.Status | Should -Be 'NoUpdate'
            }
        }
    }

    # ===========================================================================
    # Get-NvidiaGpuUtilization
    # ===========================================================================

    Context 'Get-NvidiaGpuUtilization' {

        Context 'Parsing utilization CSV from nvidia-smi' {

            BeforeAll {
                Mock -CommandName 'nvidia-smi' -ModuleName 'PC-AI.Gpu' -MockWith {
                    $script:NvidiaSmiUtilizationCsv
                }
            }

            It 'Returns two entries when two GPUs are active' {
                $result = @(Get-NvidiaGpuUtilization)
                $result.Count | Should -Be 2
            }

            It 'First entry index is 0' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[0].Index | Should -Be 0
            }

            It 'First entry GpuUtilPct is 45' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[0].GpuUtilPct | Should -Be 45
            }

            It 'First entry MemUsedMiB is 2048' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[0].MemUsedMiB | Should -Be 2048
            }

            It 'First entry MemTotalMiB is 8188' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[0].MemTotalMiB | Should -Be 8188
            }

            It 'First entry TempC is 62' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[0].TempC | Should -Be 62
            }

            It 'First entry PowerWatts is 45.5' {
                $result = @(Get-NvidiaGpuUtilization)
                [Math]::Abs($result[0].PowerWatts - 45.5) | Should -BeLessThan 0.01
            }

            It 'Second entry index is 1 (RTX 5060 Ti)' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[1].Index | Should -Be 1
            }

            It 'Second entry GpuUtilPct is 12' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[1].GpuUtilPct | Should -Be 12
            }

            It 'Second entry MemTotalMiB is 16311' {
                $result = @(Get-NvidiaGpuUtilization)
                $result[1].MemTotalMiB | Should -Be 16311
            }
        }

        Context 'nvidia-smi unavailable for utilization query' {

            BeforeAll {
                Mock -CommandName 'nvidia-smi' -ModuleName 'PC-AI.Gpu' -MockWith {
                    throw 'nvidia-smi not found'
                }
            }

            It 'Returns empty collection when nvidia-smi is not available' {
                $result = @(Get-NvidiaGpuUtilization -ErrorAction SilentlyContinue)
                $result.Count | Should -Be 0
            }
        }
    }

    # ===========================================================================
    # Private: Get-CudaVersionFromPath
    # ===========================================================================

    Context 'Private: Get-CudaVersionFromPath' {

        It 'Parses version 13.2.0 from version.json' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudaVersionFromPath -JsonPath $script:CudaVersionJsonPath
                $result | Should -Be '13.2.0'
            }
        }

        It 'Returns null when version.json does not exist' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudaVersionFromPath -JsonPath 'C:\nonexistent\version.json'
                $result | Should -BeNullOrEmpty
            }
        }

        It 'Returns null when JSON is missing the cuda key' {
            InModuleScope 'PC-AI.Gpu' {
                $emptyJsonPath = Join-Path $script:TempDir 'cuda_empty.json'
                '{"other": {"version": "1.0.0"}}' | Set-Content -Path $emptyJsonPath -Encoding UTF8
                $result = Get-CudaVersionFromPath -JsonPath $emptyJsonPath
                $result | Should -BeNullOrEmpty
            }
        }
    }

    # ===========================================================================
    # Private: Get-CudnnVersionFromHeader
    # ===========================================================================

    Context 'Private: Get-CudnnVersionFromHeader' {

        It 'Parses CUDNN_MAJOR as 9 from cudnn_version.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudnnVersionFromHeader -HeaderPath $script:CudnnHeaderPath
                $result.Major | Should -Be 9
            }
        }

        It 'Parses CUDNN_MINOR as 8 from cudnn_version.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudnnVersionFromHeader -HeaderPath $script:CudnnHeaderPath
                $result.Minor | Should -Be 8
            }
        }

        It 'Parses CUDNN_PATCHLEVEL as 0 from cudnn_version.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudnnVersionFromHeader -HeaderPath $script:CudnnHeaderPath
                $result.Patch | Should -Be 0
            }
        }

        It 'Returns version string 9.8.0' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudnnVersionFromHeader -HeaderPath $script:CudnnHeaderPath
                $result.Version | Should -Be '9.8.0'
            }
        }

        It 'Returns null when header file does not exist' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-CudnnVersionFromHeader -HeaderPath 'C:\nonexistent\cudnn_version.h'
                $result | Should -BeNullOrEmpty
            }
        }
    }

    # ===========================================================================
    # Private: Get-TensorRtVersionFromHeader
    # ===========================================================================

    Context 'Private: Get-TensorRtVersionFromHeader' {

        It 'Parses NV_TENSORRT_MAJOR as 10 from NvInferVersion.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-TensorRtVersionFromHeader -HeaderPath $script:TensorRtHeaderPath
                $result.Major | Should -Be 10
            }
        }

        It 'Parses NV_TENSORRT_MINOR as 9 from NvInferVersion.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-TensorRtVersionFromHeader -HeaderPath $script:TensorRtHeaderPath
                $result.Minor | Should -Be 9
            }
        }

        It 'Parses NV_TENSORRT_PATCH as 0 from NvInferVersion.h' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-TensorRtVersionFromHeader -HeaderPath $script:TensorRtHeaderPath
                $result.Patch | Should -Be 0
            }
        }

        It 'Returns version string 10.9.0' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-TensorRtVersionFromHeader -HeaderPath $script:TensorRtHeaderPath
                $result.Version | Should -Be '10.9.0'
            }
        }

        It 'Returns null when header file does not exist' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-TensorRtVersionFromHeader -HeaderPath 'C:\nonexistent\NvInferVersion.h'
                $result | Should -BeNullOrEmpty
            }
        }
    }

    # ===========================================================================
    # Private: Get-NsightVersions
    # ===========================================================================

    Context 'Private: Get-NsightVersions' {

        BeforeAll {
            # Create a fake Nsight Systems install directory structure
            $script:NsightBaseDir = Join-Path $script:TempDir 'NVIDIA Corporation'
            $script:NsightSystemsDir = Join-Path $script:NsightBaseDir 'Nsight Systems 2025.1.2'
            New-Item -ItemType Directory -Path $script:NsightSystemsDir -Force | Out-Null
            $script:NsightComputeDir = Join-Path $script:NsightBaseDir 'Nsight Compute 2025.1.0'
            New-Item -ItemType Directory -Path $script:NsightComputeDir -Force | Out-Null
        }

        It 'Detects Nsight Systems version 2025.1.2 from directory name' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-NsightVersions -SearchPath $script:NsightBaseDir
                $systems = $result | Where-Object { $_.Product -eq 'Nsight Systems' }
                $systems | Should -Not -BeNullOrEmpty
                $systems.Version | Should -Be '2025.1.2'
            }
        }

        It 'Detects Nsight Compute version 2025.1.0 from directory name' {
            InModuleScope 'PC-AI.Gpu' {
                $result = Get-NsightVersions -SearchPath $script:NsightBaseDir
                $compute = $result | Where-Object { $_.Product -eq 'Nsight Compute' }
                $compute | Should -Not -BeNullOrEmpty
                $compute.Version | Should -Be '2025.1.0'
            }
        }

        It 'Returns empty collection when search path does not exist' {
            InModuleScope 'PC-AI.Gpu' {
                $result = @(Get-NsightVersions -SearchPath 'C:\nonexistent\NsightPath')
                $result.Count | Should -Be 0
            }
        }
    }

    # ===========================================================================
    # Module Export Completeness
    # ===========================================================================

    Context 'Module Export Completeness' {

        It 'Should export all 4 declared Phase 1 public functions' {
            $mod = Get-Module 'PC-AI.Gpu'
            $expected = @(
                'Get-NvidiaGpuInventory',
                'Get-NvidiaSoftwareRegistry',
                'Get-NvidiaSoftwareStatus',
                'Get-NvidiaGpuUtilization'
            )
            foreach ($fn in $expected) {
                $mod.ExportedFunctions.Keys | Should -Contain $fn
            }
        }

        It 'All exported functions have comment-based help synopsis' {
            $missing = foreach ($fn in (Get-Module 'PC-AI.Gpu').ExportedFunctions.Keys) {
                $help = Get-Help $fn -ErrorAction SilentlyContinue
                if (-not $help.Synopsis -or $help.Synopsis -match '^$') { $fn }
            }
            $missing | Should -BeNullOrEmpty -Because 'Every public function must have a .SYNOPSIS'
        }

        It 'Get-NvidiaGpuInventory accepts -IncludeCim switch' {
            $cmd = Get-Command 'Get-NvidiaGpuInventory' -Module 'PC-AI.Gpu'
            $cmd.Parameters.Keys | Should -Contain 'IncludeCim'
        }

        It 'Get-NvidiaSoftwareRegistry accepts -RegistryPath parameter' {
            $cmd = Get-Command 'Get-NvidiaSoftwareRegistry' -Module 'PC-AI.Gpu'
            $cmd.Parameters.Keys | Should -Contain 'RegistryPath'
        }

        It 'Get-NvidiaSoftwareRegistry accepts -ComponentId parameter' {
            $cmd = Get-Command 'Get-NvidiaSoftwareRegistry' -Module 'PC-AI.Gpu'
            $cmd.Parameters.Keys | Should -Contain 'ComponentId'
        }

        It 'Get-NvidiaSoftwareRegistry accepts -Category parameter' {
            $cmd = Get-Command 'Get-NvidiaSoftwareRegistry' -Module 'PC-AI.Gpu'
            $cmd.Parameters.Keys | Should -Contain 'Category'
        }

        It 'Get-NvidiaSoftwareStatus accepts -Registry parameter' {
            $cmd = Get-Command 'Get-NvidiaSoftwareStatus' -Module 'PC-AI.Gpu'
            $cmd.Parameters.Keys | Should -Contain 'Registry'
        }
    }
}
