# Production orchestration is exercised with hardware and executable boundaries
# replaced before invocation. Fixture files live exclusively in Pester TestDrive.
class PcaiGpuContractsBackend {
    static [int]$Count = 1
    static [string]$Json
    static [bool]$ThrowQuery = $false
    static [int]$MarshalCalls = 0
    static [string]$Model
    static [ulong]$Context
    static [ulong]$Required
    static [int] pcai_gpu_count() { return [PcaiGpuContractsBackend]::Count }
    static [IntPtr] pcai_gpu_info_json() { return [IntPtr]42 }
    static [IntPtr] StringToUtf8Ptr([string]$model) {
        [PcaiGpuContractsBackend]::Model = $model
        return [IntPtr]::Zero
    }
    static [IntPtr] pcai_gpu_preflight_json([IntPtr]$model, [ulong]$context, [ulong]$required) {
        [PcaiGpuContractsBackend]::Context = $context
        [PcaiGpuContractsBackend]::Required = $required
        if ([PcaiGpuContractsBackend]::ThrowQuery) { throw 'Synthetic FFI query refusal' }
        return [IntPtr]42
    }
    static [string] MarshalAndFree([IntPtr]$pointer) {
        if ($pointer -ne [IntPtr]42) { throw 'Unexpected synthetic JSON custody' }
        [PcaiGpuContractsBackend]::MarshalCalls++
        return [PcaiGpuContractsBackend]::Json
    }
}

Describe 'GPU environment planning and component discovery' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        Mock Resolve-NvidiaInstallPath { @{ CUDA = 'fixture-cuda'; cuDNN = 'fixture-cudnn'; TensorRT = 'fixture-trt' } }
        Mock Get-NvidiaSoftwareRegistry { [pscustomobject]@{ Components = @([pscustomobject]@{ id = 'unrelated' }) } }
        Mock Get-CudaVersionFromPath { '13.1' }
        Mock Get-NvidiaGpuInventory {
            [pscustomobject]@{ ComputeCapability = '8.9' }
            [pscustomobject]@{ ComputeCapability = '8.9' }
            [pscustomobject]@{ ComputeCapability = '12.0' }
        }
        Mock Test-Path { $Path -like 'fixture-*' }
        Mock Set-Item { throw 'Unexpected environment mutation' }
        Mock Write-Warning {}
    }
    It 'plans initialization without environment writes in each supported scope' -ForEach @(
        @{ Scope = 'Process' }, @{ Scope = 'User' }, @{ Scope = 'Machine' }
    ) {
        $originalPath = $env:PATH
        $result = Initialize-NvidiaEnvironment -Scope $Scope -SkipBackup -WhatIf
        $result.CudaPath | Should -Be 'fixture-cuda'
        $result.CudaVersion | Should -Be '13.1'
        $result.CudnnPath | Should -Be 'fixture-cudnn'
        $result.TensorRtPath | Should -Be 'fixture-trt'
        $result.EnvVarsSet.Count | Should -Be 0
        $result.DelegatedToCudaEnv | Should -BeFalse
        $env:PATH | Should -BeExactly $originalPath
        Should -Invoke Set-Item -Exactly -Times 0
    }
    It 'reports unavailable CUDA and failed GPU discovery without claiming initialized compute capabilities' {
        Mock Resolve-NvidiaInstallPath { @{ CUDA = $null; cuDNN = $null; TensorRT = $null } }
        Mock Get-NvidiaGpuInventory { throw 'Synthetic device-query failure' }
        $result = Initialize-NvidiaEnvironment -SkipBackup -WhatIf
        $result.CudaPath | Should -BeNullOrEmpty
        $result.Notes -join '|' | Should -Match 'CUDA not found'
        $result.Notes -join '|' | Should -Match 'Synthetic device-query failure'
        Should -Invoke Set-Item -Exactly -Times 0
    }
}

Describe 'GPU private version-file consumers' -Tag 'Unit', 'Gpu', 'Portable' {
    It 'reads supported CUDA JSON layouts in preference to text or the directory version' -ForEach @(
        @{ Payload = '{"cuda":{"version":"13.1.2"}}' },
        @{ Payload = '{"cuda_cudart":{"version":"13.1.2"}}' }
    ) {
        $cuda = Join-Path $TestDrive 'v12.8'
        [IO.Directory]::CreateDirectory($cuda) | Out-Null
        [IO.File]::WriteAllText((Join-Path $cuda 'version.json'), $Payload)
        [IO.File]::WriteAllText((Join-Path $cuda 'version.txt'), 'CUDA Version 12.8.1')
        Get-CudaVersionFromPath -CudaPath $cuda | Should -Be '13.1.2'
    }
    It 'falls back from malformed CUDA JSON to readable version text' {
        $cuda = Join-Path $TestDrive 'v12.8'
        [IO.Directory]::CreateDirectory($cuda) | Out-Null
        [IO.File]::WriteAllText((Join-Path $cuda 'version.json'), '{bad json')
        [IO.File]::WriteAllText((Join-Path $cuda 'version.txt'), 'CUDA Version 12.8.1')
        Get-CudaVersionFromPath -CudaPath $cuda | Should -Be '12.8.1'
    }
    It 'uses the CUDA directory version only when metadata is absent' {
        $cuda = Join-Path $TestDrive 'v13.1'
        [IO.Directory]::CreateDirectory($cuda) | Out-Null
        Get-CudaVersionFromPath -CudaPath $cuda | Should -Be '13.1'
        Get-CudaVersionFromPath -CudaPath (Join-Path $TestDrive 'absent') | Should -BeNullOrEmpty
    }
    It 'reads cuDNN version-specific include directories and preserves a zero patch' {
        $cudnn = Join-Path $TestDrive 'cudnn'
        $include = Join-Path $cudnn 'include/13.1'
        [IO.Directory]::CreateDirectory($include) | Out-Null
        [IO.File]::WriteAllLines((Join-Path $include 'cudnn_version.h'), @('#define CUDNN_MAJOR 9', '#define CUDNN_MINOR 8', '#define CUDNN_PATCHLEVEL 0'))
        Get-CudnnVersionFromHeader -CudnnPath $cudnn | Should -Be '9.8.0'
    }
    It 'does not invent a cuDNN version from an incomplete header' {
        $cudnn = Join-Path $TestDrive 'cudnn'
        [IO.Directory]::CreateDirectory((Join-Path $cudnn 'include')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $cudnn 'include/cudnn_version.h'), '#define CUDNN_MAJOR 9')
        Get-CudnnVersionFromHeader -CudnnPath $cudnn | Should -BeNullOrEmpty
    }
    It 'reads TensorRT versions from the legacy header and preserves a zero minor' {
        $trt = Join-Path $TestDrive 'tensorrt'
        [IO.Directory]::CreateDirectory((Join-Path $trt 'include')) | Out-Null
        [IO.File]::WriteAllLines((Join-Path $trt 'include/NvInfer.h'), @('#define NV_TENSORRT_MAJOR 10', '#define NV_TENSORRT_MINOR 0', '#define NV_TENSORRT_PATCH 1'))
        Get-TensorRtVersionFromHeader -TensorRtPath $trt | Should -Be '10.0.1'
    }
    It 'does not invent TensorRT version from an incomplete header' {
        $trt = Join-Path $TestDrive 'tensorrt'
        [IO.Directory]::CreateDirectory((Join-Path $trt 'include')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $trt 'include/NvInferVersion.h'), '#define NV_TENSORRT_MAJOR 10')
        Get-TensorRtVersionFromHeader -TensorRtPath $trt | Should -BeNullOrEmpty
    }
    It 'discovers all three Nsight products without querying hardware or executables' {
        foreach ($directory in 'Nsight Compute 2025.1.0', 'Nsight Systems 2025.2', 'Nsight Graphics 2025.3') {
            [IO.Directory]::CreateDirectory((Join-Path $TestDrive $directory)) | Out-Null
        }
        $result = @(Get-NsightVersions -SearchPath $TestDrive)
        $result | Should -HaveCount 3
        ($result | Where-Object Product -eq NsightCompute).Version | Should -Be '2025.1.0'
        ($result | Where-Object Product -eq NsightSystems).Version | Should -Be '2025.2'
        ($result | Where-Object Product -eq NsightGraphics).Version | Should -Be '2025.3'
    }
}

Describe 'GPU environment backup planning' -Tag 'Unit', 'Gpu', 'Portable' {
    It 'plans backup in a private absent directory without writing it' {
        Mock Get-Item { $null } -ParameterFilter { $Path -like 'Env:*' }
        Mock Get-ChildItem { @() } -ParameterFilter { $Path -like 'Env:*' }
        Mock Get-NvidiaDriverVersion { $null }
        Mock Resolve-NvidiaInstallPath { @{ cuDNN = $null; TensorRT = $null } }
        Mock Test-Path { $false }
        $destination = Join-Path $TestDrive 'absent-backup'
        $planned = Backup-NvidiaEnvironment -BackupRoot $destination -WhatIf
        $planned | Should -Match 'nvidia-env-'
        [IO.Directory]::Exists($destination) | Should -BeFalse
        [IO.File]::Exists($planned) | Should -BeFalse
    }
    It 'plans restoring a private fixture without setting variables or PATH' {
        $backup = Join-Path $TestDrive 'backup.json'
        [IO.File]::WriteAllText($backup, '{"EnvVars":{"PCAI_GPU_CONTRACT_FIXTURE":"never-set"},"NvidiaPathSegments":["fixture-cuda"]}')
        Mock Set-Item { throw 'Unexpected restore mutation' }
        $originalPath = $env:PATH
        Backup-NvidiaEnvironment -Restore -BackupFile $backup -WhatIf | Should -Be $backup
        Should -Invoke Set-Item -Exactly -Times 0
        $env:PATH | Should -BeExactly $originalPath
    }
}

Describe 'GPU installer public early-exit contracts' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        Mock Get-NvidiaSoftwareRegistry { [pscustomobject]@{ Components = @([pscustomobject]@{ id = 'fixture'; name = 'Fixture CUDA'; category = 'runtime' }) } }
        Mock Get-NvidiaSoftwareStatus { [pscustomobject]@{ Status = 'Current'; InstalledVersion = '13.1' } }
        Mock Test-NvidiaDownloadUrl { throw 'Unexpected network validation' }
        Mock Invoke-NvidiaSilentInstall { throw 'Unexpected installer launch' }
        Mock Backup-NvidiaEnvironment { throw 'Unexpected backup' }
    }
    It 'returns an already-current success without downloading, backing up or launching an installer' {
        $result = Install-NvidiaSoftware -ComponentId fixture
        $result.Success | Should -BeTrue
        $result.VersionBefore | Should -Be '13.1'
        $result.VersionAfter | Should -BeNullOrEmpty
        $result.Message | Should -Match 'already Current'
        Should -Invoke Test-NvidiaDownloadUrl -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
    It 'rejects an absent registry component before any download or installer' {
        Mock Get-NvidiaSoftwareRegistry { [pscustomobject]@{ Components = @() } }
        { Install-NvidiaSoftware -ComponentId absent } | Should -Throw '*not found*'
        Should -Invoke Get-NvidiaSoftwareStatus -Exactly -Times 0
        Should -Invoke Invoke-NvidiaSilentInstall -Exactly -Times 0
    }
}

BeforeAll {
    $script:GpuRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.Gpu'
    $script:ModuleRoot = (Get-Item $script:GpuRoot).FullName
    foreach ($file in Get-ChildItem (Join-Path $script:GpuRoot 'Private') -File -Filter '*.ps1') { . $file.FullName }
    foreach ($file in Get-ChildItem (Join-Path $script:GpuRoot 'Public') -File -Filter '*.ps1') { . $file.FullName }
    function nvidia-smi.exe { param([Parameter(ValueFromRemainingArguments)][string[]]$Tokens) throw 'Unmocked GPU CLI forbidden.' }
    function Invoke-GpuContractPerf { param([Parameter(ValueFromRemainingArguments)][string[]]$Tokens) throw 'Unmocked preflight CLI forbidden.' }
    function Get-CimInstance { [CmdletBinding()] param($ClassName) throw 'Unmocked hardware query forbidden.' }
    function New-GpuContractRegistry {
        param([string]$Path)
        $data = [ordered]@{
            version = 'fixture'; lastUpdated = 'fixture'; detectionTimestamp = 'fixture'
            components = @(
                @{ id = 'gpu-driver'; name = 'Fixture Driver'; category = 'driver'; latestVersion = '580.0'; installedVersion = '572.83'; downloadUrl = 'https://developer.download.nvidia.com/fixture.exe' },
                @{ id = 'cuda-toolkit'; name = 'Fixture CUDA'; category = 'runtime'; latestVersion = '13.1'; installedVersion = '12.9'; downloadUrl = 'https://developer.download.nvidia.com/cuda.exe' }
            )
            compatibilityMatrix = @{ architectures = @{ 'SM 8.9' = @{
                minimumDriver = '570.0'; minimumCuda = '12.8'; recommendedCuda = '13.1'
                minimumCuDNN = $null; minimumTensorRT = '10.0'
                nsightComputeMinimum = '2025.1'; nsightSystemsMinimum = '2025.1'
            } } }
        }
        [IO.File]::WriteAllText($Path, ($data | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
    }
}

Describe 'GPU preflight backend selection and provider arguments' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        [PcaiGpuContractsBackend]::Json = '{"verdict":"go","reason":"fixture","model_estimate_mb":4096,"best_gpu_index":0,"gpus":[{"index":0,"memory_free_mb":8192}]}'
        [PcaiGpuContractsBackend]::ThrowQuery = $false
        [PcaiGpuContractsBackend]::MarshalCalls = 0
        Mock Resolve-PcaiCoreLibDll { 'fixture-core.dll' }
        Mock Initialize-PreflightInteropType { [PcaiGpuContractsBackend] }
        Mock Test-Path { $false }
        Mock Write-Warning {}
    }
    It 'forwards model/context/VRAM parameters and returns the actual FFI schema with zero GPU index intact' {
        $result = Test-PcaiGpuReadiness -ModelPath 'fixture model.gguf' -ContextLength 8192 -RequiredMB 4096
        $result.Source | Should -Be 'ffi'
        $result.Verdict | Should -Be 'go'
        $result.ModelEstimateMB | Should -Be 4096
        $result.BestGpuIndex | Should -Be 0
        $result.Gpus[0].memory_free_mb | Should -Be 8192
        [PcaiGpuContractsBackend]::Model | Should -Be 'fixture model.gguf'
        [PcaiGpuContractsBackend]::Context | Should -Be 8192
        [PcaiGpuContractsBackend]::Required | Should -Be 4096
        [PcaiGpuContractsBackend]::MarshalCalls | Should -Be 1
    }
    It 'preserves raw FFI JSON without claiming a parsed consumer result' {
        (Test-PcaiGpuReadiness -AsJson) | Should -BeExactly ([PcaiGpuContractsBackend]::Json)
    }
    It 'preserves a null best GPU index and empty GPU inventory' {
        [PcaiGpuContractsBackend]::Json = '{"verdict":"fail","reason":"none","model_estimate_mb":0,"best_gpu_index":null,"gpus":[]}'
        $result = Test-PcaiGpuReadiness
        $result.BestGpuIndex | Should -BeNullOrEmpty
        $result.Gpus | Should -HaveCount 0
        $result.Verdict | Should -Be 'fail'
    }
    It 'returns an explicit unavailable result after failed FFI and absent CLI' {
        [PcaiGpuContractsBackend]::ThrowQuery = $true
        $result = Test-PcaiGpuReadiness
        $result.Source | Should -Be 'none'
        $result.Verdict | Should -Be 'fail'
        Should -Invoke Write-Warning -Exactly -Times 1
    }
    It 'admits CLI go/warn/fail exit codes and skips non-JSON noise while forwarding arguments' -ForEach @(
        @{ Code = 0; Verdict = 'go' }, @{ Code = 1; Verdict = 'warn' }, @{ Code = 2; Verdict = 'fail' }
    ) {
        Mock Resolve-PcaiCoreLibDll { $null }
        Mock Join-Path { 'Invoke-GpuContractPerf' } -ParameterFilter { $ChildPath -like '*pcai-perf.exe' }
        Mock Test-Path { $Path -eq 'Invoke-GpuContractPerf' }
        Mock Invoke-GpuContractPerf { $global:LASTEXITCODE = $Code; 'diagnostic noise'; '{"verdict":"' + $Verdict + '","reason":"fixture","model_estimate_mb":12,"best_gpu_index":null,"gpus":[]}' }
        $result = Test-PcaiGpuReadiness -ModelPath 'fixture model.gguf' -ContextLength 2048 -RequiredMB 64
        $result.Source | Should -Be 'cli'
        $result.Verdict | Should -Be $Verdict
        Should -Invoke Invoke-GpuContractPerf -Exactly -Times 1 -ParameterFilter {
            ($Tokens -join '|') -eq 'preflight|--model|fixture model.gguf|--ctx|2048|--required-mb|64'
        }
    }
    It 'does not present malformed output or unexpected CLI failure as a GPU success' -ForEach @(
        @{ Code = 7; Frame = '{"verdict":"go"}' }, @{ Code = 0; Frame = '{malformed' }, @{ Code = 0; Frame = 'no JSON' }
    ) {
        Mock Resolve-PcaiCoreLibDll { $null }
        Mock Join-Path { 'Invoke-GpuContractPerf' } -ParameterFilter { $ChildPath -like '*pcai-perf.exe' }
        Mock Test-Path { $Path -eq 'Invoke-GpuContractPerf' }
        Mock Invoke-GpuContractPerf { $global:LASTEXITCODE = $Code; $Frame }
        (Test-PcaiGpuReadiness).Verdict | Should -Be 'fail'
    }
}

Describe 'GPU inventory and utilization data fidelity' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        Mock Resolve-PcaiCoreLibDll { $null }
        Mock Test-Path { $false }
        Mock Get-Command { if ($Name -eq 'nvidia-smi.exe') { [pscustomobject]@{ Name = $Name } } } -ParameterFilter { $Name -like '*nvidia-smi.exe' }
        Mock Get-CimInstance { throw 'Unexpected hardware fallback' }
        Mock Write-Warning {}
    }
    It 'retains quoted GPU names, UUID and unknown numeric metrics when selecting one GPU' {
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 0; '0,GPU-a,First GPU,572.83,8.9,8192,1024,50,10'; '1,GPU-b,"Fixture, GPU",572.83,12.0,16384,[Not Supported],N/A,20' }
        $result = @(Get-NvidiaGpuInventory -Index 1)
        $result | Should -HaveCount 1
        $result[0].UUID | Should -Be 'GPU-b'
        $result[0].Name | Should -Be 'Fixture, GPU'
        $result[0].ComputeCapability | Should -Be '12.0'
        $result[0].MemoryTotalMB | Should -Be 16384
        $result[0].MemoryUsedMB | Should -BeNullOrEmpty
        $result[0].Temperature | Should -BeNullOrEmpty
        $result[0].Source | Should -Be 'nvidia-smi'
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }
    It 'uses CIM fallback after a genuine CLI failure and excludes unrelated adapters' {
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 9; 'synthetic refusal' }
        Mock Get-CimInstance {
            [pscustomobject]@{ Name = 'Integrated fixture'; AdapterCompatibility = 'Other'; AdapterRAM = 1GB }
            [pscustomobject]@{ Name = 'NVIDIA fixture'; AdapterCompatibility = 'NVIDIA'; AdapterRAM = 4GB; DriverVersion = 'fixture-version' }
        }
        $result = @(Get-NvidiaGpuInventory)
        $result | Should -HaveCount 1
        $result[0].Source | Should -Be 'cim'
        $result[0].MemoryTotalMB | Should -Be 4096
        $result[0].ComputeCapability | Should -BeNullOrEmpty
        $result[0].UUID | Should -BeNullOrEmpty
    }
    It 'returns FFI inventory and performs the JSON release boundary exactly once' {
        Mock Resolve-PcaiCoreLibDll { 'fixture-core.dll' }
        Mock Initialize-NvmlInteropType { [PcaiGpuContractsBackend] }
        [PcaiGpuContractsBackend]::Count = 1; [PcaiGpuContractsBackend]::MarshalCalls = 0
        [PcaiGpuContractsBackend]::Json = '[{"index":0,"uuid":"GPU-ffi","name":"Fixture FFI","driver_version":"572.83","compute_capability":"8.9","memory_total_mb":8192,"memory_used_mb":512,"temperature_c":40}]'
        $result = @(Get-NvidiaGpuInventory)
        $result[0].Source | Should -Be 'nvml-ffi'
        $result[0].UUID | Should -Be 'GPU-ffi'
        $result[0].MemoryTotalMB | Should -Be 8192
        $result[0].Utilization | Should -BeNullOrEmpty
        [PcaiGpuContractsBackend]::MarshalCalls | Should -Be 1
    }
    It 'keeps decimal power and unsupported fan distinct from zero utilization' {
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 0; '0,"Fixture, GPU",0,512,8192,40,15.5,[Not Supported]' }
        $result = @(Get-NvidiaGpuUtilization)
        $result[0].Name | Should -Be 'Fixture, GPU'
        $result[0].GpuUtilization | Should -Be 0
        $result[0].PowerDraw | Should -Be ([decimal]15.5)
        $result[0].FanSpeed | Should -BeNullOrEmpty
        $result[0].Timestamp | Should -BeOfType [datetime]
    }
    It 'does not replace utilization failure with static hardware metrics' {
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 1; 'refused' }
        @(Get-NvidiaGpuUtilization) | Should -HaveCount 0
        Should -Invoke Get-CimInstance -Exactly -Times 0
        Should -Invoke Write-Warning -Exactly -Times 1
    }
    It 'selects the first nonempty driver version and does not query CIM on CLI success' {
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 0; ''; '572.83'; '580.0' }
        Get-NvidiaDriverVersion | Should -Be '572.83'
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }
}

Describe 'GPU compatibility requirements and private registry schema' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        $script:GpuRegistry = Join-Path $TestDrive 'gpu-registry.json'
        New-GpuContractRegistry -Path $script:GpuRegistry
        Mock Get-NvidiaGpuInventory { [pscustomobject]@{ Name = 'Fixture Ada'; Index = 0; ComputeCapability = '8.9' } }
        Mock Resolve-NvidiaInstallPath { @{ CUDA = 'fixture-cuda'; cuDNN = $null; TensorRT = $null } }
        Mock Get-NvidiaDriverVersion { '572.83.0' }
        Mock Get-CudaVersionFromPath { '12.9' }
        Mock Get-NsightVersions { [pscustomobject]@{ Product = 'NsightCompute'; Version = '2025.2' } }
        Mock Write-Warning {}
    }
    It 'preserves component filtering and private path without reading canonical config' {
        $result = Get-NvidiaSoftwareRegistry -RegistryPath $script:GpuRegistry -ComponentId cuda-toolkit -Category runtime
        $result.Components | Should -HaveCount 1
        $result.Components[0].name | Should -Be 'Fixture CUDA'
        $result.Version | Should -Be 'fixture'
    }
    It 'distinguishes driver compatibility, CUDA upgrade, absent required software and nonrequired libraries' {
        $rows = @(Get-NvidiaCompatibilityMatrix -RegistryPath $script:GpuRegistry)
        $rows | Should -HaveCount 6
        ($rows | Where-Object ComponentId -eq gpu-driver).Status | Should -Be 'Compatible'
        ($rows | Where-Object ComponentId -eq cuda-toolkit).Status | Should -Be 'UpgradeRecommended'
        ($rows | Where-Object ComponentId -eq cudnn).Status | Should -Be 'NotRequired'
        ($rows | Where-Object ComponentId -eq tensorrt).Status | Should -Be 'NotInstalled'
        ($rows | Where-Object ComponentId -eq tensorrt).IsBlocker | Should -BeTrue
        ($rows | Where-Object ComponentId -eq nsight-compute).Status | Should -Be 'Compatible'
    }
    It 'uses numeric CUDA requirements rather than string order' -ForEach @(
        @{ Version = '12.7'; Status = 'Incompatible'; Blocker = $true },
        @{ Version = '13.1'; Status = 'Compatible'; Blocker = $false },
        @{ Version = '14.0'; Status = 'Compatible'; Blocker = $false }
    ) {
        Mock Get-CudaVersionFromPath { $Version }
        $row = Get-NvidiaCompatibilityMatrix -RegistryPath $script:GpuRegistry | Where-Object ComponentId -eq cuda-toolkit
        $row.Status | Should -Be $Status
        $row.IsBlocker | Should -Be $Blocker
    }
    It 'does not invent requirements for an unknown GPU architecture' {
        Mock Get-NvidiaGpuInventory { [pscustomobject]@{ Name = 'Fixture future GPU'; Index = 0; ComputeCapability = '99.0' } }
        @(Get-NvidiaCompatibilityMatrix -RegistryPath $script:GpuRegistry) | Should -HaveCount 0
    }
    It 'plans a private registry update without changing its exact bytes under WhatIf' {
        $before = (Get-FileHash $script:GpuRegistry).Hash
        Mock New-Item { throw 'Unexpected backup directory creation' }
        $result = Update-NvidiaSoftwareRegistry -RegistryPath $script:GpuRegistry -ComponentId cuda-toolkit -LatestVersion '14.0' -WhatIf
        $result.WhatIf | Should -BeTrue
        $result.Changes | Should -HaveCount 1
        $result.Changes[0].OldValue | Should -Be '13.1'
        $result.Changes[0].NewValue | Should -Be '14.0'
        (Get-FileHash $script:GpuRegistry).Hash | Should -Be $before
        Should -Invoke New-Item -Exactly -Times 0
    }
}

Describe 'GPU download trust is bounded before transport' -Tag 'Unit', 'Gpu', 'Portable' {
    It 'admits a trusted NVIDIA subdomain with an explicit no-network check' {
        $result = Test-NvidiaDownloadUrl -Url 'https://developer.download.nvidia.com/fixture.exe' -SkipReachabilityCheck
        $result.IsValid | Should -BeTrue
        $result.IsTrusted | Should -BeTrue
        $result.StatusCode | Should -Be -1
    }
    It 'rejects deceptive host suffixes without contacting them' -ForEach @(
        @{ Url = 'https://nvidia.com.attacker.invalid/file.exe' }, @{ Url = 'https://attacker-nvidia.com/file.exe' }
    ) {
        Mock Write-Warning {}
        $result = Test-NvidiaDownloadUrl -Url $Url -SkipReachabilityCheck
        $result.IsTrusted | Should -BeFalse
        $result.IsValid | Should -BeFalse
    }
}


Describe 'GPU read-only detecting controls' {
    It 'selects CUDA 13.1 rather than 9.9 when both are installed' {
        Mock Test-Path { $Path -eq 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' }
        Mock Get-ChildItem {
            [pscustomobject]@{ Name = 'v9.9'; FullName = 'C:\fixture\v9.9' }
            [pscustomobject]@{ Name = 'v13.1'; FullName = 'C:\fixture\v13.1' }
        }
        Resolve-NvidiaInstallPath -ComponentType CUDA | Should -Be 'C:\fixture\v13.1'
    }
    It 'rejects an insecure trusted-host URL without making a network request' -ForEach @(
        @{ Url = 'http://developer.download.nvidia.com/fixture.exe' },
        @{ Url = 'ftp://developer.download.nvidia.com/fixture.exe' }
    ) {
        (Test-NvidiaDownloadUrl -Url $Url -SkipReachabilityCheck).IsValid | Should -BeFalse
    }
    It 'accepts a trusted HTTPS URL without making a network request' {
        (Test-NvidiaDownloadUrl -Url 'https://developer.download.nvidia.com/fixture.exe' -SkipReachabilityCheck).IsValid | Should -BeTrue
    }
    It 'does not create an environment backup under WhatIf' {
        Mock Backup-NvidiaEnvironment { 'fixture-only-no-write.json' }
        Mock Resolve-NvidiaInstallPath { @{ CUDA = $null; cuDNN = $null; TensorRT = $null } }
        Mock Get-NvidiaSoftwareRegistry { [pscustomobject]@{ Components = @() } }
        Mock Get-NvidiaGpuInventory { @() }
        Mock Test-Path { $false }
        Mock Set-Item { throw 'Unexpected environment write' }
        Mock Write-Warning {}
        $originalPath = $env:PATH
        Initialize-NvidiaEnvironment -Scope Process -WhatIf | Out-Null
        $env:PATH | Should -BeExactly $originalPath
        Should -Invoke Set-Item -Exactly -Times 0
        Should -Invoke Backup-NvidiaEnvironment -Exactly -Times 0
    }
    It 'does not create an installer log directory under WhatIf' {
        $principal = [pscustomobject]@{}
        $principal | Add-Member ScriptMethod IsInRole { param($Role) return $true }
        Mock New-Object { $principal } -ParameterFilter { $TypeName -like '*WindowsPrincipal*' }
        Mock Test-Path { $LiteralPath -eq 'C:\fixture\installer.exe' }
        Mock Get-Item { [pscustomobject]@{ Extension = '.exe'; BaseName = 'fixture' } }
        Mock New-Item {}
        Invoke-NvidiaSilentInstall -InstallerPath 'C:\fixture\installer.exe' -WhatIf | Out-Null
        Should -Invoke New-Item -Exactly -Times 0
    }
    It 'preserves numeric inventory fields from comma-space CSV' {
        Mock Resolve-PcaiCoreLibDll { $null }
        Mock Test-Path { $false }
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 0; '0, GPU-a, Fixture GPU, 572.83, 8.9, 8192, 512, 40, 10' }
        $gpu = @(Get-NvidiaGpuInventory)[0]
        $gpu.MemoryTotalMB | Should -Be 8192
        $gpu.MemoryUsedMB | Should -Be 512
        $gpu.Temperature | Should -Be 40
        $gpu.Utilization | Should -Be 10
    }
    It 'preserves numeric utilization fields from comma-space CSV' {
        Mock Test-Path { $false }
        Mock nvidia-smi.exe { $global:LASTEXITCODE = 0; '0, Fixture GPU, 0, 512, 8192, 40, 15.5, [Not Supported]' }
        $gpu = @(Get-NvidiaGpuUtilization)[0]
        $gpu.MemoryTotalMB | Should -Be 8192
        $gpu.GpuUtilization | Should -Be 0
        $gpu.PowerDraw | Should -Be ([decimal]15.5)
        $gpu.FanSpeed | Should -BeNullOrEmpty
    }
}

Describe 'GPU numeric installed-version and refusal contracts' -Tag 'Unit', 'Gpu', 'Portable' {
    It 'selects numeric versions and skips malformed, overflow and prerelease names for each versioned component' -ForEach @(
        @{ Component = 'CUDA'; Root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' },
        @{ Component = 'cuDNN'; Root = 'C:\Program Files\NVIDIA\CUDNN' }
    ) {
        Mock Test-Path { $Path -eq $Root }
        Mock Get-ChildItem {
            foreach ($name in 'v9.9', 'v13.1.2', 'v13.1.1', 'v99.0-preview', 'version-junk', 'v9999999999999999.0') {
                [pscustomobject]@{ Name = $name; FullName = "C:\fixture\$name" }
            }
        }
        Resolve-NvidiaInstallPath -ComponentType $Component | Should -Be 'C:\fixture\v13.1.2'
    }
    It 'returns unavailable rather than inventing a version from invalid-only installations' -ForEach @(
        @{ Component = 'CUDA'; Root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' },
        @{ Component = 'cuDNN'; Root = 'C:\Program Files\NVIDIA\CUDNN' }
    ) {
        Mock Test-Path { $Path -eq $Root }
        Mock Get-ChildItem { [pscustomobject]@{ Name = 'v99.0-preview'; FullName = 'C:\fixture\unqualified' } }
        Resolve-NvidiaInstallPath -ComponentType $Component | Should -BeNullOrEmpty
    }
    It 'preserves the embedded CUDA-header fallback for cuDNN' {
        Mock Test-Path {
            $Path -eq 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -or
            $Path -eq 'C:\fixture\v13.1\include\cudnn_version.h'
        }
        Mock Get-ChildItem { [pscustomobject]@{ Name = 'v13.1'; FullName = 'C:\fixture\v13.1' } }
        Resolve-NvidiaInstallPath -ComponentType cuDNN | Should -Be 'C:\fixture\v13.1'
    }
    It 'rejects relative, file and insecure installer URLs before any network request' -ForEach @(
        @{ Url = 'fixture.exe' }, @{ Url = 'file://developer.download.nvidia.com/fixture.exe' },
        @{ Url = 'http://developer.download.nvidia.com/fixture.exe' }, @{ Url = 'ftp://developer.download.nvidia.com/fixture.exe' }
    ) {
        Mock Write-Warning {}
        $result = Test-NvidiaDownloadUrl -Url $Url
        $result.IsTrusted | Should -BeFalse
        $result.IsValid | Should -BeFalse
        $result.StatusCode | Should -Be -1
    }
}

Describe 'GPU software status comparisons and aliases' -Tag 'Unit', 'Gpu', 'Portable' {
    BeforeEach {
        Mock Resolve-NvidiaInstallPath { @{ CUDA = 'fixture-cuda'; cuDNN = $null; TensorRT = $null } }
        Mock Get-NvidiaDriverVersion { '572.83' }
        Mock Get-CudaVersionFromPath { '13.1' }
        Mock Get-NsightVersions { @() }
        Mock Test-Path { $false }
        Mock Get-ChildItem { throw 'Unexpected real install directory query' }
    }
    It 'compares detected CUDA versions numerically and preserves the enriched-registry alias' -ForEach @(
        @{ Latest = '9.9'; Expected = 'Current' }, @{ Latest = '13.1'; Expected = 'Current' },
        @{ Latest = '14.0'; Expected = 'Outdated' }, @{ Latest = $null; Expected = 'Unknown' }
    ) {
        Mock Get-NvidiaSoftwareRegistry {
            [pscustomobject]@{ Components = @([pscustomobject]@{ id = 'cuda-toolkit'; name = 'Fixture CUDA'; latestVersion = $Latest }) }
        }
        $result = @(Get-NvidiaSoftwareStatus -RegistryPath 'fixture-only.json' -ComponentId cuda-toolkit)
        $result | Should -HaveCount 1
        $result[0].Status | Should -Be $Expected
        $result[0].InstalledVersion | Should -Be '13.1'
        $result[0].Path | Should -Be 'fixture-cuda'
        Should -Invoke Get-NvidiaSoftwareRegistry -Exactly -Times 1 -ParameterFilter {
            $RegistryPath -eq 'fixture-only.json' -and $ComponentId -eq 'cuda-toolkit'
        }
    }
    It 'reports unavailable driver, cuDNN and TensorRT without fabricated installed versions' {
        Mock Get-NvidiaDriverVersion { $null }
        Mock Get-NvidiaSoftwareRegistry {
            [pscustomobject]@{ Components = @(
                [pscustomobject]@{ id = 'gpu-driver'; name = 'Fixture Driver'; latestVersion = '580.0' },
                [pscustomobject]@{ id = 'cudnn'; name = 'Fixture cuDNN'; latestVersion = '9.8' },
                [pscustomobject]@{ id = 'tensorrt'; name = 'Fixture TensorRT'; latestVersion = '10.0' }
            ) }
        }
        $rows = @(Get-NvidiaSoftwareStatus -RegistryPath 'fixture-only.json')
        $rows | Should -HaveCount 3
        @($rows | Where-Object Status -ne NotInstalled) | Should -HaveCount 0
        @($rows | Where-Object InstalledVersion) | Should -HaveCount 0
    }
}
