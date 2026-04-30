# NVIDIA Framework Deployment & Modernization Plan

> Created: 2026-03-18 | Status: In Progress
> Phases execute in parallel where possible

## Current System State (Validated)

| Component | Installed | Latest | Status | Action |
|-----------|-----------|--------|--------|--------|
| GPU Driver | 582.41 | 591.55 | **Outdated** | Upgrade |
| CUDA Toolkit | 13.2.0 | 13.2 | Current | Fix PATH ordering |
| cuDNN | 9.8.0 | 9.8.0 | Current | None |
| TensorRT | 10.9.0 | 10.9.0 | Current | None |
| Nsight Compute | 2026.1.0 | 2026.1.0 | Current | None |
| Nsight Systems | 2025.6.3 | 2025.6.3 | Current | None |
| Nsight Graphics | Not installed | 2025.5 | **Missing** | Install |
| NVML (Rust) | 0.12.0 | 0.12.0 | Integrated | Wire to PS module |
| Warp (Python) | Not installed | 1.5.1 | **Missing** | Install |
| ONNX Model | Generated | - | Ready | Validated |

## Phase 1: Validation & Testing (Day 1)

### 1A: Run full test suite against new module
- [ ] `Invoke-Pester Tests/Unit/PC-AI.Gpu.Tests.ps1`
- [ ] `Invoke-Pester Tests/Unit/PC-AI.Media.Tests.ps1`
- [ ] `cargo test -p pcai_core_lib --features nvml`
- [ ] `cargo test -p pcai-media`
- [ ] Validate Get-NvidiaCompatibilityMatrix (was crashing)
- [ ] Validate Sync-NvidiaDriverVersion -WhatIf (format strings fixed)

### 1B: Fix remaining detection issues
- [x] cuDNN detection (v9.x subdir pattern)
- [ ] CUDA PATH ordering (nvcc 12.9 shadows 13.1/13.2)
- [ ] Set CUDA_PATH_V13_2 env var for installed v13.2
- [ ] Run Initialize-NvidiaEnvironment -Scope Process to validate

### 1C: Jules review sessions
- [ ] Launch Jules on PC-AI.Gpu module for code quality review
- [ ] Launch Jules on Sync-NvidiaDriverVersion for testing
- [ ] Launch Jules on gpu/mod.rs for Rust review

## Phase 2: Driver & SDK Modernization (Day 1-2)

### 2A: GPU Driver Update (582.41 -> 591.55)
- [ ] Run `Sync-NvidiaDriverVersion` to verify status
- [ ] Download driver installer from nvidia.com/drivers
- [ ] Backup environment: `Initialize-NvidiaEnvironment -WhatIf`
- [ ] Install driver (silent, -noreboot)
- [ ] Verify both GPUs work post-update
- [ ] Run Invoke-JanusGpuSmoke.ps1 to validate inference

### 2B: Install missing SDKs
- [ ] Install Nsight Graphics 2025.5 (developer.nvidia.com)
- [ ] Install Warp: `pip install warp-lang` in AI-Media venv
- [ ] Verify NPP+ bundled with CUDA 13.2
- [ ] Update nvidia-software-registry.json with new versions

### 2C: CUDA environment fix
- [ ] Run Initialize-NvidiaEnvironment -Scope Machine to fix PATH
- [ ] Verify nvcc --version reports 13.2 (not 12.9)
- [ ] Set CUDA_COMPUTE_CAPS=89,120 persistently
- [ ] Rebuild pcai-media with CUDA_COMPUTE_CAPS=89,120

## Phase 3: Integration & Wiring (Day 2-3)

### 3A: Wire NVML into PowerShell (replace nvidia-smi subprocess)
- [ ] Get-NvidiaGpuInventory: add NVML FFI primary path, CIM fallback
- [ ] Get-NvidiaGpuUtilization: add NVML FFI primary path
- [ ] Invoke-JanusGpuSmoke.ps1: replace Get-CudaGpuUsage with PC-AI.Gpu
- [ ] Benchmarks.Media.ps1: replace ad-hoc GPU helpers
- [ ] Collect-SystemPerformanceData.ps1: use PC-AI.Gpu module

### 3B: Wire NVML into Rust (replace nvidia-smi subprocess in Rust)
- [ ] pcai_media/config.rs: replace all_nvidia_smi_gpus() with gpu module
- [ ] rust-functiongemma-core/gpu.rs: replace query_nvidia_smi()
- [ ] Add pcai_core_lib dependency to pcai_media (for NVML access)

### 3C: C# P/Invoke updates
- [ ] Update OptimizerModule.cs for NVML FFI exports
- [ ] Update NativeResolver.cs with gpu DLL paths

## Phase 4: CI/CD & Automation (Day 3-4)

### 4A: GitHub Workflows
- [ ] Create .github/workflows/nvidia-validation.yml
- [ ] Add NVIDIA stack check to maintenance.yml
- [ ] Add NVML feature to ci.yml Rust build matrix

### 4B: Automated registry updates
- [ ] Script to query NVIDIA download pages for latest versions
- [ ] Update-NvidiaSoftwareRegistry -RefreshFromSystem automation
- [ ] Weekly cron via maintenance.yml

### 4C: Documentation
- [ ] Update CLAUDE.md with PC-AI.Gpu module docs
- [ ] Update Build.ps1 component list
- [ ] Generate auto-docs for PC-AI.Gpu functions

## Phase 5: Performance Optimization (Day 4-5)

### 5A: Benchmark baseline
- [ ] Run Invoke-PcaiToolingBenchmarks with NVML vs nvidia-smi comparison
- [ ] Measure GPU query latency: NVML FFI vs subprocess
- [ ] Measure cuda:auto device selection with NVML vs nvidia-smi

### 5B: Build optimization
- [ ] Rebuild pcai-media with CUDA 13.2 + CUDA_COMPUTE_CAPS=89,120
- [ ] Verify SM 120 (RTX 5060 Ti) kernel generation
- [ ] Test cuda:auto fallback on both GPUs
- [ ] Run Invoke-JanusGpuSmoke.ps1 on cuda:0, cuda:1, cuda:auto

### 5C: Memory optimization
- [ ] Profile NVML memory usage vs nvidia-smi subprocess overhead
- [ ] Validate OnceLock singleton prevents NVML re-init
- [ ] Check for handle leaks in continuous GPU polling

## Success Criteria

- [ ] All Pester tests pass (PC-AI.Gpu, PC-AI.Media)
- [ ] All Rust tests pass (pcai_core_lib --features nvml, pcai-media)
- [ ] GPU driver updated to 591.55+
- [ ] nvcc --version reports 13.2
- [ ] Both GPUs detected with correct ComputeCapability
- [ ] Get-NvidiaCompatibilityMatrix shows no blockers
- [ ] Sync-NvidiaDriverVersion shows UpToDate for all GPUs
- [ ] cuda:auto selects GPU correctly with NVML
- [ ] No nvidia-smi subprocess calls in hot paths
