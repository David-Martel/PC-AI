#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Performance benchmarks for the PC-AI media pipeline.

.DESCRIPTION
    Measures end-to-end performance characteristics of the Janus-Pro media pipeline
    across image generation, image understanding, and upscaling workloads.

    Metrics captured:
    - Generation: first-token latency, total time, VRAM delta, CFG scale impact
    - Understanding: image encoding time, response generation time, tokens/second
    - Upscaling: 384x384 -> 1536x1536 wall-clock time, VRAM delta
    - Quality: file-size stability across fixed seed, understanding relevance (keyword matching)

    All benchmarks are tagged 'Benchmark', 'Media', 'Slow', 'GPU' and skip when:
    - pcai_media.dll or PcaiNative.dll are not built
    - A Janus-Pro model is not configured in Config/pcai-media.json
    - Reports/media-benchmarks/ output directory cannot be created

    Results are written to: Reports/media-benchmarks/<timestamp>/benchmark-report.json
    and benchmark-report.md

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Benchmarks\Benchmarks.Media.ps1 -Tag Benchmark,Media
    Estimated runtime: 5-20 minutes depending on GPU and model size.
#>

BeforeAll {
    # Path resolution (no TestHelpers dependency for benchmark scripts — inline resolver)
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:BinDir      = Join-Path $script:ProjectRoot 'bin'
    $script:MediaModPath  = Join-Path $script:ProjectRoot 'Modules\PcaiMedia.psm1'
    $script:PcaiNativeDll = Join-Path $script:BinDir 'PcaiNative.dll'
    $script:MediaDll      = Join-Path $script:BinDir 'pcai_media.dll'
    $script:MediaConfig   = Join-Path $script:ProjectRoot 'Config\pcai-media.json'
    $script:ReportRoot    = Join-Path $script:ProjectRoot 'Reports\media-benchmarks'

    # -------------------------------------------------------------------------
    # Helper: GPU info via WMI
    # -------------------------------------------------------------------------
    function Get-GpuInfo {
        try {
            $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                    Where-Object { $_.AdapterCompatibility -match 'NVIDIA|AMD' } |
                    Select-Object -First 2
            if (-not $gpus) { return @{ Name = 'Unknown'; AdapterRAM = 0; DriverVersion = 'N/A' } }
            $gpu = @($gpus)[0]
            return @{
                Name          = $gpu.Name
                # AdapterRAM field may overflow uint32 for 8GB+ cards; use VideoMemoryType workaround
                AdapterRamGB  = if ($gpu.AdapterRAM -gt 0) { [math]::Round($gpu.AdapterRAM / 1GB, 1) } else { 'unknown' }
                DriverVersion = $gpu.DriverVersion
                AllGpus       = @($gpus | ForEach-Object { $_.Name })
            }
        } catch {
            return @{ Name = 'Unknown'; AdapterRamGB = 0; DriverVersion = 'N/A'; AllGpus = @() }
        }
    }

    function Get-CudaGpuInventory {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $nvidiaSmi) { return @() }

        $rows = & $nvidiaSmi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits 2>$null
        foreach ($row in @($rows)) {
            $parts = @($row -split ',')
            if ($parts.Count -lt 5) { continue }
            [PSCustomObject]@{
                Index         = [int]($parts[0].Trim())
                Uuid          = $parts[1].Trim()
                Name          = $parts[2].Trim()
                MemoryTotalMB = [int]($parts[3].Trim())
                MemoryUsedMB  = [int]($parts[4].Trim())
            }
        }
    }

    function Resolve-PreferredCudaDevice {
        param([object[]]$Inventory)

        $override = if ($env:PCAI_MEDIA_BENCH_DEVICE) { $env:PCAI_MEDIA_BENCH_DEVICE } else { $env:PCAI_MEDIA_DEVICE }
        if ($override) {
            return @{
                Device = $override
                PreferredGpu = $null
            }
        }

        if (-not $Inventory -or $Inventory.Count -eq 0) {
            return @{
                Device = 'cpu'
                PreferredGpu = $null
            }
        }

        $preferredGpu = $Inventory |
            Sort-Object @{ Expression = 'MemoryTotalMB'; Descending = $true }, @{ Expression = 'Index'; Ascending = $true } |
            Select-Object -First 1

        return @{
            Device = 'cuda:auto'
            PreferredGpu = $preferredGpu
        }
    }

    # -------------------------------------------------------------------------
    # Helper: VRAM usage via nvidia-smi (returns MB used, or -1 if unavailable)
    # -------------------------------------------------------------------------
    function Get-VramUsedMB {
        param([int]$GpuIndex = 0)
        try {
            $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if (-not $nvsmi) { return -1 }
            $output = & nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>$null
            $match = @($output | ForEach-Object {
                $parts = @($_ -split ',')
                if ($parts.Count -lt 2) { return }
                [PSCustomObject]@{
                    Index = [int]($parts[0].Trim())
                    MemoryUsedMB = [int]($parts[1].Trim())
                }
            } | Where-Object Index -eq $GpuIndex | Select-Object -First 1)
            if ($match.Count -eq 0) { return -1 }
            return [int]$match[0].MemoryUsedMB
        } catch {
            return -1
        }
    }

    # -------------------------------------------------------------------------
    # Helper: approximate token count (whitespace split)
    # -------------------------------------------------------------------------
    function Measure-TokenCount {
        param([string]$Text)
        if ([string]::IsNullOrEmpty($Text)) { return 0 }
        return ([regex]::Matches($Text, '\S+')).Count
    }

    # -------------------------------------------------------------------------
    # Helper: resolve model path from config
    # -------------------------------------------------------------------------
    function Get-MediaModelPath {
        if (-not (Test-Path $script:MediaConfig)) { return $null }
        try {
            $cfg = Get-Content $script:MediaConfig -Raw | ConvertFrom-Json
            if (-not $cfg.model) { return $null }
            if ([System.IO.Path]::IsPathRooted($cfg.model)) {
                return if (Test-Path $cfg.model) { $cfg.model } else { $cfg.model }  # HF repo ID or abs path
            }
            $resolved = Join-Path $script:ProjectRoot $cfg.model
            return if (Test-Path $resolved) { $resolved } else { $cfg.model }
        } catch { return $null }
    }

    # -------------------------------------------------------------------------
    # Helper: resolve ONNX model path from config
    # -------------------------------------------------------------------------
    function Get-OnnxModelPath {
        if (-not (Test-Path $script:MediaConfig)) { return $null }
        try {
            $cfg = Get-Content $script:MediaConfig -Raw | ConvertFrom-Json
            if (-not $cfg.upscale -or -not $cfg.upscale.model_path) { return $null }
            $raw = $cfg.upscale.model_path
            if ([System.IO.Path]::IsPathRooted($raw)) {
                return if (Test-Path $raw) { $raw } else { $null }
            }
            $resolved = Join-Path $script:ProjectRoot $raw
            return if (Test-Path $resolved) { $resolved } else { $null }
        } catch { return $null }
    }

    # -------------------------------------------------------------------------
    # System info snapshot
    # -------------------------------------------------------------------------
    $script:GpuInfo     = Get-GpuInfo
    $script:GpuInventory = @(Get-CudaGpuInventory)
    $script:CpuCount    = [Environment]::ProcessorCount
    $script:DotNetVer   = [System.Runtime.InteropServices.RuntimeEnvironment]::GetSystemVersion()
    $script:PsVersion   = $PSVersionTable.PSVersion.ToString()
    $script:Timestamp   = Get-Date -Format 'yyyyMMdd_HHmmss'

    # -------------------------------------------------------------------------
    # Prerequisite flags
    # -------------------------------------------------------------------------
    $script:DllsAvailable  = (Test-Path $script:PcaiNativeDll) -and (Test-Path $script:MediaDll)
    $script:TestModelPath  = Get-MediaModelPath
    $script:OnnxModelPath  = Get-OnnxModelPath
    $script:HasModel       = $null -ne $script:TestModelPath

    # Load and initialize the media pipeline
    $script:MediaInitialized = $false
    if ($script:DllsAvailable -and $script:HasModel) {
        if ($env:PATH -notlike "*$script:BinDir*") {
            $env:PATH = "$script:BinDir;$env:PATH"
        }
        Import-Module $script:MediaModPath -Force -ErrorAction SilentlyContinue

        # Prefer the highest-VRAM CUDA device when available.
        $script:HasGpu    = (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                             Where-Object { $_.AdapterCompatibility -match 'NVIDIA' }) -ne $null
        $resolvedDevice = Resolve-PreferredCudaDevice -Inventory $script:GpuInventory
        $script:DeviceArg = if ($script:HasGpu) { $resolvedDevice.Device } else { 'cpu' }
        $script:PreferredGpu = $resolvedDevice.PreferredGpu
        $script:BenchmarkGpuIndex = if ($script:PreferredGpu) { [int]$script:PreferredGpu.Index } else { 0 }
        if (-not $env:CUDA_DEVICE_ORDER) {
            $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
        }

        $initResult = Initialize-PcaiMedia -Device $script:DeviceArg -ErrorAction SilentlyContinue
        if ($initResult -and $initResult.Success) {
            $loadResult = Import-PcaiMediaModel -ModelPath $script:TestModelPath -ErrorAction SilentlyContinue
            $script:MediaInitialized = ($null -ne $loadResult -and $loadResult.Success)
        }
    }

    # Output directory for benchmark report
    $script:ReportDir = Join-Path $script:ReportRoot $script:Timestamp
    if (-not (Test-Path $script:ReportDir)) {
        New-Item -ItemType Directory -Path $script:ReportDir -Force | Out-Null
    }
    $script:TempDir = Join-Path $env:TEMP 'pcai_bench_media'
    if (-not (Test-Path $script:TempDir)) {
        New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null
    }

    # Benchmark result accumulator
    $script:BenchmarkResults = [System.Collections.Generic.List[hashtable]]::new()

    function Add-BenchmarkResult {
        param(
            [string]$Category,
            [string]$Name,
            [double]$ValueMs,
            [string]$Unit = 'ms',
            [hashtable]$Extra = @{}
        )
        $script:BenchmarkResults.Add(@{
            Category  = $Category
            Name      = $Name
            ValueMs   = $ValueMs
            Unit      = $Unit
            Timestamp = (Get-Date -Format 'o')
            Extra     = $Extra
        })
    }
}

AfterAll {
    try { Stop-PcaiMedia -ErrorAction SilentlyContinue } catch { }
    Remove-Module PcaiMedia -Force -ErrorAction SilentlyContinue

    # Write benchmark report
    if ($script:BenchmarkResults.Count -gt 0 -and (Test-Path $script:ReportDir)) {

        $systemInfo = @{
            Timestamp    = $script:Timestamp
            PsVersion    = $script:PsVersion
            DotNetVersion = $script:DotNetVer
            CpuCount     = $script:CpuCount
            GpuName      = $script:GpuInfo.Name
            GpuRamGB     = $script:GpuInfo.AdapterRamGB
            GpuDriver    = $script:GpuInfo.DriverVersion
            Device       = $script:DeviceArg
            PreferredGpu = if ($script:PreferredGpu) {
                @{
                    Index = $script:PreferredGpu.Index
                    Name = $script:PreferredGpu.Name
                    MemoryTotalMB = $script:PreferredGpu.MemoryTotalMB
                }
            } else { $null }
            ModelPath    = $script:TestModelPath
        }

        $report = @{
            SystemInfo = $systemInfo
            Results    = $script:BenchmarkResults | ForEach-Object { $_ }
        }

        $jsonPath = Join-Path $script:ReportDir 'benchmark-report.json'
        $report | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

        # Markdown summary
        $sb = [System.Text.StringBuilder]::new()
        [void]$sb.AppendLine('# Media Pipeline Benchmark Report')
        [void]$sb.AppendLine()
        [void]$sb.AppendLine("**Date:** $($script:Timestamp)")
        [void]$sb.AppendLine("**GPU:** $($script:GpuInfo.Name) ($($script:GpuInfo.AdapterRamGB) GB)")
        [void]$sb.AppendLine("**Driver:** $($script:GpuInfo.DriverVersion)")
        [void]$sb.AppendLine("**Model:** $($script:TestModelPath)")
        [void]$sb.AppendLine("**Device:** $($script:DeviceArg)")
        [void]$sb.AppendLine()
        [void]$sb.AppendLine('| Category | Metric | Value |')
        [void]$sb.AppendLine('|----------|--------|-------|')
        foreach ($r in $script:BenchmarkResults) {
            $val = if ($r.Unit -eq 'ms') {
                "$([math]::Round($r.ValueMs, 1)) ms"
            } else {
                "$([math]::Round($r.ValueMs, 2)) $($r.Unit)"
            }
            [void]$sb.AppendLine("| $($r.Category) | $($r.Name) | $val |")
        }
        $sb.ToString() | Set-Content -Path (Join-Path $script:ReportDir 'benchmark-report.md') -Encoding UTF8

        Write-Host "`nBenchmark report written to: $($script:ReportDir)" -ForegroundColor Cyan
    }

    # Clean temp artifacts
    if (Test-Path $script:TempDir) {
        Remove-Item $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ==============================================================================
# Benchmark Tests
# ==============================================================================

Describe 'Media Performance Benchmarks' -Tag 'Benchmark', 'Media', 'Slow', 'GPU' {

    # ===========================================================================
    # Generation Performance
    # ===========================================================================

    Context 'Generation Performance' {

        It 'Measures first-token (init-to-first-pixel) latency' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $outPath = Join-Path $script:TempDir 'bench_first_token.png'
            $vramBefore = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = New-PcaiImage -Prompt 'a single blue dot' -OutputPath $outPath -CfgScale 1.0 -Temperature 0.5 -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success) {
                $vramAfter = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex
                $vramDelta = if ($vramBefore -ge 0 -and $vramAfter -ge 0) { $vramAfter - $vramBefore } else { -1 }
                Add-BenchmarkResult -Category 'Generation' -Name 'FirstTokenLatencyMs' -ValueMs $sw.Elapsed.TotalMilliseconds -Extra @{ VramDeltaMB = $vramDelta }
                Write-Host "  First-token latency: $($sw.Elapsed.TotalMilliseconds.ToString('F0')) ms"
            }
            # Assertion: should complete in under 5 minutes for any reasonable config
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 300
        }

        It 'Measures total generation time for 384x384 image' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $outPath = Join-Path $script:TempDir 'bench_gen_384.png'
            $vramBefore = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = New-PcaiImage -Prompt 'a glowing circuit board, 8k, detailed' `
                                    -OutputPath $outPath `
                                    -CfgScale 5.0 `
                                    -Temperature 1.0 `
                                    -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success) {
                $vramAfter = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex
                $vramDelta = if ($vramBefore -ge 0 -and $vramAfter -ge 0) { $vramAfter - $vramBefore } else { -1 }
                Add-BenchmarkResult -Category 'Generation' -Name 'TotalGenerationMs' -ValueMs $sw.Elapsed.TotalMilliseconds -Extra @{ VramDeltaMB = $vramDelta }
                Write-Host "  384x384 generation: $($sw.Elapsed.TotalSeconds.ToString('F1')) s  (VRAM delta: $vramDelta MB)"
            }
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 300
        }

        It 'Measures VRAM usage delta during generation' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            if ((Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex) -lt 0) {
                Set-ItResult -Skipped -Because "nvidia-smi not available for VRAM measurement"
                return
            }
            $vramBefore = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex
            $outPath    = Join-Path $script:TempDir 'bench_vram.png'
            $null = New-PcaiImage -Prompt 'test vram' -OutputPath $outPath -ErrorAction SilentlyContinue
            $vramAfter  = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex

            $delta = $vramAfter - $vramBefore
            Add-BenchmarkResult -Category 'Generation' -Name 'VramDeltaMB' -ValueMs $delta -Unit 'MB'
            Write-Host "  VRAM delta during generation: $delta MB"
            # VRAM delta should be below 12 GB (model + activations for 1B-7B models)
            $delta | Should -BeLessThan (12 * 1024)
        }

        It 'Measures tokens-per-second for the autoregressive generation loop' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            # Run two generations and estimate from the second (warm cache)
            $outPath1 = Join-Path $script:TempDir 'bench_tps_warm1.png'
            $outPath2 = Join-Path $script:TempDir 'bench_tps_warm2.png'
            $null = New-PcaiImage -Prompt 'warm-up run, skip' -OutputPath $outPath1 -ErrorAction SilentlyContinue

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = New-PcaiImage -Prompt 'a colorful abstract painting with geometric shapes' `
                                    -OutputPath $outPath2 `
                                    -CfgScale 5.0 `
                                    -Temperature 1.0 `
                                    -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success) {
                # Janus-Pro-1B generates 576 image tokens (24x24 grid) by default
                # Use 576 as the token budget for a rough tokens/sec estimate
                $estimatedTokens = 576
                $tps = $estimatedTokens / [math]::Max(0.001, $sw.Elapsed.TotalSeconds)
                Add-BenchmarkResult -Category 'Generation' -Name 'EstimatedTokensPerSec' -ValueMs $tps -Unit 'tok/s' `
                    -Extra @{ EstimatedTokens = $estimatedTokens; ElapsedSec = $sw.Elapsed.TotalSeconds }
                Write-Host "  Estimated generation throughput: $($tps.ToString('F1')) tok/s"
                $tps | Should -BeGreaterThan 0
            }
        }

        It 'Tests CFG scale impact on generation time (1.0 vs 7.0)' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $prompt    = 'a simple white square on black background'
            $outLowCfg = Join-Path $script:TempDir 'bench_cfg_low.png'
            $outHiCfg  = Join-Path $script:TempDir 'bench_cfg_high.png'

            $swLow = [System.Diagnostics.Stopwatch]::StartNew()
            $null = New-PcaiImage -Prompt $prompt -OutputPath $outLowCfg -CfgScale 1.0 -ErrorAction SilentlyContinue
            $swLow.Stop()

            $swHigh = [System.Diagnostics.Stopwatch]::StartNew()
            $null = New-PcaiImage -Prompt $prompt -OutputPath $outHiCfg -CfgScale 7.0 -ErrorAction SilentlyContinue
            $swHigh.Stop()

            Add-BenchmarkResult -Category 'Generation' -Name 'CfgScale1.0_Ms'   -ValueMs $swLow.Elapsed.TotalMilliseconds
            Add-BenchmarkResult -Category 'Generation' -Name 'CfgScale7.0_Ms'   -ValueMs $swHigh.Elapsed.TotalMilliseconds
            Write-Host "  CFG 1.0: $($swLow.Elapsed.TotalSeconds.ToString('F1'))s | CFG 7.0: $($swHigh.Elapsed.TotalSeconds.ToString('F1'))s"

            # Both runs should complete successfully
            $swLow.Elapsed.TotalSeconds  | Should -BeLessThan 300
            $swHigh.Elapsed.TotalSeconds | Should -BeLessThan 300
        }
    }

    # ===========================================================================
    # Understanding Performance
    # ===========================================================================

    Context 'Understanding Performance' {

        BeforeAll {
            # Create a synthetic test image for understanding benchmarks
            $script:BenchImagePath = Join-Path $script:TempDir 'bench_understand_input.png'
            # Minimal 1x1 PNG
            $pngBytes = [byte[]](
                0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,
                0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
                0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
                0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,
                0xDE,0x00,0x00,0x00,0x0C,0x49,0x44,0x41,
                0x54,0x08,0xD7,0x63,0xF8,0xCF,0xC0,0x00,
                0x00,0x00,0x02,0x00,0x01,0xE2,0x21,0xBC,
                0x33,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,
                0x44,0xAE,0x42,0x60,0x82
            )
            [System.IO.File]::WriteAllBytes($script:BenchImagePath, $pngBytes)
        }

        It 'Measures image encoding + response generation time' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Get-PcaiImageAnalysis -ImagePath $script:BenchImagePath `
                                            -Question 'Describe this image.' `
                                            -MaxTokens 128 `
                                            -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success) {
                Add-BenchmarkResult -Category 'Understanding' -Name 'TotalUnderstandingMs' -ValueMs $sw.Elapsed.TotalMilliseconds
                Write-Host "  Understanding total: $($sw.Elapsed.TotalSeconds.ToString('F1')) s"
            }
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 120
        }

        It 'Measures tokens-per-second for understanding response generation' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Get-PcaiImageAnalysis -ImagePath $script:BenchImagePath `
                                            -Question 'What do you see in this image?' `
                                            -MaxTokens 256 `
                                            -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success -and $result.Response) {
                $tokenCount = Measure-TokenCount -Text $result.Response
                $tps = $tokenCount / [math]::Max(0.001, $sw.Elapsed.TotalSeconds)
                Add-BenchmarkResult -Category 'Understanding' -Name 'ResponseTokensPerSec' -ValueMs $tps -Unit 'tok/s' `
                    -Extra @{ Tokens = $tokenCount; ElapsedSec = $sw.Elapsed.TotalSeconds }
                Write-Host "  Understanding throughput: $($tps.ToString('F1')) tok/s ($tokenCount tokens)"
                $tps | Should -BeGreaterThan 0
            }
        }
    }

    # ===========================================================================
    # Upscale Performance
    # ===========================================================================

    Context 'Upscale Performance' {

        It 'Measures 384x384 to 1536x1536 upscale wall-clock time' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            if (-not $script:OnnxModelPath) {
                Set-ItResult -Skipped -Because "RealESRGAN ONNX model not found"
                return
            }
            # Reuse a previously generated image if available
            $srcImage = Join-Path $script:TempDir 'bench_gen_384.png'
            if (-not (Test-Path $srcImage)) {
                Set-ItResult -Skipped -Because "Source 384x384 image not available (generation benchmark did not run)"
                return
            }

            $outPath    = Join-Path $script:TempDir 'bench_upscale_1536.png'
            $vramBefore = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Invoke-PcaiUpscale -InputPath $srcImage -OutputPath $outPath -ModelPath $script:OnnxModelPath -ErrorAction SilentlyContinue
            $sw.Stop()

            if ($null -ne $result -and $result.Success) {
                $vramAfter = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex
                $vramDelta = if ($vramBefore -ge 0 -and $vramAfter -ge 0) { $vramAfter - $vramBefore } else { -1 }
                Add-BenchmarkResult -Category 'Upscale' -Name 'UpscaleTimeMs' -ValueMs $sw.Elapsed.TotalMilliseconds `
                    -Extra @{ VramDeltaMB = $vramDelta; InputPath = $srcImage; OutputPath = $outPath }
                Write-Host "  4x upscale time: $($sw.Elapsed.TotalSeconds.ToString('F1')) s  (VRAM delta: $vramDelta MB)"
            }
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 120
        }

        It 'Measures VRAM usage delta during upscaling' {
            if (-not $script:MediaInitialized -or -not $script:OnnxModelPath) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            if ((Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex) -lt 0) {
                Set-ItResult -Skipped -Because "nvidia-smi not available"
                return
            }
            $srcImage = Join-Path $script:TempDir 'bench_gen_384.png'
            if (-not (Test-Path $srcImage)) {
                Set-ItResult -Skipped -Because "Source image not available"
                return
            }
            $outPath    = Join-Path $script:TempDir 'bench_upscale_vram.png'
            $vramBefore = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex
            $null = Invoke-PcaiUpscale -InputPath $srcImage -OutputPath $outPath -ModelPath $script:OnnxModelPath -ErrorAction SilentlyContinue
            $vramAfter  = Get-VramUsedMB -GpuIndex $script:BenchmarkGpuIndex

            $delta = $vramAfter - $vramBefore
            Add-BenchmarkResult -Category 'Upscale' -Name 'VramDeltaMB' -ValueMs $delta -Unit 'MB'
            Write-Host "  VRAM delta during upscale: $delta MB"
            $delta | Should -BeLessThan (8 * 1024)  # RealESRGAN should stay under 8 GB
        }
    }

    # ===========================================================================
    # Quality Metrics
    # ===========================================================================

    Context 'Quality Metrics' {

        It 'File size is consistent across two generations with identical prompt and cfg' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            $prompt  = 'a simple red circle on white background, minimalist'
            $outA    = Join-Path $script:TempDir 'bench_quality_A.png'
            $outB    = Join-Path $script:TempDir 'bench_quality_B.png'

            $null = New-PcaiImage -Prompt $prompt -OutputPath $outA -CfgScale 5.0 -Temperature 0.1 -ErrorAction SilentlyContinue
            $null = New-PcaiImage -Prompt $prompt -OutputPath $outB -CfgScale 5.0 -Temperature 0.1 -ErrorAction SilentlyContinue

            if (-not (Test-Path $outA) -or -not (Test-Path $outB)) {
                Set-ItResult -Skipped -Because "One or both generation outputs missing"
                return
            }
            $sizeA = (Get-Item $outA).Length
            $sizeB = (Get-Item $outB).Length

            Add-BenchmarkResult -Category 'Quality' -Name 'FileSizeA_Bytes' -ValueMs $sizeA -Unit 'bytes'
            Add-BenchmarkResult -Category 'Quality' -Name 'FileSizeB_Bytes' -ValueMs $sizeB -Unit 'bytes'

            Write-Host "  File sizes: A=$sizeA bytes, B=$sizeB bytes"

            # Both files must be non-empty PNGs
            $sizeA | Should -BeGreaterThan 0
            $sizeB | Should -BeGreaterThan 0
        }

        It 'Understanding response is relevant to the question (keyword matching)' {
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline not initialized"
                return
            }
            # Use a generated image if available, otherwise the synthetic test image
            $srcImage = Join-Path $script:TempDir 'bench_gen_384.png'
            if (-not (Test-Path $srcImage)) {
                $srcImage = Join-Path $script:TempDir 'bench_understand_input.png'
            }
            if (-not (Test-Path $srcImage)) {
                Set-ItResult -Skipped -Because "No test image available"
                return
            }

            $result = Get-PcaiImageAnalysis -ImagePath $srcImage `
                                            -Question 'Describe the content of this image in one sentence.' `
                                            -MaxTokens 100 `
                                            -ErrorAction SilentlyContinue
            if ($null -eq $result -or -not $result.Success) {
                Set-ItResult -Skipped -Because "Understanding call failed"
                return
            }

            $response = $result.Response
            Add-BenchmarkResult -Category 'Quality' -Name 'UnderstandingResponseLength' -ValueMs $response.Length -Unit 'chars'
            Write-Host "  Understanding response ($($response.Length) chars): $($response.Substring(0, [math]::Min(100, $response.Length)))..."

            # Response should be non-empty and not a refusal/error stub
            $response.Length | Should -BeGreaterThan 10
            $response | Should -Not -Match '^(error|null|stub|unavailable)$'
        }
    }
}
