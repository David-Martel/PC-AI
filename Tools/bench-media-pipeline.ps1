#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Benchmark the pcai_media Rust FFI pipeline (Janus-Pro text-to-image).

.DESCRIPTION
    Tests the native Rust pipeline via PcaiNative.dll P/Invoke bridge:
      1. Load pcai_media.dll + PcaiNative.dll
      2. Initialize backend (CPU or CUDA)
      3. Load Janus-Pro model (1B or 7B)
      4. Generate images with timing
      5. Compare sync vs async generation
      6. Report throughput metrics

.PARAMETER ModelPath
    Path to the Janus-Pro model directory.

.PARAMETER Device
    Compute device: "cpu", "cuda:0", "cuda:1". Default: "cuda:0".

.PARAMETER GpuLayers
    GPU layer offload count. -1 = all layers. Default: -1.

.PARAMETER Prompts
    Array of prompts to benchmark. Uses defaults if not specified.

.PARAMETER OutputDir
    Directory for generated images. Default: Reports/media/bench/.

.EXAMPLE
    .\Tools\bench-media-pipeline.ps1 -ModelPath Models/Janus-Pro-1B -Device cuda:0
#>
[CmdletBinding()]
param(
    [string]$ModelPath = "Models/Janus-Pro-1B",
    [string]$Device = "cuda:0",
    [int]$GpuLayers = -1,
    [string[]]$Prompts,
    [string]$OutputDir = "Reports/media/bench"
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent

# Default prompts if not provided
if (-not $Prompts) {
    $Prompts = @(
        "A glowing blue circuit board floating in space, digital art, 8k"
        "A majestic wolf standing on a mountain peak at sunset, oil painting"
        "A futuristic robot reading a book in a cozy library, anime style"
    )
}

# Resolve paths
$ModelFullPath = Join-Path $ProjectRoot $ModelPath
$OutputFullDir = Join-Path $ProjectRoot $OutputDir
$BinDir = Join-Path $ProjectRoot "bin"
$NativeDll = Join-Path $BinDir "PcaiNative.dll"
$MediaDll = Join-Path $BinDir "pcai_media.dll"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PCAI MEDIA PIPELINE BENCHMARK" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Validate prerequisites
$missing = @()
if (-not (Test-Path $NativeDll)) { $missing += "PcaiNative.dll ($NativeDll)" }
if (-not (Test-Path $MediaDll)) { $missing += "pcai_media.dll ($MediaDll)" }
if (-not (Test-Path $ModelFullPath)) { $missing += "Model directory ($ModelFullPath)" }

if ($missing.Count -gt 0) {
    Write-Host "  MISSING PREREQUISITES:" -ForegroundColor Red
    foreach ($m in $missing) {
        Write-Host "    - $m" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

# Check for safetensors weights (required by Rust candle pipeline)
$safetensors = Get-ChildItem $ModelFullPath -Filter "*.safetensors" -ErrorAction SilentlyContinue
if (-not $safetensors) {
    Write-Host "  WARNING: No .safetensors files found in $ModelFullPath" -ForegroundColor Yellow
    Write-Host "  The Rust candle pipeline requires safetensors format." -ForegroundColor Yellow
    Write-Host "  Run: python Tools/convert-sharded-weights.py $ModelPath" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "  Model:      $ModelPath"
Write-Host "  Device:     $Device"
Write-Host "  GPU Layers: $GpuLayers"
Write-Host "  Prompts:    $($Prompts.Count)"
Write-Host "  Output:     $OutputDir"
Write-Host "  DLL:        $MediaDll"
Write-Host "  Weights:    $($safetensors.Name -join ', ')"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Path $OutputFullDir -Force | Out-Null

# Load PcaiNative.dll
Write-Host "  Loading PcaiNative.dll..." -ForegroundColor Gray
try {
    Add-Type -Path $NativeDll -ErrorAction Stop
    Write-Host "  [OK] PcaiNative.dll loaded" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Cannot load PcaiNative.dll: $_" -ForegroundColor Red
    exit 1
}

# Check if media module is available
$available = [PcaiNative.MediaModule]::IsAvailable
Write-Host "  [OK] pcai_media.dll available: $available" -ForegroundColor Green

# Initialize backend
Write-Host ""
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  PHASE 1: Initialize Backend ($Device)" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

$initTimer = [System.Diagnostics.Stopwatch]::StartNew()
$initResult = [PcaiNative.MediaModule]::pcai_media_init($Device)
$initTimer.Stop()

if ($initResult -ne 0) {
    $err = [PcaiNative.MediaModule]::GetLastError()
    Write-Host "  [FAIL] Init failed (code $initResult): $err" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Backend initialized in $($initTimer.ElapsedMilliseconds)ms" -ForegroundColor Green

# Load model
Write-Host ""
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  PHASE 2: Load Model" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

$loadTimer = [System.Diagnostics.Stopwatch]::StartNew()
$loadResult = [PcaiNative.MediaModule]::pcai_media_load_model($ModelFullPath, $GpuLayers)
$loadTimer.Stop()

if ($loadResult -ne 0) {
    $err = [PcaiNative.MediaModule]::GetLastError()
    Write-Host "  [FAIL] Model load failed (code $loadResult): $err" -ForegroundColor Red
    [PcaiNative.MediaModule]::pcai_media_shutdown()
    exit 1
}
Write-Host "  [OK] Model loaded in $([math]::Round($loadTimer.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green

# Generate images
Write-Host ""
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  PHASE 3: Generate Images (sync)" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

$results = @()
$cfgScale = 5.0
$temperature = 1.0

for ($i = 0; $i -lt $Prompts.Count; $i++) {
    $prompt = $Prompts[$i]
    $filename = "bench_rust_$($i + 1).png"
    $outputPath = Join-Path $OutputFullDir $filename

    Write-Host ""
    Write-Host "  [$($i + 1)/$($Prompts.Count)] Generating: '$($prompt.Substring(0, [Math]::Min(60, $prompt.Length)))...'"

    $genTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $genError = [PcaiNative.MediaModule]::GenerateImage($prompt, $outputPath, $cfgScale, $temperature)
    $genTimer.Stop()

    if ($genError) {
        Write-Host "    [FAIL] $genError" -ForegroundColor Red
        $results += @{
            Prompt = $prompt
            Status = "FAILED"
            Error = $genError
            TimeMs = $genTimer.ElapsedMilliseconds
        }
    } else {
        $fileInfo = Get-Item $outputPath
        $tokPerSec = 576.0 / $genTimer.Elapsed.TotalSeconds
        Write-Host "    [OK] $filename ($($fileInfo.Length) bytes) in $([math]::Round($genTimer.Elapsed.TotalSeconds, 1))s ($([math]::Round($tokPerSec, 1)) tok/s)" -ForegroundColor Green
        $results += @{
            Prompt = $prompt
            Status = "OK"
            File = $filename
            SizeBytes = $fileInfo.Length
            TimeMs = $genTimer.ElapsedMilliseconds
            TokPerSec = [math]::Round($tokPerSec, 1)
        }
    }
}

# Async generation test
Write-Host ""
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  PHASE 4: Async Generation Test" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

$asyncPrompt = "A crystal clear lake reflecting snow-capped mountains, photorealistic"
$asyncOutput = Join-Path $OutputFullDir "bench_rust_async.png"

Write-Host "  Submitting async request..."
$asyncTimer = [System.Diagnostics.Stopwatch]::StartNew()
$requestId = [PcaiNative.MediaModule]::pcai_media_generate_image_async($asyncPrompt, $cfgScale, $temperature, $asyncOutput)

if ($requestId -lt 0) {
    Write-Host "  [FAIL] Async submit failed: $([PcaiNative.MediaModule]::GetLastError())" -ForegroundColor Red
} else {
    Write-Host "  Request ID: $requestId"

    # Poll until complete
    $pollCount = 0
    while ($true) {
        Start-Sleep -Milliseconds 500
        $pollCount++
        $pollResult = [PcaiNative.MediaModule]::pcai_media_poll_result($requestId)

        switch ($pollResult.Status) {
            0 { Write-Host "    Poll $pollCount`: pending..." -ForegroundColor Gray }
            1 { Write-Host "    Poll $pollCount`: running..." -ForegroundColor Gray }
            2 {
                $asyncTimer.Stop()
                if ($pollResult.Text -ne [IntPtr]::Zero) {
                    [PcaiNative.MediaModule]::pcai_media_free_string($pollResult.Text)
                }
                $asyncTokPerSec = 576.0 / $asyncTimer.Elapsed.TotalSeconds
                Write-Host "    [OK] Async complete in $([math]::Round($asyncTimer.Elapsed.TotalSeconds, 1))s ($([math]::Round($asyncTokPerSec, 1)) tok/s)" -ForegroundColor Green
                break
            }
            3 {
                $asyncTimer.Stop()
                $errMsg = "unknown"
                if ($pollResult.Text -ne [IntPtr]::Zero) {
                    $errMsg = [System.Runtime.InteropServices.Marshal]::PtrToStringUTF8($pollResult.Text)
                    [PcaiNative.MediaModule]::pcai_media_free_string($pollResult.Text)
                }
                Write-Host "    [FAIL] Async failed: $errMsg" -ForegroundColor Red
                break
            }
            default {
                $asyncTimer.Stop()
                Write-Host "    [FAIL] Unknown status: $($pollResult.Status)" -ForegroundColor Red
                break
            }
        }

        if ($pollResult.Status -ge 2 -or $pollResult.Status -lt 0) { break }
        if ($pollCount -gt 600) {
            Write-Host "    [TIMEOUT] Exceeded 5 minute timeout" -ForegroundColor Red
            [PcaiNative.MediaModule]::pcai_media_cancel($requestId)
            break
        }
    }
}

# Shutdown
Write-Host ""
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Shutting down..." -ForegroundColor Gray
[PcaiNative.MediaModule]::pcai_media_shutdown()
Write-Host "  [OK] Backend shutdown complete" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  BENCHMARK RESULTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Model:       $ModelPath"
Write-Host "  Device:      $Device"
Write-Host "  Init time:   $($initTimer.ElapsedMilliseconds)ms"
Write-Host "  Load time:   $([math]::Round($loadTimer.Elapsed.TotalSeconds, 1))s"
Write-Host ""

$okResults = $results | Where-Object { $_.Status -eq "OK" }
if ($okResults.Count -gt 0) {
    $avgTime = ($okResults | Measure-Object -Property TimeMs -Average).Average / 1000
    $avgTps = ($okResults | Measure-Object -Property TokPerSec -Average).Average
    $bestTps = ($okResults | Measure-Object -Property TokPerSec -Maximum).Maximum

    Write-Host "  Generations: $($okResults.Count) / $($results.Count) succeeded"
    Write-Host "  Avg time:    $([math]::Round($avgTime, 1))s per image"
    Write-Host "  Avg tok/s:   $([math]::Round($avgTps, 1))"
    Write-Host "  Best tok/s:  $([math]::Round($bestTps, 1))"
    Write-Host ""

    foreach ($r in $okResults) {
        Write-Host "    $($r.File): $([math]::Round($r.TimeMs / 1000, 1))s ($($r.TokPerSec) tok/s)" -ForegroundColor Gray
    }
}

$failResults = $results | Where-Object { $_.Status -eq "FAILED" }
if ($failResults.Count -gt 0) {
    Write-Host ""
    Write-Host "  FAILURES:" -ForegroundColor Red
    foreach ($r in $failResults) {
        Write-Host "    - $($r.Error)" -ForegroundColor Red
    }
}

# Save JSON report
$report = @{
    timestamp = (Get-Date -Format "o")
    model = $ModelPath
    device = $Device
    gpuLayers = $GpuLayers
    initMs = $initTimer.ElapsedMilliseconds
    loadMs = $loadTimer.ElapsedMilliseconds
    results = $results
    avgTokPerSec = if ($okResults.Count -gt 0) { [math]::Round($avgTps, 1) } else { 0 }
    pipeline = "rust-candle-ffi"
}
$reportPath = Join-Path $OutputFullDir "benchmark-report.json"
$report | ConvertTo-Json -Depth 5 | Set-Content $reportPath -Encoding UTF8
Write-Host ""
Write-Host "  Report: $reportPath"
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
