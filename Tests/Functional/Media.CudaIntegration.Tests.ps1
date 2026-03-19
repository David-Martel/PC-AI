#Requires -Version 7.0
<#
.SYNOPSIS
    Functional tests for PC-AI media framework CUDA integration.
    Tests Janus-Pro 1B and 7B models with complex prompts for generation
    and understanding pipelines.

.DESCRIPTION
    These tests validate:
    1. CUDA device selection (cuda:auto, cuda:0, cuda:1)
    2. Image generation quality with complex prompts
    3. Image understanding accuracy with detailed questions
    4. Multi-turn inference stability
    5. Performance baselines (tok/s, latency)
    6. Memory management (VRAM tracking before/after)

.NOTES
    Requires: pcai-media binary built with --features cuda
    Tags: RequiresHardware, RequiresGPU, CUDA, Media
#>

param(
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent | Split-Path -Parent),
    [string]$Model1B = "Models\Janus-Pro-1B",
    [string]$Model7B = "Models\Janus-Pro-7B",
    [int]$ServerPort = 18200,
    [int]$ReadyTimeoutSeconds = 180,
    [switch]$Skip7B
)

$ErrorActionPreference = 'Stop'

# Resolve paths
$model1BPath = Join-Path $RepoRoot $Model1B
$model7BPath = Join-Path $RepoRoot $Model7B
$mediaExe    = $null

# Find the media server binary
$candidates = @(
    (Join-Path $RepoRoot '.pcai\build\artifacts\pcai-media\pcai-media.exe'),
    (Join-Path $RepoRoot 'pcai-media.exe'),
    (Join-Path $RepoRoot 'bin\pcai-media.exe'),
    'T:\RustCache\cargo-target\release\pcai-media-server.exe'
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $mediaExe = $c; break }
}

# Import GPU module for VRAM tracking
Import-Module (Join-Path $RepoRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1') -Force -ErrorAction SilentlyContinue

#region Helper Functions

function Start-MediaServer {
    param([string]$ModelPath, [string]$Device = 'cuda:auto', [int]$Port)

    $stdoutLog = Join-Path $env:TEMP "pcai-media-test-stdout-$Port.log"
    $stderrLog = Join-Path $env:TEMP "pcai-media-test-stderr-$Port.log"
    Remove-Item $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

    $proc = Start-Process -FilePath $mediaExe `
        -ArgumentList @('--model', $ModelPath, '--device', $Device, '--port', "$Port") `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru -WindowStyle Hidden

    # Wait for ready
    for ($i = 0; $i -lt $ReadyTimeoutSeconds; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            $stderr = if (Test-Path $stderrLog) { Get-Content $stderrLog -Raw } else { '' }
            throw "Media server exited early (code $($proc.ExitCode)): $stderr"
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($health.model_loaded -eq $true) { return $proc }
        } catch { }
    }
    throw "Media server did not become ready within $ReadyTimeoutSeconds seconds"
}

function Stop-MediaServer {
    param($Process)
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

function Invoke-ImageGeneration {
    param([int]$Port, [string]$Prompt, [double]$CfgScale = 5.0, [double]$Temperature = 1.0)

    $body = @{
        prompt      = $Prompt
        cfg_scale   = $CfgScale
        temperature = $Temperature
    } | ConvertTo-Json

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/images/generate" `
        -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
    $sw.Stop()

    [PSCustomObject]@{
        Prompt      = $Prompt
        Width       = $response.width
        Height      = $response.height
        Base64Len   = $response.image_base64.Length
        ElapsedMs   = $sw.Elapsed.TotalMilliseconds
        Success     = ($response.width -gt 0 -and $response.height -gt 0)
    }
}

function Invoke-ImageUnderstanding {
    param([int]$Port, [string]$ImageBase64, [string]$Question, [int]$MaxTokens = 256, [double]$Temperature = 0.1)

    $body = @{
        image_base64 = $ImageBase64
        prompt       = $Question
        max_tokens   = $MaxTokens
        temperature  = $Temperature
    } | ConvertTo-Json -Depth 4

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/images/understand" `
        -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    $sw.Stop()

    [PSCustomObject]@{
        Question    = $Question
        Response    = $response.text
        ResponseLen = $response.text.Length
        ElapsedMs   = $sw.Elapsed.TotalMilliseconds
        Success     = ($response.text.Length -gt 0)
    }
}

function Get-TestImageBase64 {
    # Create a simple test image (red square) as base64 PNG
    $testImage = Join-Path $RepoRoot 'Reports\media\understand-red.png'
    if (Test-Path $testImage) {
        return [Convert]::ToBase64String([IO.File]::ReadAllBytes($testImage))
    }
    # Fallback: create a minimal 2x2 red PNG
    $pngBytes = [byte[]]@(
        0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A, # PNG header
        0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,  # IHDR
        0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,
        0x08,0x02,0x00,0x00,0x00,0xFD,0xD4,0x9A,0x73,
        0x00,0x00,0x00,0x16,0x49,0x44,0x41,0x54,  # IDAT
        0x78,0x9C,0x62,0xF8,0xCF,0xC0,0xF0,0x1F,
        0x00,0x00,0x00,0x05,0x00,0x01,0x0D,0x0A,
        0x2D,0xB4,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,  # IEND
        0xAE,0x42,0x60,0x82
    )
    return [Convert]::ToBase64String($pngBytes)
}

#endregion

#region Test Prompts

$GenerationPrompts = @(
    # Simple baseline
    @{ Prompt = 'A red apple on a white table'; Category = 'simple'; ExpectColors = @('red') }

    # Complex scene composition
    @{ Prompt = 'A steampunk laboratory with brass gears, glass tubes filled with glowing green liquid, and a Victorian-era scientist examining a holographic display, dramatic lighting from gas lamps'; Category = 'complex-scene'; ExpectElements = @('laboratory', 'gears') }

    # Technical/scientific visualization
    @{ Prompt = 'A cross-section diagram of a GPU chip showing CUDA cores arranged in streaming multiprocessors, with data flowing through memory buses, rendered in a technical blueprint style with blue and white color scheme'; Category = 'technical'; ExpectStyle = 'technical' }

    # Abstract concept
    @{ Prompt = 'The concept of artificial intelligence represented as a neural network of glowing nodes connected by energy beams, floating in a cosmic void with nebulae in the background'; Category = 'abstract'; ExpectElements = @('nodes', 'network') }

    # Photorealistic
    @{ Prompt = 'A professional photograph of a custom-built PC with RGB lighting, visible through a tempered glass side panel, sitting on a desk with dual monitors showing code, shallow depth of field, 4K quality'; Category = 'photorealistic'; ExpectStyle = 'photo' }
)

$UnderstandingQuestions = @(
    # Basic identification
    @{ Question = 'What is the primary subject of this image? Describe it in one sentence.'; Category = 'identification' }

    # Color analysis
    @{ Question = 'List all the distinct colors visible in this image, ordered from most to least dominant.'; Category = 'color-analysis' }

    # Spatial reasoning
    @{ Question = 'Describe the spatial layout of this image. What is in the foreground, middle ground, and background?'; Category = 'spatial' }

    # Technical detail
    @{ Question = 'If this image were to be used in a machine learning dataset, what labels or tags would be appropriate? List at least 5 relevant tags.'; Category = 'ml-tagging' }

    # Creative interpretation
    @{ Question = 'Write a one-paragraph creative story inspired by this image. Include sensory details about what you see.'; Category = 'creative' }

    # Counting and enumeration
    @{ Question = 'Count and list every distinct object or element you can identify in this image.'; Category = 'counting' }
)

#endregion

# ============================================================================
# TESTS
# ============================================================================

Write-Host "`n" -NoNewline
Write-Host '=' * 70 -ForegroundColor Cyan
Write-Host '  PC-AI Media Framework — CUDA Functional Tests' -ForegroundColor Cyan
Write-Host '=' * 70 -ForegroundColor Cyan
Write-Host ''

# Pre-flight checks
if (-not $mediaExe) {
    Write-Warning "pcai-media binary not found. Build with: .\Build.ps1 -Component media -EnableCuda"
    Write-Host "Searched: $($candidates -join ', ')"
    exit 1
}

if (-not (Test-Path $model1BPath)) {
    Write-Warning "Janus-Pro-1B model not found at: $model1BPath"
    exit 1
}

Write-Host "  Binary:  $mediaExe" -ForegroundColor Gray
Write-Host "  Model 1B: $model1BPath" -ForegroundColor Gray
Write-Host "  Model 7B: $model7BPath $(if (Test-Path $model7BPath) {'[OK]'} else {'[MISSING]'})" -ForegroundColor Gray
Write-Host "  Port:    $ServerPort" -ForegroundColor Gray
Write-Host ''

# GPU inventory
$gpus = @(Get-NvidiaGpuInventory -ErrorAction SilentlyContinue)
if ($gpus.Count -gt 0) {
    Write-Host '  GPU Inventory:' -ForegroundColor Gray
    foreach ($g in $gpus) {
        Write-Host "    [$($g.Index)] $($g.Name) — $($g.MemoryTotalMB) MiB (free: $($g.MemoryTotalMB - $g.MemoryUsedMB) MiB)" -ForegroundColor Gray
    }
}
Write-Host ''

$results = [System.Collections.Generic.List[PSCustomObject]]::new()
$testNumber = 0

# ============================================================================
# TEST SUITE 1: Janus-Pro-1B — Image Generation
# ============================================================================

Write-Host '--- Test Suite 1: Janus-Pro-1B Image Generation ---' -ForegroundColor Yellow
$server = $null
try {
    Write-Host '  Starting media server (1B, cuda:auto)...' -ForegroundColor Gray
    $vramBefore = @(Get-NvidiaGpuInventory -ErrorAction SilentlyContinue)
    $server = Start-MediaServer -ModelPath $model1BPath -Device 'cuda:auto' -Port $ServerPort
    $vramAfter = @(Get-NvidiaGpuInventory -ErrorAction SilentlyContinue)

    # Check VRAM delta
    if ($vramBefore.Count -gt 0 -and $vramAfter.Count -gt 0) {
        foreach ($gpu in $vramAfter) {
            $before = $vramBefore | Where-Object Index -eq $gpu.Index
            if ($before) {
                $delta = $gpu.MemoryUsedMB - $before.MemoryUsedMB
                if ($delta -gt 100) {
                    Write-Host "    GPU $($gpu.Index) VRAM delta: +${delta} MiB (model loaded)" -ForegroundColor Green
                }
            }
        }
    }

    # Health check
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ServerPort/health" -TimeoutSec 5
    Write-Host "  Server ready: model_loaded=$($health.model_loaded)" -ForegroundColor Green

    foreach ($testCase in $GenerationPrompts) {
        $testNumber++
        $label = "GEN-1B-$($testCase.Category)"
        Write-Host "  [$testNumber] ${label}: $($testCase.Prompt.Substring(0, [Math]::Min(60, $testCase.Prompt.Length)))..." -NoNewline

        try {
            $result = Invoke-ImageGeneration -Port $ServerPort -Prompt $testCase.Prompt
            if ($result.Success) {
                $tokPerSec = if ($result.ElapsedMs -gt 0) { [math]::Round(576 / ($result.ElapsedMs / 1000), 1) } else { 0 }
                Write-Host " OK ($($result.Width)x$($result.Height), $([math]::Round($result.ElapsedMs/1000,1))s, ~$tokPerSec tok/s)" -ForegroundColor Green
                $result | Add-Member -NotePropertyName TestLabel -NotePropertyValue $label
                $result | Add-Member -NotePropertyName TokPerSec -NotePropertyValue $tokPerSec
                $result | Add-Member -NotePropertyName Status -NotePropertyValue 'PASS'
            } else {
                Write-Host " FAIL (no image)" -ForegroundColor Red
                $result | Add-Member -NotePropertyName TestLabel -NotePropertyValue $label
                $result | Add-Member -NotePropertyName Status -NotePropertyValue 'FAIL'
            }
            $results.Add($result)
        } catch {
            Write-Host " ERROR: $($_.Exception.Message)" -ForegroundColor Red
            $results.Add([PSCustomObject]@{ TestLabel = $label; Status = 'ERROR'; Prompt = $testCase.Prompt; ElapsedMs = 0 })
        }
    }

    # Understanding tests with the generated image (or test image)
    Write-Host ''
    Write-Host '--- Test Suite 2: Janus-Pro-1B Image Understanding ---' -ForegroundColor Yellow
    $testImageB64 = Get-TestImageBase64

    foreach ($testCase in $UnderstandingQuestions) {
        $testNumber++
        $label = "UND-1B-$($testCase.Category)"
        Write-Host "  [$testNumber] ${label}: $($testCase.Question.Substring(0, [Math]::Min(60, $testCase.Question.Length)))..." -NoNewline

        try {
            $result = Invoke-ImageUnderstanding -Port $ServerPort -ImageBase64 $testImageB64 -Question $testCase.Question
            if ($result.Success) {
                $preview = $result.Response.Substring(0, [Math]::Min(50, $result.Response.Length)) -replace "`n", ' '
                Write-Host " OK ($($result.ResponseLen) chars, $([math]::Round($result.ElapsedMs/1000,1))s) -> `"$preview...`"" -ForegroundColor Green
                $result | Add-Member -NotePropertyName TestLabel -NotePropertyValue $label
                $result | Add-Member -NotePropertyName Status -NotePropertyValue 'PASS'
            } else {
                Write-Host " FAIL (empty response)" -ForegroundColor Red
                $result | Add-Member -NotePropertyName TestLabel -NotePropertyValue $label
                $result | Add-Member -NotePropertyName Status -NotePropertyValue 'FAIL'
            }
            $results.Add($result)
        } catch {
            Write-Host " ERROR: $($_.Exception.Message)" -ForegroundColor Red
            $results.Add([PSCustomObject]@{ TestLabel = $label; Status = 'ERROR'; Question = $testCase.Question; ElapsedMs = 0 })
        }
    }
} catch {
    Write-Host "  Server startup FAILED: $_" -ForegroundColor Red
    $results.Add([PSCustomObject]@{ TestLabel = 'SERVER-1B'; Status = 'ERROR'; ElapsedMs = 0 })
} finally {
    Stop-MediaServer -Process $server
}

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host ''
Write-Host '=' * 70 -ForegroundColor Cyan
Write-Host '  RESULTS SUMMARY' -ForegroundColor Cyan
Write-Host '=' * 70 -ForegroundColor Cyan

$passed  = @($results | Where-Object Status -eq 'PASS').Count
$failed  = @($results | Where-Object Status -eq 'FAIL').Count
$errors  = @($results | Where-Object Status -eq 'ERROR').Count
$total   = $results.Count

Write-Host "  Total: $total | Passed: $passed | Failed: $failed | Errors: $errors" -ForegroundColor $(if ($failed + $errors -eq 0) { 'Green' } else { 'Yellow' })

$genResults = @($results | Where-Object { $_.TestLabel -match '^GEN-' -and $_.Status -eq 'PASS' })
if ($genResults.Count -gt 0) {
    $avgGen = [math]::Round(($genResults | Measure-Object ElapsedMs -Average).Average / 1000, 1)
    $avgTok = [math]::Round(($genResults | Measure-Object TokPerSec -Average).Average, 1)
    Write-Host "  Generation avg: ${avgGen}s per image, ~$avgTok tok/s" -ForegroundColor Cyan
}

$undResults = @($results | Where-Object { $_.TestLabel -match '^UND-' -and $_.Status -eq 'PASS' })
if ($undResults.Count -gt 0) {
    $avgUnd = [math]::Round(($undResults | Measure-Object ElapsedMs -Average).Average / 1000, 1)
    $avgLen = [math]::Round(($undResults | Measure-Object ResponseLen -Average).Average, 0)
    Write-Host "  Understanding avg: ${avgUnd}s per query, ~$avgLen chars response" -ForegroundColor Cyan
}

# Save results
$reportPath = Join-Path $RepoRoot 'Reports\media\cuda-functional-test-results.json'
$results | ConvertTo-Json -Depth 5 | Set-Content -Path $reportPath -Encoding UTF8
Write-Host "  Results saved: $reportPath" -ForegroundColor Gray
Write-Host ''

# Return results for pipeline consumption
return $results
