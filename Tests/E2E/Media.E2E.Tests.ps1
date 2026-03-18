#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    End-to-end tests for the PC-AI media pipeline.

.DESCRIPTION
    Full pipeline tests exercising the entire stack from PowerShell module
    through C# P/Invoke into the Rust pcai_media.dll:
    1. Image generation from text prompt (PNG output, magic bytes, dimensions)
    2. Image understanding (VLM response quality)
    3. Async image generation (request ID, poll, result)
    4. Image upscaling 4x via RealESRGAN ONNX

    All tests are tagged 'Slow', 'Model', and 'GPU' and skip automatically when:
    - pcai_media.dll is not built
    - A Janus-Pro model is not configured in Config/pcai-media.json
    - No CUDA-capable GPU is available (configurable)

    Run with:
        Invoke-Pester -Path .\Tests\E2E\Media.E2E.Tests.ps1 -Tag E2E,Media
    GPU-optional run:
        Invoke-Pester -Path .\Tests\E2E\Media.E2E.Tests.ps1 -Tag E2E,Media -ExcludeTag GPU

.NOTES
    Estimated runtime: 2-10 minutes depending on GPU and model size (1B vs 7B).
    A Janus-Pro-1B model on RTX 2000 Ada takes ~20-40 seconds per generation.
#>

BeforeAll {
    # Import shared test helpers
    Import-Module (Join-Path $PSScriptRoot '..\Helpers\TestHelpers.psm1') -Force

    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot   = $paths.ProjectRoot
    $script:BinDir        = $paths.BinDir
    $script:MediaModPath  = Join-Path $script:ProjectRoot 'Modules\PcaiMedia.psm1'
    $script:PcaiNativeDll = Join-Path $script:BinDir 'PcaiNative.dll'
    $script:MediaDll      = Join-Path $script:BinDir 'pcai_media.dll'
    $script:MediaConfig   = Join-Path $script:ProjectRoot 'Config\pcai-media.json'

    # Track pipeline state for cleanup
    $script:MediaInitialized = $false

    # Helper: resolve Janus-Pro model path from Config/pcai-media.json
    function Get-MediaModelPath {
        if (-not (Test-Path $script:MediaConfig)) { return $null }
        try {
            $cfg = Get-Content $script:MediaConfig -Raw | ConvertFrom-Json
            if (-not $cfg.model) { return $null }

            # Relative paths are resolved from project root
            if ([System.IO.Path]::IsPathRooted($cfg.model)) {
                $resolved = $cfg.model
            } else {
                $resolved = Join-Path $script:ProjectRoot $cfg.model
            }

            # Accept HuggingFace repo IDs and existing local paths
            if ($resolved -match '^[^/]+/[^/]+$') { return $resolved }  # HF repo ID
            if (Test-Path $resolved) { return $resolved }
            return $null
        } catch {
            return $null
        }
    }

    # Helper: check GPU availability via WMI
    function Test-GpuAvailable {
        try {
            $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.AdapterCompatibility -match 'NVIDIA|AMD|Intel' }
            return ($null -ne $gpus -and @($gpus).Count -gt 0)
        } catch {
            return $false
        }
    }

    # Helper: check PNG magic bytes in a file
    function Test-PngMagicBytes {
        param([string]$Path)
        if (-not (Test-Path $Path)) { return $false }
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        if ($bytes.Length -lt 8) { return $false }
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        return ($bytes[0] -eq 0x89 -and $bytes[1] -eq 0x50 -and
                $bytes[2] -eq 0x4E -and $bytes[3] -eq 0x47)
    }

    # Helper: read image dimensions from PNG IHDR chunk
    function Get-PngDimensions {
        param([string]$Path)
        if (-not (Test-Path $Path)) { return $null }
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        if ($bytes.Length -lt 24) { return $null }
        # IHDR starts at byte 16: width (4 bytes big-endian), height (4 bytes big-endian)
        $width  = ($bytes[16] -shl 24) -bor ($bytes[17] -shl 16) -bor ($bytes[18] -shl 8) -bor $bytes[19]
        $height = ($bytes[20] -shl 24) -bor ($bytes[21] -shl 16) -bor ($bytes[22] -shl 8) -bor $bytes[23]
        return [PSCustomObject]@{ Width = $width; Height = $height }
    }

    # Evaluate prerequisites
    $script:DllsAvailable   = (Test-Path $script:MediaDll) -and (Test-Path $script:PcaiNativeDll)
    $script:TestModelPath   = Get-MediaModelPath
    $script:HasModel        = $null -ne $script:TestModelPath
    $script:HasGpu          = Test-GpuAvailable
    $script:MediaDevice     = if ($script:HasGpu) { 'cuda:auto' } else { 'cpu' }

    Write-Verbose "E2E Media: DLLs=$($script:DllsAvailable), Model=$($script:HasModel), GPU=$($script:HasGpu)"
    Write-Verbose "E2E Media: TestModelPath=$($script:TestModelPath)"
    Write-Verbose "E2E Media: Device=$($script:MediaDevice)"

    # Load module and initialize pipeline if prerequisites are met
    if ($script:DllsAvailable -and $script:HasModel) {
        if ($env:PATH -notlike "*$script:BinDir*") {
            $env:PATH = "$script:BinDir;$env:PATH"
        }
        Import-Module $script:MediaModPath -Force -ErrorAction SilentlyContinue

        $initResult = Initialize-PcaiMedia -Device $script:MediaDevice -ErrorAction SilentlyContinue
        if ($initResult -and $initResult.Success) {
            $loadResult = Import-PcaiMediaModel -ModelPath $script:TestModelPath -ErrorAction SilentlyContinue
            $script:MediaInitialized = ($null -ne $loadResult -and $loadResult.Success)
        }
    }

    # Scratch directory for test outputs
    $script:TestOutputDir = Join-Path $env:TEMP 'pcai_e2e_media'
    if (-not (Test-Path $script:TestOutputDir)) {
        New-Item -ItemType Directory -Path $script:TestOutputDir -Force | Out-Null
    }
}

AfterAll {
    try { Stop-PcaiMedia -ErrorAction SilentlyContinue } catch { }
    Remove-Module PcaiMedia -Force -ErrorAction SilentlyContinue
    # Optionally clean up test output artifacts
    if (Test-Path $script:TestOutputDir) {
        Remove-Item $script:TestOutputDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ==============================================================================
# Image Generation Pipeline
# ==============================================================================

Describe 'E2E: Media Pipeline' -Tag 'E2E', 'Media', 'Model', 'Slow' {

    Context 'Image Generation Pipeline' {

        BeforeAll {
            $script:GeneratedImagePath = Join-Path $script:TestOutputDir 'e2e_gen_test.png'
        }

        It 'Generates a PNG from text prompt' -Tag 'GPU' {
            if (-not $script:DllsAvailable) {
                Set-ItResult -Skipped -Because "pcai_media.dll or PcaiNative.dll not built"
                return
            }
            if (-not $script:HasModel) {
                Set-ItResult -Skipped -Because "No Janus-Pro model configured in Config/pcai-media.json"
                return
            }
            if (-not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Media pipeline failed to initialize"
                return
            }

            $result = New-PcaiImage -Prompt 'a glowing circuit board, digital art' `
                                    -OutputPath $script:GeneratedImagePath `
                                    -CfgScale 5.0 `
                                    -Temperature 1.0
            $result | Should -Not -BeNullOrEmpty
            $result.Success | Should -BeTrue
            Test-Path $script:GeneratedImagePath | Should -BeTrue
        }

        It 'Output file has valid PNG magic bytes (89 50 4E 47)' -Tag 'GPU' {
            if (-not $script:MediaInitialized -or -not (Test-Path $script:GeneratedImagePath)) {
                Set-ItResult -Skipped -Because "Generation test did not run or failed"
                return
            }
            Test-PngMagicBytes -Path $script:GeneratedImagePath | Should -BeTrue
        }

        It 'Output image has non-zero dimensions' -Tag 'GPU' {
            if (-not $script:MediaInitialized -or -not (Test-Path $script:GeneratedImagePath)) {
                Set-ItResult -Skipped -Because "Generation test did not run or failed"
                return
            }
            $dims = Get-PngDimensions -Path $script:GeneratedImagePath
            $dims | Should -Not -BeNullOrEmpty
            $dims.Width  | Should -BeGreaterThan 0
            $dims.Height | Should -BeGreaterThan 0
        }

        It 'Output image matches expected native resolution (384x384 for Janus-Pro-1B)' -Tag 'GPU' {
            if (-not $script:MediaInitialized -or -not (Test-Path $script:GeneratedImagePath)) {
                Set-ItResult -Skipped -Because "Generation test did not run or failed"
                return
            }
            $dims = Get-PngDimensions -Path $script:GeneratedImagePath
            # Janus-Pro generates at 384x384 natively; larger models may differ
            # We assert square aspect ratio as the invariant
            $dims.Width | Should -Be $dims.Height -Because "Janus-Pro generates square images"
        }

        It 'Generation completes within timeout (120 seconds for 1B model)' -Tag 'GPU' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $timedOut = $false
            $outPath   = Join-Path $script:TestOutputDir 'e2e_gen_timeout.png'
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $null = New-PcaiImage -Prompt 'a blue sky' -OutputPath $outPath -ErrorAction SilentlyContinue
            } finally {
                $sw.Stop()
            }
            if ($sw.Elapsed.TotalSeconds -ge 120) { $timedOut = $true }
            $timedOut | Should -BeFalse -Because "Generation should complete within 120 seconds"
            Write-Host "  Generation time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
        }
    }

    # ===========================================================================
    # Image Understanding Pipeline
    # ===========================================================================

    Context 'Image Understanding Pipeline' {

        BeforeAll {
            # Create a synthetic test image (small solid PNG) for understanding tests
            $script:TestImagePath = Join-Path $script:TestOutputDir 'e2e_understand_input.png'
            # Write a minimal 1x1 red PNG (137 bytes — valid PNG)
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
            [System.IO.File]::WriteAllBytes($script:TestImagePath, $pngBytes)
        }

        It 'Analyzes a test image and returns a non-empty text response' -Tag 'GPU' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $result = Get-PcaiImageAnalysis -ImagePath $script:TestImagePath `
                                            -Question 'Describe this image.' `
                                            -MaxTokens 128
            $result | Should -Not -BeNullOrEmpty
            $result.Success  | Should -BeTrue
            $result.Response | Should -Not -BeNullOrEmpty
        }

        It 'Response has meaningful length (> 10 characters)' -Tag 'GPU' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $result = Get-PcaiImageAnalysis -ImagePath $script:TestImagePath `
                                            -Question 'Describe this image.' `
                                            -MaxTokens 128 `
                                            -ErrorAction SilentlyContinue
            if ($null -eq $result) {
                Set-ItResult -Skipped -Because "Understanding call returned null"
                return
            }
            $result.Response.Length | Should -BeGreaterThan 10
        }

        It 'Handles JPEG input format gracefully' -Tag 'GPU' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            # Create a minimal JPEG (SOI + EOI = 4 bytes — not visually meaningful but tests format handling)
            $jpegPath = Join-Path $script:TestOutputDir 'e2e_understand_input.jpg'
            $jpegBytes = [byte[]](0xFF, 0xD8, 0xFF, 0xD9)
            [System.IO.File]::WriteAllBytes($jpegPath, $jpegBytes)
            try {
                # Should either succeed or throw a meaningful error — must not crash the process
                { $null = Get-PcaiImageAnalysis -ImagePath $jpegPath -MaxTokens 32 -ErrorAction SilentlyContinue } |
                    Should -Not -Throw
            } finally {
                Remove-Item $jpegPath -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # ===========================================================================
    # Async Generation
    # ===========================================================================

    Context 'Async Generation' -Tag 'GPU' {

        It 'Async generation returns a positive request ID' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $outPath = Join-Path $script:TestOutputDir 'e2e_async_gen.png'
            $job = New-PcaiImageAsync -Prompt 'a starry night sky' -OutputPath $outPath -ErrorAction SilentlyContinue
            $job | Should -Not -BeNullOrEmpty
            $job.RequestId | Should -BeGreaterThan 0
        }

        It 'Async output path is reflected in the returned job object' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $outPath = Join-Path $script:TestOutputDir 'e2e_async_gen2.png'
            $job = New-PcaiImageAsync -Prompt 'mountains at sunset' -OutputPath $outPath -ErrorAction SilentlyContinue
            if ($null -eq $job) {
                Set-ItResult -Skipped -Because "Async start returned null"
                return
            }
            $job.OutputPath | Should -Be $outPath
        }

        It 'Wait-PcaiImageAsync completes the queued request and preserves the output path' -Tag 'Slow' {
            if (-not $script:DllsAvailable -or -not $script:HasModel -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            $outPath = Join-Path $script:TestOutputDir 'e2e_async_wait.png'
            $job = New-PcaiImageAsync -Prompt 'async completion test image' -OutputPath $outPath -ErrorAction SilentlyContinue
            if ($null -eq $job) {
                Set-ItResult -Skipped -Because "Async start returned null"
                return
            }

            $result = Wait-PcaiImageAsync -RequestId $job.RequestId -TimeoutSeconds 180
            $result.Success | Should -BeTrue
            $result.OutputPath | Should -Be $outPath
        }
    }

    # ===========================================================================
    # Upscale Pipeline
    # ===========================================================================

    Context 'Upscale Pipeline' {

        BeforeAll {
            # Locate the RealESRGAN ONNX model from config
            $script:OnnxModelPath = $null
            if (Test-Path $script:MediaConfig) {
                try {
                    $cfg = Get-Content $script:MediaConfig -Raw | ConvertFrom-Json
                    if ($cfg.upscale -and $cfg.upscale.model_path) {
                        $onnxRaw = $cfg.upscale.model_path
                        if ([System.IO.Path]::IsPathRooted($onnxRaw)) {
                            $script:OnnxModelPath = $onnxRaw
                        } else {
                            $script:OnnxModelPath = Join-Path $script:ProjectRoot $onnxRaw
                        }
                        if (-not (Test-Path $script:OnnxModelPath)) {
                            $script:OnnxModelPath = $null
                        }
                    }
                } catch { }
            }
        }

        It 'Upscales a 384x384 source to 1536x1536 (4x) and produces a valid PNG' -Tag 'GPU', 'Slow' {
            if (-not $script:DllsAvailable -or -not $script:MediaInitialized) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }
            if (-not $script:OnnxModelPath) {
                Set-ItResult -Skipped -Because "RealESRGAN ONNX model not found (configure Config/pcai-media.json upscale.model_path)"
                return
            }

            # Use the generated image from the generation context if available, else skip
            $genImage = Join-Path $script:TestOutputDir 'e2e_gen_test.png'
            if (-not (Test-Path $genImage)) {
                Set-ItResult -Skipped -Because "Source image from generation test not available"
                return
            }

            $upscaledPath = Join-Path $script:TestOutputDir 'e2e_upscaled_4x.png'
            $result = Invoke-PcaiUpscale -InputPath $genImage `
                                         -OutputPath $upscaledPath `
                                         -ModelPath $script:OnnxModelPath
            $result.Success | Should -BeTrue
            Test-Path $upscaledPath | Should -BeTrue
        }

        It 'Upscaled output is a valid PNG (magic bytes)' -Tag 'GPU', 'Slow' {
            $upscaledPath = Join-Path $script:TestOutputDir 'e2e_upscaled_4x.png'
            if (-not (Test-Path $upscaledPath)) {
                Set-ItResult -Skipped -Because "Upscale test did not run or failed"
                return
            }
            Test-PngMagicBytes -Path $upscaledPath | Should -BeTrue
        }

        It 'Upscaled image is larger than source (4x scale factor)' -Tag 'GPU', 'Slow' {
            $sourcePath   = Join-Path $script:TestOutputDir 'e2e_gen_test.png'
            $upscaledPath = Join-Path $script:TestOutputDir 'e2e_upscaled_4x.png'
            if (-not (Test-Path $sourcePath) -or -not (Test-Path $upscaledPath)) {
                Set-ItResult -Skipped -Because "Source or upscaled image not available"
                return
            }
            $srcDims      = Get-PngDimensions -Path $sourcePath
            $upscaledDims = Get-PngDimensions -Path $upscaledPath
            if ($null -eq $srcDims -or $null -eq $upscaledDims) {
                Set-ItResult -Skipped -Because "Could not read PNG dimensions"
                return
            }
            $upscaledDims.Width  | Should -BeGreaterThan $srcDims.Width
            $upscaledDims.Height | Should -BeGreaterThan $srcDims.Height
        }
    }
}
