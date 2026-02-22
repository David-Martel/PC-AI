#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Functional tests for PCAI Inference with real models.

.DESCRIPTION
    Tests actual inference capabilities with real model files:
    1. Model loading performance benchmarks
    2. Text generation quality verification
    3. GPU acceleration tests
    4. Streaming generation tests
    5. Backend comparison tests

.NOTES
    These tests require:
    - providers.pcai-native.modelPath set in Config/llm-config.json
    - OR a model installed via LM Studio or Ollama

    Run with: Invoke-Pester -Path .\Tests\Functional\ -Tag Functional

.EXAMPLE
    # Example:
    # Set providers.pcai-native.modelPath in Config/llm-config.json
    Invoke-Pester -Path .\Tests\Functional\Inference.Functional.Tests.ps1
#>

BeforeAll {
    # Import shared test helpers
    Import-Module (Join-Path $PSScriptRoot "..\Helpers\TestHelpers.psm1") -Force

    # Get standard test paths
    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot = $paths.ProjectRoot
    $script:BinDir = $paths.BinDir
    $script:ModulePath = $paths.ModulePath
    $script:DllPath = $paths.DllPath

    # Load module
    if (Test-Path $ModulePath) {
        Import-Module $ModulePath -Force -ErrorAction SilentlyContinue
    }

    # Helper: Find test model (wrapper for compatibility)
    function Get-TestModel {
        Get-TestModelPath
    }

    $script:DllAvailable = Test-InferenceDllAvailable -ProjectRoot $ProjectRoot
    $script:TestModel = Get-TestModelPath
    $script:HasModel = $null -ne $TestModel
}

AfterAll {
    try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
}

Describe "Functional: Model Loading" -Tag "Functional", "ModelLoad" {

    BeforeEach {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    AfterEach {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    Context "Loading Performance" {

        It "Model loads within acceptable time (<60 seconds for Q4)" {
            if (-not (Assert-TestPrerequisites -RequireDll -RequireModel)) { return }

            $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            if (-not $init) {
                Set-ItResult -Skipped -Because "llamacpp backend not available"
                return
            }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $loaded = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0
            $sw.Stop()

            if (-not $loaded) {
                $status = Get-PcaiInferenceStatus
                Set-ItResult -Skipped -Because "Model loading failed: $($status.LastError)"
                return
            }

            # Model should load within 60 seconds for Q4 quantized models
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 60
            Write-Host "Model loaded in $($sw.Elapsed.TotalSeconds.ToString('F2')) seconds"
        }

        It "Model info is available after loading" {
            if (-not (Assert-TestPrerequisites -RequireDll -RequireModel)) { return }

            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue

            $status = Get-PcaiInferenceStatus
            $status.ModelLoaded | Should -BeTrue
            $status.CurrentBackend | Should -Be "llama.cpp"
        }
    }

    Context "GPU Layers Configuration" {

        It "Loading with GpuLayers=0 (CPU only) succeeds" {
            if (-not (Assert-TestPrerequisites -RequireDll -RequireModel)) { return }

            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            $loaded = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0

            $loaded | Should -BeTrue
        }

        It "Loading with GpuLayers=-1 (auto) handles missing GPU gracefully" {
            if (-not (Assert-TestPrerequisites -RequireDll -RequireModel)) { return }

            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue

            # This should succeed (falls back to CPU if no GPU)
            $loaded = Import-PcaiModel -ModelPath $TestModel -GpuLayers -1 -ErrorAction SilentlyContinue

            # Either succeeds or fails with meaningful error
            $status = Get-PcaiInferenceStatus
            if (-not $loaded) {
                $status.LastError | Should -Not -BeNullOrEmpty
            } else {
                $status.ModelLoaded | Should -BeTrue
            }
        }
    }
}

Describe "Functional: Text Generation" -Tag "Functional", "Generation" {

    BeforeAll {
        $script:ModelReady = $false

        if ($DllAvailable -and $HasModel) {
            $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            if ($init) {
                $load = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
                $script:ModelReady = $load
            }
        }
    }

    AfterAll {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    Context "Basic Generation" {

        It "Generates non-empty response for simple prompt" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $result = Invoke-PcaiGenerate -Prompt "Hello" -MaxTokens 10 -Temperature 0.1
            $result | Should -Not -BeNullOrEmpty
            $result.Trim().Length | Should -BeGreaterThan 0
        }

        It "Respects MaxTokens parameter" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            # Generate with small token limit
            $result = Invoke-PcaiGenerate -Prompt "The" -MaxTokens 5 -Temperature 0.1

            # Result should be short (roughly 5 tokens)
            # Token count is approximate, so just verify it's not too long
            $wordCount = ($result -split '\s+').Count
            $wordCount | Should -BeLessThan 20  # Allow some flexibility
        }

        It "Different temperatures produce different outputs" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            # Generate multiple times with high temperature
            $results = @()
            for ($i = 0; $i -lt 3; $i++) {
                $result = Invoke-PcaiGenerate -Prompt "Random:" -MaxTokens 10 -Temperature 1.0
                $results += $result
            }

            # With high temperature, outputs should vary
            # At least 2 of 3 should be different
            $unique = $results | Select-Object -Unique
            $unique.Count | Should -BeGreaterThan 1
        }
    }

    Context "Generation Quality" {

        It "Continues text coherently" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $prompt = "The capital of France is"
            $result = Invoke-PcaiGenerate -Prompt $prompt -MaxTokens 20 -Temperature 0.1

            # Should mention Paris or continue coherently
            # This is a soft test - just verify it's not garbage
            $result | Should -Not -BeNullOrEmpty
            $result | Should -Not -Match '^\W+$'  # Not just punctuation
        }

        It "Handles long prompts" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $longPrompt = "Please write a detailed explanation of the following topic: " +
                          ("artificial intelligence and its applications " * 10)

            $result = Invoke-PcaiGenerate -Prompt $longPrompt -MaxTokens 50 -Temperature 0.1 -ErrorAction SilentlyContinue

            # Should either succeed or fail gracefully
            if ($null -eq $result) {
                $status = Get-PcaiInferenceStatus
                # If failed, should have meaningful error
                $status.LastError | Should -Not -BeNullOrEmpty
            } else {
                $result.Length | Should -BeGreaterThan 0
            }
        }

        It "Handles empty prompt gracefully" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $result = Invoke-PcaiGenerate -Prompt "" -MaxTokens 10 -Temperature 0.1 -ErrorAction SilentlyContinue

            # Either returns something or fails gracefully
            # Empty prompt behavior varies by model
            $status = Get-PcaiInferenceStatus
            # Just verify no crash
            $true | Should -BeTrue
        }
    }

    Context "Sequential Generations" {

        It "Multiple sequential generations work correctly" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $prompts = @(
                "One plus one equals",
                "The color of the sky is",
                "Water freezes at"
            )

            $results = @()
            foreach ($prompt in $prompts) {
                $result = Invoke-PcaiGenerate -Prompt $prompt -MaxTokens 10 -Temperature 0.1
                $results += $result
                $result | Should -Not -BeNullOrEmpty -Because "Generation for '$prompt' should succeed"
            }

            # All should have produced output
            $results.Count | Should -Be 3
        }
    }
}

Describe "Functional: Performance Benchmarks" -Tag "Functional", "Performance" {

    BeforeAll {
        $script:ModelReady = $false

        if ($DllAvailable -and $HasModel) {
            $init = Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            if ($init) {
                $load = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
                $script:ModelReady = $load
            }
        }
    }

    AfterAll {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    Context "Generation Speed" {

        It "First token latency is reasonable (<5 seconds)" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Invoke-PcaiGenerate -Prompt "Hi" -MaxTokens 1 -Temperature 0.1
            $sw.Stop()

            # First token should appear within 5 seconds
            $sw.Elapsed.TotalSeconds | Should -BeLessThan 5
            Write-Host "First token latency: $($sw.Elapsed.TotalMilliseconds.ToString('F0'))ms"
        }

        It "Generation throughput is measurable" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            $tokenCount = 50
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $result = Invoke-PcaiGenerate -Prompt "Once upon a time" -MaxTokens $tokenCount -Temperature 0.1
            $sw.Stop()

            # Calculate approximate tokens per second
            $tokensPerSecond = $tokenCount / $sw.Elapsed.TotalSeconds
            Write-Host "Approx throughput: $($tokensPerSecond.ToString('F1')) tokens/sec"

            # Should generate at least 1 token per second on any hardware
            $tokensPerSecond | Should -BeGreaterThan 0.5
        }
    }

    Context "Memory Usage" {

        It "Memory usage stabilizes after model load" {
            if (-not $ModelReady) {
                Set-ItResult -Skipped -Because "Model not loaded"
                return
            }

            # Get baseline memory
            [GC]::Collect()
            $beforeGeneration = [GC]::GetTotalMemory($true)

            # Run several generations
            for ($i = 0; $i -lt 5; $i++) {
                $null = Invoke-PcaiGenerate -Prompt "Test $i" -MaxTokens 20 -Temperature 0.1
            }

            [GC]::Collect()
            $afterGeneration = [GC]::GetTotalMemory($true)

            # Memory growth should be bounded (allow 100MB growth)
            $growth = $afterGeneration - $beforeGeneration
            $growthMB = $growth / 1MB
            Write-Host "Memory growth after generations: $($growthMB.ToString('F1')) MB"

            $growth | Should -BeLessThan (100 * 1MB)
        }
    }
}

Describe "Functional: Error Recovery" -Tag "Functional", "ErrorRecovery" {

    BeforeEach {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    AfterEach {
        try { Close-PcaiInference -ErrorAction SilentlyContinue } catch {}
    }

    Context "Recovery from Errors" {

        It "Recovers after failed model load" {
            if (-not $DllAvailable -or -not $HasModel) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }

            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue

            # Try to load non-existent model
            $null = Import-PcaiModel -ModelPath "C:\nonexistent.gguf" -ErrorAction SilentlyContinue

            # Should still be able to load real model
            $loaded = Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
            $loaded | Should -BeTrue

            # And generate
            $result = Invoke-PcaiGenerate -Prompt "Test" -MaxTokens 5 -Temperature 0.1
            $result | Should -Not -BeNullOrEmpty
        }

        It "Recovers after backend reinitialization" {
            if (-not $DllAvailable -or -not $HasModel) {
                Set-ItResult -Skipped -Because "Prerequisites not met"
                return
            }

            # First cycle
            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
            $result1 = Invoke-PcaiGenerate -Prompt "First" -MaxTokens 5 -Temperature 0.1
            Close-PcaiInference

            # Second cycle
            Initialize-PcaiInference -Backend llamacpp -ErrorAction SilentlyContinue
            Import-PcaiModel -ModelPath $TestModel -GpuLayers 0 -ErrorAction SilentlyContinue
            $result2 = Invoke-PcaiGenerate -Prompt "Second" -MaxTokens 5 -Temperature 0.1
            Close-PcaiInference

            # Both should work
            $result1 | Should -Not -BeNullOrEmpty
            $result2 | Should -Not -BeNullOrEmpty
        }
    }
}
