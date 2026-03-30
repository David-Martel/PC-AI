#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    FFI integration tests for the pcai-media native library.

.DESCRIPTION
    Tests the FFI boundary between PowerShell and the native Rust + C# pipeline:
    1. DLL existence and loadability (pcai_media.dll, PcaiNative.dll)
    2. [PcaiNative.MediaModule] static member availability
    3. Lifecycle: init, shutdown, double-shutdown idempotency
    4. Error propagation: init with invalid device, generate/understand before model
    5. Model loading failure with non-existent path (tagged 'Slow')

    All tests skip gracefully when the native DLL has not been built.
    Build DLLs with: .\Build.ps1 -Component media

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Integration\FFI.Media.Tests.ps1 -Tag Integration,Media,FFI
    Run slow tests: Invoke-Pester -Path .\Tests\Integration\FFI.Media.Tests.ps1 -Tag Slow
#>

BeforeAll {
    # Import shared test helpers for path resolution
    Import-Module (Join-Path $PSScriptRoot '..\Helpers\TestHelpers.psm1') -Force

    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot   = $paths.ProjectRoot
    $script:BinDir        = $paths.BinDir
    $script:PcaiNativeDll = Join-Path $script:BinDir 'PcaiNative.dll'
    $script:MediaDll      = Join-Path $script:BinDir 'pcai_media.dll'

    # Attempt to load PcaiNative.dll and make [PcaiNative.MediaModule] available.
    $script:NativeDllLoaded = $false
    if (Test-Path $script:PcaiNativeDll) {
        # Prepend bin/ to PATH so the Rust DLL is resolvable by P/Invoke
        if ($env:PATH -notlike "*$script:BinDir*") {
            $env:PATH = "$script:BinDir;$env:PATH"
        }
        try {
            Add-Type -Path $script:PcaiNativeDll -ErrorAction Stop
            $script:NativeDllLoaded = $true
        } catch {
            Write-Warning "Failed to load PcaiNative.dll: $_"
        }
    }

    # Helper: skip if DLL not available
    function Skip-IfNoDll {
        if (-not $script:NativeDllLoaded) {
            Set-ItResult -Skipped -Because "PcaiNative.dll not built (run .\Build.ps1 -Component media)"
        }
    }
}

# ==============================================================================
# DLL Loading
# ==============================================================================

Describe 'Media FFI Integration' -Tag 'Integration', 'Media', 'FFI' {

    Context 'DLL Loading' {

        It 'pcai_media.dll exists in bin/' {
            if (-not (Test-Path $script:MediaDll)) {
                Set-ItResult -Skipped -Because "pcai_media.dll not built"
            } else {
                Test-Path $script:MediaDll | Should -BeTrue
            }
        }

        It 'PcaiNative.dll exists in bin/' {
            if (-not (Test-Path $script:PcaiNativeDll)) {
                Set-ItResult -Skipped -Because "PcaiNative.dll not built"
            } else {
                Test-Path $script:PcaiNativeDll | Should -BeTrue
            }
        }

        It 'PcaiNative.dll is a valid PE file (MZ header)' {
            if (-not (Test-Path $script:PcaiNativeDll)) {
                Set-ItResult -Skipped -Because "PcaiNative.dll not built"
                return
            }
            $header = [System.IO.File]::ReadAllBytes($script:PcaiNativeDll)[0..1]
            $header[0] | Should -Be 0x4D  # 'M'
            $header[1] | Should -Be 0x5A  # 'Z'
        }

        It 'pcai_media.dll is a valid PE file (MZ header)' {
            if (-not (Test-Path $script:MediaDll)) {
                Set-ItResult -Skipped -Because "pcai_media.dll not built"
                return
            }
            $header = [System.IO.File]::ReadAllBytes($script:MediaDll)[0..1]
            $header[0] | Should -Be 0x4D
            $header[1] | Should -Be 0x5A
        }

        It 'PcaiNative.dll loads into the current process' {
            Skip-IfNoDll
            # If we reach here the BeforeAll loaded it without error
            $script:NativeDllLoaded | Should -BeTrue
        }

        It '[PcaiNative.MediaModule] type is accessible after load' {
            Skip-IfNoDll
            { [PcaiNative.MediaModule] | Out-Null } | Should -Not -Throw
        }

        It 'MediaModule.IsAvailable returns true' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::IsAvailable | Should -BeTrue
        }
    }

    # ===========================================================================
    # Lifecycle
    # ===========================================================================

    Context 'Lifecycle' {

        AfterEach {
            # Always attempt shutdown to leave clean state
            if ($script:NativeDllLoaded) {
                try { [PcaiNative.MediaModule]::pcai_media_shutdown() } catch { }
            }
        }

        It 'Init succeeds on cpu device (returns 0)' {
            Skip-IfNoDll
            $result = [PcaiNative.MediaModule]::pcai_media_init('cpu')
            $result | Should -Be 0
        }

        It 'Init returns non-zero for invalid device string' {
            Skip-IfNoDll
            $result = [PcaiNative.MediaModule]::pcai_media_init('invalid_device_xyz')
            $result | Should -Not -Be 0
        }

        It 'Shutdown after successful init does not throw' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null
            { [PcaiNative.MediaModule]::pcai_media_shutdown() } | Should -Not -Throw
        }

        It 'Shutdown is idempotent (safe to call twice)' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null
            [PcaiNative.MediaModule]::pcai_media_shutdown()
            { [PcaiNative.MediaModule]::pcai_media_shutdown() } | Should -Not -Throw
        }

        It 'Shutdown before init does not throw (cold shutdown)' {
            Skip-IfNoDll
            # No init called — shutdown should be a safe no-op
            { [PcaiNative.MediaModule]::pcai_media_shutdown() } | Should -Not -Throw
        }

        It 'GetLastError returns null when there is no error after successful init' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null
            $err = [PcaiNative.MediaModule]::GetLastError()
            $err | Should -BeNullOrEmpty
        }

        It 'Multiple init / shutdown cycles succeed without leaking state' {
            Skip-IfNoDll
            for ($cycle = 0; $cycle -lt 3; $cycle++) {
                $initResult = [PcaiNative.MediaModule]::pcai_media_init('cpu')
                $initResult | Should -Be 0 -Because "Cycle ${cycle}: init should succeed"
                [PcaiNative.MediaModule]::pcai_media_shutdown()
            }
        }
    }

    # ===========================================================================
    # Model Loading
    # ===========================================================================

    Context 'Model Loading' {

        AfterEach {
            if ($script:NativeDllLoaded) {
                try { [PcaiNative.MediaModule]::pcai_media_shutdown() } catch { }
            }
        }

        It 'Load fails gracefully with a non-existent model path (returns non-zero)' -Tag 'Slow' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null

            $result = [PcaiNative.MediaModule]::pcai_media_load_model('C:\totally\nonexistent\Janus-Pro-9999B', 0)
            $result | Should -Not -Be 0
        }

        It 'GetLastError returns a non-empty message after a model load failure' -Tag 'Slow' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null
            [PcaiNative.MediaModule]::pcai_media_load_model('C:\totally\nonexistent\Janus-Pro-9999B', 0) | Out-Null

            $err = [PcaiNative.MediaModule]::GetLastError()
            $err | Should -Not -BeNullOrEmpty
        }

        It 'Load model before init returns error or zero (no crash)' {
            Skip-IfNoDll
            # Ensure clean state (no prior init)
            [PcaiNative.MediaModule]::pcai_media_shutdown()
            $result = [PcaiNative.MediaModule]::pcai_media_load_model('deepseek-ai/Janus-Pro-1B', 0)
            # After shutdown, load_model may return 0 (no-op) or non-zero (error).
            # The key assertion is that it does not crash.
            $result | Should -BeOfType [int]
        }
    }

    # ===========================================================================
    # Error Handling
    # ===========================================================================

    Context 'Error Handling' {

        AfterEach {
            if ($script:NativeDllLoaded) {
                try { [PcaiNative.MediaModule]::pcai_media_shutdown() } catch { }
            }
        }

        It 'GenerateImage fails before model is loaded (returns non-null error string)' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null

            $tempOut = Join-Path $env:TEMP 'pcai_ffi_test_gen.png'
            $result = [PcaiNative.MediaModule]::GenerateImage('test prompt', $tempOut, 5.0, 1.0)
            # Should return an error string (not null) or GetLastError should be populated
            $gotError = ($null -ne $result) -or ($null -ne [PcaiNative.MediaModule]::GetLastError())
            $gotError | Should -BeTrue
        }

        It 'UnderstandImage fails before model is loaded (returns null result)' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null

            $tempImg = Join-Path $env:TEMP 'pcai_ffi_test_img.png'
            [System.IO.File]::WriteAllBytes($tempImg, [byte[]]@(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
            try {
                $result = [PcaiNative.MediaModule]::UnderstandImage($tempImg, 'Describe this.', 64, 0.7)
                # Either null result or GetLastError populated — both indicate correct failure mode
                $gotError = ($null -eq $result) -or ($null -ne [PcaiNative.MediaModule]::GetLastError())
                $gotError | Should -BeTrue
            } finally {
                Remove-Item $tempImg -Force -ErrorAction SilentlyContinue
            }
        }

        It 'Error codes from init with bad device are meaningful integers (non-zero)' {
            Skip-IfNoDll
            $result = [PcaiNative.MediaModule]::pcai_media_init('bad_device_name')
            $result | Should -BeOfType [int]
            $result | Should -Not -Be 0
        }

        It 'pcai_media_generate_image_async returns negative request ID before model loaded' {
            Skip-IfNoDll
            [PcaiNative.MediaModule]::pcai_media_init('cpu') | Out-Null
            $tempOut = Join-Path $env:TEMP 'pcai_ffi_async.png'
            # Call the raw P/Invoke (prompt, cfgScale, temperature, outputPath) which returns a long request ID
            $id = [PcaiNative.MediaModule]::pcai_media_generate_image_async('test', 5.0, 1.0, $tempOut)
            $id | Should -BeLessThan 0
        }
    }
}
