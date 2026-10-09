#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Unit tests for Initialize-PcaiMediaFFI (private function in PcaiMedia).

.DESCRIPTION
    Tests the DLL-loading bootstrap function responsible for loading PcaiNative.dll
    and making [PcaiNative.MediaModule] available.  All file-system and assembly-load
    operations are isolated via Pester mocks executed inside InModuleScope, so no
    actual DLLs are required.

    Scenarios covered:
    - Returns $false when PcaiNative.dll is not present on disk
    - Returns $false (with warning) when Assembly.LoadFrom throws
    - Returns $true when a valid DLL is loaded successfully
    - Does not modify script:Initialized (that is Initialize-PcaiMedia's job)

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Unit\Initialize-PcaiMediaFFI.Tests.ps1 -Tag Unit,Media,FFI
#>

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ModulePath  = Join-Path $script:ProjectRoot 'Modules\PcaiMedia.psm1'

    # Pre-load the PcaiNative stub so the module body does not try to resolve a
    # real assembly during import.
    if (-not ([System.Management.Automation.PSTypeName]'PcaiNative.MediaModule').Type) {
        Add-Type -TypeDefinition @'
namespace PcaiNative {
    public struct PcaiMediaAsyncResult { public int Status; public System.IntPtr Text; }
    public static class MediaModule {
        public static int    pcai_media_init(string d)                                     { return 0; }
        public static void   pcai_media_shutdown()                                          { }
        public static int    pcai_media_load_model(string p, int g)                        { return 0; }
        public static string GenerateImage(string p, string o, float c, float t)           { return null; }
        public static string UnderstandImage(string i, string q, uint m, float t)          { return "stub"; }
        public static System.Threading.Tasks.Task<string> GenerateImageNativeAsync(
            string p, string o, float c, float t, int ms,
            System.Threading.CancellationToken ct) {
            return System.Threading.Tasks.Task.FromResult<string>(null);
        }
        public static string UpscaleImage(string m, string i, string o)                    { return null; }
        public static void   pcai_media_free_string(System.IntPtr p)                       { }
        public static int    pcai_media_cancel(long r)                                     { return 0; }
        public static string GetLastError()                                                 { return null; }
        public static bool   IsAvailable                                                    { get { return true; } }
    }
}
'@ -ErrorAction SilentlyContinue
    }

    Import-Module $script:ModulePath -Force -ErrorAction Stop
}

AfterAll {
    Remove-Module PcaiMedia -Force -ErrorAction SilentlyContinue
}

Describe 'Initialize-PcaiMediaFFI' -Tag 'Unit', 'Media', 'FFI', 'Portable' {

    # Reset module state between every test
    BeforeEach {
        $script:SavedNativeBundleRoot = $env:PCAI_NATIVE_BUNDLE_ROOT
        $env:PCAI_NATIVE_BUNDLE_ROOT = $null
        InModuleScope PcaiMedia {
            $script:Initialized  = $false
            $script:ModelLoaded  = $false
            $script:CurrentModel = $null
        }
    }
    AfterEach { $env:PCAI_NATIVE_BUNDLE_ROOT = $script:SavedNativeBundleRoot }

    # -----------------------------------------------------------------------
    # DLL absent
    # -----------------------------------------------------------------------
    Context 'When PcaiNative.dll does not exist on disk' {

        It 'Returns $false' {
            InModuleScope PcaiMedia {
                Mock Test-Path  { return $false }       -ParameterFilter { $LiteralPath -match 'PcaiNative\.dll' }
                Mock Get-PcaiProjectRoot { return 'C:\FakeRoot' }
                Mock Write-Warning {}
                Initialize-PcaiMediaFFI | Should -BeFalse
                Should -Invoke Write-Warning -Times 0
            }
        }

        It 'Does not set script:Initialized to true' {
            InModuleScope PcaiMedia {
                Mock Test-Path  { return $false }       -ParameterFilter { $LiteralPath -match 'PcaiNative\.dll' }
                Mock Get-PcaiProjectRoot { return 'C:\FakeRoot' }
                $null = Initialize-PcaiMediaFFI
                $script:Initialized | Should -BeFalse
            }
        }
    }

    # -----------------------------------------------------------------------
    # Assembly load fails
    # -----------------------------------------------------------------------
    Context 'When the DLL exists but Assembly.LoadFrom throws' {

        BeforeEach {
            $script:InvalidBridgeRoot = Join-Path $TestDrive 'invalid-bridge'
            $bin = Join-Path $script:InvalidBridgeRoot 'bin'
            [void](New-Item -ItemType Directory -Path $bin -Force)
            Set-Content -LiteralPath (Join-Path $bin 'PcaiNative.dll') -Value 'invalid managed assembly'
            InModuleScope PcaiMedia -Parameters @{ Root = $script:InvalidBridgeRoot } {
                param($Root)
                $script:InvalidBridgeRoot = $Root
                Mock Get-PcaiProjectRoot { $script:InvalidBridgeRoot }
                Mock Get-PcaiMediaLoadedBridgePath { $null }
            }
        }

        It 'Returns $false' {
            InModuleScope PcaiMedia {
                Mock Write-Warning {}
                # Resolve the real fixture file; the actual assembly loader rejects its invalid bytes.
                $result = Initialize-PcaiMediaFFI
                $result | Should -BeFalse
            }
        }

        It 'Emits a Write-Warning message that mentions the load failure' {
            InModuleScope PcaiMedia {
                Mock Write-Warning {}
                $null = Initialize-PcaiMediaFFI
                Should -Invoke Write-Warning -Times 1 -ParameterFilter { $Message -match 'Failed to select native media bridge.*(Bad IL|BadImage|format|assembly)' }
            }
        }

        It 'Does not set script:Initialized to true after a failed load' {
            InModuleScope PcaiMedia {
                Mock Write-Warning {}
                $null = Initialize-PcaiMediaFFI
                $script:Initialized | Should -BeFalse
            }
        }
    }

    # -----------------------------------------------------------------------
    # DLL loads successfully
    # -----------------------------------------------------------------------
    Context 'When a real DLL can be loaded from a temp directory' {

        It 'Returns $true when Assembly.LoadFrom succeeds' {
            # Build a minimal valid .NET assembly in TEMP so LoadFrom does not throw.
            $tempDir  = Join-Path $env:TEMP "PcaiFfiTest_$(New-Guid)"
            $binDir   = Join-Path $tempDir 'bin'
            New-Item -ItemType Directory -Path $binDir -Force | Out-Null
            $dllPath  = Join-Path $binDir 'PcaiNative.dll'

            Add-Type -TypeDefinition @'
namespace PcaiFfiTestDummy { public class DummyClass { } }
'@ -OutputAssembly $dllPath -OutputType Library

            try {
                $probe = Join-Path $tempDir 'load-bridge.ps1'
                @'
param($ModulePath, $Root, $DllPath)
$ErrorActionPreference = 'Stop'
$env:PCAI_NATIVE_BUNDLE_ROOT = $null
Import-Module $ModulePath -Force
$success = & (Get-Module PcaiMedia) {
    param($SelectedRoot)
    $script:ModulePath = Join-Path $SelectedRoot 'Modules'
    if ((Get-PcaiProjectRoot) -ne $SelectedRoot) { throw 'Fixture project root was not selected.' }
    Initialize-PcaiMediaFFI
} $Root
if (-not $success) { throw 'Fresh-process managed loading failed.' }
$assembly = [AppDomain]::CurrentDomain.GetAssemblies() | Where-Object Location -eq $DllPath | Select-Object -First 1
if (-not $assembly) { throw 'The initializer did not load the selected fixture assembly.' }
Write-Output 'fresh-bridge-loaded'
'@ | Set-Content -LiteralPath $probe
                $shell = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
                $result = & $shell -NoLogo -NoProfile -File $probe $script:ModulePath $tempDir $dllPath
                $LASTEXITCODE | Should -Be 0
                $result | Should -Contain 'fresh-bridge-loaded'
            } finally {
                Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # -----------------------------------------------------------------------
    # Initialize-PcaiMedia integration (depends on Initialize-PcaiMediaFFI returning $false)
    # -----------------------------------------------------------------------
    Context 'Initialize-PcaiMedia throws when Initialize-PcaiMediaFFI returns false' {

        It 'Propagates a descriptive error about the missing DLL' {
            InModuleScope PcaiMedia {
                function Initialize-PcaiMediaFFI { return $false }
                { Initialize-PcaiMedia -Device cpu } | Should -Throw '*PcaiNative.dll not found*'
            }
        }

        It 'Leaves script:Initialized as false after the throw' {
            InModuleScope PcaiMedia {
                $script:Initialized = $false
                function Initialize-PcaiMediaFFI { return $false }
                try { Initialize-PcaiMedia -Device cpu } catch { }
                $script:Initialized | Should -BeFalse
            }
        }
    }

    # -----------------------------------------------------------------------
    # Initialize-PcaiMedia success path (depends on Initialize-PcaiMediaFFI returning $true)
    # -----------------------------------------------------------------------
    Context 'Initialize-PcaiMedia sets Initialized=true when FFI load succeeds' {

        It 'Sets script:Initialized to true after successful initialization' {
            InModuleScope PcaiMedia {
                $script:Initialized = $false
                function Initialize-PcaiMediaFFI { return $true }
                $null = Initialize-PcaiMedia -Device cpu
                $script:Initialized | Should -BeTrue
            }
        }

        It 'Returns a success object with Success=true' {
            InModuleScope PcaiMedia {
                function Initialize-PcaiMediaFFI { return $true }
                $result = Initialize-PcaiMedia -Device cpu
                $result.Success | Should -BeTrue
            }
        }

        It 'Reflects the requested device in the result' {
            InModuleScope PcaiMedia {
                function Initialize-PcaiMediaFFI { return $true }
                $result = Initialize-PcaiMedia -Device 'cuda:auto'
                $result.Device | Should -Be 'cuda:auto'
            }
        }
    }
}
