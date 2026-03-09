#Requires -Version 5.1
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Functional tests for the PC-AI media CLI integration.

.DESCRIPTION
    Tests the CLI integration layer for the media sub-system:
    1. Help and discovery: PC-AI.ps1 help text, subcommand listing
    2. Status command: Get-PcaiMediaStatus reporting without initialization
    3. Error messages: helpful error text for missing model or missing file inputs

    These tests do not require a built DLL or downloaded model and focus on
    the observable CLI and module-level behavior of the media pipeline.

.NOTES
    Run with: Invoke-Pester -Path .\Tests\Functional\Media.Functional.Tests.ps1 -Tag Functional,Media
#>

BeforeAll {
    # Import shared test helpers
    Import-Module (Join-Path $PSScriptRoot '..\Helpers\TestHelpers.psm1') -Force

    $paths = Get-TestPaths -StartPath $PSScriptRoot
    $script:ProjectRoot  = $paths.ProjectRoot
    $script:MainScript   = Join-Path $script:ProjectRoot 'PC-AI.ps1'
    $script:MediaModPath = Join-Path $script:ProjectRoot 'Modules\PcaiMedia.psm1'

    # Load PcaiMedia module (no DLL required for status/help paths)
    # Provide a stub [PcaiNative.MediaModule] type if not present
    if (-not ([System.Management.Automation.PSTypeName]'PcaiNative.MediaModule').Type) {
        Add-Type -TypeDefinition @'
namespace PcaiNative {
    public static class MediaModule {
        public static int  pcai_media_init(string device)                                         { return 0; }
        public static void pcai_media_shutdown()                                                   { }
        public static int  pcai_media_load_model(string modelPath, int gpuLayers)                 { return 0; }
        public static string GenerateImage(string p, string o, float c, float t)                  { return null; }
        public static string UnderstandImage(string i, string q, uint m, float t)                 { return "stub"; }
        public static long GenerateImageNativeAsync(string p, float c, float t, string o)         { return 1; }
        public static string UpscaleImage(string m, string i, string o)                           { return null; }
        public static string GetLastError()                                                        { return null; }
        public static bool   IsAvailable                                                           { get { return true; } }
    }
}
'@ -ErrorAction SilentlyContinue
    }

    Import-Module $script:MediaModPath -Force -ErrorAction SilentlyContinue
}

AfterAll {
    Remove-Module PcaiMedia -Force -ErrorAction SilentlyContinue
}

# ==============================================================================
# Help and Discovery
# ==============================================================================

Describe 'Functional: Media CLI' -Tag 'Functional', 'Media' {

    Context 'Help and Discovery' {

        It 'PC-AI.ps1 exists at project root' {
            Test-Path $script:MainScript | Should -BeTrue
        }

        It 'PcaiMedia.psm1 exists at expected path' {
            Test-Path $script:MediaModPath | Should -BeTrue
        }

        It 'PcaiMedia module exports Initialize-PcaiMedia' {
            Get-Command -Module PcaiMedia -Name 'Initialize-PcaiMedia' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'PcaiMedia module exports New-PcaiImage' {
            Get-Command -Module PcaiMedia -Name 'New-PcaiImage' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'PcaiMedia module exports Get-PcaiImageAnalysis' {
            Get-Command -Module PcaiMedia -Name 'Get-PcaiImageAnalysis' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'PcaiMedia module exports Get-PcaiMediaStatus' {
            Get-Command -Module PcaiMedia -Name 'Get-PcaiMediaStatus' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'PcaiMedia module exports Invoke-PcaiUpscale' {
            Get-Command -Module PcaiMedia -Name 'Invoke-PcaiUpscale' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'PcaiMedia module exports New-PcaiImageAsync' {
            Get-Command -Module PcaiMedia -Name 'New-PcaiImageAsync' -ErrorAction SilentlyContinue |
                Should -Not -BeNullOrEmpty
        }

        It 'Get-Help Initialize-PcaiMedia returns a synopsis' {
            $help = Get-Help Initialize-PcaiMedia -ErrorAction SilentlyContinue
            $help.Synopsis | Should -Not -BeNullOrEmpty
        }

        It 'Get-Help New-PcaiImage returns parameter help for -Prompt' {
            $help = Get-Help New-PcaiImage -Parameter Prompt -ErrorAction SilentlyContinue
            $help | Should -Not -BeNullOrEmpty
        }

        It 'Get-Help Get-PcaiImageAnalysis returns parameter help for -ImagePath' {
            $help = Get-Help Get-PcaiImageAnalysis -Parameter ImagePath -ErrorAction SilentlyContinue
            $help | Should -Not -BeNullOrEmpty
        }

        It 'Get-Help Invoke-PcaiUpscale returns at least one example' {
            $help = Get-Help Invoke-PcaiUpscale -ErrorAction SilentlyContinue
            @($help.Examples.Example).Count | Should -BeGreaterThan 0
        }

        It 'Initialize-PcaiMedia -Device parameter accepts cpu' {
            $cmd = Get-Command Initialize-PcaiMedia
            $validateSet = $cmd.Parameters['Device'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet.ValidValues | Should -Contain 'cpu'
        }

        It 'Initialize-PcaiMedia -Device parameter accepts cuda:0' {
            $cmd = Get-Command Initialize-PcaiMedia
            $validateSet = $cmd.Parameters['Device'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet.ValidValues | Should -Contain 'cuda:0'
        }

        It 'Initialize-PcaiMedia -Device parameter accepts cuda:1' {
            $cmd = Get-Command Initialize-PcaiMedia
            $validateSet = $cmd.Parameters['Device'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet.ValidValues | Should -Contain 'cuda:1'
        }
    }

    # ===========================================================================
    # Status Command
    # ===========================================================================

    Context 'Status Command' {

        BeforeEach {
            # Ensure module is in clean (uninitialized) state
            InModuleScope PcaiMedia {
                $script:Initialized  = $false
                $script:ModelLoaded  = $false
                $script:CurrentModel = $null
            }
        }

        It 'Get-PcaiMediaStatus works without initialization (no throw)' {
            { Get-PcaiMediaStatus } | Should -Not -Throw
        }

        It 'Reports Initialized=false when not initialized' {
            $status = Get-PcaiMediaStatus
            $status.Initialized | Should -BeFalse
        }

        It 'Reports ModelLoaded=false when not initialized' {
            $status = Get-PcaiMediaStatus
            $status.ModelLoaded | Should -BeFalse
        }

        It 'Reports CurrentModel=null when not initialized' {
            $status = Get-PcaiMediaStatus
            $status.CurrentModel | Should -BeNullOrEmpty
        }

        It 'Status output is a PSCustomObject (not null, not string)' {
            $status = Get-PcaiMediaStatus
            $status | Should -BeOfType [PSCustomObject]
        }

        It 'Status has exactly 3 properties (Initialized, ModelLoaded, CurrentModel)' {
            $status = Get-PcaiMediaStatus
            @($status.PSObject.Properties).Count | Should -Be 3
        }
    }

    # ===========================================================================
    # Error Messages
    # ===========================================================================

    Context 'Error Messages' {

        BeforeEach {
            InModuleScope PcaiMedia {
                $script:Initialized  = $false
                $script:ModelLoaded  = $false
                $script:CurrentModel = $null
            }
        }

        It 'New-PcaiImage without model gives a helpful error containing model-related text' {
            try {
                New-PcaiImage -Prompt 'test' -ErrorAction Stop
                # Should not reach here
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'model|Model|Import-PcaiMediaModel'
            }
        }

        It 'Get-PcaiImageAnalysis without model gives a helpful error containing model-related text' {
            try {
                Get-PcaiImageAnalysis -ImagePath 'C:\dummy.png' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'model|Model|Import-PcaiMediaModel'
            }
        }

        It 'New-PcaiImageAsync without model gives a helpful error containing model-related text' {
            try {
                New-PcaiImageAsync -Prompt 'test' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'model|Model|Import-PcaiMediaModel'
            }
        }

        It 'Invoke-PcaiUpscale without initialization gives a helpful error mentioning Initialize-PcaiMedia' {
            try {
                Invoke-PcaiUpscale -InputPath 'C:\in.png' -OutputPath 'C:\out.png' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'Initialize-PcaiMedia|not initialized|pipeline'
            }
        }

        It 'Get-PcaiImageAnalysis with missing file gives a clear file-not-found error' {
            InModuleScope PcaiMedia { $script:ModelLoaded = $true }
            try {
                Get-PcaiImageAnalysis -ImagePath 'C:\nonexistent_pcai_functional_test.png' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'not found|Image file'
            }
        }

        It 'Invoke-PcaiUpscale with missing input file gives a clear input-not-found error' {
            InModuleScope PcaiMedia { $script:Initialized = $true }
            try {
                Invoke-PcaiUpscale -InputPath 'C:\nonexistent_pcai_src.png' -OutputPath 'C:\out.png' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'not found|Input image'
            }
        }

        It 'Import-PcaiMediaModel without initialization gives helpful pre-condition error' {
            try {
                Import-PcaiMediaModel -ModelPath 'deepseek-ai/Janus-Pro-1B' -ErrorAction Stop
                $false | Should -BeTrue -Because "Expected an exception"
            } catch {
                $_.Exception.Message | Should -Match 'Initialize-PcaiMedia|not initialized'
            }
        }

        It 'Initialize-PcaiMedia with invalid device string fails at parameter validation' {
            # ValidateSet should reject unknown device before any code runs
            { Initialize-PcaiMedia -Device 'npu' -ErrorAction Stop } | Should -Throw
        }
    }
}
