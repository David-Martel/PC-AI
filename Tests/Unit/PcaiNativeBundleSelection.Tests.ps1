#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    . (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')
}

Describe 'Explicit native bundle custody' {
    BeforeEach {
        $savedBundle = $env:PCAI_NATIVE_BUNDLE_ROOT
        $savedPath = $env:PATH
        $script:PcaiNativeLoaded = $false
        $script:PcaiNativeDllPath = $null
        Mock Add-Type { throw 'Fixture stops at managed load boundary.' }
        Mock Get-Module { $null }
    }
    AfterEach {
        $env:PCAI_NATIVE_BUNDLE_ROOT = $savedBundle
        $env:PATH = $savedPath
    }

    It 'does not fall back when an explicit bundle is incomplete' {
        $env:PCAI_NATIVE_BUNDLE_ROOT = Join-Path $TestDrive 'incomplete'
        [void](New-Item -ItemType Directory -Path $env:PCAI_NATIVE_BUNDLE_ROOT)
        Set-Content -LiteralPath (Join-Path $env:PCAI_NATIVE_BUNDLE_ROOT 'PcaiNative.dll') -Value 'fixture'
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $savedPath
        Should -Invoke Add-Type -Times 0
        $script:PcaiNativeDllPath | Should -BeNullOrEmpty
    }

    It 'does not fall back when the explicit directory does not exist' {
        $env:PCAI_NATIVE_BUNDLE_ROOT = Join-Path $TestDrive 'absent'
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        Should -Invoke Add-Type -Times 0
        $env:PATH | Should -BeExactly $savedPath
    }

    It 'selects the exact paired directory and compares PATH segments exactly' {
        $bundle = Join-Path $TestDrive 'paired'
        [void](New-Item -ItemType Directory -Path $bundle)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_core_lib.dll')) {
            Set-Content -LiteralPath (Join-Path $bundle $leaf) -Value 'fixture'
        }
        $env:PCAI_NATIVE_BUNDLE_ROOT = $bundle
        $env:PATH = "$bundle-older;$savedPath"
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $script:PcaiNativeDllPath | Should -BeExactly $bundle
        ($env:PATH -split [IO.Path]::PathSeparator)[0] | Should -BeExactly $bundle
        Should -Invoke Add-Type -Times 1 -ParameterFilter { $Path -eq (Join-Path $bundle 'PcaiNative.dll') }
    }

    It 'preserves default discovery and requires both files in a candidate' {
        $complete = Join-Path $TestDrive 'default-complete'
        $partial = Join-Path $TestDrive 'default-partial'
        foreach ($root in @($complete, $partial)) { [void](New-Item -ItemType Directory -Path $root) }
        Set-Content -LiteralPath (Join-Path $partial 'PcaiNative.dll') -Value 'fixture'
        foreach ($leaf in @('PcaiNative.dll', 'pcai_core_lib.dll')) { Set-Content -LiteralPath (Join-Path $complete $leaf) -Value 'fixture' }
        @(Get-PcaiNativeCandidatePaths -BasePaths @($partial, $complete)) | Should -Be @($complete)
    }
}
