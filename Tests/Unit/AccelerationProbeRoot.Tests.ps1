#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    Import-Module (Join-Path $script:ProjectRoot 'Modules/PC-AI.Common/PC-AI.Common.psm1') -Force
    $script:SavedPcaiRoot = $env:PCAI_ROOT
    $script:SavedTestManifest = $env:PCAI_TEST_MANIFEST
}

AfterAll {
    $env:PCAI_ROOT = $script:SavedPcaiRoot
    $env:PCAI_TEST_MANIFEST = $script:SavedTestManifest
    Remove-Module PC-AI.Common -Force -ErrorAction SilentlyContinue
}

Describe 'Acceleration probe repository discovery' -Tag 'Unit', 'Acceleration', 'Portable' {
    BeforeEach {
        $script:FixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString())
        $manifestDirectory = Join-Path $script:FixtureRoot 'Modules/PC-AI.Acceleration'
        New-Item -ItemType Directory -Path $manifestDirectory -Force | Out-Null
        $env:PCAI_TEST_MANIFEST = Join-Path $manifestDirectory 'PC-AI.Acceleration.psd1'
        Set-Content -LiteralPath $env:PCAI_TEST_MANIFEST -Value '@{}'
        Set-Content -LiteralPath (Join-Path $script:FixtureRoot 'PC-AI.ps1') -Value '# Fixture repository marker'
        $env:PCAI_ROOT = $null
        Mock Resolve-PcaiAccelerationManifestPath -ModuleName PC-AI.Common { $env:PCAI_TEST_MANIFEST }
    }

    It 'finds the repository above Modules without an environment override' {
        $probe = Get-PcaiAccelerationProbe
        $probe.RepoRoot | Should -Be $script:FixtureRoot
        $probe.AccelerationModule.ManifestPath | Should -Be $env:PCAI_TEST_MANIFEST
    }

    It 'uses the manifest checkout instead of a different environment checkout' {
        $otherCheckout = Join-Path $TestDrive 'other-checkout'
        New-Item -ItemType Directory -Path $otherCheckout -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $otherCheckout 'PC-AI.ps1') -Value '# Other checkout'
        $env:PCAI_ROOT = $otherCheckout
        (Get-PcaiAccelerationProbe).RepoRoot | Should -Be $script:FixtureRoot
    }

    It 'discovers native file locations relative to the selected checkout' {
        $nativeDirectory = Join-Path $script:FixtureRoot 'bin'
        New-Item -ItemType Directory -Path $nativeDirectory -Force | Out-Null
        # These files test discovery only; no library is loaded by this probe.
        Set-Content -LiteralPath (Join-Path $nativeDirectory 'PcaiNative.dll') -Value 'fixture'
        Set-Content -LiteralPath (Join-Path $nativeDirectory 'pcai_core_lib.dll') -Value 'fixture'
        $probe = Get-PcaiAccelerationProbe
        $probe.Native.Root | Should -Be $nativeDirectory
        $probe.Native.PcaiNativeDll | Should -BeTrue
        $probe.Native.CoreLibDll | Should -BeTrue
    }

    It 'rejects an unrelated AGENTS marker and retains the environment fallback' {
        Remove-Item -LiteralPath (Join-Path $script:FixtureRoot 'PC-AI.ps1')
        Set-Content -LiteralPath (Join-Path $script:FixtureRoot 'AGENTS.md') -Value '# Unrelated project'
        $fallbackRoot = Join-Path $TestDrive 'fallback-checkout'
        New-Item -ItemType Directory -Path $fallbackRoot -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $fallbackRoot 'PC-AI.ps1') -Value '# Fallback checkout'
        $env:PCAI_ROOT = $fallbackRoot
        (Get-PcaiAccelerationProbe).RepoRoot | Should -Be $fallbackRoot
    }

    It 'retains the environment fallback when no manifest is available' {
        $env:PCAI_TEST_MANIFEST = $null
        $env:PCAI_ROOT = $script:FixtureRoot
        $probe = Get-PcaiAccelerationProbe
        $probe.RepoRoot | Should -Be $script:FixtureRoot
        $probe.AccelerationModule.Available | Should -BeFalse
    }
}
