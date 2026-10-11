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
        # Selection fixtures stop at Add-Type. Actual assemblies loaded by
        # another suite must not change this deliberately isolated boundary.
        Mock Get-PcaiNativeLoadedBridgePath { $null }
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

    It 'rejects an already loaded foreign bridge before PATH mutation or managed load' {
        $bundle = Join-Path $TestDrive 'foreign-bridge-selection'
        [void](New-Item -ItemType Directory -Path $bundle)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_core_lib.dll')) {
            Set-Content -LiteralPath (Join-Path $bundle $leaf) -Value 'selection fixture'
        }
        $env:PCAI_NATIVE_BUNDLE_ROOT = $bundle
        Mock Get-PcaiNativeLoadedBridgePath { 'C:\foreign-bundle\PcaiNative.dll' }
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $savedPath
        Should -Invoke Add-Type -Times 0
    }

    It 'canonicalizes a relative override for subsequent managed resolution' {
        $root = Join-Path $TestDrive 'relative-location'
        $bundle = Join-Path $root 'paired'
        [void](New-Item -ItemType Directory -Path $bundle -Force)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_core_lib.dll')) {
            Set-Content -LiteralPath (Join-Path $bundle $leaf) -Value 'fixture'
        }
        Push-Location $root
        try {
            $env:PCAI_NATIVE_BUNDLE_ROOT = './paired'
            Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
            $env:PCAI_NATIVE_BUNDLE_ROOT | Should -BeExactly $bundle
            $script:PcaiNativeDllPath | Should -BeExactly $bundle
            Should -Invoke Add-Type -Times 1 -ParameterFilter { $Path -eq (Join-Path $bundle 'PcaiNative.dll') }
        } finally { Pop-Location }
    }

    It 'loads and reuses the qualified short-name bundle under StrictMode and rejects another real bundle' -Skip:(-not $env:PCAI_TEST_NATIVE_MEDIA_BUNDLE) {
        $bundle = [IO.Path]::GetFullPath($env:PCAI_TEST_NATIVE_MEDIA_BUNDLE)
        $foreign = Join-Path $TestDrive 'foreign-real-native-bundle'
        [void](New-Item -ItemType Directory -Path $foreign)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_core_lib.dll')) {
            Copy-Item -LiteralPath (Join-Path $bundle $leaf) -Destination $foreign
        }
        $probe = Join-Path $TestDrive 'real-native-path-consumer.ps1'
        @'
param($Initializer, $Bundle, $Foreign)
$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Management, Microsoft.PowerShell.Utility
Add-Type -TypeDefinition @"
using System.Text;
using System.Runtime.InteropServices;
public static class PcaiNativeShortPathFixture {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern uint GetShortPathName(string path, StringBuilder output, uint capacity);
}
"@
$buffer = [Text.StringBuilder]::new(32768)
$length = [PcaiNativeShortPathFixture]::GetShortPathName($Bundle, $buffer, $buffer.Capacity)
$short = $buffer.ToString()
if (-not $length -or $length -ge $buffer.Capacity -or $short -eq $Bundle) {
    throw 'Actual native path qualification pending: no distinct existing 8.3 directory alias.'
}
$PSModuleAutoLoadingPreference = 'None'
$env:PCAI_NATIVE_BUNDLE_ROOT = $short
$env:PCAI_ROOT = $null
$script:PcaiNativeLoaded = $false
$script:PcaiNativeDllPath = $null
$script:PcaiNativeVersion = $null
. $Initializer
$initialize = (Get-Command Initialize-PcaiNative).ScriptBlock
Set-StrictMode -Version Latest
if (-not (& $initialize -Force)) { throw 'Qualified native bundle did not load from its short directory alias.' }
if ($env:PCAI_NATIVE_BUNDLE_ROOT -cne $Bundle) { throw 'Native bundle directory was not expanded to its long form.' }
$assembly = [AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.GetName().Name -eq 'PcaiNative' } | Select-Object -First 1
if (-not $assembly -or $assembly.Location -cne (Join-Path $Bundle 'PcaiNative.dll')) { throw 'Another bridge was loaded.' }
if (-not [PcaiNative.PcaiCore]::IsAvailable -or -not [PcaiNative.PcaiCore]::Version) { throw 'Actual qualified core calls failed.' }
# This detecting boundary blocks a second managed load; Core calls remain real.
function Add-Type { throw 'The already loaded bridge must be reused.' }
$env:PCAI_NATIVE_BUNDLE_ROOT = $short
if (-not (& $initialize -Force)) { throw 'Same-bridge StrictMode reuse failed.' }
$before = $env:PATH
$env:PCAI_NATIVE_BUNDLE_ROOT = $Foreign
if (& $initialize -Force) { throw 'A different real bundle containing identical DLL bytes was accepted.' }
if ($env:PATH -cne $before) { throw 'Foreign bundle rejection modified PATH.' }
Write-Output 'real-core-short-directory-and-strict-reuse-passed; foreign-bundle-rejected'
'@ | Set-Content -LiteralPath $probe
        $initializer = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1'
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $probe $initializer $bundle $foreign 2>&1
        if ($LASTEXITCODE -ne 0) { $output | ForEach-Object { Write-Host $_ } }
        $LASTEXITCODE | Should -Be 0
        $output | Should -Contain 'real-core-short-directory-and-strict-reuse-passed; foreign-bundle-rejected'
    }
}
