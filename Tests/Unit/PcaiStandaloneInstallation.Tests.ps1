#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $script:ModuleFiles = @('PcaiMedia.psd1', 'PcaiMedia.psm1', 'PcaiInference.psd1', 'PcaiInference.psm1')
    function Invoke-StandaloneChild {
        param([string]$Script, [string[]]$Arguments)
        $childPath = Join-Path $TestDrive 'standalone-child.ps1'
        Set-Content -LiteralPath $childPath -Value $Script -Encoding utf8
        $output = & (Get-Command pwsh).Source -NoLogo -NoProfile -File $childPath @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Fresh standalone child failed: $($output -join [Environment]::NewLine)" }
        return $output
    }
}

Describe 'Copied standalone module consumers without profile bootstrap' {
    BeforeEach {
        $script:FixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:Checkout = Join-Path $script:FixtureRoot 'checkout'
        $script:Installed = Join-Path $script:FixtureRoot 'installed'
        [void](New-Item -ItemType Directory -Path $script:Checkout -Force)
        '# checkout marker' | Set-Content -LiteralPath (Join-Path $script:Checkout 'PC-AI.ps1')
        foreach ($file in $script:ModuleFiles) {
            $name = [IO.Path]::GetFileNameWithoutExtension($file)
            $destination = Join-Path $script:Installed $name
            [void](New-Item -ItemType Directory -Path $destination -Force)
            Copy-Item -LiteralPath (Join-Path $script:RepoRoot "Modules/$file") -Destination $destination
        }
    }

    It 'discovers and imports both copied manifest pairs and uses a validated per-machine checkout' {
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $script:Checkout) -Script @'
param($Installed, $Checkout)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = $Installed + [IO.Path]::PathSeparator + $env:PSModulePath
$env:PCAI_ROOT = $Checkout
$env:PCAI_NATIVE_BUNDLE_ROOT = $null
foreach ($name in @('PcaiMedia', 'PcaiInference')) {
    $available = Get-Module -ListAvailable -Name $name | Where-Object { $_.Path.StartsWith($Installed, [StringComparison]::OrdinalIgnoreCase) }
    if (-not $available) { throw "Installed manifest not discoverable: $name" }
    $module = Import-Module -Name (Join-Path $Installed "$name/$name.psd1") -PassThru -Force
    $resolved = & $module { Get-PcaiProjectRoot }
    if ($resolved -ne $Checkout) { throw "Incorrect checkout for $name`: $resolved" }
}
if ((Get-PcaiMediaStatus).Initialized) { throw 'Import initialized the media engine.' }
$status = Get-PcaiInferenceStatus
if ($status.BackendInitialized -or $status.ModelLoaded -or $status.DllExists -isnot [bool]) { throw 'Read-only inference status is invalid or initialized an engine.' }
'copied pairs discoverable; statuses usable without profile'
'@
        $output | Should -Contain 'copied pairs discoverable; statuses usable without profile'
    }

    It 'rejects an unrelated PCAI_ROOT even when it contains an AGENTS file' {
        'unrelated workspace' | Set-Content -LiteralPath (Join-Path $script:FixtureRoot 'AGENTS.md')
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $script:FixtureRoot) -Script @'
param($Installed, $Unrelated)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = $Unrelated
foreach ($name in @('PcaiMedia', 'PcaiInference')) {
    $module = Import-Module -Name (Join-Path $Installed "$name/$name.psd1") -PassThru -Force
    try { & $module { Get-PcaiProjectRoot }; throw 'Invalid checkout accepted.' }
    catch { if ($_.Exception.Message -notlike '*containing PC-AI.ps1*') { throw } }
}
'unrelated checkout rejected'
'@
        $output | Should -Contain 'unrelated checkout rejected'
    }

    It 'binds source ancestry to the actual checkout without an override' {
        $modulesRoot = Join-Path $script:Checkout 'Modules'
        [void](New-Item -ItemType Directory -Path $modulesRoot)
        foreach ($file in $script:ModuleFiles) { Copy-Item -LiteralPath (Join-Path $script:RepoRoot "Modules/$file") -Destination $modulesRoot }
        $output = Invoke-StandaloneChild -Arguments @($script:Checkout) -Script @'
param($Checkout)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = $null
foreach ($name in @('PcaiMedia', 'PcaiInference')) {
    $module = Import-Module -Name (Join-Path $Checkout "Modules/$name.psd1") -PassThru -Force
    if ((& $module { Get-PcaiProjectRoot }) -ne $Checkout) { throw 'Source ancestry failed.' }
}
'source checkout resolved'
'@
        $output | Should -Contain 'source checkout resolved'
    }

    It 'rejects an incomplete explicit inference bundle before config, downloads or PATH mutation' {
        $bundle = Join-Path $script:FixtureRoot 'incomplete'
        [void](New-Item -ItemType Directory -Path $bundle)
        'not a complete pair' | Set-Content -LiteralPath (Join-Path $bundle 'PcaiNative.dll')
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $bundle) -Script @'
param($Installed, $Bundle)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = 'unavailable checkout must not be consulted'
$env:PCAI_NATIVE_BUNDLE_ROOT = $Bundle
$before = $env:PATH
$module = Import-Module -Name (Join-Path $Installed 'PcaiInference/PcaiInference.psd1') -PassThru -Force
& $module {
    function Get-PcaiConfig { throw 'Config fallback executed.' }
    function Resolve-PcaiRuntimeVariantDll { throw 'Runtime download fallback executed.' }
    try { Initialize-PcaiFFI; throw 'Incomplete bundle accepted.' }
    catch { if ($_.Exception.Message -notlike '*lacks pcai_inference.dll*') { throw } }
}
if ($env:PATH -ne $before) { throw 'Failed explicit selection modified PATH.' }
'incomplete explicit inference bundle rejected'
'@
        $output | Should -Contain 'incomplete explicit inference bundle rejected'
    }

    It 'canonicalizes a complete inference pair and rejects a conflicting DllPath without fallback' {
        $bundle = Join-Path $script:FixtureRoot 'selected'
        [void](New-Item -ItemType Directory -Path $bundle)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_inference.dll', 'other.dll')) { 'selection fixture' | Set-Content -LiteralPath (Join-Path $bundle $leaf) }
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $script:FixtureRoot, $bundle) -Script @'
param($Installed, $Root, $Bundle)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = 'unavailable checkout must not be consulted'
$env:PCAI_NATIVE_BUNDLE_ROOT = './selected'
Set-Location -LiteralPath $Root
$module = Import-Module -Name (Join-Path $Installed 'PcaiInference/PcaiInference.psd1') -PassThru -Force
$resolved = & $module { Resolve-PcaiInferenceDll }
if ($resolved -ne (Join-Path $Bundle 'pcai_inference.dll') -or $env:PCAI_NATIVE_BUNDLE_ROOT -ne $Bundle) { throw 'Explicit pair not canonicalized.' }
& $module {
    try { Resolve-PcaiInferenceDll -OverridePath './selected/other.dll'; throw 'Conflicting override accepted.' }
    catch { if ($_.Exception.Message -notlike '*conflicts with PCAI_NATIVE_BUNDLE_ROOT*') { throw } }
}
'canonical explicit pair selected; conflict rejected'
'@
        $output | Should -Contain 'canonical explicit pair selected; conflict rejected'
    }

    It 'loads the qualified CPU media pair from the copied module without a checkout or profile' -Skip:(-not $env:PCAI_TEST_NATIVE_MEDIA_BUNDLE) {
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $env:PCAI_TEST_NATIVE_MEDIA_BUNDLE) -Script @'
param($Installed, $Bundle)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = 'unavailable checkout must not be consulted'
$env:PCAI_NATIVE_BUNDLE_ROOT = $Bundle
Import-Module -Name (Join-Path $Installed 'PcaiMedia/PcaiMedia.psd1') -Force
Initialize-PcaiMedia -Device cpu
try {
    $status = Get-PcaiMediaStatus
    if (-not $status.Initialized -or $status.ModelLoaded) { throw 'CPU media lifecycle state invalid.' }
    $assembly = [AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.GetName().Name -eq 'PcaiNative' } | Select-Object -First 1
    if ($assembly.Location -ne (Join-Path $Bundle 'PcaiNative.dll')) { throw 'Another managed bridge was loaded.' }
} finally { Stop-PcaiMedia }
if ((Get-PcaiMediaStatus).Initialized) { throw 'Shutdown did not clear initialized state.' }
'actual copied CPU media lifecycle passed; trained model not exercised'
'@
        $output | Should -Contain 'actual copied CPU media lifecycle passed; trained model not exercised'
    }

    It 'rejects a previously loaded inference bridge from another bundle before PATH mutation' -Skip:(-not $env:PCAI_TEST_NATIVE_MEDIA_BUNDLE) {
        $first = Join-Path $script:FixtureRoot 'first'
        $second = Join-Path $script:FixtureRoot 'second'
        foreach ($bundle in @($first, $second)) {
            [void](New-Item -ItemType Directory -Path $bundle)
            Copy-Item -LiteralPath (Join-Path $env:PCAI_TEST_NATIVE_MEDIA_BUNDLE 'PcaiNative.dll') -Destination $bundle
            'resolver selection fixture, not an inference engine' | Set-Content -LiteralPath (Join-Path $bundle 'pcai_inference.dll')
        }
        $output = Invoke-StandaloneChild -Arguments @($script:Installed, $first, $second) -Script @'
param($Installed, $First, $Second)
$ErrorActionPreference = 'Stop'
$env:PCAI_ROOT = 'unavailable checkout must not be consulted'
$env:PCAI_NATIVE_BUNDLE_ROOT = $Second
[void][Reflection.Assembly]::LoadFrom((Join-Path $First 'PcaiNative.dll'))
$before = $env:PATH
$module = Import-Module -Name (Join-Path $Installed 'PcaiInference/PcaiInference.psd1') -Force -PassThru
if (& $module { Initialize-PcaiFFI -WarningAction SilentlyContinue }) { throw 'Already-loaded foreign bridge accepted.' }
if ($env:PATH -ne $before) { throw 'Rejected bridge selection modified PATH.' }
'foreign inference bridge rejected; actual inference engine not exercised'
'@
        $output | Should -Contain 'foreign inference bridge rejected; actual inference engine not exercised'
    }
}
