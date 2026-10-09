#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $modulePath = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Modules/PcaiMedia.psm1'
    $tokens = $null
    $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($modulePath, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Media module must parse before loader verification.' }
    foreach ($name in @('Get-PcaiMediaLoadedBridgePath', 'Import-PcaiMediaManagedBridge', 'Resolve-PcaiMediaBridgeFilePath', 'Initialize-PcaiMediaFFI')) {
        $definition = $ast.Find({ param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
        }, $true)
        . ([scriptblock]::Create($definition.Extent.Text))
    }
    function Get-PcaiProjectRoot { $script:FixtureRepo }
}

Describe 'Standalone media bridge bundle selection' {
    BeforeEach {
        $script:SavedBundle = $env:PCAI_NATIVE_BUNDLE_ROOT
        $env:PCAI_NATIVE_BUNDLE_ROOT = $null
        $script:FixtureRepo = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:DefaultBin = Join-Path $script:FixtureRepo 'bin'
        [void](New-Item -ItemType Directory -Path $script:DefaultBin -Force)
        Set-Content -LiteralPath (Join-Path $script:DefaultBin 'PcaiNative.dll') -Value 'default fixture'
        Mock Get-PcaiMediaLoadedBridgePath { $null }
        Mock Import-PcaiMediaManagedBridge { param($Path) $Path }
    }
    AfterEach { $env:PCAI_NATIVE_BUNDLE_ROOT = $script:SavedBundle }

    It 'keeps the existing default project bridge selection' {
        Initialize-PcaiMediaFFI | Should -BeTrue
        Should -Invoke Import-PcaiMediaManagedBridge -Times 1 -ParameterFilter { $Path -eq (Join-Path $script:DefaultBin 'PcaiNative.dll') }
    }

    It 'rejects an incomplete explicit bundle without falling back to repo bin' {
        $env:PCAI_NATIVE_BUNDLE_ROOT = $script:DefaultBin
        Initialize-PcaiMediaFFI -WarningAction SilentlyContinue | Should -BeFalse
        Should -Invoke Import-PcaiMediaManagedBridge -Times 0
    }

    It 'canonicalizes a relative explicit media pair before managed loading' {
        $pair = Join-Path $script:FixtureRepo 'selected'
        [void](New-Item -ItemType Directory -Path $pair)
        foreach ($leaf in @('PcaiNative.dll', 'pcai_media.dll')) { Set-Content -LiteralPath (Join-Path $pair $leaf) -Value 'selected fixture' }
        Push-Location $script:FixtureRepo
        try {
            $env:PCAI_NATIVE_BUNDLE_ROOT = './selected'
            Initialize-PcaiMediaFFI | Should -BeTrue
            $env:PCAI_NATIVE_BUNDLE_ROOT | Should -BeExactly $pair
            Should -Invoke Import-PcaiMediaManagedBridge -Times 1 -ParameterFilter { $Path -eq (Join-Path $pair 'PcaiNative.dll') }
        } finally { Pop-Location }
    }

    It 'rejects a bridge already loaded from a different directory' {
        Mock Get-PcaiMediaLoadedBridgePath { Join-Path $script:FixtureRepo 'previous/PcaiNative.dll' }
        Initialize-PcaiMediaFFI -WarningAction SilentlyContinue | Should -BeFalse
        Should -Invoke Import-PcaiMediaManagedBridge -Times 0
    }

    It 'rejects a loader that reused another assembly during initialization' {
        Mock Import-PcaiMediaManagedBridge { Join-Path $script:FixtureRepo 'racing/PcaiNative.dll' }
        Initialize-PcaiMediaFFI -WarningAction SilentlyContinue | Should -BeFalse
    }
}
