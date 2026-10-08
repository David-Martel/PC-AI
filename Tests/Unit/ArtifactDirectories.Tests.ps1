#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repoRoot 'Tools/PcaiArtifactDirectories.ps1')
    $tokens = $null
    $errors = $null
    $buildAst = [Management.Automation.Language.Parser]::ParseFile((Join-Path $repoRoot 'Build.ps1'), [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Build.ps1 must parse before testing publication.' }
    foreach ($name in @('Publish-PcaiNativeBundle', 'Publish-StagedArtifact')) {
        $functionAst = $buildAst.Find({ param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
        }, $true)
        . ([scriptblock]::Create($functionAst.Extent.Text))
    }
    function Resolve-CargoOutputDirectory { param($ProjectDir, $Configuration) Join-Path $script:ProjectRoot 'absent-target' }
}

Describe 'Stable artifact revisions preserve evidence' {
    It 'rejects unsafe components before creating directories' -ForEach @(
        @{ Component = '$null' }, @{ Component = 'NUL.txt' }, @{ Component = 'COM1.' }
    ) {
        $root = Join-Path (Join-Path $TestDrive $Component) 'measurements'
        { New-PcaiArtifactDirectory -Root $root -Name 'tooling' } | Should -Throw '*Unsafe Windows path*'
        Test-Path -LiteralPath $root | Should -BeFalse
    }

    It 'rejects a filesystem root' {
        { New-PcaiArtifactDirectory -Root ([IO.Path]::GetPathRoot($TestDrive)) -Name 'build' } | Should -Throw '*dedicated directory*'
    }

    It 'does not follow a directory junction into another custody root' -Skip:(-not $IsWindows) {
        $target = Join-Path $TestDrive 'protected'
        [void](New-Item -ItemType Directory -Path $target)
        $link = Join-Path $TestDrive 'linked'
        [void](New-Item -ItemType Junction -Path $link -Target $target)
        { New-PcaiArtifactDirectory -Root (Join-Path $link 'child') -Name 'build' } | Should -Throw '*Linked artifact roots*'
        Test-Path -LiteralPath (Join-Path $target 'child') | Should -BeFalse
    }

    It 'allocates separate revisions without changing previous payloads' {
        $root = Join-Path $TestDrive 'measurements'
        $first = New-PcaiArtifactDirectory -Root $root -Name 'tooling'
        [IO.File]::WriteAllBytes((Join-Path $first 'evidence.bin'), [byte[]]@(128, 255, 0))
        $hash = (Get-FileHash -LiteralPath (Join-Path $first 'evidence.bin')).Hash
        $second = New-PcaiArtifactDirectory -Root $root -Name 'tooling'
        (Split-Path -Leaf $first) | Should -Be 'tooling-r1'
        (Split-Path -Leaf $second) | Should -Be 'tooling-r2'
        (Get-FileHash -LiteralPath (Join-Path $first 'evidence.bin')).Hash | Should -BeExactly $hash
    }

    It 'does not claim another live publisher revision' {
        $root = Join-Path $TestDrive 'contended'
        [void](New-Item -ItemType Directory -Path $root)
        $handle = [IO.File]::Open((Join-Path $root '.artifact-revision.lock'),
            [IO.FileMode]::Create, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        try {
            { New-PcaiArtifactDirectory -Root $root -Name 'build' } | Should -Throw
            @(Get-ChildItem -LiteralPath $root -Directory).Count | Should -Be 0
        } finally { $handle.Dispose() }
    }

    It 'preserves actual published bundles from independent build roots with identical run labels' {
        $script:ProjectRoot = Join-Path $TestDrive 'publication-repo'
        $tools = Join-Path $script:ProjectRoot 'Tools'
        [void](New-Item -ItemType Directory -Path $tools -Force)
        Copy-Item -LiteralPath (Join-Path $repoRoot 'Tools/PcaiArtifactDirectories.ps1') -Destination $tools
        $script:VersionInfo = $null
        $script:BuildRunLabel = 'build-r1'
        $publications = @()
        foreach ($sourceName in @('build-root-one', 'build-root-two')) {
            $source = Join-Path $TestDrive $sourceName
            [void](New-Item -ItemType Directory -Path $source)
            foreach ($leaf in @('PcaiNative.dll', 'PcaiNative.deps.json', 'pcai_core_lib.dll')) {
                [IO.File]::WriteAllText((Join-Path $source $leaf), "$sourceName-$leaf")
            }
            $publications += Publish-PcaiNativeBundle -PublishRoot $source -Configuration Release
        }
        $publications[0].BundleRoot | Should -Not -Be $publications[1].BundleRoot
        $publications[0].BundleName | Should -Be 'native-unknown-r1'
        $publications[1].BundleName | Should -Be 'native-unknown-r2'
        foreach ($index in 0..1) {
            $manifest = Get-Content -LiteralPath $publications[$index].ManifestPath -Raw | ConvertFrom-Json
            @($manifest.files).Count | Should -Be 3
            foreach ($entry in $manifest.files) {
                (Get-FileHash -LiteralPath $entry.DestinationPath).Hash | Should -BeExactly $entry.Sha256
            }
        }
        [IO.File]::ReadAllText((Join-Path $publications[0].BundleRoot 'PcaiNative.dll')) | Should -Be 'build-root-one-PcaiNative.dll'
    }
}
