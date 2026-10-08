#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    . (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/PcaiArtifactDirectories.ps1')
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
}
