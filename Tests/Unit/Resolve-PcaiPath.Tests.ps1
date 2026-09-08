# Tests/Unit/Resolve-PcaiPath.Tests.ps1
BeforeAll {
    $ModuleRoot = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM'
    . (Join-Path $ModuleRoot 'Private\Resolve-PcaiPath.ps1')
}

Describe 'Resolve-PcaiPath' -Tag 'Unit', 'Portable' {
    Context 'Default resolution' {
        It 'Should resolve project root from module location' {
            $root = Resolve-PcaiPath -PathType 'Root'
            # Do NOT assert the folder name. GitHub checks this repo out as
            # D:\a\PC-AI\PC-AI -- the repo name, hyphenated -- while the local
            # clone is PC_AI, so `Should -Match 'PC_AI'` asserted the developer's
            # directory name rather than anything the function does. Assert that
            # the resolved path really is the repo root, which is both portable
            # and a stronger check.
            $root | Should -Not -BeNullOrEmpty
            Test-Path (Join-Path $root 'Build.ps1') | Should -BeTrue
            Test-Path (Join-Path $root 'Modules\PC-AI.LLM') | Should -BeTrue
        }

        It 'Should resolve Config path' {
            $config = Resolve-PcaiPath -PathType 'Config'
            Test-Path $config | Should -BeTrue
        }

        It 'Should resolve HVSock config' {
            $hvsock = Resolve-PcaiPath -PathType 'HVSockConfig'
            $hvsock | Should -Match 'hvsock-proxy\.conf'
        }
    }

    Context 'Environment variable override' {
        It 'Should respect PCAI_ROOT environment variable' {
            $env:PCAI_ROOT = 'C:\TestRoot'
            try {
                $root = Resolve-PcaiPath -PathType 'Root'
                $root | Should -Be 'C:\TestRoot'
            } finally {
                Remove-Item Env:\PCAI_ROOT -ErrorAction SilentlyContinue
            }
        }
    }
}
