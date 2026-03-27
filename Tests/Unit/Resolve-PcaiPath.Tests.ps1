# Tests/Unit/Resolve-PcaiPath.Tests.ps1
BeforeAll {
    $ModuleRoot = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM'
    . (Join-Path $ModuleRoot 'Private\Resolve-PcaiPath.ps1')
}

Describe 'Resolve-PcaiPath' -Tag 'Unit', 'Portable' {
    Context 'Default resolution' {
        It 'Should resolve project root from module location' {
            $root = Resolve-PcaiPath -PathType 'Root'
            $root | Should -Match 'PC_AI'
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
