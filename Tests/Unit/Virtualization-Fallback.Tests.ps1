#Requires -Version 5.1

BeforeAll {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $modulePath = Join-Path $RepoRoot 'Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1'
    if (Test-Path $modulePath) {
        Import-Module $modulePath -Force -ErrorAction Stop
    }
}

Describe 'Virtualization external script fallback' -Tag 'Unit', 'Virtualization', 'Portable' {
    Context 'Invoke-WSLDockerHealthCheck fallback' {
        It 'falls back to Get-WSLEnvironmentHealth when script missing' {
            Mock -CommandName Get-WSLEnvironmentHealth -ModuleName PC-AI.Virtualization -MockWith { @{ Status = 'OK' } }
            $result = Invoke-WSLDockerHealthCheck -ScriptPath (Join-Path $env:TEMP 'missing-wsl-docker.ps1')

            $result.Fallback | Should -BeTrue
            $result.FallbackMode | Should -Be 'Get-WSLEnvironmentHealth'
            $result.ExitCode | Should -Be 0
        }
    }

    Context 'Invoke-WSLNetworkToolkit fallback' {
        It 'uses health fallback for check mode' {
            Mock -CommandName Get-WSLEnvironmentHealth -ModuleName PC-AI.Virtualization -MockWith { @{ Status = 'OK' } }
            $result = Invoke-WSLNetworkToolkit -ScriptPath (Join-Path $env:TEMP 'missing-wsl-net.ps1') -Mode check

            $result.Fallback | Should -BeTrue
            $result.FallbackMode | Should -Be 'Get-WSLEnvironmentHealth'
            $result.RequestedFlags | Should -Contain 'Check'
        }

        It 'uses repair fallback for repair mode' {
            Mock -CommandName Repair-WSLNetworking -ModuleName PC-AI.Virtualization -MockWith { @{ Status = 'Repaired' } }
            $result = Invoke-WSLNetworkToolkit -ScriptPath (Join-Path $env:TEMP 'missing-wsl-net.ps1') -Mode repair

            $result.Fallback | Should -BeTrue
            $result.FallbackMode | Should -Be 'Repair-WSLNetworking'
            $result.RequestedFlags | Should -Contain 'Repair'
        }
    }
}
