#Requires -Version 5.1

BeforeAll {
    # The helper removes every module an earlier file left loaded first; see its header.
    . (Join-Path $PSScriptRoot '..\Helpers\Import-PcaiTestModule.ps1')
    $null = Import-PcaiTestModule -Name 'PC-AI.Virtualization'
}

AfterAll {
    # This file used to leave PC-AI.Virtualization loaded for every later file.
    Remove-PcaiTestModule
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
