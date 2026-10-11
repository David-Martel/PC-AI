#Requires -Version 7.0
BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.Common/Public/Get-PcaiRuntimeConfig.ps1')
}
Describe 'Packaged runtime root discovery' {
    It 'stops at a unified module manifest with Config even inside a checkout' {
        $checkout = Join-Path $TestDrive 'checkout'
        $package = Join-Path $checkout 'Release/PowerShell/PC-AI'
        $component = Join-Path $package 'Modules/PC-AI.LLM'
        [IO.Directory]::CreateDirectory($component) | Out-Null
        [IO.Directory]::CreateDirectory((Join-Path $package 'Config')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $checkout 'AGENTS.md'), 'fixture checkout')
        [IO.File]::WriteAllText((Join-Path $package 'PC-AI.psd1'), '@{}')
        Resolve-PcaiRepoRoot -StartPath $component | Should -Be $package
    }
    It 'requires the Config directory before treating a module manifest as a runtime root' {
        $checkout = Join-Path $TestDrive 'second-checkout'
        $component = Join-Path $checkout 'module'
        [IO.Directory]::CreateDirectory($component) | Out-Null
        [IO.File]::WriteAllText((Join-Path $checkout 'AGENTS.md'), 'fixture checkout')
        [IO.File]::WriteAllText((Join-Path $component 'PC-AI.psd1'), '@{}')
        Resolve-PcaiRepoRoot -StartPath $component | Should -Be $checkout
    }
}
