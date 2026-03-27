#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:CommonModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
    Import-Module $script:CommonModulePath -Force | Out-Null
}

Describe 'Import-PcaiAccelerationStack' -Tag 'Unit', 'Acceleration', 'Bootstrap', 'Portable' {
    It 'imports repo-local acceleration modules through the shared bootstrap' {
        $status = Import-PcaiAccelerationStack -Modules @('PC-AI.Acceleration') -RepoRoot $script:ProjectRoot

        $status | Should -Not -BeNullOrEmpty
        $status.RepoRoot | Should -Be $script:ProjectRoot
        $status.Modules.PSObject.Properties.Name | Should -Contain 'PC-AI.Acceleration'
        $status.Modules.'PC-AI.Acceleration'.Available | Should -BeTrue
        $status.FileSearchAvailable | Should -BeTrue
        $status.ContentSearchAvailable | Should -BeTrue
    }

    It 'reports missing modules without throwing when RequireAll is not set' {
        $status = Import-PcaiAccelerationStack -Modules @('DefinitelyMissingAccelerationModule') -RepoRoot $script:ProjectRoot

        $status.Modules.'DefinitelyMissingAccelerationModule'.Available | Should -BeFalse
        $status.Modules.'DefinitelyMissingAccelerationModule'.Source | Should -Be 'missing'
    }

    It 'throws when a required acceleration module cannot be loaded' {
        {
            Import-PcaiAccelerationStack -Modules @('DefinitelyMissingAccelerationModule') -RepoRoot $script:ProjectRoot -RequireAll
        } | Should -Throw
    }
}
