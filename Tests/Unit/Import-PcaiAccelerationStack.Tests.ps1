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

# Native DLL availability assertions require pcai_core_lib.dll to be built.
# These are intentionally tagged Windows (not Portable) so they run only in full CI.
Describe 'Import-PcaiAccelerationStack - NativeDll' -Tag 'Unit', 'Acceleration', 'Bootstrap', 'Windows' {
    It 'reports FileSearchAvailable and ContentSearchAvailable when DLL is present' {
        $status = Import-PcaiAccelerationStack -Modules @('PC-AI.Acceleration') -RepoRoot $script:ProjectRoot

        $status.FileSearchAvailable | Should -BeTrue
        $status.ContentSearchAvailable | Should -BeTrue
    }
}
