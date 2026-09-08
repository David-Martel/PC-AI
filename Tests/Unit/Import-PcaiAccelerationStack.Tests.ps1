#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:CommonModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
    Import-Module $script:CommonModulePath -Force | Out-Null
}

AfterAll {
    # This suite's whole purpose is to load the acceleration stack, and it left
    # PC-AI.Acceleration in the session afterwards. Every later suite then saw it:
    # PC-AI.Performance, PC-AI.Hardware and PC-AI.Drivers prefer a NATIVE probe
    # when one is available and only fall back to Get-Process / Get-CimInstance /
    # Get-WinEvent otherwise -- which is exactly what those suites mock. With the
    # stack loaded the mocks were bypassed, the forced failure never happened, and
    # five tests reporting "Expected an exception to be thrown, but no exception
    # was thrown" stopped throwing. Each passed alone and failed in the full run.
    #
    # Bisected to this file with one child process per subset (Tests\Unit, 44
    # files: N=5 clean, N=6 reproduces). Unload what this suite loaded so the
    # session is left as it was found.
    Remove-Module 'PC-AI.Acceleration' -Force -ErrorAction SilentlyContinue
    Remove-Module 'PC-AI.Common' -Force -ErrorAction SilentlyContinue
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
