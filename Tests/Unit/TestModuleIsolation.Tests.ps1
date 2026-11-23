<#
.SYNOPSIS
    Proves Tests/Helpers/Import-PcaiTestModule.ps1 leaves exactly one copy of a module loaded.

.DESCRIPTION
    Tests/Unit runs every file in one PowerShell session. A module an earlier file loaded from another path used to
    stay loaded, and the next file's `Mock -ModuleName` then failed with "Multiple script or manifest modules named
    ... are currently loaded" (93 of 166 failures in the whole-folder run of origin/main 9e69f7b4 passed when their
    file ran alone). These tests plant the second copy on purpose, so the helper is proved against the failure it
    exists to prevent rather than against a clean session. Synthetic modules keep the file independent of the host.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot '..\Helpers\Import-PcaiTestModule.ps1')

    function New-ProbeModule {
        # A throwaway module directory named like a repository module; the marker says which copy answered.
        param([string]$Root, [string]$Name, [string]$Marker)
        $dir = Join-Path $Root $Name
        $null = New-Item -ItemType Directory -Path $dir -Force
        Set-Content -LiteralPath (Join-Path $dir "$Name.psm1") -Value "function Get-ProbeMarker { '$Marker' }`nExport-ModuleMember -Function Get-ProbeMarker"
        Join-Path $dir "$Name.psm1"
    }
}

AfterAll {
    Remove-PcaiTestModule
}

Describe 'Import-PcaiTestModule' -Tag 'Unit', 'Portable' {
    BeforeEach {
        Remove-PcaiTestModule
    }

    Context 'when an earlier file left a copy loaded from another path' {
        # Protects: the premise of every test below (a positive control): two loaded copies of one module name really do make `Mock -ModuleName` throw.
        # Detects: a Pester or PowerShell change that stops refusing duplicates, which would turn every isolation test below into a test of nothing.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'the stale copy makes Pester refuse a module-scoped mock (the failure being prevented)' {
            $stale = New-ProbeModule -Root (Join-Path $TestDrive 'stale') -Name 'PcaiIsolationProbe' -Marker 'stale'
            $other = New-ProbeModule -Root (Join-Path $TestDrive 'other') -Name 'PcaiIsolationProbe' -Marker 'other'
            Import-Module $stale -Global -Force
            Import-Module $other -Global -Force
            @(Get-Module -All -Name 'PcaiIsolationProbe').Count | Should -BeGreaterThan 1 -Because 'the fixture must reproduce two copies, or the next test proves nothing'
            { Mock Get-Date { } -ModuleName 'PcaiIsolationProbe' } | Should -Throw '*Multiple script or manifest modules*'
        }

        # Protects: a test file starts with one copy of its module, whichever path an earlier file imported it from.
        # Detects: a helper that replaces only the copy it matches by path (`Import-Module -Force`), which leaves Pester two modules to choose from and fails 93 tests in the whole-folder run.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'leaves exactly one copy, the one imported from the requested path' {
            $stale = New-ProbeModule -Root (Join-Path $TestDrive 'stale') -Name 'PcaiIsolationProbe' -Marker 'stale'
            $canonical = New-ProbeModule -Root (Join-Path $TestDrive 'canonical') -Name 'PcaiIsolationProbe' -Marker 'canonical'
            Import-Module $stale -Global -Force

            $module = Import-PcaiTestModule -Name 'PcaiIsolationProbe' -Path $canonical

            $copies = @(Get-Module -All -Name 'PcaiIsolationProbe')
            $copies.Count | Should -Be 1
            $copies[0].Path | Should -Be ([System.IO.Path]::GetFullPath($canonical))
            $module.Path | Should -Be $copies[0].Path
            Get-ProbeMarker | Should -Be 'canonical'
        }

        # Protects: `Mock -ModuleName` works after the helper, which is the behaviour the six adopting files depend on.
        # Detects: a helper that imports cleanly but leaves a second copy, so the mock fails with "Multiple script or manifest modules".
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'lets a module-scoped mock resolve after the import' {
            $stale = New-ProbeModule -Root (Join-Path $TestDrive 'stale') -Name 'PcaiIsolationProbe' -Marker 'stale'
            $canonical = New-ProbeModule -Root (Join-Path $TestDrive 'canonical') -Name 'PcaiIsolationProbe' -Marker 'canonical'
            Import-Module $stale -Global -Force
            $null = Import-PcaiTestModule -Name 'PcaiIsolationProbe' -Path $canonical
            $refused = $null
            try { Mock Get-Date { } -ModuleName 'PcaiIsolationProbe' } catch { $refused = $_.Exception.Message }
            $refused | Should -BeNullOrEmpty
        }

        # Protects: a file starts with no PC-AI module loaded but its own; a leftover PC-AI.Acceleration without PC-AI.Common is what sent the autoloader to the installed `PC-AI` package.
        # Detects: a helper that clears only the named module, which leaves the leftover that triggers the autoload and the duplicate nested copies.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'removes other PC-AI family modules too, so the file starts as it does when run alone' {
            # PC-AI.Acceleration left loaded without PC-AI.Common is what sent the autoloader to the installed package.
            $other = New-ProbeModule -Root (Join-Path $TestDrive 'family') -Name 'PC-AI.IsolationProbeOther' -Marker 'other'
            $canonical = New-ProbeModule -Root (Join-Path $TestDrive 'canonical') -Name 'PcaiIsolationProbe' -Marker 'canonical'
            try {
                Import-Module $other -Global -Force
                $null = Import-PcaiTestModule -Name 'PcaiIsolationProbe' -Path $canonical
                Get-Module -All -Name 'PC-AI.IsolationProbeOther' | Should -BeNullOrEmpty
            } finally {
                Remove-PcaiTestModule
            }
        }

        # Protects: the documented opt-out: a file that needs a sibling module loaded keeps it.
        # Detects: -KeepLoaded silently clearing the family, which would break a file that relies on it.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'keeps the other family modules when -KeepLoaded is given' {
            $other = New-ProbeModule -Root (Join-Path $TestDrive 'family') -Name 'PC-AI.IsolationProbeOther' -Marker 'other'
            $canonical = New-ProbeModule -Root (Join-Path $TestDrive 'canonical') -Name 'PcaiIsolationProbe' -Marker 'canonical'
            try {
                Import-Module $other -Global -Force
                $null = Import-PcaiTestModule -Name 'PcaiIsolationProbe' -Path $canonical -KeepLoaded
                @(Get-Module -All -Name 'PC-AI.IsolationProbeOther').Count | Should -Be 1
            } finally {
                Remove-PcaiTestModule
            }
        }

        # Protects: fixture modules and Pester stay loaded, and -Alongside names a non-family module to clear.
        # Detects: the family wildcard matching too much (a fixture or Pester module removed mid-run) or -Alongside being ignored.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'leaves modules outside the family alone, and removes -Alongside names' {
            $a = New-ProbeModule -Root (Join-Path $TestDrive 'a') -Name 'PcaiIsolationProbe' -Marker 'a'
            $outside = New-ProbeModule -Root (Join-Path $TestDrive 'outside') -Name 'IsolationOutsider' -Marker 'outside'
            $named = New-ProbeModule -Root (Join-Path $TestDrive 'named') -Name 'IsolationNamed' -Marker 'named'
            try {
                Import-Module $outside -Global -Force
                Import-Module $named -Global -Force
                $null = Import-PcaiTestModule -Name 'PcaiIsolationProbe' -Path $a -Alongside 'IsolationNamed'
                @(Get-Module -All -Name 'IsolationOutsider').Count | Should -Be 1
                Get-Module -All -Name 'IsolationNamed' | Should -BeNullOrEmpty
            } finally {
                Remove-Module IsolationOutsider, IsolationNamed -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Context 'when the module does not exist' {
        # Protects: a missing module is reported by name, not as a Pester error several tests later.
        # Detects: a helper that returns nothing or a generic error for a typo in -Name.
        # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'throws naming the manifest it looked for' {
            { Import-PcaiTestModule -Name 'PC-AI.DoesNotExist' } | Should -Throw '*PC-AI.DoesNotExist.psd1*'
        }
    }

    Context 'when the module is a repository module' {
        # Protects: the helper resolves THIS checkout's module, not a copy elsewhere on PSModulePath.
        # Detects: a helper that falls back to name-based import, which would load the installed release package.
        # Needs: Pester 5 and the repository module Modules/PC-AI.Common; no hardware or admin rights; runs on Windows and Linux.
        # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Import-PcaiTestModule.
        It 'imports the manifest under this checkout''s Modules directory' {
            $repoModules = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\Modules'))
            try {
                $module = Import-PcaiTestModule -Name 'PC-AI.Common'
                $module.Path.StartsWith($repoModules, [System.StringComparison]::OrdinalIgnoreCase) | Should -BeTrue
                @(Get-Module -All -Name 'PC-AI.Common').Count | Should -Be 1
            } finally {
                Remove-PcaiTestModule -Name 'PC-AI.Common'
            }
        }
    }
}

Describe 'Remove-PcaiTestModule' -Tag 'Unit', 'Portable' {
    # Protects: a file that imported without -Global (most of the suite) is cleaned up as well.
    # Detects: a removal that only reaches the global scope, leaving the file's own import for the next file.
    # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
    # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Remove-PcaiTestModule.
    It 'removes a copy imported into the test scope without -Global' {
        $local = New-ProbeModule -Root (Join-Path $TestDrive 'local') -Name 'PcaiIsolationProbe' -Marker 'local'
        Import-Module $local -Force
        Remove-PcaiTestModule
        Get-Module -All -Name 'PcaiIsolationProbe' | Should -BeNullOrEmpty
    }

    # Protects: AfterAll leaves nothing for the next file, whichever way each copy was imported.
    # Detects: a removal that stops at the first copy of a name.
    # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
    # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Remove-PcaiTestModule.
    It 'removes a global copy and a path-distinct copy in one call' {
        $one = New-ProbeModule -Root (Join-Path $TestDrive 'one') -Name 'PcaiIsolationProbe' -Marker 'one'
        $two = New-ProbeModule -Root (Join-Path $TestDrive 'two') -Name 'PcaiIsolationProbe' -Marker 'two'
        Import-Module $one -Global -Force
        Import-Module $two -Global -Force
        Remove-PcaiTestModule -Name 'PcaiIsolationProbe'
        Get-Module -All -Name 'PcaiIsolationProbe' | Should -BeNullOrEmpty
    }

    # Protects: an explicit -Name removes exactly that name; a typo or an absent module is harmless.
    # Detects: a removal that ignores -Name and clears the family.
    # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
    # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Remove-PcaiTestModule.
    It 'removes only the names it is given, and a name that is not loaded removes nothing' {
        $one = New-ProbeModule -Root (Join-Path $TestDrive 'one') -Name 'PcaiIsolationProbe' -Marker 'one'
        Import-Module $one -Global -Force
        Remove-PcaiTestModule -Name 'PcaiNeverLoadedModule'
        @(Get-Module -All -Name 'PcaiIsolationProbe').Count | Should -Be 1
    }

    # Protects: the AfterAll default removes the PC-AI family only.
    # Detects: the default pattern matching a fixture or Pester module and breaking the rest of the run.
    # Needs: Pester 5, TestDrive and synthetic modules only; no host module, hardware or admin rights; runs on Windows and Linux.
    # Breadcrumb: Tests/Helpers/Import-PcaiTestModule.ps1::Remove-PcaiTestModule.
    It 'leaves a module outside the family loaded when called with no name' {
        $outside = New-ProbeModule -Root (Join-Path $TestDrive 'outside') -Name 'IsolationOutsider' -Marker 'outside'
        try {
            Import-Module $outside -Global -Force
            Remove-PcaiTestModule
            @(Get-Module -All -Name 'IsolationOutsider').Count | Should -Be 1
        } finally {
            Remove-Module IsolationOutsider -Force -ErrorAction SilentlyContinue
        }
    }
}
