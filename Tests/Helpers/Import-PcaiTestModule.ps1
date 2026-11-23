<#
.SYNOPSIS
    Starts a Pester file with no PC-AI module loaded but the one it tests, and leaves none behind.

.DESCRIPTION
    Why this exists: Tests/Unit runs every file in ONE PowerShell session, so a module an earlier file loaded is still
    loaded when the next file starts. Measured on origin/main 9e69f7b4 (Pester 5.7.1, Windows): the whole-folder run
    failed 166 tests, and 93 of them passed when their file ran alone. The files that failed
    (PC-AI.USB, PC-AI.Virtualization, Virtualization-Fallback, PcaiNetworkContracts, Set-LLMProviderOrder,
    PcaiLinkHealth) died on

        Multiple script or manifest modules named 'PC-AI.USB' are currently loaded. Make sure to remove any extra
        copies of the module from your session before testing.

    Pester raises that from `Mock -ModuleName`, `InModuleScope` and `Should -Invoke -ModuleName` when two loaded
    modules share the name. The measured chain (a probe in one child session: files in run order, Get-Module -All
    after each) was:

      1. An earlier file left PC-AI.Acceleration loaded from this checkout and no PC-AI.Common.
      2. PC-AI.USB.Tests.ps1 reached Resolve-PcaiRepoRoot (a PC-AI.Common command), which is not loaded, so
         PowerShell's module autoloader searched PSModulePath and imported the release package `PC-AI` installed
         under ...\PowerShell\Modules\PC-AI. That package bundles every module as a nested module, so each one was
         then loaded twice (this checkout and the installed copy), and the installed copy stayed loaded for the rest
         of the run.
      3. Every later file that mocks into a module hit the "Multiple script or manifest modules" error. The installed
         package also exports Write-Error and Write-Warning, which shadowed the ones PcaiLinkHealth.Tests.ps1
         mocks, so a test that waits for a thrown error saw none.

    `Import-Module -Force` replaces only the copy it matches by path, so a file importing its own module cannot
    repair that. Import-PcaiTestModule removes the whole family first (PC-AI, PC-AI.*, Pcai*), found with
    `Get-Module -All` so nested copies go too, then imports this checkout's manifest by absolute path. A file now
    starts as it does when run alone. Remove-PcaiTestModule is the matching AfterAll call: it also removes anything
    the autoloader pulled in while the file ran, so the next file starts clean as well.

    Dot-source this file from BeforeAll; it defines functions only and has no side effects. It uses Join-Path and
    [System.IO.Path] only, so it also runs on the Linux self-hosted gate (Tests/linux-ci-pester.txt).

.EXAMPLE
    BeforeAll {
        . (Join-Path $PSScriptRoot '..\Helpers\Import-PcaiTestModule.ps1')
        $null = Import-PcaiTestModule -Name 'PC-AI.USB'
    }
    AfterAll {
        Remove-PcaiTestModule
    }

.NOTES
    Tests/Unit/TestModuleIsolation.Tests.ps1 plants stale copies and proves the helper removes them.
#>

$script:PcaiTestRepoRoot = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $PSScriptRoot '..') '..'))

# Module names that belong to this repository's own surface. Anything else a test loads (Pester, MockData, a fixture
# module) is the test's to manage.
$script:PcaiTestModuleFamily = @('PC-AI', 'PC-AI.*', 'Pcai*')

function Remove-PcaiTestModule {
    <#
    .SYNOPSIS
        Removes loaded copies of PC-AI modules, whichever path each was imported from.
    .PARAMETER Name
        Module names or wildcards to remove. The default is the whole PC-AI family (PC-AI, PC-AI.*, Pcai*), which is
        what an AfterAll wants: it also removes a copy the autoloader imported while the file ran.
    #>
    [CmdletBinding()]
    param(
        [ValidateNotNullOrEmpty()]
        [string[]]$Name = $script:PcaiTestModuleFamily
    )

    # -All: a copy loaded as another module's nested module is not in the plain listing. A nested copy cannot be
    # removed on its own, so removing the module that nests it (matched by the same names) is what clears it.
    $loaded = @(Get-Module -All | Where-Object {
            $moduleName = $_.Name
            foreach ($pattern in $Name) { if ($moduleName -like $pattern) { return $true } }
            $false
        })
    foreach ($module in $loaded) {
        Remove-Module -ModuleInfo $module -Force -ErrorAction SilentlyContinue
    }
}

function Import-PcaiTestModule {
    <#
    .SYNOPSIS
        Removes every loaded PC-AI module, then imports this checkout's copy of one of them.
    .DESCRIPTION
        Returns the PSModuleInfo that Import-Module -PassThru gives. Throws when the repository has no such module,
        or when more than one copy of it is loaded after the import, so a failure names the cause instead of
        surfacing as a "Multiple script or manifest modules" error several tests later.
    .PARAMETER Name
        Directory name under Modules\ (PC-AI.USB, PC-AI.Network, PcaiInference ...).
    .PARAMETER Path
        Absolute path of the manifest or script module to import instead of Modules\<Name>\<Name>.psd1. Use it for a
        copy that is deliberately elsewhere, such as a relocated source tree.
    .PARAMETER Alongside
        Extra module names (wildcards allowed) to remove first, for modules outside the PC-AI family that this file
        replaces.
    .PARAMETER KeepLoaded
        Remove only the named module and -Alongside, not the family. Use it when the file needs another PC-AI module
        loaded first, and say why at the call site.
    #>
    [CmdletBinding()]
    [OutputType([System.Management.Automation.PSModuleInfo])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$Name,

        [string]$Path,

        [string[]]$Alongside = @(),

        [switch]$KeepLoaded
    )

    if (-not $Path) {
        $moduleDir = Join-Path (Join-Path $script:PcaiTestRepoRoot 'Modules') $Name
        $candidates = @((Join-Path $moduleDir "$Name.psd1"), (Join-Path $moduleDir "$Name.psm1"))
        $Path = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (-not $Path) {
            throw "Import-PcaiTestModule: neither $($candidates[0]) nor $($candidates[1]) exists."
        }
    }
    $Path = [System.IO.Path]::GetFullPath($Path)

    $remove = if ($KeepLoaded) { @($Name) + $Alongside } else { @($script:PcaiTestModuleFamily) + @($Name) + $Alongside }
    Remove-PcaiTestModule -Name $remove

    # -Global: Import-Module inside a function would otherwise bind the commands to this function's scope and the
    # test file would not see them. Remove-PcaiTestModule in AfterAll takes them out again.
    $module = Import-Module -Name $Path -Force -Global -PassThru -ErrorAction Stop
    $copies = @(Get-Module -All -Name $module.Name)
    if ($copies.Count -ne 1) {
        throw "Import-PcaiTestModule: $($copies.Count) copies of $($module.Name) are loaded after the import: $(($copies | ForEach-Object Path) -join '; ')"
    }
    $module
}
