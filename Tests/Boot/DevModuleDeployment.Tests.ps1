#Requires -Version 7.0

BeforeAll {
    $script:installer = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Install-PcaiDevModules.ps1'
    . $script:installer -Help | Out-Null

    function New-TestDevModule {
        param([string]$Root, [string]$Name, [string]$Value = 'new')
        $path = Join-Path $Root $Name
        [IO.Directory]::CreateDirectory($path) | Out-Null
        [IO.File]::WriteAllText((Join-Path $path "$Name.psd1"), "@{ RootModule='$Name.psm1'; ModuleVersion='1.0.0'; FileList=@('$Name.psm1','payload.txt') }")
        [IO.File]::WriteAllText((Join-Path $path "$Name.psm1"), "function Get-FixtureValue { '$Value' }")
        [IO.File]::WriteAllText((Join-Path $path 'payload.txt'), $Value)
        return $path
    }
}

Describe 'Scoped module publication with recovery' {
    BeforeEach {
        $script:caseRoot = Join-Path $TestDrive ([Guid]::NewGuid().ToString('N'))
        $script:repo = Join-Path $caseRoot 'repo'
        $script:modules = Join-Path $repo 'Modules'
        $script:install = Join-Path $caseRoot 'installed'
        [IO.Directory]::CreateDirectory($modules) | Out-Null
        $script:oldProcessPath = $env:PSModulePath
    }
    AfterEach { $env:PSModulePath = $script:oldProcessPath }

    It 'copies only the selected module, keeps source immutable, and retains the old payload' {
        $source = New-TestDevModule $modules Alpha
        New-TestDevModule $modules Beta | Out-Null
        New-TestDevModule $install Alpha 'old' | Out-Null
        New-TestDevModule $install Beta 'untouched' | Out-Null
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false
        $result.Status | Should -Be 'Installed'
        $result.Mode | Should -Be 'Copy'
        [IO.File]::ReadAllText((Join-Path $install 'Alpha\payload.txt')) | Should -Be 'new'
        [IO.File]::ReadAllText((Join-Path $install 'Beta\payload.txt')) | Should -Be 'untouched'
        [IO.File]::ReadAllText((Join-Path $source 'payload.txt')) | Should -Be 'new'
        [IO.File]::ReadAllText((Join-Path $result.BackupPath 'payload.txt')) | Should -Be 'old'
        $result.BackupPath | Should -BeLike "$install\.pcai-module-backups\*\Alpha"
    }

    It 'deep merges installed CargoTools host overrides with new defaults and preserves other Config files' {
        $source = New-TestDevModule (Join-Path $caseRoot 'external') CargoTools
        $old = New-TestDevModule $install CargoTools 'old'
        [IO.Directory]::CreateDirectory((Join-Path $source 'Config')) | Out-Null
        [IO.Directory]::CreateDirectory((Join-Path $old 'Config\nested')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $source 'Config\machines.json'), '{"NUC":{"BuildJobs":4,"NewDefault":true},"NewHost":{"BuildJobs":3}}')
        [IO.File]::WriteAllText((Join-Path $old 'Config\machines.json'), '{"NUC":{"BuildJobs":2,"UserKey":"keep"},"OldHost":{"BuildJobs":1}}')
        [IO.File]::WriteAllText((Join-Path $old 'Config\nested\secret.config'), 'preserve')
        [IO.File]::WriteAllText((Join-Path $old 'Config\only-installed.json'), '{"Custom":true}')
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames CargoTools -CargoToolsSourcePath (Join-Path $source 'CargoTools.psd1') -UpdatePSModulePath:$false
        $config = [IO.File]::ReadAllText((Join-Path $result.InstalledPath 'Config\machines.json')) | ConvertFrom-Json
        $config.NUC.BuildJobs | Should -Be 2
        $config.NUC.NewDefault | Should -BeTrue
        $config.NUC.UserKey | Should -Be 'keep'
        $config.NewHost.BuildJobs | Should -Be 3
        $config.OldHost.BuildJobs | Should -Be 1
        [IO.File]::ReadAllText((Join-Path $result.InstalledPath 'Config\nested\secret.config')) | Should -Be 'preserve'
        ([IO.File]::ReadAllText((Join-Path $result.InstalledPath 'Config\only-installed.json')) | ConvertFrom-Json).Custom | Should -BeTrue
        ([IO.File]::ReadAllText((Join-Path $source 'Config\machines.json')) | ConvertFrom-Json).NUC.BuildJobs | Should -Be 4
    }

    It 'stages every selected module before replacing any active module' {
        New-TestDevModule $modules Alpha | Out-Null
        $broken = New-TestDevModule $modules Beta
        New-TestDevModule $install Alpha 'old-alpha' | Out-Null
        New-TestDevModule $install Beta 'old-beta' | Out-Null
        [IO.File]::WriteAllText((Join-Path $broken 'Beta.psm1'), 'function Broken {')
        { & $installer -RepoRoot $repo -InstallRoot $install -UpdatePSModulePath:$false } | Should -Throw '*Script syntax failed*'
        [IO.File]::ReadAllText((Join-Path $install 'Alpha\payload.txt')) | Should -Be 'old-alpha'
        [IO.File]::ReadAllText((Join-Path $install 'Beta\payload.txt')) | Should -Be 'old-beta'
        @(Get-ChildItem $install -Directory -Filter '.pcai-stage-*').Count | Should -Be 1
        [IO.Directory]::Exists((Join-Path $install '.pcai-module-backups')) | Should -BeFalse
    }

    It 'rolls back an already published module if another target cannot be published' {
        New-TestDevModule $modules Alpha | Out-Null
        New-TestDevModule $modules Beta | Out-Null
        New-TestDevModule $install Alpha 'original' | Out-Null
        [IO.File]::WriteAllText((Join-Path $install 'Beta'), 'unrelated file')
        { & $installer -RepoRoot $repo -InstallRoot $install -UpdatePSModulePath:$false } | Should -Throw '*Installed target is a file*'
        [IO.File]::ReadAllText((Join-Path $install 'Alpha\payload.txt')) | Should -Be 'original'
        [IO.File]::ReadAllText((Join-Path $install 'Beta')) | Should -Be 'unrelated file'
        $stage = Get-ChildItem $install -Directory -Filter '.pcai-stage-*'
        [IO.File]::ReadAllText((Join-Path $stage.FullName 'Alpha.failed\payload.txt')) | Should -Be 'new'
        $env:PSModulePath | Should -Be $oldProcessPath
    }

    It 'does not publish malformed installed Config' {
        New-TestDevModule $modules Alpha | Out-Null
        $old = New-TestDevModule $install Alpha 'original'
        [IO.Directory]::CreateDirectory((Join-Path $old 'Config')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $old 'Config\machines.json'), 'invalid JSON')
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*Deployment failed*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
        [IO.File]::ReadAllText((Join-Path $old 'Config\machines.json')) | Should -Be 'invalid JSON'
    }

    It 'rejects manifest files outside the staged module' {
        $source = New-TestDevModule $modules Alpha
        [IO.File]::WriteAllText((Join-Path $source 'Alpha.psd1'), "@{ RootModule='Alpha.psm1'; ModuleVersion='1.0'; FileList=@('..\outside.txt') }")
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*outside the staged module*'
        [IO.Directory]::Exists((Join-Path $install 'Alpha')) | Should -BeFalse
    }

    It 'rejects unsafe deployment paths before creating any files' -ForEach @(
        @{ Leaf = 'NUL.txt' }, @{ Leaf = 'AUX. ' }, @{ Leaf = '$null' },
        @{ Leaf = 'COM1.log' }, @{ Leaf = 'LPT9' }
    ) {
        New-TestDevModule $modules Alpha | Out-Null
        $badRoot = Join-Path $caseRoot $Leaf
        { & $installer -RepoRoot $repo -InstallRoot $badRoot -ModuleNames Alpha } | Should -Throw '*Reserved path component*'
        [IO.Directory]::Exists($install) | Should -BeFalse
    }

    It 'rejects relative roots and overlapping backup or source payloads' {
        $source = New-TestDevModule $modules Alpha
        { & $installer -RepoRoot $repo -InstallRoot 'relative' -ModuleNames Alpha } | Should -Throw '*absolute*'
        { & $installer -RepoRoot $repo -InstallRoot $source -ModuleNames Alpha } | Should -Throw '*must not overlap*'
        { & $installer -RepoRoot $repo -InstallRoot $install -BackupRoot (Join-Path $install 'Alpha\backups') -ModuleNames Alpha } | Should -Throw '*must not overlap*'
        { & $installer -RepoRoot $repo -InstallRoot ($install + ':stream') -ModuleNames Alpha } | Should -Throw '*alternate data streams*'
        [IO.Directory]::Exists($install) | Should -BeFalse
    }

    # Protect the selected source/installation/backup identity before Windows or
    # Join-Path can erase an ordinary component's trailing dot or space. These
    # real-copy cases use only TestDrive payloads; no global module path is changed.
    It 'rejects aliased <PathArgument> before any publication IO (<SuffixName>)' -ForEach @(
        @{ PathArgument = 'RepoRoot'; Suffix = ' '; SuffixName = 'space' },
        @{ PathArgument = 'RepoRoot'; Suffix = '.'; SuffixName = 'dot' },
        @{ PathArgument = 'InstallRoot'; Suffix = ' '; SuffixName = 'space' },
        @{ PathArgument = 'InstallRoot'; Suffix = '.'; SuffixName = 'dot' },
        @{ PathArgument = 'BackupRoot'; Suffix = ' '; SuffixName = 'space' },
        @{ PathArgument = 'BackupRoot'; Suffix = '.'; SuffixName = 'dot' },
        @{ PathArgument = 'CargoToolsSourcePath'; Suffix = ' '; SuffixName = 'space' },
        @{ PathArgument = 'CargoToolsSourcePath'; Suffix = '.'; SuffixName = 'dot' }
    ) {
        New-TestDevModule $modules Alpha | Out-Null
        New-TestDevModule $install Alpha 'original' | Out-Null
        $arguments = @{ RepoRoot = $repo; InstallRoot = $install; ModuleNames = @('Alpha'); UpdatePSModulePath = $false }
        switch ($PathArgument) {
            RepoRoot { $arguments.RepoRoot = $repo + $Suffix }
            InstallRoot { $arguments.InstallRoot = $install + $Suffix }
            BackupRoot {
                $backupParent = Join-Path $caseRoot 'backups'
                [IO.Directory]::CreateDirectory($backupParent) | Out-Null
                $arguments.BackupRoot = ($backupParent + $Suffix) + '\saved'
            }
            CargoToolsSourcePath {
                $cargo = New-TestDevModule (Join-Path $caseRoot 'external') CargoTools
                $arguments.ModuleNames = @('CargoTools')
                $arguments.CargoToolsSourcePath = $cargo + $Suffix
            }
        }
        $before = @(Get-ChildItem -LiteralPath $caseRoot -Recurse -Force | Select-Object -ExpandProperty FullName | Sort-Object)
        { & $installer @arguments } | Should -Throw '*must not end with a dot or space*'
        @(Get-ChildItem -LiteralPath $caseRoot -Recurse -Force | Select-Object -ExpandProperty FullName | Sort-Object) | Should -Be $before
        [IO.File]::ReadAllText((Join-Path $install 'Alpha\payload.txt')) | Should -Be 'original'
        $env:PSModulePath | Should -Be $oldProcessPath
    }

    # Keep ordinary names and explicit navigation usable: only a named component
    # ending in a dot/space is ambiguous; spaces/dots inside names and . / .. are valid.
    It 'publishes ordinary spaced names through dot navigation without changing source or module paths' {
        $source = New-TestDevModule $modules Alpha
        $normalInstall = Join-Path $caseRoot 'installed modules.normal'
        New-TestDevModule $normalInstall Alpha 'original' | Out-Null
        $result = & $installer -RepoRoot $repo -InstallRoot ($normalInstall + '\child\..\.') -ModuleNames Alpha -UpdatePSModulePath:$false
        $result.Status | Should -Be 'Installed'
        [IO.File]::ReadAllText((Join-Path $normalInstall 'Alpha\payload.txt')) | Should -Be 'new'
        [IO.File]::ReadAllText((Join-Path $result.BackupPath 'payload.txt')) | Should -Be 'original'
        [IO.File]::ReadAllText((Join-Path $source 'payload.txt')) | Should -Be 'new'
        $env:PSModulePath | Should -Be $oldProcessPath
    }

    It 'keeps DryRun and WhatIf free of file and environment writes' {
        New-TestDevModule $modules Alpha | Out-Null
        $before = @(Get-ChildItem $caseRoot -Recurse -Force | Select-Object -ExpandProperty FullName)
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -DryRun
        $result.Status | Should -Be 'DryRun'
        & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -WhatIf
        @(Get-ChildItem $caseRoot -Recurse -Force | Select-Object -ExpandProperty FullName) | Should -Be $before
        [IO.Directory]::Exists($install) | Should -BeFalse
        $env:PSModulePath | Should -Be $oldProcessPath
    }

    It 'supports both help aliases without deployment writes' {
        & $installer -RepoRoot $repo -InstallRoot $install -h | Out-Null
        & $installer -RepoRoot $repo -InstallRoot $install --help | Out-Null
        [IO.Directory]::Exists($install) | Should -BeFalse
        $env:PSModulePath | Should -Be $oldProcessPath
    }

    It 'preserves every module path entry while prepending and deduplicating the root' {
        Get-MergedPsModulePath -PreferredRoot 'C:\Modules' -CurrentValue 'C:\First;c:\modules\;C:\Second;C:\FIRST;' |
            Should -Be 'C:\Modules;C:\First;C:\Second'
        New-TestDevModule $modules Alpha | Out-Null
        $env:PSModulePath = 'C:\First;C:\Second'
        & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -PSModulePathScope Process | Out-Null
        $env:PSModulePath | Should -Be "$install;C:\First;C:\Second"
    }

    It 'allows one explicit repo-local development junction and retains the replaced payload' {
        $source = New-TestDevModule $modules Alpha
        New-TestDevModule $install Alpha 'old' | Out-Null
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -Mode Junction -UpdatePSModulePath:$false
        (Get-Item $result.InstalledPath).LinkType | Should -Be 'Junction'
        [IO.File]::ReadAllText((Join-Path $result.InstalledPath 'payload.txt')) | Should -Be 'new'
        [IO.File]::ReadAllText((Join-Path $result.BackupPath 'payload.txt')) | Should -Be 'old'
        [IO.File]::ReadAllText((Join-Path $source 'payload.txt')) | Should -Be 'new'
    }

    It 'refuses a junction when installed Config would require mutating the source' {
        New-TestDevModule $modules Alpha | Out-Null
        $old = New-TestDevModule $install Alpha 'old'
        [IO.Directory]::CreateDirectory((Join-Path $old 'Config')) | Out-Null
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -Mode Junction } | Should -Throw '*Use Copy*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'old'
    }

    It 'rejects a destination reached through a junction without touching its external files' {
        New-TestDevModule $modules Alpha | Out-Null
        $outside = Join-Path $caseRoot 'outside'
        [IO.Directory]::CreateDirectory($outside) | Out-Null
        [IO.File]::WriteAllText((Join-Path $outside 'keep.txt'), 'outside data')
        New-Item -ItemType Junction -Path $install -Target $outside | Out-Null
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha } | Should -Throw '*Destination ancestor is a reparse point*'
        [IO.File]::ReadAllText((Join-Path $outside 'keep.txt')) | Should -Be 'outside data'
        [IO.Directory]::Exists((Join-Path $outside 'Alpha')) | Should -BeFalse
    }

    It 'rejects source payload links before replacing the installed module' {
        $source = New-TestDevModule $modules Alpha
        $old = New-TestDevModule $install Alpha 'original'
        $outside = Join-Path $caseRoot 'outside'
        [IO.Directory]::CreateDirectory($outside) | Out-Null
        New-Item -ItemType Junction -Path (Join-Path $source 'linked') -Target $outside | Out-Null
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*Source reparse points are unsupported*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
    }

    It 'rejects a malformed child script and leaves the old installed payload intact' {
        $source = New-TestDevModule $modules Alpha
        $old = New-TestDevModule $install Alpha 'original'
        [IO.Directory]::CreateDirectory((Join-Path $source 'Public')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $source 'Public\Broken.ps1'), 'function Broken {')
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*Script syntax failed*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
    }

    It 'rejects escaping manifest path references without replacing the old module' -ForEach @(
        @{ Field = 'ScriptsToProcess' }, @{ Field = 'TypesToProcess' },
        @{ Field = 'FormatsToProcess' }, @{ Field = 'RequiredAssemblies' },
        @{ Field = 'NestedModules' }
    ) {
        $source = New-TestDevModule $modules Alpha
        $old = New-TestDevModule $install Alpha 'original'
        [IO.File]::WriteAllText((Join-Path $source 'Alpha.psd1'), "@{ RootModule='Alpha.psm1'; ModuleVersion='1.0'; $Field=@('..\outside.dll') }")
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*outside the staged module*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
    }

    It 'allows named module dependencies and assemblies while checking valid local references' {
        $source = New-TestDevModule $modules Alpha
        [IO.File]::WriteAllText((Join-Path $source 'Alpha.psd1'), "@{ RootModule='Alpha.psm1'; ModuleVersion='1.0'; NestedModules=@('Microsoft.PowerShell.Management',@{ModuleName='Microsoft.PowerShell.Utility';ModuleVersion='7.0'}); RequiredAssemblies=@('System.Xml'); ScriptsToProcess=@('init.ps1') }")
        [IO.File]::WriteAllText((Join-Path $source 'init.ps1'), '# This file is parsed without execution.')
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false
        $result.Status | Should -Be 'Installed'
    }

    It 'validates child data files without executing them' {
        $source = New-TestDevModule $modules Alpha
        $old = New-TestDevModule $install Alpha 'original'
        [IO.File]::WriteAllText((Join-Path $source 'settings.psd1'), '@{ Invalid = Get-Process }')
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*Deployment failed*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
    }

    It 'preserves an unused platform manifest and reports its stale FileList without blocking the active module' {
        $source = New-TestDevModule $modules Alpha
        [IO.File]::WriteAllText((Join-Path $source 'Alpha-other-host.psd1'), "@{ RootModule='missing.psm1'; ModuleVersion='1.0'; FileList=@('missing.help.txt') }")
        $result = & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false
        $result.Status | Should -Be 'Installed'
        $result.InactiveManifestDiagnostics.Count | Should -Be 1
        $result.InactiveManifestDiagnostics[0] | Should -Match 'Inactive manifest.*stale FileList'
        [IO.File]::Exists((Join-Path $result.InstalledPath 'Alpha-other-host.psd1')) | Should -BeTrue
    }

    It 'validates a referenced nested manifest and rejects its missing initialization script' {
        $source = New-TestDevModule $modules Alpha
        $old = New-TestDevModule $install Alpha 'original'
        [IO.File]::WriteAllText((Join-Path $source 'Alpha.psd1'), "@{ RootModule='Alpha.psm1'; ModuleVersion='1.0'; NestedModules=@('nested.psd1') }")
        [IO.File]::WriteAllText((Join-Path $source 'nested.psd1'), "@{ ModuleVersion='1.0'; ScriptsToProcess=@('missing.ps1') }")
        { & $installer -RepoRoot $repo -InstallRoot $install -ModuleNames Alpha -UpdatePSModulePath:$false } | Should -Throw '*ScriptsToProcess*'
        [IO.File]::ReadAllText((Join-Path $old 'payload.txt')) | Should -Be 'original'
    }
}
