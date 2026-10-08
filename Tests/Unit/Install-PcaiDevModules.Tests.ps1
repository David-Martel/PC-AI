#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:Installer = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/Install-PcaiDevModules.ps1'
    function New-InstallFixtureModule {
        param([string]$Root, [string]$Name = 'FixtureModule', [string]$Payload = 'original')
        $path = Join-Path $Root $Name
        [void](New-Item -ItemType Directory -Path $path -Force)
        "@{ RootModule = '$Name.psm1'; ModuleVersion = '1.0.0'; FunctionsToExport = @() }" | Set-Content -LiteralPath (Join-Path $path "$Name.psd1")
        "# $Payload" | Set-Content -LiteralPath (Join-Path $path "$Name.psm1")
        [void](New-Item -ItemType Directory -Path (Join-Path $path 'fixtures') -Force)
        [IO.File]::WriteAllBytes((Join-Path $path 'fixtures/payload.bin'), [byte[]]@(0, 1, 2, 128, 255))
        return $path
    }
}

Describe 'Development module installation custody and mutation safety' {
    BeforeEach {
        $script:SavedModulePath = $env:PSModulePath
        $script:SavedCargoManifest = $env:CARGOTOOLS_MANIFEST
        $script:FixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:FixtureRepo = Join-Path $script:FixtureRoot 'repo'
        $script:FixtureInstall = Join-Path $script:FixtureRoot 'installed'
        $script:FixtureSource = New-InstallFixtureModule -Root (Join-Path $script:FixtureRepo 'Modules')
    }
    AfterEach {
        $env:PSModulePath = $script:SavedModulePath
        $env:CARGOTOOLS_MANIFEST = $script:SavedCargoManifest
    }

    It 'does no file, process or user-environment writes in <Flag>' -ForEach @(
        @{ Flag = 'WhatIf' }, @{ Flag = 'DryRun' }, @{ Flag = 'Help' }, @{ Flag = 'h' }
    ) {
        $beforeUser = [Environment]::GetEnvironmentVariable('PSModulePath', 'User')
        $arguments = @{ RepoRoot = $script:FixtureRepo; InstallRoot = $script:FixtureInstall }
        $arguments[$Flag] = $true
        & $script:Installer @arguments
        Test-Path -LiteralPath $script:FixtureInstall | Should -BeFalse
        $env:PSModulePath | Should -BeExactly $script:SavedModulePath
        [Environment]::GetEnvironmentVariable('PSModulePath', 'User') | Should -BeExactly $beforeUser
    }

    It 'accepts --help before any discovery or filesystem operation' {
        & $script:Installer -RepoRoot (Join-Path $TestDrive 'absent') -InstallRoot $script:FixtureInstall --help | Should -Match '^Usage:'
        Test-Path -LiteralPath $script:FixtureInstall | Should -BeFalse
    }

    It 'copies exact payload hashes, records custody, and is idempotent' {
        $first = & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false
        $first.State | Should -Be 'Installed'
        $receipt = Get-Content -LiteralPath $first.Receipt -Raw | ConvertFrom-Json
        $receipt.State | Should -Be 'Installed'
        $receipt.Files.Count | Should -Be 3
        foreach ($file in $receipt.Files) {
            (Get-FileHash -LiteralPath (Join-Path $first.InstalledPath $file.Path)).Hash | Should -BeExactly $file.Sha256
            (Get-FileHash -LiteralPath (Join-Path $script:FixtureSource $file.Path)).Hash | Should -BeExactly $file.Sha256
        }
        $receiptHash = (Get-FileHash -LiteralPath $first.Receipt).Hash
        $second = & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false
        $second.State | Should -Be 'Unchanged'
        (Get-FileHash -LiteralPath $first.Receipt).Hash | Should -BeExactly $receiptHash
        @(Get-ChildItem -LiteralPath (Split-Path (Split-Path $first.Receipt -Parent) -Parent) -Directory).Count | Should -Be 1
    }

    It 'retains unique previous files and unrelated user modules during replacement' {
        $old = New-InstallFixtureModule -Root $script:FixtureInstall -Payload 'old install'
        'unique WIP' | Set-Content -LiteralPath (Join-Path $old 'unique.txt')
        $uniqueHash = (Get-FileHash -LiteralPath (Join-Path $old 'unique.txt')).Hash
        $other = New-InstallFixtureModule -Root $script:FixtureInstall -Name 'UnrelatedModule'
        $otherHash = (Get-FileHash -LiteralPath (Join-Path $other 'UnrelatedModule.psm1')).Hash
        $result = & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        (Get-FileHash -LiteralPath (Join-Path $receipt.PreviousPath 'unique.txt')).Hash | Should -BeExactly $uniqueHash
        ($receipt.PreviousFiles | Where-Object Path -eq 'unique.txt').Sha256 | Should -BeExactly $uniqueHash
        (Get-FileHash -LiteralPath (Join-Path $other 'UnrelatedModule.psm1')).Hash | Should -BeExactly $otherHash
    }

    It 'restores exact previous installation after publication fails' {
        $old = New-InstallFixtureModule -Root $script:FixtureInstall -Payload 'old install'
        $oldHash = (Get-FileHash -LiteralPath (Join-Path $old 'FixtureModule.psm1')).Hash
        Mock Move-Item {
            if ($LiteralPath -like '*\staged') { throw 'Injected publication failure' }
            [IO.Directory]::Move($LiteralPath, $Destination)
        }
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*Injected publication failure*'
        (Get-FileHash -LiteralPath (Join-Path $old 'FixtureModule.psm1')).Hash | Should -BeExactly $oldHash
        $receiptPath = Join-Path $script:FixtureInstall '.pcai-module-install/FixtureModule/r1/receipt.json'
        (Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json).State | Should -Be 'RolledBack'
        Test-Path -LiteralPath (Join-Path (Split-Path $receiptPath -Parent) 'staged/FixtureModule.psm1') | Should -BeTrue
        $env:PSModulePath | Should -BeExactly $script:SavedModulePath
    }

    It 'preserves all process module roots and deduplicates the preferred root' {
        $env:PSModulePath = "$script:FixtureInstall;C:\fixture-custom;C:\fixture-system"
        & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -PSModulePathScope Process -Confirm:$false
        $env:PSModulePath | Should -BeExactly "$script:FixtureInstall;C:\fixture-custom;C:\fixture-system"
    }

    It 'retains staging and never publishes corrupted copied bytes' {
        Mock Copy-Item { [IO.File]::WriteAllText($Destination, 'corrupted staging') }
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*hashes differ*'
        Test-Path -LiteralPath (Join-Path $script:FixtureInstall 'FixtureModule') | Should -BeFalse
        $receiptPath = Join-Path $script:FixtureInstall '.pcai-module-install/FixtureModule/r1/receipt.json'
        (Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json).State | Should -Be 'RolledBack'
        Test-Path -LiteralPath (Join-Path (Split-Path $receiptPath -Parent) 'staged') | Should -BeTrue
    }

    It 'rejects an escaping manifest RootModule without publishing it' {
        "@{ RootModule = '..\escape.psm1'; ModuleVersion = '1.0.0' }" | Set-Content -LiteralPath (Join-Path $script:FixtureSource 'FixtureModule.psd1')
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*outside staged payload*'
        Test-Path -LiteralPath (Join-Path $script:FixtureInstall 'FixtureModule') | Should -BeFalse
    }

    It 'refuses publication while another installer owns its exclusive lock' {
        $old = New-InstallFixtureModule -Root $script:FixtureInstall -Payload 'old install'
        $oldHash = (Get-FileHash -LiteralPath (Join-Path $old 'FixtureModule.psm1')).Hash
        $stateRoot = Join-Path $script:FixtureInstall '.pcai-module-install/FixtureModule'
        [void](New-Item -ItemType Directory -Path $stateRoot -Force)
        $lock = [IO.File]::Open((Join-Path $stateRoot 'publish.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        try {
            { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw
            (Get-FileHash -LiteralPath (Join-Path $old 'FixtureModule.psm1')).Hash | Should -BeExactly $oldHash
        } finally { $lock.Dispose() }
    }

    It 'rejects linked child payloads without copying or modifying their targets' -Skip:(-not $IsWindows) {
        $foreign = Join-Path $script:FixtureRoot 'foreign-payload'
        [void](New-Item -ItemType Directory -Path $foreign)
        'foreign data' | Set-Content -LiteralPath (Join-Path $foreign 'data.txt')
        $hash = (Get-FileHash -LiteralPath (Join-Path $foreign 'data.txt')).Hash
        [void](New-Item -ItemType Junction -Path (Join-Path $script:FixtureSource 'linked') -Target $foreign)
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*Linked payload*'
        Test-Path -LiteralPath $script:FixtureInstall | Should -BeFalse
        (Get-FileHash -LiteralPath (Join-Path $foreign 'data.txt')).Hash | Should -BeExactly $hash
    }

    It 'uses the actual merge function to preserve existing user roots without registry mutation' {
        $tokens = $null; $errors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile($script:Installer, [ref]$tokens, [ref]$errors)
        $functionAst = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Merge-ModulePath' }, $true)
        . ([scriptblock]::Create($functionAst.Extent.Text))
        Merge-ModulePath -Existing 'C:\user-extra;C:\second-user;C:\USER-EXTRA\' -Preferred $script:FixtureInstall | Should -BeExactly "$script:FixtureInstall;C:\user-extra;C:\second-user"
    }

    It 'honors explicit CargoTools manifest precedence' {
        $cargo = New-InstallFixtureModule -Root (Join-Path $script:FixtureRoot 'cargo-source') -Name CargoTools
        $env:CARGOTOOLS_MANIFEST = Join-Path $cargo 'CargoTools.psd1'
        $results = @(& $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false)
        ($results | Where-Object Name -eq CargoTools).SourcePath | Should -BeExactly $cargo
    }

    It 'rejects source/destination overlap and nested Git before altering their contents' {
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureSource -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*overlap*'
        [void](New-Item -ItemType Directory -Path (Join-Path $script:FixtureSource 'fixtures/.git'))
        { & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*Nested Git*'
        Test-Path -LiteralPath $script:FixtureInstall | Should -BeFalse
    }

    It 'preserves a previous junction without traversing or deleting its target' -Skip:(-not $IsWindows) {
        $original = New-InstallFixtureModule -Root (Join-Path $script:FixtureRoot 'foreign') -Payload 'foreign original'
        $originalHash = (Get-FileHash -LiteralPath (Join-Path $original 'FixtureModule.psm1')).Hash
        [void](New-Item -ItemType Directory -Path $script:FixtureInstall)
        [void](New-Item -ItemType Junction -Path (Join-Path $script:FixtureInstall 'FixtureModule') -Target $original)
        $result = & $script:Installer -RepoRoot $script:FixtureRepo -InstallRoot $script:FixtureInstall -UpdatePSModulePath:$false -Confirm:$false
        $receipt = Get-Content -LiteralPath $result.Receipt -Raw | ConvertFrom-Json
        $receipt.PreviousLinkTarget | Should -BeExactly $original
        (Get-Item -LiteralPath $receipt.PreviousPath).LinkType | Should -Be Junction
        (Get-FileHash -LiteralPath (Join-Path $original 'FixtureModule.psm1')).Hash | Should -BeExactly $originalHash
    }
}
