#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
BeforeAll {
    $script:ExistingInstaller = Join-Path $PSScriptRoot '../../Tools/Install-PcaiDevModules.ps1'
    function New-TransactionFixtureModule([string]$Root, [string]$Name, [string]$Marker='canonical synthetic source') {
        $path = Join-Path $Root $Name
        $null = New-Item -ItemType Directory -Path $path -Force
        "@{ RootModule='$Name.psm1'; ModuleVersion='1.0.0'; FunctionsToExport=@() }" | Set-Content -LiteralPath (Join-Path $path "$Name.psd1")
        "# $Marker" | Set-Content -LiteralPath (Join-Path $path "$Name.psm1")
        [IO.File]::WriteAllBytes((Join-Path $path 'fixture.bin'), [byte[]]@(0,1,128,255))
        return $path
    }
}
Describe 'Existing maintained CargoTools-only catalog transaction without new API' {
    BeforeEach {
        $script:SavedManifest = $env:CARGOTOOLS_MANIFEST
        $script:SavedModulePath = $env:PSModulePath
        $script:Root = Join-Path $TestDrive ([Guid]::NewGuid().ToString('N'))
        $script:Catalog = Join-Path $script:Root 'catalog'
        $script:Install = Join-Path $script:Root 'installed'
        $null = New-Item -ItemType Directory -Path (Join-Path $script:Catalog 'Modules') -Force
        $script:Cargo = New-TransactionFixtureModule -Root (Join-Path $script:Root 'source') -Name CargoTools
        $env:CARGOTOOLS_MANIFEST = Join-Path $script:Cargo 'CargoTools.psd1'
    }
    AfterEach {
        $env:CARGOTOOLS_MANIFEST = $script:SavedManifest
        $env:PSModulePath = $script:SavedModulePath
    }
    It 'performs only the CargoTools transaction and preserves unrelated installed module bytes' {
        $other = New-TransactionFixtureModule -Root $script:Install -Name 'PC-AI.Fixture' -Marker 'local installed PC fixture'
        $otherHash = (Get-FileHash -LiteralPath (Join-Path $other 'PC-AI.Fixture.psm1')).Hash
        $results = @(& $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -Mode Copy -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false)
        $results.Count | Should -Be 1
        $results[0].Name | Should -Be CargoTools
        $receipt = Get-Content -Raw -LiteralPath $results[0].Receipt | ConvertFrom-Json
        $receipt.Files.Count | Should -Be 3
        foreach ($file in $receipt.Files) {
            (Get-FileHash -LiteralPath (Join-Path $script:Cargo $file.Path)).Hash | Should -BeExactly $file.Sha256
            (Get-FileHash -LiteralPath (Join-Path $results[0].InstalledPath $file.Path)).Hash | Should -BeExactly $file.Sha256
        }
        (Get-FileHash -LiteralPath (Join-Path $other 'PC-AI.Fixture.psm1')).Hash | Should -BeExactly $otherHash
        Test-Path -LiteralPath (Join-Path $script:Install '.pcai-module-install/PC-AI.Fixture') | Should -BeFalse
        $env:PSModulePath | Should -BeExactly $script:SavedModulePath
    }
    It 'preserves installed-only synthetic private bytes in previous custody' {
        $old = New-TransactionFixtureModule -Root $script:Install -Name CargoTools -Marker 'previous synthetic installation'
        $file = Join-Path $old 'local-only-fixture.txt'
        'synthetic fixture, never real credentials' | Set-Content -LiteralPath $file
        $hash = (Get-FileHash -LiteralPath $file).Hash
        $result = & $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false
        $receipt = Get-Content -Raw -LiteralPath $result.Receipt | ConvertFrom-Json
        (Get-FileHash -LiteralPath (Join-Path $receipt.PreviousPath 'local-only-fixture.txt')).Hash | Should -BeExactly $hash
        ($receipt.PreviousFiles | Where-Object Path -eq 'local-only-fixture.txt').Sha256 | Should -BeExactly $hash
    }
    It 'retains the existing transaction idempotence and avoids a new revision' {
        $first = & $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false
        $hash = (Get-FileHash -LiteralPath $first.Receipt).Hash
        $second = & $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false
        $second.State | Should -Be Unchanged
        (Get-FileHash -LiteralPath $first.Receipt).Hash | Should -BeExactly $hash
        Test-Path -LiteralPath (Join-Path $script:Install '.pcai-module-install/CargoTools/r2') | Should -BeFalse
    }
    It 'keeps existing Cargo-only <Flag> non-mutating for files and process/user module paths' -ForEach @(
        @{Flag='DryRun'}, @{Flag='WhatIf'}
    ) {
        $userBefore = [Environment]::GetEnvironmentVariable('PSModulePath','User')
        $arguments = @{RepoRoot=$script:Catalog;InstallRoot=$script:Install;IncludeCargoTools=$true}
        $arguments[$Flag] = $true
        & $script:ExistingInstaller @arguments
        Test-Path -LiteralPath $script:Install | Should -BeFalse
        $env:PSModulePath | Should -BeExactly $script:SavedModulePath
        [Environment]::GetEnvironmentVariable('PSModulePath','User') | Should -BeExactly $userBefore
    }
    It 'demonstrates intended IncludeCargoTools behavior with an unconstrained catalog' {
        $null = New-TransactionFixtureModule -Root (Join-Path $script:Catalog 'Modules') -Name 'PC-AI.Fixture'
        $results = @(& $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false)
        $results.Count | Should -Be 2
        @($results.Name | Sort-Object) | Should -Be @('CargoTools','PC-AI.Fixture')
    }
    It 'rejects nested Git payload before publication with the unchanged safety guard' {
        $null = New-Item -ItemType Directory -Path (Join-Path $script:Cargo 'nested/.git') -Force
        { & $script:ExistingInstaller -RepoRoot $script:Catalog -InstallRoot $script:Install -IncludeCargoTools -UpdatePSModulePath:$false -Confirm:$false } | Should -Throw '*Nested Git*'
        Test-Path -LiteralPath $script:Install | Should -BeFalse
    }
}
