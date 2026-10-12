#Requires -Version 7.0

BeforeAll {
    $script:Installer=if($env:PCAI_TEST_CLOUD_INSTALLER){[IO.Path]::GetFullPath($env:PCAI_TEST_CLOUD_INSTALLER)}else{Join-Path $PSScriptRoot '../../Tools/Install-PcaiDevModules.ps1'}
    $tokens=$null;$parseErrors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($script:Installer,[ref]$tokens,[ref]$parseErrors)
    if($parseErrors.Count){throw 'Installer source parse failure'}
    foreach($name in @('Test-ContainedPath','Test-PcaiCloudReparseTag','Get-PcaiInstallReparseTag','Test-PcaiCloudReparsePath','Assert-SafePath','Get-ModuleInventory','Test-SameInventory')){
        $functionMatches=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true))
        if($functionMatches.Count -ne 1){throw "Installer function anchor is missing or ambiguous: $name"}
        . ([scriptblock]::Create($functionMatches[0].Extent.Text))
    }
    function New-CloudInstallFixture([string]$Root,[string]$Name='CloudFixture',[string]$Marker='new source'){
        $module=Join-Path $Root $Name
        $null=New-Item -ItemType Directory -Path $module -Force
        "@{RootModule='$Name.psm1';ModuleVersion='1.0.0';FunctionsToExport=@()}"|Set-Content -LiteralPath (Join-Path $module "$Name.psd1")
        "# $Marker"|Set-Content -LiteralPath (Join-Path $module "$Name.psm1")
        [IO.File]::WriteAllBytes((Join-Path $module 'fixture.bin'),[byte[]]@(0,1,128,255))
        return $module
    }
    # Positive host evidence is read-only and bounded to one explicit/default
    # directory's metadata. No real installed module payload is enumerated.
    $script:ActualCloudRoot=$null
    if($IsWindows){
        $candidate=if($env:PCAI_TEST_CLOUD_ROOT){$env:PCAI_TEST_CLOUD_ROOT}else{Join-Path $HOME 'OneDrive'}
        if(Test-Path -LiteralPath $candidate -PathType Container){
            $item=Get-Item -LiteralPath $candidate -Force
            if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -and (Test-PcaiCloudReparsePath -Path $item.FullName)){$script:ActualCloudRoot=$item.FullName}
        }
        if($env:PCAI_TEST_CLOUD_ROOT -and -not$script:ActualCloudRoot){throw 'Explicit Cloud fixture must be an existing known Cloud Files directory.'}
    }
}

Describe 'Exact documented Cloud Files family classifier' -Tag 'Unit','Portable' {
    It 'admits documented cloud variant <Variant> only' -ForEach @(0..15|ForEach-Object {@{Variant=$_;Tag=[uint32]([Convert]::ToUInt32('9000001A',16)+($_*4096))}}){
        param($Variant,$Tag)
        Test-PcaiCloudReparseTag -Tag $Tag|Should -BeTrue
    }
    It 'rejects non-cloud, link, unknown or name-surrogate tag <Hex>' -ForEach @('00000000','A0000003','A000000C','9000001C','80000021','80000015','9001001A','B000001A','1000001A','9000001B'|ForEach-Object {@{Hex=$_;Tag=[Convert]::ToUInt32($_,16)}}){
        param($Hex,$Tag)
        Test-PcaiCloudReparseTag -Tag $Tag|Should -BeFalse
    }
}

Describe 'Genuine no-follow Windows tag and existing guard behavior' -Tag 'Unit','Windows' -Skip:(-not$IsWindows) {
    BeforeEach {
        $script:Fixture=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Directory -Path $script:Fixture
    }
    It 'queries an ordinary directory as zero and admits the existing ordinary path' {
        Get-PcaiInstallReparseTag -Path $script:Fixture|Should -Be 0
        Test-PcaiCloudReparsePath -Path $script:Fixture|Should -BeFalse
        {Assert-SafePath -Path $script:Fixture}|Should -Not -Throw
    }
    It 'queries a real junction itself, not its ordinary target, and rejects it' {
        $target=Join-Path $script:Fixture 'target';$null=New-Item -ItemType Directory -Path $target
        $junction=Join-Path $script:Fixture 'junction';$null=New-Item -ItemType Junction -Path $junction -Target $target
        Get-PcaiInstallReparseTag -Path $junction|Should -Be ([Convert]::ToUInt32('A0000003',16))
        {Assert-SafePath -Path $junction}|Should -Throw '*Reparse point*'
        {Get-ModuleInventory -Root $junction}|Should -Throw '*Linked module root*'
        {Assert-SafePath -Path (Join-Path $junction 'future-child')}|Should -Throw '*Reparse point*'
    }
    It 'rejects a real nested junction payload before reading or copying its target' {
        $target=Join-Path $script:Fixture 'external';$null=New-Item -ItemType Directory -Path $target
        $source=Join-Path $script:Fixture 'source';$null=New-Item -ItemType Directory -Path $source
        $null=New-Item -ItemType Junction -Path (Join-Path $source 'nested') -Target $target
        {Get-ModuleInventory -Root $source}|Should -Throw '*Linked payload*'
    }
    It 'fails closed on a real missing-handle tag query' {
        {Get-PcaiInstallReparseTag -Path (Join-Path $script:Fixture 'absent')}|Should -Throw '*Cannot inspect Windows reparse tag*'
    }
    It 'fails closed when a real opened device handle does not support file tag metadata' {
        {Get-PcaiInstallReparseTag -Path '\\.\NUL'}|Should -Throw '*FileAttributeTagInfo metadata query failed*'
    }
    It 'retains filesystem-root and reserved-path refusal' {
        {Assert-SafePath -Path 'C:\'}|Should -Throw '*filesystem root*'
        {Assert-SafePath -Path (Join-Path $script:Fixture 'NUL')}|Should -Throw '*Unsafe Windows path*'
    }
    It 'retains nested Git-store refusal independently of cloud tags' {
        $null=New-Item -ItemType Directory -Path (Join-Path $script:Fixture 'nested/.git') -Force
        {Get-ModuleInventory -Root $script:Fixture}|Should -Throw '*Nested Git*'
    }
    It 'propagates payload hash-read failure without silently accepting omitted bytes' {
        $file=Join-Path $script:Fixture 'locked.bin';[IO.File]::WriteAllBytes($file,[byte[]]@(1,2))
        $lock=[IO.File]::Open($file,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        try {$saved=$ErrorActionPreference;$ErrorActionPreference='Stop';{Get-ModuleInventory -Root $script:Fixture}|Should -Throw}
        finally{$ErrorActionPreference=$saved;$lock.Dispose()}
    }
}

Describe 'Controlled metadata failures preserve fail-closed path admission' -Tag 'Unit','Portable' {
    BeforeEach {
        $script:ControlPath=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Directory -Path $script:ControlPath
        Mock Get-Item {[pscustomobject]@{FullName=$script:ControlPath;Attributes=([IO.FileAttributes]::Directory -bor [IO.FileAttributes]::ReparsePoint);LinkType=$null;PSIsContainer=$true}} -ParameterFilter {$LiteralPath -eq $script:ControlPath}
    }
    It 'rejects a metadata-query failure before admitting the path' {
        Mock Get-PcaiInstallReparseTag {throw 'synthetic metadata query failure'}
        {Assert-SafePath -Path $script:ControlPath}|Should -Throw '*synthetic metadata query failure*'
        Should -Invoke Get-PcaiInstallReparseTag -Times 1 -Exactly
    }
    It 'rejects unknown metadata independently of the reparse attribute' {
        Mock Get-PcaiInstallReparseTag {[Convert]::ToUInt32('9001001A',16)}
        {Assert-SafePath -Path $script:ControlPath}|Should -Throw '*Reparse point*'
    }
    It 'does not let leaf-link replacement authorization admit an unknown non-link tag' {
        Mock Get-PcaiInstallReparseTag {[Convert]::ToUInt32('9001001A',16)}
        {Assert-SafePath -Path $script:ControlPath -AllowLeafLink}|Should -Throw '*Reparse point*'
    }
}

Describe 'Actual host Cloud metadata and non-mutating preflight' -Tag 'Unit','Windows' -Skip:(-not$IsWindows) {
    It 'reads and admits one actual Cloud Files directory through a no-follow metadata handle' {
        if(-not$script:ActualCloudRoot){Set-ItResult -Skipped -Because 'No actual host Cloud Files metadata fixture is available.';return}
        $beforeTag=Get-PcaiInstallReparseTag -Path $script:ActualCloudRoot
        Test-PcaiCloudReparseTag -Tag $beforeTag|Should -BeTrue
        {Assert-SafePath -Path $script:ActualCloudRoot}|Should -Not -Throw
        Get-PcaiInstallReparseTag -Path $script:ActualCloudRoot|Should -Be $beforeTag
    }
    It 'admits an actual Cloud ancestor in <Flag> without files or environment mutation' -ForEach @(@{Flag='DryRun'},@{Flag='WhatIf'}) {
        param($Flag)
        if(-not$script:ActualCloudRoot){Set-ItResult -Skipped -Because 'No actual host Cloud Files metadata fixture is available.';return}
        $process=$env:PSModulePath;$user=[Environment]::GetEnvironmentVariable('PSModulePath','User');$machine=[Environment]::GetEnvironmentVariable('PSModulePath','Machine')
        $repo=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $name='CloudFixture'+[guid]::NewGuid().ToString('N')
        $null=New-CloudInstallFixture -Root (Join-Path $repo 'Modules') -Name $name
        $destination=Join-Path $script:ActualCloudRoot $name
        Test-Path -LiteralPath $destination|Should -BeFalse
        $beforeTag=Get-PcaiInstallReparseTag -Path $script:ActualCloudRoot
        $arguments=@{RepoRoot=$repo;InstallRoot=$script:ActualCloudRoot;Mode='Copy';UpdatePSModulePath=$false};$arguments[$Flag]=$true
        @(& $script:Installer @arguments).Count|Should -Be 0
        Test-Path -LiteralPath $destination|Should -BeFalse
        Get-PcaiInstallReparseTag -Path $script:ActualCloudRoot|Should -Be $beforeTag
        $env:PSModulePath|Should -BeExactly $process
        [Environment]::GetEnvironmentVariable('PSModulePath','User')|Should -BeExactly $user
        [Environment]::GetEnvironmentVariable('PSModulePath','Machine')|Should -BeExactly $machine
    }
}

Describe 'Private synthetic transaction failures retain original installations' -Tag 'Unit','Portable' {
    BeforeEach {
        $script:SavedManifest=$env:CARGOTOOLS_MANIFEST
        $script:Fixture=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:Catalog=Join-Path $script:Fixture 'catalog';$null=New-Item -ItemType Directory -Path (Join-Path $script:Catalog 'Modules') -Force
        $script:Source=New-CloudInstallFixture (Join-Path $script:Fixture 'source') CargoTools 'new source'
        $script:InstallRoot=Join-Path $script:Fixture 'installed'
        $script:Old=New-CloudInstallFixture $script:InstallRoot CargoTools 'old installed source'
        $script:OldInventory=@(Get-ModuleInventory -Root $script:Old)
        $env:CARGOTOOLS_MANIFEST=Join-Path $script:Source 'CargoTools.psd1'
    }
    AfterEach {$env:CARGOTOOLS_MANIFEST=$script:SavedManifest}
    It 'rejects a genuine unreadable source payload before publication, with old bytes intact' {
        $file=Join-Path $script:Source 'fixture.bin';$lock=[IO.File]::Open($file,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        try {
            {& $script:Installer -RepoRoot $script:Catalog -InstallRoot $script:InstallRoot -IncludeCargoTools -Mode Copy -UpdatePSModulePath:$false -Confirm:$false}|Should -Throw
            Test-SameInventory -Left $script:OldInventory -Right @(Get-ModuleInventory -Root $script:Old)|Should -BeTrue
            Test-Path -LiteralPath (Join-Path $script:InstallRoot '.pcai-module-install')|Should -BeFalse
        }finally{$lock.Dispose()}
    }
    It 'records an injected copy failure and rolls back without replacing the original tree' {
        Mock Copy-Item {throw 'synthetic copy failure'} -ParameterFilter {$LiteralPath -eq (Join-Path $script:Source 'fixture.bin')}
        {& $script:Installer -RepoRoot $script:Catalog -InstallRoot $script:InstallRoot -IncludeCargoTools -Mode Copy -UpdatePSModulePath:$false -Confirm:$false}|Should -Throw '*synthetic copy failure*'
        Test-SameInventory -Left $script:OldInventory -Right @(Get-ModuleInventory -Root $script:Old)|Should -BeTrue
        $receipt=Get-Content -LiteralPath (Join-Path $script:InstallRoot '.pcai-module-install/CargoTools/r1/receipt.json') -Raw|ConvertFrom-Json
        $receipt.State|Should -BeExactly 'RolledBack'
        $receipt.Error|Should -BeExactly 'synthetic copy failure'
        Should -Invoke Copy-Item -Times 1 -Exactly -ParameterFilter {$LiteralPath -eq (Join-Path $script:Source 'fixture.bin')}
    }
}
