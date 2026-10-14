#Requires -Modules Pester
BeforeAll {
    $script:Tool = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/SystemScripts/Networking/Invoke-PcaiCloudMountGate.ps1'
    . $script:Tool
    $script:Sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    function New-PcaiRunnerFixture {
        [CmdletBinding(SupportsShouldProcess)]
        param()
        $root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        if (-not $PSCmdlet.ShouldProcess($root,'Create isolated runner fixture files')) { throw 'Test fixture creation declined.' }
        $null = [IO.Directory]::CreateDirectory($root)
        $script:PcaiGateWrapper = Join-Path $root 'wrapper.ps1'
        $script:PcaiGateHelper = Join-Path $root 'helper.ps1'
        $script:Marker = Join-Path $root 'entered.json'
        $source = @'
param($TaskName,$Drive,$ProviderUnc,$CacheDirectory,$ExpectedUserSid,$ExpectedTaskXmlSha256,$TimeoutSeconds,[switch]$Apply,[switch]$ValidateOnly)
[IO.File]::WriteAllText('__MARKER__',($PSBoundParameters | ConvertTo-Json))
[pscustomobject]@{State='__STATE__';Drive=$Drive;StartRequested=$false}
'@
        [IO.File]::WriteAllText($script:PcaiGateWrapper,$source.Replace('__MARKER__',$script:Marker.Replace("'","''")).Replace('__STATE__','AlreadyRunning'))
        [IO.File]::WriteAllText($script:PcaiGateHelper,'# reviewed fixture helper')
        $script:Binding = @{Drive='U';TaskName='Rclone-Dropbox-Gmail-U';ProviderUnc='\\dropbox-cloud\gmail-u';CacheDirectory='F:\RcloneCache\gmail-u'
            ExpectedUserSid=$script:Sid;ExpectedTaskXmlSha256='A'*64;ExpectedWrapperSha256=(Get-FileHash $script:PcaiGateWrapper).Hash
            ExpectedHelperSha256=(Get-FileHash $script:PcaiGateHelper).Hash;ReceiptDirectory=$root;TimeoutSeconds=17;Apply=$true;ValidateOnly=$true }
        return $root
    }
    function Get-PcaiReceiptFixture {
        param([string]$State='AlreadyRunning')
        [pscustomobject]@{SchemaVersion=1;Owner='PcaiCloudMountGate';Drive='U';TaskName='Rclone-Dropbox-Gmail-U';UserSid=$script:Sid;State=$State;Result=$null}
    }
}
Describe 'Cloud gate runner controls' {
    BeforeEach {
        $script:Root = New-PcaiRunnerFixture
        # Only OS identity is replaced. Real runner, file hashing/sharing, script
        # invocation, validation and durable receipt rotation execute unchanged.
        Mock Get-PcaiGateContext { [pscustomobject]@{Sid=$script:Sid;SessionId=1;Elevated=$false} }
    }
    It 'plans with no identity probe, source opening, wrapper call or writes' {
        $script:Binding.Apply=$false
        $before=@(Get-ChildItem $script:Root).Count
        (Invoke-PcaiGateRunner @script:Binding).State | Should -Be 'Plan'
        Should -Invoke Get-PcaiGateContext -Times 0
        Test-Path $script:Marker | Should -BeFalse
        @(Get-ChildItem $script:Root).Count | Should -Be $before
    }
    It 'DryRun and WhatIf suppress all runtime effects even with Apply' {
        foreach ($switch in @('DryRun','WhatIf')) {
            $callArguments=$script:Binding.Clone();$callArguments[$switch]=$true
            (Invoke-PcaiGateRunner @callArguments).SideEffects | Should -BeFalse
        }
        Should -Invoke Get-PcaiGateContext -Times 0
        Test-Path $script:Marker | Should -BeFalse
        Test-Path (Join-Path $script:Root 'deployment-gate-U-latest.json') | Should -BeFalse
    }
    It 'supports explicit and long help without probes' {
        (Invoke-PcaiGateRunner -Help).State | Should -Be 'Help'
        (Invoke-PcaiGateRunner -CliArgs '--help').State | Should -Be 'Help'
        Should -Invoke Get-PcaiGateContext -Times 0
    }
    It 'rejects unknown CLI arguments' { {Invoke-PcaiGateRunner -CliArgs '--bogus'} | Should -Throw '*Unknown*' }
    It 'forwards exactly reviewed parameters to the actual fixture boundary' {
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'AlreadyRunning'
        $callArguments=[IO.File]::ReadAllText($script:Marker)|ConvertFrom-Json
        $callArguments.TaskName | Should -Be 'Rclone-Dropbox-Gmail-U'
        $callArguments.Drive | Should -Be 'U';$callArguments.ProviderUnc | Should -Be '\\dropbox-cloud\gmail-u'
        $callArguments.CacheDirectory | Should -Be 'F:\RcloneCache\gmail-u'
        $callArguments.ExpectedUserSid | Should -Be $script:Sid;$callArguments.ExpectedTaskXmlSha256 | Should -Be ('A'*64)
        $callArguments.TimeoutSeconds | Should -Be 17;$callArguments.Apply.IsPresent | Should -BeTrue;$callArguments.ValidateOnly.IsPresent | Should -BeTrue
        $saved=[IO.File]::ReadAllText((Join-Path $script:Root 'deployment-gate-U-latest.json'))|ConvertFrom-Json
        $saved.State | Should -Be 'AlreadyRunning';$saved.WrapperSha256 | Should -Be $script:Binding.ExpectedWrapperSha256
    }
    It 'rejects source hash changes before admission and records failure' {
        [IO.File]::AppendAllText($script:PcaiGateWrapper,'# changed')
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'Failed';$result.Error | Should -Match 'bytes changed'
        Test-Path $script:Marker | Should -BeFalse
    }
    It 'rejects helper hash changes before admission' {
        $script:Binding.ExpectedHelperSha256='B'*64
        (Invoke-PcaiGateRunner @script:Binding).State | Should -Be 'Failed'
        Test-Path $script:Marker | Should -BeFalse
    }
    It 'holds source bytes against writes while a reader lease is open' {
        $lease=Open-PcaiGateSource $script:PcaiGateWrapper $script:Binding.ExpectedWrapperSha256
        try { {[IO.File]::AppendAllText($script:PcaiGateWrapper,'changed')} | Should -Throw } finally {$lease.Dispose()}
        [IO.File]::AppendAllText($script:PcaiGateWrapper,'now writable')
        (Get-FileHash $script:PcaiGateWrapper).Hash | Should -Not -Be $script:Binding.ExpectedWrapperSha256
    }
    It 'refuses invalid source pins and oversized sources' {
        {Open-PcaiGateSource $script:PcaiGateWrapper 'bad'} | Should -Throw '*SHA256*'
        [IO.File]::WriteAllBytes($script:PcaiGateHelper,[byte[]]::new(2MB+1))
        {Open-PcaiGateSource $script:PcaiGateHelper ((Get-FileHash $script:PcaiGateHelper).Hash)} | Should -Throw '*limit*'
    }
    It 'refuses elevated, session-zero and foreign-owner contexts' {
        foreach ($context in @(@{Sid=$script:Sid;SessionId=1;Elevated=$true},@{Sid=$script:Sid;SessionId=0;Elevated=$false},@{Sid='S-foreign';SessionId=1;Elevated=$false})) {
            $script:Context=$context
            Mock Get-PcaiGateContext { [pscustomobject]$script:Context }
            {Invoke-PcaiGateRunner @script:Binding} | Should -Throw '*Limited*'
        }
        Test-Path $script:Marker | Should -BeFalse
    }
    It 'refuses R, crossed provider bindings, invalid pins and deadlines' {
        foreach ($change in @(@{Drive='R'},@{TaskName='Rclone-VIGIL-Team'},@{ExpectedTaskXmlSha256='bad'},@{TimeoutSeconds=0},@{TimeoutSeconds=601})) {
            $script:InvalidBinding=$script:Binding.Clone();foreach($key in $change.Keys){$script:InvalidBinding[$key]=$change[$key]}
            {Invoke-PcaiGateRunner @script:InvalidBinding} | Should -Throw '*binding*'
        }
        Should -Invoke Get-PcaiGateContext -Times 0
    }
    It 'retains Hold and uncertain scheduler outcomes without success claims' {
        $text=[IO.File]::ReadAllText($script:PcaiGateWrapper).Replace("State='AlreadyRunning'","State='Hold';StartOutcome='Unknown'")
        [IO.File]::WriteAllText($script:PcaiGateWrapper,$text)
        $script:Binding.ExpectedWrapperSha256=(Get-FileHash $script:PcaiGateWrapper).Hash
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'Hold';$result.Result.StartOutcome | Should -Be 'Unknown'
    }
    It 'continues after an actual fixture wrapper emits Hold and exits one' {
        $text=[IO.File]::ReadAllText($script:PcaiGateWrapper).Replace("State='AlreadyRunning'","State='Hold'")+"`nexit 1"
        [IO.File]::WriteAllText($script:PcaiGateWrapper,$text)
        $script:Binding.ExpectedWrapperSha256=(Get-FileHash $script:PcaiGateWrapper).Hash
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'Hold'
        ([IO.File]::ReadAllText((Join-Path $script:Root 'deployment-gate-U-latest.json'))|ConvertFrom-Json).State | Should -Be 'Hold'
    }
    It 'rejects unqualified output from the launcher' {
        [IO.File]::AppendAllText($script:PcaiGateWrapper,"`n'noise'")
        $script:Binding.ExpectedWrapperSha256=(Get-FileHash $script:PcaiGateWrapper).Hash
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'Failed';$result.Error | Should -Match 'unqualified'
    }
    It 'records actual launcher exceptions' {
        [IO.File]::WriteAllText($script:PcaiGateWrapper,"throw 'real fixture failure'")
        $script:Binding.ExpectedWrapperSha256=(Get-FileHash $script:PcaiGateWrapper).Hash
        $result=Invoke-PcaiGateRunner @script:Binding
        $result.State | Should -Be 'Failed';$result.Error | Should -Be 'real fixture failure'
    }
    It 'rotates three durable receipts into bounded latest and previous' {
        foreach($state in @('first','second','third')) { Write-PcaiGateReceipt (Get-PcaiReceiptFixture $state) $script:Root }
        ([IO.File]::ReadAllText((Join-Path $script:Root 'deployment-gate-U-latest.json'))|ConvertFrom-Json).State | Should -Be 'third'
        ([IO.File]::ReadAllText((Join-Path $script:Root 'deployment-gate-U-previous.json'))|ConvertFrom-Json).State | Should -Be 'second'
        @(Get-ChildItem $script:Root -Filter '*.pending').Count | Should -Be 0
        @(Get-ChildItem $script:Root -Filter 'deployment-gate-*').Count | Should -Be 3
    }
    It 'preserves foreign receipts and refuses to overwrite them' {
        $path=Join-Path $script:Root 'deployment-gate-U-latest.json'
        [IO.File]::WriteAllText($path,'{"SchemaVersion":1,"Owner":"foreign"}')
        $before=(Get-FileHash $path).Hash
        {Write-PcaiGateReceipt (Get-PcaiReceiptFixture) $script:Root} | Should -Throw '*Foreign*'
        (Get-FileHash $path).Hash | Should -Be $before
    }
    It 'preserves prior receipts when replacement is denied by a live reader' {
        Write-PcaiGateReceipt (Get-PcaiReceiptFixture 'old') $script:Root
        $path=Join-Path $script:Root 'deployment-gate-U-latest.json'
        $before=(Get-FileHash $path).Hash
        $held=[IO.FileStream]::new($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
        try { {Write-PcaiGateReceipt (Get-PcaiReceiptFixture 'new') $script:Root} | Should -Throw } finally {$held.Dispose()}
        (Get-FileHash $path).Hash | Should -Be $before
        @(Get-ChildItem $script:Root -Filter '*.pending').Count | Should -Be 0
    }
    It 'refuses a competing writer lock while preserving receipt bytes' {
        Write-PcaiGateReceipt (Get-PcaiReceiptFixture 'old') $script:Root
        $path=Join-Path $script:Root 'deployment-gate-U-latest.json';$before=(Get-FileHash $path).Hash
        $lock=[IO.FileStream]::new((Join-Path $script:Root 'deployment-gate-U.lock'),[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        try { {Write-PcaiGateReceipt (Get-PcaiReceiptFixture 'new') $script:Root} | Should -Throw } finally {$lock.Dispose()}
        (Get-FileHash $path).Hash | Should -Be $before
        @(Get-ChildItem $script:Root -Filter '*.pending').Count | Should -Be 0
    }
    It 'rejects actual directory-junction ancestry before source reads or receipt writes' {
        $alias=Join-Path $TestDrive ('alias-'+[guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Junction -Path $alias -Target $script:Root -ErrorAction Stop
        {Assert-PcaiGateLocalPath $alias -Directory} | Should -Throw '*Reparse*'
        {Open-PcaiGateSource (Join-Path $alias 'wrapper.ps1') $script:Binding.ExpectedWrapperSha256} | Should -Throw '*Reparse*'
        {Write-PcaiGateReceipt (Get-PcaiReceiptFixture) $alias} | Should -Throw '*Reparse*'
        Test-Path (Join-Path $script:Root 'deployment-gate-U-latest.json') | Should -BeFalse
    }
    It 'refuses oversized old and new receipts while preserving the current one' {
        Write-PcaiGateReceipt (Get-PcaiReceiptFixture) $script:Root
        $path=Join-Path $script:Root 'deployment-gate-U-latest.json';$before=(Get-FileHash $path).Hash
        $new=Get-PcaiReceiptFixture;$new.Result='x'*128KB
        {Write-PcaiGateReceipt $new $script:Root} | Should -Throw '*limit*'
        (Get-FileHash $path).Hash | Should -Be $before
        [IO.File]::WriteAllText($path,'x'*129KB)
        {Write-PcaiGateReceipt (Get-PcaiReceiptFixture) $script:Root} | Should -Throw '*limit*'
    }
    It 'rejects UNC, missing and non-directory receipt roots' {
        {Assert-PcaiGateLocalPath '\\server\share' -Directory} | Should -Throw '*local*'
        {Assert-PcaiGateLocalPath (Join-Path $script:Root 'absent') -Directory} | Should -Throw
        {Assert-PcaiGateLocalPath $script:PcaiGateHelper -Directory} | Should -Throw '*directory*'
        {Assert-PcaiGateLocalPath $script:Root} | Should -Throw '*file*'
    }
}
Describe 'Actual runner identity and script planning boundary' {
    It 'observes the native caller token and interactive session without mutation' {
        $actual=Get-PcaiGateContext
        $actual.Sid | Should -Be ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
        $actual.SessionId | Should -Be ([Diagnostics.Process]::GetCurrentProcess().SessionId)
        $actual.Elevated | Should -Be (([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))
    }
    It 'executes the actual script entrypoint with DryRun and exits zero' {
        $output=& (Join-Path $PSHOME 'pwsh.exe') -NoProfile -NonInteractive -File $script:Tool -Apply -DryRun
        $LASTEXITCODE | Should -Be 0
        ($output|ConvertFrom-Json).State | Should -Be 'Plan'
    }
}
