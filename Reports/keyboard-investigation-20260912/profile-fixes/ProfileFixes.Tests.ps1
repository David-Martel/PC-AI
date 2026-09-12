param([string]$ProfilePath = 'C:/Users/david/.config/powershell/Microsoft.PowerShell_profile.ps1')

BeforeAll {
    $script:ExitFixtureRunner = Join-Path $PSScriptRoot 'ExitContext.Smoke.ps1'
    $script:ProfileText = [System.IO.File]::ReadAllText($ProfilePath)
    $tokens = $null; $parseErrors = $null
    $script:ProfileAst = [System.Management.Automation.Language.Parser]::ParseInput($script:ProfileText, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw 'Profile must parse before blocks can be tested.' }
    function Get-TestProfileNode {
        param([scriptblock]$Predicate)
        @($script:ProfileAst.FindAll($Predicate, $true))[0]
    }
    function Test-AcceleratorAvailable { $true }
    function Test-FileFast { param($Path) [System.IO.File]::Exists($Path) }
    function Invoke-ProfileTestBoundary { }
    function Close-JsonlAccelerator { }
    function Get-ProfileGuardHarness {
        param([switch]$Minimal)
        $guard = Get-TestProfileNode { param($node) $node -is [System.Management.Automation.Language.IfStatementAst] -and $node.Clauses[0].Item1.Extent.Text -match '^\$global:__MicrosoftPowerShellProfileLoaded' }
        $outerTry = @($script:ProfileAst.EndBlock.Statements | Where-Object { $_ -is [System.Management.Automation.Language.TryStatementAst] -and $_.Finally -and $_.Finally.Extent.Text -match '__MicrosoftPowerShellProfileLoading' }) | Select-Object -First 1
        if ($outerTry) {
            $prefix = $script:ProfileText.Substring($guard.Extent.StartOffset, $outerTry.Extent.StartOffset - $guard.Extent.StartOffset)
            $success = $outerTry.Body.Statements | Where-Object { $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and $_.Left.Extent.Text -eq '$script:ProfileInitializationSucceeded' } | Select-Object -Last 1
            if (-not $success) { throw 'Missing real completion assignment.' }
            $body = 'Invoke-ProfileTestBoundary; ' + $success.Extent.Text
            if ($Minimal) {
                $minimalNode = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text -match 'InitializedComponents.MinimalProfile' }
                $body = $minimalNode.Extent.Text
            }
            return [scriptblock]::Create($prefix + 'try { ' + $body + ' } finally ' + $outerTry.Finally.Extent.Text)
        }
        $end = $script:ProfileText.IndexOf('$script:ProfileStartTime =')
        $body = 'Invoke-ProfileTestBoundary'
        if ($Minimal) {
            $minimalNode = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text -match 'InitializedComponents.MinimalProfile' }
            $body = $minimalNode.Extent.Text
        }
        [scriptblock]::Create($script:ProfileText.Substring($guard.Extent.StartOffset, $end - $guard.Extent.StartOffset) + $body)
    }
}

Describe 'Canonical profile isolated regression blocks' {
    BeforeEach {
        $global:__MicrosoftPowerShellProfileLoaded = $false
        $global:__MicrosoftPowerShellProfileLoading = $false
        $global:__PrimaryProfilePipelineLoaded = $false
        $global:__ProfileSystemBootstrapLoaded = $false
        $global:__ProfileSystemBootstrapContext = 'prior-context'
        $global:ProfileFixTestReloadCalls = 0
        $script:SavedSkip = $env:PS_SKIP_PROFILE_ACCELERATOR
        $script:SavedBootstrap = $env:PS_ENABLE_PROFILESYSTEM
        $script:UseProfileAccelerator = $false
        $script:ProfileHistoryWriterOwned = $false
        $script:ProfileExitSubscriptionId = $null
        $script:ProfileExitJob = $null
        $script:ProfileInitializationSucceeded = $false
        Mock Write-Host { }
    }
    AfterEach {
        $env:PS_SKIP_PROFILE_ACCELERATOR = $script:SavedSkip
        $env:PS_ENABLE_PROFILESYSTEM = $script:SavedBootstrap
        Remove-Variable -Scope Global -Name __MicrosoftPowerShellProfileLoaded,__MicrosoftPowerShellProfileLoading,__PrimaryProfilePipelineLoaded,__ProfileSystemBootstrapLoaded,__ProfileSystemBootstrapContext,ProfileFixTestReloadCalls -ErrorAction SilentlyContinue
    }

    It 'honors the accelerator opt-out before module import' {
        $script:AcceleratorPsd1 = 'fixture.psd1'
        $env:PS_SKIP_PROFILE_ACCELERATOR = '1'
        Mock Import-Module { }
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match '^\$script:AcceleratorPsd1' }
        . ([scriptblock]::Create($node.Extent.Text))
        Should -Invoke Import-Module -Times 0
        $script:UseProfileAccelerator | Should -BeFalse
    }
    It 'still imports the available accelerator when opted in' {
        $script:AcceleratorPsd1 = 'fixture.psd1'
        $env:PS_SKIP_PROFILE_ACCELERATOR = '0'
        Mock Import-Module { }
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match '^\$script:AcceleratorPsd1' }
        . ([scriptblock]::Create($node.Extent.Text))
        Should -Invoke Import-Module -Times 1 -ParameterFilter { $Name -eq 'fixture.psd1' }
        $script:UseProfileAccelerator | Should -BeTrue
    }
    It 'classifies host arguments without running the payload: <Case>' -ForEach @(
        @{ Case='implicit file'; HostArgs=@('pwsh','fixture.ps1'); Minimal=$true }
        @{ Case='standard interactive flags'; HostArgs=@('pwsh','-NoLogo','-NoProfile'); Minimal=$false }
        @{ Case='unknown switch stays minimal'; HostArgs=@('pwsh','-unknown'); Minimal=$true }
        @{ Case='interactive'; HostArgs=@('pwsh'); Minimal=$false }
        @{ Case='NoExit Command'; HostArgs=@('pwsh','-NoExit','-Command','Initialize-Test'); Minimal=$false }
        @{ Case='NoExit short command'; HostArgs=@('pwsh','-noexit','-c','Initialize-Test'); Minimal=$false }
        @{ Case='literal standalone noninteractive payload'; HostArgs=@('pwsh','-NoExit','-Command','Write-Output','-NonInteractive'); Minimal=$false }
        @{ Case='payload NoExit is not host NoExit'; HostArgs=@('pwsh','-Command','Write-Output','-NoExit'); Minimal=$true }
        @{ Case='option value is not host command'; HostArgs=@('pwsh','-WorkingDirectory','-Command'); Minimal=$false }
        @{ Case='explicit short noninteractive'; HostArgs=@('pwsh','-NoExit','-NonI'); Minimal=$true }
        @{ Case='automated command'; HostArgs=@('pwsh','-Command','Initialize-Test'); Minimal=$true }
        @{ Case='NoExit file stays minimal'; HostArgs=@('pwsh','-NoExit','-File','fixture.ps1'); Minimal=$true }
        @{ Case='NoExit encoded command stays minimal'; HostArgs=@('pwsh','-NoExit','-EncodedCommand','fixture'); Minimal=$true }
        @{ Case='explicit noninteractive'; HostArgs=@('pwsh','-NoExit','-NonInteractive','-Command','fixture'); Minimal=$true }
        @{ Case='payload does not become a host switch'; HostArgs=@('pwsh','-NoExit','-Command','Write-Output -NonInteractive'); Minimal=$false }
    ) {
        $helper = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'script:Test-ProfileNonInteractiveInvocation' }
        if ($helper) { . ([scriptblock]::Create($helper.Extent.Text)) }
        $script:CommandLine = $HostArgs -join ' '
        $script:CommandLineArgs = $HostArgs
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$script:HasNonInteractiveArgs' }
        . ([scriptblock]::Create($node.Extent.Text))
        $script:HasNonInteractiveArgs | Should -Be $Minimal
    }
    It 'clears partial initialization guards after a terminating failure and permits retry' {
        $guard = Get-ProfileGuardHarness
        Mock Invoke-ProfileTestBoundary { throw 'fixture initialization failure' }
        { & $guard } | Should -Throw '*fixture initialization failure*'
        $global:__MicrosoftPowerShellProfileLoaded | Should -BeFalse
        $global:__MicrosoftPowerShellProfileLoading | Should -BeFalse
        $global:__PrimaryProfilePipelineLoaded | Should -BeFalse
        Mock Invoke-ProfileTestBoundary { }
        & $guard
        $global:__MicrosoftPowerShellProfileLoaded | Should -BeTrue
        Should -Invoke Invoke-ProfileTestBoundary -Times 2
    }
    It 'does not initialize twice after successful completion' {
        Mock Invoke-ProfileTestBoundary { }
        $guard = Get-ProfileGuardHarness
        & $guard
        & $guard
        Should -Invoke Invoke-ProfileTestBoundary -Times 1
    }
    It 'does not reenter while initialization is in progress' {
        $global:__MicrosoftPowerShellProfileLoading = $true
        Mock Invoke-ProfileTestBoundary { }
        & (Get-ProfileGuardHarness)
        Should -Invoke Invoke-ProfileTestBoundary -Times 0
    }
    It 'reloads through the real reload function after an earlier successful load' {
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Import-Profile' }
        . ([scriptblock]::Create($node.Extent.Text))
        $PROFILE = Join-Path $TestDrive 'reload.ps1'
        Set-Content -LiteralPath $PROFILE -Value 'if ($global:__MicrosoftPowerShellProfileLoaded) { return }; $global:ProfileFixTestReloadCalls++; function Test-ReloadedProfileDefinition { 42 }; $script:ReloadedProfileState = 23; $ReloadedOrdinaryValue = 17; $global:__MicrosoftPowerShellProfileLoaded = $true'
        $global:__MicrosoftPowerShellProfileLoaded = $true
        . Import-Profile
        $global:ProfileFixTestReloadCalls | Should -Be 1
        Test-ReloadedProfileDefinition | Should -Be 42
        $script:ReloadedProfileState | Should -Be 23
        $ReloadedOrdinaryValue | Should -Be 17
    }
    It 'rejects ordinary invocation before reload changes scope or guards' {
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Import-Profile' }
        . ([scriptblock]::Create($node.Extent.Text))
        $global:__MicrosoftPowerShellProfileLoaded = $true
        $PROFILE = Join-Path $TestDrive 'must-not-execute.ps1'
        Set-Content -LiteralPath $PROFILE -Value '$global:ProfileFixTestReloadCalls++'
        { Import-Profile } | Should -Throw '*caller scope*'
        $global:ProfileFixTestReloadCalls | Should -Be 0
        $global:__MicrosoftPowerShellProfileLoaded | Should -BeTrue
    }
    It 'closes owned writer and removes only owned exit resources before reload even when accelerator skipped' {
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Import-Profile' }
        . ([scriptblock]::Create($node.Extent.Text))
        $PROFILE = Join-Path $TestDrive 'reload-cleanup.ps1'
        Set-Content -LiteralPath $PROFILE -Value '$global:ProfileFixTestReloadCalls++'
        $script:ProfileHistoryWriterOwned = $true
        $script:UseProfileAccelerator = $false
        $script:ProfileExitSubscriptionId = 41
        $script:ProfileExitJob = [pscustomobject]@{ Id = 73 }
        Mock Close-JsonlAccelerator { }
        Mock Unregister-Event { }
        Mock Remove-Job { } -RemoveParameterType Job
        . Import-Profile
        Should -Invoke Close-JsonlAccelerator -Times 1
        Should -Invoke Unregister-Event -Times 1 -ParameterFilter { $SubscriptionId -eq 41 }
        Should -Invoke Remove-Job -Times 1 -ParameterFilter { $Job.Id -eq 73 }
        $script:ProfileHistoryWriterOwned | Should -BeFalse
        $global:ProfileFixTestReloadCalls | Should -Be 1
    }
    It 'marks the actual minimal early return complete and clears loading' {
        $script:ProfileLevel = 'minimal'
        $script:ProfileStartTime = Get-Date
        $script:InitializedComponents = @{}
        function Trace-Section { }
        & (Get-ProfileGuardHarness -Minimal)
        $global:__MicrosoftPowerShellProfileLoaded | Should -BeTrue
        $global:__MicrosoftPowerShellProfileLoading | Should -BeFalse
        $script:InitializedComponents.MinimalProfile | Should -BeTrue
    }
    It 'does not mark a minimal trace failure as completed' {
        $script:ProfileLevel = 'minimal'
        $script:ProfileStartTime = Get-Date
        $script:InitializedComponents = @{}
        function Trace-Section { throw 'fixture trace failure' }
        { & (Get-ProfileGuardHarness -Minimal) } | Should -Throw '*fixture trace failure*'
        $global:__MicrosoftPowerShellProfileLoaded | Should -BeFalse
        $global:__MicrosoftPowerShellProfileLoading | Should -BeFalse
    }
    It 'replaces real owned event registration while preserving an unrelated subscriber' {
        $registration = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$script:ProfileExitJob' -and $n.Right.Extent.Text -match 'Register-EngineEvent' }
        if (-not $registration) {
            $registration = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Register-EngineEvent' }
        }
        $subscription = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$script:ProfileExitSubscriptionId' -and $n.Right.Extent.Text -match 'Get-EventSubscriber' }
        $cleanup = @($script:ProfileAst.FindAll({ param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match '^\$script:ProfileExit(SubscriptionId|Job)$' }, $true) | Where-Object { $_.Extent.StartOffset -lt $registration.Extent.StartOffset } | Select-Object -Last 2)
        $eventSource = 'ProfileFixes.Registration.' + [guid]::NewGuid().ToString('N')
        $unrelatedJob = Register-EngineEvent -SourceIdentifier $eventSource -Action { }
        $unrelated = Get-EventSubscriber -SourceIdentifier $eventSource
        # Only the event name is substituted; the real registration and cleanup are exercised; the exit action is not invoked.
        $body = (($cleanup | ForEach-Object { $_.Extent.Text }) -join "`n") + "`n" + $registration.Extent.Text + "`n" + $subscription.Extent.Text
        $body = $body.Replace('PowerShell.Exiting', $eventSource)
        try {
            . ([scriptblock]::Create($body))
            $script:ProfileExitJob | Should -Not -BeNullOrEmpty
            $firstId = $script:ProfileExitSubscriptionId
            $firstId | Should -Not -BeNullOrEmpty
            . ([scriptblock]::Create($body))
            $script:ProfileExitSubscriptionId | Should -Not -Be $firstId
            @(Get-EventSubscriber -SourceIdentifier $eventSource).Count | Should -Be 2
            (Get-EventSubscriber -SubscriptionId $unrelated.SubscriptionId).SubscriptionId | Should -Be $unrelated.SubscriptionId
        } finally {
            Get-EventSubscriber -SourceIdentifier $eventSource -Force -ErrorAction SilentlyContinue | Unregister-Event -Force
            if ($script:ProfileExitJob) { Remove-Job -Job $script:ProfileExitJob -Force -ErrorAction SilentlyContinue }
            Remove-Job -Job $unrelatedJob -Force -ErrorAction SilentlyContinue
        }
    }
    It 'fires actual child-process exit with captured history state: writer owned <WriterOwned>' -ForEach @(@{WriterOwned=$true}, @{WriterOwned=$false}) {
        $fixtureDirectory = Join-Path $TestDrive ('exit-' + $WriterOwned)
        $null = New-Item -ItemType Directory -Path $fixtureDirectory -Force
        $startInfo = [System.Diagnostics.ProcessStartInfo]::new((Get-Command pwsh).Source)
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $mode = if ($WriterOwned) { 'owned' } else { 'unowned' }
        foreach ($argument in @('-NoLogo','-NoProfile','-File',$script:ExitFixtureRunner,'-ProfilePath',$ProfilePath,'-OutputDirectory',$fixtureDirectory,'-WriterMode',$mode)) {
            $startInfo.ArgumentList.Add($argument)
        }
        $process = [System.Diagnostics.Process]::Start($startInfo)
        try {
            if (-not $process.WaitForExit(10000)) { $process.Kill($true); throw 'Exit fixture exceeded ten seconds.' }
            $process.ExitCode | Should -Be 0
            [System.IO.File]::Exists((Join-Path $fixtureDirectory 'exit-smoke.closed')) | Should -Be $WriterOwned
            [System.IO.File]::Exists((Join-Path $fixtureDirectory 'exit-smoke.merged')) | Should -BeTrue
        } finally { $process.Dispose() }
    }
    It 'does not mark a missing optional bootstrap loaded' {
        $env:PS_ENABLE_PROFILESYSTEM = '1'
        Mock Test-FileFast { $false }
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match 'PS_ENABLE_PROFILESYSTEM' }
        . ([scriptblock]::Create("`$PSScriptRoot = 'C:/fixture-profile-root'; " + $node.Extent.Text))
        $global:__ProfileSystemBootstrapLoaded | Should -BeFalse
    }
    It 'marks a successful bootstrap loaded and restores its context' {
        $env:PS_ENABLE_PROFILESYSTEM = '1'
        $path = Join-Path $TestDrive 'bootstrap.ps1'
        Set-Content -LiteralPath $path -Value '$global:ProfileFixTestReloadCalls++'
        $script:BootstrapFixturePath = $path
        Mock Join-Path { $script:BootstrapFixturePath }
        Mock Test-FileFast { $true }
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match 'PS_ENABLE_PROFILESYSTEM' }
        . ([scriptblock]::Create("`$PSScriptRoot = 'C:/fixture-profile-root'; " + $node.Extent.Text))
        $global:__ProfileSystemBootstrapLoaded | Should -BeTrue
        $global:__ProfileSystemBootstrapContext | Should -Be 'prior-context'
        $global:ProfileFixTestReloadCalls | Should -Be 1
    }
    It 'leaves failed bootstrap retryable and restores context: <Failure>' -ForEach @(
        @{ Failure='terminating'; Body="throw 'fixture bootstrap failure'" }
        @{ Failure='nonterminating'; Body="Write-Error 'fixture bootstrap failure'" }
    ) {
        $env:PS_ENABLE_PROFILESYSTEM = '1'
        $path = Join-Path $TestDrive 'bootstrap.ps1'
        Set-Content -LiteralPath $path -Value $Body
        $script:BootstrapFixturePath = $path
        Mock Join-Path { $script:BootstrapFixturePath }
        Mock Test-FileFast { $true }
        $node = Get-TestProfileNode { param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -match 'PS_ENABLE_PROFILESYSTEM' }
        { . ([scriptblock]::Create("`$PSScriptRoot = 'C:/fixture-profile-root'; " + $node.Extent.Text)) } | Should -Throw '*fixture bootstrap failure*'
        $global:__ProfileSystemBootstrapLoaded | Should -BeFalse
        $global:__ProfileSystemBootstrapContext | Should -Be 'prior-context'
    }
}
