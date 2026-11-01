#Requires -Version 7.0
#Requires -Modules @{ ModuleName='Pester'; ModuleVersion='5.0.0' }
param([string]$LauncherPath)

BeforeAll {
    $script:SelectedLauncher = if ($LauncherPath) { $LauncherPath } else { Join-Path $PSScriptRoot '../../Tools/SystemScripts/SerenaLauncher/run-serena-codex.ps1' }
    $allowedHashes = @('BB4368B2E05DF04E834E5D24D32D34A88A1712B922C27794BF858C21BDC6A4A8',
        '0097966855B17B6403657E6A20272921A42B4EE9479C941829DBB30DB2A896D6',
        '957743868333CCE466FFBC9EDC0085965D35287FC15201C0F92840CB14245352')
    if ((Get-FileHash -LiteralPath $script:SelectedLauncher).Hash -cnotin $allowedHashes) { throw 'Selected complete launcher is not the reviewed exact baseline/candidate' }
    $tokens=$null; $errors=$null
    $script:LauncherAst=[Management.Automation.Language.Parser]::ParseFile($script:SelectedLauncher,[ref]$tokens,[ref]$errors)
    if ($errors.Count) { throw 'Complete launcher source did not parse' }
    $required=@('Get-PostgresConfig','Test-TcpPort','Ensure-ManagedPostgres')
    $functions=@($script:LauncherAst.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -in $required })
    if ($functions.Count -ne $required.Count) { throw 'Complete actual prerequisite/provider functions missing' }
    foreach ($function in $functions) { . ([scriptblock]::Create($function.Extent.Text)) }
    $dockerProvider=@($script:LauncherAst.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -eq 'Invoke-SerenaDockerCommand' })
    $script:HasDockerProvider=$dockerProvider.Count -eq 1
    if ($script:HasDockerProvider) { . ([scriptblock]::Create($dockerProvider[0].Extent.Text)) }
    $script:ManagedContainerName='owned-serena-container'
    $script:ManagedVolumeName='owned-serena-volume'
    $script:OwnedLocalUrl='postgresql://owned:owned@localhost:5544/owned'
    $script:OwnedExternalUrl='postgresql://owned:owned@owned.invalid:5544/owned'
}

Describe 'Serena launcher managed PostgreSQL policy' -Tag 'Unit','Portable','NoDocker' {
    BeforeEach {
        # All values belong to this fixture. None comes from the user environment.
        $state=[pscustomobject]@{
            Probes=0; DockerLookups=0; Ready=$false; Existing=$true
            BecomeReady=$true; DockerMissing=$false; ProviderFault=$false
            Waits=0; Calls=[Collections.Generic.List[string]]::new()
        }
        $script:Scenario=$state
        $probe={ param($Address,$Port); $state.Probes++; return $state.Ready }.GetNewClosure()
        Mock Test-TcpPort -MockWith $probe
        # Default refusal tests install an independent prohibited-provider sentinel.
        # Legacy source reaches this mocked lookup; no Docker executable is called.
        $lookup={ param($Name,$CommandType); $state.DockerLookups++; if ($state.DockerMissing) { return $null }; throw 'Owned prohibited Docker lookup was invoked' }.GetNewClosure()
        Mock Get-Command -ParameterFilter { $Name -contains 'docker.exe' } -MockWith $lookup
        $wait={ param($Seconds); $state.Waits++ }.GetNewClosure()
        Mock Start-Sleep -MockWith $wait
        if ($script:HasDockerProvider) {
            $provider={
                param($Executable,[string[]]$ArgumentList)
                $verb=$ArgumentList[0]
                $state.Calls.Add($verb)
                if ($state.ProviderFault) { throw [IO.IOException]::new('Owned Docker provider failure') }
                if ($verb -eq 'ps') { if ($state.Existing) { return 'owned-container-id' }; return '' }
                if ($verb -in @('run','start')) { if ($state.BecomeReady) { $state.Ready=$true }; return 'owned-result' }
                throw 'Unexpected Docker provider operation'
            }.GetNewClosure()
            Mock Invoke-SerenaDockerCommand -MockWith $provider
        }
    }

    It 'accepts literal --help arguments before any provider or child access' {
        Mock Test-Path { throw 'Help must not query a wrapper path' } -ParameterFilter { $LiteralPath -like '*run-serena-codex.js' }
        Mock Get-Command { throw 'Help must not discover Node' } -ParameterFilter { $Name -contains 'node.exe' }
        $originalError = [Console]::Error
        $stderr = [IO.StringWriter]::new()
        try {
            [Console]::SetError($stderr)
            $output = @(& $script:SelectedLauncher '--help')
            $output.Count | Should -Be 0
            $stderr.ToString() | Should -Match 'Serena launcher'
        } finally { [Console]::SetError($originalError); $stderr.Dispose() }
        $script:Scenario.Probes | Should -Be 0
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        Should -Invoke Test-Path -Times 0 -Exactly -ParameterFilter { $LiteralPath -like '*run-serena-codex.js' }
        Should -Invoke Get-Command -Times 0 -Exactly -ParameterFilter { $Name -contains 'node.exe' }
    }

    It 'refuses unavailable local PostgreSQL by default before any Docker discovery' {
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl } | Should -Throw '*Automatic Docker startup is disabled*'
        $script:Scenario.Probes | Should -Be 1
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        $script:Scenario.Waits | Should -Be 0
        Should -Invoke Get-Command -Times 0 -Exactly -ParameterFilter { $Name -contains 'docker.exe' }
    }

    It 'keeps an already healthy local database without Docker or sleeping' {
        $script:Scenario.Ready=$true
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl } | Should -Not -Throw
        $script:Scenario.Probes | Should -Be 1
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        $script:Scenario.Waits | Should -Be 0
    }

    It 'refuses an explicitly false startup switch before Docker discovery' {
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup:$false } | Should -Throw '*Automatic Docker startup is disabled*'
        $script:Scenario.Probes | Should -Be 1
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        $script:Scenario.Waits | Should -Be 0
    }

    It 'preserves external database delegation without local TCP or Docker bootstrap' {
        { Ensure-ManagedPostgres -Url $script:OwnedExternalUrl } | Should -Not -Throw
        $script:Scenario.Probes | Should -Be 0
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        $script:Scenario.Waits | Should -Be 0
    }

    It 'preserves no-bootstrap behavior for an absent or unrecognized wrapper URL: <Case>' -ForEach @(
        @{Case='Absent';Url=$null}, @{Case='Unrecognized';Url='owned-not-a-url'}
    ) {
        param($Case,$Url)
        { Ensure-ManagedPostgres -Url $Url } | Should -Not -Throw
        $script:Scenario.Probes | Should -Be 0
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        # This is only preserved wrapper delegation; actual backend readiness is not asserted.
    }

    It 'does not start Docker merely because opt-in was supplied to a healthy local database' {
        $script:Scenario.Ready=$true
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Not -Throw
        $script:Scenario.Probes | Should -Be 1
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
    }

    It 'restarts an existing managed container only with explicit opt-in and observes readiness' {
        $state=$script:Scenario
        $lookup={ param($Name,$CommandType); $state.DockerLookups++; return [pscustomobject]@{Source='owned-never-executed-docker.exe'} }.GetNewClosure()
        Mock Get-Command -ParameterFilter { $Name -contains 'docker.exe' } -MockWith $lookup
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Not -Throw
        ($state.Calls -join '|') | Should -BeExactly 'ps|start'
        $state.Probes | Should -Be 2
        $state.DockerLookups | Should -Be 1
        $state.Waits | Should -Be 0
    }

    It 'creates a missing managed container only with explicit opt-in and observes readiness' {
        $state=$script:Scenario; $state.Existing=$false
        $lookup={ param($Name,$CommandType); $state.DockerLookups++; return [pscustomobject]@{Source='owned-never-executed-docker.exe'} }.GetNewClosure()
        Mock Get-Command -ParameterFilter { $Name -contains 'docker.exe' } -MockWith $lookup
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Not -Throw
        ($state.Calls -join '|') | Should -BeExactly 'ps|run'
        $state.Probes | Should -Be 2
        $state.Waits | Should -Be 0
    }

    It 'fails truthfully when explicit startup is requested but Docker is missing' {
        $script:Scenario.DockerMissing=$true
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Throw '*docker.exe was not found*'
        $script:Scenario.DockerLookups | Should -Be 1
        $script:Scenario.Calls.Count | Should -Be 0
        $script:Scenario.Waits | Should -Be 0
    }

    It 'propagates a genuine mocked Docker provider failure without readiness success' {
        $state=$script:Scenario; $state.ProviderFault=$true
        $lookup={ param($Name,$CommandType); $state.DockerLookups++; return [pscustomobject]@{Source='owned-never-executed-docker.exe'} }.GetNewClosure()
        Mock Get-Command -ParameterFilter { $Name -contains 'docker.exe' } -MockWith $lookup
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Throw '*Owned Docker provider failure*'
        ($state.Calls -join '|') | Should -BeExactly 'ps'
        $state.Probes | Should -Be 1
        $state.Waits | Should -Be 0
    }

    It 'does not report success when explicitly started database never becomes ready' {
        $state=$script:Scenario; $state.BecomeReady=$false
        $lookup={ param($Name,$CommandType); $state.DockerLookups++; return [pscustomobject]@{Source='owned-never-executed-docker.exe'} }.GetNewClosure()
        Mock Get-Command -ParameterFilter { $Name -contains 'docker.exe' } -MockWith $lookup
        { Ensure-ManagedPostgres -Url $script:OwnedLocalUrl -AllowManagedPostgresStartup } | Should -Throw '*did not become reachable*'
        ($state.Calls -join '|') | Should -BeExactly 'ps|start'
        $state.Probes | Should -Be 46
        $state.Waits | Should -Be 45
    }

    It 'returns complete-launcher <Mode> before path, DB, Docker or node operations and keeps stdout empty' -ForEach @(
        @{Mode='Help';Parameter='Help'}, @{Mode='ShortHelp';Parameter='h'},
        @{Mode='LongHelp';Parameter='-help'}, @{Mode='DryRun';Parameter='DryRun'}
    ) {
        param($Mode,$Parameter)
        Mock Test-Path -ParameterFilter { $LiteralPath -like '*run-serena-codex.js' } -MockWith { throw 'Inert launcher accessed JS path' }
        Mock Get-Command -ParameterFilter { $Name -contains 'node.exe' } -MockWith { throw 'Inert launcher discovered Node' }
        $originalError=[Console]::Error; $stderr=[IO.StringWriter]::new()
        try {
            [Console]::SetError($stderr)
            $parameters=@{}; $parameters[$Parameter]=$true
            $output=@(& $script:SelectedLauncher @parameters)
            $output.Count | Should -Be 0
            $stderr.ToString() | Should -Match $(if($Parameter -eq 'DryRun'){'DRY_RUN_NO_IO'}else{'Serena launcher'})
        } finally { [Console]::SetError($originalError); $stderr.Dispose() }
        $script:Scenario.Probes | Should -Be 0
        $script:Scenario.DockerLookups | Should -Be 0
        $script:Scenario.Calls.Count | Should -Be 0
        Should -Invoke Test-Path -Times 0 -Exactly -ParameterFilter { $LiteralPath -like '*run-serena-codex.js' }
        Should -Invoke Get-Command -Times 0 -Exactly -ParameterFilter { $Name -contains 'node.exe' }
    }
}
