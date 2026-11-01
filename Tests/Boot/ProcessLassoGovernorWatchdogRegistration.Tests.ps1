#Requires -Version 7.0
param([string]$SourcePath = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools/Register-ProcessLassoGovernorWatchdog.ps1'))

BeforeAll {
    $script:PreviousWatchdogFixtureState = Get-Variable -Name PcaiWatchdogRegistrationFixtureState -Scope Global -ErrorAction SilentlyContinue
    $script:PreviousWatchdogFixtureStatePresent = $null -ne $script:PreviousWatchdogFixtureState
    $script:PreviousWatchdogFixtureStateValue = if ($script:PreviousWatchdogFixtureStatePresent) { $script:PreviousWatchdogFixtureState.Value } else { $null }
    if (-not $IsWindows) {
        throw 'The watchdog registration suite requires Windows ScheduledTasks; select portable suites on other platforms.'
    }
    Import-Module ScheduledTasks -ErrorAction Stop
    foreach ($requiredCommand in @('New-ScheduledTaskTrigger', 'New-ScheduledTaskAction', 'New-ScheduledTaskPrincipal', 'New-ScheduledTaskSettingsSet', 'Register-ScheduledTask', 'Unregister-ScheduledTask', 'Start-ScheduledTask', 'Disable-ScheduledTask')) {
        [void](Get-Command -Name $requiredCommand -Module ScheduledTasks -ErrorAction Stop)
    }
    $script:QualifiedSource = (Resolve-Path -LiteralPath $SourcePath).Path
    $script:FixtureRepo = Join-Path $TestDrive 'watchdog-repo'
    $script:FixtureTools = Join-Path $script:FixtureRepo 'Tools'
    [void][IO.Directory]::CreateDirectory($script:FixtureTools)
    $script:RegisterScript = Join-Path $script:FixtureTools 'Register-ProcessLassoGovernorWatchdog.ps1'
    [IO.File]::Copy($script:QualifiedSource, $script:RegisterScript, $false)
    $script:WorkerScript = Join-Path $script:FixtureTools 'Ensure-ProcessLassoGovernor.ps1'
    [IO.File]::WriteAllText($script:WorkerScript, '# Synthetic worker; never executed by these registration fixtures.')
    $script:FrozenNow = [datetime]'2026-10-09T12:00:00'
    function New-FixtureSchedulerCim {
        param([string]$ClassName, [hashtable]$Properties = @{})
        # Offline metadata only: no CimSession, provider, server or task registration.
        $instance = [Microsoft.Management.Infrastructure.CimInstance]::new($ClassName, 'root/Microsoft/Windows/TaskScheduler')
        foreach ($entry in $Properties.GetEnumerator()) {
            $property = [Microsoft.Management.Infrastructure.CimProperty]::Create(
                $entry.Key, [string]$entry.Value,
                [Microsoft.Management.Infrastructure.CimType]::String,
                [Microsoft.Management.Infrastructure.CimFlags]::None)
            [void]$instance.CimInstanceProperties.Add($property)
        }
        # Cmdlet PSTypeName constraints include provider inheritance, which an
        # offline class-name-only constructor cannot obtain from a server.
        $baseClass = switch ($ClassName) {
            'MSFT_TaskLogonTrigger' { 'MSFT_TaskTrigger' }
            'MSFT_TaskTimeTrigger' { 'MSFT_TaskTrigger' }
            'MSFT_TaskExecAction' { 'MSFT_TaskAction' }
            'MSFT_TaskPrincipal2' { 'MSFT_TaskPrincipal' }
            'MSFT_TaskSettings3' { 'MSFT_TaskSettings' }
            default { throw "Unexpected offline scheduler fixture class: $ClassName" }
        }
        $requiredIdentity = 'Microsoft.Management.Infrastructure.CimInstance#' + $baseClass
        if ($requiredIdentity -notin $instance.PSObject.TypeNames) {
            $instance.PSObject.TypeNames.Insert(0, $requiredIdentity)
        }
        return $instance
    }
    function Get-FixtureFiles {
        @(Get-ChildItem -LiteralPath $script:FixtureRepo -Recurse -File |
            ForEach-Object { '{0}|{1}' -f $_.FullName, (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } |
            Sort-Object)
    }
}

AfterAll {
    if ($script:PreviousWatchdogFixtureStatePresent) {
        Set-Variable -Name PcaiWatchdogRegistrationFixtureState -Scope Global -Value $script:PreviousWatchdogFixtureStateValue
    }
    else {
        Remove-Variable -Name PcaiWatchdogRegistrationFixtureState -Scope Global -ErrorAction SilentlyContinue
    }
}
Describe 'Process Lasso watchdog registration preserves logon and recurring contracts' -Tag 'Unit', 'Boot', 'Windows' {
    BeforeEach {
        $global:PcaiWatchdogRegistrationFixtureState = [pscustomobject]@{
            TriggerCalls = [Collections.Generic.List[object]]::new()
            Registered = $null
            ActionParameters = $null
            PrincipalParameters = $null
            SettingsParameters = $null
            FrozenNow = $script:FrozenNow
        }
        Mock Get-Date { $global:PcaiWatchdogRegistrationFixtureState.FrozenNow }
        Mock New-ScheduledTaskTrigger {
            [CmdletBinding()]
            param([switch]$AtLogOn, [switch]$Once, [datetime]$At, [timespan]$RepetitionInterval, [timespan]$RepetitionDuration)
            $captured = @{} + $PSBoundParameters
            $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls.Add($captured)
            if ($AtLogOn) { return New-FixtureSchedulerCim -ClassName 'MSFT_TaskLogonTrigger' -Properties @{ Delay = '' } }
            if ($Once) { return New-FixtureSchedulerCim -ClassName 'MSFT_TaskTimeTrigger' -Properties @{ StartBoundary = $At.ToString('o') } }
            throw 'Unexpected trigger contract.'
        }
        Mock New-ScheduledTaskAction {
            [CmdletBinding()]
            param([string]$Execute, [string]$Argument, [string]$WorkingDirectory)
            $global:PcaiWatchdogRegistrationFixtureState.ActionParameters = @{} + $PSBoundParameters
            New-FixtureSchedulerCim -ClassName 'MSFT_TaskExecAction' -Properties @{ Execute = $Execute; Arguments = $Argument; WorkingDirectory = $WorkingDirectory }
        }
        Mock New-ScheduledTaskPrincipal {
            [CmdletBinding()]
            param([string]$UserId, $LogonType, $RunLevel)
            $global:PcaiWatchdogRegistrationFixtureState.PrincipalParameters = @{} + $PSBoundParameters
            New-FixtureSchedulerCim -ClassName 'MSFT_TaskPrincipal2' -Properties @{ UserId = $UserId; LogonType = $LogonType; RunLevel = $RunLevel }
        }
        Mock New-ScheduledTaskSettingsSet {
            [CmdletBinding()]
            param([switch]$StartWhenAvailable, [switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries, $MultipleInstances, [int]$RestartCount, [timespan]$RestartInterval, [timespan]$ExecutionTimeLimit)
            $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters = @{} + $PSBoundParameters
            New-FixtureSchedulerCim -ClassName 'MSFT_TaskSettings3'
        }
        Mock Register-ScheduledTask {
            [CmdletBinding()]
            param(
                [string]$TaskName,
                [PSTypeName('Microsoft.Management.Infrastructure.CimInstance#MSFT_TaskAction')][Microsoft.Management.Infrastructure.CimInstance[]]$Action,
                [PSTypeName('Microsoft.Management.Infrastructure.CimInstance#MSFT_TaskTrigger')][Microsoft.Management.Infrastructure.CimInstance[]]$Trigger,
                [PSTypeName('Microsoft.Management.Infrastructure.CimInstance#MSFT_TaskPrincipal')][Microsoft.Management.Infrastructure.CimInstance]$Principal,
                [PSTypeName('Microsoft.Management.Infrastructure.CimInstance#MSFT_TaskSettings')][Microsoft.Management.Infrastructure.CimInstance]$Settings,
                [string]$Description,
                [switch]$Force
            )
            $global:PcaiWatchdogRegistrationFixtureState.Registered = @{} + $PSBoundParameters
        }
        Mock Unregister-ScheduledTask {}
        Mock Start-ScheduledTask {}
        Mock Disable-ScheduledTask {}
        Mock New-Item { throw 'Registration must not create files or directories.' }
        Mock Set-Content { throw 'Registration must not write reports.' }
        Mock Out-File { throw 'Registration must not write files.' }
        Mock Add-Content { throw 'Registration must not append files.' }
        Mock New-EventLog { throw 'Registration must not create event sources.' }
        Mock Write-EventLog { throw 'Registration must not emit event records.' }
        Mock Start-Process { throw 'Registration must not start native processes.' }
    }

    It 'registers the default delayed logon plus indefinite five-minute repetition with the existing principal and settings' {
        & $script:RegisterScript -ScriptPath $script:WorkerScript
        Should -Invoke Register-ScheduledTask -Times 1 -Exactly
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls.Count | Should -Be 2
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[0].AtLogOn | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].Once | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].At | Should -Be ($script:FrozenNow.AddSeconds(180))
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].RepetitionInterval | Should -Be ([timespan]::FromMinutes(5))
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].ContainsKey('RepetitionDuration') | Should -BeFalse
        @($global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger).Count | Should -Be 2
        $global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger[0].CimSystemProperties.ClassName | Should -Be 'MSFT_TaskLogonTrigger'
        $global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger[0].Delay | Should -Be 'PT3M'
        $global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger[1].CimSystemProperties.ClassName | Should -Be 'MSFT_TaskTimeTrigger'
        $global:PcaiWatchdogRegistrationFixtureState.Registered.Force | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.PrincipalParameters.UserId | Should -Be "$env:USERDOMAIN\$env:USERNAME"
        $global:PcaiWatchdogRegistrationFixtureState.PrincipalParameters.LogonType | Should -Be 'Interactive'
        $global:PcaiWatchdogRegistrationFixtureState.PrincipalParameters.RunLevel | Should -Be 'Highest'
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.StartWhenAvailable | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.AllowStartIfOnBatteries | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.DontStopIfGoingOnBatteries | Should -BeTrue
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.MultipleInstances | Should -Be 'IgnoreNew'
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.RestartCount | Should -Be 3
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.RestartInterval | Should -Be ([timespan]::FromMinutes(1))
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.ExecutionTimeLimit | Should -Be ([timespan]::FromMinutes(5))
        $global:PcaiWatchdogRegistrationFixtureState.ActionParameters.WorkingDirectory | Should -Be $script:FixtureTools
        $defaultReport = Join-Path $script:FixtureRepo 'Reports/processlasso-governor-watchdog.json'
        $global:PcaiWatchdogRegistrationFixtureState.ActionParameters.Argument | Should -Match ([regex]::Escape('-ReportPath "' + $defaultReport + '"'))
        $global:PcaiWatchdogRegistrationFixtureState.ActionParameters.Argument | Should -Match '^-NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File '
        ([regex]::Matches($global:PcaiWatchdogRegistrationFixtureState.ActionParameters.Argument, '-ExecutionPolicy')).Count | Should -Be 1
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
        Should -Invoke Disable-ScheduledTask -Times 0 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }

    It 'passes an explicit D report path and custom delays, interval and settings without replacing the logon trigger' {
        $report = 'D:\pcai-relocation\responsiveness\processlasso-governor-watchdog.json'
        & $script:RegisterScript -ScriptPath $script:WorkerScript -StartupDelaySeconds 45 -RepetitionIntervalMinutes 7 -ReportPath $report -ExecutionTimeLimitMinutes 4 -RestartCount 2 -RestartIntervalMinutes 3
        $global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger[0].Delay | Should -Be 'PT45S'
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].At | Should -Be ($script:FrozenNow.AddSeconds(45))
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].RepetitionInterval | Should -Be ([timespan]::FromMinutes(7))
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].ContainsKey('RepetitionDuration') | Should -BeFalse
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.ExecutionTimeLimit | Should -Be ([timespan]::FromMinutes(4))
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.RestartCount | Should -Be 2
        $global:PcaiWatchdogRegistrationFixtureState.SettingsParameters.RestartInterval | Should -Be ([timespan]::FromMinutes(3))
        $global:PcaiWatchdogRegistrationFixtureState.ActionParameters.Argument | Should -Match ([regex]::Escape('-ReportPath "' + $report + '"'))
        $global:PcaiWatchdogRegistrationFixtureState.ActionParameters.Argument | Should -Not -Match ([regex]::Escape($script:FixtureRepo + '\Reports'))
    }

    It 'accepts the interval boundary <Minutes> and retains infinite repetition' -TestCases @(@{ Minutes = 1 }, @{ Minutes = 1440 }) {
        param($Minutes)
        & $script:RegisterScript -ScriptPath $script:WorkerScript -RepetitionIntervalMinutes $Minutes
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].RepetitionInterval | Should -Be ([timespan]::FromMinutes($Minutes))
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls[1].ContainsKey('RepetitionDuration') | Should -BeFalse
        @($global:PcaiWatchdogRegistrationFixtureState.Registered.Trigger).Count | Should -Be 2
    }

    It 'rejects interval <Minutes> before calling any scheduler mutation' -TestCases @(@{ Minutes = 0 }, @{ Minutes = -1 }, @{ Minutes = 1441 }) {
        param($Minutes)
        { & $script:RegisterScript -ScriptPath $script:WorkerScript -RepetitionIntervalMinutes $Minutes } | Should -Throw
        Should -Invoke New-ScheduledTaskTrigger -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }

    It 'keeps DryRun <Case> free of task, process, report and event mutations' -TestCases @(
        @{ Case = 'plain'; Options = @{} },
        @{ Case = 'run-now'; Options = @{ RunNow = $true } },
        @{ Case = 'disabled'; Options = @{ Disable = $true } },
        @{ Case = 'run-now-and-disabled'; Options = @{ RunNow = $true; Disable = $true } }
    ) {
        param($Case, $Options)
        $beforeFiles = Get-FixtureFiles
        $output = & $script:RegisterScript -ScriptPath $script:WorkerScript -DryRun @Options 6>&1 | Out-String
        $output | Should -Match 'trigger: at logon, delayed 180 seconds'
        $output | Should -Match 'trigger: every 5 minutes starting .*, indefinitely'
        $global:PcaiWatchdogRegistrationFixtureState.TriggerCalls.Count | Should -Be 2
        (Get-FixtureFiles) -join "`n" | Should -Be ($beforeFiles -join "`n")
        foreach ($command in @('Register-ScheduledTask', 'Unregister-ScheduledTask', 'Start-ScheduledTask', 'Disable-ScheduledTask', 'New-Item', 'Set-Content', 'Out-File', 'Add-Content', 'New-EventLog', 'Write-EventLog', 'Start-Process')) {
            Should -Invoke $command -Times 0 -Exactly
        }
    }

    It 'keeps dry-run unregister free of task and file mutations' {
        & $script:RegisterScript -ScriptPath $script:WorkerScript -Unregister -DryRun
        Should -Invoke New-ScheduledTaskTrigger -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
        Should -Invoke Unregister-ScheduledTask -Times 0 -Exactly
        Should -Invoke New-Item -Times 0 -Exactly
        Should -Invoke Set-Content -Times 0 -Exactly
    }

    It 'rejects unknown positional registration input before scheduler work' {
        { & $script:RegisterScript 'unexpected-positional' } | Should -Throw '*Unknown CLI argument*unexpected-positional*'
        Should -Invoke New-ScheduledTaskTrigger -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }
    It 'documents portable report and indefinite repetition without scheduler work for <HelpFlag>' -TestCases @(@{ HelpFlag = '-h' }, @{ HelpFlag = '--help' }) {
        param($HelpFlag)
        $output = if ($HelpFlag -eq '-h') { & $script:RegisterScript -h | Out-String } else { & $script:RegisterScript --help | Out-String }
        $output | Should -Match 'RepetitionIntervalMinutes'
        $output | Should -Match 'repetition has no end duration'
        $output | Should -Match 'Empty uses the portable repo Reports path'
        Should -Invoke New-ScheduledTaskTrigger -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }
}
