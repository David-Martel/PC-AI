#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: Convert-TaskToLauncher.ps1 rewrites the action of live scheduled tasks. The
# string transformation is the part that can silently break a task (a dropped quote, a lost
# argument), so it is pinned here without touching Task Scheduler.

BeforeAll {
    $script:Tool = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Convert-TaskToLauncher.ps1'
    # Dot-sourcing defines the functions and skips the conversion (see the script's last line).
    . $script:Tool -TaskName 'unused'
    Set-StrictMode -Off
    $ErrorActionPreference = 'Continue'

    function New-TestExecAction {
        param([string]$Execute, [string]$Argument, [string]$WorkingDirectory)
        # Every field is present (as the real cmdlet's object has them), and the type name Set-ScheduledTask's
        # -Action parameter checks for is added.
        $props = @{ Execute = $Execute; Arguments = [string]$Argument; WorkingDirectory = [string]$WorkingDirectory }
        $action = New-CimInstance -ClassName MSFT_TaskExecAction -Namespace 'Root/Microsoft/Windows/TaskScheduler' -ClientOnly -Property $props
        $action.PSObject.TypeNames.Insert(0, 'Microsoft.Management.Infrastructure.CimInstance#MSFT_TaskAction')
        $action
    }
}

Describe 'ConvertTo-LauncherArguments' {

    It 'carries the child command line over character for character' {
        $child = '"C:\Program Files\PowerShell\7\pwsh.exe" -NonInteractive -File "C:\a b\x.ps1" -Quiet'
        $r = ConvertTo-LauncherArguments -Execute 'C:\WINDOWS\System32\conhost.exe' -Arguments "--headless $child" -Name 'My Task'
        $r | Should -Be "--name `"My Task`" -- $child"
    }

    It 'adds --timeout and --notify-bus only when asked' {
        $r = ConvertTo-LauncherArguments -Execute 'conhost.exe' -Arguments '--headless cmd.exe /c x.cmd' -Name 'T' -TimeoutSeconds 590 -NotifyBus
        $r | Should -Be '--name "T" --timeout 590 --notify-bus -- cmd.exe /c x.cmd'
        (ConvertTo-LauncherArguments -Execute 'conhost.exe' -Arguments '--headless cmd.exe' -Name 'T') | Should -Not -Match 'timeout|notify'
    }

    It 'accepts a quoted conhost.exe path' {
        ConvertTo-LauncherArguments -Execute '"C:\WINDOWS\System32\conhost.exe"' -Arguments '--headless a.exe' -Name 'T' | Should -Not -BeNullOrEmpty
    }

    It 'refuses anything that is not a conhost --headless wrapper' -ForEach @(
        @{ Execute = 'pwsh.exe'; Arguments = '--headless x' }
        @{ Execute = 'C:\WINDOWS\System32\conhost.exe'; Arguments = '-File x.ps1' }
        @{ Execute = 'C:\x\dtm-task-run.exe'; Arguments = '--name T -- a.exe' }
        @{ Execute = 'conhost.exe'; Arguments = '--headless' }
    ) {
        ConvertTo-LauncherArguments -Execute $Execute -Arguments $Arguments -Name 'T' | Should -BeNullOrEmpty
    }

    It 'is idempotent: an already-converted action is not wrapped twice' {
        $once = ConvertTo-LauncherArguments -Execute 'conhost.exe' -Arguments '--headless a.exe' -Name 'T'
        ConvertTo-LauncherArguments -Execute 'C:\x\dtm-task-run.exe' -Arguments $once -Name 'T' | Should -BeNullOrEmpty
    }
}

Describe 'Get-DefaultTimeoutSeconds' {
    It 'is the execution time limit minus ten seconds (<Limit>)' -ForEach @(
        @{ Limit = 'PT10M'; Expected = 590 }
        @{ Limit = 'PT3M'; Expected = 170 }
        @{ Limit = 'PT2M'; Expected = 110 }
        @{ Limit = 'PT24H'; Expected = 86390 }
    ) {
        Get-DefaultTimeoutSeconds $Limit | Should -Be $Expected
    }

    It 'never goes below one second' {
        Get-DefaultTimeoutSeconds 'PT5S' | Should -Be 1
    }

    It 'is zero (no launcher timeout) for daemons and runners with no limit (<Limit>)' -ForEach @(
        @{ Limit = 'PT0S' }; @{ Limit = '' }; @{ Limit = $null }
    ) {
        Get-DefaultTimeoutSeconds $Limit | Should -Be 0
    }
}

Describe 'Get-TaskBackupName' {
    It 'follows the scheduler agent convention (<Name>)' -ForEach @(
        @{ Path = '\'; Name = 'UDP Socket Monitor'; Expected = 'UDP_Socket_Monitor' }
        @{ Path = '\PowerShell\'; Name = 'ProfileLogSync'; Expected = 'PowerShell_ProfileLogSync' }
        @{ Path = '\'; Name = 'PC-AI Process Lasso Governor Watchdog'; Expected = 'PC-AI_Process_Lasso_Governor_Watchdog' }
    ) {
        Get-TaskBackupName $Path $Name | Should -Be $Expected
    }
}

Describe 'Get-DefaultTimeoutSeconds launcher bound' {
    # Why: dtm-task-run rejects --timeout above 86400 s with exit 87 and never starts the child
    # (LEARNED_RULES 85). Task Scheduler's default limit is PT72H, so without this bound the converter
    # would turn every default task into one that never runs. Detects: the bound being dropped or moved.
    It 'gives no launcher timeout when the limit minus ten seconds exceeds 86400 (<Limit>)' -ForEach @(
        @{ Limit = 'PT72H' }; @{ Limit = 'P1DT2H' }; @{ Limit = 'P3D' }
    ) {
        Get-DefaultTimeoutSeconds $Limit | Should -Be 0
    }

    It 'keeps a limit whose result is exactly the maximum' {
        Get-DefaultTimeoutSeconds 'PT86410S' | Should -Be 86400
    }
}

Describe 'Get-BackupExecAction' {
    # Why: -Revert rebuilds the action from the pre-launcher backup, so every field the forward path
    # keeps must be read back. Detects: WorkingDirectory (or Arguments) silently dropped on rollback.
    It 'reads Command, Arguments and WorkingDirectory from an exported task' {
        $xml = Join-Path $TestDrive 'pre.xml'
        Set-Content -LiteralPath $xml -Encoding utf8 -Value @'
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Actions Context="Author">
    <Exec>
      <Command>C:\WINDOWS\System32\conhost.exe</Command>
      <Arguments>--headless pwsh.exe -File x.ps1</Arguments>
      <WorkingDirectory>C:\work dir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
'@
        $a = Get-BackupExecAction -Path $xml
        $a.Command | Should -Be 'C:\WINDOWS\System32\conhost.exe'
        $a.Arguments | Should -Be '--headless pwsh.exe -File x.ps1'
        $a.WorkingDirectory | Should -Be 'C:\work dir'
    }

    It 'returns empty strings for absent optional elements' {
        $xml = Join-Path $TestDrive 'pre-min.xml'
        Set-Content -LiteralPath $xml -Encoding utf8 -Value '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Actions><Exec><Command>a.exe</Command></Exec></Actions></Task>'
        $a = Get-BackupExecAction -Path $xml
        $a.Arguments | Should -Be ''
        $a.WorkingDirectory | Should -Be ''
    }
}

Describe 'Invoke-Conversion against a mocked Task Scheduler' {
    # Why: AGENTS.md requires that a write-capable script's dry run is proven to change nothing. These
    # tests run the real conversion and rollback code with Task Scheduler mocked, and check both the
    # scheduler calls and the file system. Needs: the ScheduledTasks module (Windows). Detects: a dry run
    # that creates the backup directory, exports XML or calls Set-ScheduledTask; a rollback that loses
    # WorkingDirectory; a --timeout above the launcher maximum.
    BeforeAll {
        $script:FakeTask = {
            param([string]$Arguments = '--headless pwsh.exe -File x.ps1', [string]$Limit = 'PT10M')
            [pscustomobject]@{
                TaskPath = '\'; TaskName = 'Fake Task'
                Actions  = @([pscustomobject]@{ Execute = 'C:\WINDOWS\System32\conhost.exe'; Arguments = $Arguments; WorkingDirectory = 'C:\wd' })
                Settings = [pscustomobject]@{ ExecutionTimeLimit = $Limit }
            }
        }
    }

    BeforeEach {
        $TaskName = @('\Fake Task')
        $NotifyBus = @()
        $TimeoutSeconds = @{}
        $Revert = $false
        $BackupDir = Join-Path $TestDrive ("backups-" + [guid]::NewGuid().ToString('N'))
        $LauncherPath = Join-Path $TestDrive 'dtm-task-run.exe'
        Set-Content -LiteralPath $LauncherPath -Value 'stub'

        Mock Get-ScheduledTask { & $script:FakeTask }
        Mock Get-ScheduledTaskInfo { [pscustomobject]@{ LastTaskResult = 0 } }
        Mock Export-ScheduledTask { '<Task />' }
        # Set-ScheduledTask is a CDXML cmdlet whose -Action must be a real CimInstance, so the mock builds a
        # client-only MSFT_TaskExecAction (no scheduler access) carrying the fields the script passed.
        Mock New-ScheduledTaskAction { New-TestExecAction -Execute $Execute -Argument $Argument -WorkingDirectory $WorkingDirectory }
        Mock Set-ScheduledTask { }
    }

    It 'previews under -WhatIf without touching Task Scheduler or the file system' {
        $rows = @(Invoke-Conversion -WhatIf)
        $rows.Count | Should -Be 1
        $rows[0].NewArguments | Should -Be '--name "Fake Task" --timeout 590 -- pwsh.exe -File x.ps1'
        Should -Invoke Set-ScheduledTask -Times 0 -Exactly
        Should -Invoke Export-ScheduledTask -Times 0 -Exactly
        Should -Invoke New-ScheduledTaskAction -Times 0 -Exactly
        Test-Path -LiteralPath $BackupDir | Should -BeFalse
    }

    It 'converts for real when not previewing (positive control for the dry-run test)' {
        @(Invoke-Conversion).Count | Should -Be 1
        Should -Invoke Set-ScheduledTask -Times 1 -Exactly -ParameterFilter {
            $Action.Execute -eq $LauncherPath -and $Action.WorkingDirectory -eq 'C:\wd'
        }
        Test-Path -LiteralPath (Join-Path $BackupDir 'pre-launcher__Fake_Task.xml') | Should -BeTrue
    }

    It 'gives a PT72H task no --timeout instead of one the launcher rejects' {
        Mock Get-ScheduledTask { & $script:FakeTask -Limit 'PT72H' }
        $row = @(Invoke-Conversion -WhatIf)[0]
        $row.TimeoutSeconds | Should -Be 0
        $row.NewArguments | Should -Not -Match '--timeout'
    }

    It 'refuses an explicit -TimeoutSeconds above 86400' {
        $TimeoutSeconds = @{ '\Fake Task' = 90000 }
        { Invoke-Conversion -WhatIf } | Should -Throw '*86400*'
        Should -Invoke Set-ScheduledTask -Times 0 -Exactly
    }

    It 'restores command, arguments and working directory on -Revert' {
        $Revert = $true
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $BackupDir 'pre-launcher__Fake_Task.xml') -Encoding utf8 -Value '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Actions><Exec><Command>C:\WINDOWS\System32\conhost.exe</Command><Arguments>--headless a.exe</Arguments><WorkingDirectory>C:\wd</WorkingDirectory></Exec></Actions></Task>'
        Invoke-Conversion
        Should -Invoke Set-ScheduledTask -Times 1 -Exactly -ParameterFilter {
            $Action.Execute -eq 'C:\WINDOWS\System32\conhost.exe' -and $Action.Arguments -eq '--headless a.exe' -and $Action.WorkingDirectory -eq 'C:\wd'
        }
    }

    It 'changes nothing on -Revert under -WhatIf' {
        $Revert = $true
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $BackupDir 'pre-launcher__Fake_Task.xml') -Encoding utf8 -Value '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Actions><Exec><Command>a.exe</Command></Exec></Actions></Task>'
        Invoke-Conversion -WhatIf
        Should -Invoke Set-ScheduledTask -Times 0 -Exactly
        Should -Invoke New-ScheduledTaskAction -Times 0 -Exactly
    }
}

Describe 'Convert-TaskToLauncher CLI contract' {
    # Why: AGENTS.md requires -h, --help and a non-mutating -DryRun on write-capable scripts. Running the
    # script file proves the switches are wired to the conversion, not just declared. Detects: help that
    # reaches Task Scheduler, a -DryRun/--DryRun that does not reach ShouldProcess, stray arguments
    # accepted silently.
    BeforeEach {
        $script:Backups = Join-Path $TestDrive ("cli-" + [guid]::NewGuid().ToString('N'))
        $script:Launcher = Join-Path $TestDrive 'dtm-task-run.exe'
        Set-Content -LiteralPath $script:Launcher -Value 'stub'
        Mock Get-ScheduledTask {
            [pscustomobject]@{
                TaskPath = '\'; TaskName = 'Fake Task'
                Actions  = @([pscustomobject]@{ Execute = 'conhost.exe'; Arguments = '--headless a.exe'; WorkingDirectory = '' })
                Settings = [pscustomobject]@{ ExecutionTimeLimit = 'PT10M' }
            }
        }
        Mock Get-ScheduledTaskInfo { [pscustomobject]@{ LastTaskResult = 0 } }
        Mock Export-ScheduledTask { '<Task />' }
        Mock New-ScheduledTaskAction { New-TestExecAction -Execute $Execute }
        Mock Set-ScheduledTask { }
    }

    It 'prints help for <Flag> without reading Task Scheduler' -ForEach @(
        @{ Flag = '-h' }; @{ Flag = '--help' }; @{ Flag = '-Help' }
    ) {
        $out = (& ([scriptblock]::Create("& '$($script:Tool)' $Flag"))) -join "`n"
        $out | Should -Match 'SYNOPSIS'
        $out | Should -Match 'DryRun'
        Should -Invoke Get-ScheduledTask -Times 0 -Exactly
    }

    It 'previews with <Flag> and changes nothing' -ForEach @(
        @{ Flag = '-DryRun' }; @{ Flag = '--DryRun' }
    ) {
        $rows = @(& ([scriptblock]::Create("& '$($script:Tool)' -TaskName '\Fake Task' -LauncherPath '$($script:Launcher)' -BackupDir '$($script:Backups)' $Flag")))
        $rows.Count | Should -Be 1
        Should -Invoke Get-ScheduledTask -Times 1 -Exactly
        Should -Invoke Set-ScheduledTask -Times 0 -Exactly
        Should -Invoke Export-ScheduledTask -Times 0 -Exactly
        Test-Path -LiteralPath $script:Backups | Should -BeFalse
    }

    It 'rejects an unknown argument' {
        { & $script:Tool -TaskName '\Fake Task' --bogus } | Should -Throw '*Unknown arguments*'
    }

    It 'requires -TaskName when not asking for help' {
        { & $script:Tool -LauncherPath $script:Launcher -BackupDir $script:Backups -DryRun } | Should -Throw '*TaskName*'
    }
}
