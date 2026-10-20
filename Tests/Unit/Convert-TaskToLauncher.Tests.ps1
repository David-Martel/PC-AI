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
        @{ Limit = 'PT72H'; Expected = 259190 }
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
