#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: Tools/Measure-TaskLauncherOverhead.ps1 starts hundreds of processes and registers
# throwaway Task Scheduler tasks. AGENTS.md requires -h/--help and a -DryRun that is proven to change
# nothing, so these tests run the script file and check that a preview starts no process, registers no
# task and writes no file. Needs: Windows (the ScheduledTasks module). The benchmark itself is not run.

BeforeAll {
    $script:Tool = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Measure-TaskLauncherOverhead.ps1'
    function Get-BenchDirs { @(Get-ChildItem -LiteralPath ([IO.Path]::GetTempPath()) -Directory -Filter 'dtm-launcher-bench-*' -ErrorAction SilentlyContinue).FullName }
    function Invoke-Tool([string]$ArgText) { & ([scriptblock]::Create("& '$($script:Tool)' $ArgText")) }
}

Describe 'Measure-TaskLauncherOverhead CLI contract' {
    BeforeEach {
        $script:Launcher = Join-Path $TestDrive 'dtm-task-run.exe'
        Set-Content -LiteralPath $script:Launcher -Value 'stub'
        $script:Json = Join-Path $TestDrive 'out\results.json'
        $script:BenchBefore = Get-BenchDirs
        # Anything that would mutate the machine or start the benchmark. None may be called in these tests.
        Mock Register-ScheduledTask { }
        Mock Start-ScheduledTask { }
        Mock Get-CimInstance { }
        Mock Set-Content { } -ParameterFilter { $LiteralPath -like '*results.json' }
    }

    It 'prints help for <Flag> and does nothing else' -ForEach @(
        @{ Flag = '-h' }; @{ Flag = '--help' }; @{ Flag = '-Help' }
    ) {
        $out = (Invoke-Tool $Flag) -join "`n"
        $out | Should -Match 'SYNOPSIS'
        $out | Should -Match 'DryRun'
        Should -Invoke Get-CimInstance -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
    }

    It 'previews with <Flag>: a plan, no process, no task, no file' -ForEach @(
        @{ Flag = '-DryRun' }; @{ Flag = '--DryRun' }
    ) {
        $plan = Invoke-Tool "-LauncherExe 'fd=$($script:Launcher)' -Runs 5 -WarmUp 1 -PopupRuns 2 -OutputJson '$($script:Json)' $Flag"
        $plan.DryRun | Should -BeTrue
        $plan.PopupTasks | Should -Be 3            # direct, conhost, one launcher
        $plan.PopupLaunches | Should -Be 6
        $plan.OverheadProcesses | Should -Be 36    # 2 children x 3 variants x (5 + 1) runs
        Should -Invoke Get-CimInstance -Times 0 -Exactly
        Should -Invoke Register-ScheduledTask -Times 0 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0 -Exactly
        Test-Path -LiteralPath (Split-Path -Parent $script:Json) | Should -BeFalse
        Get-BenchDirs | Where-Object { $_ -notin $script:BenchBefore } | Should -BeNullOrEmpty
    }

    It 'still validates launcher paths in a dry run' {
        { Invoke-Tool "-LauncherExe 'x=$(Join-Path $TestDrive 'missing.exe')' -DryRun" } | Should -Throw '*Launcher not found*'
    }

    It 'requires -LauncherExe when not asking for help' {
        { Invoke-Tool '-DryRun' } | Should -Throw '*LauncherExe*'
    }

    It 'rejects an unknown argument' {
        { Invoke-Tool "-LauncherExe 'fd=$($script:Launcher)' --bogus" } | Should -Throw '*Unknown arguments*'
    }
}
