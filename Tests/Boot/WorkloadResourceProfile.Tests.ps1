#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
BeforeAll {
    # Portable tests mock the Windows-only commands; no platform provider is used.
    if (-not (Get-Command Get-Counter -ErrorAction SilentlyContinue)) {
        function Get-Counter { [CmdletBinding()] param([string[]]$Counter) throw 'Windows counter provider is unavailable.' }
    }
    if (-not (Get-Command Get-CimInstance -ErrorAction SilentlyContinue)) {
        function Get-CimInstance { [CmdletBinding()] param([string]$ClassName, [string[]]$Property, [int]$OperationTimeoutSec) throw 'Windows CIM provider is unavailable.' }
    }
    $script:CollectorPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools/Collect-WorkloadResourceProfile.ps1'
    $tokens = $null; $parseErrors = $null
    $script:CollectorAst = [Management.Automation.Language.Parser]::ParseFile($script:CollectorPath, [ref]$tokens, [ref]$parseErrors)
    $script:ParseErrors = $parseErrors
    # Load real pure helpers and query helpers without executing the collector.
    foreach ($definition in $script:CollectorAst.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
        . ([scriptblock]::Create($definition.Extent.Text))
    }
    function New-ProcessFixture {
        param([double]$Cpu = 1, [string]$Creation = '2026-10-05T12:00:00Z', [int]$ProcessId = 42)
        [pscustomobject]@{ Pid = $ProcessId; CreationUtc = $Creation; Identity = "${ProcessId}:$Creation";
            CpuSeconds = $Cpu; IoReadBytes = 1000.0; IoWriteBytes = 2000.0 }
    }
}
Describe 'Workload resource collector' -Tag 'Unit', 'Boot', 'Portable' {
    It 'parses and excludes machine mutation commands' {
        @($script:ParseErrors).Count | Should -Be 0
        $commands = $script:CollectorAst.FindAll({ param($node) $node -is [Management.Automation.Language.CommandAst] }, $true) |
            ForEach-Object GetCommandName
        @($commands | Where-Object { $_ -match '^(Stop-Process|Start-Process|Write-EventLog|Register-ScheduledTask|Set-ScheduledTask|Mount-VHD|Set-ItemProperty|Restart-Service)$' }).Count | Should -Be 0
    }
    It 'calculates fractional interval CPU and IO normalized to host capacity' {
        $previous = New-ProcessFixture -Cpu 20
        $current = New-ProcessFixture -Cpu 20.125
        $current.IoReadBytes = 1250; $current.IoWriteBytes = 2500
        $delta = Get-WorkloadDelta $current $previous 0.5 8
        $delta.CpuSeconds | Should -Be 0.125
        $delta.CpuHostPercent | Should -Be 3.125
        $delta.IoReadBytesPerSecond | Should -Be 500
        $delta.IoWriteBytesPerSecond | Should -Be 1000
    }
    It 'does not treat reused PIDs or new identities as lifetime interval CPU' {
        $previous = New-ProcessFixture -Cpu 800
        $current = New-ProcessFixture -Cpu 1000 -Creation '2026-10-05T13:00:00Z'
        (Get-WorkloadDelta $current $previous 10 22).CpuHostPercent | Should -BeNullOrEmpty
        (Get-WorkloadDelta $current $null 10 22).IoReadBytesPerSecond | Should -BeNullOrEmpty
    }
    It 'normalizes native and CIM creation precision without merging adjacent microseconds' {
        $native = Get-WorkloadCreationKey ([datetime]'2026-10-05T17:02:22.4697965Z')
        $cim = Get-WorkloadCreationKey ([datetime]'2026-10-05T17:02:22.4697960Z')
        $native | Should -Be $cim
        Get-WorkloadCreationKey ([datetime]'2026-10-05T17:02:22.4697970Z') | Should -Not -Be $cim
    }
    It 'keeps unavailable metrics and reset counters unknown' {
        $previous = New-ProcessFixture -Cpu 10
        $current = New-ProcessFixture -Cpu 9
        $current.IoReadBytes = $null; $current.IoWriteBytes = 10
        $delta = Get-WorkloadDelta $current $previous 10 22
        $delta.CpuSeconds | Should -BeNullOrEmpty
        $delta.IoReadBytesPerSecond | Should -BeNullOrEmpty
        $delta.IoWriteBytesPerSecond | Should -BeNullOrEmpty
        $current.CreationUtc = $null
        (Get-WorkloadDelta $current $previous 10 22).CpuHostPercent | Should -BeNullOrEmpty
    }
    It 'excludes idle CPU and zero-duration intervals' {
        $idle = New-ProcessFixture -ProcessId 0
        (Get-WorkloadDelta $idle $idle 1 22).CpuHostPercent | Should -BeNullOrEmpty
        $process = New-ProcessFixture
        (Get-WorkloadDelta $process $process 0 22).CpuHostPercent | Should -BeNullOrEmpty
    }
    It 'calculates null-aware nearest-rank percentiles' {
        Get-WorkloadPercentile @(1, 2, $null, 3, 4, 5) 50 | Should -Be 3
        Get-WorkloadPercentile @(1, 2, 3, 4, 5) 95 | Should -Be 5
        Get-WorkloadPercentile @($null, $null) 99 | Should -BeNullOrEmpty
    }
    It 'retains unknown host metrics with actionable errors' {
        Mock Get-Counter { throw 'test counter unavailable' }
        $issues = [System.Collections.Generic.List[string]]::new()
        $result = Get-WorkloadHostSample -Issues $issues
        $result.CpuPercent | Should -BeNullOrEmpty
        $result.CommitPercent | Should -BeNullOrEmpty
        $issues[0] | Should -Match 'host-counters-unavailable:test counter unavailable'
    }
    It 'reads valid counters, skips invalid values and converts disk units' {
        Mock Get-Counter {
            [pscustomobject]@{ CounterSamples = @(
                [pscustomobject]@{ Path = '\\host\processor(_total)\% processor time'; Status = 0; CookedValue = 12.5 }
                [pscustomobject]@{ Path = '\\host\memory\committed bytes'; Status = 0; CookedValue = 300 }
                [pscustomobject]@{ Path = '\\host\memory\commit limit'; Status = 0; CookedValue = 1000 }
                [pscustomobject]@{ Path = '\\host\physicaldisk(_total)\avg. disk sec/transfer'; Status = 0; CookedValue = 0.0025 }
                [pscustomobject]@{ Path = '\\host\memory\available mbytes'; Status = 99; CookedValue = 0 }
                [pscustomobject]@{ Path = '\\host\network interface(test)\bytes total/sec'; InstanceName = 'test'; Status = 0; CookedValue = 125 }
            ) }
        }
        $issues = [System.Collections.Generic.List[string]]::new()
        $result = Get-WorkloadHostSample -Issues $issues
        $result.CpuPercent | Should -Be 12.5
        $result.CommitPercent | Should -Be 30
        $result.DiskTransferLatencyMs | Should -Be 2.5
        $result.AvailableMB | Should -BeNullOrEmpty
        $result.NetworkAdapters[0].BytesPerSecond | Should -Be 125
        $issues.Count | Should -Be 1
    }
    It 'classifies shared runtimes as needing context rather than waste' {
        Get-WorkloadCohort 'node.exe' | Should -Be 'SharedRuntimesNeedsContext'
        Get-WorkloadCohort 'codex.exe' | Should -Be 'LLMTools'
        Get-WorkloadCohort 'git.exe' | Should -Be 'BuildGitEvaluation'
        Get-WorkloadCohort 'ms-teams.exe' | Should -Be 'RealtimeMedia'
    }
    It 'performs no queries, sleeps or output writes during dry run' {
        Mock Get-CimInstance { throw 'DryRun queried CIM' }
        Mock Get-Counter { throw 'DryRun queried counters' }
        Mock Start-Sleep { throw 'DryRun slept' }
        $target = Join-Path $TestDrive 'dryrun-output'
        $result = & $script:CollectorPath -DryRun -OutputPath $target
        $result.DryRun | Should -BeTrue
        $target | Should -Not -Exist
        Should -Invoke Get-CimInstance -Times 0
        Should -Invoke Get-Counter -Times 0
        Should -Invoke Start-Sleep -Times 0
    }
    It 'uses unique non-date default artifact identities without creating directories' {
        $first = & $script:CollectorPath -DryRun
        $second = & $script:CollectorPath -DryRun
        (Split-Path -Leaf $first.OutputPath) | Should -Match '^capture-[0-9a-f]{32}$'
        (Split-Path -Leaf (Split-Path -Parent $first.OutputPath)) | Should -Be 'workload-profiles'
        $first.OutputPath | Should -Not -Be $second.OutputPath
        $first.OutputPath | Should -Not -Exist
        $second.OutputPath | Should -Not -Exist
    }
    It 'supports help forms without writes or queries' {
        Mock Get-CimInstance { throw 'Help queried CIM' }
        $target = Join-Path $TestDrive 'help-output'
        (& $script:CollectorPath -h -OutputPath $target) | Should -Match 'SYNOPSIS'
        (& $script:CollectorPath --help -OutputPath $target) | Should -Match 'SYNOPSIS'
        $target | Should -Not -Exist
        Should -Invoke Get-CimInstance -Times 0
    }
    It 'rejects reserved output leaves even in dry run' {
        { & $script:CollectorPath -DryRun -OutputPath (Join-Path $TestDrive 'NUL.txt') } | Should -Throw '*Unsafe output*'
        { & $script:CollectorPath -DryRun -OutputPath (Join-Path $TestDrive '$null') } | Should -Throw '*Unsafe output*'
    }
    It 'rejects unrecognized CLI arguments' {
        { & $script:CollectorPath --mistyped -DryRun } | Should -Throw '*Unknown CLI*'
    }
}
Describe 'Workload collector reporting window' -Tag 'Unit', 'Boot' {
    BeforeEach {
        Mock Get-CimInstance {
            param($ClassName)
            switch ($ClassName) {
                'Win32_Processor' { [pscustomobject]@{ Name = 'Fixture CPU'; NumberOfCores = 4; NumberOfLogicalProcessors = 8 } }
                'Win32_ComputerSystem' { [pscustomobject]@{ TotalPhysicalMemory = 16GB } }
                'Win32_Process' { [pscustomobject]@{ ProcessId = 42; ParentProcessId = 0; CreationDate = [datetime]'2026-10-05T12:00:00Z' } }
            }
        }
        Mock Get-Process {
            $process = [pscustomobject]@{ Id = 42; StartTime = [datetime]'2026-10-05T12:00:00Z'; ProcessName = 'node';
                TotalProcessorTime = [timespan]::FromSeconds(0.6); PrivateMemorySize64 = 1MB; WorkingSet64 = 2MB;
                Handle = [IntPtr]::Zero; HandleCount = 3; BasePriority = 8 }
            $process | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            $process
        }
        Mock Get-Counter { [pscustomobject]@{ CounterSamples = @() } }
    }
    It 'streams bounded samples with actual duration and first-sample unknown rates' -Skip:(-not $IsWindows) {
        $target = Join-Path $TestDrive 'capture'
        $result = & $script:CollectorPath -DurationMinutes 0.018 -IntervalSeconds 1 -OutputPath $target -Label test-window
        $summary = [IO.File]::ReadAllText((Join-Path $target 'summary.json')) | ConvertFrom-Json
        $samples = @([IO.File]::ReadAllLines((Join-Path $target 'samples-0001.jsonl')) | ForEach-Object { $_ | ConvertFrom-Json })
        $result.Status | Should -Be 'completed'
        $summary.Samples | Should -Be $samples.Count
        $samples.Count | Should -BeGreaterOrEqual 1
        $samples[0].Processes[0].IntervalCpuHostPercent | Should -BeNullOrEmpty
        $summary.ActualDurationSeconds | Should -BeGreaterOrEqual 1.08
        $summary.TopByPrivateBytes[0].PeakPrivateBytes | Should -Be 1MB
        $summary.TopByPrivateBytes[0].PeakInstances | Should -Be 1
        $summary.ObservedProcessIdentities | Should -Be 1
        [IO.File]::ReadAllText((Join-Path $target 'samples-0001.jsonl')) | Should -Not -Match 'CommandLine|WindowTitle|RemoteEndpoint'
        $before = Get-FileHash (Join-Path $target 'summary.json')
        { & $script:CollectorPath -DurationMinutes 0.01 -OutputPath $target } | Should -Throw '*existing evidence*'
        (Get-FileHash (Join-Path $target 'summary.json')).Hash | Should -Be $before.Hash
    }
    It 'preserves fully unknown memory and CPU in per-name summaries' -Skip:(-not $IsWindows) {
        Mock Get-Process {
            $process = [pscustomobject]@{ Id = 42; StartTime = $null; ProcessName = 'node'; TotalProcessorTime = $null;
                PrivateMemorySize64 = $null; WorkingSet64 = $null; Handle = [IntPtr]::Zero; HandleCount = 3; BasePriority = 8 }
            $process | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            $process
        }
        $target = Join-Path $TestDrive 'unknown-capture'
        & $script:CollectorPath -DurationMinutes 0.035 -IntervalSeconds 1 -OutputPath $target | Out-Null
        $summary = [IO.File]::ReadAllText((Join-Path $target 'summary.json')) | ConvertFrom-Json
        $summary.Samples | Should -BeGreaterOrEqual 2
        $summary.TopByPrivateBytes[0].PeakPrivateBytes | Should -BeNullOrEmpty
        $summary.TopByPrivateBytes[0].PeakWorkingSetBytes | Should -BeNullOrEmpty
        $summary.TopByPrivateBytes[0].MeanPrivateBytesWhenPresent | Should -BeNullOrEmpty
        $summary.TopByPrivateBytes[0].KnownPrivateSamples | Should -Be 0
        $summary.TopByCpuSeconds[0].CpuSeconds | Should -BeNullOrEmpty
        $summary.HostStats.SampledProcessCpuPercent.KnownSamples | Should -Be 0
    }
    It 'excludes rejected samples and identities when the output budget is exhausted' -Skip:(-not $IsWindows) {
        Mock Get-Process {
            $process = [pscustomobject]@{ Id = 42; StartTime = [datetime]'2026-10-05T12:00:00Z'; ProcessName = ('n' * 1MB);
                TotalProcessorTime = [timespan]::FromSeconds(0.6); PrivateMemorySize64 = 1MB; WorkingSet64 = 2MB;
                Handle = [IntPtr]::Zero; HandleCount = 3; BasePriority = 8 }
            $process | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            $process
        }
        $target = Join-Path $TestDrive 'budget-capture'
        $result = & $script:CollectorPath -DurationMinutes 0.01 -MaxOutputMB 1 -OutputPath $target
        $summary = [IO.File]::ReadAllText((Join-Path $target 'summary.json')) | ConvertFrom-Json
        $result.Status | Should -Be 'output-budget'
        $summary.Samples | Should -Be 0
        $summary.ObservedProcessIdentities | Should -Be 0
        $summary.Parts | Should -Be 0
        $summary.HostStats.CpuPercent.KnownSamples | Should -Be 0
        @(Get-ChildItem $target -Filter 'samples-*.jsonl').Count | Should -Be 0
    }
}
