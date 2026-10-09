# All processes and state paths in this suite are fixture-owned. No proxy,
# service, registry, profile or default ProgramData state is touched.
BeforeAll {
    $script:ProxySourceRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.Virtualization/Public'
    foreach ($leaf in 'Start-HVSockProxy.ps1', 'Stop-HVSockProxy.ps1', 'Get-HVSockProxyStatus.ps1') {
        . (Join-Path $script:ProxySourceRoot $leaf)
    }

    function New-OwnedProxyFixtureChild {
        $start = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardInput = $true
        foreach ($argument in '-NoLogo', '-NoProfile', '-EncodedCommand',
            [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes('[Console]::ReadLine() | Out-Null'))) {
            $null = $start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::Start($start)
        $script:OwnedProxyChildren.Add($process)
        if ($process.HasExited) { throw 'Owned fixture child exited before admission.' }
        $ready = [Diagnostics.Stopwatch]::StartNew()
        while (-not $process.MainModule.FileName) {
            if ($ready.ElapsedMilliseconds -ge 5000 -or $process.HasExited) { throw 'Owned child executable identity did not become available.' }
            Start-Sleep -Milliseconds 10
            $process.Refresh()
        }
        return $process
    }

    function Write-OwnedProxyFixtureState {
        param([Diagnostics.Process]$Process, [string]$ExecutablePath, [long]$StartTicks)
        $entry = [pscustomobject]@{
            Name = 'synthetic-proxy'; ServiceId = 'synthetic-service'; TcpTarget = 'fixture:80'
            Pid = $Process.Id; Command = 'synthetic command'; Started = 'synthetic display metadata'
            ExecutablePath = $ExecutablePath; ProcessStartTimeUtcTicks = $StartTicks
        }
        [IO.File]::WriteAllText($script:ProxyStatePath, (ConvertTo-Json -InputObject @($entry) -Depth 5))
        return (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash
    }

    function Register-HVSockServices {
        [CmdletBinding()] param($ConfigPath, [switch]$Force)
        throw 'Live registration boundary forbidden.'
    }
}

Describe 'HVSOCK proxy process custody' -Tag 'Unit', 'Virtualization', 'ProcessCustody' {
    BeforeEach {
        $script:OwnedProxyChildren = [Collections.Generic.List[Diagnostics.Process]]::new()
        $script:ProxyStatePath = Join-Path $TestDrive "private-proxy-state-$([guid]::NewGuid().ToString('N')).json"
        # An unexpected termination cannot affect any workstation process.
        Mock Stop-Process { throw 'Unadmitted termination boundary.' }
    }
    AfterEach {
        foreach ($child in $script:OwnedProxyChildren) {
            try {
                if (-not $child.HasExited) { $child.Kill() }
                if (-not $child.WaitForExit(5000)) { throw "Fixture child $($child.Id) did not close." }
            } finally { $child.Dispose() }
        }
    }

    It 'returns empty status and stop results for an absent private state file' {
        (Get-HVSockProxyStatus -StatePath $script:ProxyStatePath).Running | Should -Be 0
        (Stop-HVSockProxy -StatePath $script:ProxyStatePath).Stopped | Should -Be 0
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'does not report an unrelated live child as a proxy merely because its PID exists' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child (Join-Path $TestDrive 'foreign-winsocat.exe') $child.StartTime.ToUniversalTime().Ticks
        $result = Get-HVSockProxyStatus -StatePath $script:ProxyStatePath
        $result.Running | Should -Be 0
        $child.HasExited | Should -BeFalse
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
    }

    It 'rejects a stale process start identity even when the executable and PID match' {
        $child = New-OwnedProxyFixtureChild
        $null = Write-OwnedProxyFixtureState $child $child.MainModule.FileName ($child.StartTime.ToUniversalTime().Ticks - 1)
        (Get-HVSockProxyStatus -StatePath $script:ProxyStatePath).Running | Should -Be 0
        $child.HasExited | Should -BeFalse
    }

    It 'reads an explicitly selected existing state path literally' {
        $script:ProxyStatePath = Join-Path $TestDrive 'literal[proxy]-state.json'
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        $recorded = Get-Content -LiteralPath $script:ProxyStatePath -Raw | ConvertFrom-Json
        $recorded.Pid | Should -Be $child.Id
        $recorded.ExecutablePath | Should -Be $child.MainModule.FileName
        $recorded.ProcessStartTimeUtcTicks | Should -Be $child.StartTime.ToUniversalTime().Ticks
        $result = Get-HVSockProxyStatus -StatePath $script:ProxyStatePath
        $result.Entries[0].CustodyError | Should -BeNullOrEmpty
        $result.Running | Should -Be 1
        $result.Entries | Should -HaveCount 1
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
    }

    It 'does not terminate an unrelated real child recorded in a stale private state file' {
        $child = New-OwnedProxyFixtureChild
        $script:OnlyAdmittedTerminationPid = $child.Id
        $script:OnlyAdmittedTerminationProcess = $child
        $before = Write-OwnedProxyFixtureState $child (Join-Path $TestDrive 'foreign-winsocat.exe') $child.StartTime.ToUniversalTime().Ticks
        Mock Stop-Process {
            if ($Id -ne $script:OnlyAdmittedTerminationPid) { throw 'Foreign process termination forbidden.' }
            $script:OnlyAdmittedTerminationProcess.Kill()
            if (-not $script:OnlyAdmittedTerminationProcess.WaitForExit(5000)) { throw 'Owned termination did not complete.' }
        }
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath
        $child.HasExited | Should -BeFalse
        $result.Stopped | Should -Be 0
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'retains exact state bytes after a terminating stop refusal' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        Mock Stop-Process { throw 'Synthetic owned-child termination refusal.' }
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath
        $result.Stopped | Should -Be 0
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeTrue
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 1 -Exactly
    }

    It 'does not count a nonterminating stop failure as confirmed process exit' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        Mock Stop-Process { Write-Error 'Synthetic termination refused.' }
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath
        $result.Stopped | Should -Be 0
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeTrue
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 1 -Exactly
    }

    It 'supports ShouldProcess for explicit stop dry runs' {
        (Get-Command Stop-HVSockProxy).Parameters.ContainsKey('WhatIf') | Should -BeTrue
    }

    It 'supports ShouldProcess before launching proxies or registering services' {
        (Get-Command Start-HVSockProxy).Parameters.ContainsKey('WhatIf') | Should -BeTrue
    }

    It 'records exact process creation identity for a successfully launched owned child' {
        $child = New-OwnedProxyFixtureChild
        $script:SyntheticProxyProcess = $child
        $script:SyntheticProxyExecutable = $child.MainModule.FileName
        $configPath = Join-Path $TestDrive 'synthetic-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { Get-Process -Id $script:SyntheticProxyProcess.Id }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        $result = Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath
        $result.Count | Should -Be 1
        $state = Get-Content -LiteralPath $script:ProxyStatePath -Raw | ConvertFrom-Json
        $state.Pid | Should -Be $child.Id
        $state.ExecutablePath | Should -Be $child.MainModule.FileName
        $state.ProcessStartTimeUtcTicks | Should -Be $child.StartTime.ToUniversalTime().Ticks
        $child.HasExited | Should -BeFalse
    }

    It 'refuses to overwrite existing unresolved custody or launch a duplicate without Force' {
        $child = New-OwnedProxyFixtureChild
        $script:SyntheticProxyProcess = $child
        $script:SyntheticProxyExecutable = $child.MainModule.FileName
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        $configPath = Join-Path $TestDrive 'synthetic-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { Get-Process -Id $script:SyntheticProxyProcess.Id }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } | Should -Throw
        Should -Invoke Start-Process -Times 0 -Exactly
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        $child.HasExited | Should -BeFalse
    }

    It 'confirms an actual owned exit and retains the displaced original state' {
        $child = New-OwnedProxyFixtureChild
        $script:OnlyAdmittedTerminationPid = $child.Id
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        Mock Stop-Process {
            if ($InputObject.Id -ne $script:OnlyAdmittedTerminationPid) { throw 'Unowned process object.' }
            $InputObject.Kill()
        }
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath
        $timer.Stop()
        $result.Stopped | Should -Be 1
        $result.Unresolved | Should -HaveCount 0
        $child.HasExited | Should -BeTrue
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeFalse
        (Get-FileHash -LiteralPath $result.PreviousStatePath).Hash | Should -Be $before
        Write-Information "Actual owned-exit call elapsed milliseconds: $($timer.ElapsedMilliseconds)" -InformationAction Continue
    }

    It 'retains custody when an actual live child has not exited at the zero-wait deadline' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        Mock Stop-Process { }
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath -WaitForExitMilliseconds 0
        $result.Stopped | Should -Be 0
        $result.Unresolved | Should -HaveCount 1
        $result.Errors[0].Error | Should -Match 'wait expired'
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 1 -Exactly
    }

    It 'preserves a concurrent state writer after a confirmed process exit' {
        $child = New-OwnedProxyFixtureChild
        $script:OnlyAdmittedTerminationPid = $child.Id
        $null = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        $script:ConcurrentProxyBytes = '{"owner":"synthetic-later-writer"}'
        Mock Stop-Process {
            if ($InputObject.Id -ne $script:OnlyAdmittedTerminationPid) { throw 'Unowned process object.' }
            $InputObject.Kill()
            [IO.File]::WriteAllText($script:ProxyStatePath, $script:ConcurrentProxyBytes)
        }
        { Stop-HVSockProxy -StatePath $script:ProxyStatePath } | Should -Throw '*changed concurrently*'
        [IO.File]::ReadAllText($script:ProxyStatePath) | Should -BeExactly $script:ConcurrentProxyBytes
        $child.HasExited | Should -BeTrue
    }

    It 'detects same-byte replacement as a different file object' {
        $child = New-OwnedProxyFixtureChild
        $null = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        $snapshot = Get-HVSockStateSnapshot -Path $script:ProxyStatePath
        [IO.File]::Move($script:ProxyStatePath, "$script:ProxyStatePath.original")
        [IO.File]::WriteAllBytes($script:ProxyStatePath, $snapshot.Bytes)
        (Get-HVSockStateSnapshot -Path $script:ProxyStatePath).Hash | Should -Be $snapshot.Hash
        Test-HVSockStateSnapshot -Path $script:ProxyStatePath -Expected $snapshot | Should -BeFalse
        { Set-HVSockCustodyState -Path $script:ProxyStatePath -Expected $snapshot -Entries @() } | Should -Throw '*changed concurrently*'
        [IO.File]::ReadAllBytes($script:ProxyStatePath) | Should -Be $snapshot.Bytes
    }

    It 'leaves state bytes and the verified process unchanged under Stop WhatIf' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child $child.MainModule.FileName $child.StartTime.ToUniversalTime().Ticks
        $result = Stop-HVSockProxy -StatePath $script:ProxyStatePath -WhatIf
        $result.Stopped | Should -Be 0
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'creates no directory or state and launches or registers nothing under Start WhatIf' {
        $configPath = Join-Path $TestDrive 'whatif-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        $script:ProxyStatePath = Join-Path $TestDrive 'new-whatif-directory/state.json'
        Mock Start-Process { throw 'Unexpected launch.' }
        Mock Register-HVSockServices { throw 'Unexpected registration.' }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        $result = Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath -RegisterServices -WhatIf
        $result.Count | Should -Be 0
        Test-Path -LiteralPath (Split-Path $script:ProxyStatePath -Parent) | Should -BeFalse
        Should -Invoke Start-Process -Times 0 -Exactly
        Should -Invoke Register-HVSockServices -Times 0 -Exactly
    }

    It 'Force retains unverified existing state and never launches replacements' {
        $child = New-OwnedProxyFixtureChild
        $before = Write-OwnedProxyFixtureState $child (Join-Path $TestDrive 'unverified.exe') $child.StartTime.ToUniversalTime().Ticks
        $configPath = Join-Path $TestDrive 'force-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { throw 'Unexpected duplicate.' }
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath -Force } | Should -Throw '*unresolved*'
        (Get-FileHash -LiteralPath $script:ProxyStatePath).Hash | Should -Be $before
        Should -Invoke Start-Process -Times 0 -Exactly
        Should -Invoke Stop-Process -Times 0 -Exactly
        $child.HasExited | Should -BeFalse
    }

    It 'closes only the exact newly launched child after a later launch fails' {
        $child = New-OwnedProxyFixtureChild
        $script:SyntheticProxyProcess = $child
        $script:ProxyLaunchCount = 0
        $configPath = Join-Path $TestDrive 'partial-proxy.conf'
        [IO.File]::WriteAllText($configPath, "synthetic:service:fixture:80`nsecond:service:fixture:81")
        Mock Start-Process {
            $script:ProxyLaunchCount++
            if ($script:ProxyLaunchCount -eq 2) { throw 'Synthetic second launch failed.' }
            Get-Process -Id $script:SyntheticProxyProcess.Id
        }
        Mock Stop-Process {
            if ($InputObject.Id -ne $script:SyntheticProxyProcess.Id) { throw 'Unowned process object.' }
            $InputObject.Kill()
        }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } | Should -Throw '*second launch failed*'
        $child.HasExited | Should -BeTrue
        Should -Invoke Stop-Process -Times 1 -Exactly
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeFalse
    }

    It 'retains durable recovery identity when partial-launch cleanup is refused' {
        $child = New-OwnedProxyFixtureChild
        $script:SyntheticProxyProcess = $child
        $script:ProxyLaunchCount = 0
        $configPath = Join-Path $TestDrive 'refused-partial-proxy.conf'
        [IO.File]::WriteAllText($configPath, "synthetic:service:fixture:80`nsecond:service:fixture:81")
        Mock Start-Process {
            $script:ProxyLaunchCount++
            if ($script:ProxyLaunchCount -eq 2) { throw 'Synthetic second launch failed.' }
            Get-Process -Id $script:SyntheticProxyProcess.Id
        }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        $failure = $null
        try { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } catch { $failure = $_ }
        $failure.Exception.Message | Should -Match 'second launch failed'
        $recovery = $failure.Exception.Data['ProxyRecoveryStatePath']
        Test-Path -LiteralPath $recovery | Should -BeTrue
        $state = Get-Content -LiteralPath $recovery -Raw | ConvertFrom-Json
        $state.Pid | Should -Be $child.Id
        $state.ProcessStartTimeUtcTicks | Should -Be $child.StartTime.ToUniversalTime().Ticks
        $state.ExecutablePath | Should -Be $child.MainModule.FileName
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 1 -Exactly
        $launchCount = $script:ProxyLaunchCount
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath -Force } | Should -Throw '*Pending proxy recovery*'
        $script:ProxyLaunchCount | Should -Be $launchCount
        Test-Path -LiteralPath $recovery | Should -BeTrue
    }

    It 'closes its exact new child when primary state publication fails' {
        $child = New-OwnedProxyFixtureChild
        $script:SyntheticProxyProcess = $child
        $configPath = Join-Path $TestDrive 'publication-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { Get-Process -Id $script:SyntheticProxyProcess.Id }
        Mock Set-HVSockCustodyState { throw 'Synthetic publication failed.' }
        Mock Stop-Process {
            if ($InputObject.Id -ne $script:SyntheticProxyProcess.Id) { throw 'Unowned process object.' }
            $InputObject.Kill()
        }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } | Should -Throw '*publication failed*'
        $child.HasExited | Should -BeTrue
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeFalse
    }

    It 'preserves the original failure and exact retained child when creation-time recovery metadata is unavailable' {
        $child = New-OwnedProxyFixtureChild
        $script:FaultyReturnedProxy = Get-Process -Id $child.Id
        $script:ProxyGetterReads = 0
        Add-Member -InputObject $script:FaultyReturnedProxy -MemberType ScriptProperty -Name StartTime -Value {
            $script:ProxyGetterReads++
            throw 'Synthetic creation-time getter unavailable.'
        } -Force
        $configPath = Join-Path $TestDrive 'getter-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { $script:FaultyReturnedProxy }
        Mock Stop-Process { throw 'Synthetic exact-child termination refused.' }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        $failure = $null
        try { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } catch { $failure = $_ }
        $script:ProxyGetterReads | Should -Be 1
        $failure.Exception.Data['ProxyRecoveryEntries'] | Should -HaveCount 1
        $reference = $failure.Exception.Data['ProxyRecoveryProcessReferences'][0]
        [object]::ReferenceEquals($reference, $script:FaultyReturnedProxy) | Should -BeTrue
        [object]::ReferenceEquals($failure.Exception, $failure.Exception.Data['ProxyOriginalOperationException']) | Should -BeTrue
        $failure.Exception.Data['ProxyOriginalErrorRecord'].Exception | Should -Be $failure.Exception
        $child.HasExited | Should -BeFalse
        Should -Invoke Stop-Process -Times 1 -Exactly
        $recovery = $failure.Exception.Data['ProxyRecoveryStatePath']
        $state = Get-Content -LiteralPath $recovery -Raw | ConvertFrom-Json
        $state.MetadataIncomplete | Should -BeTrue
        $state.ProcessStartTimeUtcTicks | Should -BeNullOrEmpty
        (Get-HVSockProxyStatus -StatePath $recovery).Running | Should -Be 0
        (Stop-HVSockProxy -StatePath $recovery).Stopped | Should -Be 0
        Should -Invoke Stop-Process -Times 1 -Exactly
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath -Force } | Should -Throw '*Pending proxy recovery*'
        # A native method on the exact caller-owned reference can close it even
        # though PowerShell's targeted metadata getter is unavailable.
        $reference.Kill(); $reference.WaitForExit(5000) | Should -BeTrue
        $reference.Dispose()
        $child.HasExited | Should -BeTrue
    }

    It 'retains unverified-handle child custody without auto-kill and permits exact caller closure' {
        $child = New-OwnedProxyFixtureChild
        $script:FaultyReturnedProxy = Get-Process -Id $child.Id
        $script:ProxyGetterReads = 0
        Add-Member -InputObject $script:FaultyReturnedProxy -MemberType ScriptProperty -Name Handle -Value {
            $script:ProxyGetterReads++
            throw 'Synthetic native handle getter unavailable.'
        } -Force
        $configPath = Join-Path $TestDrive 'handle-proxy.conf'
        [IO.File]::WriteAllText($configPath, 'synthetic:service:fixture:80')
        Mock Start-Process { $script:FaultyReturnedProxy }
        Mock Get-Command { [pscustomobject]@{ Path = [Environment]::ProcessPath } } -ParameterFilter { $Name -eq 'winsocat.exe' }
        $failure = $null
        try { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath } catch { $failure = $_ }
        $failure.Exception.Message | Should -Match 'handle could not be verified'
        $script:ProxyGetterReads | Should -Be 1
        Should -Invoke Stop-Process -Times 0 -Exactly
        $child.HasExited | Should -BeFalse
        $reference = $failure.Exception.Data['ProxyRecoveryProcessReferences'][0]
        [object]::ReferenceEquals($reference, $script:FaultyReturnedProxy) | Should -BeTrue
        [object]::ReferenceEquals($failure.Exception, $failure.Exception.Data['ProxyOriginalOperationException']) | Should -BeTrue
        $recovery = $failure.Exception.Data['ProxyRecoveryStatePath']
        (Get-Content -LiteralPath $recovery -Raw | ConvertFrom-Json).MetadataIncomplete | Should -BeTrue
        (Get-HVSockProxyStatus -StatePath $recovery).Running | Should -Be 0
        (Stop-HVSockProxy -StatePath $recovery).Stopped | Should -Be 0
        { Start-HVSockProxy -ConfigPath $configPath -StatePath $script:ProxyStatePath -Force } | Should -Throw '*Pending proxy recovery*'
        Should -Invoke Stop-Process -Times 0 -Exactly
        Test-Path -LiteralPath $script:ProxyStatePath | Should -BeFalse
        $reference.Kill(); $reference.WaitForExit(5000) | Should -BeTrue
        $reference.Dispose()
        $child.HasExited | Should -BeTrue
    }

    It 'rejects reserved state leaf <Leaf> before termination or creation' -TestCases @(
        @{ Leaf = '$null' }, @{ Leaf = 'NUL' }, @{ Leaf = 'con.txt' }, @{ Leaf = 'COM1.log' }, @{ Leaf = 'AUX. ' }, @{ Leaf = 'LPT9' }
    ) {
        param($Leaf)
        { Stop-HVSockProxy -StatePath (Join-Path $TestDrive $Leaf) } | Should -Throw '*Reserved*'
        Should -Invoke Stop-Process -Times 0 -Exactly
    }

    It 'rejects an actual junction ancestor before touching its target state' {
        $target = Join-Path $TestDrive 'junction-target'
        $alias = Join-Path $TestDrive 'junction-alias'
        $null = New-Item -ItemType Directory -Path $target
        $null = New-Item -ItemType Junction -Path $alias -Target $target
        try {
            { Stop-HVSockProxy -StatePath (Join-Path $alias 'state.json') } | Should -Throw '*Linked*'
            Test-Path -LiteralPath (Join-Path $target 'state.json') | Should -BeFalse
            Should -Invoke Stop-Process -Times 0 -Exactly
        } finally { Remove-Item -LiteralPath $alias -Force }
    }
}
