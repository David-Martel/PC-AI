BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $helper = Join-Path $repoRoot 'Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1'
    . $helper
}

Describe 'Fleet share LAN selection' {
    BeforeEach {
        $script:testAddresses = @(
            [pscustomobject]@{ IPAddress = '192.168.50.42'; PrefixLength = 24; InterfaceIndex = 30; AddressState = 'Preferred' },
            [pscustomobject]@{ IPAddress = '10.60.4.4'; PrefixLength = 29; InterfaceIndex = 11; AddressState = 'Preferred' }
        )
        $script:testAdapters = @(
            [pscustomobject]@{ ifIndex = 30; Status = 'Up' },
            [pscustomobject]@{ ifIndex = 11; Status = 'Up' }
        )
        Mock Test-VigilNfsPort { $true }
    }
    It 'prefers the approved wired LAN and binds its source address' {
        $selected = Select-VigilNfsEndpoint -Addresses $script:testAddresses -Adapters $script:testAdapters
        $selected.Remote | Should -Be '192.168.50.2:/srv/vigil-share'
        Should -Invoke Test-VigilNfsPort -Exactly 1 -ParameterFilter {
            $Server -eq '192.168.50.2' -and $SourceAddress -eq '192.168.50.42' -and $TimeoutMilliseconds -eq 750
        }
    }
    It 'falls back to the approved direct LAN only when the primary probe fails' {
        Mock Test-VigilNfsPort { $Server -eq '10.60.4.1' }
        (Select-VigilNfsEndpoint -Addresses $script:testAddresses -Adapters $script:testAdapters).Remote | Should -Be '10.60.4.1:/srv/vigil-share'
        Should -Invoke Test-VigilNfsPort -Exactly 2
    }
    It 'rejects tentative, down, wrong-prefix and unapproved source addresses before probing' {
        foreach ($case in @('Tentative', 'Down', 'WrongPrefix', 'WrongSource')) {
            $address = [pscustomobject]@{ IPAddress = '192.168.50.42'; PrefixLength = 24; InterfaceIndex = 30; AddressState = 'Preferred' }
            $adapter = [pscustomobject]@{ ifIndex = 30; Status = 'Up' }
            switch ($case) {
                Tentative { $address.AddressState = 'Tentative' }
                Down { $adapter.Status = 'Disconnected' }
                WrongPrefix { $address.PrefixLength = 32 }
                WrongSource { $address.IPAddress = '192.168.50.99' }
            }
            Select-VigilNfsEndpoint -Addresses @($address) -Adapters @($adapter) | Should -BeNullOrEmpty
        }
        Should -Invoke Test-VigilNfsPort -Exactly 0
    }
    It 'returns unavailable when neither bounded port probe succeeds' {
        Mock Test-VigilNfsPort { $false }
        Select-VigilNfsEndpoint -Addresses $script:testAddresses -Adapters $script:testAdapters | Should -BeNullOrEmpty
        Should -Invoke Test-VigilNfsPort -Exactly 2
    }
}

Describe 'NFS mount identity and changes' {
    BeforeEach {
        $script:nfsMounted = $false
        $script:nfsRemote = '192.168.50.2:/srv/vigil-share'
        $script:nativeFailure = $false
        $script:occupied = @()
        # Optional Windows feature presence is external I/O, like the native
        # mount command below. Hosted runners do not install Client for NFS.
        Mock Test-Path { $true } -ParameterFilter {
            $PathType -eq 'Leaf' -and $LiteralPath -in @(
                (Join-Path $env:WINDIR 'System32/mount.exe'),
                (Join-Path $env:WINDIR 'System32/umount.exe')
            )
        }
        Mock Get-VigilDriveLetter { $script:occupied }
        Mock Get-NetIPAddress {
            [pscustomobject]@{ IPAddress = '192.168.50.42'; PrefixLength = 24; InterfaceIndex = 30; AddressState = 'Preferred' }
        }
        Mock Get-NetAdapter { [pscustomobject]@{ ifIndex = 30; Status = 'Up' } }
        Mock Test-VigilNfsPort { $true }
        Mock Invoke-VigilNfsNative {
            param($FilePath, $Arguments)
            if ($Arguments -and $FilePath.EndsWith('umount.exe')) {
                $script:nfsMounted = $false
            }
            elseif ($Arguments) {
                if ($script:nativeFailure) { return [pscustomobject]@{ ExitCode = 53; Stdout = 'failure'; Stderr = 'unreachable' } }
                $script:nfsMounted = $true
            }
            $text = if ($script:nfsMounted) { "N: $script:nfsRemote Properties" } else { 'Local Remote Properties' }
            [pscustomobject]@{ ExitCode = 0; Stdout = $text; Stderr = '' }
        }
    }
    It 'parses real mount row forms and rejects the private backup export' {
        $parsed = @(ConvertFrom-VigilNfsMount "Local Remote Properties`nN: \\192.168.50.2\srv\vigil-share UID=-2`nP: 10.60.4.1:/srv/vigil-backups other")
        $parsed.Count | Should -Be 2
        Test-VigilOwnedNfsMount $parsed[0].Remote | Should -BeTrue
        Test-VigilOwnedNfsMount $parsed[1].Remote | Should -BeFalse
        Test-VigilOwnedNfsMount '192.168.50.2:/srv/vigil-share-extra' | Should -BeFalse
    }
    It 'has a side-effect-free dry run' {
        (Invoke-VigilFleetShare -Action Mount -Apply -DryRun).State | Should -Be 'Plan'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0
        Should -Invoke Get-NetIPAddress -Exactly 0
        Should -Invoke Test-VigilNfsPort -Exactly 0
    }
    It 'fails closed without native or network calls when Client for NFS is absent' {
        Mock Test-Path { $false } -ParameterFilter {
            $LiteralPath -eq (Join-Path $env:WINDIR 'System32/mount.exe')
        }
        { Invoke-VigilFleetShare -Action Mount -Apply } | Should -Throw '*Client for NFS mount.exe is unavailable*'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0
        Should -Invoke Test-VigilNfsPort -Exactly 0
    }
    It 'requires Apply and honors WhatIf for both state changes' {
        (Invoke-VigilFleetShare -Action Mount).State | Should -Be 'WouldMount'
        (Invoke-VigilFleetShare -Action Mount -Apply -WhatIf).State | Should -Be 'WouldMount'
        $script:nfsMounted = $true
        (Invoke-VigilFleetShare -Action Unmount).State | Should -Be 'WouldUnmount'
        (Invoke-VigilFleetShare -Action Unmount -Apply -WhatIf).State | Should -Be 'WouldUnmount'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'preserves a foreign filesystem drive without trying the server' {
        $script:occupied = @('N:')
        (Invoke-VigilFleetShare -Action Mount -Apply).State | Should -Be 'ForeignDrivePreserved'
        Should -Invoke Test-VigilNfsPort -Exactly 0
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'preserves a foreign NFS mount even for an explicit unmount' {
        $script:nfsMounted = $true
        $script:nfsRemote = '192.168.50.2:/srv/vigil-backups'
        (Invoke-VigilFleetShare -Action Unmount -Apply).State | Should -Be 'ForeignDrivePreserved'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'performs the exact bounded anonymous mount and verifies it idempotently' {
        (Invoke-VigilFleetShare -Action Mount -Apply).State | Should -Be 'Mounted'
        (Invoke-VigilFleetShare -Action Mount -Apply).State | Should -Be 'AlreadyMounted'
        Should -Invoke Invoke-VigilNfsNative -Exactly 1 -ParameterFilter {
            $FilePath.EndsWith('mount.exe') -and $Arguments.Count -eq 7 -and
            ($Arguments -join '|') -eq '-o|anon|mtype=soft|timeout=1|retry=1|192.168.50.2:/srv/vigil-share|N:' -and $TimeoutSeconds -eq 8
        }
    }
    It 'unmounts only the recognized public share and is idempotent' {
        $script:nfsMounted = $true
        (Invoke-VigilFleetShare -Action Unmount -Apply).State | Should -Be 'Unmounted'
        (Invoke-VigilFleetShare -Action Unmount -Apply).State | Should -Be 'AlreadyUnmounted'
        Should -Invoke Invoke-VigilNfsNative -Exactly 1 -ParameterFilter { $FilePath.EndsWith('umount.exe') -and ($Arguments -join '|') -eq 'N:' }
    }
    It 'skips an unreachable LAN without a mount call' {
        Mock Test-VigilNfsPort { $false }
        (Invoke-VigilFleetShare -Action Mount -Apply).State | Should -Be 'LanUnavailable'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'surfaces native mount failure without reporting success' {
        $script:nativeFailure = $true
        { Invoke-VigilFleetShare -Action Mount -Apply } | Should -Throw '*NFS mount failed*53*'
    }
    It 'preserves a drive that becomes occupied after the LAN probe' {
        Mock Test-VigilNfsPort { $script:occupied = @('N:'); $true }
        { Invoke-VigilFleetShare -Action Mount -Apply } | Should -Throw '*became occupied*preserved*'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'rejects false native success without an observed expected mapping' {
        Mock Invoke-VigilNfsNative { [pscustomobject]@{ ExitCode = 0; Stdout = 'Local Remote Properties'; Stderr = '' } }
        { Invoke-VigilFleetShare -Action Mount -Apply } | Should -Throw '*expected mapping was not observed*'
    }
    It 'reports an offline existing endpoint while preserving the public share' {
        $script:nfsMounted = $true
        Mock Test-VigilNfsPort { $false }
        (Invoke-VigilFleetShare -Action Mount -Apply).State | Should -Be 'ExistingEndpointUnavailable'
        $script:nfsMounted | Should -BeTrue
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'rechecks exact NFS identity immediately before unmount and preserves a replacement' {
        $script:nfsMounted = $true
        $script:inventoryReads = 0
        Mock Invoke-VigilNfsNative {
            $script:inventoryReads++
            $text = if ($script:inventoryReads -eq 1) { 'N: 192.168.50.2:/srv/vigil-share public' } else { 'N: 192.168.50.2:/srv/vigil-backups foreign' }
            [pscustomobject]@{ ExitCode = 0; Stdout = $text; Stderr = '' }
        }
        { Invoke-VigilFleetShare -Action Unmount -Apply } | Should -Throw '*identity changed*preserved*'
        Should -Invoke Invoke-VigilNfsNative -Exactly 0 -ParameterFilter { $Arguments.Count -gt 0 }
    }
    It 'rejects false unmount success if a drive mapping remains' {
        $script:nfsMounted = $true
        Mock Invoke-VigilNfsNative { [pscustomobject]@{ ExitCode = 0; Stdout = 'N: 192.168.50.2:/srv/vigil-share public'; Stderr = '' } }
        { Invoke-VigilFleetShare -Action Unmount -Apply } | Should -Throw '*mapping is still observed*'
    }
    It 'rejects unsafe drive and invalid deadline before native I/O' {
        { Invoke-VigilFleetShare -Drive 'C:' -Apply } | Should -Throw
        { Invoke-VigilFleetShare -Drive 'N:;whoami' -Apply } | Should -Throw
        { Invoke-VigilFleetShare -NativeTimeoutSeconds 0 } | Should -Throw
        { Invoke-VigilFleetShare -ProbeTimeoutMilliseconds 3001 } | Should -Throw
        Should -Invoke Invoke-VigilNfsNative -Exactly 0
    }
}

Describe 'Real native process deadlines' {
    It 'captures output and preserves the real exit code' {
        $result = Invoke-VigilNfsNative -FilePath (Get-Command pwsh).Source -Arguments @('-NoProfile', '-Command', '[Console]::Write("receipt"); exit 7') -TimeoutSeconds 5
        $result.Stdout | Should -Be 'receipt'
        $result.ExitCode | Should -Be 7
    }
    It 'terminates a stalled native process and its child within a bounded deadline' {
        $pidFile = Join-Path $TestDrive 'child.pid'
        $escapedPath = $pidFile.Replace("'", "''")
        $command = @"
`$start = [Diagnostics.ProcessStartInfo]::new()
`$start.FileName = '$((Get-Command pwsh).Source.Replace("'", "''"))'
`$start.UseShellExecute = `$false
`$start.CreateNoWindow = `$true
foreach (`$arg in @('-NoProfile', '-Command', 'Start-Sleep -Seconds 20')) { `$start.ArgumentList.Add(`$arg) }
`$child = [Diagnostics.Process]::Start(`$start)
[IO.File]::WriteAllText('$escapedPath', [string]`$child.Id)
Start-Sleep -Seconds 20
"@
        $clock = [Diagnostics.Stopwatch]::StartNew()
        { Invoke-VigilNfsNative -FilePath (Get-Command pwsh).Source -Arguments @('-NoProfile', '-Command', $command) -TimeoutSeconds 2 } | Should -Throw '*timed out*process tree terminated*'
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 5
        Test-Path -LiteralPath $pidFile | Should -BeTrue
        $childPid = [int](Get-Content -LiteralPath $pidFile -Raw)
        Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'kills a pipe-inheriting descendant even after its parent exits normally' {
        $pidFile = Join-Path $TestDrive 'orphan.pid'
        $escapedPath = $pidFile.Replace("'", "''")
        $pwshPath = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
        $command = @"
`$start = [Diagnostics.ProcessStartInfo]::new()
`$start.FileName = '$($pwshPath.Replace("'", "''"))'
`$start.UseShellExecute = `$false
`$start.CreateNoWindow = `$true
foreach (`$arg in @('-NoProfile', '-Command', 'Start-Sleep -Seconds 20')) { `$start.ArgumentList.Add(`$arg) }
`$child = [Diagnostics.Process]::Start(`$start)
[IO.File]::WriteAllText('$escapedPath', [string]`$child.Id)
[Console]::Write('parent-receipt')
exit 0
"@
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $receipt = Invoke-VigilNfsNative -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', $command) -TimeoutSeconds 5
        $receipt.ExitCode | Should -Be 0
        $receipt.Stdout | Should -Be 'parent-receipt'
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 5
        $childPid = [int](Get-Content -LiteralPath $pidFile -Raw)
        Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'preserves the launch error and closes resources when a program is absent' {
        { Invoke-VigilNfsNative -FilePath (Join-Path $TestDrive 'absent.exe') -TimeoutSeconds 1 } | Should -Throw '*CreateProcess suspended*'
    }
}

Describe 'Inert per-user task template' {
    It 'supports both documented short and GNU-style help aliases without reaching status' {
        Mock Get-Help { 'help receipt' }
        (& $helper -h) | Should -Be 'help receipt'
        (& $helper --help) | Should -Be 'help receipt'
        Should -Invoke Get-Help -Exactly 2 -ParameterFilter { $Detailed }
    }
    It 'produces parseable login/network task data with hidden least-privilege execution and no registration' {
        Mock Register-ScheduledTask { throw 'Unexpected task registration' }
        $plan = Get-VigilNfsTaskTemplate -ScriptPath $helper -UserId 'S-1-5-21-1-2-3-1001'
        $plan.Registered | Should -BeFalse
        $task = [xml]$plan.Xml
        $task.Task.Principals.Principal.LogonType | Should -Be 'InteractiveToken'
        $task.Task.Principals.Principal.RunLevel | Should -Be 'LeastPrivilege'
        $task.Task.Triggers.LogonTrigger.UserId | Should -Be 'S-1-5-21-1-2-3-1001'
        $task.Task.Triggers.EventTrigger.Subscription | Should -Match 'EventID=10000'
        $task.Task.Settings.MultipleInstancesPolicy | Should -Be 'IgnoreNew'
        $task.Task.Actions.Exec.Arguments | Should -Match '-WindowStyle Hidden.*-Action Mount -Apply'
        Should -Invoke Register-ScheduledTask -Exactly 0
    }
    It 'rejects invalid user identity and nonexistent task script' {
        { Get-VigilNfsTaskTemplate -ScriptPath $helper -UserId 'user;<bad>' } | Should -Throw
        { Get-VigilNfsTaskTemplate -ScriptPath (Join-Path $TestDrive 'absent.ps1') -UserId 'S-1-5-21-1-2-3-1001' } | Should -Throw
    }
}
