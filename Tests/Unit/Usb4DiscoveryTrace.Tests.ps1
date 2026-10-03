BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1')
}

Describe 'USB4 trace planning is inert' {
    BeforeEach {
        Mock Test-Usb4TraceAdministrator { throw 'Planning must not check administrator state.' }
        Mock Invoke-Usb4TraceNative { throw 'Planning must not start native processes.' }
        Mock New-Item { throw 'Planning must not create directories.' }
        Mock Add-Content { throw 'Planning must not write native logs.' }
        Mock Start-Sleep { throw 'Planning must not sleep.' }
    }
    It 'returns a default plan without an output directory or any effects' {
        $result = Invoke-Usb4DiscoveryTraceMain
        $result.State | Should -Be Planned
        $result.Applied | Should -BeFalse
        $result.ReportDirectory | Should -BeNullOrEmpty
        Should -Invoke Invoke-Usb4TraceNative -Times 0
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
    }
    It 'DryRun suppresses Apply including all direct file writes' {
        $directory = Join-Path $TestDrive 'dryrun'
        $result = Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply -IsDryRun
        $result.Applied | Should -BeFalse
        Test-Path -LiteralPath $directory | Should -BeFalse
        Should -Invoke Invoke-Usb4TraceNative -Times 0
        Should -Invoke Test-Usb4TraceAdministrator -Times 0
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
    }
    It 'WhatIf suppresses Apply including direct writes and sleep' {
        $directory = Join-Path $TestDrive 'whatif'
        $result = Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply -WhatIf
        $result.Applied | Should -BeFalse
        Test-Path -LiteralPath $directory | Should -BeFalse
        Should -Invoke Invoke-Usb4TraceNative -Times 0
        Should -Invoke Test-Usb4TraceAdministrator -Times 0
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke New-Item -Times 0
    }
    It 'rejects duration and size outside the bounded limits' {
        { Invoke-Usb4DiscoveryTraceMain -Seconds 0 } | Should -Throw
        { Invoke-Usb4DiscoveryTraceMain -Seconds 61 } | Should -Throw
        { Invoke-Usb4DiscoveryTraceMain -Megabytes 31 } | Should -Throw
        { Invoke-Usb4DiscoveryTraceMain -Megabytes 129 } | Should -Throw
    }
    It 'supports literal --help without producing a capture plan' {
        $help = & (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1') --help
        $help.Synopsis | Should -Match 'Capture bounded USB4 router discovery telemetry'
        $help.PSObject.Properties.Name | Should -Not -Contain 'Applied'
        Should -Invoke Invoke-Usb4TraceNative -Times 0
    }
    It 'executes the actual default entrypoint without an omitted-arguments exception' {
        $result = & (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1')
        $result.State | Should -Be Planned
        $result.Applied | Should -BeFalse
        Should -Invoke Invoke-Usb4TraceNative -Times 0
        Should -Invoke Start-Sleep -Times 0
    }
    It 'executes the actual Apply DryRun entrypoint without writes or native calls' {
        $directory = Join-Path $TestDrive 'entrypointdry'
        $result = & (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1') -ReportDirectory $directory -Apply -DryRun
        $result.State | Should -Be Planned
        $result.Applied | Should -BeFalse
        Test-Path -LiteralPath $directory | Should -BeFalse
        Should -Invoke Invoke-Usb4TraceNative -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Start-Sleep -Times 0
    }
    It 'plans only the two official discovery providers with exact keywords and level' {
        $result = Invoke-Usb4DiscoveryTraceMain -Directory $TestDrive -Seconds 3 -Megabytes 32
        $result.Providers.Count | Should -Be 2
        $result.Providers[0] | Should -Be '{575BA31F-2B45-58C2-64FD-F5DC757B6137} 0xffffffffffffffff 5'
        $result.Providers[1] | Should -Be '{AE795D36-2B11-5EFB-C7E0-5D552BC55D6C} 0xffffffffffffffff 5'
        $result.Commands.Create.Arguments[-4] | Should -Be '-p'
        $result.Commands.Create.Arguments[-3] | Should -Be '{575BA31F-2B45-58C2-64FD-F5DC757B6137}'
        $result.Commands.Create.Arguments[-2] | Should -Be '0xffffffffffffffff'
        $result.Commands.Create.Arguments[-1] | Should -Be '5'
        $result.Commands.Create.Arguments[6] | Should -Be 'bincirc'
        $result.Commands.Create.Arguments[8] | Should -Be '32'
        $result.Commands.Update.Arguments[3] | Should -Be '-pf'
        $result.Commands.Update.Arguments[4] | Should -Be $result.Paths.Providers
        foreach ($path in $result.Paths.PSObject.Properties.Value) {
            [IO.Path]::GetDirectoryName($path) | Should -Be ([IO.Path]::GetFullPath($TestDrive))
        }
    }
}

Describe 'USB4 trace application and failure cleanup' {
    BeforeEach {
        $script:usb4Calls = [Collections.Generic.List[object]]::new()
        $script:usb4FailureStep = ''
        $script:usb4EtlPath = ''
        $script:usb4SuffixEtl = $false
        $script:usb4SecondEtl = $false
        Mock Test-Usb4TraceAdministrator { $true }
        Mock Start-Sleep {}
        Mock Invoke-Usb4TraceNative {
            param($FilePath, $Arguments, $WorkingDirectory)
            $script:usb4Calls.Add([pscustomobject]@{ FilePath = $FilePath; Arguments = $Arguments; WorkingDirectory = $WorkingDirectory })
            $step = $Arguments[0]
            if ($FilePath -eq 'tracerpt.exe') { $step = 'decode' }
            if ($step -eq 'create') { $script:usb4EtlPath = $Arguments[4] }
            if ($step -eq $script:usb4FailureStep) {
                return [pscustomobject]@{ ExitCode = 9; StandardOutput = ''; StandardError = "controlled $step failure" }
            }
            if ($step -eq 'stop') {
                $actualPath = $(if ($script:usb4SuffixEtl) { $script:usb4EtlPath -replace '\.etl$', '_000001.etl' } else { $script:usb4EtlPath })
                [IO.File]::WriteAllBytes($actualPath, [byte[]]@(1, 2, 3))
                if ($script:usb4SecondEtl) { [IO.File]::WriteAllBytes(($script:usb4EtlPath -replace '\.etl$', '_000002.etl'), [byte[]]@(4, 5, 6)) }
                [IO.File]::WriteAllBytes(($script:usb4EtlPath -replace '\.etl$', '_foreign.etl'), [byte[]]@(7, 8, 9))
            }
            if ($step -eq 'decode') { [IO.File]::WriteAllText($Arguments[2], '<Events><Event /></Events>') }
            return [pscustomobject]@{ ExitCode = 0; StandardOutput = "$step succeeded"; StandardError = '' }
        }
    }
    It 'runs create/update/start/stop/delete/decode in order and writes only the supplied directory' {
        $directory = Join-Path $TestDrive 'capture'
        $result = Invoke-Usb4DiscoveryTraceMain -Directory $directory -Seconds 2 -EnableApply
        $result.Applied | Should -BeTrue
        $result.State | Should -Be Captured
        $script:usb4Calls.Count | Should -Be 6
        @($script:usb4Calls | ForEach-Object { if ($_.FilePath -eq 'tracerpt.exe') { 'decode' } else { $_.Arguments[0] } }) -join ',' |
            Should -Be 'create,update,start,stop,delete,decode'
        foreach ($call in $script:usb4Calls) { $call.WorkingDirectory | Should -Be $directory }
        foreach ($call in @($script:usb4Calls[2], $script:usb4Calls[3], $script:usb4Calls[4])) {
            $call.Arguments[1] | Should -Be $result.CollectorName
            $call.Arguments.Count | Should -Be 2
        }
        (Get-Content -LiteralPath $result.Paths.Providers) -join ',' | Should -Be ($result.Providers -join ',')
        (Get-Content -LiteralPath $result.Paths.NativeLog).Count | Should -Be 6
        @(Get-Content -LiteralPath $result.Paths.NativeLog | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object ExitCode -NE 0).Count | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -ParameterFilter { $Seconds -eq 2 }
    }
    It 'uses unique collectors and artifacts on repeated runs' {
        $directory = Join-Path $TestDrive 'twice'
        $first = Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply
        $before = [IO.File]::ReadAllBytes($first.Paths.Etl)
        $second = Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply
        $second.CollectorName | Should -Not -Be $first.CollectorName
        $second.Paths.Etl | Should -Not -Be $first.Paths.Etl
        [IO.File]::ReadAllBytes($first.Paths.Etl) -join ',' | Should -Be ($before -join ',')
    }
    It 'resolves the real logman numeric ETL suffix and passes exactly that path to tracerpt' {
        $script:usb4SuffixEtl = $true
        $directory = Join-Path $TestDrive 'suffixcapture'
        $null = New-Item -ItemType Directory -Path $directory
        [IO.File]::WriteAllBytes((Join-Path $directory 'unrelated.etl'), [byte[]]@(9, 9, 9))
        $result = Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply
        $result.Applied | Should -BeTrue
        $result.Paths.Etl | Should -Match '_000001\.etl$'
        $result.Paths.Etl | Should -Not -Be $script:usb4EtlPath
        $result.Commands.Decode.Arguments[0] | Should -Be $result.Paths.Etl
        $script:usb4Calls[-1].Arguments[0] | Should -Be $result.Paths.Etl
        Test-Path -LiteralPath $script:usb4EtlPath | Should -BeFalse
        [IO.File]::ReadAllBytes((Join-Path $directory 'unrelated.etl')) -join ',' | Should -Be '9,9,9'
    }
    It 'refuses ambiguous nonempty own numeric ETLs after cleanup without decoding' {
        $script:usb4SuffixEtl = $true
        $script:usb4SecondEtl = $true
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'ambiguous') -EnableApply } | Should -Throw '*exactly one nonempty ETL*found 2*'
        $script:usb4Calls[-1].Arguments[0] | Should -Be delete
    }
    It 'requires explicit directory and elevation before writes' {
        { Invoke-Usb4DiscoveryTraceMain -EnableApply } | Should -Throw '*explicit ReportDirectory*'
        Mock Test-Usb4TraceAdministrator { $false }
        $directory = Join-Path $TestDrive 'notadmin'
        { Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply } | Should -Throw '*elevated*'
        Test-Path -LiteralPath $directory | Should -BeFalse
        Should -Invoke Invoke-Usb4TraceNative -Times 0
    }
    It 'cleans its collector when provider update fails without starting capture' {
        $script:usb4FailureStep = 'update'
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'updatefailed') -EnableApply } | Should -Throw '*controlled update failure*'
        @($script:usb4Calls | ForEach-Object { $_.Arguments[0] }) -join ',' | Should -Be 'create,update,delete'
        Should -Invoke Start-Sleep -Times 0
    }
    It 'attempts only its own deletion after a create failure that might be partial' {
        $script:usb4FailureStep = 'create'
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'createfailed') -EnableApply } | Should -Throw '*controlled create failure*'
        @($script:usb4Calls | ForEach-Object { $_.Arguments[0] }) -join ',' | Should -Be 'create,delete'
        $script:usb4Calls[1].Arguments[1] | Should -Be $script:usb4Calls[0].Arguments[2]
        $script:usb4Calls[1].Arguments[1] | Should -Match '^pcai-usb4-[0-9a-f]{32}$'
        Should -Invoke Start-Sleep -Times 0
    }
    It 'cleans its running collector if the capture wait fails' {
        Mock Start-Sleep { throw 'controlled wait failure' }
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'waitfailed') -EnableApply } | Should -Throw '*controlled wait failure*'
        @($script:usb4Calls | ForEach-Object { $_.Arguments[0] }) -join ',' | Should -Be 'create,update,start,stop,delete'
    }
    It 'attempts stop and delete when start fails and surfaces cleanup errors' {
        $script:usb4FailureStep = 'start'
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'startfailed') -EnableApply } | Should -Throw '*controlled start failure*'
        @($script:usb4Calls | ForEach-Object { $_.Arguments[0] }) -join ',' | Should -Be 'create,update,start,stop,delete'
        Should -Invoke Start-Sleep -Times 0
    }
    It 'still deletes its collector when stop fails and does not decode a failed capture' {
        $script:usb4FailureStep = 'stop'
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'stopfailed') -EnableApply } | Should -Throw '*controlled stop failure*'
        @($script:usb4Calls | ForEach-Object { $_.Arguments[0] }) -join ',' | Should -Be 'create,update,start,stop,delete'
    }
    It 'surfaces delete failure and does not claim completion' {
        $script:usb4FailureStep = 'delete'
        { Invoke-Usb4DiscoveryTraceMain -Directory (Join-Path $TestDrive 'deletefailed') -EnableApply } | Should -Throw '*controlled delete failure*'
        $script:usb4Calls[-1].Arguments[0] | Should -Be delete
    }
    It 'retains native decode failure evidence after collector cleanup' {
        $script:usb4FailureStep = 'decode'
        $directory = Join-Path $TestDrive 'decodefailed'
        { Invoke-Usb4DiscoveryTraceMain -Directory $directory -EnableApply } | Should -Throw '*controlled decode failure*'
        $script:usb4Calls[4].Arguments[0] | Should -Be delete
        $log = @(Get-ChildItem -LiteralPath $directory -Filter '*.native.jsonl')
        (Get-Content -LiteralPath $log[0].FullName | Select-Object -Last 1 | ConvertFrom-Json).ExitCode | Should -Be 9
    }
}

Describe 'USB4 native process exit handling' {
    It 'kills the owned process when it exceeds the native deadline' {
        $pidFile = Join-Path $TestDrive 'usb4-stalled.pid'
        $command = "[IO.File]::WriteAllText('$($pidFile.Replace("'", "''"))', [string]`$PID); Start-Sleep -Seconds 30"
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-Usb4TraceNative -FilePath (Join-Path $PSHOME 'pwsh.exe') -Arguments @('-NoProfile', '-Command', $command) -WorkingDirectory $TestDrive -TimeoutSeconds 2
        $result.ExitCode | Should -Be -1
        $result.StandardError | Should -Match 'timed out.*process tree terminated'
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 4
        Test-Path -LiteralPath $pidFile | Should -BeTrue
        Get-Process -Id ([int](Get-Content -LiteralPath $pidFile -Raw)) -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'captures parent output promptly and kills an exited parents pipe-inheriting descendant' {
        $pidFile = Join-Path $TestDrive 'usb4-orphan.pid'
        $pwshPath = Join-Path $PSHOME 'pwsh.exe'
        $command = @"
`$start = [Diagnostics.ProcessStartInfo]::new()
`$start.FileName = '$($pwshPath.Replace("'", "''"))'
`$start.UseShellExecute = `$false
`$start.CreateNoWindow = `$true
foreach (`$value in @('-NoProfile', '-Command', 'Start-Sleep -Seconds 30')) { `$start.ArgumentList.Add(`$value) }
`$child = [Diagnostics.Process]::Start(`$start)
[IO.File]::WriteAllText('$($pidFile.Replace("'", "''"))', [string]`$child.Id)
[Console]::Write('usb4-parent-receipt')
exit 0
"@
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-Usb4TraceNative -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', $command) -WorkingDirectory $TestDrive -TimeoutSeconds 3
        $result.ExitCode | Should -Be 0
        $result.StandardOutput | Should -BeExactly 'usb4-parent-receipt'
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 3
        Get-Process -Id ([int](Get-Content -LiteralPath $pidFile -Raw)) -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'returns actual native failure output and exit status without hiding it' {
        $result = Invoke-Usb4TraceNative -FilePath (Join-Path $PSHOME 'pwsh.exe') `
            -Arguments @('-NoProfile', '-Command', '[Console]::Error.WriteLine("native controlled failure"); exit 7') -WorkingDirectory $TestDrive
        $result.ExitCode | Should -Be 7
        $result.StandardError | Should -Match 'native controlled failure'
    }
}
