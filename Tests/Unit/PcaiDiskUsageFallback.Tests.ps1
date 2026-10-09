BeforeAll {
    $script:RepoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:DiskSource=Join-Path $script:RepoRoot 'Modules/PC-AI.Acceleration/Public/Get-DiskUsageFast.ps1'
    . $script:DiskSource
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.Acceleration/Private/Get-PcaiPerfToolPath.ps1')
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1')
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.Acceleration/Private/Invoke-PcaiPerfWorker.ps1')
}
Describe 'Actual disk production pipeline fatal-failure boundary' {
    BeforeEach {
        $script:SavedEnvironment=@{}
        foreach($name in @('PCAI_NATIVE_BUNDLE_ROOT','PCAI_PREFER_RUST_CLI','PCAI_PREFER_PERF_WORKER_DISK')){$script:SavedEnvironment[$name]=[Environment]::GetEnvironmentVariable($name)}
        $env:PCAI_NATIVE_BUNDLE_ROOT=$TestDrive
        [IO.File]::WriteAllText((Join-Path $TestDrive 'pcai-perf.exe'),'No child is executed in this injected transport fixture')
        $env:PCAI_PREFER_RUST_CLI='1'
        $env:PCAI_PREFER_PERF_WORKER_DISK=$null
        $script:ToolPaths=@{dust=$null}
        $script:RustToolCache=@{dust=$null}
        $script:SearchPaths=@()
        $script:DownstreamCount=0
        $script:InjectedFailure=$null
        $script:ObservedFailure=$null
        Mock Invoke-PcaiPerfCliCommand {throw $script:InjectedFailure}
        Mock Invoke-PcaiPerfWorkerRequest {throw $script:InjectedFailure}
        Mock Get-DiskUsageParallel {
            $script:DownstreamCount++
            [pscustomobject]@{Path='downstream-marker';SizeBytes=1;FileCount=1;Tool='injected-parallel-marker'}
        }
        Mock Get-DiskUsageWithDust {
            $script:DownstreamCount++
            [pscustomobject]@{Path='downstream-marker';SizeBytes=1;FileCount=1;Tool='injected-dust-marker'}
        }
        Mock Get-DiskUsageWithNative {
            $script:DownstreamCount++
            [pscustomobject]@{Path='downstream-marker';SizeBytes=1;FileCount=1;Tool='injected-native-marker'}
        }
    }
    AfterEach {
        foreach($entry in $script:SavedEnvironment.GetEnumerator()){[Environment]::SetEnvironmentVariable($entry.Key,$entry.Value)}
    }
    It 'does not invoke downstream <Fallback> after a production <FailureKind> in <Transport>' -ForEach @(
        @{FailureKind='timeout';Transport='direct';Fallback='parallel'}
        @{FailureKind='timeout';Transport='worker';Fallback='parallel'}
        @{FailureKind='custody';Transport='direct';Fallback='parallel'}
        @{FailureKind='custody';Transport='worker';Fallback='parallel'}
        @{FailureKind='timeout';Transport='direct';Fallback='dust'}
    ) {
        param($FailureKind,$Transport,$Fallback)
        if($Transport -eq 'worker'){$env:PCAI_PREFER_PERF_WORKER_DISK='1'}
        if($Fallback -eq 'dust'){
            $dust=Join-Path $TestDrive 'dust.exe'
            [IO.File]::WriteAllText($dust,'No downstream process is executed')
            $script:ToolPaths.dust=$dust
        }
        $script:InjectedFailure=if($FailureKind -eq 'timeout'){[TimeoutException]::new('Injected existing child deadline')}
        else{
            $failure=[InvalidOperationException]::new('Injected exact owned resource closure failure')
            $failure.Data['PcaiProcessCustody']=[pscustomobject]@{SyntheticExactCustody=$true}
            $failure
        }
        $result=$null
        try{$result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)}catch{$script:ObservedFailure=$_.Exception}
        $script:DownstreamCount|Should -Be 0 -Because 'a fatal child deadline or unconfirmed custody must stop this invocation'
        $script:ObservedFailure|Should -Not -BeNullOrEmpty
        $result|Should -BeNullOrEmpty
        if($FailureKind -eq 'custody') {
            $script:ObservedFailure.Data['PcaiProcessCustody']|Should -Be $script:InjectedFailure.Data['PcaiProcessCustody']
        }
    }
    It 'retains a terminal <FailureKind> through its <Wrapper> wrapper' -ForEach @(
        @{FailureKind='timeout';Wrapper='inner'}
        @{FailureKind='cancellation';Wrapper='inner'}
        @{FailureKind='custody';Wrapper='inner'}
        @{FailureKind='timeout';Wrapper='operation-data'}
        @{FailureKind='cancellation';Wrapper='operation-data'}
        @{FailureKind='custody';Wrapper='operation-data'}
        @{FailureKind='timeout';Wrapper='aggregate-second'}
        @{FailureKind='cancellation';Wrapper='aggregate-second'}
        @{FailureKind='custody';Wrapper='aggregate-second'}
    ) {
        param($FailureKind,$Wrapper)
        $cause=switch($FailureKind) {
            'timeout' { [TimeoutException]::new('Original deadline') }
            'cancellation' { [OperationCanceledException]::new('Original cancellation') }
            'custody' {
                $failure=[InvalidOperationException]::new('Original retained custody')
                $failure.Data['PcaiProcessCustody']=[object]::new()
                $failure
            }
        }
        $script:InjectedFailure=switch($Wrapper) {
            'inner' { [InvalidOperationException]::new('Transport wrapper',$cause) }
            'operation-data' {
                $failure=[InvalidOperationException]::new('Cleanup wrapper')
                $failure.Data['OperationException']=$cause
                $failure
            }
            'aggregate-second' { [AggregateException]::new([Exception[]]@([IO.IOException]::new('Other failure'),$cause)) }
        }
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        $script:DownstreamCount|Should -Be 0
        [object]::ReferenceEquals($script:ObservedFailure,$script:InjectedFailure)|Should -BeTrue
    }
    It 'does not treat legacy NotSupported with retained custody as permission to launch direct CLI' {
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        $script:InjectedFailure=[NotSupportedException]::new('Legacy negotiation and failed cleanup')
        $custody=[object]::new()
        $script:InjectedFailure.Data['PcaiProcessCustody']=$custody
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        [object]::ReferenceEquals($script:ObservedFailure,$script:InjectedFailure)|Should -BeTrue
        [object]::ReferenceEquals($script:ObservedFailure.Data['PcaiProcessCustody'],$custody)|Should -BeTrue
        $script:DownstreamCount|Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
    It 'uses direct CLI after genuine unsupported worker negotiation' {
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        $script:InjectedFailure=[NotSupportedException]::new('Only legacy negotiation')
        Mock Invoke-PcaiPerfCliCommand { [pscustomobject]@{Path='direct-row';SizeBytes=7;FileCount=1;Tool='pcai-perf'} }
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $result.Count|Should -Be 1
        $result[0].Transport|Should -Be 'direct-cli'
        $result[0].Path|Should -Be 'direct-row'
        $script:DownstreamCount|Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 1 -Exactly
    }
    It 'keeps genuine <Transport> unavailability eligible for a downstream backend' -ForEach @(
        @{Transport='direct'}
        @{Transport='worker'}
    ) {
        param($Transport)
        if($Transport -eq 'worker'){$env:PCAI_PREFER_PERF_WORKER_DISK='1'}
        $script:InjectedFailure=[IO.FileNotFoundException]::new('Selected tool disappeared before launch')
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $script:DownstreamCount|Should -Be 1
        $result[0].Path|Should -Be 'downstream-marker'
    }
    It 'preserves successful worker transport, useful output ordering and Top contract' {
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        Mock Invoke-PcaiPerfWorkerRequest {
            [pscustomobject]@{Path='small';SizeBytes=1;FileCount=1;Tool='pcai-perf'}
            [pscustomobject]@{Path='large';SizeBytes=10;FileCount=2;Tool='pcai-perf'}
        }
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $result.Count|Should -Be 1
        $result[0].Path|Should -Be 'large'
        $result[0].SizeBytes|Should -Be 10
        $result[0].Transport|Should -Be 'worker'
        $script:DownstreamCount|Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
    It 'does not loop on a cyclic operation cause with genuine unavailability' {
        $script:InjectedFailure=[IO.FileNotFoundException]::new('Missing selected tool')
        $script:InjectedFailure.Data['OperationException']=$script:InjectedFailure
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $script:DownstreamCount|Should -Be 1
        $result[0].Path|Should -Be 'downstream-marker'
    }
    It 'preserves the original <Transport> error when pending custody appears during transport' -ForEach @(
        @{Transport='direct'}
        @{Transport='unsupported-worker'}
    ) {
        param($Transport)
        $registry=$script:PcaiPerfCustodyRegistry
        $pending=$registry.Pending
        $beforeCount=$pending.Count
        $process=[Diagnostics.Process]::new()
        $script:LatePendingState=[pscustomobject]@{Process=$process;OriginRegistry=$registry;ClosurePending=$true}
        $cause=if($Transport -eq 'direct'){[InvalidOperationException]::new('Pending appeared during direct transport')}
        else{[NotSupportedException]::new('Legacy negotiation with pending custody appearing')}
        $script:LateErrorRecord=[Management.Automation.ErrorRecord]::new($cause,'Injected.LateCustody',[Management.Automation.ErrorCategory]::ResourceBusy,$script:LatePendingState)
        Mock Invoke-PcaiPerfCliCommand {
            Register-PcaiPerfPendingState $script:LatePendingState
            throw $script:LateErrorRecord
        }
        Mock Invoke-PcaiPerfWorkerRequest {
            Register-PcaiPerfPendingState $script:LatePendingState
            throw $script:LateErrorRecord
        }
        if($Transport -eq 'unsupported-worker'){$env:PCAI_PREFER_PERF_WORKER_DISK='1'}
        $observedRecord=$null
        try {
            try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $observedRecord=$_ }
            [object]::ReferenceEquals($observedRecord.Exception,$cause)|Should -BeTrue
            [object]::ReferenceEquals($observedRecord.TargetObject,$script:LatePendingState)|Should -BeTrue
            $observedRecord.FullyQualifiedErrorId|Should -Match '^Injected.LateCustody'
            $script:DownstreamCount|Should -Be 0
            [object]::ReferenceEquals($registry.Pending,$pending)|Should -BeTrue
            [object]::ReferenceEquals($script:LatePendingState.Process,$process)|Should -BeTrue
            $pending.Count|Should -Be ($beforeCount+1)
            $pending.Contains($script:LatePendingState)|Should -BeTrue
            if($Transport -eq 'unsupported-worker'){Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly}
        } finally {
            Unregister-PcaiPerfPendingState $script:LatePendingState
            $process.Dispose()
        }
        $pending.Count|Should -Be $beforeCount
    }
    It 'propagates actual explicit-bundle binding rejection without a downstream scan' {
        $foreignDirectory=Join-Path $TestDrive 'foreign'
        [IO.Directory]::CreateDirectory($foreignDirectory)|Out-Null
        $script:ForeignTool=Join-Path $foreignDirectory 'pcai-perf.exe'
        [IO.File]::WriteAllText($script:ForeignTool,'No executable is launched')
        Mock Invoke-PcaiPerfCliCommand { Assert-PcaiPerfBundleBinding $script:ForeignTool }
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        $script:ObservedFailure|Should -BeOfType ([IO.InvalidDataException])
        $script:ObservedFailure.Message|Should -Match 'explicitly selected native bundle'
        $script:DownstreamCount|Should -Be 0
    }
    It 'propagates actual frame EOF without a downstream scan' {
        Mock Invoke-PcaiPerfCliCommand {
            $reader=[IO.StringReader]::new('')
            try {
                Read-PcaiPerfFrame ([pscustomobject]@{Output=$reader;Remaining=''}) ([Diagnostics.Stopwatch]::StartNew()) 10000 ([Threading.CancellationToken]::None)
            } finally { $reader.Dispose() }
        }
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        $script:ObservedFailure|Should -BeOfType ([IO.EndOfStreamException])
        $script:DownstreamCount|Should -Be 0
    }
    It 'propagates an actual malformed JSON frame instead of another backend scan' {
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        Mock Invoke-PcaiPerfWorkerRequest {
            $writer=[IO.StringWriter]::new()
            $reader=[IO.StringReader]::new(('not-json'+[char]10))
            try {
                Invoke-PcaiPerfFrame ([pscustomobject]@{Input=$writer;Output=$reader;Remaining=''}) @{command='disk';request_id='fixture';protocol=1} ([Diagnostics.Stopwatch]::StartNew()) 10000 ([Threading.CancellationToken]::None)
            } finally { $writer.Dispose();$reader.Dispose() }
        }
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        $script:ObservedFailure|Should -BeOfType ([ArgumentException])
        $script:ObservedFailure.InnerException.GetType().FullName|Should -Be 'Newtonsoft.Json.JsonReaderException'
        $script:DownstreamCount|Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
    It 'keeps a valid worker operation error eligible for the existing fallback' {
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        Mock Invoke-PcaiPerfWorkerRequest {
            $writer=[IO.StringWriter]::new()
            $reader=[IO.StringReader]::new(('{"protocol":1,"request_id":"fixture","ok":false,"error":"Private fixture metadata operation failed"}'+[char]10))
            try {
                Invoke-PcaiPerfFrame ([pscustomobject]@{Input=$writer;Output=$reader;Remaining=''}) @{command='disk';request_id='fixture';protocol=1} ([Diagnostics.Stopwatch]::StartNew()) 10000 ([Threading.CancellationToken]::None)
            } finally { $writer.Dispose();$reader.Dispose() }
        }
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $script:DownstreamCount|Should -Be 1
        $result[0].Path|Should -Be 'downstream-marker'
    }
    It 'keeps an ordinary direct operation error eligible for the existing fallback' {
        $script:InjectedFailure=[InvalidOperationException]::new('pcai-perf CLI failed with exit code 7.')
        $result=@(Get-DiskUsageFast -Path $TestDrive -Top 1)
        $script:DownstreamCount|Should -Be 1
        $result[0].Path|Should -Be 'downstream-marker'
    }
    It 'propagates worker <Validation> corruption instead of another backend scan' -ForEach @(
        @{Validation='request-id'}
        @{Validation='changed-protocol'}
        @{Validation='missing-status'}
        @{Validation='missing-result'}
    ) {
        param($Validation)
        $env:PCAI_PREFER_PERF_WORKER_DISK='1'
        $script:InjectedFailure=[IO.InvalidDataException]::new("Injected $Validation validation failure")
        try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
        [object]::ReferenceEquals($script:ObservedFailure,$script:InjectedFailure)|Should -BeTrue
        $script:DownstreamCount|Should -Be 0
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
    It 'rejects actual rooted pending custody before launch or fallback and retains its exact objects' {
        $registry=$script:PcaiPerfCustodyRegistry
        $pending=$registry.Pending
        $beforeCount=$pending.Count
        $process=[Diagnostics.Process]::new()
        $process.StartInfo=[Diagnostics.ProcessStartInfo]::new((Get-Command pwsh -ErrorAction Stop).Source)
        foreach($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 20')) {
            $process.StartInfo.ArgumentList.Add($argument)
        }
        $process.StartInfo.UseShellExecute=$false
        $process.StartInfo.CreateNoWindow=$true
        $gate=[Threading.SemaphoreSlim]::new(1,1)
        $state=[pscustomobject]@{Process=$process;OwnedPid=$null;OriginRegistry=$registry;Gate=$gate;ClosurePending=$true}
        try {
            if(-not $process.Start()){throw 'Pending-custody fixture child did not start'}
            $state.OwnedPid=$process.Id
            Register-PcaiPerfPendingState $state
            try { Get-DiskUsageFast -Path $TestDrive -Top 1|Out-Null } catch { $script:ObservedFailure=$_.Exception }
            $script:ObservedFailure.Message|Should -Match 'retained custody blocks replacement'
            $script:DownstreamCount|Should -Be 0
            Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
            Should -Invoke Invoke-PcaiPerfWorkerRequest -Times 0 -Exactly
            [object]::ReferenceEquals($script:PcaiPerfCustodyRegistry,$registry)|Should -BeTrue
            [object]::ReferenceEquals($registry.Pending,$pending)|Should -BeTrue
            [object]::ReferenceEquals($state.Process,$process)|Should -BeTrue
            [object]::ReferenceEquals($state.Gate,$gate)|Should -BeTrue
            $pending.Count|Should -Be ($beforeCount+1)
            $pending.Contains($state)|Should -BeTrue
            $process.HasExited|Should -BeFalse
        } finally {
            # Close only the fixture's exact owned handle; never look up a PID.
            if($state.OwnedPid -and -not $process.HasExited){$process.Kill()}
            if($state.OwnedPid -and -not $process.WaitForExit(1000)) {
                throw 'Fixture child closure was not confirmed; exact rooted state was preserved'
            }
            Unregister-PcaiPerfPendingState $state
            $process.Dispose()
            $gate.Dispose()
        }
        $pending.Count|Should -Be $beforeCount
    }
    It 'keeps ordinary no-CLI default selection as a downstream result' {
        $env:PCAI_PREFER_RUST_CLI=$null
        $result=@(Get-DiskUsageFast -Path $TestDrive)
        $result[0].Tool|Should -Be 'injected-parallel-marker'
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
    It 'does not select CLI with default Top0 even when Rust preference is enabled' {
        $result=@(Get-DiskUsageFast -Path $TestDrive)
        $result[0].Tool|Should -Be 'injected-parallel-marker'
        Should -Invoke Invoke-PcaiPerfCliCommand -Times 0 -Exactly
    }
}
