#Requires -Version 7.0

BeforeAll {
    if (-not ('PcaiWorkerCancellationFixture' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
public static class PcaiWorkerCancellationFixture {
 [System.Runtime.InteropServices.DllImport("kernel32.dll",CharSet=System.Runtime.InteropServices.CharSet.Unicode,SetLastError=true)]
 private static extern uint GetShortPathName(string input,System.Text.StringBuilder output,uint capacity);
 public static string ShortPath(string input) {
  var output=new System.Text.StringBuilder(32768);
  uint size=GetShortPathName(input,output,32768);
  if(size==0)throw new System.ComponentModel.Win32Exception();
  if(size>=32768)throw new System.IO.PathTooLongException();
  return output.ToString();
 }
 public static Task<bool> CancelAfterRequest(string marker,CancellationTokenSource source) {
  return Task.Run(() => {
   var timer=System.Diagnostics.Stopwatch.StartNew();
   while(timer.ElapsedMilliseconds<5000) {
    if(File.Exists(marker)){source.Cancel();return true;}
    Thread.Sleep(10);
   }
   source.Cancel();return false;
  });
 }
}
'@
    }
    $script:HelperPath = Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/Invoke-PcaiPerfWorker.ps1'
    $script:ResolverPath = Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/Get-PcaiPerfToolPath.ps1'
    $script:PwshPath = (Get-Command pwsh -ErrorAction Stop).Source
    $script:FakePath = Join-Path $TestDrive 'fake-perf-worker.ps1'
    @'
param([string]$Mode='good',[string]$PidFile)
if ($PidFile) { [IO.File]::WriteAllText($PidFile,[string]$PID) }
if ($Mode -like 'cli*') {
    [Console]::Out.WriteLine('{"value":"direct","Tool":"pcai-perf"}')
    if ($Mode -eq 'cli-hang') { Start-Sleep -Seconds 30 }
    if ($Mode -eq 'cli-error') { exit 7 }
    exit 0
}
while ($null -ne ($line=[Console]::In.ReadLine())) {
    $request=$line | ConvertFrom-Json
    if ($Mode -eq 'legacy') {
        [Console]::Out.WriteLine('{"ok":false,"error":"unsupported hello"}')
        continue
    }
    if ($request.command -eq 'hello') { $result=@{protocol=1} }
    else {
        if ($Mode -eq 'hang') { [IO.File]::WriteAllText($PidFile+'.request','received'); Start-Sleep -Seconds 30 }
        if ($Mode -eq 'partial') { [Console]::Out.Write('{"ok":true'); [Console]::Out.Flush(); Start-Sleep -Seconds 30; continue }
        if ($Mode -eq 'oversize') { [Console]::Out.Write([string]::new([char]'x',1100000)); [Console]::Out.Flush(); continue }
        if ($Mode -eq 'bad-json') { [Console]::Out.WriteLine('{broken'); continue }
        if ($request.delay_ms) { Start-Sleep -Milliseconds $request.delay_ms }
        $result=if($request.command -eq 'processes'){@(@{PID=$PID;Name='fixture';CPU=12.5;MemoryMB=8;Tool='pcai_rust'})}else{@{marker=$request.marker;owner_pid=$PID;Tool='pcai-perf'}}
    }
    $id=if($Mode -eq 'wrong-id' -and $request.command -ne 'hello'){'foreign-request'}else{$request.request_id}
    $response=@{ok=($Mode -ne 'error' -or $request.command -eq 'hello');result=$result;error='fixture failure';request_id=$id;protocol=1}
    if($request.command -ne 'hello' -and $Mode -eq 'wrong-protocol'){$response.protocol=2}
    if($request.command -ne 'hello' -and $Mode -eq 'missing-result'){$response.Remove('result')}
    [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress -Depth 6))
    [Console]::Out.Flush()
}
'@ | Set-Content -LiteralPath $script:FakePath -Encoding utf8NoBOM
    function New-WorkerArguments {
        param([string]$Mode)
        @('-NoLogo','-NoProfile','-File',$script:FakePath,'-Mode',$Mode,'-PidFile',$script:PidFile)
    }
}

Describe 'Bounded correlated pcai-perf worker' -Tag 'Unit','Portable','Acceleration' {
    BeforeEach {
        $script:OriginalBundle = $env:PCAI_NATIVE_BUNDLE_ROOT
        $env:PCAI_NATIVE_BUNDLE_ROOT = $null
        $script:PcaiPerfWorkerState = $null
        . $script:HelperPath
        . (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Public/Get-ProcessesFast.ps1')
        $script:PidFile = Join-Path $TestDrive 'worker.pid'
        if (Test-Path -LiteralPath $script:PidFile) { Remove-Item -LiteralPath $script:PidFile }
        if (Test-Path -LiteralPath ($script:PidFile+'.request')) { Remove-Item -LiteralPath ($script:PidFile+'.request') }
    }
    AfterEach {
        Stop-PcaiPerfWorker
        $env:PCAI_NATIVE_BUNDLE_ROOT = $script:OriginalBundle
    }

    It 'reuses one owned child and preserves useful response values across requests' {
        $workerArguments = New-WorkerArguments good
        $first = Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments $workerArguments -Command echo -Payload @{marker='first'}
        $second = Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments $workerArguments -Command echo -Payload @{marker='second'}
        $first.marker | Should -Be 'first'
        $second.marker | Should -Be 'second'
        $second.owner_pid | Should -Be $first.owner_pid
        $script:PcaiPerfWorkerState.OwnedPid | Should -Be $first.owner_pid
        $script:PcaiPerfWorkerState.ToolSha256 | Should -Be (Get-FileHash $script:PwshPath).Hash
        Stop-PcaiPerfWorker
        Get-Process -Id $first.owner_pid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }

    It 'labels real managed lifetime CPU without inventing a CPU percentage' {
        $rows=@(Get-ProcessesParallel -Name ([Diagnostics.Process]::GetCurrentProcess().ProcessName) -SortBy mem)
        $row=@($rows|Where-Object PID -eq $PID)[0]
        $row.CPUUnit|Should -Be 'seconds'
        $row.CPUPercent|Should -BeNullOrEmpty
        $row.TotalProcessorTimeSeconds|Should -BeGreaterOrEqual 0
        $row.CPU|Should -Be ([Math]::Round($row.TotalProcessorTimeSeconds,2))
    }

    It 'rejects a legacy uncorrelated worker and closes its owned process' {
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments legacy) -Command echo } | Should -Throw '*correlated protocol1*'
        $childPid = [int](Get-Content -LiteralPath $script:PidFile)
        Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
    }

    It 'fails closed on <Mode> rather than accepting a corrupt response' -TestCases @(
        @{Mode='wrong-id';Expected='*ID mismatch*'}
        @{Mode='wrong-protocol';Expected='*protocol changed*'}
        @{Mode='missing-result';Expected='*missing its result*'}
        @{Mode='bad-json';Expected='*'}
        @{Mode='error';Expected='*fixture failure*'}
        @{Mode='oversize';Expected='*character limit*'}
    ) {
        param($Mode,$Expected)
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments $Mode) -Command echo -Payload @{marker='expected'} } | Should -Throw $Expected
        $childPid = [int](Get-Content -LiteralPath $script:PidFile)
        Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }

    It 'times out a real partial frame, closes its PID, and leaves a foreign process alive' {
        $foreignInfo = [Diagnostics.ProcessStartInfo]::new($script:PwshPath)
        foreach ($arg in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 30')) { $foreignInfo.ArgumentList.Add($arg) }
        $foreignInfo.UseShellExecute=$false
        $foreignInfo.CreateNoWindow=$true
        $foreign = [Diagnostics.Process]::Start($foreignInfo)
        try {
            $workerArguments = New-WorkerArguments partial
            $null = Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments
            $childPid = $script:PcaiPerfWorkerState.OwnedPid
            $clock = [Diagnostics.Stopwatch]::StartNew()
            { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments $workerArguments -Command echo -TimeoutMilliseconds 150 } | Should -Throw '*deadline*'
            $clock.ElapsedMilliseconds | Should -BeLessThan 2500
            Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
            $foreign.HasExited | Should -BeFalse
        } finally {
            if (-not $foreign.HasExited) { $foreign.Kill(); $null=$foreign.WaitForExit(1000) }
            $foreign.Dispose()
        }
    }

    It 'honors cancellation while a real child is silent and releases serialization ownership' {
        $workerArguments = New-WorkerArguments hang
        $null = Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments
        $childPid = $script:PcaiPerfWorkerState.OwnedPid
        $witness = [Diagnostics.Process]::GetProcessById($childPid)
        $null = $witness.Handle
        $cancel = [Threading.CancellationTokenSource]::new()
        try {
            $signal = [PcaiWorkerCancellationFixture]::CancelAfterRequest($script:PidFile+'.request',$cancel)
            { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments $workerArguments -Command echo -CancellationToken $cancel.Token } | Should -Throw
            $signal.GetAwaiter().GetResult() | Should -BeTrue
            $witness.HasExited | Should -BeTrue
            $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
            $script:PcaiPerfWorkerState.Gate.CurrentCount | Should -Be 1
        } finally { $witness.Dispose();$cancel.Dispose() }
    }

    It 'rejects reserved or oversized requests before starting any child' {
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -Command echo -Payload @{request_id='forged'} } | Should -Throw '*Reserved*'
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -Command echo -Payload @{marker=('x' * 1100000)} } | Should -Throw '*byte limit*'
        Test-Path -LiteralPath $script:PidFile | Should -BeFalse
        $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
    }

    It 'checks a spent serialization/startup budget before starting any child' {
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good) -Command echo -Payload @{marker=('x'*900000)} -TimeoutMilliseconds 1 } | Should -Throw '*deadline*'
        Test-Path -LiteralPath $script:PidFile|Should -BeFalse
        $script:PcaiPerfWorkerState.Process|Should -BeNullOrEmpty
    }

    It 'rejects lossy depth truncation before starting any child' {
        $nested = @{leaf='useful-value'}
        for ($depth=0;$depth -lt 25;$depth++) { $nested = @{next=$nested} }
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good) -Command echo -Payload @{marker=$nested} } | Should -Throw
        Test-Path -LiteralPath $script:PidFile | Should -BeFalse
        $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
    }

    It 'sends the exact payload snapshot that was serialized and validated once' {
        $readCounter = [pscustomobject]@{Value=0}
        $marker = [pscustomobject]@{}
        $getter = { $readCounter.Value++; "snapshot-$($readCounter.Value)" }.GetNewClosure()
        $marker | Add-Member -MemberType ScriptProperty -Name value -Value $getter
        $result = Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good) -Command echo -Payload @{marker=$marker}
        $result.marker.value | Should -Be 'snapshot-1'
        $readCounter.Value | Should -Be 1
    }

    It 'bounds lock contention without interrupting the active owner' {
        $script:PcaiPerfWorkerState.Gate.Wait() | Out-Null
        try {
            { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -Command echo -TimeoutMilliseconds 100 } | Should -Throw '*busy*'
            $script:PcaiPerfWorkerState.Gate.CurrentCount | Should -Be 0
            $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
        } finally { $null=$script:PcaiPerfWorkerState.Gate.Release() }
    }

    It 'retains exact real process and pipe custody on <Failure> and blocks replacement' -TestCases @(
        @{Failure='kill'}
        @{Failure='wait'}
    ) {
        param($Failure)
        $workerArguments = New-WorkerArguments good
        $null = Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments
        $owned = $script:PcaiPerfWorkerState.Process
        $inputPipe = $script:PcaiPerfWorkerState.Input
        $outputPipe = $script:PcaiPerfWorkerState.Output
        $childPid = $owned.Id
        try {
            if ($Failure -eq 'kill') {
                $owned | Add-Member -MemberType ScriptMethod -Name Kill -Value { throw 'injected owned Kill failure' } -Force
            } else {
                $owned | Add-Member -MemberType ScriptMethod -Name Kill -Value { } -Force
                $owned | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { param($milliseconds) $false } -Force
            }
            $failureRecord = $null
            try { Close-PcaiPerfOwnedProcess $script:PcaiPerfWorkerState -OperationException ([IO.InvalidDataException]::new('corrupt frame fixture')) } catch { $failureRecord = $_ }
            $failureRecord | Should -Not -BeNullOrEmpty
            $failureRecord.Exception.Message | Should -BeLike '*corrupt frame fixture*exact process and pipe custody retained*'
            [object]::ReferenceEquals($script:PcaiPerfWorkerState.Process,$owned) | Should -BeTrue
            [object]::ReferenceEquals($script:PcaiPerfWorkerState.Input,$inputPipe) | Should -BeTrue
            [object]::ReferenceEquals($script:PcaiPerfWorkerState.Output,$outputPipe) | Should -BeTrue
            [object]::ReferenceEquals($failureRecord.Exception.Data['PcaiProcessCustody'],$script:PcaiPerfWorkerState) | Should -BeTrue
            $script:PcaiPerfWorkerState.OwnedPid | Should -Be $childPid
            $owned.HasExited | Should -BeFalse
            Test-PcaiPerfWorkerHealthy | Should -BeFalse
            { Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments } | Should -Throw '*retained custody blocks replacement*'
            $owned.HasExited | Should -BeFalse
        } finally {
            $owned.PSObject.Members.Remove('Kill')
            if ($Failure -eq 'wait') { $owned.PSObject.Members.Remove('WaitForExit') }
            Stop-PcaiPerfWorker
        }
        Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        $script:PcaiPerfPendingCustody.Count | Should -Be 0
    }

    It 'refuses a real <Route> launch while its registry monitor is held' -TestCases @(
        @{Route='worker'}
        @{Route='direct'}
        @{Route='profiler'}
    ) {
        param($Route)
        $origin=$script:PcaiPerfCustodyRegistry
        $held=[Threading.ManualResetEventSlim]::new($false)
        $release=[Threading.ManualResetEventSlim]::new($false)
        $holder=[PowerShell]::Create()
        $handle=$null
        try {
            $null=$holder.AddScript({param($gate,$held,$release)
                [Threading.Monitor]::Enter($gate)
                try{$held.Set();if(-not$release.Wait(5000)){throw 'Registry launch fixture guard expired'}}finally{[Threading.Monitor]::Exit($gate)}
            }).AddArgument($origin.Gate).AddArgument($held).AddArgument($release)
            $handle=$holder.BeginInvoke()
            $held.Wait(1000)|Should -BeTrue
            $failure=$null
            try {
                switch($Route){
                    'worker'{$null=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good)}
                    'direct'{$null=Invoke-PcaiPerfCliCommand -ToolPath $script:PwshPath -Arguments (New-WorkerArguments cli)}
                    'profiler'{$null=& (Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiProfiling.ps1') -Scenario custom -CommandPath $script:PwshPath -ArgumentList (New-WorkerArguments cli) -OutputRoot (Join-Path $TestDrive 'held-registry-profile') -PassThru}
                }
            }catch{$failure=$_.Exception}
            $failure.Message|Should -BeLike '*registry is busy*'
            Test-Path -LiteralPath $script:PidFile|Should -BeFalse
            $script:PcaiPerfWorkerState.Process|Should -BeNullOrEmpty
        } finally {
            $release.Set()
            if($handle){$null=$holder.EndInvoke($handle)}
            $holder.Dispose();$held.Dispose();$release.Dispose()
        }
        $origin.Pending.Count|Should -Be 0
    }

    It 'removes empty state after an actual <Route> process startup failure' -TestCases @(
        @{Route='worker'}
        @{Route='direct'}
        @{Route='profiler'}
    ) {
        param($Route)
        $invalidExecutable=Join-Path $TestDrive 'invalid-executable.exe'
        [IO.File]::WriteAllText($invalidExecutable,'Not an executable image')
        $failure=$null
        try {
            switch($Route){
                'worker'{$null=Start-PcaiPerfWorker -ToolPath $invalidExecutable}
                'direct'{$null=Invoke-PcaiPerfCliCommand -ToolPath $invalidExecutable -Arguments @('fixture')}
                'profiler'{$null=& (Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiProfiling.ps1') -Scenario custom -CommandPath $invalidExecutable -OutputRoot (Join-Path $TestDrive 'failed-start-profile') -PassThru}
            }
        }catch{$failure=$_.Exception}
        $failure|Should -Not -BeNullOrEmpty
        $script:PcaiPerfCustodyRegistry.Pending.Count|Should -Be 0
        $script:PcaiPerfWorkerState.Process|Should -BeNullOrEmpty
        Assert-PcaiPerfNoPendingCustody
    }

    It 'keeps unrelated active custody alive while independent script work succeeds and cleans its own state' {
        $state=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good)
        $origin=$state.OriginRegistry
        $owned=$state.Process
        $origin.Pending.Count|Should -Be 1
        [object]::ReferenceEquals($origin.Pending[0],$state)|Should -BeTrue
        Assert-PcaiPerfNoPendingCustody
        $independent=Join-Path $TestDrive 'independent-active-state.ps1'
        @'
param($Helper,$Tool,$Arguments)
. $Helper
Stop-PcaiPerfWorker
Invoke-PcaiPerfCliCommand -ToolPath $Tool -Arguments $Arguments
'@|Set-Content -LiteralPath $independent -Encoding utf8NoBOM
        $result=& $independent $script:HelperPath $script:PwshPath (New-WorkerArguments cli)
        $result.value|Should -Be 'direct'
        $owned.HasExited|Should -BeFalse
        $origin.Pending.Count|Should -Be 1
        Stop-PcaiPerfWorker
        $origin.Pending.Count|Should -Be 0
    }

    It 'keeps failed closure rooted when its registry monitor is held by another runspace' {
        $state=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good)
        $owned=$state.Process
        $origin=$state.OriginRegistry
        $owned|Add-Member -MemberType ScriptMethod -Name Kill -Value {throw 'registry contention Kill failure'} -Force
        $held=[Threading.ManualResetEventSlim]::new($false)
        $release=[Threading.ManualResetEventSlim]::new($false)
        $holder=[PowerShell]::Create()
        $handle=$null
        try {
            $null=$holder.AddScript({param($gate,$held,$release)
                [Threading.Monitor]::Enter($gate)
                try{$held.Set();if(-not$release.Wait(5000)){throw 'Registry holder fixture guard expired'}}
                finally{[Threading.Monitor]::Exit($gate)}
            }).AddArgument($origin.Gate).AddArgument($held).AddArgument($release)
            $handle=$holder.BeginInvoke()
            $held.Wait(1000)|Should -BeTrue
            $failure=$null
            try{Close-PcaiPerfOwnedProcess $state -OperationException ([IO.InvalidDataException]::new('registry contention original frame'))}catch{$failure=$_.Exception}
            $failure|Should -Not -BeNullOrEmpty
            $failure.Data['OperationException'].Message|Should -Be 'registry contention original frame'
            [object]::ReferenceEquals($failure.Data['PcaiProcessCustody'],$state)|Should -BeTrue
            [object]::ReferenceEquals($state.Process,$owned)|Should -BeTrue
            $owned.HasExited|Should -BeFalse
            $release.Set()
            $null=$holder.EndInvoke($handle)
            $handle=$null
            $holder.Streams.Error.Count|Should -Be 0
            $origin.Pending.Count|Should -Be 1
            [object]::ReferenceEquals($origin.Pending[0],$state)|Should -BeTrue
            {Assert-PcaiPerfNoPendingCustody}|Should -Throw '*retained custody blocks replacement*'
            $nextScript=Join-Path $TestDrive 'held-registry-next-invocation.ps1'
            @'
param($Helper,$Tool,$Arguments)
. $Helper
$failure=$null;$result=$null
try{$result=Invoke-PcaiPerfWorkerRequest -ToolPath $Tool -WorkerArguments $Arguments -Command echo}catch{$failure=$_.Exception}
finally{if($script:PcaiPerfWorkerState.Process){Stop-PcaiPerfWorker}}
[pscustomobject]@{Result=$result;Failure=$failure;OwnedPid=$script:PcaiPerfWorkerState.OwnedPid}
'@|Set-Content -LiteralPath $nextScript -Encoding utf8NoBOM
            $next=& $nextScript $script:HelperPath $script:PwshPath (New-WorkerArguments good)
            $next.Result|Should -BeNullOrEmpty
            $next.Failure.Message|Should -BeLike '*retained custody blocks replacement*'
            $next.OwnedPid|Should -BeNullOrEmpty
            $owned.HasExited|Should -BeFalse
        } finally {
            $release.Set()
            if($handle){$null=$holder.EndInvoke($handle)}
            $holder.Dispose()
            $held.Dispose()
            $release.Dispose()
            $owned.PSObject.Members.Remove('Kill')
            Stop-PcaiPerfWorker
        }
        $origin.Pending.Count|Should -Be 0
    }

    It 'retains original failure and exact custody when the cleanup gate is busy' {
        $state=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments (New-WorkerArguments good)
        $owned=$state.Process
        $inputPipe=$state.Input
        $originalFailure=[IO.InvalidDataException]::new('original frame failure before cleanup gate')
        $acquired=$state.Gate.Wait(1000)
        $acquired|Should -BeTrue
        try {
            $failure=$null
            try{Close-PcaiPerfOwnedProcess $state -OperationException $originalFailure -TimeoutMilliseconds 1}catch{$failure=$_.Exception}
            $failure|Should -Not -BeNullOrEmpty
            $failure.Message|Should -BeLike '*original frame failure*worker is busy*'
            [object]::ReferenceEquals($failure.Data['OperationException'],$originalFailure)|Should -BeTrue
            [object]::ReferenceEquals($failure.Data['PcaiProcessCustody'],$state)|Should -BeTrue
            [object]::ReferenceEquals($state.Process,$owned)|Should -BeTrue
            [object]::ReferenceEquals($state.Input,$inputPipe)|Should -BeTrue
            $owned.HasExited|Should -BeFalse
        } finally {if($acquired){$null=$state.Gate.Release()}}
    }

    It 'retains each failed Dispose object and retries exact custody after <FailedMember> fails' -TestCases @(
        @{FailedMember='Input'}
        @{FailedMember='Output'}
        @{FailedMember='Process'}
        @{FailedMember='Stderr'}
    ) {
        param($FailedMember)
        $workerArguments=New-WorkerArguments good
        $state=$script:PcaiPerfWorkerState
        if($FailedMember -eq 'Stderr') {
            $ownedInfo=[Diagnostics.ProcessStartInfo]::new($script:PwshPath)
            foreach($argument in $workerArguments){$ownedInfo.ArgumentList.Add($argument)}
            $ownedInfo.UseShellExecute=$false
            $ownedInfo.CreateNoWindow=$true
            $ownedInfo.RedirectStandardInput=$true
            $ownedInfo.RedirectStandardOutput=$true
            $ownedInfo.RedirectStandardError=$true
            $state.Process=[Diagnostics.Process]::Start($ownedInfo)
            $state.OwnedPid=$state.Process.Id
            $state.Input=$state.Process.StandardInput
            $state.Output=$state.Process.StandardOutput
            $state|Add-Member -NotePropertyName Stderr -NotePropertyValue $state.Process.StandardError
        } else { $null=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments }
        $originals=@{Process=$state.Process;Input=$state.Input;Output=$state.Output}
        if($state.PSObject.Properties['Stderr']){$originals.Stderr=$state.Stderr}
        $ownedPid=$state.OwnedPid
        $witness=[Diagnostics.Process]::GetProcessById($ownedPid)
        $null=$witness.Handle
        $foreignInfo=[Diagnostics.ProcessStartInfo]::new($script:PwshPath)
        foreach($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 30')){$foreignInfo.ArgumentList.Add($argument)}
        $foreignInfo.UseShellExecute=$false
        $foreignInfo.CreateNoWindow=$true
        $foreign=[Diagnostics.Process]::Start($foreignInfo)
        $operationFailure=[IO.InvalidDataException]::new('original failed frame')
        try {
            $originals[$FailedMember]|Add-Member -MemberType ScriptMethod -Name Dispose -Value {throw 'injected Dispose failure'} -Force
            $failureRecord=$null
            try {Close-PcaiPerfOwnedProcess $state -OperationException $operationFailure}catch{$failureRecord=$_}
            $failureRecord|Should -Not -BeNullOrEmpty
            $failureRecord.Exception.Message|Should -BeLike '*original failed frame*Dispose*'
            [object]::ReferenceEquals($failureRecord.Exception.Data['OperationException'],$operationFailure)|Should -BeTrue
            [object]::ReferenceEquals($failureRecord.Exception.Data['PcaiProcessCustody'],$state)|Should -BeTrue
            foreach($member in $originals.Keys){
                if($member -eq $FailedMember){[object]::ReferenceEquals($state.$member,$originals[$member])|Should -BeTrue}
                else{$state.$member|Should -BeNullOrEmpty}
            }
            $state.OwnedPid|Should -Be $ownedPid
            $state.ClosurePending|Should -BeTrue
            $script:PcaiPerfPendingCustody.Count|Should -Be 1
            $witness.HasExited|Should -BeTrue
            $foreign.HasExited|Should -BeFalse
            {Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments}|Should -Throw '*retained custody blocks replacement*'
            {Stop-PcaiPerfWorker}|Should -Throw '*original failed frame*Dispose*'
            [object]::ReferenceEquals($state.$FailedMember,$originals[$FailedMember])|Should -BeTrue
            $originals[$FailedMember].PSObject.Members.Remove('Dispose')
            Stop-PcaiPerfWorker
            $state.Process|Should -BeNullOrEmpty
            $state.Input|Should -BeNullOrEmpty
            $state.Output|Should -BeNullOrEmpty
            if($state.PSObject.Properties['Stderr']){$state.Stderr|Should -BeNullOrEmpty}
            $state.OwnedPid|Should -BeNullOrEmpty
            $script:PcaiPerfPendingCustody.Count|Should -Be 0
            $foreign.HasExited|Should -BeFalse
        } finally {
            $originals[$FailedMember].PSObject.Members.Remove('Dispose')
            Stop-PcaiPerfWorker
            # The detecting pre-repair run can lose this object from State.
            # Fixture ownership still retains it, so no live pipe is abandoned.
            foreach($original in $originals.Values){$original.Dispose()}
            $witness.Dispose()
            if(-not$foreign.HasExited){$foreign.Kill();$null=$foreign.WaitForExit(1000)}
            $foreign.Dispose()
        }
    }

    It 'retains failed closure across independent script invocations and blocks a second launch' {
        $firstScript=Join-Path $TestDrive 'first-independent-invocation.ps1'
        $secondScript=Join-Path $TestDrive 'second-independent-invocation.ps1'
        $freshScript=Join-Path $TestDrive 'fresh-process-invocation.ps1'
        @'
param($HelperPath,$ToolPath,$Arguments)
. $HelperPath
$null=Start-PcaiPerfWorker -ToolPath $ToolPath -WorkerArguments $Arguments
$state=$script:PcaiPerfWorkerState
$inputPipe=$state.Input
$ownedPid=$state.OwnedPid
$witness=[Diagnostics.Process]::GetProcessById($ownedPid)
$null=$witness.Handle
$inputPipe|Add-Member -MemberType ScriptMethod -Name Dispose -Value {throw 'independent invocation Dispose failure'} -Force
$failure=$null
try{Close-PcaiPerfOwnedProcess $state -OperationException ([IO.InvalidDataException]::new('first invocation frame failed'))}catch{$failure=$_.Exception}
[pscustomobject]@{State=$state;Input=$inputPipe;OwnedPid=$ownedPid;Witness=$witness;Failure=$failure;Registry=$script:PcaiPerfPendingCustody}
'@|Set-Content -LiteralPath $firstScript -Encoding utf8NoBOM
        @'
param($HelperPath,$ToolPath,$Arguments)
. $HelperPath
$result=$null;$failure=$null
try{$result=Invoke-PcaiPerfWorkerRequest -ToolPath $ToolPath -WorkerArguments $Arguments -Command echo}
catch{$failure=$_.Exception}
finally{if($script:PcaiPerfWorkerState.Process){Stop-PcaiPerfWorker}}
[pscustomobject]@{Result=$result;Failure=$failure;State=$script:PcaiPerfWorkerState;Registry=$script:PcaiPerfPendingCustody}
'@|Set-Content -LiteralPath $secondScript -Encoding utf8NoBOM
        @'
param($HelperPath,$ReceiptPath)
$ErrorActionPreference='Stop'
. $HelperPath
Assert-PcaiPerfNoPendingCustody
if($script:PcaiPerfPendingCustody.Count -ne 0){throw 'A foreign process registry was inherited'}
[IO.File]::WriteAllText($ReceiptPath,'fresh process admission passed')
'@|Set-Content -LiteralPath $freshScript -Encoding utf8NoBOM
        $workerArguments=New-WorkerArguments good
        $foreignInfo=[Diagnostics.ProcessStartInfo]::new($script:PwshPath)
        foreach($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 30')){$foreignInfo.ArgumentList.Add($argument)}
        $foreignInfo.UseShellExecute=$false
        $foreignInfo.CreateNoWindow=$true
        $foreign=[Diagnostics.Process]::Start($foreignInfo)
        $first=$null
        $freshProcess=$null
        try {
            $first=& $firstScript $script:HelperPath $script:PwshPath $workerArguments
            $first.Failure|Should -Not -BeNullOrEmpty
            [object]::ReferenceEquals($first.State.Input,$first.Input)|Should -BeTrue
            $first.Witness.HasExited|Should -BeTrue
            $freshReceipt=Join-Path $TestDrive 'fresh-process-admission.txt'
            $freshInfo=[Diagnostics.ProcessStartInfo]::new($script:PwshPath)
            foreach($argument in @('-NoLogo','-NoProfile','-File',$freshScript,$script:HelperPath,$freshReceipt)){$freshInfo.ArgumentList.Add($argument)}
            $freshInfo.UseShellExecute=$false
            $freshInfo.CreateNoWindow=$true
            $freshProcess=[Diagnostics.Process]::Start($freshInfo)
            $freshProcess.WaitForExit(10000)|Should -BeTrue
            $freshProcess.ExitCode|Should -Be 0
            Get-Content -LiteralPath $freshReceipt|Should -Be 'fresh process admission passed'
            $second=& $secondScript $script:HelperPath $script:PwshPath $workerArguments
            $second.Result|Should -BeNullOrEmpty
            $second.Failure.Message|Should -BeLike '*retained custody blocks replacement*'
            [object]::ReferenceEquals($first.Registry,$second.Registry)|Should -BeTrue
            [object]::ReferenceEquals($first.State.Input,$first.Input)|Should -BeTrue
            $foreign.HasExited|Should -BeFalse
            $first.Input.PSObject.Members.Remove('Dispose')
            Close-PcaiPerfOwnedProcess $first.State
            $first.Registry.Count|Should -Be 0
            $first.State.Input|Should -BeNullOrEmpty
            $first.State.OwnedPid|Should -BeNullOrEmpty
            $foreign.HasExited|Should -BeFalse
        } finally {
            if($freshProcess){
                if(-not$freshProcess.HasExited){$freshProcess.Kill();$freshProcess.WaitForExit(1000)|Should -BeTrue}
                $freshProcess.Dispose()
            }
            if($first){
                $first.Input.PSObject.Members.Remove('Dispose')
                Close-PcaiPerfOwnedProcess $first.State
                $first.Input.Dispose()
                $first.Witness.Dispose()
            }
            if(-not$foreign.HasExited){$foreign.Kill();$null=$foreign.WaitForExit(1000)}
            $foreign.Dispose()
        }
    }

    It 'preserves and rejects an unknown rooted custody <Kind>' -TestCases @(
        @{Kind='store-type'}
        @{Kind='version'}
        @{Kind='runspace'}
        @{Kind='pending-type'}
        @{Kind='gate-type'}
        @{Kind='missing-gate'}
    ) {
        param($Kind)
        $runspaceId=[System.Management.Automation.Runspaces.Runspace]::DefaultRunspace.InstanceId
        $key="PC_AI.PcaiPerfPendingCustody.v1:$($runspaceId.ToString('D'))"
        $original=[AppDomain]::CurrentDomain.GetData($key)
        $invalid=[pscustomobject]@{Version=1;RunspaceId=$runspaceId;Pending=[Collections.Generic.List[object]]::new();Gate=[object]::new()}
        $invalid.PSObject.TypeNames.Insert(0,'PC_AI.PcaiPerfPendingCustodyRegistry.v1')
        switch($Kind){
            'store-type'{$invalid=@{Version=1;RunspaceId=$runspaceId;Pending=[Collections.Generic.List[object]]::new()}}
            'version'{$invalid.Version=2}
            'runspace'{$invalid.RunspaceId=[guid]::NewGuid()}
            'pending-type'{$invalid.Pending=@()}
            'gate-type'{$invalid.Gate=[Threading.SemaphoreSlim]::new(1,1)}
            'missing-gate'{$invalid.PSObject.Properties.Remove('Gate')}
        }
        try {
            [AppDomain]::CurrentDomain.SetData($key,$invalid)
            { . $script:HelperPath }|Should -Throw '*rooted custody was preserved*'
            [object]::ReferenceEquals([AppDomain]::CurrentDomain.GetData($key),$invalid)|Should -BeTrue
            $script:PcaiPerfWorkerState.Process|Should -BeNullOrEmpty
        } finally { [AppDomain]::CurrentDomain.SetData($key,$original) }
    }

    It 'retries exact origin custody across foreign runspaces without duplicate or stale entries' {
        $workerArguments=New-WorkerArguments good
        $state=Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments
        $origin=$state.OriginRegistry
        $inputPipe=$state.Input
        $inputPipe|Add-Member -MemberType ScriptMethod -Name Dispose -Value {throw 'foreign-runspace disposal fixture'} -Force
        $foreignInfo=[Diagnostics.ProcessStartInfo]::new($script:PwshPath)
        foreach($argument in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 30')){$foreignInfo.ArgumentList.Add($argument)}
        $foreignInfo.UseShellExecute=$false
        $foreignInfo.CreateNoWindow=$true
        $foreign=[Diagnostics.Process]::Start($foreignInfo)
        $pipelines=@()
        try {
            {Close-PcaiPerfOwnedProcess $state -OperationException ([IO.InvalidDataException]::new('original corrupt frame'))}|Should -Throw '*exact failed pipe/handle custody retained*'
            $origin.Pending.Count|Should -Be 1
            $retry=[PowerShell]::Create()
            try {
                $null=$retry.AddScript({param($helper,$shared)
                    . $helper
                    $failure=$null
                    try{Close-PcaiPerfOwnedProcess $shared}catch{$failure=$_.Exception}
                    [pscustomobject]@{Failure=$failure;Count=$shared.OriginRegistry.Pending.Count;SameOrigin=[object]::ReferenceEquals((Get-PcaiPerfStateRegistry $shared),$shared.OriginRegistry)}
                }).AddArgument($script:HelperPath).AddArgument($state)
                $failedRetry=@($retry.Invoke())
                $retry.Streams.Error.Count|Should -Be 0
                $failedRetry.Count|Should -Be 1
                $failedRetry[0].Failure.Data['OperationException'].Message|Should -Be 'original corrupt frame'
                $failedRetry[0].SameOrigin|Should -BeTrue
                $failedRetry[0].Count|Should -Be 1
                [object]::ReferenceEquals($origin.Pending[0],$state)|Should -BeTrue
            } finally {$retry.Dispose()}
            foreach($marker in @('alpha','beta')) {
                $pipeline=[PowerShell]::Create()
                $null=$pipeline.AddScript({param($helper,$shared,$marker)
                    . $helper
                    if(-not$shared.Gate.Wait(1000)){throw 'Fixture worker gate timed out'}
                    try{Register-PcaiPerfPendingState $shared}finally{$null=$shared.Gate.Release()}
                    $marker
                }).AddArgument($script:HelperPath).AddArgument($state).AddArgument($marker)
                $pipelines+=@{Pipeline=$pipeline;Handle=$pipeline.BeginInvoke()}
            }
            $markers=foreach($item in $pipelines){$item.Pipeline.EndInvoke($item.Handle);$item.Pipeline.Streams.Error.Count|Should -Be 0}
            @($markers|Sort-Object)|Should -Be @('alpha','beta')
            $origin.Pending.Count|Should -Be 1
            $inputPipe.PSObject.Members.Remove('Dispose')
            $retry=[PowerShell]::Create()
            try {
                $null=$retry.AddScript({param($helper,$shared)
                    . $helper
                    Close-PcaiPerfOwnedProcess $shared
                    [pscustomobject]@{Count=$shared.OriginRegistry.Pending.Count;Input=$shared.Input;OwnedPid=$shared.OwnedPid}
                }).AddArgument($script:HelperPath).AddArgument($state)
                $successfulRetry=@($retry.Invoke())
                $retry.Streams.Error.Count|Should -Be 0
                $successfulRetry[0].Count|Should -Be 0
                $successfulRetry[0].Input|Should -BeNullOrEmpty
                $successfulRetry[0].OwnedPid|Should -BeNullOrEmpty
            } finally {$retry.Dispose()}
            Assert-PcaiPerfNoPendingCustody
            $result=Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -WorkerArguments $workerArguments -Command echo -Payload @{marker='after-exact-retry'}
            $result.marker|Should -Be 'after-exact-retry'
            $foreign.HasExited|Should -BeFalse
        } finally {
            foreach($item in $pipelines){$item.Pipeline.Dispose()}
            $inputPipe.PSObject.Members.Remove('Dispose')
            Stop-PcaiPerfWorker
            $inputPipe.Dispose()
            if(-not$foreign.HasExited){$foreign.Kill();$foreign.WaitForExit(1000)|Should -BeTrue}
            $foreign.Dispose()
        }
    }

    It 'serializes two real concurrent runspaces without crossing response IDs or payloads' {
        $workerArguments = New-WorkerArguments good
        $null = Start-PcaiPerfWorker -ToolPath $script:PwshPath -WorkerArguments $workerArguments
        $pipelines = @()
        try {
            foreach ($marker in @('alpha','beta')) {
                $pipeline=[PowerShell]::Create()
                $null=$pipeline.AddScript({param($helper,$state,$tool,$arguments,$marker)
                    $script:PcaiPerfWorkerState=$state
                    . $helper
                    Invoke-PcaiPerfWorkerRequest -ToolPath $tool -WorkerArguments $arguments -Command echo -Payload @{marker=$marker;delay_ms=100}
                }).AddArgument($script:HelperPath).AddArgument($script:PcaiPerfWorkerState).AddArgument($script:PwshPath).AddArgument($workerArguments).AddArgument($marker)
                $pipelines+=@{Pipeline=$pipeline;Handle=$pipeline.BeginInvoke()}
            }
            $results=foreach($item in $pipelines){$item.Pipeline.EndInvoke($item.Handle);$item.Pipeline.Streams.Error.Count | Should -Be 0}
            @($results.marker | Sort-Object) | Should -Be @('alpha','beta')
            @($results.owner_pid | Select-Object -Unique).Count | Should -Be 1
        } finally { foreach($item in $pipelines){$item.Pipeline.Dispose()} }
    }

    It 'retains legacy direct JSON results and rejects real nonzero child exits' {
        $result = Invoke-PcaiPerfCliCommand -ToolPath $script:PwshPath -Arguments (New-WorkerArguments cli)
        $result.value | Should -Be 'direct'
        { Invoke-PcaiPerfCliCommand -ToolPath $script:PwshPath -Arguments (New-WorkerArguments cli-error) } | Should -Throw '*exit code 7*'
    }

    It 'cancels a direct CLI after its JSON frame while it has not exited' {
        $cancel = [Threading.CancellationTokenSource]::new()
        try {
            $cancel.CancelAfter(3000)
            { Invoke-PcaiPerfCliCommand -ToolPath $script:PwshPath -Arguments (New-WorkerArguments cli-hang) -CancellationToken $cancel.Token } | Should -Throw
            $childPid=[int](Get-Content -LiteralPath $script:PidFile)
            Get-Process -Id $childPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        } finally { $cancel.Dispose() }
    }

    It 'rejects a wrong explicit bundle and does not choose a global fallback' {
        $bundle=Join-Path $TestDrive 'explicit-bundle'
        $null=New-Item -ItemType Directory -Path $bundle
        $env:PCAI_NATIVE_BUNDLE_ROOT=$bundle
        . $script:ResolverPath
        Get-PcaiPerfToolPath | Should -BeNullOrEmpty
        { Invoke-PcaiPerfWorkerRequest -ToolPath $script:PwshPath -Command echo } | Should -Throw '*explicitly selected native bundle*'
        $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
    }

    It 'admits actual Windows short directory and leaf spellings of the selected executable' -Skip:(-not $IsWindows) {
        $bundle=Join-Path $TestDrive 'explicit-bundle-with-long-name'
        $null=New-Item -ItemType Directory -Path $bundle
        $tool=Join-Path $bundle 'pcai-perf.exe'
        [IO.File]::WriteAllText($tool,'binding-only fixture; never executed')
        $shortBundle=[PcaiWorkerCancellationFixture]::ShortPath($bundle)
        $shortTool=[PcaiWorkerCancellationFixture]::ShortPath($tool)
        if ([string]::Equals($shortBundle,$bundle,[StringComparison]::OrdinalIgnoreCase) -or
            [string]::Equals([IO.Path]::GetFileName($shortTool),'pcai-perf.exe',[StringComparison]::OrdinalIgnoreCase)) {
            Set-ItResult -Skipped -Because 'This filesystem did not provide actual short directory and executable leaf aliases.'
            return
        }
        . $script:ResolverPath
        foreach($pair in @(@{Root=$shortBundle;Tool=$tool},@{Root=$bundle;Tool=$shortTool},@{Root=$shortBundle;Tool=$shortTool})) {
            $env:PCAI_NATIVE_BUNDLE_ROOT=$pair.Root
            Assert-PcaiPerfBundleBinding -ToolPath $pair.Tool | Should -Be ([IO.Path]::GetFullPath($tool))
            Get-PcaiPerfToolPath | Should -Be ([IO.Path]::GetFullPath($tool))
        }
    }

    It 'rejects an identically named executable from a different explicit bundle' {
        $bundle=Join-Path $TestDrive 'selected-bundle'
        $foreign=Join-Path $TestDrive 'foreign-bundle'
        $null=New-Item -ItemType Directory -Path $bundle,$foreign
        foreach($root in @($bundle,$foreign)){[IO.File]::WriteAllText((Join-Path $root 'pcai-perf.exe'),'same bytes and basename; distinct files')}
        $env:PCAI_NATIVE_BUNDLE_ROOT=$bundle
        { Assert-PcaiPerfBundleBinding -ToolPath (Join-Path $foreign 'pcai-perf.exe') } | Should -Throw '*explicitly selected native bundle*'
        $script:PcaiPerfWorkerState.Process | Should -BeNullOrEmpty
    }
}

Describe 'Actual paired native process consumer contracts' -Tag 'Integration','Acceleration','Native' {
    BeforeAll {
        $script:NativeBundle = $env:PCAI_TEST_NATIVE_BUNDLE_ROOT
        $script:SavedBundle = $env:PCAI_NATIVE_BUNDLE_ROOT
        $script:SavedDisableWorker = $env:PCAI_DISABLE_PERF_WORKER
        if ($script:NativeBundle) {
            $env:PCAI_NATIVE_BUNDLE_ROOT = $script:NativeBundle
            $env:PCAI_DISABLE_PERF_WORKER = $null
            $script:NativeModule = Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/PC-AI.Acceleration.psd1') -Force -PassThru
        }
    }
    AfterAll {
        if ($script:NativeModule) { & $script:NativeModule { Stop-PcaiPerfWorker } }
        $env:PCAI_NATIVE_BUNDLE_ROOT = $script:SavedBundle
        $env:PCAI_DISABLE_PERF_WORKER = $script:SavedDisableWorker
    }

    It 'uses a negotiated owned native worker and preserves direct CLI CPU units and sorting' -Skip:(-not $env:PCAI_TEST_NATIVE_BUNDLE_ROOT) {
        $expectedTool = [IO.Path]::GetFullPath((Join-Path $script:NativeBundle 'pcai-perf.exe'))
        $workerRows = @(& $script:NativeModule { param($tool) Get-ProcessesWithPcaiPerf -Top 15 -SortBy mem -ToolPath $tool } $expectedTool)
        $ownedPid = & $script:NativeModule { $script:PcaiPerfWorkerState.OwnedPid }
        $secondRows = @(& $script:NativeModule { param($tool) Get-ProcessesWithPcaiPerf -Top 15 -SortBy mem -ToolPath $tool } $expectedTool)
        (& $script:NativeModule { $script:PcaiPerfWorkerState.OwnedPid }) | Should -Be $ownedPid
        (& $script:NativeModule { $script:PcaiPerfWorkerState.ToolSha256 }) | Should -Be (Get-FileHash -LiteralPath $expectedTool).Hash
        $env:PCAI_DISABLE_PERF_WORKER = '1'
        $directRows = @(& $script:NativeModule { param($tool) Get-ProcessesWithPcaiPerf -Top 15 -SortBy mem -ToolPath $tool } $expectedTool)
        foreach ($rows in @($workerRows,$secondRows,$directRows)) {
            $rows.Count | Should -BeGreaterThan 0
            $rows.Count | Should -BeLessOrEqual 15
            foreach ($row in $rows) {
                $row.Tool | Should -Be 'pcai_rust'
                $row.CPUUnit | Should -Be 'percent'
                $row.CPUPercent | Should -Be $row.CPU
                $row.TotalProcessorTimeSeconds | Should -BeNullOrEmpty
                $row.CPU | Should -BeGreaterOrEqual 0
            }
            for ($index=1;$index -lt $rows.Count;$index++) { $rows[$index-1].MemoryMB | Should -BeGreaterOrEqual $rows[$index].MemoryMB }
        }
        $workerRows[0].Transport | Should -Be 'worker'
        $directRows[0].Transport | Should -Be 'direct-cli'
        & $script:NativeModule { Stop-PcaiPerfWorker }
        Get-Process -Id $ownedPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }

    It 'loads the explicitly paired bridge and preserves native ABI CPU unit labels' -Skip:(-not $env:PCAI_TEST_NATIVE_BUNDLE_ROOT) {
        (& $script:NativeModule { Initialize-PcaiNative -Force }) | Should -BeTrue
        $rows = @(& $script:NativeModule { Get-ProcessesWithNative -Top 15 -SortBy mem })
        $rows.Count | Should -BeGreaterThan 0
        foreach ($row in $rows) {
            $row.CPUUnit | Should -Be 'percent'
            $row.CPUPercent | Should -Be $row.CPU
            $row.TotalProcessorTimeSeconds | Should -BeNullOrEmpty
        }
        [IO.Path]::GetFullPath([PcaiNative.PcaiCore].Assembly.Location) | Should -Be ([IO.Path]::GetFullPath((Join-Path $script:NativeBundle 'PcaiNative.dll')))
    }

    It 'admits unknown system CPU in JSON while exposing raw ABI serialization limits' -Skip:(-not $env:PCAI_TEST_NATIVE_BUNDLE_ROOT) {
        (& $script:NativeModule { Initialize-PcaiNative -Force }) | Should -BeTrue
        # Rust emits JSON null for a nonfinite system statistic. The bridge's
        # row DTO deliberately omits that global statistic, preserving rows.
        $json='{"status":"success","sort_by":"memory","system_cpu_usage":null,"processes":[{"pid":1234,"name":"fixture","cpu_usage":1.25,"memory_bytes":1024}]}'
        $admitted=[System.Text.Json.JsonSerializer]::Deserialize[PcaiNative.TopProcessesResult]($json)
        $admitted.Processes.Count|Should -Be 1
        $admitted.Processes[0].Pid|Should -Be 1234
        $admitted.Processes[0].CpuUsage|Should -Be 1.25
        $raw=[PcaiNative.ProcessStats]::new()
        $raw.SystemCpuUsage=[single]::NaN
        [single]::IsNaN($raw.SystemCpuUsage)|Should -BeTrue
        $options=[System.Text.Json.JsonSerializerOptions]::new()
        $options.IncludeFields=$true
        # This is a declared raw-struct consumer limitation, not a successful
        # JSON export: callers must explicitly map unknown CPU to nullable DTOs.
        { [System.Text.Json.JsonSerializer]::Serialize[PcaiNative.ProcessStats]($raw,$options) }|Should -Throw
    }
}
