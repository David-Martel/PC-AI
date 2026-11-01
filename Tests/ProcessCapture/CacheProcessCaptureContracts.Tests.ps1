#Requires -Version 7.5
param([string]$CandidateRoot=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Common'),
    [string]$SourceManifestSHA256,[string]$PwshPath=(Join-Path $PSHOME 'pwsh.exe'),
    [string]$EvidenceDirectory,[string]$SourceManifestPath=(Join-Path $PSScriptRoot 'capture-source-register.json'))
BeforeAll {
    if('Pcai.Common.ProcessCaptureV1'-as[type]){throw 'Fresh capture-control child required'}
    . (Join-Path $PSScriptRoot 'Test-ProcessCaptureSources.ps1')
    $manifestPath=$SourceManifestPath
    if([string]::IsNullOrEmpty($SourceManifestSHA256)){$SourceManifestSHA256=(Get-FileHash -LiteralPath $manifestPath).Hash}
    $script:ProjectionManifest=Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot $manifestPath $SourceManifestSHA256
    $script:CaptureManifest=$script:ProjectionManifest
    $script:CaptureHash=@($script:CaptureManifest.Files|Where-Object{$_.Path-ceq'Private/PcaiProcessCapture.cs'})[0].SHA256
    if([string]::IsNullOrEmpty($EvidenceDirectory)-or-not[IO.Directory]::Exists($EvidenceDirectory)){throw 'Fresh enclosing entrypoint evidence directory required'}
    $script:SavedProvider=[Environment]::GetEnvironmentVariable('PCAI_CACHE_PROVIDER','Process')
    $env:PCAI_CACHE_PROVIDER='memory'
    . (Join-Path $CandidateRoot 'Public/Get-PcaiSharedCache.ps1')
    $script:ChildPath=Join-Path $PSScriptRoot 'Fixtures/CacheProcessChild.ps1'
    $script:Results=[Collections.Generic.List[object]]::new()
    function Invoke-ActualCapture([string]$Mode,[int]$Timeout=10000,[Threading.CancellationToken]$Token=[Threading.CancellationToken]::None,[string]$ReadyEventName=''){
        $fixtureArguments=@('-NoLogo','-NoProfile','-NonInteractive','-File',$script:ChildPath,'-Mode',$Mode);if($ReadyEventName){$fixtureArguments+=@('-ReadyEventName',$ReadyEventName)}
        $value=Invoke-PcaiProcessCapture -FilePath $PwshPath -Arguments $fixtureArguments -TimeoutMs $Timeout -CancellationToken $Token
        @($value).Count|Should -Be 1
        $script:Results.Add(@{Mode=$Mode;Success=$value.Success;ExitCode=$value.ExitCode;TimedOut=$value.TimedOut;Closure=$value.Closure;PrimaryFailure=$value.PrimaryFailure;SecondaryFailures=@($value.SecondaryFailures);StdoutBytes=$value.StdoutBytes;StderrBytes=$value.StderrBytes;FirstCleanup=$value.FirstCleanup;OriginalPid=$value.OriginalPid;OriginalExitCode=$value.OriginalExitCode;ParentExited=$value.ParentExited;JobEmpty=$value.JobEmpty;ReadersSettled=$value.ReadersSettled;HandlesClosed=$value.HandlesClosed;ElapsedMilliseconds=$value.ElapsedMilliseconds})
        return $value
    }
    function Assert-ActualClosure([object]$Value){
        $Value.Closure|Should -BeExactly 'CLOSED'
        foreach($field in @('AssignedBeforeResume','ParentExited','JobEmpty','ReadersSettled','HandlesClosed')){$Value.$field|Should -BeTrue}
        $Value.OriginalPid|Should -BeGreaterThan 0
        $Value.FirstCleanup|Should -Not -BeNullOrEmpty
        $Value.FirstCleanup.Closure|Should -BeExactly 'CLOSED'
        $Value.FirstCleanup.HandlesClosed|Should -BeTrue
        $Value.FirstCleanup.CleanupElapsedMilliseconds|Should -BeLessThan 1000
        $Value.FirstCleanup.SourceBinding|Should -BeExactly $script:CaptureHash
        [Pcai.Common.ProcessCaptureV1]::PendingCustody().Count|Should -Be 0
        [Pcai.Common.ProcessCaptureV1]::ActiveCustody()|Should -BeNullOrEmpty
    }
    $script:CancellationWaiterSource=@'
// Test-only original readiness owner. No native capture API, PID lookup or PowerShell callbacks.
namespace Pcai.Tests.CacheCancellationR1 {
 public sealed class Receipt {
  public bool ReadyObserved, CancellationRequested, ThreadClosed, ReadyEventClosed, StopEventClosed, CancellationClosed, OriginalResourcesClosed;
  public long ReadyElapsedMilliseconds, RepeatedFailures;
  public string FirstFailure, SourceBinding;
  public string[] SecondaryFailures;
 }
 public sealed class Waiter {
  public const int ContractVersion=1;
  public string SourceBinding { get; private set; }
  public System.Threading.CancellationTokenSource Cancellation { get; private set; }
  public System.Threading.EventWaitHandle ReadyEvent { get; private set; }
  public System.Threading.EventWaitHandle StopEvent { get; private set; }
  public Microsoft.Win32.SafeHandles.SafeWaitHandle ReadyHandle { get; private set; }
  public Microsoft.Win32.SafeHandles.SafeWaitHandle StopHandle { get; private set; }
  public Microsoft.Win32.SafeHandles.SafeWaitHandle CancellationHandle { get; private set; }
  public System.Threading.Thread OriginalThread { get; private set; }
  public System.Exception FirstFailure { get; private set; }
  public readonly System.Collections.Generic.List<System.Exception> SecondaryFailures=new System.Collections.Generic.List<System.Exception>();
  public long RepeatedFailures { get; private set; }
  public volatile bool ReadyObserved, CancellationRequested;
  public bool ThreadStarted, ThreadClosed, ReadyDisposalReturned, StopDisposalReturned, CancellationDisposalReturned, ReadyEventClosed, StopEventClosed, CancellationClosed;
  public long ReadyElapsedMilliseconds;
  readonly object failures=new object();
  public System.Threading.CancellationToken Token { get { return Cancellation.Token; } }
  public bool OriginalResourcesClosed { get { return ThreadClosed && ReadyEventClosed && StopEventClosed && CancellationClosed; } }
  void Fail(System.Exception error) {
   lock(failures) {
    if(object.ReferenceEquals(FirstFailure,error))return;
    foreach(var original in SecondaryFailures)if(object.ReferenceEquals(original,error))return;
    if(FirstFailure==null)FirstFailure=error;else if(SecondaryFailures.Count<64)SecondaryFailures.Add(error);else if(RepeatedFailures<long.MaxValue)RepeatedFailures++;
   }
  }
  void CancelOriginal(){try{Cancellation.Cancel();CancellationRequested=Cancellation.IsCancellationRequested;}catch(System.Exception error){Fail(error);}}
  public void Start(string eventName,int readinessMilliseconds,string sourceBinding) {
   if(ReadyEvent!=null||OriginalThread!=null||Cancellation!=null)throw new System.InvalidOperationException("Original waiter cannot be replaced");
   if(!System.Text.RegularExpressions.Regex.IsMatch(eventName,@"^Local\\PcaiCaptureCancellation_[a-f0-9]{32}$")||readinessMilliseconds<=0||readinessMilliseconds>10000||sourceBinding==null||sourceBinding.Length!=64)throw new System.ArgumentException("Bounded owned readiness admission required");
   SourceBinding=sourceBinding;
   try {
    bool created;ReadyEvent=new System.Threading.EventWaitHandle(false,System.Threading.EventResetMode.ManualReset,eventName,out created);ReadyHandle=ReadyEvent.SafeWaitHandle;
    if(!created)throw new System.InvalidOperationException("Owned readiness event collision");
    StopEvent=new System.Threading.EventWaitHandle(false,System.Threading.EventResetMode.ManualReset);StopHandle=StopEvent.SafeWaitHandle;
    Cancellation=new System.Threading.CancellationTokenSource();CancellationHandle=Cancellation.Token.WaitHandle.SafeWaitHandle;
    OriginalThread=new System.Threading.Thread(()=>{
     var clock=System.Diagnostics.Stopwatch.StartNew();
     try {
      int signal=System.Threading.WaitHandle.WaitAny(new System.Threading.WaitHandle[]{ReadyEvent,StopEvent},readinessMilliseconds);
      ReadyElapsedMilliseconds=clock.ElapsedMilliseconds;
      if(signal==0){ReadyObserved=true;CancelOriginal();}
      else {Fail(signal==System.Threading.WaitHandle.WaitTimeout ? (System.Exception)new System.TimeoutException("Owned child readiness was not observed within its finite startup budget") : new System.InvalidOperationException("Capture returned before owned child readiness"));CancelOriginal();}
     }catch(System.Exception error){Fail(error);CancelOriginal();}
     finally{clock.Stop();}
    }){IsBackground=false,Name="PC-AI original cancellation fixture waiter"};
    OriginalThread.Start();ThreadStarted=true;
   }catch(System.Exception error){Fail(error);throw;}
  }
  public void CloseOriginal() {
   // Keep the same foreground caller until the SAME managed thread and handles close.
   while(!ThreadClosed) {
    if(OriginalThread==null){ThreadClosed=true;break;}
    try {
     if(!ThreadStarted&&(OriginalThread.ThreadState&System.Threading.ThreadState.Unstarted)!=0){ThreadClosed=true;break;}
     if(OriginalThread.Join(0)){ThreadClosed=true;break;}
     if(StopEvent!=null)StopEvent.Set();
     if(OriginalThread.Join(1000)){ThreadClosed=true;break;}
    }catch(System.Exception error){Fail(error);}
    System.Threading.Thread.Sleep(1000);
   }
   while(!ReadyEventClosed||!StopEventClosed||!CancellationClosed) {
    if(ReadyEvent==null)ReadyEventClosed=true;
    else {if(ReadyHandle==null)try{ReadyHandle=ReadyEvent.SafeWaitHandle;}catch(System.Exception error){Fail(error);}if(!ReadyDisposalReturned)try{ReadyEvent.Dispose();ReadyDisposalReturned=true;}catch(System.Exception error){Fail(error);}try{ReadyEventClosed=ReadyDisposalReturned&&ReadyHandle!=null&&ReadyHandle.IsClosed;}catch(System.Exception error){Fail(error);}}
    if(StopEvent==null)StopEventClosed=true;
    else {if(StopHandle==null)try{StopHandle=StopEvent.SafeWaitHandle;}catch(System.Exception error){Fail(error);}if(!StopDisposalReturned)try{StopEvent.Dispose();StopDisposalReturned=true;}catch(System.Exception error){Fail(error);}try{StopEventClosed=StopDisposalReturned&&StopHandle!=null&&StopHandle.IsClosed;}catch(System.Exception error){Fail(error);}}
    if(Cancellation==null)CancellationClosed=true;
    else {if(CancellationHandle==null)try{CancellationHandle=Cancellation.Token.WaitHandle.SafeWaitHandle;}catch(System.Exception error){Fail(error);}if(!CancellationDisposalReturned)try{Cancellation.Dispose();CancellationDisposalReturned=true;}catch(System.Exception error){Fail(error);}try{CancellationClosed=CancellationDisposalReturned&&CancellationHandle!=null&&CancellationHandle.IsClosed;}catch(System.Exception error){Fail(error);}}
    if(!ReadyEventClosed||!StopEventClosed||!CancellationClosed)System.Threading.Thread.Sleep(1000);
   }
  }
  public Receipt Snapshot() {
   lock(failures){return new Receipt{SourceBinding=SourceBinding,ReadyObserved=ReadyObserved,CancellationRequested=CancellationRequested,ThreadClosed=ThreadClosed,ReadyEventClosed=ReadyEventClosed,StopEventClosed=StopEventClosed,CancellationClosed=CancellationClosed,OriginalResourcesClosed=OriginalResourcesClosed,ReadyElapsedMilliseconds=ReadyElapsedMilliseconds,RepeatedFailures=RepeatedFailures,FirstFailure=FirstFailure==null?null:FirstFailure.ToString(),SecondaryFailures=SecondaryFailures.ConvertAll(value=>value.ToString()).ToArray()};}
  }
 }
}
'@
    function Assert-RepeatedText([string]$Text,[int]$Count,[char]$Character){
        $Text.Length|Should -Be $Count
        # Hash complete actual text against independent known bytes, not a prefix.
        $expected=[byte[]]::new($Count);[Array]::Fill($expected,[byte]$Character)
        $actual=[Text.Encoding]::UTF8.GetBytes($Text)
        [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($actual))|Should -BeExactly ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($expected)))
    }
}

Describe 'Current-source proposed cache capture contracts' -Tag 'Unit','Cache','Windows','Capture' -Skip:(-not $IsWindows) {
    # Protects: loading current cache definitions and using memory cache need no compiler.
    # Detects: eager native initialization added to cache import or memory-only access.
    # Needs: fresh no-profile child before any actual capture, current full Public source.
    # Breadcrumb: current global cache initialization and memory-first public functions.
    It 'keeps definition load and local memory access free of capture initialization' {
        ('Pcai.Common.ProcessCaptureV1'-as[type]) | Should -BeNullOrEmpty
        Get-Command Initialize-PcaiProcessCaptureRunner -CommandType Function -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        Set-PcaiSharedCacheEntry -Namespace 'capture-owned' -Key 'public' -Value 'PUBLIC_ONLY'
        Get-PcaiSharedCacheEntry -Namespace 'capture-owned' -Key 'public' | Should -BeExactly 'PUBLIC_ONLY'
        ('Pcai.Common.ProcessCaptureV1'-as[type]) | Should -BeNullOrEmpty
    }
    # Protects: both redirected pipes drain while the real child exceeds pipe capacity.
    # Detects: predecessor wait-before-drain blockage, partial text or stream swaps.
    # Needs: actual2MiB A stdout and512KiB B stderr, exact frozen library and assigned outer job.
    # Breadcrumb: actual R4 Start/Reader/Buffer and real DualStream child.
    It 'captures complete dual-stream pressure' {
        $value=Invoke-ActualCapture DualStream
        $value.Success|Should -BeTrue;$value.ExitCode|Should -Be 0;$value.TimedOut|Should -BeFalse
        $value.PrimaryFailure|Should -BeNullOrEmpty;@($value.SecondaryFailures).Count|Should -Be 0
        $value.StdoutBytes|Should -Be 2097152;$value.StderrBytes|Should -Be 524288
        Assert-RepeatedText $value.StdOut 2097152 'A';Assert-RepeatedText $value.StdErr 524288 'B';Assert-ActualClosure $value
    }
    # Protects: actual nonzero process exit retains complete BOM-aware Unicode text and Trim.
    # Detects: nonzero code replaced, BOM retained, Unicode damaged or failed capture treated as empty.
    # Needs: actual Exit23 child with UTF8 BOM, spaces and CRLF on stdout and Unicode stderr.
    # Breadcrumb: R4 Buffer.Decode/Result and real Exit23 fixture.
    It 'preserves actual23 BOM Unicode and trimmed complete streams' {
        $value=Invoke-ActualCapture Exit23
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be 23;$value.OriginalExitCode|Should -Be 23;$value.TimedOut|Should -BeFalse
        $value.StdOut.Length|Should -Be 8;$value.StdErr.Length|Should -Be 8
        $value.StdOut|Should -BeExactly 'actual-π';$value.StdErr|Should -BeExactly 'actual-π'
        $value.PrimaryFailure|Should -BeNullOrEmpty;Assert-ActualClosure $value
    }
    # Protects: a successful empty capture is a valid result.
    # Detects: empty strings confused with missing or incomplete output.
    # Needs: actual Empty child with exit0 and both real EOFs.
    # Breadcrumb: R4 Result completeCapture/Reader.Eof and Empty fixture.
    It 'accepts successful empty complete output' {
        $value=Invoke-ActualCapture Empty
        $value.Success|Should -BeTrue;$value.ExitCode|Should -Be 0
        $value.StdOut.Length|Should -Be 0;$value.StdErr.Length|Should -Be 0
        $value.StdoutBytes|Should -Be 0;$value.StderrBytes|Should -Be 0;Assert-ActualClosure $value
    }
    # Protects: stdout accepts the entire8MiB approved boundary.
    # Detects: off-by-one cap rejection, truncated success or decode corruption.
    # Needs: actual StdoutExact producer and complete known A byte hash.
    # Breadcrumb: R4 Buffer.Append/StdoutLimit and real StdoutExact fixture.
    It 'accepts exact8MiB stdout' {
        $value=Invoke-ActualCapture StdoutExact 20000
        $value.Success|Should -BeTrue;$value.StdoutBytes|Should -Be 8388608;$value.StderrBytes|Should -Be 0
        Assert-RepeatedText $value.StdOut 8388608 'A';Assert-ActualClosure $value
    }
    # Protects: stderr independently accepts the entire1MiB approved boundary.
    # Detects: swapped stream limits or partial stderr success.
    # Needs: actual StderrExact producer and complete known B byte hash.
    # Breadcrumb: R4 Buffer.Append/StderrLimit and real StderrExact fixture.
    It 'accepts exact1MiB stderr' {
        $value=Invoke-ActualCapture StderrExact
        $value.Success|Should -BeTrue;$value.StderrBytes|Should -Be 1048576;$value.StdoutBytes|Should -Be 0
        Assert-RepeatedText $value.StdErr 1048576 'B';Assert-ActualClosure $value
    }
    # Protects: stdout overflow fails instead of returning an accepted prefix.
    # Detects: cap silently truncates, reports timeout or loses the real first cause.
    # Needs: real8MiB+1 producer and original parent/job/readers/handle closure.
    # Breadcrumb: R4 stdout Reader append failure and Result completeCapture gate.
    It 'fails stdout overflow without prefix success' {
        $value=Invoke-ActualCapture StdoutOver 20000
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -3;$value.TimedOut|Should -BeFalse
        $value.StdOut.Length|Should -Be 0;$value.PrimaryFailure|Should -Match 'stdout capture exceeded its finite byte limit'
        $value.StdoutBytes|Should -BeLessOrEqual 8388608;Assert-ActualClosure $value
    }
    # Protects: stderr overflow enforces its own finite limit.
    # Detects: stdout limit applied to stderr, partial capture promotion or cause loss.
    # Needs: real1MiB+1 stderr producer and exact original resource closure.
    # Breadcrumb: R4 stderr Reader append failure and Result failure preservation.
    It 'fails stderr overflow without prefix success' {
        $value=Invoke-ActualCapture StderrOver
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -3;$value.TimedOut|Should -BeFalse
        $value.StdOut.Length|Should -Be 0;$value.PrimaryFailure|Should -Match 'stderr capture exceeded its finite byte limit'
        $value.StderrBytes|Should -BeLessOrEqual 1048576;Assert-ActualClosure $value
    }
    # Protects: operation timeout retires only exact-owned stalled resources.
    # Detects: timeout becomes success, cancellation is mislabeled or closure remains unproved.
    # Needs: actual5sec Stall child with1000ms aggregate operation budget.
    # Breadcrumb: R4 Admit OperationDeadlineException and original Cleanup.
    It 'times out a real stalled child with exact closure' {
        $value=Invoke-ActualCapture Stall 1000
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -2;$value.TimedOut|Should -BeTrue
        $value.PrimaryFailure|Should -Match 'Capture operation deadline exceeded';Assert-ActualClosure $value
    }
    # Protects: zero deadline is rejected before Begin and child creation.
    # Detects: unbounded zero deadline, initializer invocation or real process admission.
    # Needs: real existing interpreter path, zero timeout and literal NOT_STARTED result.
    # Breadcrumb: actual R4 wrapper first TimeoutMs guard.
    It 'rejects zero deadline before a child' {
        $value=Invoke-PcaiProcessCapture -FilePath $PwshPath -Arguments @('-NoProfile','-File',$script:ChildPath,'-Mode','Stall') -TimeoutMs 0
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -3;$value.TimedOut|Should -BeFalse;$value.Closure|Should -BeExactly 'NOT_STARTED'
        $value.StdErr|Should -BeExactly 'TimeoutMs must be positive.'
        [Pcai.Common.ProcessCaptureV1]::PendingCustody().Count|Should -Be 0;[Pcai.Common.ProcessCaptureV1]::ActiveCustody()|Should -BeNullOrEmpty
    }
    # Protects: negative deadline fails through the same prelaunch guard.
    # Detects: negative interpreted as infinite or passed into native creation.
    # Needs: real interpreter path and negative timeout, no malformed-path shortcut.
    # Breadcrumb: actual R4 wrapper TimeoutMs -le0 guard.
    It 'rejects negative deadline before a child' {
        $value=Invoke-PcaiProcessCapture -FilePath $PwshPath -Arguments @('-NoProfile','-File',$script:ChildPath,'-Mode','Stall') -TimeoutMs -1
        $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -3;$value.Closure|Should -BeExactly 'NOT_STARTED'
        $value.StdErr|Should -BeExactly 'TimeoutMs must be positive.'
        [Pcai.Common.ProcessCaptureV1]::PendingCustody().Count|Should -Be 0;[Pcai.Common.ProcessCaptureV1]::ActiveCustody()|Should -BeNullOrEmpty
    }
    # Protects: a caller-owned cancellation token preserves cancellation separately from timeout.
    # Detects: ignored token, false TimedOut or cleanup of foreign identities.
    # Needs: real CancellationStall child signals owned readiness; retained managed waiter then cancels its original caller token within10sec budget.
    # Breadcrumb: actual R4 Begin linked token, Admit and owned cleanup; not StopProcessing proof.
    It 'cancels a real child through its caller token' {
        if('Pcai.Tests.CacheCancellationR1.Waiter'-as[type]){throw 'Fresh owned cancellation waiter type required'}
        $waiterHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.UTF8Encoding]::new($false,$true).GetBytes($script:CancellationWaiterSource)))
        Add-Type -TypeDefinition $script:CancellationWaiterSource -ErrorAction Stop
        if(Get-Variable PcaiCacheCancellationReadinessR1 -Scope Global -ErrorAction SilentlyContinue){throw 'Original cancellation waiter must not be replaced'}
        $global:PcaiCacheCancellationReadinessR1=@{Original=[Pcai.Tests.CacheCancellationR1.Waiter]::new();SourceBinding=$waiterHash;FirstErrorRecord=$null;CaptureRow=$null}
        $owned=$global:PcaiCacheCancellationReadinessR1;$waiter=$owned.Original
        try{
            $readyName='Local\PcaiCaptureCancellation_'+[Guid]::NewGuid().ToString('N')
            $waiter.Start($readyName,10000,$waiterHash)
            $value=Invoke-ActualCapture CancellationStall 10000 $waiter.Token $readyName
            $owned.CaptureRow=$script:Results[-1]
            $waiter.ReadyObserved|Should -BeTrue
            $value.Success|Should -BeFalse;$value.ExitCode|Should -Be -3;$value.TimedOut|Should -BeFalse
            $value.PrimaryFailure|Should -Match 'Capture was cancelled by its original caller';Assert-ActualClosure $value
        }catch{$owned.FirstErrorRecord=$_;throw}
        finally{
            $waiter.CloseOriginal()
            if($null-ne$owned.CaptureRow){$owned.CaptureRow.CancellationReadiness=$waiter.Snapshot()}
        }
        $waiter.CancellationRequested|Should -BeTrue
        $waiter.FirstFailure|Should -BeNullOrEmpty;$waiter.SecondaryFailures.Count|Should -Be 0;$waiter.RepeatedFailures|Should -Be 0
        $waiter.OriginalResourcesClosed|Should -BeTrue
    }
}

AfterAll {
    # Retain the original writer/error objects after Pester returns. This is not
    # a PID/path recovery registry; the same foreground child owns settlement.
    if(Get-Variable -Name PcaiCacheCaptureReceiptWriterR2 -Scope Global -ErrorAction SilentlyContinue){throw 'Fresh receipt-writer registry required'}
    $global:PcaiCacheCaptureReceiptWriterR2=@{
        State='PREPARING';Stream=$null;OriginalHandle=$null;Path=$null
        OpenAttempted=$false;OpenFailed=$false;DisposalReturned=$false
        OriginalHandleClosed=$false;Closed=$false;PostDisposalAttested=$false
        SerializedOwnClosure=$false;FirstErrorRecord=$null
        SecondaryErrorRecords=[Collections.Generic.List[object]]::new()
        RepeatedSecondaryFailures=0;UnreturnedOpenInternals='NOT_ATTESTED'
    }
    $writer=$global:PcaiCacheCaptureReceiptWriterR2
    function Add-CacheCaptureReceiptFailure([object]$Cause){
        if($null-eq$writer.FirstErrorRecord){$writer.FirstErrorRecord=$Cause}
        elseif($writer.SecondaryErrorRecords.Count-lt64){$writer.SecondaryErrorRecords.Add($Cause)}
        else{$writer.RepeatedSecondaryFailures++}
    }
    function Settle-CacheCaptureReceiptWriter {
        if($null-eq$writer.Stream){
            # Open throwing before returning is failure, not evidence about its
            # unreturned constructor internals. There is no returned resource.
            $writer.Closed=(-not$writer.OpenAttempted)-or$writer.OpenFailed
            return
        }
        if($null-eq$writer.OriginalHandle){
            try{$writer.OriginalHandle=$writer.Stream.SafeFileHandle}
            catch{Add-CacheCaptureReceiptFailure $_}
        }
        if(-not$writer.DisposalReturned){
            try{$writer.Stream.Dispose();$writer.DisposalReturned=$true}
            catch{Add-CacheCaptureReceiptFailure $_}
        }
        try{
            if($null-eq$writer.OriginalHandle){throw 'Actual original receipt SafeHandle unavailable; closure unconfirmed'}
            $writer.OriginalHandleClosed=$writer.OriginalHandle.IsClosed
            $writer.Closed=$writer.DisposalReturned-and$writer.OriginalHandleClosed
            $writer.PostDisposalAttested=$writer.Closed
        }catch{Add-CacheCaptureReceiptFailure $_;$writer.Closed=$false}
    }
    try{
        [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER',$script:SavedProvider,'Process')
        [void](Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot $manifestPath $SourceManifestSHA256)
        # Serialize self-closure as FALSE while this receipt has not settled.
        # The enclosing child must inspect the retained original registry after
        # Pester returns, before result/XML exports, and persist a separate
        # post-disposal attestation with IsClosed, raw first cause and secondaries.
        $payload=@{Rows=$script:Results.ToArray();ReceiptWriter=@{Registry='PcaiCacheCaptureReceiptWriterR2';Closed=$false;OriginalHandleClosed=$false;PostDisposalAttested=$false}}
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes((ConvertTo-Json -InputObject $payload -Depth 14))
        if($bytes.Length-gt1048576){throw 'Finite fixture receipt exceeds1MiB.'}
        $writer.Path=Join-Path $EvidenceDirectory 'capture-rows.json'
        $writer.OpenAttempted=$true
        try{$writer.Stream=[IO.File]::Open($writer.Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)}
        catch{$writer.OpenFailed=$true;throw}
        $writer.OriginalHandle=$writer.Stream.SafeFileHandle
        $writer.State='WRITING'
        $writer.Stream.Write($bytes,0,$bytes.Length);$writer.Stream.Flush($true)
    }catch{Add-CacheCaptureReceiptFailure $_}
    finally{
        Settle-CacheCaptureReceiptWriter
        while(-not$writer.Closed){
            $writer.State='HOLD'
            try{[Console]::Error.WriteLine('HOLD: original cache test receipt writer closure unresolved; same foreground child retained')}
            catch{Add-CacheCaptureReceiptFailure $_}
            [Threading.Thread]::Sleep(1000)
            Settle-CacheCaptureReceiptWriter
        }
        $writer.State=if($null-ne$writer.FirstErrorRecord){'CLOSED_FAILURE'}else{'CLOSED'}
    }
    if($null-ne$writer.FirstErrorRecord){throw $writer.FirstErrorRecord}
}
