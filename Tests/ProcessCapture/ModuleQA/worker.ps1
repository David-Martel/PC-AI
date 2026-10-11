#Requires -Version 7.6
[CmdletBinding(PositionalBinding=$false)]
param([switch]$Apply,[switch]$DryRun,[Alias('h')][switch]$Help,[string]$PacketSHA256,[string]$AdmissionPath,[string]$AdmissionSHA256,[Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments)
if($Help-or($null-ne$RemainingArguments-and$RemainingArguments-ccontains'--help')){'worker.ps1 -Apply requires Root input and assigned-job readiness; strict full module, memory, cold/warm captures and unchanged cache tests. Default/DryRun performs no IO.';return}
if($DryRun-or-not$Apply){'NO_IO';return}
if($null-ne$RemainingArguments-and$RemainingArguments.Count){throw 'Unknown arguments'}
foreach($hash in @($PacketSHA256,$AdmissionSHA256)){if($hash-cnotmatch'^[A-F0-9]{64}$'){throw 'Exact admission/source hashes before IO'}}
if([Console]::ReadLine()-cne'PCAI_CARBON_HOOK_READY'){throw 'Assigned original Supervisor4 readiness before IO/import'}
Set-StrictMode -Version Latest;$ErrorActionPreference='Stop'
if(-not$IsWindows-or(Get-Module PC-AI.Common)-or(Get-Module Pester)-or('Pcai.Common.ProcessCaptureV1'-as[type])){throw 'Fresh Windows module/Pester/native type required'}
foreach($scope in @('Global','Script','Local')){if(Get-Variable PesterPreference -Scope $scope -ErrorAction SilentlyContinue){throw 'No ambient Pester preferences'}}
if(Get-Variable PcaiCanonicalCacheModuleQAChildR1 -Scope Global -ErrorAction SilentlyContinue){throw 'Fresh child globals required'}
$common=Join-Path $PSScriptRoot 'qa-common.ps1';if((Get-FileHash -LiteralPath $common).Hash-cne'8948DAD8AE948AA916E034823DF92855981F2E4C04D23608F67998E1711DBD2C'){throw 'Frozen QA definitions required'};. $common
$global:PcaiCanonicalCacheModuleQAChildR1=@{State='PREPARING';First=$null;Secondary=[Collections.Generic.List[object]]::new();RepeatedSecondaryFailures=0;Writers=[Collections.Generic.List[object]]::new();Identity=$null;IdentityHandle=$null;IdentityClosed=$true;CaptureWriter=$null;Xml=$null;XmlStream=$null;XmlReader=$null;XmlStreamDisposed=$false;XmlReaderDisposed=$false;Clock=[Diagnostics.Stopwatch]::StartNew();SourceStable=$false;RuntimeStable=$false;Pester=$null;Module=$null;Measurements=[Collections.Generic.List[object]]::new();Captures=[Collections.Generic.List[object]]::new();Rows=@();NUnitProjection=[Collections.Generic.List[object]]::new();NUnitNodes=[Collections.Generic.List[object]]::new();NUnitFlatTests=[Collections.Generic.List[object]]::new();NUnitSeenIndexes=[Collections.Generic.HashSet[int]]::new();NUnitIdentities=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal);ImportNoCaptureType=$false;MemoryNoCaptureType=$false;Packet=$null;Admission=$null}
$state=$global:PcaiCanonicalCacheModuleQAChildR1;$EvidenceDirectory=$null;$result=$null
function Measure-QA([string]$Purpose,[scriptblock]$Body){
    $before=[GC]::GetTotalAllocatedBytes($false);$clock=[Diagnostics.Stopwatch]::StartNew();$success=$false
    try{& $Body;$success=$true}finally{$clock.Stop();$state.Measurements.Add(@{Purpose=$Purpose;ElapsedMilliseconds=$clock.Elapsed.TotalMilliseconds;ManagedAllocatedBytes=([GC]::GetTotalAllocatedBytes($false)-$before);ManagedAllocationScope='PROCESS_CUMULATIVE_DELTA_INCLUDES_ASYNC_AND_OBSERVER_ALLOCATIONS_NOT_RSS';Completed=$success})}
}
function Invoke-QACapture([string]$Purpose){
    $argsForFixture=@('-NoLogo','-NoProfile','-NonInteractive','-File',(Join-Path $PSScriptRoot 'Fixtures/CaptureChild.ps1'),'-Emit')
    $capture=& $state.Module {param($executable,$fixtureArguments) Invoke-PcaiProcessCapture -FilePath $executable -Arguments $fixtureArguments -TimeoutMs 10000} $state.Admission.PwshPath $argsForFixture
    $state.Captures.Add(@{Purpose=$Purpose;Result=$capture;Arguments=$argsForFixture})
    QA-Assert ($null-ne$capture-and$capture.Success-is[bool]-and$capture.Success-and$capture.ExitCode-eq0-and-not$capture.TimedOut-and$capture.StdOut-ceq'PCAI_MODULE_CAPTURE_OK'-and$capture.StdErr-ceq'PCAI_MODULE_CAPTURE_ERR'-and$capture.StdoutBytes-eq22-and$capture.StderrBytes-eq23-and$capture.Closure-ceq'CLOSED'-and$capture.AssignedBeforeResume-and$capture.ParentExited-and$capture.JobEmpty-and$capture.ReadersSettled-and$capture.HandlesClosed-and$null-eq$capture.CustodyId-and$null-eq$capture.PrimaryFailure-and$capture.SecondaryFailures.Count-eq0-and$capture.RepeatedFailures-eq0) 'Real bounded capture semantic bytes and all ordinary native closure fields'
    $native='Pcai.Common.ProcessCaptureV1'-as[type];QA-Assert ($null-ne$native-and$native::ContractVersion-eq1-and$native::SourceBinding-ceq'14A046751C3EB14A2AB58A3DE785AC6D4B330F2036BAFB9CFC9E6908BEADCCCD'-and@($native::PendingCustody()).Count-eq0-and$null-eq$native::ActiveCustody()) 'Exact native binding and pending zero after each capture'
}
try{
    $state.Packet=QA-Packet $PacketSHA256;$state.Admission=QA-Admission $AdmissionPath $AdmissionSHA256 $PacketSHA256;$admission=$state.Admission;$packet=$state.Packet;$EvidenceDirectory=$admission.EvidenceDirectory
    Assert-ControlOrdinary $EvidenceDirectory
    $state.Identity=[Diagnostics.Process]::GetCurrentProcess();$state.IdentityClosed=$false;$state.IdentityHandle=$state.Identity.SafeHandle
    QA-Assert ([IO.Path]::GetFullPath($state.Identity.MainModule.FileName)-ieq[IO.Path]::GetFullPath($admission.PwshPath)) 'Actual selected worker interpreter'
    $loadedClr=@($state.Identity.Modules|Where-Object ModuleName -IEQ coreclr.dll|ForEach-Object FileName)
    QA-Assert ($loadedClr.Count-eq1-and[IO.Path]::GetFullPath($loadedClr[0])-ieq[IO.Path]::GetFullPath($admission.CoreClrPath)) 'Actual loaded worker CLR'
    Assert-ControlRuntime $admission.RuntimePins $admission.PesterPins $packet.PesterPayload $admission.PesterPath
    foreach($key in @('PwshPath','CoreClrPath','PesterPath')){
        QA-Assert ($admission[$key]-is[string]-and@($admission.RuntimePins|Where-Object{[IO.Path]::GetFullPath($_.Path)-ieq[IO.Path]::GetFullPath($admission[$key])}).Count-eq1) 'Selected runtime path must occur exactly once in physical hash inventory'
    }
    $state.Identity.Dispose();$state.IdentityClosed=$state.IdentityHandle.IsClosed;QA-Assert $state.IdentityClosed 'Original identity handle closed'
    $env:PCAI_CACHE_PROVIDER='memory'
    foreach($name in @('PCAI_REDIS_CLI_PATH','PCAI_REDIS_HOST','PCAI_REDIS_PORT','PCAI_REDIS_KEY_PREFIX','PCAI_REDIS_TIMEOUT_MS')){[Environment]::SetEnvironmentVariable($name,$null,'Process')}
    $temp=Join-Path $EvidenceDirectory 'temp';QA-Assert (-not(Test-Path -LiteralPath $temp)) 'Fresh temp';[void][IO.Directory]::CreateDirectory($temp);Assert-ControlOrdinary $temp;$env:TEMP=$temp;$env:TMP=$temp
    Measure-QA 'FullModuleImportStrictLatest' {
        $state.Module=Import-Module (Join-Path $admission.ModuleRoot 'PC-AI.Common.psm1') -PassThru -ErrorAction Stop
        QA-Assert ($null-ne$state.Module-and[IO.Path]::GetFullPath($state.Module.Path)-ieq[IO.Path]::GetFullPath((Join-Path $admission.ModuleRoot 'PC-AI.Common.psm1'))) 'Actual full module identity'
        $expectedExports=@($packet.ExportedFunctions);$actualExports=@($state.Module.ExportedFunctions.Keys)
        QA-Assert ($actualExports.Count-eq$expectedExports.Count) 'Complete exported function set';foreach($name in $expectedExports){QA-Assert ($actualExports-ccontains$name) 'Required exact export'}
        QA-Assert ($null-eq('Pcai.Common.ProcessCaptureV1'-as[type])) 'Import must not initialize native capture';$state.ImportNoCaptureType=$true
    }
    Measure-QA 'ExportedMemoryCacheConsumer' {
        & $state.Module.ExportedCommands['Clear-PcaiSharedCache'] -Namespace 'module-qa'
        $payload=@{Keys='literal';Nested=@{Value='public-memory-marker'}}
        & $state.Module.ExportedCommands['Set-PcaiSharedCacheEntry'] -Namespace 'module-qa' -Key 'entry' -Value $payload -DependencyStamp 'qa-stamp' -TtlSeconds 30|Out-Null
        $payload.Nested.Value='mutated-input';$hit=& $state.Module.ExportedCommands['Get-PcaiSharedCacheEntry'] -Namespace 'module-qa' -Key 'entry' -DependencyStamp 'qa-stamp' -TtlSeconds 30
        QA-Assert ($hit['Keys']-ceq'literal'-and$hit['Nested']['Value']-ceq'public-memory-marker') 'Real exported cache consumer clone'
        $miss=& $state.Module.ExportedCommands['Get-PcaiSharedCacheEntry'] -Namespace 'module-qa' -Key 'entry' -DependencyStamp 'different-stamp' -TtlSeconds 30
        QA-Assert ($null-eq$miss) 'Actual exported stamp miss';QA-Assert ($null-eq('Pcai.Common.ProcessCaptureV1'-as[type])) 'Memory use must not initialize native capture';$state.MemoryNoCaptureType=$true
    }
    Measure-QA 'ColdInternalCaptureIncludingNativeInitializer' {Invoke-QACapture 'cold'}
    for($warmIndex=1;$warmIndex-le5;$warmIndex++){Measure-QA "WarmInternalCapture$warmIndex" {Invoke-QACapture "warm$warmIndex"}}
    QA-Assert ($state.Captures.Count-eq6-and$state.Measurements.Count-eq8) 'One cold and five warm actual calls'
    Import-Module $admission.PesterPath -ErrorAction Stop
    QA-Assert ((Get-Module Pester).Version.ToString()-ceq'5.7.1') 'Selected complete Pester inventory'
    $testPath=Join-Path (Resolve-QARepositoryRoot) 'Tests/Unit/PcaiSharedCache.Tests.ps1';$configuration=New-PesterConfiguration;$configuration.Run.Path=@();$configuration.Run.Container=@((New-PesterContainer -Path $testPath -Data @{CacheSourcePath=(Join-Path $admission.ModuleRoot 'Public/Get-PcaiSharedCache.ps1');SkipRedisProbe=$true}))
    $configuration.Run.PassThru=$true;$configuration.Run.Exit=$false;$configuration.Filter.FullName=@();$configuration.Filter.Tag=@();$configuration.Filter.ExcludeTag=@();$configuration.TestResult.Enabled=$false;$configuration.CodeCoverage.Enabled=$false;$configuration.Output.Verbosity='Detailed'
    $result=Invoke-Pester -Configuration $configuration;$state.Pester=$result
    foreach($container in $result.Containers){foreach($cause in @($container.ErrorRecord)){if($null-ne$cause){Add-ControlCause $cause}}};foreach($block in $result.FailedBlocks){foreach($cause in @($block.ErrorRecord)){if($null-ne$cause){Add-ControlCause $cause}}}
    $expectedRows=@(foreach($row in $packet.MaintainedControls){QA-Assert ($row.SourcePath-ceq'Tests/Unit/PcaiSharedCache.Tests.ps1'-and$row.Name-is[string]-and$row.ExpandedName-is[string]-and$row.ExpandedPath-is[string]-and-not[string]::IsNullOrEmpty($row.ExpandedPath)-and$row.Result-cin@('Passed','Skipped')) 'Reviewed maintained identity and result';@{Name=$row.Name;ExpandedName=$row.ExpandedName;ExpandedPath=$row.ExpandedPath;Result=$row.Result;SourcePath=$testPath}})
    $expectedTotal=$expectedRows.Count;$expectedPassed=@($expectedRows|Where-Object Result -CEQ 'Passed').Count;$expectedSkipped=@($expectedRows|Where-Object Result -CEQ 'Skipped').Count
    $declaredSkip=@($expectedRows|Where-Object Result -CEQ 'Skipped');$expectedIdentities=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    QA-Assert ($expectedTotal-ge1-and$expectedTotal-le64-and$expectedPassed-eq($expectedTotal-1)-and$expectedSkipped-eq1-and$declaredSkip[0].Name-ceq'hydrates a local cache miss from Redis when the provider is enabled'-and$declaredSkip[0].ExpandedName-ceq$declaredSkip[0].Name) 'Only the exact reviewed live Redis case may skip'
    foreach($row in $expectedRows){QA-Assert ($expectedIdentities.Add($row.ExpandedPath)) 'Unique complete maintained case identity'}
    foreach($originalTest in $result.Tests){
        foreach($cause in @($originalTest.ErrorRecord)){
            if($null-eq$cause){continue}
            if($originalTest.Result-ceq'Skipped'-and$originalTest.Name-ceq'hydrates a local cache miss from Redis when the provider is enabled'-and$cause.FullyQualifiedErrorId-ceq'PesterTestSkipped'-and$cause.Exception.Message-ceq'is skipped, because Redis provider was not available or its live probe was disabled.'){continue}
            Add-ControlCause $cause
        }
    }
    QA-Assert ($result.TotalCount-eq$expectedTotal-and$result.PassedCount-eq$expectedPassed-and$result.FailedCount-eq0-and$result.SkippedCount-eq$expectedSkipped-and$result.NotRunCount-eq0) 'Genuine counts from exact reviewed maintained cases and sole declared Redis skip'
    $state.Rows=@($result.Tests|ForEach-Object {@{Name=$_.Name;ExpandedName=$_.ExpandedName;ExpandedPath=$_.ExpandedPath;Result=$_.Result;Source=$_.ScriptBlock.File}})
    QA-Assert ($result.Tests.Count-eq$expectedTotal) 'Complete original maintained test objects'
    for($i=0;$i-lt$expectedTotal;$i++){
        $test=$result.Tests[$i];$expected=$expectedRows[$i]
        QA-Assert ($test.Name-ceq$expected.Name-and$test.ExpandedName-ceq$expected.ExpandedName-and$test.ExpandedPath-ceq$expected.ExpandedPath-and$test.Result-ceq$expected.Result-and[IO.Path]::GetFullPath($test.ScriptBlock.File)-ieq[IO.Path]::GetFullPath($testPath)) 'Exact reviewed maintained case identity/result/source'
        $state.NUnitFlatTests.Add($test)
    }
    if($state.First){throw $state.First}
    $projectionGroupCommand=Get-Command Group-Object -Module Microsoft.PowerShell.Utility -CommandType Cmdlet
    foreach($container in $result.Containers){if($container.ShouldRun){Add-ControlNUnitProjection $container $container.Name 0}}
    QA-Assert ($state.NUnitProjection.Count-eq$expectedTotal-and$state.NUnitSeenIndexes.Count-eq$expectedTotal) 'Each real original test represented once'
    $xml=ConvertTo-NUnitReport -Result $result -AsString -Format NUnit2.5;$old='<?xml version="1.0" encoding="utf-16" standalone="no"?>';$new='<?xml version="1.0" encoding="utf-8" standalone="no"?>'
    QA-Assert ($xml-is[string]-and$xml.StartsWith($old,[StringComparison]::Ordinal)-and[regex]::Matches($xml,[regex]::Escape($old)).Count-eq1) 'Pinned sole exporter UTF16 prolog';$bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes($xml.Replace($old,$new));$xmlPath=Join-Path $EvidenceDirectory 'tests.xml';Write-ControlFresh $xmlPath $bytes
    QA-Assert ($null-eq$state.First-and$state.Secondary.Count-eq0-and@($state.Writers|Where-Object{-not$_.Closed}).Count-eq0) 'XML original writer closed before physical parse'
    $physical=[IO.File]::ReadAllBytes($xmlPath);QA-Assert ($physical.Length-le2097152-and$physical.Length-eq$bytes.Length-and[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($physical))-ceq[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))) 'Closed physical XML bytes/hash'
    try{
        $state.XmlStream=[IO.MemoryStream]::new($physical,$false);$settings=[Xml.XmlReaderSettings]::new();$settings.DtdProcessing=[Xml.DtdProcessing]::Prohibit;$settings.XmlResolver=$null;$settings.CloseInput=$false;$settings.MaxCharactersInDocument=2097152;$settings.MaxCharactersFromEntities=0;$state.XmlReader=[Xml.XmlReader]::Create($state.XmlStream,$settings);$document=[Xml.XmlDocument]::new();$document.XmlResolver=$null;$document.Load($state.XmlReader)
        foreach($pair in @(@('total',[string]$expectedTotal),@('errors','0'),@('failures','0'),@('not-run','0'),@('inconclusive','0'),@('ignored','0'),@('skipped',[string]$expectedSkipped),@('invalid','0'))){QA-Assert ($document.DocumentElement.GetAttribute($pair[0])-ceq$pair[1]) 'Actual physical NUnit counters'}
        $cases=$document.SelectNodes('//test-case');QA-Assert ($cases.Count-eq$expectedTotal) 'Complete physical XML cases'
        for($i=0;$i-lt$expectedTotal;$i++){$case=$cases[$i];$row=$state.NUnitProjection[$i];QA-Assert ($case.GetAttribute('name')-ceq$row.ExpandedPath-and$case.GetAttribute('description')-ceq$row.ExpandedName) 'Full original XML identity';$ancestor=$case.ParentNode;$matches=0;$depth=0;while($null-ne$ancestor){QA-Assert (++$depth-le32) 'Bounded XML ancestry';if($ancestor-is[Xml.XmlElement]-and$ancestor.LocalName-ceq'test-suite'-and[IO.Path]::IsPathRooted($ancestor.GetAttribute('name'))-and[IO.Path]::GetFullPath($ancestor.GetAttribute('name'))-ieq[IO.Path]::GetFullPath($testPath)){$matches++};$ancestor=$ancestor.ParentNode};QA-Assert ($matches-eq1) 'Actual source container';if($row.Result-ceq'Passed'){QA-Assert ($case.GetAttribute('result')-ceq'Success'-and$case.GetAttribute('executed')-ceq'True'-and$case.GetAttribute('success')-ceq'True') 'Actual successful XML case'}else{QA-Assert ($row.Result-ceq'Skipped'-and$case.GetAttribute('result')-ceq'Ignored'-and$case.GetAttribute('executed')-ceq'False'-and$case.GetAttribute('success')-ceq'False') 'Sole declared physical Redis skip'}}
        $state.Xml=@{Path=$xmlPath;Bytes=$physical.Length;SHA256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($physical));Total=$expectedTotal;Passed=$expectedPassed;Skipped=$expectedSkipped;Projection=$state.NUnitProjection.ToArray();PhysicalUTF8=$true;DTD='PROHIBITED';XmlWriterOriginalClosed=$true}
    }catch{Add-ControlCause $_}finally{Close-ControlXmlObjects}
    $pending=Get-ControlPending;QA-Assert (-not($pending.Native-or$pending.CaptureWriter-or$pending.OwnWriters-or$pending.Identity-or$pending.Xml)) 'All actual capture/writer/reader originals settled'
    $state.State='PASS_MODULE_SMOKE_SIX_CAPTURE_AND_EXACT_MAINTAINED_CONTROLS_ONLY'
}catch{Add-ControlCause $_;$state.State='FAIL'}
finally{
    Close-ControlXmlObjects
    if($null-ne$state.Identity-and-not$state.IdentityClosed){try{$state.Identity.Dispose();$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-ControlCause $_}}
    try{[void](QA-Packet $PacketSHA256);[void](QA-Read $AdmissionPath $AdmissionSHA256);$state.SourceStable=$true;if($null-ne$state.Admission){Assert-ControlRuntime $state.Admission.RuntimePins $state.Admission.PesterPins $state.Packet.PesterPayload $state.Admission.PesterPath;$state.RuntimeStable=$true}}catch{Add-ControlCause $_;$state.State='FAIL'}
    if($null-ne$EvidenceDirectory){try{Write-ControlFresh (Join-Path $EvidenceDirectory 'result.json') ([Text.UTF8Encoding]::new($false).GetBytes((@{StateAtSerialization=$state.State;FinalDispositionNotSerialized=$true;SourceSeal=$PacketSHA256;AdmissionSHA256=$AdmissionSHA256;ImportNoCaptureType=$state.ImportNoCaptureType;MemoryNoCaptureType=$state.MemoryNoCaptureType;Measurements=$state.Measurements.ToArray();Captures=$state.Captures.ToArray();Rows=$state.Rows;Total=$(if($result){$result.TotalCount}else{0});Passed=$(if($result){$result.PassedCount}else{0});Failed=$(if($result){$result.FailedCount}else{0});Skipped=$(if($result){$result.SkippedCount}else{0});NotRun=$(if($result){$result.NotRunCount}else{0});Xml=$state.Xml;FirstFailure=QA-Cause $state.First;SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures;SourceStable=$state.SourceStable;RuntimeStable=$state.RuntimeStable;SelfWriterClosed=$false;Timing='DESCRIPTIVE_SINGLE_WORKER_NO_MATCHED_BASELINE_OR_SPEEDUP';Allocation='MANAGED_PROCESS_DELTA_NOT_RSS';Limits='Module QA only; Windows TRUE0 defect NOT_REPRODUCED remains OPEN; portable deadline/Redis/crash/StopProcessing/caller-loss not accepted'}|ConvertTo-Json -Depth 20)))}catch{Add-ControlCause $_;$state.State='FAIL'}}
    $pending=Get-ControlPending
    while($pending.Native-or$pending.CaptureWriter-or$pending.OwnWriters-or$pending.Identity-or$pending.Xml){$state.State='HOLD';Close-ControlXmlObjects;foreach($writer in $state.Writers){if(-not$writer.Closed){Settle-ControlWriter $writer}};if(-not$state.IdentityClosed){try{$state.Identity.Dispose();$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-ControlCause $_}};[Threading.Thread]::Sleep(1000);$pending=Get-ControlPending}
    $state.Clock.Stop();$state.ElapsedMilliseconds=$state.Clock.ElapsedMilliseconds
    if($state.ElapsedMilliseconds-ge160000){Add-ControlCause ([TimeoutException]::new('QA worker cooperative deadline overrun'));$state.State='FAIL'}
    if($state.First-or$state.Secondary.Count-or$state.RepeatedSecondaryFailures){$state.State='FAIL'}
}
[Console]::Out.WriteLine((@{FinalState=$state.State;ElapsedMilliseconds=$state.ElapsedMilliseconds;ClockStopped=(-not$state.Clock.IsRunning);OriginalWritersClosed=(@($state.Writers|Where-Object{-not$_.Closed}).Count-eq0);IdentityClosed=$state.IdentityClosed;XmlReaderDisposed=$state.XmlReaderDisposed;XmlStreamDisposed=$state.XmlStreamDisposed;FirstFailure=QA-Cause $state.First;SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures}|ConvertTo-Json -Compress -Depth 5))
if($state.State-cne'PASS_MODULE_SMOKE_SIX_CAPTURE_AND_EXACT_MAINTAINED_CONTROLS_ONLY'){exit 1}
