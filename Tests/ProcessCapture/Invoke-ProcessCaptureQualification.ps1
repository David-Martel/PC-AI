#Requires -Version 7.6
<#
.SYNOPSIS
Qualify the maintained cache capture and full module in separate fresh children.
.DESCRIPTION
Default/help/DryRun perform no I/O. Apply (alias Run) requires explicit reviewed
source/register/peer pins, a complete compatible Pester installation, fresh
Windows pwsh -NoProfile parent and an absent owned output namespace. An alive
managed child or unresolved original resources keep this same parent foreground.
This entrypoint never enters ordinary Unit discovery or grants coverage credit.
#>
[CmdletBinding(PositionalBinding=$false)]
param([Alias('Run')][switch]$Apply,[switch]$DryRun,[Alias('h')][switch]$Help,
 [string]$SourceRegisterSHA256,[string]$SourcePeerPath,[string]$SourcePeerSHA256,
 [string]$PesterManifestPath,[string]$EvidenceDirectory,
 [ValidatePattern('^[A-Za-z][A-Za-z0-9_.-]{0,63}$')][string]$OutputId='cache-qualification-r1',
 [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments)
if($Help-or($null-ne$RemainingArguments-and$RemainingArguments-ccontains'--help')){'Usage: Invoke-ProcessCaptureQualification.ps1 (-Apply | -Run) -SourceRegisterSHA256 <reviewed hash> -SourcePeerPath <review.json> -SourcePeerSHA256 <hash> -PesterManifestPath <complete 5.7.1 manifest> [-EvidenceDirectory <absent owned directory>] [-OutputId <stable id>]';return}
if($DryRun-or-not$Apply){'DRY_RUN_NO_IO: explicit Apply and separately reviewed source/runtime admission required';return}
if($null-ne$RemainingArguments-and$RemainingArguments.Count){throw 'Unknown arguments'}
Set-StrictMode -Version Latest;$ErrorActionPreference='Stop'
if(-not$IsWindows-or-not[Environment]::Is64BitProcess-or[Environment]::GetCommandLineArgs()-inotcontains'-NoProfile'){throw 'Fresh Windows x64 pwsh -NoProfile parent required'}
if((Get-Module Pester)-or(Get-Module PC-AI.Common)-or('Pcai.Common.ProcessCaptureV1'-as[type])-or('PcaiCarbonQualificationR4.Supervisor'-as[type])){throw 'Fresh parent before modules/native supervisor required'}
foreach($scope in @('Global','Script','Local')){if(Get-Variable PesterPreference -Scope $scope -ErrorAction SilentlyContinue){throw 'No ambient Pester preference'} }
if(Get-Variable PcaiMaintainedCacheDefaultsOwnerR1 -Scope Global -ErrorAction SilentlyContinue){throw 'Retain original owner; no repeat or replacement'}
$global:PcaiMaintainedCacheDefaultsOwnerR1=@{State='PREPARING';FirstErrorRecord=$null;Secondary=[Collections.Generic.List[object]]::new();RepeatedSecondaryFailures=0;Writers=[Collections.Generic.List[object]]::new();Operations=[Collections.Generic.List[object]]::new();Children=[Collections.Generic.List[object]]::new();ChildInputs=[Collections.Generic.List[object]]::new();Identity=$null;IdentityHandle=$null;IdentityDisposalReturned=$false;IdentityClosed=$true;XmlReaders=[Collections.Generic.List[object]]::new();Clock=[Diagnostics.Stopwatch]::StartNew();SourceStable=$false;RuntimeStable=$false;Capacity=$null;SourceRegister=$null;Peer=$null;Runtime=$null;OriginalPID=$PID;OriginalBirthUtc=$null;SelfWriterClosed=$false;PriorSnapshots=@()}
$state=$global:PcaiMaintainedCacheDefaultsOwnerR1
function Add-DefaultsCause([object]$Cause){if($null-eq$Cause){return};if([object]::ReferenceEquals($state.FirstErrorRecord,$Cause)){return};foreach($original in $state.Secondary){if([object]::ReferenceEquals($original,$Cause)){return}};if($null-eq$state.FirstErrorRecord){$state.FirstErrorRecord=$Cause}elseif($state.Secondary.Count-lt64){$state.Secondary.Add($Cause)}else{$state.RepeatedSecondaryFailures++}}
function Assert-DefaultsOrdinary([string]$Path){$item=Get-Item -LiteralPath $Path -Force;for($depth=0;$null-ne$item;$depth++){if($depth-ge32-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Ordinary bounded ancestry required'};$item=if($item-is[IO.DirectoryInfo]){$item.Parent}else{$item.Directory}}}
function Read-DefaultsJson([string]$Path,[string]$SHA256,[ValidateRange(1,2097152)][int]$Cap=131072){if($SHA256-cnotmatch'^[A-F0-9]{64}$'){throw 'Explicit reviewed hash required'};Assert-DefaultsOrdinary $Path;$item=Get-Item -LiteralPath $Path -Force;if($item-isnot[IO.FileInfo]-or$item.Length-gt$Cap){throw 'Bounded ordinary JSON required'};$bytes=[IO.File]::ReadAllBytes($Path);if($bytes.Length-gt$Cap-or[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))-cne$SHA256){throw 'Same-byte reviewed JSON differs'};[Text.UTF8Encoding]::new($false,$true).GetString($bytes)|ConvertFrom-Json -AsHashtable -DateKind String}
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$registerPath=Join-Path $PSScriptRoot 'qualification-source-register.json'
$supervisor=$null;$commonLoaded=$false;$outputCreated=$false
try{
 $register=Read-DefaultsJson $registerPath $SourceRegisterSHA256;$state.SourceRegister=$register
 if($register.State-cne'MAINTAINED_DEFAULTS_SOURCE_BINDING_ONLY_NOT_RUNTIME_ACCEPTANCE'){throw 'Scoped maintained source register required'}
 $commonPath=Join-Path $PSScriptRoot 'qualification-common.ps1';Assert-DefaultsOrdinary $commonPath
 $commonPins=@($register.Files|Where-Object{$_.Path-ceq'Tests/ProcessCapture/qualification-common.ps1'})
 if($commonPins.Count-ne1-or(Get-Item -LiteralPath $commonPath).Length-ne$commonPins[0].Bytes-or(Get-FileHash -LiteralPath $commonPath).Hash-cne$commonPins[0].SHA256){throw 'Reviewed owner definitions differ'}
 . $commonPath;$commonLoaded=$true
 Assert-DefaultsSources $register $repositoryRoot
 $peer=Read-DefaultsJson $SourcePeerPath $SourcePeerSHA256;$state.Peer=$peer
 if($peer.State-cne'SOURCE_GO_FOR_SEPARATELY_ADMITTED_MAINTAINED_CACHE_DEFAULTS_ONLY'-or$peer.SourceRegisterSHA256-cne$SourceRegisterSHA256-or$peer.ModuleQaSourceRegisterSHA256-cne$register.ModuleQaSourceRegisterSHA256){throw 'Distinct source peer must bind both maintained source registers'}
 $state.Identity=[Diagnostics.Process]::GetCurrentProcess();$state.IdentityClosed=$false;$state.IdentityHandle=$state.Identity.SafeHandle
 $state.OriginalBirthUtc=$state.Identity.StartTime.ToUniversalTime().ToString('o')
 if($state.Identity.Id-ne$PID){throw 'Actual original parent identity required'}
 $runtime=Select-DefaultsRuntime $state.Identity $PesterManifestPath $register.PesterPayload;$state.Runtime=$runtime
 $state.Capacity=Get-DefaultsCapacity $repositoryRoot $EvidenceDirectory $OutputId
 Settle-DefaultsOriginals
 if(-not$state.IdentityClosed-or$state.FirstErrorRecord){throw 'Original identity closure before bootstrap required'}
 $controller=Join-Path $PSScriptRoot 'QualificationSupervisor.cs';$controllerHash=(Get-FileHash -LiteralPath $controller).Hash
 if($controllerHash-cne'019A7A1D97239C4F5F7D3C6C90925E9D74047F8E721693D9C832C4644276E227'){throw 'Exact accepted supervisor source required'}
 $controllerBytes=[IO.File]::ReadAllBytes($controller);if([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($controllerBytes))-cne$controllerHash){throw 'Same-byte bootstrap source differs'}
 Add-Type -TypeDefinition ([Text.UTF8Encoding]::new($false,$true).GetString($controllerBytes)) -ErrorAction Stop
 $supervisor='PcaiCarbonQualificationR4.Supervisor'-as[type]
 if($null-eq$supervisor-or$supervisor::ContractVersion-ne4){throw 'Exact supervisor contract required'}
 $supervisor::BindSource($controllerHash)
 if($supervisor::SourceBinding-cne$controllerHash-or@($supervisor::PendingCustody()).Count){throw 'Original supervisor binding/pending differs'}
 Assert-DefaultsSources $register $repositoryRoot;Assert-DefaultsRuntime $runtime $register.PesterPayload
 $EvidenceDirectory=$state.Capacity.OutputRoot
 if(Test-Path -LiteralPath $EvidenceDirectory){throw 'Fresh absent owned output ID required'}
 [void][IO.Directory]::CreateDirectory($EvidenceDirectory);$outputCreated=$true;Assert-DefaultsOrdinary $EvidenceDirectory
 foreach($mode in @('Eighteen','ModuleQA')){
  if($state.FirstErrorRecord-or$state.Secondary.Count-or$state.RepeatedSecondaryFailures){throw 'Sticky original failure refuses next child'}
  Wait-DefaultsOriginals $supervisor
  $capacity=Get-DefaultsCapacity $repositoryRoot $EvidenceDirectory $OutputId -ExistingRoot
  Assert-DefaultsSources $register $repositoryRoot;Assert-DefaultsRuntime $runtime $register.PesterPayload
  $now=[DateTime]::UtcNow;$childDirectory=Join-Path $EvidenceDirectory $(if($mode-ceq'Eighteen'){'eighteen'}else{'module-qa'})
  if(Test-Path -LiteralPath $childDirectory){throw 'Separate fresh child namespace required'}
  $inputPath=Join-Path $EvidenceDirectory $(if($mode-ceq'Eighteen'){'eighteen-admission.json'}else{'module-qa-admission.json'})
  $admission=@{State=$(if($mode-ceq'Eighteen'){'ROOT_ADMITTED_CANONICAL_CACHE_EIGHTEEN_ONLY'}else{'ROOT_ADMITTED_CANONICAL_CACHE_MODULE_QA_ONLY'});SourceReviewed=$true;SerializedOwnerConfirmed=$true;NoCompetingHeavyJobs=$true;QualifiedSupervisor4=$true;ManagedHandoffConfirmed=$true;PriorOriginalCustodyClosed=$true;HistoricalErrorsPreserved=$true;ObservedUtc=$now.ToString('o');ExpiresUtc=$now.AddSeconds(300).ToString('o');ModuleRoot=(Join-Path $repositoryRoot 'Modules/PC-AI.Common');EvidenceDirectory=$childDirectory;OwnedOutputRoot=$EvidenceDirectory;SourceRegisterSHA256=$register.EighteenSourceRegisterSHA256;SourceSeal=$register.ModuleQaSourceRegisterSHA256;PwshPath=$runtime.PwshPath;CoreClrPath=$runtime.CoreClrPath;PesterPath=$runtime.PesterManifestPath;PesterPins=$runtime.PesterPins;RuntimePins=$runtime.RuntimePins;SourcePeerPath=$SourcePeerPath;SourcePeerSHA256=$SourcePeerSHA256;ParentSourceSHA256=(Get-FileHash -LiteralPath $PSCommandPath).Hash;ProjectionSourceSeal='8E602A068F954F8D5702BC85482BAD5980690DEA6B8D7E49C7E81AC0B8D93B5E';ProjectionPeerSHA256='B435BE67E37640134C1FA898612894EBF261F841B47B1189CC1F1B1A7B2D7FB0';Capacity=$capacity;OriginalPID=$PID;OriginalBirthUtc=$state.OriginalBirthUtc;PriorOriginalLedgers=@()}
  if($mode-ceq'ModuleQA'){$admission.PriorOriginalLedgers=@(@{Name='PcaiMaintainedCacheDefaultsOwnerR1';Schema='CACHECCB';WriterCount=($state.Writers.Count+1)});$state.PriorSnapshots=@(Get-DefaultsLedgerSnapshot)}
  Write-DefaultsFresh $inputPath ([Text.UTF8Encoding]::new($false,$true).GetBytes(($admission|ConvertTo-Json -Depth 20)))
  Wait-DefaultsOriginals $supervisor
  if($state.FirstErrorRecord-or$state.Secondary.Count-or$state.RepeatedSecondaryFailures){throw 'Sticky original input writer failure refuses child launch'}
  if($mode-ceq'ModuleQA'){QA-Assert ($admission.PriorOriginalLedgers[0].WriterCount-eq$state.Writers.Count) 'Declaration matches actual complete ledger after input writer creation';[void](Get-DefaultsLedgerSnapshot)}
  # Input creation adds one owned writer; declare the actual current full ledger.
  # ModuleQA child does not consume parent objects; snapshots below remain local.
  $inputHash=(Get-FileHash -LiteralPath $inputPath).Hash
  $worker=if($mode-ceq'Eighteen'){Join-Path $PSScriptRoot 'Invoke-ProcessCaptureContracts.ps1'}else{Join-Path $PSScriptRoot 'ModuleQA/worker.ps1'}
  $arguments=@('-NoLogo','-NoProfile','-NonInteractive','-File',$worker,'-Apply','-AdmissionPath',$inputPath,'-AdmissionSHA256',$inputHash)
  if($mode-ceq'ModuleQA'){$arguments+=@('-PacketSHA256',$register.ModuleQaSourceRegisterSHA256);[void][IO.Directory]::CreateDirectory($childDirectory);Assert-DefaultsOrdinary $childDirectory}
  $handoff=Join-Path $EvidenceDirectory $(if($mode-ceq'Eighteen'){'eighteen-original.hold'}else{'module-qa-original.hold'})
  $operation=$supervisor::Run($runtime.PwshPath,$arguments,$repositoryRoot,(Join-Path $EvidenceDirectory "$mode-stdout.log"),(Join-Path $EvidenceDirectory "$mode-stderr.log"),180,2097152,$handoff)
  $state.Operations.Add($operation);Wait-DefaultsOriginals $supervisor
  QA-Closed $operation -AllowFailedExit
  $child=Read-DefaultsChild $mode $childDirectory (Join-Path $EvidenceDirectory "$mode-stdout.log") $inputHash $register $repositoryRoot
  $state.Children.Add($child);QA-Closed $operation
  [void](Read-DefaultsJson $inputPath $inputHash)
  Assert-DefaultsSources $register $repositoryRoot;Assert-DefaultsRuntime $runtime $register.PesterPayload
  Assert-DefaultsLedgerSnapshot $state.PriorSnapshots;$state.PriorSnapshots=@()
 }
 $state.SourceStable=$true;$state.RuntimeStable=$true;$state.State='PASS_SEPARATE_EIGHTEEN_AND_MODULE_QA_PENDING_INDEPENDENT_ACTUAL_REVIEW'
}catch{Add-DefaultsCause $_;$state.State='FAIL'}
finally{
 if($commonLoaded){
  Wait-DefaultsOriginals $supervisor
  try{if($null-ne$state.SourceRegister){[void](Read-DefaultsJson $registerPath $SourceRegisterSHA256);[void](Read-DefaultsJson $SourcePeerPath $SourcePeerSHA256);Assert-DefaultsSources $state.SourceRegister $repositoryRoot;$state.SourceStable=$true};if($null-ne$state.Runtime){Assert-DefaultsRuntime $state.Runtime $state.SourceRegister.PesterPayload;$state.RuntimeStable=$true};Assert-DefaultsLedgerSnapshot $state.PriorSnapshots}catch{Add-DefaultsCause $_;$state.State='FAIL'}
  if($outputCreated){try{Write-DefaultsFresh (Join-Path $EvidenceDirectory 'owner-result.json') ([Text.UTF8Encoding]::new($false,$true).GetBytes((@{StateAtSerialization=$state.State;SourceRegisterSHA256=$SourceRegisterSHA256;ModuleQaSourceRegisterSHA256=$state.SourceRegister.ModuleQaSourceRegisterSHA256;SourcePeerSHA256=$SourcePeerSHA256;OriginalPID=$PID;OriginalBirthUtc=$state.OriginalBirthUtc;Operations=$state.Operations.ToArray();Children=$state.Children.ToArray();SourceStable=$state.SourceStable;RuntimeStable=$state.RuntimeStable;FirstFailure=(QA-Cause $state.FirstErrorRecord);SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures;SelfWriterClosed=$false;ActualIndependentReview='REQUIRED';Limits='No coverage/performance/TRUE0/portable/liveRedis/crash/rareDispose/callerLoss claim'}|ConvertTo-Json -Depth 30)))}catch{Add-DefaultsCause $_;$state.State='FAIL'}}
  Wait-DefaultsOriginals $supervisor
 }elseif($null-ne$state.Identity){try{$state.Identity.Dispose();$state.IdentityDisposalReturned=$true;$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-DefaultsCause $_};while(-not$state.IdentityClosed){$state.State='HOLD';try{$state.Identity.Dispose();$state.IdentityDisposalReturned=$true;$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-DefaultsCause $_};[Threading.Thread]::Sleep(1000)}}
 $state.Clock.Stop();$state.SelfWriterClosed=@($state.Writers|Where-Object{-not$_.Closed}).Count-eq0
 if($state.FirstErrorRecord-or$state.Secondary.Count-or$state.RepeatedSecondaryFailures){$state.State='FAIL'}
}
try{[Console]::Out.WriteLine((@{FinalState=$state.State;ClockStopped=(-not$state.Clock.IsRunning);ElapsedMilliseconds=$state.Clock.ElapsedMilliseconds;OriginalIdentityClosed=$state.IdentityClosed;OriginalWritersClosed=$state.SelfWriterClosed;OriginalXmlReadersClosed=(@($state.XmlReaders|Where-Object{-not$_.ReaderClosed-or-not$_.StreamClosed}).Count-eq0);NativePending=$(if($null-ne$supervisor){@($supervisor::PendingCustody()).Count}else{0});FirstFailure=$(if($commonLoaded){QA-Cause $state.FirstErrorRecord}elseif($null-ne$state.FirstErrorRecord){$state.FirstErrorRecord.Exception.Message}else{$null});SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures}|ConvertTo-Json -Compress -Depth 6))}catch{Add-DefaultsCause $_;$state.State='FAIL'}
if($state.FirstErrorRecord){throw $state.FirstErrorRecord};if($state.Secondary.Count-or$state.RepeatedSecondaryFailures){throw 'Sticky original secondary failure'}
$state.State
