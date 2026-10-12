#Requires -Version 7.5
<#
.SYNOPSIS
Run eighteen maintained capture controls in an already supervised fresh child.
.DESCRIPTION
Default, -h, --help and -DryRun perform no I/O. Apply requires separate Root
admission and the qualified parent's assigned-job handshake before any I/O.
This worker does not create a controller or grant its own execution authority.
#>
[CmdletBinding(PositionalBinding=$false)]
param([switch]$Apply,[switch]$DryRun,[Alias('h')][switch]$Help,
    [string]$CandidateRoot,[string]$EvidenceDirectory,
    [string]$AdmissionPath,[string]$AdmissionSHA256,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments)
if($Help-or($null-ne$RemainingArguments-and$RemainingArguments-ccontains'--help')){
    'Usage: Invoke-ProcessCaptureContracts.ps1 -Apply -AdmissionPath <Root input> -AdmissionSHA256 <SHA256>';return
}
if($DryRun-or-not$Apply){'DRY_RUN_NO_IO: eighteen capture controls require separate reviewed retained-owner execution';return}
if($null-ne$RemainingArguments-and$RemainingArguments.Count-gt0){throw 'Unknown arguments'}
if($AdmissionSHA256-cnotmatch'^[A-F0-9]{64}$'){throw 'Exact admission hash required before IO'}
if([Console]::ReadLine()-cne'PCAI_CARBON_HOOK_READY'){throw 'Original assigned-job readiness required before IO/import/compile'}
Set-StrictMode -Version Latest;$ErrorActionPreference='Stop'
if(-not$IsWindows){throw 'Dedicated native capture controls require Windows; portable fallback qualification is separate'}
if(Get-Variable PcaiMaintainedCaptureControlsR2 -Scope Global -ErrorAction SilentlyContinue){throw 'Fresh child registry required'}
if(('Pcai.Common.ProcessCaptureV1'-as[type])-or(Get-Module Pester -ErrorAction SilentlyContinue)){throw 'Fresh unloaded capture/Pester child required'}
foreach($scope in @('Global','Script','Local')){if(Get-Variable PesterPreference -Scope $scope -ErrorAction SilentlyContinue){throw 'No inherited Pester preference permitted'}}
$global:PcaiMaintainedCaptureControlsR2=@{State='PREPARING';First=$null;Secondary=[Collections.Generic.List[object]]::new();RepeatedSecondaryFailures=0;Writers=[Collections.Generic.List[object]]::new();Identity=$null;IdentityHandle=$null;IdentityClosed=$true;Pester=$null;CaptureWriter=$null;Rows=@();Xml=$null;OutputCreated=$false;SourceStable=$false;RuntimeStable=$false;Clock=[Diagnostics.Stopwatch]::StartNew()}
$state=$global:PcaiMaintainedCaptureControlsR2
$state.XmlStream=$null;$state.XmlReader=$null;$state.XmlStreamDisposed=$false;$state.XmlReaderDisposed=$false
$state.NUnitProjection=[Collections.Generic.List[object]]::new();$state.NUnitNodes=[Collections.Generic.List[object]]::new();$state.NUnitFlatTests=[Collections.Generic.List[object]]::new();$state.NUnitSeenIndexes=[Collections.Generic.HashSet[int]]::new();$state.NUnitIdentities=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
function Add-ControlCause([object]$Cause){if($null-eq$state.First){$state.First=$Cause}elseif($state.Secondary.Count-lt64){$state.Secondary.Add($Cause)}else{$state.RepeatedSecondaryFailures++}}
function Assert-ControlOrdinary([string]$Path){$item=Get-Item -LiteralPath $Path -Force;for($i=0;$null-ne$item;$i++){if($i-ge32-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Ordinary bounded ancestry required'};$item=if($item-is[IO.DirectoryInfo]){$item.Parent}else{$item.Directory}}}
function Assert-ControlRuntime($Pins,$PesterPins,$Payload,[string]$PesterPath){
    if($PesterPins-isnot[Array]-or$Payload-isnot[Collections.IDictionary]-or$Payload.Required-isnot[Array]-or$Payload.Required.Count-ne18-or$PesterPins.Count-lt$Payload.Required.Count-or$PesterPins.Count-gt($Payload.Required.Count+1)){throw 'Separately selected complete source-bound Pester inventory required'}
    if($Pins-isnot[Array]-or$Pins.Count-ne($PesterPins.Count+2)){throw 'Runtime closure must be exact selected Pester leaves plus interpreter and loaded CLR'}
    $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($pin in $Pins){if($pin.Path-isnot[string]-or$pin.SHA256-cnotmatch'^[A-F0-9]{64}$'-or-not$seen.Add([IO.Path]::GetFullPath($pin.Path))){throw 'Exact unique runtime pins required'};Assert-ControlOrdinary $pin.Path;$leaf=Get-Item -LiteralPath $pin.Path -Force;if($leaf-isnot[IO.FileInfo]-or$leaf.Length-ne$pin.Bytes-or(Get-FileHash -LiteralPath $pin.Path).Hash-cne$pin.SHA256){throw 'Runtime physical pin differs'}}
    $pesterRoot=[IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($PesterPath))
    $prefix=$pesterRoot.TrimEnd([char[]]@('/','\'))+[IO.Path]::DirectorySeparatorChar
    $selected=[Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach($pin in $PesterPins){
        if($pin.Path-isnot[string]-or$pin.SHA256-cnotmatch'^[A-F0-9]{64}$'){throw 'Typed selected Pester pin required'}
        $full=[IO.Path]::GetFullPath($pin.Path);if(-not$full.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Selected Pester leaf outside exact root'}
        $relative=[IO.Path]::GetRelativePath($pesterRoot,$full).Replace('\','/')
        if($relative.Split('/')-ccontains'..'-or$selected.ContainsKey($relative)){throw 'Traversal or duplicate selected Pester path'}
        $matched=@($Pins|Where-Object{[IO.Path]::GetFullPath($_.Path)-ieq$full})
        if($matched.Count-ne1-or$matched[0].Bytes-ne$pin.Bytes-or$matched[0].SHA256-cne$pin.SHA256){throw 'Selected Pester row not exactly in runtime inventory'}
        $selected.Add($relative,$pin)
    }
    foreach($required in $Payload.Required){
        if($required.Path-isnot[string]-or-not$selected.ContainsKey($required.Path)){throw 'Required complete Pester source payload missing'}
        $pin=$selected[$required.Path];if($pin.Bytes-ne$required.Bytes-or$pin.SHA256-cne$required.SHA256){throw 'Selected Pester source bytes differ from reviewed payload'}
    }
    foreach($relative in $selected.PSBase.Keys){
        if(@($Payload.Required|Where-Object{$_.Path-ceq$relative}).Count-eq1){continue}
        if($relative-cne'PSGetModuleInfo.xml'-or$Payload.OptionalMetadata-cne'PSGetModuleInfo.xml'){throw 'Unexpected extra Pester payload leaf'}
        # Installer metadata is admitted only when actually present and explicitly
        # pinned in both selected and physical runtime inventories; never invented.
    }
    $queue=[Collections.Generic.Queue[string]]::new();$queue.Enqueue($pesterRoot);$physical=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal);$directoryCount=0
    while($queue.Count){$directory=$queue.Dequeue();Assert-ControlOrdinary $directory;if(++$directoryCount-gt32){throw 'Pester directory cap'};foreach($item in Get-ChildItem -LiteralPath $directory -Force){Assert-ControlOrdinary $item.FullName;if($item-is[IO.DirectoryInfo]){$queue.Enqueue($item.FullName)}elseif($item-is[IO.FileInfo]){$relative=[IO.Path]::GetRelativePath($pesterRoot,$item.FullName).Replace('\','/');if(-not$physical.Add($relative)-or$physical.Count-gt($Payload.Required.Count+1)){throw 'Pester physical leaf cap/duplicate'}}else{throw 'Unsupported Pester item'}}}
    if($physical.Count-ne$selected.PSBase.Count){throw 'Selected Pester complete physical set count differs'}
    foreach($relative in $physical){if(-not$selected.ContainsKey($relative)){throw 'Physical Pester file omitted from selected inventory'}}
    foreach($relative in $selected.PSBase.Keys){if(-not$physical.Contains($relative)){throw 'Selected Pester leaf absent from physical inventory'}}
}
function Settle-ControlWriter($Entry){
    if($null-eq$Entry.Stream){$Entry.Closed=(-not$Entry.OpenAttempted)-or$Entry.OpenFailed;return}
    if($null-eq$Entry.Handle){try{$Entry.Handle=$Entry.Stream.SafeFileHandle}catch{Add-ControlCause $_}}
    if(-not$Entry.DisposalReturned){try{$Entry.Stream.Dispose();$Entry.DisposalReturned=$true}catch{Add-ControlCause $_}}
    try{$Entry.Closed=$Entry.DisposalReturned-and$null-ne$Entry.Handle-and$Entry.Handle.IsClosed}catch{Add-ControlCause $_;$Entry.Closed=$false}
}
function Write-ControlFresh([string]$Path,[byte[]]$Bytes){
    if($Bytes.Length-gt2097152){throw 'Per-receipt two MiB cap'}
    $full=[IO.Path]::GetFullPath($Path);$prefix=[IO.Path]::GetFullPath($EvidenceDirectory).TrimEnd([char[]]@('/','\'))+[IO.Path]::DirectorySeparatorChar
    if(-not$full.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)-or[IO.Path]::GetDirectoryName($full)-ine[IO.Path]::GetFullPath($EvidenceDirectory)){throw 'Only direct owned output leaves permitted'}
    $entry=@{Path=$full;Stream=$null;Handle=$null;OpenAttempted=$false;OpenFailed=$false;DisposalReturned=$false;Closed=$false};$state.Writers.Add($entry)
    try{$entry.OpenAttempted=$true;try{$entry.Stream=[IO.File]::Open($full,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)}catch{$entry.OpenFailed=$true;throw};$entry.Handle=$entry.Stream.SafeFileHandle;$entry.Stream.Write($Bytes,0,$Bytes.Length);$entry.Stream.Flush($true)}catch{Add-ControlCause $_}finally{Settle-ControlWriter $entry}
}
function Add-ControlNUnitProjection($Node,[string]$ContainerPath,[int]$Depth){
    if($Depth-gt32-or$state.NUnitNodes.Count-ge128){throw 'Bounded original Pester tree required'}
    if($Depth-eq0){if($Node-isnot[Pester.Container]){throw 'Original typed Pester container required'}}elseif($Node-isnot[Pester.Block]){throw 'Original typed Pester block required'}
    foreach($seenNode in $state.NUnitNodes){if([object]::ReferenceEquals($seenNode,$Node)){throw 'Repeated original Pester tree node'}}
    $state.NUnitNodes.Add($Node)
    foreach($block in $Node.Blocks){
        if($block.ShouldRun-isnot[bool]){throw 'Typed original block ShouldRun required'}
        if($block.ShouldRun){Add-ControlNUnitProjection $block $ContainerPath ($Depth+1)}
    }
    # Pester.Container contains Blocks only; tests belong to typed Block nodes.
    # The pinned exporter treats its absent container.Tests as null in loose mode.
    if($Depth-eq0){return}
    $groups=@($Node.Tests|& $projectionGroupCommand -Property GroupId)
    foreach($group in $groups){
        $parameterizedName=$null;if($group.Name){$parameterizedName=$group.Group[0].Path-join'.'}
        foreach($originalTest in $group.Group){
            if($originalTest.ShouldRun-isnot[bool]){throw 'Typed original test ShouldRun required'}
            if(-not$originalTest.ShouldRun){continue}
            $matches=[Collections.Generic.List[int]]::new()
            for($flatIndex=0;$flatIndex-lt$state.NUnitFlatTests.Count;$flatIndex++){if([object]::ReferenceEquals($originalTest,$state.NUnitFlatTests[$flatIndex])){$matches.Add($flatIndex)}}
            if($matches.Count-ne1-or-not$state.NUnitSeenIndexes.Add($matches[0])){throw 'Missing/repeated/nonoriginal NUnit test identity'}
            $flatIndex=$matches[0];$row=$state.Rows[$flatIndex];$declared=$expectedRows[$flatIndex]
            foreach($field in @('Name','ExpandedName','ExpandedPath','Result')){if($originalTest.$field-isnot[string]-or$originalTest.$field.Length-lt1-or$originalTest.$field.Length-gt4096-or$originalTest.$field-cne$row[$field]){throw 'Original projected test scalar identity differs'}}
            foreach($field in @('Name','ExpandedName','Result')){if($row[$field]-cne$declared[$field]){throw 'Frozen flat source-order selector/result differs'}}
            if($originalTest.ScriptBlock.File-isnot[string]-or[IO.Path]::GetFullPath($originalTest.ScriptBlock.File)-ine[IO.Path]::GetFullPath($ContainerPath)-or[IO.Path]::GetFullPath($row.Source)-ine[IO.Path]::GetFullPath($declared.SourcePath)){throw 'Original projected test source/container differs'}
            # Current reviewed matrix names expand <Kind>. Do not guess the
            # separate exporter Data-value suffix dialect for unreviewed names.
            if($null-ne$parameterizedName-and$originalTest.ExpandedPath-ceq$parameterizedName){throw 'Unreviewed parameterized argument-suffix XML name'}
            $identity=[IO.Path]::GetFullPath($ContainerPath).ToUpperInvariant()+[char]0+$originalTest.ExpandedPath
            if(-not$state.NUnitIdentities.Add($identity)){throw 'Ambiguous duplicate complete XML identity'}
            $state.NUnitProjection.Add(@{FlatIndex=$flatIndex;Name=$row.Name;ExpandedName=$row.ExpandedName;ExpandedPath=$row.ExpandedPath;SourcePath=$row.Source;ContainerPath=$ContainerPath;Result=$row.Result})
        }
    }
}
function Close-ControlXmlObjects {
    if($null-ne$state.XmlReader-and-not$state.XmlReaderDisposed){try{$state.XmlReader.Dispose();$state.XmlReaderDisposed=$true}catch{Add-ControlCause $_}}
    if($null-ne$state.XmlStream-and-not$state.XmlStreamDisposed){try{$state.XmlStream.Dispose();$state.XmlStreamDisposed=$true}catch{Add-ControlCause $_}}
}
function Get-ControlPending {
    $native='Pcai.Common.ProcessCaptureV1'-as[type]
    $capturePending=$null-ne$native-and(@($native::PendingCustody()).Count-gt0-or$null-ne$native::ActiveCustody())
    $writerPending=$null-ne$state.CaptureWriter-and(-not$state.CaptureWriter.Closed-or$null-eq$state.CaptureWriter.OriginalHandle-or-not$state.CaptureWriter.OriginalHandle.IsClosed)
    @{Native=$capturePending;CaptureWriter=$writerPending;OwnWriters=@($state.Writers|Where-Object{-not$_.Closed}).Count;Identity=(-not$state.IdentityClosed);Xml=(($null-ne$state.XmlReader-and-not$state.XmlReaderDisposed)-or($null-ne$state.XmlStream-and-not$state.XmlStreamDisposed))}
}
$inputRecord=$null;$registerHash=$null;$register=$null;$result=$null
try{
    Assert-ControlOrdinary $AdmissionPath;$admissionLeaf=Get-Item -LiteralPath $AdmissionPath -Force
    if($admissionLeaf-isnot[IO.FileInfo]-or$admissionLeaf.Length-gt131072){throw 'Bounded admission leaf required'}
    $bytes=[IO.File]::ReadAllBytes($AdmissionPath)
    if($bytes.Length-gt131072-or[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))-cne$AdmissionSHA256){throw 'Bounded same-byte Root admission differs'}
    $inputRecord=[Text.UTF8Encoding]::new($false,$true).GetString($bytes)|ConvertFrom-Json -AsHashtable -DateKind String
    if($inputRecord.State-cne'ROOT_ADMITTED_CANONICAL_CACHE_EIGHTEEN_ONLY'){throw 'Separate scoped Root admission required'}
    foreach($key in @('SourceReviewed','SerializedOwnerConfirmed','NoCompetingHeavyJobs','QualifiedSupervisor4','ManagedHandoffConfirmed','PriorOriginalCustodyClosed')){if($inputRecord[$key]-isnot[bool]-or-not$inputRecord[$key]){throw 'Root original custody and serialization attestations required'}}
    foreach($key in @('ObservedUtc','ExpiresUtc')){if($inputRecord[$key]-isnot[string]-or$inputRecord[$key]-cnotmatch'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z$'){throw 'Raw UTC admission required'}}
    $format="yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'";$culture=[Globalization.CultureInfo]::InvariantCulture;$styles=[Globalization.DateTimeStyles]::AssumeUniversal
    $observed=[DateTimeOffset]::ParseExact($inputRecord.ObservedUtc,$format,$culture,$styles);$expires=[DateTimeOffset]::ParseExact($inputRecord.ExpiresUtc,$format,$culture,$styles);$now=[DateTimeOffset]::UtcNow
    if($observed-gt$now-or($now-$observed).TotalSeconds-gt300-or$expires-le$now-or($expires-$observed).TotalSeconds-gt300){throw 'Fresh five-minute admission required'}
    if([string]::IsNullOrEmpty($CandidateRoot)){$CandidateRoot=Join-Path $PSScriptRoot '../../Modules/PC-AI.Common'}
    if([string]::IsNullOrEmpty($EvidenceDirectory)){$EvidenceDirectory=$inputRecord.EvidenceDirectory}
    if([IO.Path]::GetFullPath($CandidateRoot)-ine[IO.Path]::GetFullPath($inputRecord.ModuleRoot)-or[IO.Path]::GetFullPath($EvidenceDirectory)-ine[IO.Path]::GetFullPath($inputRecord.EvidenceDirectory)){throw 'Actual default or injected paths must match Root admission'}
    $registerPath=Join-Path $PSScriptRoot 'capture-source-register.json';$registerHash=$inputRecord.SourceRegisterSHA256
    . (Join-Path $PSScriptRoot 'Test-ProcessCaptureSources.ps1')
    $register=Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot $registerPath $registerHash
    if($register.Controls.Count-ne18){throw 'Exact eighteen control register required'}
    Assert-ControlRuntime $inputRecord.RuntimePins $inputRecord.PesterPins $register.PesterPayload $inputRecord.PesterPath
    foreach($key in @('PwshPath','CoreClrPath','PesterPath')){if($inputRecord[$key]-isnot[string]-or@($inputRecord.RuntimePins|Where-Object{[IO.Path]::GetFullPath($_.Path)-ieq[IO.Path]::GetFullPath($inputRecord[$key])}).Count-ne1){throw 'Selected runtime must be uniquely pinned'}}
    $state.Identity=[Diagnostics.Process]::GetCurrentProcess();$state.IdentityClosed=$false;$state.IdentityHandle=$state.Identity.SafeHandle
    if([IO.Path]::GetFullPath($state.Identity.MainModule.FileName)-ine[IO.Path]::GetFullPath($inputRecord.PwshPath)){throw 'Actual selected interpreter differs'}
    $loadedClr=@($state.Identity.Modules|Where-Object{$_.ModuleName-ieq'coreclr.dll'}|ForEach-Object FileName)
    if($loadedClr.Count-ne1-or[IO.Path]::GetFullPath($loadedClr[0])-ine[IO.Path]::GetFullPath($inputRecord.CoreClrPath)){throw 'Actual loaded CLR differs'}
    $state.Identity.Dispose();$state.IdentityClosed=$state.IdentityHandle.IsClosed
    if(-not$state.IdentityClosed){throw 'Original identity handle remains open'}
    if(Test-Path -LiteralPath $EvidenceDirectory){throw 'Fresh absent evidence namespace required'}
    Assert-ControlOrdinary ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($EvidenceDirectory)));[void][IO.Directory]::CreateDirectory($EvidenceDirectory);$state.OutputCreated=$true;Assert-ControlOrdinary $EvidenceDirectory
    Import-Module $inputRecord.PesterPath -ErrorAction Stop
    if((Get-Module Pester).Version.ToString()-cne'5.7.1'){throw 'Selected Pester contract differs'}
    $configuration=New-PesterConfiguration;$configuration.Run.Path=@();$configuration.Run.PassThru=$true;$configuration.Run.Exit=$false
    $configuration.Run.Container=@(
        (New-PesterContainer -Path (Join-Path $PSScriptRoot 'CacheProcessCaptureContracts.Tests.ps1') -Data @{CandidateRoot=$CandidateRoot;SourceManifestPath=$registerPath;SourceManifestSHA256=$registerHash;PwshPath=$inputRecord.PwshPath;EvidenceDirectory=$EvidenceDirectory}),
        (New-PesterContainer -Path (Join-Path $PSScriptRoot 'CacheCaptureCollaborators.Tests.ps1') -Data @{CandidateRoot=$CandidateRoot;SourceManifestPath=$registerPath;SourceManifestSHA256=$registerHash}))
    $configuration.Filter.FullName=@();$configuration.Filter.Tag=@();$configuration.Filter.ExcludeTag=@();$configuration.TestResult.Enabled=$false;$configuration.CodeCoverage.Enabled=$false;$configuration.Output.Verbosity='Detailed'
    $result=Invoke-Pester -Configuration $configuration;$state.Pester=$result
    foreach($container in @($result.Containers)){foreach($cause in @($container.ErrorRecord)){if($null-ne$cause){Add-ControlCause $cause}}}
    foreach($block in @($result.FailedBlocks)){foreach($cause in @($block.ErrorRecord)){if($null-ne$cause){Add-ControlCause $cause}}}
    foreach($test in @($result.Tests)){foreach($cause in @($test.ErrorRecord)){if($null-ne$cause){Add-ControlCause $cause}}}
    $readiness=(Get-Variable PcaiCacheCancellationReadinessR1 -Scope Global -ErrorAction Stop).Value
    if($null-ne$readiness.FirstErrorRecord){Add-ControlCause $readiness.FirstErrorRecord}
    $originalWaiter=$readiness.Original
    if($null-ne$originalWaiter.FirstFailure){try{throw $originalWaiter.FirstFailure}catch{Add-ControlCause $_}}
    foreach($originalFailure in $originalWaiter.SecondaryFailures){try{throw $originalFailure}catch{Add-ControlCause $_}}
    if([Pcai.Tests.CacheCancellationR1.Waiter]::ContractVersion-ne1-or$originalWaiter.SourceBinding-cne$readiness.SourceBinding-or-not$originalWaiter.ReadyObserved-or-not$originalWaiter.CancellationRequested-or-not$originalWaiter.OriginalResourcesClosed-or$originalWaiter.RepeatedFailures-ne0){throw 'Original readiness waiter/cancellation/errors/handles must settle before exports'}
    $state.CancellationReadinessOriginal=$originalWaiter;$state.CancellationReadiness=$originalWaiter.Snapshot()
    $writerVariable=Get-Variable PcaiCacheCaptureReceiptWriterR2 -Scope Global -ErrorAction Stop;$state.CaptureWriter=$writerVariable.Value
    $writer=$state.CaptureWriter
    if($writer.State-cne'CLOSED'-or-not$writer.Closed-or-not$writer.DisposalReturned-or-not$writer.PostDisposalAttested-or$null-eq$writer.OriginalHandle-or-not$writer.OriginalHandle.IsClosed-or$null-ne$writer.FirstErrorRecord-or$writer.SecondaryErrorRecords.Count-or$writer.RepeatedSecondaryFailures){throw 'Original post-Pester capture receipt writer did not settle cleanly'}
    $state.Rows=@($result.Tests|ForEach-Object{@{Name=$_.Name;ExpandedName=$_.ExpandedName;ExpandedPath=$_.ExpandedPath;Result=$_.Result;Source=$_.ScriptBlock.File}})
    if($null-ne$state.First){throw $state.First}
    if($result.TotalCount-ne18-or$result.PassedCount-ne18-or$result.FailedCount-or$result.SkippedCount-or$result.NotRunCount){throw 'All eighteen actual controls must pass with no filtered/missing rows'}
    for($i=0;$i-lt18;$i++){if($state.Rows[$i].Name-cne$register.Controls[$i].Name-or$state.Rows[$i].Result-cne'Passed'){throw 'Actual control names/order differ'}}
    $pending=Get-ControlPending;if($pending.Native-or$pending.CaptureWriter-or$pending.OwnWriters-or$pending.Identity-or$pending.Xml){throw 'Original resources remain pending'}
    $expectedRows=@(foreach($control in $register.Controls){@{Name=$control.Name;ExpandedName=$control.Name;Result='Passed';SourcePath=(Join-Path $PSScriptRoot ([IO.Path]::GetFileName($control.Source)))}})
    foreach($test in $result.Tests){$state.NUnitFlatTests.Add($test)}
    if($state.NUnitFlatTests.Count-ne18-or$state.Rows.Count-ne18){throw 'Exact eighteen original flat tests required'}
    $projectionGroupCommand=Get-Command -Name Group-Object -Module Microsoft.PowerShell.Utility -CommandType Cmdlet -ErrorAction Stop
    if($projectionGroupCommand.ModuleName-cne'Microsoft.PowerShell.Utility'-or$projectionGroupCommand.Name-cne'Group-Object'){throw 'Exact pinned exporter grouping cmdlet required'}
    foreach($container in $result.Containers){if($container.ShouldRun-isnot[bool]-or$container.Name-isnot[string]){throw 'Typed original Pester container required'};if($container.ShouldRun){Add-ControlNUnitProjection $container $container.Name 0}}
    if($state.NUnitProjection.Count-ne18-or$state.NUnitSeenIndexes.Count-ne18){throw 'Projection must include all eighteen original tests exactly once'}
    # Qualified R6 exact declaration correction; parse actual closed UTF8 bytes.
    $xml=ConvertTo-NUnitReport -Result $result -AsString -Format NUnit2.5
    $oldProlog='<?xml version="1.0" encoding="utf-16" standalone="no"?>';$newProlog='<?xml version="1.0" encoding="utf-8" standalone="no"?>'
    if($xml-isnot[string]-or-not$xml.StartsWith($oldProlog,[StringComparison]::Ordinal)-or[regex]::Matches($xml,[regex]::Escape($oldProlog)).Count-ne1){throw 'Exact sole Pester UTF16 prolog required'}
    $xml=$xml.Replace($oldProlog,$newProlog);$xmlBytes=[Text.UTF8Encoding]::new($false,$true).GetBytes($xml)
    $xmlPath=Join-Path $EvidenceDirectory 'tests.xml';Write-ControlFresh $xmlPath $xmlBytes
    if($null-ne$state.First-or$state.Secondary.Count){throw 'Original XML writer failed'}
    $xmlWriters=@($state.Writers|Where-Object{$_.Path-ceq[IO.Path]::GetFullPath($xmlPath)})
    if($xmlWriters.Count-ne1-or-not$xmlWriters[0].Closed-or-not$xmlWriters[0].DisposalReturned-or$xmlWriters[0].Handle-isnot[Microsoft.Win32.SafeHandles.SafeFileHandle]-or-not$xmlWriters[0].Handle.IsClosed){throw 'Original XML writer must close before physical read/parse'}
    Assert-ControlOrdinary $xmlPath;$xmlLeaf=Get-Item -LiteralPath $xmlPath -Force
    if($xmlLeaf-isnot[IO.FileInfo]-or$xmlLeaf.Length-lt1-or$xmlLeaf.Length-gt2097152){throw 'Physical XML byte cap'}
    $actualXmlBytes=[IO.File]::ReadAllBytes($xmlPath);$expectedXmlHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($xmlBytes));$actualXmlHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($actualXmlBytes))
    if($actualXmlBytes.Length-ne$xmlBytes.Length-or$actualXmlBytes.Length-gt2097152-or$actualXmlHash-cne$expectedXmlHash){throw 'Physical XML differs from exact UTF8 export'}
    try{
        $state.XmlStream=[IO.MemoryStream]::new($actualXmlBytes,$false)
        $settings=[Xml.XmlReaderSettings]::new();$settings.DtdProcessing=[Xml.DtdProcessing]::Prohibit;$settings.XmlResolver=$null;$settings.CloseInput=$false;$settings.MaxCharactersInDocument=2097152;$settings.MaxCharactersFromEntities=0
        $state.XmlReader=[Xml.XmlReader]::Create($state.XmlStream,$settings)
        $document=[Xml.XmlDocument]::new();$document.XmlResolver=$null;$document.Load($state.XmlReader)
        $top=$document.DocumentElement;if($null-eq$top-or$top.LocalName-cne'test-results'-or$top.NamespaceURI-cne''){throw 'Exact NUnit top element required'}
        foreach($pair in @(@('total','18'),@('errors','0'),@('failures','0'),@('not-run','0'),@('inconclusive','0'),@('ignored','0'),@('skipped','0'),@('invalid','0'))){if($top.GetAttribute($pair[0])-cne$pair[1]){throw 'Physical NUnit eighteen counters differ'}}
        $cases=$document.SelectNodes('//test-case');if($cases.Count-ne18){throw 'Physical XML requires all eighteen source-bound cases'}
        for($caseIndex=0;$caseIndex-lt18;$caseIndex++){
            $case=$cases[$caseIndex];$row=$state.NUnitProjection[$caseIndex]
            if($case.GetAttribute('description')-cne$row.ExpandedName-or$case.GetAttribute('name')-cne$row.ExpandedPath){throw 'Physical XML projected full identity differs'}
            $ancestor=$case.ParentNode;$matches=0;$depth=0
            while($null-ne$ancestor){if(++$depth-gt32){throw 'Bounded physical XML ancestry required'};if($ancestor-is[Xml.XmlElement]-and$ancestor.LocalName-ceq'test-suite'){$suiteName=$ancestor.GetAttribute('name');if([IO.Path]::IsPathRooted($suiteName)-and[IO.Path]::GetFullPath($suiteName)-ieq[IO.Path]::GetFullPath($row.ContainerPath)){$matches++}};$ancestor=$ancestor.ParentNode}
            if($matches-ne1-or$case.GetAttribute('result')-cne'Success'-or$case.GetAttribute('success')-cne'True'-or$case.GetAttribute('executed')-cne'True'){throw 'Physical XML original container/passed case differs'}
        }
        $state.Xml=@{State='PHYSICAL_UTF8_NUNIT_EIGHTEEN_ROWS_VERIFIED';Path=$xmlPath;Bytes=$actualXmlBytes.Length;SHA256=$actualXmlHash;Total=18;Passed=18;Skipped=0;DTD='PROHIBITED';ExternalResolver='NONE';Encoding='utf-8';ProjectionContract='PINNED_PESTER_NUNIT2_5_BLOCKS_THEN_GROUPID';Projection=$state.NUnitProjection.ToArray();ReaderDisposed=$false;StreamDisposed=$false;IndependentReview='REQUIRED_BEFORE_ACTUAL_ACCEPTANCE'}
    }catch{Add-ControlCause $_}finally{Close-ControlXmlObjects}
    if($null-ne$state.First-or$state.Secondary.Count-or$null-eq$state.Xml-or-not$state.XmlReaderDisposed-or-not$state.XmlStreamDisposed){throw 'Physical XML validation or original reader disposal failed'}
    $state.Xml.ReaderDisposed=$state.XmlReaderDisposed;$state.Xml.StreamDisposed=$state.XmlStreamDisposed
    if($null-ne$state.First){throw $state.First}
    $state.State='PASS_EIGHTEEN_CONTROLS_ONLY_PENDING_PHYSICAL_XML_AND_PARENT_CLOSURE'
}catch{Add-ControlCause $_;$state.State='FAIL'}
finally{
    Close-ControlXmlObjects
    if($null-ne$state.Identity-and-not$state.IdentityClosed){try{$state.Identity.Dispose();$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-ControlCause $_}}
    try{if($null-ne$registerHash){[void](Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot (Join-Path $PSScriptRoot 'capture-source-register.json') $registerHash);$state.SourceStable=$true};if($null-ne$inputRecord){Assert-ControlRuntime $inputRecord.RuntimePins $inputRecord.PesterPins $register.PesterPayload $inputRecord.PesterPath;$state.RuntimeStable=$true}}catch{Add-ControlCause $_;$state.State='FAIL'}
    if($state.OutputCreated){
        $payload=@{StateAtSerialization=$state.State;FinalDispositionNotSerialized=$true;Rows=$state.Rows;SourceRegisterSHA256=$registerHash;AdmissionSHA256=$AdmissionSHA256;SourceStable=$state.SourceStable;RuntimeStable=$state.RuntimeStable;FirstFailure=$(if($null-ne$state.First){@{Type=$state.First.Exception.GetType().FullName;Message=$state.First.Exception.Message;ErrorId=$state.First.FullyQualifiedErrorId}}else{$null});SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures;Xml=$state.Xml;OriginalCaptureWriterPostPesterClosed=($null-ne$state.CaptureWriter-and$state.CaptureWriter.Closed-and$state.CaptureWriter.OriginalHandle.IsClosed);SelfWriterClosed=$false;Limits='No canonical adoption/performance/portable/rare-fault acceptance; same-child and parent post-return original closure required'}
        try{Write-ControlFresh (Join-Path $EvidenceDirectory 'result.json') ([Text.UTF8Encoding]::new($false,$true).GetBytes(($payload|ConvertTo-Json -Depth 12)))}catch{Add-ControlCause $_;$state.State='FAIL'}
    }
    $pending=Get-ControlPending
    while($pending.Native-or$pending.CaptureWriter-or$pending.OwnWriters-or$pending.Identity-or$pending.Xml){
        Close-ControlXmlObjects
        $state.State='HOLD';foreach($entry in $state.Writers){if(-not$entry.Closed){Settle-ControlWriter $entry}}
        if($null-ne$state.Identity-and-not$state.IdentityClosed){try{$state.Identity.Dispose();$state.IdentityClosed=$null-ne$state.IdentityHandle-and$state.IdentityHandle.IsClosed}catch{Add-ControlCause $_}}
        # No termination/reconstructed resources. Qualified parent retains alive child
        # even if marker IO fails; capture controller owns its original retries.
        [Threading.Thread]::Sleep(1000);$pending=Get-ControlPending
    }
    $state.Clock.Stop();$state.ElapsedMilliseconds=$state.Clock.ElapsedMilliseconds
    if($state.ElapsedMilliseconds-ge180000){Add-ControlCause ([TimeoutException]::new('Dedicated controls exceeded cooperative 180-second budget'));$state.State='FAIL'}
    if($null-ne$state.First-or$state.Secondary.Count-or$state.RepeatedSecondaryFailures){$state.State='FAIL'}
}
try{
    $finalPending=Get-ControlPending
    $finalNative='Pcai.Common.ProcessCaptureV1'-as[type]
    [Console]::Out.WriteLine((@{FinalState=$state.State;ClockStopped=(-not$state.Clock.IsRunning);ElapsedMilliseconds=$state.ElapsedMilliseconds;OriginalWritersClosed=(@($state.Writers|Where-Object{-not$_.Closed-or-not$_.DisposalReturned-or$null-eq$_.Handle-or-not$_.Handle.IsClosed}).Count-eq0);CaptureWriterClosed=(-not$finalPending.CaptureWriter);IdentityClosed=$state.IdentityClosed;XmlReaderDisposed=$state.XmlReaderDisposed;XmlStreamDisposed=$state.XmlStreamDisposed;NativePending=$(if($null-ne$finalNative){@($finalNative::PendingCustody()).Count}else{0});FirstFailure=$(if($null-ne$state.First){@{Type=$state.First.Exception.GetType().FullName;Message=$state.First.Exception.Message;ErrorId=$state.First.FullyQualifiedErrorId}}else{$null});SecondaryCount=$state.Secondary.Count;RepeatedSecondaryFailures=$state.RepeatedSecondaryFailures}|ConvertTo-Json -Compress -Depth 5))
}catch{Add-ControlCause $_;$state.State='FAIL'}
if($state.State-cne'PASS_EIGHTEEN_CONTROLS_ONLY_PENDING_PHYSICAL_XML_AND_PARENT_CLOSURE'){exit 1}
$state.State
