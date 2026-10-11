function QA-Assert([bool]$Condition,[string]$Message){if(-not$Condition){throw [IO.InvalidDataException]::new($Message)}}

function QA-Closed($Operation,[switch]$AllowFailedExit){
    QA-Assert ($null-ne$Operation-and$Operation.Disposition-ceq'CLOSED'-and-not$Operation.ChildHoldRetained-and$null-eq$Operation.CustodyId) 'Original outer operation closed'
    foreach($key in @('Started','ProcessHandleCaptured','JobAssigned','ParentExitConfirmed','JobEmptyConfirmed','DrainsSettled','DrainsSucceeded','FilesFlushed','ResourcesDisposed','OriginalHandleDisposed')){QA-Assert ($Operation.$key-is[bool]-and$Operation.$key) 'All ten original outer closure flags'}
    if(-not$AllowFailedExit){QA-Assert ($Operation.ExitCode-eq0-and$null-eq$Operation.PrimaryFailure-and$Operation.SecondaryFailures.Count-eq0) 'No outer failure promotion'}
}

function QA-Cause($Cause){if($null-eq$Cause){return $null};if($Cause-is[Management.Automation.ErrorRecord]){return @{Type=$Cause.Exception.GetType().FullName;Message=$Cause.Exception.Message;ErrorId=$Cause.FullyQualifiedErrorId}};@{Type=$Cause.GetType().FullName;Message=$Cause.ToString()}}

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

# Definitions only; appended after unchanged source-bound QA/runtime validators.
function Assert-DefaultsSources($Register,[string]$Root){
 QA-Assert ($Register.Files-is[Array]-and$Register.Files.Count-ge16-and$Register.Files.Count-le32) 'Bounded complete maintained source register'
 $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
 foreach($row in $Register.Files){QA-Assert ($row.Path-is[string]-and$row.Path-cmatch'^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$'-and$row.Path.Split('/')-cnotcontains'..'-and$seen.Add($row.Path)-and$row.SHA256-cmatch'^[A-F0-9]{64}$') 'Safe unique source row';$path=Join-Path $Root $row.Path;Assert-DefaultsOrdinary $path;$leaf=Get-Item -LiteralPath $path -Force;QA-Assert ($leaf-is[IO.FileInfo]-and$leaf.Length-eq$row.Bytes-and(Get-FileHash -LiteralPath $path).Hash-ceq$row.SHA256) 'Physical maintained source differs'}
 $module=Join-Path $Root 'Modules/PC-AI.Common';$physical=@(Get-ChildItem -LiteralPath $module -Recurse -File -Force);QA-Assert ($physical.Count-eq8) 'Exactly eight complete maintained module leaves';foreach($leaf in $physical){Assert-DefaultsOrdinary $leaf.FullName;$relative=[IO.Path]::GetRelativePath($Root,$leaf.FullName).Replace('\','/');QA-Assert ($seen.Contains($relative)) 'No omitted module source'}
 $registerPaths=@(@{Path='Tests/ProcessCapture/capture-source-register.json';Hash=$Register.EighteenSourceRegisterSHA256},@{Path='Tests/ProcessCapture/ModuleQA/module-qa-source-register.json';Hash=$Register.ModuleQaSourceRegisterSHA256})
 foreach($row in $registerPaths){QA-Assert ((Get-FileHash -LiteralPath (Join-Path $Root $row.Path)).Hash-ceq$row.Hash) 'Mode source register pairing'}
}
function Select-DefaultsRuntime($Original,[string]$PesterPath,$Payload){
 QA-Assert (-not[string]::IsNullOrEmpty($PesterPath)) 'Explicit complete compatible Pester manifest required; missing runtime refused'
 $image=$Original.MainModule.FileName;$clr=@($Original.Modules|Where-Object ModuleName -IEQ 'coreclr.dll'|ForEach-Object FileName);QA-Assert ($clr.Count-eq1) 'Exactly one actual loaded CLR'
 $selected=[Collections.Generic.List[object]]::new();$pesterRoot=[IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($PesterPath))
 QA-Assert ([IO.Path]::GetFileName($PesterPath)-ceq'Pester.psd1'-and$Payload.Version-ceq'5.7.1') 'Source-compatible Pester manifest/version'
 foreach($row in $Payload.Required){$path=Join-Path $pesterRoot $row.Path;Assert-DefaultsOrdinary $path;$item=Get-Item -LiteralPath $path -Force;QA-Assert ($item-is[IO.FileInfo]-and$item.Length-eq$row.Bytes-and(Get-FileHash -LiteralPath $path).Hash-ceq$row.SHA256) 'Complete compatible Pester payload required';$selected.Add(@{Path=[IO.Path]::GetFullPath($path);Bytes=$item.Length;SHA256=$row.SHA256})}
 $metadata=Join-Path $pesterRoot 'PSGetModuleInfo.xml';if(Test-Path -LiteralPath $metadata){Assert-DefaultsOrdinary $metadata;$item=Get-Item -LiteralPath $metadata -Force;$selected.Add(@{Path=$item.FullName;Bytes=$item.Length;SHA256=(Get-FileHash -LiteralPath $metadata).Hash})}
 $runtime=[Collections.Generic.List[object]]::new();foreach($path in @($image,$clr[0])){Assert-DefaultsOrdinary $path;$item=Get-Item -LiteralPath $path -Force;$runtime.Add(@{Path=[IO.Path]::GetFullPath($path);Bytes=$item.Length;SHA256=(Get-FileHash -LiteralPath $path).Hash})};foreach($row in $selected){$runtime.Add($row)}
 $selection=@{PwshPath=[IO.Path]::GetFullPath($image);CoreClrPath=[IO.Path]::GetFullPath($clr[0]);PesterManifestPath=[IO.Path]::GetFullPath($PesterPath);PesterPins=$selected.ToArray();RuntimePins=$runtime.ToArray()};Assert-DefaultsRuntime $selection $Payload;$selection
}
function Assert-DefaultsRuntime($Runtime,$Payload){Assert-ControlRuntime $Runtime.RuntimePins $Runtime.PesterPins $Payload $Runtime.PesterManifestPath}
function Get-DefaultsCapacity([string]$Root,[string]$Requested,[string]$Id,[switch]$ExistingRoot){
 $output=if([string]::IsNullOrEmpty($Requested)){Join-Path $Root ".pcai/qualifications/$Id"}else{[IO.Path]::GetFullPath($Requested)}
 QA-Assert (-not$Id.EndsWith('.',[StringComparison]::Ordinal)-and$Id.Split('.')[0]-notmatch'^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])$') 'Reject reserved Windows output basenames and trailing dot'
 QA-Assert ($Id-cnotmatch'\d{4}[-_]\d{2}[-_]\d{2}'-and[IO.Path]::GetFileName($output)-ceq$Id) 'Stable exact owned output ID'
 $ancestor=$output;while(-not(Test-Path -LiteralPath $ancestor)){$next=[IO.Path]::GetDirectoryName($ancestor);QA-Assert (-not[string]::IsNullOrEmpty($next)-and$next-cne$ancestor) 'Ordinary output ancestry';$ancestor=$next};Assert-DefaultsOrdinary $ancestor
 QA-Assert ((Get-Item -LiteralPath $ancestor)-is[IO.DirectoryInfo]) 'Output ancestor directory'
 if($ExistingRoot){QA-Assert (Test-Path -LiteralPath $output) 'Same original output root'}else{QA-Assert (-not(Test-Path -LiteralPath $output)) 'Absent output namespace; no overwrite'}
 $os=Get-CimInstance Win32_OperatingSystem;$memory=Get-Counter @('\Memory\% Committed Bytes In Use','\Memory\Commit Limit','\Memory\Committed Bytes') -MaxSamples 1
 $commit=@($memory.CounterSamples|Where-Object Path -Like '*\% committed bytes in use')[0].CookedValue;$limit=@($memory.CounterSamples|Where-Object Path -Like '*\commit limit')[0].CookedValue;$used=@($memory.CounterSamples|Where-Object Path -Like '*\committed bytes')[0].CookedValue
 $drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($output));$capacity=@{AvailableMiB=[math]::Floor($os.FreePhysicalMemory/1024);CommitPercent=$commit;CommitHeadroomBytes=[long]($limit-$used);OutputFreeBytes=$drive.AvailableFreeSpace;OutputRoot=[IO.Path]::GetFullPath($output);ObservedUtc=[DateTime]::UtcNow.ToString('o')}
 QA-Assert ($capacity.AvailableMiB-ge8192-and$capacity.CommitPercent-lt80-and$capacity.CommitHeadroomBytes-ge10GB-and$capacity.OutputFreeBytes-ge10GB) 'Carry admitted 8GiB/<80%/10GiB headroom/free bounds'
 QA-Assert (@(Get-Process|Where-Object{$_.Name-in@('cargo','rustc','dotnet','csc','MSBuild','cl')}).Count-eq0) 'Known competing compiler requires serialization'
 $capacity
}
function Settle-DefaultsOriginals {
 if($null-ne$state.Identity-and-not$state.IdentityClosed){if($null-eq$state.IdentityHandle){try{$state.IdentityHandle=$state.Identity.SafeHandle}catch{Add-DefaultsCause $_}};if(-not$state.IdentityDisposalReturned){try{$state.Identity.Dispose();$state.IdentityDisposalReturned=$true}catch{Add-DefaultsCause $_}};if($state.IdentityDisposalReturned-and$null-ne$state.IdentityHandle){try{$state.IdentityClosed=$state.IdentityHandle.IsClosed}catch{Add-DefaultsCause $_}}}
 foreach($writer in $state.Writers){if($writer.Closed){continue};if($null-eq$writer.Stream){$writer.Closed=(-not$writer.OpenAttempted)-or$writer.OpenFailed;continue};if($null-eq$writer.OriginalHandle){try{$writer.OriginalHandle=$writer.Stream.SafeFileHandle}catch{Add-DefaultsCause $_}};if(-not$writer.DisposalReturned){try{$writer.Stream.Dispose();$writer.DisposalReturned=$true}catch{Add-DefaultsCause $_}};try{$writer.Closed=$writer.DisposalReturned-and$null-ne$writer.OriginalHandle-and$writer.OriginalHandle.IsClosed}catch{Add-DefaultsCause $_}}
 foreach($entry in $state.XmlReaders){if(-not$entry.ReaderClosed){try{$entry.Reader.Dispose();$entry.ReaderClosed=$true}catch{Add-DefaultsCause $_}};if(-not$entry.StreamClosed){try{$entry.Stream.Dispose();$entry.StreamClosed=$true}catch{Add-DefaultsCause $_}}}
}
function Wait-DefaultsOriginals($Supervisor){
 while($true){
  Settle-DefaultsOriginals
  if($null-ne$Supervisor){foreach($operation in $state.Operations){if($operation.PrimaryFailure){Add-DefaultsCause $operation.PrimaryFailure};foreach($cause in $operation.SecondaryFailures){Add-DefaultsCause $cause};if($null-ne$operation.CustodyId-and$operation.Disposition-cne'CLOSED'){try{$recovery=$Supervisor::RetryOwnedCustody($operation.CustodyId,15000);QA-Assert ([object]::ReferenceEquals($recovery.LatestObservation,$operation)) 'Recovery remains original observation';if($recovery.OriginalPrimaryFailure){Add-DefaultsCause $recovery.OriginalPrimaryFailure}}catch{Add-DefaultsCause $_}}}}
  $localPending=(-not$state.IdentityClosed)-or(@($state.Writers|Where-Object{-not$_.Closed}).Count-gt0)-or(@($state.XmlReaders|Where-Object{-not$_.ReaderClosed-or-not$_.StreamClosed}).Count-gt0)
  $nativePending=$null-ne$Supervisor-and@($Supervisor::PendingCustody()).Count-gt0
  if(-not$localPending-and-not$nativePending){break}
  $state.State='HOLD';try{[Console]::Error.WriteLine('HOLD: retain SAME foreground parent and original child/resources; no termination or PID reconstruction')}catch{Add-DefaultsCause $_};[Threading.Thread]::Sleep(1000)
 }
}
function Write-DefaultsFresh([string]$Path,[byte[]]$Bytes){
 QA-Assert ($Bytes.Length-le2097152) 'Retain two MiB bounded own receipts'
 $full=[IO.Path]::GetFullPath($Path);QA-Assert ([IO.Path]::GetDirectoryName($full)-ieq[IO.Path]::GetFullPath($EvidenceDirectory)) 'Only direct owned receipt leaves'
 $entry=@{Path=$full;Stream=$null;OriginalHandle=$null;OpenAttempted=$false;OpenFailed=$false;DisposalReturned=$false;Closed=$false};$state.Writers.Add($entry)
 try{$entry.OpenAttempted=$true;try{$entry.Stream=[IO.File]::Open($full,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)}catch{$entry.OpenFailed=$true;throw};$entry.OriginalHandle=$entry.Stream.SafeFileHandle;$entry.Stream.Write($Bytes,0,$Bytes.Length);$entry.Stream.Flush($true)}catch{Add-DefaultsCause $_}finally{Settle-DefaultsOriginals}
}
function Get-DefaultsLedgerSnapshot {
 QA-Assert ($state.Writers.Count-ge1-and$state.Writers.Count-le64) 'Actual current CACHECCB originals'
 $writers=@(foreach($writer in $state.Writers){QA-Assert ($writer.Closed-and$writer.DisposalReturned-and$null-ne$writer.Stream-and$null-ne$writer.OriginalHandle-and$writer.OriginalHandle.IsClosed) 'Original input writer closed';@{Original=$writer;Stream=$writer.Stream;Handle=$writer.OriginalHandle}})
 @{Original=$state;First=$state.FirstErrorRecord;Secondary=@($state.Secondary);Writers=$writers}
}
function Assert-DefaultsLedgerSnapshot($Snapshots){foreach($snapshot in $Snapshots){QA-Assert ([object]::ReferenceEquals($state,$snapshot.Original)-and$state.Writers.Count-ge$snapshot.Writers.Count) 'Original parent ledger remains retained';if($null-ne$snapshot.First){QA-Assert ([object]::ReferenceEquals($state.FirstErrorRecord,$snapshot.First)) 'Original first error reference retained'};for($i=0;$i-lt$snapshot.Secondary.Count;$i++){QA-Assert ([object]::ReferenceEquals($state.Secondary[$i],$snapshot.Secondary[$i])) 'Original secondary reference retained'};for($i=0;$i-lt$snapshot.Writers.Count;$i++){$saved=$snapshot.Writers[$i];$current=$state.Writers[$i];QA-Assert ([object]::ReferenceEquals($saved.Original,$current)-and[object]::ReferenceEquals($saved.Stream,$current.Stream)-and[object]::ReferenceEquals($saved.Handle,$current.OriginalHandle)-and$current.Closed-and$current.DisposalReturned-and$current.OriginalHandle.IsClosed) 'Prior original writer references closed unchanged'}}}
function Read-DefaultsChild([string]$Mode,[string]$Directory,[string]$Stdout,[string]$AdmissionHash,$Register,[string]$Root){
 $resultPath=Join-Path $Directory 'result.json';$result=Read-DefaultsJson $resultPath (Get-FileHash -LiteralPath $resultPath).Hash 2097152
 $state.ChildInputs.Add(@{Mode=$Mode;Path=$resultPath;OriginalParsedResult=$result;FinalSummary=$null})
 if($result.FirstFailure){Add-DefaultsCause ([InvalidOperationException]::new(($result.FirstFailure|ConvertTo-Json -Compress -Depth 8)))}
 Assert-DefaultsOrdinary $Stdout;$stdoutBytes=[IO.File]::ReadAllBytes($Stdout);QA-Assert ($stdoutBytes.Length-le2097152) 'Bounded original stdout'
 $lines=[Text.UTF8Encoding]::new($false,$true).GetString($stdoutBytes).TrimEnd()-split"`n"
 $summaryLine=if($Mode-ceq'Eighteen'){$lines[-2]}else{$lines[-1]};$summary=$summaryLine|ConvertFrom-Json -AsHashtable -DateKind String;$state.ChildInputs[-1].FinalSummary=$summary
 if($summary.FirstFailure){Add-DefaultsCause ([InvalidOperationException]::new(($summary.FirstFailure|ConvertTo-Json -Compress -Depth 8)))}
 QA-Assert ($result.AdmissionSHA256-ceq$AdmissionHash-and$result.SourceStable-and$result.RuntimeStable-and$null-eq$result.FirstFailure-and$result.SecondaryCount-eq0-and$result.RepeatedSecondaryFailures-eq0) 'Source-bound child original errors retained before counters'
 QA-Assert ($summary.ClockStopped-and$summary.OriginalWritersClosed-and$summary.IdentityClosed-and$summary.XmlReaderDisposed-and$summary.XmlStreamDisposed-and$null-eq$summary.FirstFailure-and$summary.SecondaryCount-eq0-and$summary.RepeatedSecondaryFailures-eq0) 'Final original clock/readers/writers without write-time self-closure inference'
 $captureMode=$Mode-ceq'Eighteen';$total=if($captureMode){18}else{12};$skipped=if($captureMode){0}else{1}
 if($captureMode){QA-Assert ($result.StateAtSerialization-ceq'PASS_EIGHTEEN_CONTROLS_ONLY_PENDING_PHYSICAL_XML_AND_PARENT_CLOSURE'-and$summary.FinalState-ceq$result.StateAtSerialization-and$lines[-1].Trim()-ceq$result.StateAtSerialization-and$summary.ElapsedMilliseconds-lt180000-and$result.Rows.Count-eq18-and$result.SourceRegisterSHA256-ceq$Register.EighteenSourceRegisterSHA256-and$result.OriginalCaptureWriterPostPesterClosed-and$summary.CaptureWriterClosed-and$summary.NativePending-eq0) 'Complete maintained eighteen actual controls'}else{QA-Assert ($result.StateAtSerialization-ceq'PASS_MODULE_SMOKE_SIX_CAPTURE_AND_MAINTAINED_ELEVEN_ONLY'-and$summary.FinalState-ceq$result.StateAtSerialization-and$summary.ElapsedMilliseconds-lt160000-and$result.SourceSeal-ceq$Register.ModuleQaSourceRegisterSHA256-and$result.ImportNoCaptureType-and$result.MemoryNoCaptureType-and$result.Captures.Count-eq6-and$result.Measurements.Count-eq8-and$result.Total-eq12-and$result.Passed-eq11-and$result.Skipped-eq1-and$result.Failed-eq0-and$result.NotRun-eq0) 'Full real module/memory/six captures and genuine11/declared Redis skip'}
 $xmlPath=Join-Path $Directory 'tests.xml';Assert-DefaultsOrdinary $xmlPath;$bytes=[IO.File]::ReadAllBytes($xmlPath);QA-Assert ($bytes.Length-le2097152-and[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))-ceq$result.Xml.SHA256-and$bytes.Length-eq$result.Xml.Bytes) 'Physical same-byte UTF8 NUnit'
 QA-Assert ([Text.UTF8Encoding]::new($false,$true).GetString($bytes).StartsWith('<?xml version="1.0" encoding="utf-8" standalone="no"?>',[StringComparison]::Ordinal)) 'Actual UTF8 prolog'
 $entry=@{Stream=$null;Reader=$null;ReaderClosed=$true;StreamClosed=$true};$state.XmlReaders.Add($entry);$entry.Stream=[IO.MemoryStream]::new($bytes,$false);$entry.StreamClosed=$false
 try{
  $settings=[Xml.XmlReaderSettings]::new();$settings.DtdProcessing=[Xml.DtdProcessing]::Prohibit;$settings.XmlResolver=$null;$settings.CloseInput=$false;$settings.MaxCharactersInDocument=2097152;$settings.MaxCharactersFromEntities=0;$entry.Reader=[Xml.XmlReader]::Create($entry.Stream,$settings);$entry.ReaderClosed=$false;$document=[Xml.XmlDocument]::new();$document.XmlResolver=$null;$document.Load($entry.Reader)
  QA-Assert ($document.DocumentElement.LocalName-ceq'test-results'-and$document.DocumentElement.NamespaceURI-ceq'') 'Exact NUnit2.5 document'
  foreach($pair in @(@('total',[string]$total),@('errors','0'),@('failures','0'),@('not-run','0'),@('inconclusive','0'),@('ignored','0'),@('skipped',[string]$skipped),@('invalid','0'))){QA-Assert ($document.DocumentElement.GetAttribute($pair[0])-ceq$pair[1]) 'Actual physical counters'}
  $cases=$document.SelectNodes('//test-case');$projection=$result.Xml.Projection;QA-Assert ($cases.Count-eq$total-and$projection.Count-eq$total) 'Complete ordered physical identities'
  $seen=[Collections.Generic.HashSet[int]]::new();$modeRows=if($captureMode){$Register.CaptureControls}else{$Register.MaintainedControls}
  for($i=0;$i-lt$total;$i++){$row=$projection[$i];$case=$cases[$i];QA-Assert (($row.FlatIndex-is[int]-or$row.FlatIndex-is[long])-and$row.FlatIndex-ge0-and$row.FlatIndex-lt$total-and$seen.Add([int]$row.FlatIndex)) 'Distinct original flat identity';$expected=$modeRows[$row.FlatIndex];$source=Join-Path $Root $expected.SourcePath;QA-Assert ($row.Name-ceq$expected.Name-and$row.ExpandedName-ceq$expected.ExpandedName-and$row.Result-ceq$expected.Result-and[IO.Path]::GetFullPath($row.SourcePath)-ieq[IO.Path]::GetFullPath($source)) 'Actual ordered source identity';QA-Assert ($case.GetAttribute('name')-ceq$row.ExpandedPath-and$case.GetAttribute('description')-ceq$row.ExpandedName) 'Complete XML case identity';$ancestor=$case.ParentNode;$matches=0;$depth=0;while($null-ne$ancestor){QA-Assert (++$depth-le32) 'Bounded XML ancestry';if($ancestor-is[Xml.XmlElement]-and$ancestor.LocalName-ceq'test-suite'-and[IO.Path]::IsPathRooted($ancestor.GetAttribute('name'))-and[IO.Path]::GetFullPath($ancestor.GetAttribute('name'))-ieq[IO.Path]::GetFullPath($source)){$matches++};$ancestor=$ancestor.ParentNode};QA-Assert ($matches-eq1) 'Exact actual test source container';if($expected.Result-ceq'Passed'){QA-Assert ($case.GetAttribute('result')-ceq'Success'-and$case.GetAttribute('success')-ceq'True'-and$case.GetAttribute('executed')-ceq'True') 'Real passed physical case'}else{QA-Assert (-not$captureMode-and$expected.Name-ceq'hydrates a local cache miss from Redis when the provider is enabled'-and$case.GetAttribute('result')-ceq'Ignored'-and$case.GetAttribute('success')-ceq'False'-and$case.GetAttribute('executed')-ceq'False') 'Only declared live Redis skip'}}
 }catch{Add-DefaultsCause $_;throw}finally{Settle-DefaultsOriginals}
 @{Mode=$Mode;ResultPath=$resultPath;ResultSHA256=(Get-FileHash -LiteralPath $resultPath).Hash;XMLPath=$xmlPath;XMLSHA256=$result.Xml.SHA256;Total=$total;Passed=($total-$skipped);Skipped=$skipped;FinalOriginalSummary=$summary;OriginalXmlReaderClosed=$entry.ReaderClosed;OriginalXmlStreamClosed=$entry.StreamClosed}
}
