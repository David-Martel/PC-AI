# Definitions only. Original retained resource ledgers are never reconstructed.
function Resolve-QARepositoryRoot {[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))}

function QA-Assert([bool]$Condition,[string]$Message){if(-not$Condition){throw [IO.InvalidDataException]::new($Message)}}
function QA-Pin([string]$Path){Assert-ControlOrdinary $Path;$leaf=Get-Item -LiteralPath $Path -Force;QA-Assert ($leaf-is[IO.FileInfo]) 'Ordinary source leaf';@{Path=$Path;Bytes=$leaf.Length;SHA256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash}}
function QA-Read([string]$Path,[string]$Hash,[int]$Cap=131072){
    QA-Assert ($Hash-cmatch'^[A-F0-9]{64}$') 'Exact bounded JSON hash';$pin=QA-Pin $Path;QA-Assert ($pin.Bytes-gt0-and$pin.Bytes-le$Cap) 'Bounded JSON leaf';$bytes=[IO.File]::ReadAllBytes($Path)
    QA-Assert ($bytes.Length-le$Cap-and[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))-ceq$Hash) 'Same-byte JSON admission';[Text.UTF8Encoding]::new($false,$true).GetString($bytes)|ConvertFrom-Json -AsHashtable -DateKind String
}
function QA-Packet([string]$Hash){
    $packet=QA-Read (Join-Path $PSScriptRoot 'module-qa-source-register.json') $Hash
    QA-Assert ($packet.State-ceq'SOURCE_ONLY_PENDING_ROOT_AND_DISTINCT_REVIEW'-and$packet.Files.Count-le32-and$packet.Files.Count-ge12) 'Exact QA source packet'
    $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($row in $packet.Files){QA-Assert ($row.RelativePath-is[string]-and$row.RelativePath-cmatch'^[A-Za-z0-9_./-]+$'-and$row.RelativePath.Split('/')-cnotcontains'..'-and-not[IO.Path]::IsPathRooted($row.RelativePath)-and$seen.Add($row.RelativePath)) 'Unique safe relative source';$pin=QA-Pin (Join-Path (Resolve-QARepositoryRoot) $row.RelativePath);QA-Assert ($pin.Bytes-eq$row.Bytes-and$pin.SHA256-ceq$row.SHA256) 'QA packet source differs'}
    $moduleRoot=Join-Path (Resolve-QARepositoryRoot) 'Modules/PC-AI.Common';$expected=@($packet.Files|Where-Object{$_.RelativePath.StartsWith('Modules/PC-AI.Common/',[StringComparison]::Ordinal)})
    QA-Assert ($expected.Count-eq9) 'Nine complete candidate module leaves'
    $queue=[Collections.Generic.Queue[string]]::new();$queue.Enqueue($moduleRoot);$actual=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal);$directories=0
    while($queue.Count){$directory=$queue.Dequeue();Assert-ControlOrdinary $directory;QA-Assert (++$directories-le8) 'Bounded complete module directories';foreach($item in Get-ChildItem -LiteralPath $directory -Force){Assert-ControlOrdinary $item.FullName;if($item-is[IO.DirectoryInfo]){$queue.Enqueue($item.FullName)}elseif($item-is[IO.FileInfo]){QA-Assert ($actual.Add(('Modules/PC-AI.Common/'+[IO.Path]::GetRelativePath($moduleRoot,$item.FullName).Replace('\','/')))-and$actual.Count-le9) 'Complete module physical leaf set'}else{throw 'Unsupported module item'}}}
    QA-Assert ($actual.Count-eq9) 'No omitted/extra candidate module files';foreach($row in $expected){QA-Assert ($actual.Contains($row.RelativePath)) 'Complete candidate module path equality'}
    $packet
}
function QA-Admission([string]$Path,[string]$Hash,[string]$PacketHash){
    $data=QA-Read $Path $Hash
    QA-Assert ($data.State-ceq'ROOT_ADMITTED_CANONICAL_CACHE_MODULE_QA_ONLY'-and$data.SourceSeal-ceq$PacketHash) 'Scoped original Root admission'
    foreach($key in @('SourceReviewed','SerializedOwnerConfirmed','NoCompetingHeavyJobs','QualifiedSupervisor4','ManagedHandoffConfirmed','PriorOriginalCustodyClosed','HistoricalErrorsPreserved')){QA-Assert ($data[$key]-is[bool]-and$data[$key]) 'Explicit Root custody/serialization attestation'}
    foreach($key in @('ObservedUtc','ExpiresUtc','OriginalBirthUtc')){QA-Assert ($data[$key]-is[string]-and$data[$key]-cmatch'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z$') 'Raw UTC input'}
    $format="yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'";$culture=[Globalization.CultureInfo]::InvariantCulture;$style=[Globalization.DateTimeStyles]::AssumeUniversal
    $observed=[DateTimeOffset]::ParseExact($data.ObservedUtc,$format,$culture,$style);$expires=[DateTimeOffset]::ParseExact($data.ExpiresUtc,$format,$culture,$style);$now=[DateTimeOffset]::UtcNow
    QA-Assert ($observed-le$now-and($now-$observed).TotalSeconds-le300-and$expires-gt$now-and($expires-$observed).TotalSeconds-le300) 'Fresh five-minute admission'
    QA-Assert (($data.OriginalPID-is[int]-or$data.OriginalPID-is[long])-and$data.OriginalPID-gt0-and$data.OriginalPID-le[int]::MaxValue) 'Actual original PID'
    $peer=QA-Read $data.SourcePeerPath $data.SourcePeerSHA256
    QA-Assert ($peer.State-ceq'SOURCE_GO_FOR_SEPARATELY_ADMITTED_MAINTAINED_CACHE_DEFAULTS_ONLY'-and$peer.ModuleQaSourceRegisterSHA256-ceq$PacketHash) 'Distinct source peer exact state/seal'
    QA-Assert ($data.ProjectionSourceSeal-ceq'8E602A068F954F8D5702BC85482BAD5980690DEA6B8D7E49C7E81AC0B8D93B5E'-and$data.ProjectionPeerSHA256-ceq'B435BE67E37640134C1FA898612894EBF261F841B47B1189CC1F1B1A7B2D7FB0') 'Accepted projection lineage'
    QA-Assert ([IO.Path]::GetFullPath($data.ModuleRoot)-ieq[IO.Path]::GetFullPath((Join-Path (Resolve-QARepositoryRoot) 'Modules/PC-AI.Common'))) 'Exact complete packet module only'
    QA-Assert ($data.OwnedOutputRoot-is[string]-and$data.OwnedOutputRoot.Length-gt0) 'Explicit original parent-owned root'
    $output=[IO.Path]::GetFullPath($data.EvidenceDirectory);$owned=[IO.Path]::GetFullPath($data.OwnedOutputRoot);Assert-ControlOrdinary $owned
    QA-Assert ($output.Equals([IO.Path]::GetFullPath((Join-Path $owned 'module-qa')),[StringComparison]::OrdinalIgnoreCase)) 'Exact fresh separately owned module-QA namespace'
    QA-Assert ($data.Capacity.AvailableMiB-ge8192-and$data.Capacity.CommitPercent-lt85-and$data.Capacity.CommitHeadroomBytes-ge10GB-and$data.Capacity.OutputFreeBytes-ge10GB) 'Admitted tiny bounded capacity: 8GiB/<85%/10GiB headroom/free'
    $data
}
function QA-Closed($Operation,[switch]$AllowFailedExit){
    QA-Assert ($null-ne$Operation-and$Operation.Disposition-ceq'CLOSED'-and-not$Operation.ChildHoldRetained-and$null-eq$Operation.CustodyId) 'Original outer operation closed'
    foreach($key in @('Started','ProcessHandleCaptured','JobAssigned','ParentExitConfirmed','JobEmptyConfirmed','DrainsSettled','DrainsSucceeded','FilesFlushed','ResourcesDisposed','OriginalHandleDisposed')){QA-Assert ($Operation.$key-is[bool]-and$Operation.$key) 'All ten original outer closure flags'}
    if(-not$AllowFailedExit){QA-Assert ($Operation.ExitCode-eq0-and$null-eq$Operation.PrimaryFailure-and$Operation.SecondaryFailures.Count-eq0) 'No outer failure promotion'}
}
function QA-Cause($Cause){if($null-eq$Cause){return $null};if($Cause-is[Management.Automation.ErrorRecord]){return @{Type=$Cause.Exception.GetType().FullName;Message=$Cause.Exception.Message;ErrorId=$Cause.FullyQualifiedErrorId}};@{Type=$Cause.GetType().FullName;Message=$Cause.ToString()}}
function QA-Ledgers($Declarations){
    QA-Assert ($Declarations-is[Array]-and$Declarations.Count-ge1-and$Declarations.Count-le32) 'Root selected prior original ledger closure'
    $snapshots=[Collections.Generic.List[object]]::new()
    foreach($declared in $Declarations){
        QA-Assert ($declared.Name-is[string]-and$declared.Name-cmatch'^Pcai[A-Za-z0-9]+$'-and($declared.WriterCount-is[int]-or$declared.WriterCount-is[long])-and$declared.WriterCount-ge1-and$declared.WriterCount-le64) 'Exact current original ledger declaration'
        QA-Assert ($declared.Schema-ceq'CACHECCB') 'Only matching current CACHECCB Carbon original ledgers admitted'
        $ledger=(Get-Variable -Name $declared.Name -Scope Global -ErrorAction Stop).Value
        QA-Assert ($ledger.Writers.Count-eq$declared.WriterCount) 'No missing original writers'
        foreach($writer in $ledger.Writers){
            $handle=$writer.OriginalHandle
            QA-Assert ($writer.Closed-is[bool]-and$writer.Closed-and$null-ne$handle-and$handle.IsClosed-and$null-ne$writer.Stream) 'Original retained writer handle closed'
            QA-Assert ($writer.DisposalReturned-is[bool]-and$writer.DisposalReturned) 'Original writer disposal returned'
        }
        $snapshots.Add(@{Declaration=$declared;Original=$ledger;First=$ledger.FirstErrorRecord;Secondary=@($ledger.Secondary)})
    }
    $snapshots.ToArray()
}
function QA-LedgersUnchanged($Snapshots){foreach($snapshot in $Snapshots){$current=(Get-Variable $snapshot.Declaration.Name -Scope Global -ErrorAction Stop).Value;QA-Assert ([object]::ReferenceEquals($current,$snapshot.Original)-and[object]::ReferenceEquals($current.FirstErrorRecord,$snapshot.First)-and$current.Secondary.Count-eq$snapshot.Secondary.Count) 'Original prior failure references preserved';for($i=0;$i-lt$snapshot.Secondary.Count;$i++){QA-Assert ([object]::ReferenceEquals($current.Secondary[$i],$snapshot.Secondary[$i])) 'Original secondary unchanged'};[void](QA-Ledgers @($snapshot.Declaration))}}

function Add-ControlCause([object]$Cause){if($null-eq$Cause){return};if([object]::ReferenceEquals($state.First,$Cause)){return};foreach($original in $state.Secondary){if([object]::ReferenceEquals($original,$Cause)){return}};if($null-eq$state.First){$state.First=$Cause}elseif($state.Secondary.Count-lt64){$state.Secondary.Add($Cause)}else{$state.RepeatedSecondaryFailures++}}

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
