#Requires -Version 7.0
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-ThunderboltEnrolledPeer.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $SourcePath)) { throw 'Partner-bound Windows reconciliation is missing.' }
. $SourcePath
function Assert-Refused([scriptblock]$Operation) {
    $refused = $false
    try { & $Operation | Out-Null } catch { $refused = $true }
    if (-not $refused) { throw 'Unsafe enrolled edge was admitted.' }
}
$now = [DateTimeOffset]::Parse('2026-01-01T00:00:30Z')
$node = '11111111-1111-1111-1111-111111111111'
$peer = @{ node_id = '22222222-2222-2222-2222-222222222222'; controller_uuids = @('33333333-3333-3333-3333-333333333333');
    minimum_epoch = 1; host_key_alias = 'fixture-peer'; ssh_user = 'fixture'; identity_file = 'fixture-key'; known_hosts_file = 'fixture-hosts';
    local_root_ids = @('USB4\ROOT_DEVICE_ROUTER\fixture'); ula_pool = 'fd00:7462:7032::/48' }
$edge = @{ edge_id = ('usb4-' + ('a' * 64)); peer_node_id = $node; local_controller_uuid = $peer.controller_uuids[0];
    remote_controller_uuid = '44444444-4444-4444-4444-444444444444'; local_ula = 'fd00:7462:7032:1234::2';
    peer_ula = 'fd00:7462:7032:1234::1'; prefix_len = 64; mtu = 1500 }
$observation = @{ schema = 1; epoch = 1; policy_sha256 = ('b' * 64); boot_id = '55555555-5555-5555-5555-555555555555';
    observed_at = '2026-01-01T00:00:20Z'; namespace = 'net:[42]'; namespace_inventory_complete = $false;
    edges = @(@{ state = 'ENROLLED_UNVERIFIED'; ifindex = 342; interface = 'renamed-peer'; edge = $edge }) }
$frame = @{ node_id = $peer.node_id; connection = @('fe80::1%renamed-peer','12345','fe80::2%renamed-peer','22');
    observation = $observation; addresses = @(@{ ifindex = 342; ifname = 'renamed-peer'; address = '02:00:00:00:00:02'; addr_info = @(@{family='inet6';local='fe80::2'}) }) }
function Get-Proof($value = $frame, $previous = @{}) {
    Confirm-Usb4PartnerProof -Frame $value -Peer $peer -LocalNodeId $node -LocalAddress 'fe80::1' -PeerAddress 'fe80::2' -Previous $previous -Now $now -TtlSeconds 90
}
# Selector: FreshSocketBoundEnrolledEdgePasses. Protects actual partner selection;
# detects alias/ifindex assumptions and global-ready promotion. Needs fixtures.
# Breadcrumb: Confirm-Usb4PartnerProof, shared Linux observe-current schema.
$proof = Get-Proof
if ($proof.windows_address -ne $edge.peer_ula -or $proof.peer_address -ne $edge.local_ula -or $proof.scope -ne 'current-root-namespace-edge') { throw 'Fresh partner proof did not retain scoped derived pair.' }
Write-Output 'PASS: FreshSocketBoundEnrolledEdgePasses'
# Selector: WrongSocketNodeOrControllerRefused. Protects pin/socket/node binding;
# detects another interface, peer or unapproved hardware being promoted.
# Needs fixture clones; breadcrumb: Confirm-Usb4PartnerProof identity guards.
$wrong = $frame.Clone(); $wrong.node_id = $node; Assert-Refused { Get-Proof $wrong }
$wrong = $frame.Clone(); $wrong.connection = @('fe80::9','12345','fe80::2','22'); Assert-Refused { Get-Proof $wrong }
$wrong = $frame.Clone(); $wrong.connection = @('fe80::1','12345','fe80::2','80'); Assert-Refused { Get-Proof $wrong }
$badEdge = $edge.Clone(); $badEdge.local_controller_uuid = '66666666-6666-6666-6666-666666666666'
$wrongObservation = $observation.Clone(); $wrongObservation.edges = @(@{state='ENROLLED_UNVERIFIED';ifindex=342;interface='renamed-peer';edge=$badEdge})
$wrong = $frame.Clone(); $wrong.observation = $wrongObservation; Assert-Refused { Get-Proof $wrong }
Write-Output 'PASS: WrongSocketNodeOrControllerRefused'
# Selector: StaleFutureAndEpochReplayRefused. Protects reconnect/lease freshness;
# detects a replay, same-epoch policy substitution and rollback. Needs frozen clock.
# Breadcrumb: Confirm-Usb4PartnerProof freshness and monotonic peer state.
$old = $observation.Clone(); $old.observed_at = '2025-12-31T23:55:00Z'
$wrong = $frame.Clone(); $wrong.observation = $old; Assert-Refused { Get-Proof $wrong }
$future = $observation.Clone(); $future.observed_at = '2026-01-01T00:05:00Z'
$wrong = $frame.Clone(); $wrong.observation = $future; Assert-Refused { Get-Proof $wrong }
Assert-Refused { Get-Proof $frame @{epoch=2;policy_sha256=('b'*64)} }
Assert-Refused { Get-Proof $frame @{epoch=1;policy_sha256=('c'*64)} }
Write-Output 'PASS: StaleFutureAndEpochReplayRefused'
# Selector: AmbiguousAndForeignAddressPlansRefused. Protects multihomed sources;
# detects ambiguous inventory, foreign interface allocation and broad address pools.
# Needs fixture data; breadcrumb: Confirm-Usb4PartnerProof and Get-Usb4AddressDisposition.
$wrong = $frame.Clone(); $dup = $observation.Clone(); $dup.edges = @($observation.edges[0],$observation.edges[0]); $wrong.observation = $dup
Assert-Refused { Get-Proof $wrong }
$outside=$edge.Clone();$outside.peer_ula='fd01:7462:7032:1234::1'
$outsideObservation=$observation.Clone();$outsideObservation.edges=@(@{state='ENROLLED_UNVERIFIED';ifindex=342;interface='renamed-peer';edge=$outside})
$wrong=$frame.Clone();$wrong.observation=$outsideObservation;Assert-Refused { Get-Proof $wrong }
$adapter = @{ interface_guid='77777777-7777-7777-7777-777777777777'; interface_index=28; mac='00-00-00-00-00-01'; root_id=$peer.local_root_ids[0] }
$foreign = @(@{IPAddress=$edge.peer_ula;PrefixLength=64;InterfaceIndex=99})
Assert-Refused { Get-Usb4AddressDisposition $proof $adapter $foreign @() @() }
$existing = @(@{IPAddress=$edge.peer_ula;PrefixLength=64;InterfaceIndex=28})
if ((Get-Usb4AddressDisposition $proof $adapter $existing @() @()).action -ne 'PreserveUnowned') { throw 'Preexisting address silently adopted.' }
if ((Get-Usb4AddressDisposition $proof $adapter @() @() @()).action -ne 'AddOwned') { throw 'Clean reenrolled interface cannot recover.' }
Write-Output 'PASS: AmbiguousAndForeignAddressPlansRefused'
# Selector: ExpiryDeletesOnlyCommittedExactCustody. Protects foreign and pending
# addresses; detects cleanup after GUID/MAC drift or incomplete writes.
# Needs fixtures; breadcrumb: Get-Usb4ExpiredAddressPlan, no live network mutation.
$owned = @{owned=$true;committed=$true;edge_id=$edge.edge_id;interface_guid=$adapter.interface_guid;interface_index=28;mac=$adapter.mac;root_id=$adapter.root_id;address=$edge.peer_ula;prefix_length=64;expires_at='2026-01-01T00:00:00Z'}
if (@(Get-Usb4ExpiredAddressPlan @($owned) @($adapter) $existing $now).Count -ne 1) { throw 'Expired exact custody was not withdrawn.' }
$unowned=$owned.Clone();$unowned.owned=$false
$pending=$owned.Clone();$pending.committed=$false
$changed=$adapter.Clone();$changed.interface_guid='88888888-8888-8888-8888-888888888888'
if (@(Get-Usb4ExpiredAddressPlan @($unowned,$pending) @($adapter) $existing $now).Count -ne 0 -or
    @(Get-Usb4ExpiredAddressPlan @($owned) @($changed) $existing $now).Count -ne 0) { throw 'Foreign or pending custody admitted deletion.' }
Write-Output 'PASS: ExpiryDeletesOnlyCommittedExactCustody'
# Selector: BootstrapTransportCannotFallBack. Protects actual USB4 bootstrap;
# detects proxy/multiplex reuse, missing pin, source scope or interactive launch.
# Needs string argument construction only; breadcrumb: Get-Usb4PartnerSshArguments.
$transport=@(Get-Usb4PartnerSshArguments $peer 'fe80::1' 'fe80::2' 28)
foreach($required in @('ProxyCommand=none','ProxyJump=none','ControlMaster=no','ControlPath=none','BatchMode=yes','StrictHostKeyChecking=yes','HostKeyAlias=fixture-peer','IdentityAgent=none')) {
    if ($required -notin $transport) { throw "Missing transport guard $required" }
}
if ($transport[[Array]::IndexOf($transport,'-b')+1] -ne 'fe80::1%28' -or 'fixture@fe80::2%28' -notin $transport) { throw 'Bootstrap lost exact interface scope.' }
Write-Output 'PASS: BootstrapTransportCannotFallBack'

# Selector: RuntimeProbeParsesSocketAndChecksRouteCounters. Protects the actual
# probe/parser integration; detects malformed frames, gateway fallback and missing
# bidirectional traffic. Needs mocked native process and host APIs, no real SSH.
# Breadcrumb: Invoke-Usb4PartnerProbe, ConvertFrom-Usb4PartnerFrame, direct route guard.
$currentObservation=$observation.Clone();$currentObservation.observed_at=[DateTimeOffset]::UtcNow.ToString('o')
$script:probeFrame="PCAI_NODE`n$($peer.node_id)`nPCAI_CONNECTION`n$($frame.connection -join ' ')`nPCAI_OBSERVATION`n$($currentObservation|ConvertTo-Json -Compress -Depth 20)`nPCAI_ADDRESSES`n$($frame.addresses|ConvertTo-Json -Compress -Depth 20)`n"
$parsed=ConvertFrom-Usb4PartnerFrame $script:probeFrame.Replace("`n","`r`n")
if ($parsed.node_id -ne $peer.node_id) { throw 'CRLF partner frame was not normalized.' }
Assert-Refused { ConvertFrom-Usb4PartnerFrame $script:probeFrame.Replace('PCAI_ADDRESSES','DROPPED') }
$script:stats=0;$script:gateway=$false;$script:flatCounters=$false
function Find-NetRoute { param($LocalIPAddress,$RemoteIPAddress,$InterfaceIndex,$ErrorAction)
    [pscustomobject]@{InterfaceIndex=$InterfaceIndex;IPAddress=$LocalIPAddress}
    [pscustomobject]@{InterfaceIndex=$InterfaceIndex;DestinationPrefix='fe80::/64';NextHop=$(if($script:gateway){'fe80::99'}else{'::'})}
}
function Get-NetAdapterStatistics { param($Name,$ErrorAction)
    if (-not $script:flatCounters) {$script:stats++}
    @{ReceivedBytes=$script:stats;SentBytes=$script:stats}
}
function Invoke-VigilBoundedProcess { param($FilePath,$Arguments,$TimeoutSeconds)
    if ($TimeoutSeconds -ne 6 -or 'ProxyCommand=none' -notin $Arguments -or 'fixture@fe80::2%28' -notin $Arguments) { throw 'Actual probe lost transport controls.' }
    @{ExitCode=0;Stdout=$script:probeFrame;Stderr=''}
}
$script:probeAdapter=$adapter.Clone();$script:probeAdapter.name='fixture';$script:probeAdapter.up=$true
function Get-Usb4HostSnapshot { @{adapters=@($script:probeAdapter)} }
$savedSystemRoot=$env:SystemRoot
try {
    # The process boundary is mocked: Linux needs a synthetic path root only.
    # Production invocation remains Windows-only and uses the real SystemRoot.
    if (-not $IsWindows) { $env:SystemRoot=[IO.Path]::GetTempPath() }
    $actual=Invoke-Usb4PartnerProbe $peer $script:probeAdapter 'fe80::1' 'fe80::2' $node @{} 90
    if ($actual.edge_id -ne $edge.edge_id) { throw 'Actual probe did not retain enrolled edge.' }
    $script:gateway=$true
    Assert-Refused { Invoke-Usb4PartnerProbe $peer $script:probeAdapter 'fe80::1' 'fe80::2' $node @{} 90 }
    $script:gateway=$false;$script:flatCounters=$true
    Assert-Refused { Invoke-Usb4PartnerProbe $peer $script:probeAdapter 'fe80::1' 'fe80::2' $node @{} 90 }
} finally { $env:SystemRoot=$savedSystemRoot }
Write-Output 'PASS: RuntimeProbeParsesSocketAndChecksRouteCounters'

# Selector: RuntimePlanAndReenumeration. Protects the real reconcile control flow;
# detects DryRun writes and ifIndex/alias coupling. Needs mocked host APIs only.
# Breadcrumb: Invoke-Usb4PartnerReconciliation journal and address mutation calls.
$script:fixtureProof=$proof
$script:fixtureAdapter=$adapter.Clone();$script:fixtureAdapter.up=$true
$script:fixtureAdapter.name='renamed-windows';$script:fixtureAdapter.local_llas=@('fe80::1')
$script:fixtureAdapter.neighbors=@(@{IPAddress='fe80::2'})
$script:fixtureAddresses=@();$script:fixturePersistent=@();$script:writes=0;$script:adds=0;$script:removes=0
$script:failCanary=$false;$script:noOpDelete=$false;$script:saved=@{}
function Get-Usb4HostSnapshot { @{adapters=@($script:fixtureAdapter);addresses=@($script:fixtureAddresses);persistent_addresses=@($script:fixturePersistent)} }
function Invoke-Usb4PartnerProbe {
    param($Peer,$Adapter,$LocalAddress,$PeerAddress,$LocalNodeId,$Previous,$TtlSeconds)
    if ($script:failCanary -and $LocalAddress -notlike 'fe80:*') { throw 'fixture canary failed' }
    $script:fixtureProof
}
function Save-Usb4OwnedState { param($Path,$State)
    $script:writes++;$script:saved=$State|ConvertTo-Json -Depth 30|ConvertFrom-Json -AsHashtable
}
function New-NetIPAddress { param($InterfaceIndex,$IPAddress,$PrefixLength,$PolicyStore,$ErrorAction)
    if ($PolicyStore -ne 'ActiveStore') { throw 'Persistent mutation attempted' }
    $script:adds++;$script:fixtureAddresses+=@(@{InterfaceIndex=$InterfaceIndex;IPAddress=$IPAddress;PrefixLength=$PrefixLength;AddressState='Preferred'})
}
function Get-NetIPAddress { param($InterfaceIndex,$AddressFamily,$PolicyStore,$ErrorAction)
    @($script:fixtureAddresses|Where-Object InterfaceIndex -EQ $InterfaceIndex)
}
function Remove-NetIPAddress { param($InterfaceIndex,$IPAddress,$PolicyStore,$Confirm,$ErrorAction)
    if ($PolicyStore -ne 'ActiveStore') { throw 'Persistent deletion attempted' }
    $script:removes++
    if (-not $script:noOpDelete) { $script:fixtureAddresses=@($script:fixtureAddresses|Where-Object {-not($_.InterfaceIndex -eq $InterfaceIndex -and $_.IPAddress -eq $IPAddress)}) }
}
$policy=@{schema=1;node_id=$node;ttl_seconds=90;peers=@($peer);adoptions=@()}
$fixtureState=Join-Path ([IO.Path]::GetTempPath()) ('usb4-fixture-'+[guid]::NewGuid().ToString('N')+'.json')
$result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply -DryRun
if ($result.results[0].state -ne 'PLANNED' -or $writes -or $adds -or $removes) { throw 'DryRun mutated or failed to plan.' }
$result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply
if ($result.results[0].state -ne 'AUTHENTICATED' -or $adds -ne 1 -or $saved.addresses.Count -ne 1 -or -not $saved.addresses[0].committed -or -not $saved.addresses[0].qualified) { throw 'Runtime add/canary/journal qualification failed.' }
$reindexed=$adapter.Clone();$reindexed.interface_index=87
$sameOwned=$owned.Clone();$sameOwned.expires_at=[DateTimeOffset]::UtcNow.AddMinutes(1).ToString('o')
$moved=@(@{IPAddress=$edge.peer_ula;PrefixLength=64;InterfaceIndex=87})
if ((Get-Usb4AddressDisposition $proof $reindexed $moved @($sameOwned) @()).action -ne 'RefreshOwned') { throw 'Same GUID reindex lost exact custody.' }
$newIncarnation=$changed.Clone();$newIncarnation.interface_index=87
Assert-Refused { Get-Usb4AddressDisposition $proof $newIncarnation $existing @($sameOwned) @() }
Write-Output 'PASS: RuntimePlanAndReenumeration'

# Selector: RuntimeRollbackRequiresPositiveAbsence. Protects failed canary cleanup;
# detects silent no-op removal and forgotten committed custody. Needs host mocks.
# Breadcrumb: reconcile catch/rollback path, Save-Usb4OwnedState intent barrier.
$script:fixtureAddresses=@();$script:writes=0;$script:adds=0;$script:removes=0;$script:failCanary=$true
$result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply
if ($result.results[0].state -ne 'REFUSED' -or $adds -ne 1 -or $removes -ne 1 -or $saved.addresses.Count -ne 0) { throw 'Failed canary leaked its new address.' }
$script:fixtureAddresses=@();$script:noOpDelete=$true
Assert-Refused { Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply }
if ($saved.addresses.Count -ne 1 -or -not $saved.addresses[0].committed) { throw 'No-op rollback forgot live custody.' }
Write-Output 'PASS: RuntimeRollbackRequiresPositiveAbsence'

# Selector: PersistentCustodyIsNeverLeasedImplicitly. Protects reboot semantics;
# detects an Active-only receipt hiding a persistent address. Needs host mocks.
# Breadcrumb: reconcile persistent-store gate before disposition/mutation.
$script:fixturePersistent=@(@{IPAddress=$edge.peer_ula;InterfaceIndex=28;PrefixLength=64})
$script:fixtureAddresses=@($script:fixturePersistent);$script:adds=0;$script:removes=0;$script:failCanary=$false
$result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply
if ($result.results[0].state -ne 'REFUSED' -or $adds -or $removes -or $result.results[0].error -notlike '*PersistentStore*') { throw 'Persistent address silently admitted lease custody.' }
Write-Output 'PASS: PersistentCustodyIsNeverLeasedImplicitly'

# Selector: RuntimeRebootRecoveryRetainsSingleCustody. Protects persisted leases
# after ActiveStore loss; detects duplicate journals that wedge later expiry.
# Needs persisted fixture journal and host mocks; breadcrumb: AddOwned prior-row reuse.
function Assert-Usb4AdministrativeFile { param($Path) }
$script:fixturePersistent=@();$script:fixtureAddresses=@();$script:noOpDelete=$false
$script:adds=0;$script:removes=0
$prior=$owned.Clone();$prior.qualified=$true;$prior.peer_node_id=$peer.node_id
$journal=@{schema=1;node_id=$node;peers=@{};addresses=@($prior);results=@()}
try {
    [IO.File]::WriteAllText($fixtureState,($journal|ConvertTo-Json -Depth 20))
    $result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply
    if ($result.results[0].state -ne 'AUTHENTICATED' -or $saved.addresses.Count -ne 1 -or $adds -ne 1) { throw 'ActiveStore loss multiplied or failed custody recovery.' }
    $saved.addresses[0].expires_at=[DateTimeOffset]::UtcNow.AddSeconds(-1).ToString('o')
    [IO.File]::WriteAllText($fixtureState,($saved|ConvertTo-Json -Depth 20))
    $result=Invoke-Usb4PartnerReconciliation $policy $fixtureState -Apply
    if ($result.results[0].state -ne 'AUTHENTICATED' -or $saved.addresses.Count -ne 1 -or $removes -ne 1 -or $adds -ne 2) { throw 'Recovered custody could not expire and reenroll.' }
} finally { if ([IO.File]::Exists($fixtureState)) { [IO.File]::Delete($fixtureState) } }
Write-Output 'PASS: RuntimeRebootRecoveryRetainsSingleCustody'

# Selector: RuntimeCandidateBudgetIsGlobal. Protects roaming latency and scheduler
# capacity; detects peer/adapter loop multiplication beyond four attempts.
# Needs mocked failed probes; breadcrumb: reconcile global attempt/deadline guards.
$script:fixturePersistent=@();$script:fixtureAddresses=@();$script:probeCalls=0
function Invoke-Usb4PartnerProbe { param($Peer,$Adapter,$LocalAddress,$PeerAddress,$LocalNodeId,$Previous,$TtlSeconds)
    $script:probeCalls++;throw 'fixture unavailable partner'
}
$script:fixtureAdapter.neighbors=@(@{IPAddress='fe80::2'},@{IPAddress='fe80::3'},@{IPAddress='fe80::4'},@{IPAddress='fe80::5'})
$many=$policy.Clone();$many.peers=@($peer,$peer,$peer,$peer,$peer,$peer,$peer,$peer)
$result=Invoke-Usb4PartnerReconciliation $many $fixtureState -DryRun
if ($probeCalls -ne 4 -or $result.results.Count -ne 4) { throw 'Candidate budget multiplied across peers/adapters.' }
Write-Output 'PASS: RuntimeCandidateBudgetIsGlobal'
