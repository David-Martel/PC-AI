#Requires -Version 7.0
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath
$script:checks = 0

function Assert-Check([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Assert-Refused([scriptblock]$Operation, [string]$Message) {
    $refused = $false
    try { & $Operation | Out-Null } catch { $refused = $true }
    Assert-Check $refused $Message
}
function Initialize-Fixture([string]$Mode = 'Peer') {
    $script:policy = @{ schemaVersion = 1; hostName = 'fixture-host'; mode = $Mode; adapterMac = 'C0-94-EB-14-4D-CA';
        peerMac = '08-93-C1-7B-39-51'; peerLinkLocal = 'fe80::d700:9'; localIPv4 = '172.31.252.133'; peerIPv4 = '172.31.252.134'; prefixLength = 30;
        localUla = 'fd00:7462:7032::7'; peerUla = 'fd00:7462:7032::9'
    }
    if ($Mode -ne 'Peer') {
        $script:policy.sharing = @{ providerAddress = '192.168.137.1'; consumerAddress = '192.168.137.2'; prefixLength = 24;
            preferSharedInternet = $true; routeMetric = 10; probeAddress = '1.1.1.1'; probePort = 443; allowWirelessFallback = $false
        }
    }
    $usb = @{ mac = 'C094EB144DCA'; guid = '11111111-1111-1111-1111-111111111111'; index = 31; name = 'renumberable-usb4';
        description = 'USB4(TM) P2P Network Adapter'; up = $true; hardware = $false; wired = $false; wireless = $false;
        metric = 5; mtu = 62000; dhcp = 'Disabled'; weakHostSend = 'Disabled'; weakHostReceive = 'Disabled'; forwarding = 'Disabled';
        addresses = @(); dns = @('8.8.8.8'); dns6 = @('2001:4860:4860::8888'); dnsMode = 'Dhcp'
    }
    $wiredAddress = @{ address = '192.168.50.42'; prefix = 24; index = 30; family = 'IPv4'; state = 'Preferred'; prefixOrigin = 'Manual'; suffixOrigin = 'Manual'; skipAsSource = $false }
    $wired = @{ mac = '8CAE4CBCFE12'; guid = '22222222-2222-2222-2222-222222222222'; index = 30; name = 'approved-wired';
        description = 'Realtek USB 2.5GbE'; up = $true; hardware = $true; wired = $true; wireless = $false;
        metric = 10; mtu = 1500; dhcp = 'Enabled'; weakHostSend = 'Disabled'; weakHostReceive = 'Disabled'; forwarding = 'Disabled';
        addresses = @($wiredAddress); dns = @('192.168.50.1'); dns6 = @(); dnsMode = 'Dhcp'
    }
    $wslAddress = @{ address = '172.16.0.1'; prefix = 16; index = 89; family = 'IPv4'; state = 'Preferred'; prefixOrigin = 'Manual'; suffixOrigin = 'Manual'; skipAsSource = $false }
    $wsl = @{ mac = '00155D123456'; guid = '33333333-3333-3333-3333-333333333333'; index = 89; name = 'vEthernet WSL';
        description = 'Hyper-V Virtual Ethernet Adapter'; up = $true; hardware = $false; wired = $false; wireless = $false;
        metric = 5000; mtu = 1500; dhcp = 'Disabled'; weakHostSend = 'Disabled'; weakHostReceive = 'Disabled'; forwarding = 'Disabled';
        addresses = @($wslAddress); dns = @(); dns6 = @(); dnsMode = 'Dhcp'
    }
    $script:model = @{ hostName = 'fixture-host'; adapters = @($usb, $wired, $wsl); addresses = @($wiredAddress, $wslAddress);
        routes = @(@{ destination = '0.0.0.0/0'; nextHop = '192.168.50.1'; index = 30; metric = 10 }); sharing = @(); nat = @(@{ name = 'WSL'; prefix = '172.16.0.0/16' })
    }
    $script:calls = [Collections.Generic.List[hashtable]]::new()
    $script:peerHealthy = $true; $script:internetHealthy = $true; $script:dnsHealthy = $true; $script:transportHealthy = $true
    $script:peerMac = '0893C17B3951'; $script:failOperation = ''; $script:failAfterMutation = $false; $script:failRollback = $false
    $script:corruptUnrelated = $false; $script:checkpointCalls = 0; $script:lastCheckpoint = @{}
    $script:noOpOperation = ''
    $script:timeline = [Collections.Generic.List[hashtable]]::new()
    $script:failureReports = [Collections.Generic.List[hashtable]]::new()
    $script:failCommit = $false
    $script:natInventory = $null
    $script:automaticDns6Fallback = $false
    $script:unknownRoutingMetricIndex = -1
}
function Initialize-NatFixture([string]$Mode = 'Provider') {
    Initialize-Fixture $Mode
    $script:policy.sharing.backend = 'ExistingWinNat'; $script:policy.sharing.natName = 'WSL'; $script:policy.sharing.natPrefix = '172.16.0.0/16'
    $script:policy.sharing.providerAddress = '172.16.252.1'; $script:policy.sharing.consumerAddress = '172.16.252.2'; $script:policy.sharing.prefixLength = 30
    $script:policy.sharing.dnsServers = @('1.1.1.1')
    @($script:model.addresses | Where-Object index -EQ 89)[0].address = '192.168.240.1'
    @($script:model.addresses | Where-Object index -EQ 89)[0].prefix = 20
    $script:natInventory = @{ complete = $true; nat = @(@{ Name = 'WSL'; InternalIPInterfaceAddressPrefix = '172.16.0.0/16'; ExternalIPInterfaceAddressPrefix = $null; Active = 1;
                IcmpQueryTimeout = 30; TcpEstablishedConnectionTimeout = 1800; TcpTransientConnectionTimeout = 120; TcpFilteringBehavior = 1; UdpFilteringBehavior = 1; UdpIdleSessionTimeout = 120; UdpInboundRefresh = 0; Store = 0
            });
        networks = @(@{ id = 'wsl'; type = 'ICS'; prefixes = @('172.26.32.0/20') }, @{ id = 'firewall-wsl'; type = 'ICS'; prefixes = @('192.168.240.0/20') }, @{ id = 'default'; type = 'ICS'; prefixes = @('172.20.192.0/20') });
        endpoints = @(@{ id = 'endpoint'; addresses = @('192.168.247.82') }); mappings = @(); sessions = @()
    }
}
function Add-FixtureAddress([hashtable]$Arguments) {
    $entry = @{ address = $Arguments.address; prefix = $Arguments.prefix; index = $Arguments.index; family = $(if ($Arguments.address.Contains(':')) { 'IPv6' } else { 'IPv4' }); state = 'Preferred'; prefixOrigin = 'Manual'; suffixOrigin = 'Manual'; skipAsSource = $false }
    $script:model.addresses += $entry
    $adapter = @($script:model.adapters | Where-Object index -EQ $Arguments.index)[0]
    $adapter.addresses += $entry
}
function Get-FixtureSnapshot {
    $snapshot = Copy-Usb4NetworkValue $script:model
    $snapshot.routingInterfaces = @(foreach ($a in $snapshot.adapters) {
            foreach ($family in @('IPv4', 'IPv6')) {
                @{ index = $a.index; family = $family; metric = $(if ($a.index -eq $script:unknownRoutingMetricIndex) { $null } else { $a.metric });
                    state = $(if ($a.up) { 'Connected' } else { 'Disconnected' })
                }
            }
        })
    $snapshot
}
function Invoke-FixtureOperation {
    param([string]$Name, [hashtable]$Arguments)
    $script:calls.Add(@{ name = $Name; arguments = Copy-Usb4NetworkValue $Arguments })
    $script:timeline.Add(@{ kind = 'Operation'; name = $Name; pendingName = $(if ($script:lastCheckpoint.ContainsKey('pendingTransaction') -and $script:lastCheckpoint.pendingTransaction.ContainsKey('lastOperation')) { $script:lastCheckpoint.pendingTransaction.lastOperation.name } else { $null }) })
    if ($script:failOperation -eq $Name -and -not $script:failAfterMutation) { throw "Injected $Name failure before mutation." }
    if ($script:noOpOperation -eq $Name) { return }
    switch ($Name) {
        { $_ -in @('Snapshot', 'ReadAddressState', 'ReadRouteState', 'ReadAdapterState', 'ReadSharingState') } { return Get-FixtureSnapshot }
        'ReadExistingNat' { if ($null -eq $script:natInventory) { throw 'Unexpected NAT inventory dependency.' }; return Copy-Usb4NetworkValue $script:natInventory }
        'ProbePeer' {
            if (-not $script:peerHealthy) { return @{ success = $false; error = 'Peer disconnected.' } }
            return @{ success = $true; interfaceIndex = $Arguments.adapter.index; peerMac = $script:peerMac; peerAddress = "$($Arguments.address)%$($Arguments.adapter.index)" }
        }
        'ProbeTransport' {
            if (-not $script:transportHealthy -or @($script:model.addresses | Where-Object { $_.address -eq $Arguments.localIPv4 -and $_.index -eq $Arguments.adapter.index }).Count -ne 1) { return @{ success = $false; error = 'No qualified private IPv4 transport.' } }
            return @{ success = $true; source = $Arguments.localIPv4; peerAddress = $Arguments.peerIPv4; interfaceIndex = $Arguments.adapter.index; peerMac = $Arguments.peerMac }
        }
        'ProbeInternet' { return @{ success = $script:internetHealthy; source = $Arguments.source; remote = $Arguments.remote; tlsAuthenticated = $script:internetHealthy; httpStatus = $(if ($script:internetHealthy) { 301 } else { 0 }) } }
        'ProbeDns' { return @{ success = $script:dnsHealthy; source = $Arguments.source; server = $Arguments.server } }
        'AddAddress' { Add-FixtureAddress $Arguments }
        'RemoveAddress' {
            if ($script:failRollback) { throw 'Injected rollback refusal.' }
            $script:model.addresses = @($script:model.addresses | Where-Object { $_.address -ne $Arguments.address -or $_.index -ne $Arguments.index })
            $a = @($script:model.adapters | Where-Object index -EQ $Arguments.index)[0]
            $a.addresses = @($a.addresses | Where-Object address -NE $Arguments.address)
        }
        'AddRoute' { $script:model.routes += Copy-Usb4NetworkValue $Arguments }
        'RemoveRoute' {
            if ($script:failRollback) { throw 'Injected rollback refusal.' }
            $script:model.routes = @($script:model.routes | Where-Object { $_.destination -ne $Arguments.destination -or $_.index -ne $Arguments.index -or $_.nextHop -ne $Arguments.nextHop -or $_.metric -ne $Arguments.metric })
        }
        'SetDns' {
            $a = @($script:model.adapters | Where-Object index -EQ $Arguments.index)[0]
            $a.dns = @($Arguments.servers); $a.dnsMode = $Arguments.mode
            if ($script:automaticDns6Fallback) {
                if ($a.dns.Count) { $a.dns6 = @() } else { $a.dns6 = @('fec0:0:0:ffff::1', 'fec0:0:0:ffff::2', 'fec0:0:0:ffff::3') }
            }
        }
        'RestoreDhcp' { @($script:model.adapters | Where-Object guid -EQ $Arguments.guid)[0].dhcp = $Arguments.mode }
        'SetForwarding' {
            $a = @($script:model.adapters | Where-Object index -EQ $Arguments.index)[0]
            if ($a.guid -cne $Arguments.expectedAdapter.guid -or $a.mac -cne $Arguments.expectedAdapter.mac) { throw 'Fixture forwarding adapter incarnation changed.' }
            if ($Arguments.ContainsKey('allowAlreadyDesired') -and $Arguments.allowAlreadyDesired -and $a.forwarding -eq $Arguments.mode) { return }
            if ($a.forwarding -cne $Arguments.expectedCurrent) { throw 'Fixture forwarding custody changed.' }
            $a.forwarding = $Arguments.mode
        }
        'SetSharing' {
            $actual = @($script:model.sharing | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|'
            $expected = @($Arguments.expected | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|'
            if ($actual -cne $expected) { throw 'Fixture sharing preimage changed.' }
            $script:model.sharing = @()
            if ($Arguments.publicGuid) {
                $script:model.sharing = @(@{ guid = $Arguments.publicGuid; type = 0 }, @{ guid = $Arguments.privateGuid; type = 1 })
                $a = @($script:model.adapters | Where-Object guid -EQ $Arguments.privateGuid)[0]
                $script:model.addresses = @($script:model.addresses | Where-Object index -NE $a.index); $a.addresses = @()
                Add-FixtureAddress @{ index = $a.index; address = '192.168.137.1'; prefix = 24 }
                if ($script:corruptUnrelated) { $script:model.addresses[0].address = '192.168.50.99' }
            }
        }
        default { throw "Fixture received unexpected operation $Name." }
    }
    if ($script:failOperation -eq $Name -and $script:failAfterMutation) { throw "Injected $Name failure after mutation." }
}
function Save-FixtureCheckpoint([hashtable]$State) {
    if ($script:failCommit -and $State.Count -gt 0 -and -not $State.ContainsKey('pendingTransaction')) { $script:failCommit = $false; throw 'Injected final-state commit refusal.' }
    $script:checkpointCalls++; $script:lastCheckpoint = Copy-Usb4NetworkValue $State
    $script:timeline.Add(@{ kind = 'Checkpoint'; firstFailure = $(if ($State.ContainsKey('pendingTransaction') -and $State.pendingTransaction.ContainsKey('firstFailure')) { Copy-Usb4NetworkValue $State.pendingTransaction.firstFailure } else { $null }) })
}
function Invoke-Fixture([hashtable]$Previous = @{}, [switch]$Apply) {
    Invoke-Usb4NetworkReconcile -Policy $script:policy -Snapshot (Get-FixtureSnapshot) -PreviousState $Previous -Operation ${function:Invoke-FixtureOperation} -StateCheckpoint ${function:Save-FixtureCheckpoint} -ReportCheckpoint { param($Report); $script:failureReports.Add((Copy-Usb4NetworkValue $Report)); $script:timeline.Add(@{ kind = 'FailureReport'; firstFailure = Copy-Usb4NetworkValue $Report.firstFailure }) } -Apply:$Apply
}
function Get-FixtureMutation {
    @($script:calls | Where-Object { $_.name -notin @('Snapshot', 'ReadAddressState', 'ReadRouteState', 'ReadAdapterState', 'ReadSharingState', 'ReadExistingNat', 'ProbePeer', 'ProbeTransport', 'ProbeInternet', 'ProbeDns') })
}

# Real validation rejects malformed, cross-host, unknown, or unsafe policy inputs.
Initialize-Fixture
$bad = Copy-Usb4NetworkValue $policy; $bad.localIPv4 = '169.254.1.1'
Assert-Refused { Assert-Usb4NetworkPolicy $bad } 'APIPA was admitted for routed transport.'
$bad = Copy-Usb4NetworkValue $policy; $bad.peerMac = $bad.adapterMac
Assert-Refused { Assert-Usb4NetworkPolicy $bad } 'Self-peer loop was admitted.'
$bad = Copy-Usb4NetworkValue $policy; $bad.unhandledParameter = 1
Assert-Refused { Assert-Usb4NetworkPolicy $bad } 'An unknown policy parameter was ignored.'
Assert-Refused { Assert-Usb4NetworkPolicy $policy -CurrentHost wrong-host } 'Cross-host policy was admitted.'
Write-Output 'PASS: ExplicitPolicyValidation'

Initialize-Fixture
$before = Get-Usb4NetworkFingerprint $model
$r = Invoke-Fixture
Assert-Check ($r.exitCode -eq 2 -and $r.linkLocalReady -and -not $r.ipv4Ready -and -not $r.networkReady) 'Plan claimed unconfigured IPv4 readiness.'
Assert-Check (@(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0 -and (Get-Usb4NetworkFingerprint $model) -ceq $before) 'Plan mutated network or persistent state.'
Write-Output 'PASS: NoWritePlanAndObservedReadiness'

Initialize-Fixture
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.peerReady -and $r.ipv4Ready -and $r.networkReady -and -not $r.internetReady) 'Peer apply failed real transport qualification.'
Assert-Check (@($model.addresses | Where-Object address -EQ $policy.localIPv4).Count -eq 1 -and @($model.routes | Where-Object destination -EQ "$($policy.peerUla)/128").Count -eq 1) 'Peer address/route was not actually applied.'
Assert-Check (@(Get-FixtureMutation | Where-Object name -In @('SetSharing', 'SetDns')).Count -eq 0 -and @($model.routes | Where-Object destination -EQ '0.0.0.0/0').Count -eq 1) 'Peer-only mode changed internet sharing or default routing.'
Assert-Check ($checkpointCalls -ge 2 -and -not $lastCheckpoint.ContainsKey('pendingTransaction')) 'Successful apply did not journal progress and commit state.'
foreach ($entry in @($timeline | Where-Object { $_.kind -eq 'Operation' -and $_.name -in @('AddAddress', 'AddRoute', 'SetSharing', 'SetDns') })) {
    Assert-Check ($entry.pendingName -ceq $entry.name) 'A mutation started without its exact operation name durably checkpointed.'
}
Assert-Check (@($calls | Where-Object name -EQ 'Snapshot').Count -eq 1 -and @($calls | Where-Object name -EQ 'ReadAddressState').Count -eq 2 -and @($calls | Where-Object name -EQ 'ReadRouteState').Count -eq 1) 'Peer convergence repeated full global snapshots instead of fresh narrow custody reads.'
$owned = $r.state; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 0 -and @(Get-FixtureMutation).Count -eq 0) 'Repeated peer apply was not idempotent.'
# Same GUID/MAC, changing ifIndex/name is a normal discovery event.
$model.adapters[0].index = 88; $model.adapters[0].name = 'new-interface-name'
foreach ($entry in @($model.addresses | Where-Object index -EQ 31)) { $entry.index = 88 }
foreach ($entry in @($model.routes | Where-Object index -EQ 31)) { $entry.index = 88 }
$script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.peerProof.interfaceIndex -eq 88 -and @(Get-FixtureMutation).Count -eq 0) 'Adapter renumbering changed ownership or lost the direct scope.'
Write-Output 'PASS: PeerApplyIdempotenceAndRenumbering'

Initialize-Fixture
$script:peerMac = '020000000099'; $r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 2 -and -not $r.peerReady -and @(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0) 'Unknown physical peer was auto-enrolled or mutated.'
Write-Output 'PASS: UnknownPeerRefusal'

Initialize-Fixture
Add-FixtureAddress @{ index = 31; address = $policy.localIPv4; prefix = 30 }
Add-FixtureAddress @{ index = 31; address = $policy.localUla; prefix = 64 }
$model.routes += @{ destination = "$($policy.peerUla)/128"; nextHop = '::'; index = 31; metric = 5 }
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and @(Get-FixtureMutation).Count -eq 0 -and $r.state.ownedAddresses.Count -eq 0 -and $r.state.ownedRoutes.Count -eq 0) 'Sufficient foreign ULA prefix/route was overwritten or adopted.'
Write-Output 'PASS: ExistingUlaCustodyPreserved'

Initialize-Fixture Provider
$model.sharing = @(@{ guid = 'foreign-public'; type = 0 }, @{ guid = 'foreign-private'; type = 1 })
Assert-Refused { Invoke-Fixture -Apply } 'Foreign ICS pair was stolen.'
Assert-Check (@(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0) 'Foreign sharing refusal occurred after mutation.'
Write-Output 'PASS: ForeignSharingRefusedBeforeMutation'

Initialize-Fixture Consumer
$model.sharing = @(@{ guid = $model.adapters[0].guid; type = 0 }, @{ guid = 'foreign-private'; type = 1 })
Assert-Refused { Invoke-Fixture -Apply } 'Consumer reused a USB4 interface already owned by an ICS pair.'
Assert-Check (@(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0) 'Consumer sharing refusal occurred after mutation.'
Initialize-Fixture Consumer
Add-FixtureAddress @{ index = 31; address = $policy.sharing.providerAddress; prefix = 24 }
Assert-Refused { Invoke-Fixture -Apply } 'Consumer accepted its own local address as an upstream gateway.'
Assert-Check (@(Get-FixtureMutation).Count -eq 0) 'Self-gateway refusal changed foreign addressing.'
Write-Output 'PASS: ConsumerForeignSharingAndSelfGatewayRefused'

Initialize-Fixture Provider
Add-FixtureAddress @{ index = 31; address = '172.31.252.133'; prefix = 30 }
$natBefore = $model.nat | ConvertTo-Json -Compress
$unrelatedBefore = Get-Usb4UnrelatedNetworkFingerprint $model 31
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.networkReady -and $r.internetReady -and $model.sharing.Count -eq 2) 'Provider did not verify and enable its exact ICS pair.'
Assert-Check (($model.nat | ConvertTo-Json -Compress) -ceq $natBefore -and (Get-Usb4UnrelatedNetworkFingerprint $model 31) -ceq $unrelatedBefore) 'Provider changed WSL/NAT or another interface.'
Assert-Check (@($model.addresses | Where-Object address -EQ '172.31.252.133').Count -eq 1 -and @($r.state.ownedAddresses | Where-Object address -EQ '172.31.252.133').Count -eq 0) 'ICS reset lost or adopted the preexisting transport address.'
$owned = $r.state; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 0 -and @(Get-FixtureMutation).Count -eq 0) 'Provider changed an already matching sharing pair.'
Write-Output 'PASS: ProviderPreservesWslAndRestoresTransport'

foreach ($loss in @('ChangedMac', 'Disconnected')) {
    Initialize-Fixture Provider
    $r = Invoke-Fixture -Apply; $owned = $r.state; $script:calls.Clear()
    if ($loss -eq 'ChangedMac') { $script:peerMac = '020000000099' } else { $script:peerHealthy = $false }
    $unrelatedBefore = Get-Usb4UnrelatedNetworkFingerprint $model 31
    $r = Invoke-Fixture $owned -Apply
    Assert-Check ($r.exitCode -eq 2 -and -not $r.peerReady -and $model.sharing.Count -eq 0 -and -not $r.state.ics) "Provider $loss left owned sharing enabled for an unqualified peer."
    Assert-Check ((Get-Usb4UnrelatedNetworkFingerprint $model 31) -ceq $unrelatedBefore -and -not $lastCheckpoint.ContainsKey('pendingTransaction')) "Provider $loss damaged unrelated networking or lost completion custody."
}
Write-Output 'PASS: ProviderPeerLossDisablesOnlyOwnedSharing'

Initialize-Fixture Provider
$r = Invoke-Fixture -Apply; $owned = $r.state
$model.adapters[1].up = $false; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and $r.peerReady -and -not $r.internetReady -and $model.sharing.Count -eq 0 -and -not $r.state.ics) 'Provider upstream loss left previously owned sharing enabled.'
Assert-Check ($model.nat[0].name -eq 'WSL' -and $model.nat[0].prefix -eq '172.16.0.0/16') 'Provider upstream withdrawal changed WSL NAT.'
Write-Output 'PASS: ProviderUpstreamLossWithdrawsOwnedSharing'

Initialize-Fixture Provider
$model.adapters[1].up = $false
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 2 -and $r.peerReady -and -not $r.internetReady -and @($calls | Where-Object name -EQ 'SetSharing').Count -eq 0) 'Provider enabled sharing without a healthy wired upstream.'
Write-Output 'PASS: NoHealthyWiredUplinkNoSharing'

Initialize-Fixture Consumer
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.internetReady -and $r.internetProof.source -eq '192.168.137.2' -and $r.dnsProof.server -eq '192.168.137.1') 'Consumer was not qualified through the explicit private source and DNS gateway.'
Assert-Check (@($model.routes | Where-Object destination -EQ '0.0.0.0/0').Count -eq 2 -and $model.adapters[0].dns[0] -eq '192.168.137.1' -and $model.adapters[0].dns6[0] -eq '2001:4860:4860::8888') 'Consumer damaged fallback default or IPv6 DNS.'
$probeIndex = @($calls | ForEach-Object name).IndexOf('ProbeInternet'); $dnsIndex = @($calls | ForEach-Object name).IndexOf('SetDns')
$defaultCall = @($calls | Where-Object { $_.name -eq 'AddRoute' -and $_.arguments.destination -eq '0.0.0.0/0' })[0]
Assert-Check ($probeIndex -ge 0 -and $dnsIndex -gt $probeIndex -and $calls.IndexOf($defaultCall) -gt $probeIndex) 'Consumer promoted DNS/default before proving shared internet.'
$owned = $r.state; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 0 -and @(Get-FixtureMutation).Count -eq 0) 'Consumer repeated apply was not idempotent.'
Write-Output 'PASS: SourceBoundConsumerPromotionAndIdempotence'

$script:internetHealthy = $false; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and -not $r.internetReady -and @($model.routes | Where-Object { $_.destination -eq '0.0.0.0/0' -and $_.index -eq 31 }).Count -eq 0 -and @($model.routes | Where-Object destination -EQ '1.1.1.1/32').Count -eq 0) 'Upstream loss left an owned default/probe blackhole.'
Assert-Check ($model.adapters[0].dnsMode -eq 'Dhcp' -and $model.adapters[0].dns[0] -eq '8.8.8.8' -and $model.adapters[0].dns6[0] -eq '2001:4860:4860::8888') 'Upstream loss failed DNS-origin/IPv6 preservation.'
Write-Output 'PASS: UpstreamLossRestoresFallbackAndDnsOrigin'

Initialize-Fixture Consumer
$r = Invoke-Fixture -Apply; $owned = $r.state; $script:peerHealthy = $false; $script:calls.Clear()
$r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and @($model.routes | Where-Object { $_.index -eq 31 -and $_.destination -in @('0.0.0.0/0', '1.1.1.1/32') }).Count -eq 0 -and $model.adapters[0].dnsMode -eq 'Dhcp') 'Disconnected peer bypassed owned fallback withdrawal.'
Assert-Check (-not $lastCheckpoint.ContainsKey('pendingTransaction')) 'Completed peer-loss cleanup left an interrupted journal.'
Write-Output 'PASS: PeerLossWithdrawsOwnedInternet'

Initialize-Fixture Consumer
$r = Invoke-Fixture -Apply; $owned = $r.state; $model.adapters[0].up = $false
$r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and @($model.routes | Where-Object { $_.index -eq 31 -and $_.destination -in @('0.0.0.0/0', '1.1.1.1/32') }).Count -eq 0 -and $model.adapters[0].dnsMode -eq 'Dhcp' -and -not $r.state.ContainsKey('pendingTransaction')) 'A disconnected adapter lost its identity before owned fallback cleanup.'
Write-Output 'PASS: AdapterDownWithdrawsOwnedInternet'

Initialize-Fixture Consumer
$model.routes += @{ destination = '1.1.1.1/32'; nextHop = '192.168.50.1'; index = 30; metric = 1 }
$before = Get-Usb4NetworkFingerprint $model; $r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and (Get-Usb4NetworkFingerprint $model) -ceq $before -and $r.rollbackErrors.Count -eq 0) 'Foreign probe route was overwritten or failed rollback preservation.'
Write-Output 'PASS: ForeignRouteCollisionAndRollback'

Initialize-Fixture Consumer
$script:failOperation = 'AddRoute'; $script:failAfterMutation = $true
$before = Get-Usb4NetworkFingerprint $model; $r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and (Get-Usb4NetworkFingerprint $model) -ceq $before -and $r.rollbackErrors.Count -eq 0) 'A backend mutation followed by an error escaped rollback.'
Assert-Check ($lastCheckpoint.Count -eq 0) 'Successful failed-run rollback did not restore original state custody.'
Assert-Check ($r.firstFailure.operation -eq 'AddRoute' -and $r.firstFailure.message -like '*after mutation*' -and $r.operationTimings.Count -gt 0 -and $r.reconcileElapsedMilliseconds -ge 0) 'Original operation/cause or elapsed timing was lost after rollback.'
$durableError = @($timeline | Where-Object { $_.kind -eq 'Checkpoint' -and $_.firstFailure })[0]
$rollbackOperation = @($timeline | Where-Object { $_.kind -eq 'Operation' -and $_.name -eq 'RemoveRoute' })[0]
Assert-Check ($timeline.IndexOf($durableError) -lt $timeline.IndexOf($rollbackOperation) -and $durableError.firstFailure.operation -eq 'AddRoute') 'The first cause was checkpointed only after rollback started.'
Assert-Check ($failureReports.Count -gt 0 -and @($failureReports | Where-Object { $_.exitCode -ne 1 -or $_.networkReady -or $_.internetReady -or $_.peerReady -or $_.ipv4Ready }).Count -eq 0) 'An interim failure report remained green before rollback.'
Write-Output 'PASS: PostMutationFailureRollsBackExactPreimage'

Initialize-Fixture Consumer
$script:failOperation = 'AddRoute'; $script:failAfterMutation = $true; $script:failRollback = $true
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and $r.rollbackErrors.Count -gt 0 -and $r.state.ContainsKey('pendingTransaction') -and $lastCheckpoint.ContainsKey('pendingTransaction')) 'Unresolved rollback dropped pending ownership.'
$blocked = Invoke-Fixture $r.state -Apply
Assert-Check ($blocked.exitCode -eq 1 -and $blocked.phase -eq 'PendingOwnershipReview' -and $blocked.originalFailureKnown) 'An interrupted transaction was silently adopted or lost its original cause.'
Assert-Check ($blocked.originalAttemptId -eq $r.attemptId -and $blocked.attemptId -ne $r.attemptId -and $blocked.firstFailure.message -ceq $r.firstFailure.message) 'A later pending refusal replaced the original failed attempt or cause.'
Write-Output 'PASS: FailedRollbackRetainsDurableCustody'

Initialize-Fixture
$script:failCommit = $true; $script:failRollback = $true
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and $r.state.ContainsKey('pendingTransaction') -and $lastCheckpoint.pendingTransaction.firstFailure.message -eq 'Injected final-state commit refusal.' -and $r.firstFailure.operation -eq 'CommitState') 'A final commit failure cleared the journal before its original cause could be retained.'
$blocked = Invoke-Fixture $r.state -Apply
Assert-Check ($blocked.originalFailureKnown -and $blocked.firstFailure.message -eq 'Injected final-state commit refusal.') 'A pending refusal lost the final commit failure cause.'
Write-Output 'PASS: FinalCommitFailureRetainsOriginalCauseAndJournal'

Initialize-Fixture Provider
$script:corruptUnrelated = $true; $r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and $r.rollbackErrors.Count -gt 0 -and $r.state.ContainsKey('pendingTransaction')) 'Unrelated addressing corruption passed or lost pending custody.'
Write-Output 'PASS: UnrelatedNetworkDriftCannotPass'
Assert-Check ($failureReports.Count -gt 0 -and @($failureReports | Where-Object { $_.exitCode -ne 1 -or $_.networkReady -or $_.internetReady -or $_.peerReady -or $_.ipv4Ready }).Count -eq 0) 'Late invariant failure published a ready report before rollback.'

Initialize-Fixture
$model.routes += @{ destination = "$($policy.peerUla)/128"; nextHop = 'fd00:7462:7032::bad'; index = 30; metric = 1 }
$r = Invoke-Fixture
Assert-Check ($r.exitCode -eq 1 -and @(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0 -and $failureReports.Count -eq 0) 'An exceptional DryRun wrote state/report checkpoints or network state.'
Write-Output 'PASS: ExceptionalDryRunRetainsNoWriteContract'

Initialize-Fixture
$r = Invoke-Usb4NetworkReconcile -Policy $policy -Snapshot (Copy-Usb4NetworkValue $model) -Operation ${function:Invoke-FixtureOperation} -Apply -StateCheckpoint { param($State); if ($State.schemaVersion -ne 1) { throw 'Unexpected journal payload.' }; throw 'Storage refused before mutation.' }
Assert-Check ($r.exitCode -eq 1 -and @(Get-FixtureMutation).Count -eq 0) 'First journal failure allowed a network mutation.'
Write-Output 'PASS: JournalFailurePreventsMutation'

Initialize-Fixture
$r = Invoke-Fixture -Apply
$r = Publish-Usb4NetworkResult -Result $r -Apply -StateWriter { param($State); if ($State.schemaVersion -ne 1) { throw 'Unexpected state payload.' }; throw 'State publication refused.' } -ReportWriter { param($Report); throw "Unexpected report publication: $($Report.scope)" }
Assert-Check ($r.exitCode -eq 1 -and -not $r.networkReady -and $r.errors[-1] -like 'Publication failed:*') 'Publication failure was masked as success.'
$script:published = 0
$r = Publish-Usb4NetworkResult -Result $r -StateWriter { param($State); $script:published++; if ($State.Count -lt 1) { throw 'Unexpected empty state.' } } -ReportWriter { param($Report); $script:published++; if ($Report.Count -lt 1) { throw 'Unexpected empty report.' } }
Assert-Check ($published -eq 0) 'Read-only publication wrote artifacts.'
Write-Output 'PASS: PublicationErrorsAndDryRunWrites'

foreach ($noOp in @('AddRoute', 'SetDns')) {
    Initialize-Fixture Consumer
    $script:noOpOperation = $noOp
    $before = Get-Usb4NetworkFingerprint $model; $r = Invoke-Fixture -Apply
    Assert-Check ($r.exitCode -eq 1 -and -not $r.networkReady -and (Get-Usb4NetworkFingerprint $model) -ceq $before) "A falsely successful $noOp backend bypassed readback or preservation."
}
Write-Output 'PASS: BackendFalseSuccessDetectedByReadback'

Initialize-Fixture
Add-FixtureAddress @{ index = 31; address = $policy.localIPv4; prefix = 30 }
$preimage = Copy-Usb4NetworkValue $model
foreach ($change in @(@{ key = 'prefixOrigin'; value = 'WellKnown' }, @{ key = 'suffixOrigin'; value = 'Link' }, @{ key = 'skipAsSource'; value = $true })) {
    $changed = Copy-Usb4NetworkValue $preimage
    @($changed.addresses | Where-Object index -EQ 31)[0][$change.key] = $change.value
    Assert-Check ((Get-Usb4NetworkFingerprint $changed) -cne (Get-Usb4NetworkFingerprint $preimage)) "Managed $($change.key) drift was omitted from rollback admission."
}
Write-Output 'PASS: AddressMetadataDriftChangesRollbackFingerprint'

Initialize-NatFixture
$natBefore = Get-Usb4NatConfigurationFingerprint $natInventory
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.networkReady -and $r.sharingConfigured -and $r.upstreamInternetReady -and -not $r.internetReady -and $null -eq $r.peerInternetVerified -and $r.qualification -like 'ProviderConfigurationOnly*') 'Existing NAT provider configuration was confused with peer-originated packet proof.'
Assert-Check ($r.state.natReservation.committed -and $r.state.natReservation.childCidr -eq '172.16.252.0/30' -and $r.state.ownedForwarding.Count -eq 2) 'Existing NAT did not retain exact reservation and forwarding custody.'
Assert-Check ($model.adapters[0].forwarding -eq 'Enabled' -and $model.adapters[1].forwarding -eq 'Enabled' -and $model.adapters[2].forwarding -eq 'Disabled' -and @($calls | Where-Object name -EQ 'SetSharing').Count -eq 0) 'Existing NAT changed ICS or an unapproved interface forwarding flag.'
Assert-Check ((Get-Usb4NatConfigurationFingerprint $natInventory) -ceq $natBefore -and $model.nat[0].name -eq 'WSL' -and $model.nat[0].prefix -eq '172.16.0.0/16') 'Existing NAT configuration or HNS allocations were mutated.'
$owned = $r.state; $natInventory.sessions += @{ source = '172.16.252.2'; destination = '1.1.1.1' }; $script:calls.Clear()
$r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 0 -and @(Get-FixtureMutation).Count -eq 0) 'Expected owned consumer NAT sessions broke periodic idempotence.'
Write-Output 'PASS: ExistingNatProviderConfigurationAndOwnedSessionIdempotence'

foreach ($conflict in @('Unknown', 'HnsSubnet', 'Endpoint', 'HostPrefix', 'StaticMapping', 'ForeignSession', 'Inactive', 'SecondNat', 'UnownedPeerForwarding')) {
    Initialize-NatFixture
    switch ($conflict) {
        'Unknown' { $natInventory.complete = $false }
        'HnsSubnet' { $natInventory.networks[0].prefixes = @('172.16.252.0/24') }
        'Endpoint' { $natInventory.endpoints[0].addresses = @('172.16.252.2') }
        'HostPrefix' { $model.addresses[0].address = '172.16.10.5'; $model.addresses[0].prefix = 16 }
        'StaticMapping' { $natInventory.mappings += @{ internalAddress = '172.16.252.2'; natName = 'WSL'; configuration = 'foreign-map' } }
        'ForeignSession' { $natInventory.sessions += @{ source = '172.16.252.2'; destination = '1.1.1.1' } }
        'Inactive' { $natInventory.nat[0].Active = 0 }
        'SecondNat' { $natInventory.nat += Copy-Usb4NetworkValue $natInventory.nat[0] }
        'UnownedPeerForwarding' { $model.adapters[0].forwarding = 'Enabled' }
    }
    Assert-Refused { Invoke-Fixture -Apply } "Existing NAT admitted $conflict before reservation."
    Assert-Check (@(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq 0) "Existing NAT $conflict refusal occurred after mutation."
}
Write-Output 'PASS: ExistingNatUnknownAndOccupiedAllocationRefusal'

Initialize-NatFixture
$policy.sharing.dnsServers = @('169.254.1.1')
Assert-Refused { Assert-Usb4NetworkPolicy $policy } 'Existing NAT policy admitted forwarding APIPA as an external DNS target.'
Write-Output 'PASS: ExistingNatRejectsLinkLocalDnsForwarding'

Initialize-NatFixture
$natInventory.nat[0].InternalIPInterfaceAddressPrefix = '172.16.123.1/16'
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.state.natReservation.childCidr -eq '172.16.252.0/30') 'Equivalent native parent-prefix representations were incorrectly compared as different networks.'
Write-Output 'PASS: ExistingNatEquivalentParentPrefix'

Initialize-NatFixture
$script:failOperation = 'SetForwarding'; $script:failAfterMutation = $true
$before = Get-Usb4NetworkFingerprint $model; $natBefore = Get-Usb4NatConfigurationFingerprint $natInventory
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 1 -and (Get-Usb4NetworkFingerprint $model) -ceq $before -and (Get-Usb4NatConfigurationFingerprint $natInventory) -ceq $natBefore) 'A failed forwarding mutation escaped exact host/NAT preservation.'
Write-Output 'PASS: ExistingNatForwardingFailureRollback'

foreach ($loss in @('Peer', 'Wired')) {
    Initialize-NatFixture
    $r = Invoke-Fixture -Apply; $owned = $r.state
    if ($loss -eq 'Peer') { $script:peerHealthy = $false } else { $model.adapters[1].up = $false }
    $script:calls.Clear(); $natBefore = Get-Usb4NatConfigurationFingerprint $natInventory
    $r = Invoke-Fixture $owned -Apply
    Assert-Check ($r.exitCode -eq 2 -and $model.adapters[0].forwarding -eq 'Disabled' -and $model.adapters[1].forwarding -eq 'Disabled' -and $r.state.ownedForwarding.Count -eq 0) "Existing NAT $loss loss left owned forwarding enabled."
    Assert-Check ((Get-Usb4NatConfigurationFingerprint $natInventory) -ceq $natBefore -and @($calls | Where-Object name -EQ 'SetSharing').Count -eq 0) "Existing NAT $loss withdrawal changed NAT/ICS."
}
Write-Output 'PASS: ExistingNatPeerAndUpstreamLossRestoreOnlyOwnedForwarding'

Initialize-NatFixture Consumer
$script:natInventory = $null; $model.nat = @()
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.internetReady -and $r.peerInternetVerified -and $r.internetProof.source -eq '172.16.252.2' -and $r.dnsProof.server -eq '1.1.1.1') 'Existing NAT consumer did not prove source-bound HTTPS and external DNS.'
Assert-Check ($model.adapters[0].dns[0] -eq '1.1.1.1' -and @($calls | Where-Object name -EQ 'ReadExistingNat').Count -eq 0 -and @($calls | Where-Object { $_.name -eq 'Snapshot' -and (-not $_.arguments.ContainsKey('skipNat') -or -not $_.arguments.skipNat) }).Count -eq 0) 'Existing NAT consumer required local NAT inventory or configured a gateway DNS proxy.'
$owned = $r.state; $script:peerHealthy = $false; $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and $model.adapters[0].dnsMode -eq 'Dhcp' -and @($model.routes | Where-Object { $_.index -eq 31 -and $_.destination -in @('0.0.0.0/0', '1.1.1.1/32') }).Count -eq 0) 'Existing NAT consumer fallback failed to withdraw its owned public probes/default/DNS.'
Write-Output 'PASS: ExistingNatConsumerHasNoNatDependencyOrDnsProxyAssumption'

# Actual Windows IPv4 DNS assignment hides the automatic legacy IPv6 fallback;
# raw observations remain intact while real IPv6 configuration stays protected.
Initialize-NatFixture Consumer
$script:natInventory = $null; $model.nat = @()
$script:automaticDns6Fallback = $true
$model.adapters[0].dns = @()
$model.adapters[0].dns6 = @('fec0:0:0:ffff::1', 'fec0:0:0:ffff::2', 'fec0:0:0:ffff::3')
$model.adapters[0].dns6Configuration = @{ observed = $true; nameServer = ''; dhcpServers = '' }
$before = Copy-Usb4NetworkValue $model
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.internetReady -and $model.adapters[0].dns[0] -eq '1.1.1.1' -and $model.adapters[0].dns6.Count -eq 0 -and $r.rollbackErrors.Count -eq 0) 'Automatic IPv6 fallback suppression falsely rolled back proven consumer Internet.'
Assert-Check ($r.adapter.dns6.Count -eq 3 -and (Get-Usb4NetworkFingerprint $before) -cne (Get-Usb4NetworkFingerprint $model)) 'Normalization erased raw IPv6 fallback observations or weakened exact rollback fingerprints.'
Assert-Check ((Get-Usb4UnrelatedNetworkFingerprint $before 31) -cne (Get-Usb4UnrelatedNetworkFingerprint $model 31)) 'Automatic fallback normalization leaked outside explicitly owned IPv4 DNS reconciliation.'
$owned = $r.state; $script:peerHealthy = $false; $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.exitCode -eq 2 -and $model.adapters[0].dns.Count -eq 0 -and $model.adapters[0].dns6.Count -eq 3 -and $r.rollbackErrors.Count -eq 0) 'Peer loss did not restore automatic IPv4 DNS and observed IPv6 fallback.'
Write-Output 'PASS: ObservedWindowsAutomaticIpv6DnsFallbackTransition'

foreach ($kind in @('StaticSentinels', 'DhcpSentinels', 'UnknownOrigin', 'RealServer', 'PartialFallback', 'ForeignAdapter', 'ChangedConfiguration')) {
    $original = Copy-Usb4NetworkValue $before
    $changed = Copy-Usb4NetworkValue $before
    switch ($kind) {
        'StaticSentinels' { $original.adapters[0].dns6Configuration.nameServer = $original.adapters[0].dns6 -join ','; $changed.adapters[0].dns6Configuration.nameServer = $original.adapters[0].dns6Configuration.nameServer }
        'DhcpSentinels' { $original.adapters[0].dns6Configuration.dhcpServers = 'FEC00000'; $changed.adapters[0].dns6Configuration.dhcpServers = 'FEC00000' }
        'UnknownOrigin' { $original.adapters[0].dns6Configuration.observed = $false; $changed.adapters[0].dns6Configuration.observed = $false }
        'RealServer' { $original.adapters[0].dns6 = @('2001:4860:4860::8888') }
        'PartialFallback' { $original.adapters[0].dns6 = @('fec0:0:0:ffff::1') }
        'ForeignAdapter' {
            $original.adapters[1].dns6 = $original.adapters[0].dns6
            $original.adapters[1].dns6Configuration = Copy-Usb4NetworkValue $original.adapters[0].dns6Configuration
            $changed.adapters[1].dns6Configuration = Copy-Usb4NetworkValue $original.adapters[1].dns6Configuration
        }
        'ChangedConfiguration' { $changed.adapters[0].dns6Configuration.nameServer = '2001:4860:4860::8888' }
    }
    if ($kind -ne 'ChangedConfiguration') { $changed.adapters[$(if ($kind -eq 'ForeignAdapter') { 1 } else { 0 })].dns6 = @() }
    Assert-Check ((Get-Usb4UnrelatedNetworkFingerprint $original 31 -AllowManagedAutomaticDnsFallback) -cne (Get-Usb4UnrelatedNetworkFingerprint $changed 31 -AllowManagedAutomaticDnsFallback)) "IPv6 DNS invariant ignored $kind drift."
}
Write-Output 'PASS: ConfiguredUnknownAndForeignIpv6DnsRemainStrict'

Initialize-NatFixture Consumer
$script:natInventory = $null; $model.nat = @()
$model.routes += @{ destination = '9.9.9.9/32'; nextHop = '192.168.50.1'; index = 30; metric = 1 }
$foreignBefore = Get-Usb4UnrelatedNetworkFingerprint $model 31
$r = Invoke-Fixture -Apply
Assert-Check ($r.exitCode -eq 0 -and $r.internetReady -and $r.sharedPathInternetReady -and $r.desiredDefaultPreferenceMet -and $r.ipv4DefaultSelection.selectedInterfaceIndex -eq 31 -and $r.ipv4DefaultSelection.selectedEffectiveMetric -eq 15) 'Default preference ignored the effective interface-plus-route metric.'
Assert-Check ($null -eq $r.ipv4DefaultSelection.sourceAddress -and $r.ipv4DefaultSelection.sourceSelection -eq 'NotObserved' -and $r.internetProofScope -eq 'SourceBoundIpv4TlsAndDnsViaDedicatedProbeRoutes') 'Default receipt invented an unbound source or broadened dedicated probe evidence.'
Assert-Check ((Get-Usb4UnrelatedNetworkFingerprint $model 31) -ceq $foreignBefore -and @($model.routes | Where-Object destination -EQ '9.9.9.9/32').Count -eq 1) 'Default observation rewrote foreign or more-specific routes.'
Write-Output 'PASS: EffectiveDefaultPreferenceAndScopedInternetEvidence'

foreach ($case in @('ForeignWins', 'Tie', 'UnknownMetric')) {
    Initialize-NatFixture Consumer
    $script:natInventory = $null; $model.nat = @()
    switch ($case) {
        'ForeignWins' { $model.adapters[0].metric = 20; $expectedState = 'Selected' }
        'Tie' { $model.adapters[0].metric = 10; $expectedState = 'Tie' }
        'UnknownMetric' { $script:unknownRoutingMetricIndex = 31; $expectedState = 'Unknown' }
    }
    $foreignBefore = Get-Usb4UnrelatedNetworkFingerprint $model 31
    $r = Invoke-Fixture -Apply
    Assert-Check ($r.exitCode -eq 2 -and -not $r.networkReady -and -not $r.internetReady -and $r.sharedPathInternetReady -and $r.peerInternetVerified -and -not $r.desiredDefaultPreferenceMet -and $r.ipv4DefaultSelection.state -eq $expectedState) "$case was falsely claimed as preferred shared Internet."
    Assert-Check ($r.state.dns -and $r.rollbackErrors.Count -eq 0 -and -not $r.state.ContainsKey('pendingTransaction') -and (Get-Usb4UnrelatedNetworkFingerprint $model 31) -ceq $foreignBefore) "$case rewrote foreign policy or discarded healthy scoped custody."
    $owned = $r.state; $script:calls.Clear(); $r = Invoke-Fixture $owned -Apply
    Assert-Check ($r.exitCode -eq 2 -and $r.sharedPathInternetReady -and @(Get-FixtureMutation).Count -eq 0) "$case caused recurring route or metric mutations."
    $script:calls.Clear(); $checkpointBefore = $checkpointCalls; $r = Invoke-Fixture $owned
    Assert-Check ($r.exitCode -eq 2 -and -not $r.networkReady -and -not $r.internetReady -and $r.sharedPathInternetReady -and -not $r.desiredDefaultPreferenceMet -and @(Get-FixtureMutation).Count -eq 0 -and $checkpointCalls -eq $checkpointBefore) "$case idempotent DryRun falsely claimed preference or wrote state."
}
Write-Output 'PASS: UnmetDefaultPreferenceDoesNotRewriteForeignRouting'

Initialize-Fixture
$empty = Get-FixtureSnapshot; $empty.routes = @()
$v4 = Get-Usb4DefaultRouteSelection $empty IPv4 -ExpectedInterfaceIndex 31
Assert-Check ($v4.state -eq 'Absent' -and -not $v4.desiredPreferenceMet -and $null -eq $v4.selectedInterfaceIndex) 'Absent default route acquired a fabricated choice.'
$empty.routes = @(@{ destination = '0.0.0.0/0'; nextHop = '192.168.50.1'; index = 30; metric = 10 })
$empty.routingInterfaces = @()
Assert-Check ((Get-Usb4DefaultRouteSelection $empty IPv4).state -eq 'Unknown') 'Unobserved routing interface metric became a selected route.'
$empty = Get-FixtureSnapshot; $empty.routes += @{ destination = '0.0.0.0/0'; nextHop = '192.168.50.2'; index = 30; metric = 10 }
Assert-Check ((Get-Usb4DefaultRouteSelection $empty IPv4).state -eq 'Tie') 'Different equal-cost gateways on one interface were collapsed into a unique route.'
Write-Output 'PASS: AbsentUnknownAndEqualCostGatewaySelection'

Initialize-Fixture
$policy.localUla = 'fd00:7462:7032::7'; $policy.peerUla = 'fd00:7462:7032::9'
$r = Invoke-Fixture -Apply
Assert-Check ($r.ipv6.linkLocalPeerReady -and $r.ipv6.ulaConfigured -and $null -eq $r.ipv6.ulaPeerVerified -and -not $r.ipv6.defaultRoutePresent -and -not $r.ipv6.internetVerified -and $r.ipv6.internetState -eq 'NotVerified') 'IPv6 peer transport was confused with ULA packet proof or routed Internet.'
$model.routes += @{ destination = '::/0'; nextHop = 'fe80::1'; index = 30; metric = 10 }
$owned = $r.state; $r = Invoke-Fixture $owned -Apply
Assert-Check ($r.ipv6.defaultRoutePresent -and $r.ipv6.defaultSelection.state -eq 'Selected' -and $r.ipv6.defaultSelection.selectedInterfaceIndex -eq 30 -and -not $r.ipv6.internetVerified) 'IPv6 default presence was falsely elevated to verified IPv6 Internet.'
Write-Output 'PASS: Ipv6TransportConfigurationAndInternetAreDistinct'

Write-Output "PASS: $script:checks detecting assertions across policy, real reconciliation, recovery, and publication controls."
