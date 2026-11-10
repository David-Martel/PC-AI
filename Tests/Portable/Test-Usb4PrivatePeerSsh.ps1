#Requires -Version 7.0
<#
.SYNOPSIS
Detect private IPv4 admission without exact enrolled USB4 peer custody.
#>
[CmdletBinding()]
param([string]$SourcePath=(Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. $SourcePath
$script:checks=0
function Assert-Private([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message};$script:checks++}
function Copy-Private($Value){
    if($Value -is [Collections.IDictionary]){$copy=@{};foreach($key in $Value.Keys){$copy[$key]=Copy-Private $Value[$key]};return $copy}
    if($Value -is [Array]){return ,@($Value|ForEach-Object {Copy-Private $_})}
    $Value
}
function Assert-PrivateRefusal([scriptblock]$Operation){$caught=$false;try{& $Operation|Out-Null}catch{$caught=$true};Assert-Private $caught 'Expected private enrollment refusal.'}
$request=@{ExpectedHostName='fixture-host';InterfaceGuid='11111111-2222-3333-4444-555555555555';ExpectedInterfaceAlias='Fixture USB4';ExpectedMac='02-00-00-00-00-01';ExpectedPeerMac='02-00-00-00-00-02';LocalAddress=@('172.31.251.242');RemoteAddress=@('172.31.251.241');ProgramPath='C:\Windows\System32\OpenSSH\sshd.exe';ServiceBinding='Executable';Action='Add';EnrolledPrivatePolicy=@{schemaVersion=1;hostName='fixture-host';mode='Peer';adapterMac='02-00-00-00-00-01';peerMac='02-00-00-00-00-02';localIPv4='172.31.251.242';peerIPv4='172.31.251.241';prefixLength=30}}
function New-PrivateFixture {
    $script:store=@{};$script:calls=[Collections.Generic.List[string]]::new();$script:snapshotCount=0;$script:drift=$null
    $script:snapshot=@{hostName='fixture-host';interfaceGuid=$request.InterfaceGuid;alias='Fixture USB4';mac=$request.ExpectedMac;up=$true;usb4=$true;serviceRunning=$true;serviceSidType=0;serviceImagePath=$request.ProgramPath;runningImagePath=$request.ProgramPath;processId=4242;processStartedUtc='2030-01-01T00:00:00Z';localAddresses=@('172.31.251.242');prefixLengths=@{'172.31.251.242'=30};peerMac=$request.ExpectedPeerMac;peerNeighborState=5}
    @{
        Runtime={$script:calls.Add('runtime');@{windows=$true;hostName='fixture-host';administrator=$true}}
        Acquire={$script:calls.Add('acquire');'fixture-lease'}
        Release={param($Lease)$script:calls.Add('release')}
        Snapshot={param($Plan)$script:calls.Add('snapshot');$script:snapshotCount++;$copy=Copy-Private $script:snapshot;if($script:snapshotCount -eq 2 -and $script:drift){$copy[$script:drift]='changed'};$copy}
        ReadRules={param($Name)$script:calls.Add('read');if($script:store.ContainsKey($Name)){Copy-Private $script:store[$Name]}}
        AddRule={param($Rule)$script:calls.Add('add');$script:store[$Rule.name]=Copy-Private $Rule}
        RemoveRule={param($Name)$script:calls.Add('remove');$script:store.Remove($Name)}
    }
}

# Selector group: ExplicitPrivatePairKeepsOfflineExactHostScope
# Protects: explicit Peer enrollment only; detects implicit private admission,
# widened host scopes, unstable naming and provider calls while planning.
# Needs: inert dictionaries only; no Windows/network. Breadcrumb: New-Usb4PeerSshAccessPlan.
$ops=New-PrivateFixture
$result=Invoke-Usb4PeerSshAccess -Request $request -Operations $ops
$plan=$result.plan
Assert-Private ($result.status -eq 'Planned' -and $script:calls.Count -eq 0) 'Private planning reached providers.'
Assert-Private ($plan.rule.localAddresses -ceq '172.31.251.242' -and $plan.rule.remoteAddresses -ceq '172.31.251.241') 'Private host addresses widened.'
Assert-Private ($plan.rule.localPorts -eq '22' -and $plan.rule.protocol -eq 6 -and $plan.rule.interfaces.Count -eq 1 -and $plan.rule.applicationName -ceq $request.ProgramPath) 'Private rule lost exact SSH application/interface scope.'
$repeat=New-Usb4PeerSshAccessPlan @request
Assert-Private ($repeat.rule.name -ceq $plan.rule.name) 'Equivalent private enrollment changed identity.'
$default=Copy-Private $request;$default.Remove('EnrolledPrivatePolicy');$default.Remove('ExpectedPeerMac')
Assert-PrivateRefusal {New-Usb4PeerSshAccessPlan @default}
Write-Output 'PASS: explicit private pair keeps offline exact-host scope'

# Selector group: MalformedOrMismatchedEnrollmentRefusesBeforeProviders
# Protects: exact host/MAC/pair/private /30 declarations; detects public/broad,
# wrong-role, wrong-host, wrong-peer, network/broadcast and address-family coercion.
# Needs: inert plans/provider traps. Breadcrumb: private policy normalization and address admission.
$cases=@(
    @('policy','schemaVersion',2),@('policy','hostName','other-host'),@('policy','mode','Provider'),
    @('policy','adapterMac','02-00-00-00-00-09'),@('policy','peerMac','02-00-00-00-00-09'),
    @('policy','prefixLength',24),@('policy','prefixLength','30'),@('policy','localIPv4','8.8.8.2'),
    @('policy','peerIPv4','172.31.251.245'),@('policy','peerIPv4','172.31.251.240'),
    @('policy','peerIPv4','172.31.251.243'),@('policy','peerIPv4','172.31.251.242'),
    @('request','ExpectedPeerMac','01-00-00-00-00-02'),@('request','ExpectedPeerMac',''),
    @('request','LocalAddress',@('172.31.251.242/30')),@('request','RemoteAddress',@('172.31.251.241/30')),
    @('request','RemoteAddress',@('172.31.251.245')),@('request','LocalAddress',@('169.254.12.34'))
)
foreach($case in $cases){$candidate=Copy-Private $request;if($case[0] -eq 'policy'){$candidate.EnrolledPrivatePolicy[$case[1]]=$case[2]}else{$candidate[$case[1]]=$case[2]};$ops=New-PrivateFixture;Assert-PrivateRefusal {Invoke-Usb4PeerSshAccess -Request $candidate -Apply -Operations $ops};Assert-Private ($script:calls.Count -eq 0) 'Invalid enrollment reached providers.'}
Write-Output 'PASS: malformed or mismatched enrollment refuses before providers'

# Selector group: FreshUsb4PrefixAndResolvedPeerAreRequired
# Protects: fresh positive USB4 GUID/MAC, assigned /30 and resolved peer MAC.
# Detects: foreign interface, unknown prefix, stale policy, unresolved/foreign peer
# and a generation change immediately before Add. Needs inert runtime snapshots.
# Breadcrumb: Assert-Usb4SshSnapshot and second preflight.
$ops=New-PrivateFixture;$accepted=Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops
Assert-Private ($accepted.status -eq 'Added') 'Known-good private runtime fixture did not pass.'
foreach($change in @(@('interfaceGuid','other'),@('usb4',$false),@('prefixLengths',@{}),@('prefixLengths',@{'172.31.251.242'=24}),@('peerMac','02-00-00-00-00-09'),@('peerNeighborState',1),@('peerNeighborState',4))){$ops=New-PrivateFixture;$script:snapshot[$change[0]]=$change[1];Assert-PrivateRefusal {Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops};Assert-Private (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove') -and $script:calls.Contains('release')) 'Foreign/unresolved private endpoint mutated or stranded lease.'}
$ops=New-PrivateFixture;$script:drift='peerMac';Assert-PrivateRefusal {Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops};Assert-Private (-not $script:calls.Contains('add')) 'Peer drift passed second preflight.'
Write-Output 'PASS: fresh USB4 prefix and resolved peer are required'

# Selector group: PrivateRuleReadbackAndRemovalRetainOwnedScope
# Protects: exact owned private rule lifecycle and foreign-rule custody.
# Detects: masked-address widening, drift overwrite/removal or default-mode leakage.
# Needs: inert firewall store only. Breadcrumb: rule identity, readback and Remove.
$ops=New-PrivateFixture;$added=Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops
Assert-Private ($added.status -eq 'Added' -and $script:store.Count -eq 1) 'Private add did not read back.'
$name=$added.plan.rule.name;$script:store[$name].localAddresses='172.31.251.242/32';$script:store[$name].remoteAddresses='172.31.251.241/255.255.255.255'
$again=Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops
Assert-Private ($again.status -eq 'NoOp') 'Exact-host mask readback was refused.'
$script:store[$name].remoteAddresses='172.31.251.240/30';$script:calls.Clear();Assert-PrivateRefusal {Invoke-Usb4PeerSshAccess -Request $request -Apply -Operations $ops};Assert-Private (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove')) 'Broader observed private scope was overwritten.'
$script:store[$name].remoteAddresses='172.31.251.241';$removeRequest=Copy-Private $request;$removeRequest.Action='Remove';$removed=Invoke-Usb4PeerSshAccess -Request $removeRequest -Apply -Operations $ops
Assert-Private ($removed.status -eq 'Removed' -and $script:store.Count -eq 0) 'Exact private removal failed.'
Write-Output 'PASS: private rule readback and removal retain owned scope'

# Selector group: UnregisteredWindowsPrivateComRuleRetainsExactHosts
# Protects: actual Windows COM rule value conversion for private exact hosts.
# Detects: unsupported VARIANT arrays or OS normalization mismatches before Apply.
# Needs: Windows and a current interface alias; never opens live firewall policy.
# Breadcrumb: New-Usb4SshComRule and observed exact-host identity.
if($IsWindows){
    $nic=@([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces())[0]
    $comRequest=Copy-Private $request;$comRequest.ExpectedInterfaceAlias=$nic.Name
    $comPlan=New-Usb4PeerSshAccessPlan @comRequest
    $comRule=New-Usb4SshComRule $comPlan.rule
    try{$observed=ConvertFrom-Usb4SshComRule $comRule;Assert-Usb4SshRule @($observed) $comPlan.rule;Assert-Private ($observed.localAddresses -match '172\.31\.251\.242' -and $observed.remoteAddresses -match '172\.31\.251\.241') 'Windows COM private addresses changed.'}
    finally{[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($comRule)}
    Write-Output 'PASS: unregistered Windows private COM rule retains exact hosts'
}else{Write-Output 'NOT_RUN: private COM value conversion requires Windows'}
Write-Output "PASS: $script:checks private SSH assertions"
