#Requires -Version 7.0
<#
.SYNOPSIS
Reconcile owned Windows USB4 addresses through an authenticated enrolled partner.
.DESCRIPTION
Bootstrap uses only a source-bound link-local SSH connection with an existing pin.
The partner's fresh observe-current record supplies the enrolled controller pair
and derived address, while actual SSH socket endpoints select the physical edge.
Apply manages only explicitly owned addresses; existing addresses remain unowned
unless the local administrative policy records an exact custody transfer.
See docs/usb4/windows-boot.md. No default routes, DNS, firewall or MTU changes occur.
.PARAMETER DryRun
Read current partner evidence and plan recovery without writing files or addresses.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$PolicyPath = (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/windows-partners.json'),
    [string]$StatePath = (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/windows-owned-edges.json'),
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h','help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-Usb4Address {
    <# .SYNOPSIS
    Normalize IPv6 socket addresses independently of the peer's zone spelling.
    #>
    param([string]$Address)
    $ip = [Net.IPAddress]::Parse(($Address -split '%',2)[0])
    if ($ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetworkV6) { throw 'IPv6 edge address required.' }
    $ip.ToString()
}

function Get-Usb4PartnerSshArguments {
    <# .SYNOPSIS
    Build isolated pinned SSH arguments that cannot inherit proxy or socket reuse.
    #>
    param([hashtable]$Peer,[string]$LocalAddress,[string]$PeerAddress,[int]$InterfaceIndex)
    $local = ConvertTo-Usb4Address $LocalAddress; $remote = ConvertTo-Usb4Address $PeerAddress
    if ([Net.IPAddress]::Parse($local).IsIPv6LinkLocal) { $local += "%$InterfaceIndex" }
    if ([Net.IPAddress]::Parse($remote).IsIPv6LinkLocal) { $remote += "%$InterfaceIndex" }
    @('-6','-F','NUL','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent=none',
        '-o','StrictHostKeyChecking=yes','-o',"HostKeyAlias=$($Peer.host_key_alias)",
        '-o',"UserKnownHostsFile=$($Peer.known_hosts_file)",'-o','GlobalKnownHostsFile=NUL',
        '-o','ProxyCommand=none','-o','ProxyJump=none','-o','ControlMaster=no','-o','ControlPath=none',
        '-o','PermitLocalCommand=no','-o','ClearAllForwardings=yes','-o','ConnectTimeout=3',
        '-i',$Peer.identity_file,'-b',$local,"$($Peer.ssh_user)@$remote")
}

function ConvertFrom-Usb4PartnerFrame {
    <# .SYNOPSIS
    Parse the bounded node, actual socket and fresh native observer response.
    #>
    param([string]$Text)
    $Text=$Text.Replace("`r`n","`n")
    if ($Text.Length -gt 1048576 -or -not $Text.StartsWith("PCAI_NODE`n")) { throw 'Unexpected partner frame.' }
    $parts = $Text -split "`nPCAI_CONNECTION`n",2
    if ($parts.Count -ne 2) { throw 'Missing partner socket evidence.' }
    $node = $parts[0].Substring(10).Trim()
    $parts = $parts[1] -split "`nPCAI_OBSERVATION`n",2
    if ($parts.Count -ne 2) { throw 'Missing enrolled observation.' }
    $connection = @($parts[0].Trim() -split '\s+')
    $parts = $parts[1] -split "`nPCAI_ADDRESSES`n",2
    if ($parts.Count -ne 2) { throw 'Missing partner interface addresses.' }
    @{node_id=$node;connection=$connection;observation=($parts[0]|ConvertFrom-Json -AsHashtable);
        addresses=@($parts[1]|ConvertFrom-Json -AsHashtable)}
}

function Confirm-Usb4PartnerProof {
    <# .SYNOPSIS
    Bind one fresh enrolled edge to peer identity, actual socket and policy epoch.
    #>
    param([hashtable]$Frame,[hashtable]$Peer,[string]$LocalNodeId,[string]$LocalAddress,[string]$PeerAddress,
        [hashtable]$Previous=@{},[DateTimeOffset]$Now=[DateTimeOffset]::UtcNow,[int]$TtlSeconds=90)
    if ($Frame.node_id -cne $Peer.node_id -or ([guid]$LocalNodeId) -eq [guid]::Empty) { throw 'Partner node identity mismatch.' }
    $connection = @($Frame.connection)
    if ($connection.Count -ne 4 -or $connection[3] -ne '22' -or [int]$connection[1] -notin 1..65535 -or
        (ConvertTo-Usb4Address $connection[0]) -ne (ConvertTo-Usb4Address $LocalAddress) -or
        (ConvertTo-Usb4Address $connection[2]) -ne (ConvertTo-Usb4Address $PeerAddress)) { throw 'Actual SSH socket does not use the selected edge.' }
    $observation=$Frame.observation
    if ($observation.schema -ne 1 -or $observation.policy_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $observation.namespace -notmatch '^net:\[\d+\]$' -or ([guid]$observation.boot_id) -eq [guid]::Empty) { throw 'Malformed native observation.' }
    $observed=[DateTimeOffset]$observation.observed_at
    if (($Now-$observed).TotalSeconds -gt $TtlSeconds -or ($observed-$Now).TotalSeconds -gt 5) { throw 'Partner observation is stale or future-dated.' }
    if ([long]$observation.epoch -lt [long]$Peer.minimum_epoch -or
        ($Previous.ContainsKey('epoch') -and ([long]$observation.epoch -lt [long]$Previous.epoch -or
        ([long]$observation.epoch -eq [long]$Previous.epoch -and $observation.policy_sha256 -cne $Previous.policy_sha256)))) { throw 'Policy epoch rollback or same-epoch substitution.' }
    $interfaces=@($Frame.addresses|Where-Object {
        @($_.addr_info|Where-Object { $_.family -eq 'inet6' -and
            (ConvertTo-Usb4Address $_.local) -eq (ConvertTo-Usb4Address $PeerAddress) }).Count -gt 0
    })
    if ($interfaces.Count -ne 1) { throw 'Actual SSH server address does not select one partner interface.' }
    $remote=$interfaces[0]
    $edges=@($observation.edges|Where-Object {
        $_.state -eq 'ENROLLED_UNVERIFIED' -and $_.ifindex -eq $remote.ifindex -and $_.interface -ceq $remote.ifname -and $_.edge.peer_node_id -ceq $LocalNodeId
    })
    if ($edges.Count -ne 1) { throw 'Actual partner socket lacks one enrolled unprotected edge.' }
    $edge=$edges[0].edge
    if ($edge.local_controller_uuid -notin $Peer.controller_uuids -or ([guid]$edge.remote_controller_uuid) -eq [guid]::Empty -or
        $edge.edge_id -notmatch '^usb4-[0-9a-f]{64}$' -or $edge.prefix_len -ne 64 -or $edge.mtu -lt 1280 -or $edge.mtu -gt 65535) { throw 'Unapproved controller or derived edge.' }
    $windows=ConvertTo-Usb4Address $edge.peer_ula; $partner=ConvertTo-Usb4Address $edge.local_ula
    $localBytes=[Net.IPAddress]::Parse($windows).GetAddressBytes();$remoteBytes=[Net.IPAddress]::Parse($partner).GetAddressBytes()
    $poolParts=$Peer.ula_pool -split '/',2
    if ($poolParts.Count -ne 2 -or $poolParts[1] -ne '48') { throw 'Approved ULA pool must be /48.' }
    $pool=[Net.IPAddress]::Parse($poolParts[0]).GetAddressBytes()
    if ($pool[0] -ne 253 -or $windows -eq $partner -or
        [Convert]::ToHexString([byte[]]$localBytes[0..5]) -cne [Convert]::ToHexString([byte[]]$pool[0..5]) -or
        [Convert]::ToHexString([byte[]]$localBytes[0..7]) -cne [Convert]::ToHexString([byte[]]$remoteBytes[0..7])) { throw 'Derived addresses leave the approved isolated edge.' }
    @{peer_node_id=$Frame.node_id;edge_id=$edge.edge_id;epoch=[long]$observation.epoch;policy_sha256=$observation.policy_sha256;
        boot_id=$observation.boot_id;windows_address=$windows;peer_address=$partner;prefix_length=64;
        remote_controller_uuid=$edge.local_controller_uuid;windows_controller_uuid=$edge.remote_controller_uuid;
        remote_ifindex=$remote.ifindex;remote_mac=$remote.address;observed_at=$observed.ToString('o');scope='current-root-namespace-edge'}
}

function Get-Usb4AddressDisposition {
    <# .SYNOPSIS
    Distinguish a new owned allocation from foreign or explicitly transferred custody.
    #>
    param([hashtable]$Proof,[hashtable]$Adapter,[object[]]$Addresses,[object[]]$Owned,[object[]]$Adoptions)
    $existing=@($Addresses|Where-Object { (ConvertTo-Usb4Address $_.IPAddress) -eq $Proof.windows_address })
    if (@($existing|Where-Object InterfaceIndex -NE $Adapter.interface_index).Count -gt 0 -or $existing.Count -gt 1) { throw 'Derived address remains on another incarnation; inspect the exact old GUID/MAC/root and explicitly transfer or retire its custody before recovery.' }
    if ($existing.Count -eq 1) {
        if ($existing[0].PrefixLength -ne 64) { throw 'Existing derived address has foreign prefix custody.' }
        $ownedMatch=@($Owned|Where-Object {$_.owned -and $_.committed -and $_.address -eq $Proof.windows_address -and $_.edge_id -eq $Proof.edge_id -and
            $_.interface_guid -eq $Adapter.interface_guid -and $_.mac -eq $Adapter.mac -and $_.root_id -eq $Adapter.root_id})
        if ($ownedMatch.Count -eq 1) { return @{action='RefreshOwned'} }
        $adopt=@($Adoptions|Where-Object {$_.address -eq $Proof.windows_address -and $_.prefix_length -eq 64 -and
            $_.edge_id -eq $Proof.edge_id -and $_.interface_guid -eq $Adapter.interface_guid -and $_.mac -eq $Adapter.mac -and $_.root_id -eq $Adapter.root_id})
        if ($adopt.Count -eq 1) { return @{action='AdoptExplicitCustody'} }
        return @{action='PreserveUnowned'}
    }
    @{action='AddOwned'}
}

function Get-Usb4ExpiredAddressPlan {
    <# .SYNOPSIS
    Withdraw only expired committed custody still bound to the exact local adapter.
    #>
    param([object[]]$Owned,[object[]]$Adapters,[object[]]$Addresses,[DateTimeOffset]$Now=[DateTimeOffset]::UtcNow)
    foreach($row in $Owned) {
        if (-not $row.owned -or -not $row.committed -or [DateTimeOffset]$row.expires_at -gt $Now) { continue }
        $adaptersFound=@($Adapters|Where-Object {$_.interface_guid -eq $row.interface_guid -and $_.mac -eq $row.mac -and $_.root_id -eq $row.root_id})
        if ($adaptersFound.Count -ne 1) { continue }
        $adapter=$adaptersFound[0]
        $addressesFound=@($Addresses|Where-Object {$_.InterfaceIndex -eq $adapter.interface_index -and
            (ConvertTo-Usb4Address $_.IPAddress) -eq $row.address -and $_.PrefixLength -eq $row.prefix_length})
        if ($addressesFound.Count -eq 1 -and $row.prefix_length -eq 64 -and [Net.IPAddress]::Parse($row.address).GetAddressBytes()[0] -eq 253) {
            @{address=$row.address;prefix_length=$row.prefix_length;interface_index=$adapter.interface_index;interface_guid=$adapter.interface_guid}
        }
    }
}

function Assert-Usb4AdministrativeFile {
    <# .SYNOPSIS
    Refuse reparse points and policy/journal files writable by nonadministrators.
    #>
    param([string]$Path)
    Assert-Usb4AdministrativeDirectory ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path)))
    $item=Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Administrative USB4 file must be regular.' }
    $acl=Get-Acl -LiteralPath $Path
    $trusted=@('S-1-5-18','S-1-5-32-544')
    if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) { throw 'USB4 administrative file owner is not SYSTEM/Administrators.' }
    $write=[Security.AccessControl.FileSystemRights]::WriteData -bor [Security.AccessControl.FileSystemRights]::AppendData -bor
        [Security.AccessControl.FileSystemRights]::WriteAttributes -bor [Security.AccessControl.FileSystemRights]::WriteExtendedAttributes -bor
        [Security.AccessControl.FileSystemRights]::Delete -bor [Security.AccessControl.FileSystemRights]::ChangePermissions -bor [Security.AccessControl.FileSystemRights]::TakeOwnership
    foreach($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq 'Allow' -and ($rule.FileSystemRights -band $write) -and $rule.IdentityReference.Value -notin $trusted) {
            throw 'USB4 administrative file permits a nonadministrator writer.'
        }
    }
}

function Assert-Usb4AdministrativeDirectory {
    <# .SYNOPSIS
    Require a protected nonreparse journal directory before creating any intent.
    #>
    param([string]$Path)
    $item=Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'USB4 administrative directory must be a regular directory.' }
    $acl=Get-Acl -LiteralPath $Path
    $trusted=@('S-1-5-18','S-1-5-32-544')
    if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) { throw 'USB4 directory owner is not SYSTEM/Administrators.' }
    $write=[Security.AccessControl.FileSystemRights]::Write -bor [Security.AccessControl.FileSystemRights]::Delete -bor
        [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor [Security.AccessControl.FileSystemRights]::ChangePermissions -bor [Security.AccessControl.FileSystemRights]::TakeOwnership
    foreach($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq 'Allow' -and ($rule.FileSystemRights -band $write) -and $rule.IdentityReference.Value -notin $trusted) { throw 'USB4 directory permits a nonadministrator writer.' }
    }
}

function Get-Usb4PartnerPolicy {
    <# .SYNOPSIS
    Load the local administrative allowlist without creating or enrolling peers.
    #>
    param([string]$Path)
    Assert-Usb4AdministrativeFile $Path
    $policy=[IO.File]::ReadAllText($Path)|ConvertFrom-Json -AsHashtable
    $nodePath=Join-Path $env:ProgramData 'pcai-usb4/node-id'
    Assert-Usb4AdministrativeFile $nodePath
    $node=[IO.File]::ReadAllText($nodePath).Trim()
    if ($policy.schema -ne 1 -or $policy.node_id -cne $node -or ([guid]$node) -eq [guid]::Empty -or
        $policy.ttl_seconds -lt 30 -or $policy.ttl_seconds -gt 300 -or @($policy.peers).Count -notin 1..8) { throw 'Invalid local partner policy.' }
    $nodes=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($peer in $policy.peers) {
        if (-not $nodes.Add($peer.node_id) -or ([guid]$peer.node_id) -eq [guid]::Empty -or $peer.node_id -eq $node -or
            $peer.host_key_alias -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$' -or
            $peer.ssh_user -notmatch '^[a-zA-Z_][a-zA-Z0-9_-]{0,31}$' -or $peer.minimum_epoch -lt 1 -or
            @($peer.controller_uuids).Count -eq 0 -or @($peer.local_root_ids).Count -eq 0) { throw 'Invalid approved partner.' }
        foreach($controller in $peer.controller_uuids) { if (([guid]$controller) -eq [guid]::Empty) { throw 'Nil controller enrollment.' } }
        foreach($root in $peer.local_root_ids) { if ($root -notlike 'USB4\ROOT_DEVICE_ROUTER*\*') { throw 'Full Windows root-router identity required.' } }
        foreach($key in @('identity_file','known_hosts_file')) {
            if (-not [IO.Path]::IsPathFullyQualified($peer[$key]) -or -not [IO.File]::Exists($peer[$key])) { throw 'Explicit existing SSH identity and pin paths required.' }
        }
    }
    if (-not $policy.ContainsKey('adoptions')) { $policy.adoptions=@() }
    $policy
}

function Get-Usb4HostSnapshot {
    <# .SYNOPSIS
    Capture present USB4 identity and bounded parent ancestry without alias selection.
    #>
    $addresses=@(Get-NetIPAddress -AddressFamily IPv6 -PolicyStore ActiveStore -ErrorAction Stop)
    $persistent=@(Get-NetIPAddress -AddressFamily IPv6 -PolicyStore PersistentStore -ErrorAction Stop)
    $adapters=@()
    foreach($adapter in @(Get-NetAdapter -IncludeHidden -ErrorAction Stop|Where-Object {$_.InterfaceDescription -match 'USB4.*P2P|Thunderbolt.*Network'})) {
        $pnp=[string]$adapter.PnPDeviceID
        if (-not $pnp.StartsWith('USB4\',[StringComparison]::OrdinalIgnoreCase)) { continue }
        $present=Get-PnpDevice -InstanceId $pnp -PresentOnly -ErrorAction SilentlyContinue
        if (-not $present) { continue }
        $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $current=$pnp;$root=$null
        for($depth=0;$depth -lt 16 -and $current;$depth++) {
            if (-not $seen.Add($current)) { throw 'PnP parent cycle.' }
            if ($current.StartsWith('USB4\ROOT_DEVICE_ROUTER',[StringComparison]::OrdinalIgnoreCase)) { $root=$current;break }
            $current=[string](Get-PnpDeviceProperty -InstanceId $current -KeyName DEVPKEY_Device_Parent -ErrorAction Stop).Data
        }
        if (-not $root) { continue }
        $index=[int]$adapter.ifIndex
        $neighbors=@(Get-NetNeighbor -InterfaceIndex $index -AddressFamily IPv6 -ErrorAction Stop|Where-Object {
            [Net.IPAddress]::Parse(($_.IPAddress -split '%',2)[0]).IsIPv6LinkLocal -and
            $_.LinkLayerAddress -notmatch '^(00[-:]?){6}$' -and $_.State.ToString() -in @('Reachable','Stale','Delay','Probe','Permanent')
        }|Select-Object -First 4)
        $adapters+=@(@{interface_index=$index;interface_guid=([guid]$adapter.InterfaceGuid).ToString();mac=([string]$adapter.MacAddress).ToUpperInvariant();
            root_id=$root;up=($adapter.InterfaceOperationalStatus -eq 1 -and $adapter.AdminStatus -eq 1);name=$adapter.Name;neighbors=$neighbors;
            local_llas=@($addresses|Where-Object {$_.InterfaceIndex -eq $index -and $_.AddressState -eq 'Preferred' -and
                [Net.IPAddress]::Parse(($_.IPAddress -split '%',2)[0]).IsIPv6LinkLocal}|Select-Object -ExpandProperty IPAddress)})
    }
    @{adapters=$adapters;addresses=$addresses;persistent_addresses=$persistent}
}

function Assert-Usb4DirectRoute {
    <# .SYNOPSIS
    Refuse a source lookup leaving the selected interface or using a gateway.
    #>
    param([string]$LocalAddress,[string]$PeerAddress,[int]$InterfaceIndex)
    $local=ConvertTo-Usb4Address $LocalAddress;$remote=ConvertTo-Usb4Address $PeerAddress
    if ([Net.IPAddress]::Parse($local).IsIPv6LinkLocal) {$local+="%$InterfaceIndex"}
    if ([Net.IPAddress]::Parse($remote).IsIPv6LinkLocal) {$remote+="%$InterfaceIndex"}
    $rows=@(Find-NetRoute -LocalIPAddress $local -RemoteIPAddress $remote -InterfaceIndex $InterfaceIndex -ErrorAction Stop)
    $routes=@($rows|Where-Object {$_.PSObject.Properties['DestinationPrefix'] -and $_.DestinationPrefix})
    $sources=@($rows|Where-Object {$_.PSObject.Properties['IPAddress'] -and $_.IPAddress})
    if ($routes.Count -ne 1 -or $sources.Count -ne 1 -or @($rows|Where-Object InterfaceIndex -NE $InterfaceIndex).Count -or
        $routes[0].NextHop -ne '::' -or $routes[0].DestinationPrefix -eq '::/0' -or
        (ConvertTo-Usb4Address $sources[0].IPAddress) -ne (ConvertTo-Usb4Address $LocalAddress)) { throw 'Partner path is not direct on the selected USB4 source.' }
}

function Invoke-Usb4PartnerProbe {
    <# .SYNOPSIS
    Query enrolled identity through the actual edge with bounded hidden SSH.
    #>
    param([hashtable]$Peer,[hashtable]$Adapter,[string]$LocalAddress,[string]$PeerAddress,
        [string]$LocalNodeId,[hashtable]$Previous,[int]$TtlSeconds)
    Assert-Usb4DirectRoute $LocalAddress $PeerAddress $Adapter.interface_index
    $before=Get-NetAdapterStatistics -Name $Adapter.name -ErrorAction Stop
    $command='set -eu; printf "PCAI_NODE\n"; cat /etc/pcai-usb4/node-id; printf "\nPCAI_CONNECTION\n%s\nPCAI_OBSERVATION\n" "$SSH_CONNECTION"; sudo -n /usr/local/libexec/pcai-usb4-agent observe-current /etc/pcai-usb4/policy.json; printf "\nPCAI_ADDRESSES\n"; ip -j address show'
    $arguments=@(Get-Usb4PartnerSshArguments $Peer $LocalAddress $PeerAddress $Adapter.interface_index)+@($command)
    $result=Invoke-VigilBoundedProcess -FilePath (Join-Path $env:SystemRoot 'System32/OpenSSH/ssh.exe') -Arguments $arguments -TimeoutSeconds 6
    if ($result.ExitCode -ne 0) { throw "Partner SSH refused: $($result.Stderr.Trim())" }
    $frame=ConvertFrom-Usb4PartnerFrame $result.Stdout
    $proof=Confirm-Usb4PartnerProof $frame $Peer $LocalNodeId $LocalAddress $PeerAddress $Previous -TtlSeconds $TtlSeconds
    $after=Get-NetAdapterStatistics -Name $Adapter.name -ErrorAction Stop
    Assert-Usb4DirectRoute $LocalAddress $PeerAddress $Adapter.interface_index
    $current=@((Get-Usb4HostSnapshot).adapters|Where-Object {$_.interface_index -eq $Adapter.interface_index -and
        $_.interface_guid -eq $Adapter.interface_guid -and $_.mac -eq $Adapter.mac -and $_.root_id -eq $Adapter.root_id -and $_.up})
    if ($current.Count -ne 1 -or $after.ReceivedBytes -le $before.ReceivedBytes -or $after.SentBytes -le $before.SentBytes) { throw 'USB4 identity changed or counters did not prove bidirectional traffic.' }
    $proof
}

function Save-Usb4OwnedState {
    <# .SYNOPSIS
    Publish a private atomic journal without following an existing reparse point.
    #>
    param([string]$Path,[hashtable]$State)
    if ([IO.File]::Exists($Path)) { Assert-Usb4AdministrativeFile $Path }
    $directory=[IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path))
    Assert-Usb4AdministrativeDirectory $directory
    $staging=Join-Path $directory ('pending-'+[guid]::NewGuid().ToString('N')+'.json')
    try {
        [IO.File]::WriteAllText($staging,($State|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
        Assert-Usb4AdministrativeFile $staging
        [IO.File]::Move($staging,$Path,$true)
    } finally { if ([IO.File]::Exists($staging)) {[IO.File]::Delete($staging)} }
}

function Invoke-Usb4PartnerReconciliation {
    <# .SYNOPSIS
    Reconcile only owned addresses and publish current-edge authentication leases.
    #>
    [CmdletBinding()]
    param([hashtable]$Policy,[string]$StatePath,[switch]$Apply,[switch]$DryRun)
    $mutate=$Apply -and -not $DryRun
    $state=@{schema=1;node_id=$Policy.node_id;peers=@{};addresses=@();results=@()}
    if ([IO.File]::Exists($StatePath)) {
        Assert-Usb4AdministrativeFile $StatePath
        $state=[IO.File]::ReadAllText($StatePath)|ConvertFrom-Json -AsHashtable
        if ($state.schema -ne 1 -or $state.node_id -cne $Policy.node_id) { throw 'Owned journal node/schema mismatch.' }
    }
    $snapshot=Get-Usb4HostSnapshot
    $results=[Collections.Generic.List[object]]::new();$clock=[Diagnostics.Stopwatch]::StartNew();$attempts=0
    if ($mutate) {
        foreach($expired in @(Get-Usb4ExpiredAddressPlan $state.addresses $snapshot.adapters $snapshot.addresses)) {
            # Fresh identity and positive address membership precede each mutation.
            $fresh=Get-Usb4HostSnapshot
            $verified=@(Get-Usb4ExpiredAddressPlan $state.addresses $fresh.adapters $fresh.addresses|Where-Object {
                $_.address -eq $expired.address -and $_.interface_guid -eq $expired.interface_guid})
            if ($verified.Count -ne 1) { throw 'Expired custody changed before withdrawal.' }
            if (@($fresh.persistent_addresses|Where-Object IPAddress -EQ $expired.address).Count) { throw 'Expired address also has PersistentStore custody; explicit migration required.' }
            Remove-NetIPAddress -InterfaceIndex $verified[0].interface_index -IPAddress $expired.address -PolicyStore ActiveStore -Confirm:$false -ErrorAction Stop
            $remaining=@(Get-NetIPAddress -InterfaceIndex $verified[0].interface_index -AddressFamily IPv6 -PolicyStore ActiveStore -ErrorAction Stop|Where-Object IPAddress -EQ $expired.address)
            if ($remaining.Count) { throw 'Expired owned address remains after withdrawal.' }
            $state.addresses=@($state.addresses|Where-Object {-not($_.address -eq $expired.address -and $_.interface_guid -eq $expired.interface_guid)})
            Save-Usb4OwnedState $StatePath $state
        }
        $snapshot=Get-Usb4HostSnapshot
    }
    foreach($peer in $Policy.peers) {
        if ($clock.Elapsed.TotalSeconds -gt 20 -or $attempts -ge 4) { break }
        $previous=if ($state.peers.ContainsKey($peer.node_id)) {$state.peers[$peer.node_id]} else {@{}}
        foreach($adapter in @($snapshot.adapters|Where-Object {$_.up -and $_.root_id -in $peer.local_root_ids}|Select-Object -First 4)) {
            if ($clock.Elapsed.TotalSeconds -gt 20 -or $attempts -ge 4) { break }
            if ($adapter.local_llas.Count -ne 1) { continue }
            foreach($neighbor in $adapter.neighbors) {
                if ($clock.Elapsed.TotalSeconds -gt 20 -or $attempts -ge 4) { break }
                $attempts++
                $added=$false;$row=$null
                try {
                    $proof=Invoke-Usb4PartnerProbe $peer $adapter $adapter.local_llas[0] $neighbor.IPAddress $Policy.node_id $previous $Policy.ttl_seconds
                    $fresh=Get-Usb4HostSnapshot
                    $freshAdapter=@($fresh.adapters|Where-Object {$_.interface_guid -eq $adapter.interface_guid -and $_.interface_index -eq $adapter.interface_index -and $_.mac -eq $adapter.mac -and $_.root_id -eq $adapter.root_id -and $_.up})
                    if ($freshAdapter.Count -ne 1) { throw 'Local USB4 identity changed before planning.' }
                    if (@($fresh.persistent_addresses|Where-Object {(ConvertTo-Usb4Address $_.IPAddress) -eq $proof.windows_address}).Count) { throw 'Derived address has PersistentStore custody; preserve both stores and explicitly migrate this exact identity to ActiveStore before lease management.' }
                    $plan=Get-Usb4AddressDisposition $proof $adapter $fresh.addresses $state.addresses $Policy.adoptions
                    if ($plan.action -eq 'AddOwned' -and -not $mutate) {
                        $results.Add(@{state='PLANNED';action='AddOwned';edge_id=$proof.edge_id;interface_index=$adapter.interface_index;authenticated_data_path=$false});break
                    }
                    if ($plan.action -eq 'AddOwned') {
                        # A pending record never grants deletion custody after a crash.
                        # ActiveStore may disappear at reboot while its custody journal
                        # survives. Reuse that exact row rather than multiply leases.
                        $prior=@($state.addresses|Where-Object {$_.owned -and $_.address -eq $proof.windows_address -and
                            $_.edge_id -eq $proof.edge_id -and $_.interface_guid -eq $adapter.interface_guid -and
                            $_.mac -eq $adapter.mac -and $_.root_id -eq $adapter.root_id})
                        if ($prior.Count -gt 1) { throw 'Duplicate exact custody rows require journal reconciliation.' }
                        if ($prior.Count -eq 1) {
                            $row=$prior[0];$row.committed=$false;$row.qualified=$false;$row.interface_index=$adapter.interface_index
                            $row.expires_at=[DateTimeOffset]::UtcNow.AddSeconds($Policy.ttl_seconds).ToString('o')
                        } else {
                            $row=@{owned=$true;committed=$false;qualified=$false;edge_id=$proof.edge_id;peer_node_id=$peer.node_id;
                                interface_guid=$adapter.interface_guid;interface_index=$adapter.interface_index;mac=$adapter.mac;root_id=$adapter.root_id;
                                address=$proof.windows_address;prefix_length=64;expires_at=[DateTimeOffset]::UtcNow.AddSeconds($Policy.ttl_seconds).ToString('o')}
                            $state.addresses+=@($row)
                        }
                        Save-Usb4OwnedState $StatePath $state
                        $null=New-NetIPAddress -InterfaceIndex $adapter.interface_index -IPAddress $proof.windows_address -PrefixLength 64 -PolicyStore ActiveStore -ErrorAction Stop
                        $added=$true;$row.committed=$true;Save-Usb4OwnedState $StatePath $state
                        $deadline=[DateTimeOffset]::UtcNow.AddSeconds(3)
                        do {
                            $address=Get-NetIPAddress -InterfaceIndex $adapter.interface_index -AddressFamily IPv6 -ErrorAction Stop|Where-Object IPAddress -EQ $proof.windows_address
                            if ($address -and $address.AddressState -eq 'Preferred') { break }
                            Start-Sleep -Milliseconds 100
                        } while ([DateTimeOffset]::UtcNow -lt $deadline)
                        if (-not $address -or $address.AddressState -ne 'Preferred') { throw 'Derived address did not pass duplicate-address detection.' }
                    }
                    $canary=Invoke-Usb4PartnerProbe $peer $adapter $proof.windows_address $proof.peer_address $Policy.node_id $previous $Policy.ttl_seconds
                    foreach($field in @('edge_id','epoch','policy_sha256','boot_id','windows_address','peer_address','remote_ifindex','remote_mac')) {
                        if ($canary[$field] -cne $proof[$field]) { throw 'Enrolled partner changed between bootstrap and data canary.' }
                    }
                    if ($mutate) {
                        $state.peers[$peer.node_id]=@{epoch=$canary.epoch;policy_sha256=$canary.policy_sha256;boot_id=$canary.boot_id}
                        if ($plan.action -eq 'AdoptExplicitCustody') {
                            $row=@{owned=$true;committed=$true;qualified=$true;edge_id=$canary.edge_id;peer_node_id=$peer.node_id;
                                interface_guid=$adapter.interface_guid;interface_index=$adapter.interface_index;mac=$adapter.mac;root_id=$adapter.root_id;
                                address=$canary.windows_address;prefix_length=64;expires_at=''}
                            $state.addresses+=@($row)
                        }
                        foreach($owned in @($state.addresses|Where-Object {$_.owned -and $_.committed -and $_.address -eq $canary.windows_address -and $_.interface_guid -eq $adapter.interface_guid})) {
                            $owned.expires_at=[DateTimeOffset]::UtcNow.AddSeconds($Policy.ttl_seconds).ToString('o');$owned.qualified=$true;$owned.interface_index=$adapter.interface_index
                        }
                    }
                    $results.Add(@{state='AUTHENTICATED';edge_id=$canary.edge_id;interface_index=$adapter.interface_index;address=$canary.windows_address;
                        peer_address=$canary.peer_address;scope=$canary.scope;authenticated_data_path=$true;managed_address=($plan.action -ne 'PreserveUnowned');
                        expires_at=[DateTimeOffset]::UtcNow.AddSeconds($Policy.ttl_seconds).ToString('o')})
                    break
                } catch {
                    if ($added) {
                        # Roll back only this call's newly created address, after exact recheck.
                        $fresh=Get-Usb4HostSnapshot
                        $same=@($fresh.adapters|Where-Object {$_.interface_guid -eq $adapter.interface_guid -and $_.mac -eq $adapter.mac -and $_.root_id -eq $adapter.root_id -and $_.interface_index -eq $adapter.interface_index})
                        if ($same.Count -eq 1) {
                            Remove-NetIPAddress -InterfaceIndex $adapter.interface_index -IPAddress $row.address -PolicyStore ActiveStore -Confirm:$false -ErrorAction Stop
                            $remaining=@(Get-NetIPAddress -InterfaceIndex $adapter.interface_index -AddressFamily IPv6 -PolicyStore ActiveStore -ErrorAction Stop|Where-Object IPAddress -EQ $row.address)
                            if ($remaining.Count) { throw 'New owned address remains after rollback; custody retained for explicit recovery.' }
                            $state.addresses=@($state.addresses|Where-Object {-not($_.address -eq $row.address -and $_.interface_guid -eq $row.interface_guid)})
                        }
                    }
                    $results.Add(@{state='REFUSED';interface_index=$adapter.interface_index;peer_node_id=$peer.node_id;error=$_.Exception.Message;authenticated_data_path=$false})
                }
            }
        }
    }
    $state.results=@($results);$state.observed_at=[DateTimeOffset]::UtcNow.ToString('o')
    if ($mutate) { Save-Usb4OwnedState $StatePath $state }
    @{schema=1;scope='individual-enrolled-edges';global_topology_ready=$false;results=@($results);mutated=$mutate}
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if (-not $IsWindows) { throw 'Windows enrolled peer reconciliation requires Windows.' }
foreach($module in @('NetAdapter','NetTCPIP','PnpDevice')) { Import-Module (Join-Path $env:SystemRoot "System32/WindowsPowerShell/v1.0/Modules/$module/$module.psd1") }
. (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')
$mutex=$null;$acquired=$false
try {
    $policy=Get-Usb4PartnerPolicy $PolicyPath
    if ($Apply -and -not $DryRun -and -not $WhatIfPreference) {
        $mutex=[Threading.Mutex]::new($false,('Global\PC-AI.Usb4OwnedEdges.'+$policy.node_id))
        $acquired=$mutex.WaitOne(0)
        if (-not $acquired) { throw 'Another owned-edge reconciliation is active.' }
    }
    $result=Invoke-Usb4PartnerReconciliation -Policy $policy -StatePath $StatePath -Apply:$Apply -DryRun:($DryRun -or $WhatIfPreference)
    $result|ConvertTo-Json -Depth 20
    $success=@($result.results|Where-Object {$_.state -in @('AUTHENTICATED','PLANNED')}).Count -gt 0
    exit $(if($success){0}else{2})
} catch {
    @{state='REFUSED';authenticated_data_path=$false;error=$_.Exception.Message}|ConvertTo-Json -Depth 5
    exit 1
} finally {
    if ($acquired) {$mutex.ReleaseMutex()}
    if ($mutex) {$mutex.Dispose()}
}
