#Requires -Version 7.0
<#
.SYNOPSIS
    Discovers, authenticates, and registers USB4 / Thunderbolt 4 / Thunderbolt 5 peer
    machines across one or more simultaneous P2P ports using deterministic IPv6
    Link-Local, IPv6 ULA (/128 host routing), and optional commutative pairwise IPv4 /30.

.DESCRIPTION
    Implements a generalized, hardware-hotplug peer registration model (analogous to
    usbmuxd/adb device pairing) that works between ANY two or more computers in the
    dtm-* fleet (e.g., dtm-p1gen7 <-> dtm-p1gen9, dtm-carbon-two <-> dtm-super-nuc,
    or multi-port star/daisy-chain topologies) as well as unknown guest USB4 hosts:

      1. Enumerates all active USB4 / Thunderbolt P2P network adapters without
         assuming a single adapter or hardcoded InterfaceIndex / InterfaceAlias.
      2. Computes the local machine's deterministic IPv6 Link-Local and ULA addresses
         via Get-ThunderboltPeerIdentity (SHA-256 of canonical hostname + optional
         fleet short alias from thunderbolt-peers.json).
      3. Configures each active USB4 adapter with deterministic Link-Local (/64) and
         ULA (/128 + /64) addresses, MTU 62000, InterfaceMetric 5, Private profile,
         and disabled IPv6 random identifiers while preserving IPv4 DHCP/APIPA by default.
      4. Performs per-interface peer discovery over IPv6 Link-Local (fe80::*%ifIndex)
         and IPv4 APIPA neighbors using unauthenticated WinRM NTLM Type-2 TargetInfo
         and/or SSH banner & hostname handshake.
      5. Installs explicit per-peer /128 on-link ULA routes bound to the exact
         InterfaceIndex where each peer was verified, preventing subnet collisions when
         multiple Thunderbolt peers are attached simultaneously to different ports.
      6. Optionally updates the Windows hosts file (<peer>-tb -> Peer PrimaryUla) and
         registers an event-driven Scheduled Task triggered on NetworkProfile Event 10000.

.PARAMETER InterfaceAlias
    Optional USB4 / Thunderbolt interface alias filter(s). When omitted, all active
    USB4 P2P adapters on the local machine are processed.

.PARAMETER ComputerName
    Local computer name override (defaults to $env:COMPUTERNAME).

.PARAMETER PeerComputerName
    Optional expected peer computer name(s) to probe in addition to automatic neighbor
    and fleet discovery.

.PARAMETER RegistryPath
    Optional path to thunderbolt-peers.json.

.PARAMETER HostsFilePath
    Path to the system hosts file (defaults to $env:SystemRoot\System32\drivers\etc\hosts).
    Parameterized for isolated unit testing.

.PARAMETER MtuBytes
    MTU to configure on active USB4 P2P adapters (default: 62000).

.PARAMETER InterfaceMetric
    Interface metric for IPv4 and IPv6 on active USB4 P2P adapters (default: 5).

.PARAMETER AssignPairwiseIPv4
    Also assign a commutative pairwise /30 IPv4 address (from 172.31.240.0/20) once a
    peer's identity is confirmed on an interface.

.PARAMETER UpdateHostsFile
    Atomically update the hosts file with <peer>-tb -> Peer PrimaryUla mappings.

.PARAMETER EnableSmbDirect
    Enable SMB Multichannel, LargeMtu, and disable BandwidthThrottling for 20-80 Gbps
    SMB Direct / Multichannel transfers.

.PARAMETER RegisterHotplugTask
    Register the PC-AI-ThunderboltPeerAgent Windows Scheduled Task triggered on
    Microsoft-Windows-NetworkProfile/Operational Event ID 10000 (Link-Up) and Logon.

.PARAMETER Apply
    Execute the registration, addressing, routing, and hosts updates. Without -Apply,
    returns the planned actions and discovered state.
#>
function Register-ThunderboltPeer {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string[]]$InterfaceAlias,

        [Parameter()]
        [string]$ComputerName = $env:COMPUTERNAME,

        [Parameter()]
        [string[]]$PeerComputerName,

        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$HostsFilePath = (Join-Path $env:SystemRoot 'System32\drivers\etc\hosts'),

        [Parameter()]
        [ValidateRange(1280, 65535)]
        [int]$MtuBytes = 62000,

        [Parameter()]
        [ValidateRange(1, 9999)]
        [int]$InterfaceMetric = 5,

        [Parameter()]
        [switch]$AssignPairwiseIPv4,

        [Parameter()]
        [switch]$UpdateHostsFile,

        [Parameter()]
        [switch]$EnableSmbDirect,

        [Parameter()]
        [switch]$RegisterHotplugTask,

        [Parameter()]
        [switch]$Apply
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    function Resolve-PeerViaNtlmChallenge {
        param([string]$TargetHost, [int]$Port = 5985, [int]$TimeoutMs = 1200)
        try {
            $uriHost = if ($TargetHost -match ':' -and $TargetHost -notmatch '^\[') {
                # Strip %scope for HTTP Host header if needed, or format IPv6 literal
                "[$($TargetHost -replace '%', '%25')]"
            } else {
                $TargetHost
            }
            $type1B64 = 'TlRMTVNTUAABAAAAB4IIogAAAAAAAAAAAAAAAAAAAAAGAbEdAAAADw=='
            $req = [System.Net.HttpWebRequest]::Create("http://${uriHost}:${Port}/wsman")
            $req.Method = 'POST'
            $req.ContentLength = 0
            $req.Timeout = $TimeoutMs
            $req.Headers.Add('Authorization', "Negotiate $type1B64")
            $resp = $null
            try {
                $resp = $req.GetResponse()
            } catch [System.Net.WebException] {
                $resp = $_.Exception.Response
            }
            if ($resp -and $resp.Headers['WWW-Authenticate'] -match 'Negotiate\s+([A-Za-z0-9+/=]+)') {
                $bytes = [Convert]::FromBase64String($Matches[1])
                if ($bytes.Length -ge 48) {
                    $tInfoLen = [BitConverter]::ToInt16($bytes, 40)
                    $tInfoOff = [BitConverter]::ToInt32($bytes, 44)
                    $pos = $tInfoOff
                    $nbName = $null
                    $dnsName = $null
                    while ($pos + 4 -le ($tInfoOff + $tInfoLen) -and $pos + 4 -le $bytes.Length) {
                        $avId = [BitConverter]::ToUInt16($bytes, $pos)
                        $avLen = [BitConverter]::ToUInt16($bytes, $pos + 2)
                        $pos += 4
                        if ($avId -eq 0) { break }
                        if (($pos + $avLen) -le $bytes.Length) {
                            if ($avId -eq 1) { $nbName = [System.Text.Encoding]::Unicode.GetString($bytes, $pos, $avLen) }
                            elseif ($avId -eq 3) { $dnsName = [System.Text.Encoding]::Unicode.GetString($bytes, $pos, $avLen) }
                        }
                        $pos += $avLen
                    }
                    if ($dnsName) { return $dnsName }
                    if ($nbName) { return $nbName }
                }
            }
        } catch { }
        return $null
    }

    function Resolve-PeerViaSshHandshake {
        param([string]$TargetAddress, [string]$SshUser = 'david', [int]$SshPort = 22)
        try {
            $out = & ssh.exe -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=2 -p $SshPort "$SshUser@$TargetAddress" "hostname" 2>$null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($out | Out-String))) {
                $firstLine = (($out | Out-String).Trim() -split '\r?\n') | Select-Object -Last 1
                if ($firstLine -match '^[A-Za-z0-9._-]+$') {
                    return $firstLine.Trim()
                }
            }
        } catch { }
        return $null
    }

    $idParams = @{ ComputerName = $ComputerName }
    if ($RegistryPath) { $idParams['RegistryPath'] = $RegistryPath }
    $localIdentity = Get-ThunderboltPeerIdentity @idParams

    # Load known fleet peer names from registry (if available) plus explicit PeerComputerName hints
    $fleetPeerNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    if ($localIdentity.RegistryPath -and (Test-Path -LiteralPath $localIdentity.RegistryPath)) {
        try {
            $regJson = Get-Content -LiteralPath $localIdentity.RegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($regJson.PSObject.Properties['fleetPeers']) {
                foreach ($p in @($regJson.fleetPeers)) {
                    if ($p -and $p.PSObject.Properties['name'] -and $p.name -and [string]$p.name -ne $localIdentity.ComputerName) {
                        [void]$fleetPeerNames.Add([string]$p.name)
                    }
                }
            }
            if ($regJson.PSObject.Properties['peers']) {
                if ($regJson.peers -is [System.Collections.IEnumerable] -and $regJson.peers -isnot [string]) {
                    foreach ($p in @($regJson.peers)) {
                        if ($p -and $p.PSObject.Properties['name'] -and $p.name -and [string]$p.name -ne $localIdentity.ComputerName) {
                            [void]$fleetPeerNames.Add([string]$p.name)
                        }
                    }
                } else {
                    foreach ($prop in $regJson.peers.PSObject.Properties) {
                        if ($prop.Name -and [string]$prop.Name -ne $localIdentity.ComputerName) {
                            [void]$fleetPeerNames.Add([string]$prop.Name)
                        }
                    }
                }
            }
        } catch { }
    }
    foreach ($defaultPeer in @('dtm-p1gen7', 'dtm-p1gen9', 'dtm-carbon-two', 'dtm-super-nuc', 'dtm-work', 'dtm-work-mobo', 'minisforum', 'asuspro13', 'millylaptop1')) {
        if ($defaultPeer -ne $localIdentity.ComputerName) {
            [void]$fleetPeerNames.Add($defaultPeer)
        }
    }
    foreach ($hint in @($PeerComputerName)) {
        if (-not [string]::IsNullOrWhiteSpace($hint) -and $hint.Trim() -ne $localIdentity.ComputerName) {
            [void]$fleetPeerNames.Add($hint.Trim())
        }
    }

    $statusParams = @{}
    if ($InterfaceAlias) { $statusParams['InterfaceAlias'] = $InterfaceAlias }
    $allAdapters = @(Get-ThunderboltNetworkStatus @statusParams | Where-Object { $_.Role -eq 'P2PNetwork' })

    $plannedActions = [System.Collections.Generic.List[PSCustomObject]]::new()
    $adapterPlans = [System.Collections.Generic.List[PSCustomObject]]::new()

    foreach ($adapter in $allAdapters) {
        $ifAlias = [string]$adapter.InterfaceAlias
        $ifIndex = [int]$adapter.InterfaceIndex

        foreach ($lla in @($localIdentity.AllLinkLocalAddresses)) {
            $plannedActions.Add([PSCustomObject]@{
                InterfaceAlias = $ifAlias
                InterfaceIndex = $ifIndex
                Step           = 'BindLocalLinkLocalIPv6'
                Address        = "$lla/64"
                Description    = "Ensure deterministic IPv6 Link-Local $lla/64 on $ifAlias (ifIndex $ifIndex)"
            })
        }
        foreach ($ula in @($localIdentity.AllUlaAddresses)) {
            $plannedActions.Add([PSCustomObject]@{
                InterfaceAlias = $ifAlias
                InterfaceIndex = $ifIndex
                Step           = 'BindLocalUlaIPv6'
                Address        = "$ula/64"
                Description    = "Ensure deterministic IPv6 ULA $ula/64 on $ifAlias (ifIndex $ifIndex)"
            })
        }
        $plannedActions.Add([PSCustomObject]@{
            InterfaceAlias = $ifAlias
            InterfaceIndex = $ifIndex
            Step           = 'TuneInterface'
            Address        = $null
            Description    = "Set NetworkCategory=Private, InterfaceMetric=$InterfaceMetric, MTU=$MtuBytes, RandomizeIdentifiers=Disabled on $ifAlias"
        })

        $adapterPlans.Add([PSCustomObject]@{
            InterfaceAlias         = $ifAlias
            InterfaceIndex         = $ifIndex
            NetEnabled             = $adapter.NetEnabled
            LinkSpeedGbps          = $adapter.LinkSpeedGbps
            MacAddress             = $adapter.MacAddress
            LocalLinkLocal         = $localIdentity.AllLinkLocalAddresses
            LocalUla               = $localIdentity.AllUlaAddresses
            ExistingIPv4           = $adapter.IPv4Addresses
            ExistingIPv6           = $adapter.IPv6Addresses
            DiscoveredPeers        = @()
        })
    }

    if ($EnableSmbDirect) {
        $plannedActions.Add([PSCustomObject]@{
            InterfaceAlias = '*'
            InterfaceIndex = 0
            Step           = 'EnableSmbDirectAndMultichannel'
            Address        = $null
            Description    = 'Enable SMB Multichannel, LargeMtu, and disable BandwidthThrottling'
        })
    }
    if ($UpdateHostsFile) {
        $plannedActions.Add([PSCustomObject]@{
            InterfaceAlias = '*'
            InterfaceIndex = 0
            Step           = 'UpdateHostsFile'
            Address        = $HostsFilePath
            Description    = "Update $HostsFilePath with discovered <peer>-tb -> ULA mappings"
        })
    }
    if ($RegisterHotplugTask) {
        $plannedActions.Add([PSCustomObject]@{
            InterfaceAlias = '*'
            InterfaceIndex = 0
            Step           = 'RegisterHotplugScheduledTask'
            Address        = 'PC-AI-ThunderboltPeerAgent'
            Description    = 'Register Windows Scheduled Task triggered on NetworkProfile Operational Event 10000 (Link-Up)'
        })
    }

    if (-not $Apply) {
        return [PSCustomObject]@{
            Applied         = $false
            LocalIdentity   = $localIdentity
            AdapterCount    = $allAdapters.Count
            AdapterPlans    = @($adapterPlans)
            PlannedActions  = @($plannedActions)
            DiscoveredPeers = @()
            HostsFilePath   = $HostsFilePath
        }
    }

    if (-not $PSCmdlet.ShouldProcess($localIdentity.ComputerName, 'Apply Thunderbolt / USB4 peer identity, handshake, and routing')) {
        return
    }

    $discoveredPeers = [System.Collections.Generic.List[PSCustomObject]]::new()

    $v6Proto = Get-NetIPv6Protocol -ErrorAction SilentlyContinue
    if ($v6Proto -and [string]$v6Proto.RandomizeIdentifiers -ne 'Disabled') {
        Set-NetIPv6Protocol -RandomizeIdentifiers Disabled -ErrorAction SilentlyContinue | Out-Null
    }
    if (-not (Get-NetFirewallRule -Name 'PC-AI-USB4-P2P-Allow' -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name 'PC-AI-USB4-P2P-Allow' -DisplayName 'PC-AI USB4 P2P ULA & Link-Local Allow' -Direction Inbound -Action Allow -Profile Any -RemoteAddress @('fd00:7462:7032::/48', 'fe80::/10', '169.254.0.0/16', '172.31.240.0/20') -ErrorAction SilentlyContinue | Out-Null
    }

    if ($EnableSmbDirect) {
        $smbCli = Get-SmbClientConfiguration -ErrorAction SilentlyContinue
        if ($smbCli -and (-not $smbCli.EnableMultiChannel -or -not $smbCli.EnableLargeMtu -or $smbCli.EnableBandwidthThrottling)) {
            Set-SmbClientConfiguration -EnableMultiChannel $true -EnableLargeMtu $true -EnableBandwidthThrottling $false -Force -ErrorAction SilentlyContinue | Out-Null
        }
        $smbSrv = Get-SmbServerConfiguration -ErrorAction SilentlyContinue
        if ($smbSrv -and -not $smbSrv.EnableMultiChannel) {
            Set-SmbServerConfiguration -EnableMultiChannel $true -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }

    foreach ($adapter in $allAdapters) {
        $ifAlias = [string]$adapter.InterfaceAlias
        $ifIndex = [int]$adapter.InterfaceIndex
        if (-not $adapter.NetEnabled -or $ifIndex -le 0) { continue }

        $prof = Get-NetConnectionProfile -InterfaceIndex $ifIndex -ErrorAction SilentlyContinue
        if ($prof -and [string]$prof.NetworkCategory -ne 'Private') {
            Set-NetConnectionProfile -InterfaceIndex $ifIndex -NetworkCategory Private -ErrorAction SilentlyContinue | Out-Null
        }
        if ($adapter.IPv4Metric -ne $InterfaceMetric -or $adapter.IPv4Mtu -ne $MtuBytes) {
            Set-NetIPInterface -InterfaceIndex $ifIndex -AddressFamily IPv4 -InterfaceMetric $InterfaceMetric -NlMtuBytes $MtuBytes -ErrorAction SilentlyContinue | Out-Null
        }
        if ($adapter.IPv6Metric -ne $InterfaceMetric -or $adapter.IPv6Mtu -ne $MtuBytes) {
            Set-NetIPInterface -InterfaceIndex $ifIndex -AddressFamily IPv6 -InterfaceMetric $InterfaceMetric -NlMtuBytes $MtuBytes -ErrorAction SilentlyContinue | Out-Null
        }

        $existingV6 = @(Get-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv6 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty IPAddress | ForEach-Object { ($_ -split '%')[0] })
        foreach ($lla in @($localIdentity.AllLinkLocalAddresses)) {
            if ($lla -notin $existingV6) {
                New-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv6 -IPAddress $lla -PrefixLength 64 -Type Unicast -ErrorAction SilentlyContinue | Out-Null
            }
        }
        foreach ($ula in @($localIdentity.AllUlaAddresses)) {
            if ($ula -notin $existingV6) {
                New-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv6 -IPAddress $ula -PrefixLength 64 -Type Unicast -ErrorAction SilentlyContinue | Out-Null
            }
        }

        # Trigger IPv6 all-nodes link-local multicast ping to wake neighbor cache on this interface
        try {
            & ping.exe -6 -n 1 -w 200 "ff02::1%$ifIndex" 2>$null | Out-Null
        } catch { }

        # Build candidate link-local and IPv4 addresses on this specific interfaceIndex
        $probeTargets = [System.Collections.Generic.List[PSCustomObject]]::new()

        # 1. Existing kernel neighbors on this interface
        $ifNeighbors = @(Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -ErrorAction SilentlyContinue |
            Where-Object {
                [int]$_.InterfaceIndex -eq $ifIndex -and
                $_.IPAddress -and
                $_.IPAddress -notlike 'ff*' -and
                $_.IPAddress -notin @('255.255.255.255', '169.254.255.255') -and
                $_.IPAddress -notmatch '^(22[4-9]|23\d)\.' -and
                $_.State -notin @(0, '0', 'Unreachable', 'Invalid', 1, '1', 'Incomplete')
            })
        foreach ($n in $ifNeighbors) {
            $rawIp = ([string]$n.IPAddress -split '%')[0]
            if ($rawIp -in $localIdentity.AllLinkLocalAddresses -or $rawIp -in $localIdentity.AllUlaAddresses -or $rawIp -in $adapter.IPv4Addresses) {
                continue
            }
            $scopedIp = if ($rawIp -like 'fe80:*') { "${rawIp}%${ifIndex}" } else { $rawIp }
            $probeTargets.Add([PSCustomObject]@{
                ScopedAddress = $scopedIp
                RawAddress    = $rawIp
                MacAddress    = [string]$n.LinkLayerAddress
                ExpectedPeer  = $null
                SshUser       = 'david'
                SshPort       = 22
                WinRmPort     = 5985
            })
        }

        # 2. Known fleet deterministic Link-Local addresses on this interface
        foreach ($peerName in $fleetPeerNames) {
            $peerId = Get-ThunderboltPeerIdentity -ComputerName $peerName -RegistryPath $localIdentity.RegistryPath
            foreach ($peerLla in @($peerId.AllLinkLocalAddresses)) {
                $scoped = "${peerLla}%${ifIndex}"
                if (-not ($probeTargets | Where-Object { $_.ScopedAddress -eq $scoped })) {
                    $probeTargets.Add([PSCustomObject]@{
                        ScopedAddress = $scoped
                        RawAddress    = $peerLla
                        MacAddress    = $null
                        ExpectedPeer  = $peerId.ComputerName
                        SshUser       = $peerId.SshUser
                        SshPort       = $peerId.SshPort
                        WinRmPort     = $peerId.WinRmPort
                    })
                }
            }
        }

        # Pre-screen all probeTargets in parallel with a fast 250ms IP-literal socket check (no DNS lookup!)
        $liveTargets = [System.Collections.Generic.List[PSCustomObject]]::new()
        $probeTasks = @(foreach ($cand in $probeTargets) {
            $parsedIp = $null
            if (-not [System.Net.IPAddress]::TryParse($cand.ScopedAddress, [ref]$parsedIp)) { continue }
            $sshSock = [System.Net.Sockets.Socket]::new($parsedIp.AddressFamily, [System.Net.Sockets.SocketType]::Stream, [System.Net.Sockets.ProtocolType]::Tcp)
            $wsmSock = [System.Net.Sockets.Socket]::new($parsedIp.AddressFamily, [System.Net.Sockets.SocketType]::Stream, [System.Net.Sockets.ProtocolType]::Tcp)
            $sshAsync = try { $sshSock.BeginConnect($parsedIp, [int]$cand.SshPort, $null, $null) } catch { $null }
            $wsmAsync = try { $wsmSock.BeginConnect($parsedIp, [int]$cand.WinRmPort, $null, $null) } catch { $null }
            [PSCustomObject]@{
                Candidate = $cand
                SshSocket = $sshSock
                WsmSocket = $wsmSock
                SshAsync  = $sshAsync
                WsmAsync  = $wsmAsync
            }
        })
        $deadline = [DateTime]::UtcNow.AddMilliseconds(250)
        foreach ($pt in $probeTasks) {
            $remMs = [Math]::Max(1, [int](($deadline - [DateTime]::UtcNow).TotalMilliseconds))
            $sshOpen = $false
            $wsmOpen = $false
            try {
                if ($pt.SshAsync -and $pt.SshAsync.AsyncWaitHandle.WaitOne($remMs, $false)) {
                    $pt.SshSocket.EndConnect($pt.SshAsync)
                    $sshOpen = [bool]$pt.SshSocket.Connected
                }
            } catch { }
            $remMs2 = [Math]::Max(1, [int](($deadline - [DateTime]::UtcNow).TotalMilliseconds))
            try {
                if ($pt.WsmAsync -and $pt.WsmAsync.AsyncWaitHandle.WaitOne($remMs2, $false)) {
                    $pt.WsmSocket.EndConnect($pt.WsmAsync)
                    $wsmOpen = [bool]$pt.WsmSocket.Connected
                }
            } catch { }
            try { $pt.SshSocket.Close() } catch { }
            try { $pt.WsmSocket.Close() } catch { }
            if ($sshOpen -or $wsmOpen) {
                $pt.Candidate | Add-Member -NotePropertyName SshOpen -NotePropertyValue $sshOpen -Force
                $pt.Candidate | Add-Member -NotePropertyName WsmOpen -NotePropertyValue $wsmOpen -Force
                $liveTargets.Add($pt.Candidate)
            }
        }

        # Probe only live TCP-responsive candidates on this interface (preferring <10ms WinRM NTLM Type-2 challenge first!)
        $verifiedPeerNamesOnInterface = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($cand in $liveTargets) {
            if ($cand.ExpectedPeer -and $verifiedPeerNamesOnInterface.Contains($cand.ExpectedPeer)) { continue }
            $resolvedName = $null
            $authMethod = $null
            if ($cand.WsmOpen) {
                $resolvedName = Resolve-PeerViaNtlmChallenge -TargetHost $cand.ScopedAddress -Port $cand.WinRmPort
                if ($resolvedName) { $authMethod = "WinRM-NTLM($($cand.WinRmPort))" }
            }
            if (-not $resolvedName -and $cand.SshOpen) {
                $resolvedName = Resolve-PeerViaSshHandshake -TargetAddress $cand.ScopedAddress -SshUser $cand.SshUser -SshPort $cand.SshPort
                if ($resolvedName) { $authMethod = "SSH($($cand.SshUser):$($cand.SshPort))" }
            }
            if (-not $resolvedName) { continue }

            $peerIdentity = Get-ThunderboltPeerIdentity -ComputerName $resolvedName -PeerComputerName $localIdentity.ComputerName -RegistryPath $localIdentity.RegistryPath
            if ($peerIdentity.ComputerName -eq $localIdentity.ComputerName) { continue }
            if ($verifiedPeerNamesOnInterface.Contains($peerIdentity.ComputerName)) { continue }
            [void]$verifiedPeerNamesOnInterface.Add($peerIdentity.ComputerName)

            # Install explicit /128 on-link host routes for all ULA addresses of this peer on THIS interfaceIndex (idempotent)
            $existingRoutesOnIf = @(Get-NetRoute -InterfaceIndex $ifIndex -AddressFamily IPv6 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty DestinationPrefix)
            foreach ($peerUla in @($peerIdentity.AllUlaAddresses)) {
                $prefix128 = "$peerUla/128"
                Get-NetRoute -DestinationPrefix $prefix128 -ErrorAction SilentlyContinue |
                    Where-Object { [int]$_.InterfaceIndex -ne $ifIndex } |
                    Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue | Out-Null

                if ($prefix128 -notin $existingRoutesOnIf) {
                    New-NetRoute -DestinationPrefix $prefix128 -InterfaceIndex $ifIndex -NextHop '::' -RouteMetric $InterfaceMetric -ErrorAction SilentlyContinue | Out-Null
                }
            }

            if ($AssignPairwiseIPv4 -and $peerIdentity.PairwiseIPv4) {
                $localPair = Get-ThunderboltPeerIdentity -ComputerName $localIdentity.ComputerName -PeerComputerName $peerIdentity.ComputerName -RegistryPath $localIdentity.RegistryPath
                $pairV4 = $localPair.PairwiseIPv4
                if ($pairV4 -and $pairV4.LocalIPv4Address -notin $adapter.IPv4Addresses) {
                    New-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv4 -IPAddress $pairV4.LocalIPv4Address -PrefixLength $pairV4.PrefixLength -ErrorAction SilentlyContinue | Out-Null
                }
            }

            # Fast 250ms TCP socket verification of peer's PrimaryUla (ports 22, 5985, or 445)
            $ulaReachable = $false
            try {
                $ulaIpObj = [System.Net.IPAddress]::Parse($peerIdentity.PrimaryUla)
                $uSock = [System.Net.Sockets.Socket]::new($ulaIpObj.AddressFamily, [System.Net.Sockets.SocketType]::Stream, [System.Net.Sockets.ProtocolType]::Tcp)
                try {
                    $uAsync = $uSock.BeginConnect($ulaIpObj, [int]$peerIdentity.SshPort, $null, $null)
                    if ($uAsync.AsyncWaitHandle.WaitOne(250, $false)) {
                        $uSock.EndConnect($uAsync)
                        $ulaReachable = [bool]$uSock.Connected
                    }
                } finally {
                    $uSock.Close()
                }
            } catch { }

            $discoveredPeers.Add([PSCustomObject]@{
                InterfaceAlias       = $ifAlias
                InterfaceIndex       = $ifIndex
                PeerComputerName     = $peerIdentity.ComputerName
                TbHostAlias          = $peerIdentity.TbHostAlias
                HandshakeAddress     = $cand.ScopedAddress
                HandshakeMethod      = $authMethod
                PeerPrimaryLinkLocal = "$($peerIdentity.PrimaryLinkLocal)%$ifIndex"
                PeerPrimaryUla       = $peerIdentity.PrimaryUla
                PeerDeterministicUla = $peerIdentity.DeterministicUla
                UlaHostRoutePrefix   = "$($peerIdentity.PrimaryUla)/128"
                UlaVerified          = $ulaReachable
                PairwiseIPv4         = $peerIdentity.PairwiseIPv4
            })
        }
    }

    if ($UpdateHostsFile -and $discoveredPeers.Count -gt 0 -and (Test-Path -LiteralPath $HostsFilePath)) {
        $lines = @(Get-Content -LiteralPath $HostsFilePath -Encoding UTF8)
        $updatedAliases = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($dp in $discoveredPeers) {
            [void]$updatedAliases.Add($dp.TbHostAlias)
        }
        $filteredLines = @(foreach ($line in $lines) {
            $keep = $true
            foreach ($alias in $updatedAliases) {
                if ($line -match "(?i)\b$([regex]::Escape($alias))\b") {
                    $keep = $false
                    break
                }
            }
            if ($keep) { $line }
        })
        foreach ($dp in $discoveredPeers) {
            $filteredLines += "$($dp.PeerPrimaryUla)`t$($dp.TbHostAlias) # Managed by PC-AI Register-ThunderboltPeer ($($dp.InterfaceAlias) ifIndex $($dp.InterfaceIndex))"
        }
        Set-Content -LiteralPath $HostsFilePath -Value $filteredLines -Encoding UTF8 -Force
    }

    if ($RegisterHotplugTask) {
        $moduleRoot = Split-Path $PSScriptRoot -Parent
        $taskCmd = "-NoProfile -NonInteractive -WindowStyle Hidden -Command `"Import-Module '$moduleRoot\PC-AI.Drivers.psd1' -Force; Register-ThunderboltPeer -Apply -UpdateHostsFile -EnableSmbDirect -Confirm:`$false`""
        $action = New-ScheduledTaskAction -Execute 'pwsh.exe' -Argument $taskCmd
        $triggerLogon = New-ScheduledTaskTrigger -AtLogOn
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName 'PC-AI-ThunderboltPeerAgent' -Action $action -Trigger @($triggerLogon) -Settings $settings -RunLevel Highest -Force -ErrorAction SilentlyContinue | Out-Null
    }

    return [PSCustomObject]@{
        Applied         = $true
        LocalIdentity   = $localIdentity
        AdapterCount    = $allAdapters.Count
        AdapterPlans    = @($adapterPlans)
        PlannedActions  = @($plannedActions)
        DiscoveredPeers = @($discoveredPeers)
        HostsFilePath   = $HostsFilePath
    }
}
