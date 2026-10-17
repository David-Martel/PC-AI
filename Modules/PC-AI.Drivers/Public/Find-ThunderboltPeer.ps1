#Requires -Version 7.0
function Find-ThunderboltPeer {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string[]]$ComputerNameCandidates = @(
            'dtm-p1gen9',
            'dtm-p1gen7',
            'dtm-carbon-two',
            'dtm-super-nuc',
            'dtm-supernuc',
            'dtm-work',
            'dtm-work-mobo',
            'minisforum',
            'asuspro13',
            'millylaptop1',
            'DESKTOP-86FC90B'
        ),

        [Parameter()]
        [int]$WinRmPort = 5985,

        [Parameter()]
        [int]$SshPort = 22,

        [Parameter()]
        [ValidateRange(250, 10000)]
        [int]$TcpTimeoutMs = 1500,

        [Parameter()]
        [switch]$IncludeRaw
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    # ── Phase 1: Discover Thunderbolt/USB4 adapters via CIM ──────────────
    # Note: Win32_NetworkAdapter.Index is the WMI DeviceID, whereas InterfaceIndex
    # is the TCP/IP stack interface index used by MSFT_NetIPAddress/MSFT_NetNeighbor.
    $adapters = @(Get-CimInstance -ClassName Win32_NetworkAdapter -ErrorAction SilentlyContinue |
        Where-Object {
            $_.NetConnectionID -and (
                $_.Description -match 'USB4|Thunderbolt|P2P' -or
                $_.Name -match 'USB4|Thunderbolt|P2P' -or
                $_.PNPDeviceID -match 'PROT_USB4NET|THUNDERBOLT|USB4'
            )
        } |
        Sort-Object Index |
        ForEach-Object {
            $propIfIndex = $_.PSObject.Properties['InterfaceIndex']
            $resolvedIfIndex = if ($propIfIndex -and $null -ne $propIfIndex.Value -and [int]$propIfIndex.Value -gt 0) {
                [int]$propIfIndex.Value
            } else {
                [int]$_.Index
            }
            [PSCustomObject]@{
                Name                 = $_.Name
                InterfaceDescription = $_.Description
                InterfaceAlias       = $_.NetConnectionID
                Status               = if ($_.NetEnabled) { 'Up' } else { 'Disconnected' }
                MacAddress           = $_.MACAddress
                LinkSpeed            = $_.Speed
                ifIndex              = $resolvedIfIndex
                WmiIndex             = [int]$_.Index
                PnpDeviceId          = $_.PNPDeviceID
            }
        })

    # ── Phase 1b: Discover USB4/Thunderbolt Host Routers & Dock/Hub Topology ──
    $topologyRouters = @(Get-CimInstance -ClassName Win32_PnPEntity -ErrorAction SilentlyContinue |
        Where-Object {
            $_.PNPDeviceID -match '^USB4\\VID_' -or
            $_.Name -match 'USB4\(TM\) Host Router|USB4.*Device Router|Thunderbolt.*Router'
        } |
        ForEach-Object {
            [PSCustomObject]@{
                Name        = $_.Name
                PnpDeviceId = $_.PNPDeviceID
                Status      = $_.Status
                Role        = if ($_.Name -match 'Host Router') { 'HostRouter' } elseif ($_.Name -match 'Device Router|Hub|Dock') { 'DeviceRouter' } else { 'USB4Peripheral' }
            }
        })

    # ── Phase 2: Gather IP addresses, neighbors, and APIPA route conflicts ──
    $interfaceIndexes = @($adapters | ForEach-Object { [int]$_.ifIndex })
    $localAddresses = @()
    $neighbors = @()
    $apipaRouteConflicts = @()

    if ($interfaceIndexes.Count -gt 0) {
        $indexSet = [System.Collections.Generic.HashSet[int]]::new()
        foreach ($idx in $interfaceIndexes) { [void]$indexSet.Add($idx) }

        # IP addresses via StandardCimv2
        $localAddresses = @(
            Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetIPAddress -ErrorAction SilentlyContinue |
            Where-Object { $indexSet.Contains([int]$_.InterfaceIndex) } |
            ForEach-Object {
                $family = switch ([int]$_.AddressFamily) { 2 { 'IPv4' }; 23 { 'IPv6' }; default { 'Other' } }
                [PSCustomObject]@{
                    InterfaceIndex  = [int]$_.InterfaceIndex
                    InterfaceAlias  = $_.InterfaceAlias
                    AddressFamily   = $family
                    IPAddress       = $_.IPAddress
                    PrefixLength    = $_.PrefixLength
                }
            })

        # Neighbors via StandardCimv2
        $neighbors = @(
            Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -ErrorAction SilentlyContinue |
            Where-Object {
                $indexSet.Contains([int]$_.InterfaceIndex) -and
                $_.IPAddress -and
                $_.State -notin @(0, '0', 'Unreachable', 'Invalid', 1, '1', 'Incomplete') -and
                $_.IPAddress -notin @('255.255.255.255', '169.254.255.255') -and
                $_.IPAddress -notmatch '^(22[4-9]|23\d)\.' -and
                $_.IPAddress -notlike 'ff*' -and
                $_.LinkLayerAddress -and
                $_.LinkLayerAddress -notin @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff') -and
                $_.IPAddress -ne 'ff02::1'
            } |
            ForEach-Object {
                [PSCustomObject]@{
                    InterfaceIndex   = [int]$_.InterfaceIndex
                    InterfaceAlias   = $_.InterfaceAlias
                    IPAddress        = $_.IPAddress
                    LinkLayerAddress = $_.LinkLayerAddress
                    State            = [string]$_.State
                }
            })

        # Detect competing 169.254.0.0/16 routes on non-USB4 interfaces (e.g. vEthernet (WSL))
        $allApipaRoutes = @(
            Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetRoute -ErrorAction SilentlyContinue |
            Where-Object { $_.DestinationPrefix -eq '169.254.0.0/16' }
        )
        $allIpInterfaces = @{}
        foreach ($ipIf in @(Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetIPInterface -ErrorAction SilentlyContinue | Where-Object { $_.AddressFamily -eq 2 })) {
            $allIpInterfaces[[int]$ipIf.InterfaceIndex] = $ipIf
        }
        $usb4ApipaMetrics = @(foreach ($idx in $interfaceIndexes) {
            if ($allIpInterfaces.ContainsKey($idx)) {
                [int]$allIpInterfaces[$idx].InterfaceMetric
            }
        })
        $minUsb4Metric = if ($usb4ApipaMetrics.Count -gt 0) { ($usb4ApipaMetrics | Measure-Object -Minimum).Minimum } else { 15 }
        foreach ($r in $allApipaRoutes) {
            $rIdx = [int]$r.InterfaceIndex
            if (-not $indexSet.Contains($rIdx) -and $allIpInterfaces.ContainsKey($rIdx)) {
                $otherIf = $allIpInterfaces[$rIdx]
                $otherEff = [int]$r.RouteMetric + [int]$otherIf.InterfaceMetric
                $usb4Eff  = 256 + $minUsb4Metric
                if ($otherEff -lt $usb4Eff -or ($otherEff -eq $usb4Eff -and $rIdx -lt ($interfaceIndexes | Measure-Object -Minimum).Minimum)) {
                    $apipaRouteConflicts += [PSCustomObject]@{
                        ConflictingInterfaceAlias = $otherIf.InterfaceAlias
                        ConflictingInterfaceIndex = $rIdx
                        ConflictingInterfaceMetric = [int]$otherIf.InterfaceMetric
                        Usb4InterfaceMetric       = $minUsb4Metric
                        Recommendation            = "Lower USB4 InterfaceMetric to 5 or raise '$($otherIf.InterfaceAlias)' InterfaceMetric so 169.254.0.0/16 traffic routes over USB4 P2P."
                    }
                }
            }
        }
    }

    # ── Phase 3: Build candidate target set & per-interface source IP map ──
    $candidateTargets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $sourceIpByTarget = @{}
    $localIpv4ByIfIndex = @{}
    foreach ($addr in $localAddresses) {
        if ($addr.AddressFamily -eq 'IPv4' -and $addr.IPAddress -match '^\d+\.\d+\.\d+\.\d+$') {
            if (-not $localIpv4ByIfIndex.ContainsKey([int]$addr.InterfaceIndex)) {
                $localIpv4ByIfIndex[[int]$addr.InterfaceIndex] = [string]$addr.IPAddress
            }
        }
    }
    foreach ($name in $ComputerNameCandidates) {
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            [void]$candidateTargets.Add($name.Trim())
        }
    }
    foreach ($neighbor in $neighbors) {
        if (-not [string]::IsNullOrWhiteSpace($neighbor.IPAddress) -and $neighbor.IPAddress -match '^\d+\.\d+\.\d+\.\d+$') {
            $nIp = [string]$neighbor.IPAddress
            [void]$candidateTargets.Add($nIp)
            $nIfIdx = [int]$neighbor.InterfaceIndex
            if ($localIpv4ByIfIndex.ContainsKey($nIfIdx)) {
                $sourceIpByTarget[$nIp] = $localIpv4ByIfIndex[$nIfIdx]
            }
        }
    }

    # ── Phase 4: DNS resolution (async with timeout) ────────────────────
    $dnsTimeoutMs = [Math]::Min($TcpTimeoutMs, 2000)
    $dnsTasks = [System.Collections.Generic.Dictionary[string, System.Threading.Tasks.Task]]::new()
    foreach ($target in $candidateTargets) {
        $dnsTasks[$target] = [System.Net.Dns]::GetHostAddressesAsync($target)
    }

    $nameResolution = @(foreach ($target in @($candidateTargets)) {
        $task = $dnsTasks[$target]
        $resolved = @()
        try {
            if ($task.Wait($dnsTimeoutMs)) {
                $resolved = @($task.Result | ForEach-Object { $_.IPAddressToString })
            }
        } catch { }
        [PSCustomObject]@{
            Target          = $target
            Resolved        = $resolved.Count -gt 0
            ResolvedAddress = $resolved
        }
    })

    # ── Phase 5: Source-bound TCP WinRM/SSH/SMB/TBShare probe & NTLM identity ──
    $usb4LocalIpv4 = @(
        $localAddresses |
        Where-Object { $_.AddressFamily -eq 'IPv4' -and $_.IPAddress -match '^\d+\.\d+\.\d+\.\d+$' } |
        Select-Object -ExpandProperty IPAddress
    )
    $primarySourceIp = if ($usb4LocalIpv4.Count -gt 0) { [string]$usb4LocalIpv4[0] } else { $null }
    $sourceMapCapture = $sourceIpByTarget
    $portCapture = $WinRmPort
    $sshPortCapture = $SshPort
    $timeoutCapture = $TcpTimeoutMs

    $winRmCandidates = @($candidateTargets | ForEach-Object -ThrottleLimit 8 -Parallel {
        $target = $_
        $winRmPortLocal = $using:portCapture
        $sshPortLocal = $using:sshPortCapture
        $timeoutLocal = $using:timeoutCapture
        $sourceMapLocal = $using:sourceMapCapture
        $sourceIpLocal = if ($sourceMapLocal -and $sourceMapLocal.ContainsKey($target)) {
            [string]$sourceMapLocal[$target]
        } else {
            $using:primarySourceIp
        }

        function Test-BoundTcpPort {
            param([string]$HostTarget, [int]$Port, [int]$Timeout, [string]$SourceIp)
            $client = $null
            try {
                if ($SourceIp -and $HostTarget -match '^169\.254\.') {
                    $localEp = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Parse($SourceIp), 0)
                    $client = [System.Net.Sockets.TcpClient]::new($localEp)
                } else {
                    $client = [System.Net.Sockets.TcpClient]::new()
                }
                $async = $client.BeginConnect($HostTarget, $Port, $null, $null)
                if ($async.AsyncWaitHandle.WaitOne($Timeout, $false)) {
                    $client.EndConnect($async)
                    return [bool]$client.Connected
                }
                return $false
            } catch {
                return $false
            } finally {
                if ($null -ne $client) { $client.Close() }
            }
        }

        function Get-WsManNtlmPeerIdentity {
            param([string]$HostTarget, [int]$Port, [int]$Timeout)
            try {
                $type1B64 = 'TlRMTVNTUAABAAAAB4IIogAAAAAAAAAAAAAAAAAAAAAGAbEdAAAADw=='
                $req = [System.Net.HttpWebRequest]::Create("http://${HostTarget}:${Port}/wsman")
                $req.Method = 'POST'
                $req.ContentLength = 0
                $req.Timeout = [Math]::Max($Timeout, 1500)
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
                        if ($dnsName -or $nbName) {
                            return [PSCustomObject]@{
                                NetBiosComputerName = $nbName
                                DnsComputerName     = $dnsName
                            }
                        }
                    }
                }
            } catch { }
            return $null
        }

        $tcpReachable = Test-BoundTcpPort -HostTarget $target -Port $winRmPortLocal -Timeout $timeoutLocal -SourceIp $sourceIpLocal
        $sshReachable = Test-BoundTcpPort -HostTarget $target -Port $sshPortLocal -Timeout $timeoutLocal -SourceIp $sourceIpLocal
        $smbReachable = $false
        $tbShareReachable = $false
        $peerIdentity = $null

        if ($tcpReachable -or $sshReachable -or $target -match '^\d+\.\d+\.\d+\.\d+$') {
            $smbReachable = Test-BoundTcpPort -HostTarget $target -Port 445 -Timeout $timeoutLocal -SourceIp $sourceIpLocal
            $tbShareReachable = Test-BoundTcpPort -HostTarget $target -Port 6547 -Timeout ([Math]::Min($timeoutLocal, 750)) -SourceIp $sourceIpLocal
        }
        if ($tcpReachable) {
            $peerIdentity = Get-WsManNtlmPeerIdentity -HostTarget $target -Port $winRmPortLocal -Timeout $timeoutLocal
        }

        [PSCustomObject]@{
            Target                  = $target
            PeerHostname            = if ($peerIdentity) { $peerIdentity.DnsComputerName } else { $null }
            PeerNetBiosName         = if ($peerIdentity) { $peerIdentity.NetBiosComputerName } else { $null }
            WinRmPort               = $winRmPortLocal
            SshPort                 = $sshPortLocal
            TcpTimeoutMs            = $timeoutLocal
            TcpReachable            = [bool]$tcpReachable
            SshReachable            = [bool]$sshReachable
            SmbReachable            = [bool]$smbReachable
            ThunderboltShareChannel = [bool]$tbShareReachable
        }
    })

    # ── Phase 6: Optional raw diagnostics ────────────────────────────────
    $raw = $null
    if ($IncludeRaw) {
        $raw = [PSCustomObject]@{
            IpConfig = (ipconfig /all | Out-String).Trim()
            Arp      = (arp -a | Out-String).Trim()
            Route    = (route print | Out-String).Trim()
        }
    }

    # ── Result ───────────────────────────────────────────────────────────
    return [PSCustomObject]@{
        Timestamp           = Get-Date
        LocalComputer       = $env:COMPUTERNAME
        AdapterCount        = $adapters.Count
        Adapters            = @($adapters)
        Usb4Topology        = @($topologyRouters)
        LocalAddresses      = @($localAddresses)
        Neighbors           = @($neighbors)
        ApipaRouteConflicts = @($apipaRouteConflicts)
        NameResolution      = @($nameResolution)
        WinRmCandidates     = @($winRmCandidates)
        Raw                 = $raw
    }
}
