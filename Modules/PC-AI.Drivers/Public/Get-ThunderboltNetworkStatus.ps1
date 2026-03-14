#Requires -Version 7.0
<#
.SYNOPSIS
    Discovers Thunderbolt / USB4 networking adapters and reports link state, IP data,
    peer candidates, and tuning hints.

.DESCRIPTION
    Uses CIM and StandardCimv2 classes instead of NetTCPIP cmdlets so it remains
    usable on constrained Windows shells where the networking module is unavailable.
    The function focuses on USB4 / Thunderbolt networking surfaces such as:

      - USB4(TM) P2P Network Adapter
      - USB4(TM) Host Router
      - Thunderbolt routers and controllers

    Peer discovery is based on MSFT_NetNeighbor entries for the selected interface.
    When -ProbeWinRM is specified, IPv4 peer candidates are tested with Test-WSMan.

.PARAMETER InterfaceAlias
    Optional interface alias filter such as 'Ethernet 11'. When omitted, all
    Thunderbolt / USB4 networking adapters are returned.

.PARAMETER ProbeWinRM
    Attempt Test-WSMan against discovered IPv4 peer candidates.

.OUTPUTS
    PSCustomObject[] with adapter, IP, neighbor, and recommendation details.
#>
function Get-ThunderboltNetworkStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string[]]$InterfaceAlias,

        [Parameter()]
        [switch]$ProbeWinRM
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    function Get-SafeCimInstance {
        param(
            [Parameter(Mandatory)]
            [string]$ClassName,

            [Parameter()]
            [string]$Namespace = 'root/cimv2'
        )

        try {
            return @(Get-CimInstance -Namespace $Namespace -ClassName $ClassName -ErrorAction Stop)
        } catch {
            Write-Verbose "CIM query failed for ${Namespace}:${ClassName}: $($_.Exception.Message)"
            return @()
        }
    }

    function Convert-LinkSpeedToGbps {
        param([object]$Speed)

        if ($null -eq $Speed) { return $null }
        $numeric = 0.0
        if (-not [double]::TryParse([string]$Speed, [ref]$numeric)) {
            return $null
        }
        if ($numeric -le 0) { return $null }
        return [math]::Round(($numeric / 1000000000.0), 2)
    }

    function Resolve-NeighborState {
        param([object]$State)

        $text = [string]$State
        if ([string]::IsNullOrWhiteSpace($text)) { return 'Unknown' }
        switch ($text) {
            '0' { return 'Unknown' }
            '1' { return 'Incomplete' }
            '2' { return 'Reachable' }
            '3' { return 'Stale' }
            '4' { return 'Delay' }
            '5' { return 'Probe' }
            '6' { return 'Permanent' }
        }
        return $text
    }

    function Get-PnpSignedDriverInfo {
        param([string]$DeviceId)

        if ([string]::IsNullOrWhiteSpace($DeviceId)) { return $null }

        $escapedDeviceId = $DeviceId.Replace('\', '\\').Replace("'", "''")
        $query = "SELECT DeviceID, DriverVersion, DriverProviderName FROM Win32_PnPSignedDriver WHERE DeviceID = '$escapedDeviceId'"

        try {
            return Get-CimInstance -Query $query -ErrorAction Stop | Select-Object -First 1
        } catch {
            Write-Verbose "Signed driver lookup failed for '$DeviceId': $($_.Exception.Message)"
            return $null
        }
    }

    $aliasFilter = @{}
    foreach ($alias in @($InterfaceAlias)) {
        if (-not [string]::IsNullOrWhiteSpace($alias)) {
            $aliasFilter[$alias] = $true
        }
    }

    $adapters = @(Get-SafeCimInstance -ClassName 'Win32_NetworkAdapter' |
        Where-Object {
            $_.NetConnectionID -and (
                $_.Name -match 'Thunderbolt|USB4|P2P Network Adapter' -or
                $_.Description -match 'Thunderbolt|USB4|P2P Network Adapter' -or
                $_.PNPDeviceID -match 'PROT_USB4NET|THUNDERBOLT|USB4'
            )
        })

    if ($aliasFilter.Count -gt 0) {
        $adapters = @($adapters | Where-Object { $aliasFilter.ContainsKey($_.NetConnectionID) })
    }

    if ($adapters.Count -eq 0) {
        return @()
    }

    $configsByIndex = @{}
    foreach ($cfg in Get-SafeCimInstance -ClassName 'Win32_NetworkAdapterConfiguration') {
        $configsByIndex[$cfg.Index] = $cfg
    }

    $interfacesByAlias = @{}
    foreach ($iface in Get-SafeCimInstance -Namespace 'root/StandardCimv2' -ClassName 'MSFT_NetIPInterface') {
        if (-not $iface.InterfaceAlias) { continue }
        if (-not $interfacesByAlias.ContainsKey($iface.InterfaceAlias)) {
            $interfacesByAlias[$iface.InterfaceAlias] = @()
        }
        $interfacesByAlias[$iface.InterfaceAlias] += $iface
    }

    $neighborsByAlias = @{}
    foreach ($neighbor in Get-SafeCimInstance -Namespace 'root/StandardCimv2' -ClassName 'MSFT_NetNeighbor') {
        if (-not $neighbor.InterfaceAlias) { continue }
        if (-not $neighborsByAlias.ContainsKey($neighbor.InterfaceAlias)) {
            $neighborsByAlias[$neighbor.InterfaceAlias] = @()
        }
        $neighborsByAlias[$neighbor.InterfaceAlias] += $neighbor
    }

    $results = foreach ($adapter in $adapters) {
        $cfg = $configsByIndex[$adapter.Index]
        $ifaces = @($interfacesByAlias[$adapter.NetConnectionID])

        $ipv4 = @()
        $ipv6 = @()
        if ($cfg -and $cfg.IPAddress) {
            foreach ($ip in @($cfg.IPAddress)) {
                if ($ip -match '^\d+\.\d+\.\d+\.\d+$') {
                    $ipv4 += $ip
                } elseif ($ip -match ':') {
                    $ipv6 += $ip
                }
            }
        }

        $ipv4Interface = $ifaces | Where-Object { $_.AddressFamily -eq 2 } | Select-Object -First 1
        $ipv6Interface = $ifaces | Where-Object { $_.AddressFamily -eq 23 } | Select-Object -First 1

        $driver = Get-PnpSignedDriverInfo -DeviceId $adapter.PNPDeviceID

        $peerCandidates = @()
        foreach ($neighbor in @($neighborsByAlias[$adapter.NetConnectionID])) {
            $ip = [string]$neighbor.IPAddress
            if ([string]::IsNullOrWhiteSpace($ip)) { continue }
            if ($ip -eq '255.255.255.255' -or $ip -eq '169.254.255.255') { continue }
            if ($ip -match '^(22[4-9]|23\d)\.' -or $ip -like 'ff*') { continue }

            $state = Resolve-NeighborState -State $neighbor.State
            $mac = [string]$neighbor.LinkLayerAddress
            if ($state -match 'Unreachable|Invalid|Unknown|Incomplete') { continue }
            if ($mac -in @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff')) { continue }

            $peer = [PSCustomObject]@{
                IPAddress        = $ip
                LinkLayerAddress = if ([string]::IsNullOrWhiteSpace($mac)) { $null } else { $mac }
                State            = $state
                WsManReachable   = $null
            }

            if ($ProbeWinRM -and $ip -match '^\d+\.\d+\.\d+\.\d+$') {
                try {
                    Test-WSMan -ComputerName $ip -ErrorAction Stop | Out-Null
                    $peer.WsManReachable = $true
                } catch {
                    $peer.WsManReachable = $false
                }
            }

            $peerCandidates += $peer
        }

        $role = if ($adapter.PNPDeviceID -match 'PROT_USB4NET') {
            'P2PNetwork'
        } elseif ($adapter.Name -match 'Host Router') {
            'HostRouter'
        } elseif ($adapter.Name -match 'Router|Thunderbolt') {
            'Router'
        } else {
            'Peripheral'
        }

        $recommendations = [System.Collections.Generic.List[string]]::new()
        if ($role -eq 'P2PNetwork' -and $ipv4.Count -gt 0 -and (@($ipv4 | Where-Object { $_ -like '169.254.*' })).Count -gt 0) {
            if ($peerCandidates.Count -eq 0) {
                $recommendations.Add('The USB4 peer link is using APIPA but no peer was discovered. Confirm the remote Windows host exposes a live USB4 P2P adapter and that cable authorization succeeded.')
            } else {
                $recommendations.Add('The USB4 peer link is in APIPA mode. For predictable SMB and WinRM targeting, consider assigning a dedicated /30 instead of relying on 169.254/16.')
            }
        }
        if ($role -eq 'P2PNetwork' -and $ipv4Interface -and $ipv4Interface.NlMtu -lt 62000) {
            $recommendations.Add('The IPv4 MTU is below the observed USB4 baseline of 62000 bytes. Review interface tuning if large-transfer throughput is below expectation.')
        }
        if ($role -eq 'P2PNetwork' -and $peerCandidates.Count -gt 0 -and (@($peerCandidates | Where-Object { $_.WsManReachable -eq $true })).Count -eq 0 -and $ProbeWinRM) {
            $recommendations.Add('A peer address was discovered on the USB4 link, but WinRM did not answer. Enable WinRM and set the connection profile to Private on the remote machine.')
        }
        if ($role -eq 'HostRouter') {
            $recommendations.Add('Host router devices are inbox-managed. Use OEM firmware tooling or Windows Update rather than forcing a third-party driver package.')
        }

        [PSCustomObject]@{
            InterfaceAlias      = $adapter.NetConnectionID
            Name                = $adapter.Name
            Description         = $adapter.Description
            Role                = $role
            NetEnabled          = [bool]$adapter.NetEnabled
            NetConnectionStatus = $adapter.NetConnectionStatus
            LinkSpeedGbps       = Convert-LinkSpeedToGbps -Speed $adapter.Speed
            MacAddress          = $adapter.MACAddress
            PnpDeviceId         = $adapter.PNPDeviceID
            DriverVersion       = if ($driver) { $driver.DriverVersion } else { $null }
            DriverProvider      = if ($driver) { $driver.DriverProviderName } else { $null }
            IPv4Addresses       = @($ipv4)
            IPv6Addresses       = @($ipv6)
            IPv4Metric          = if ($ipv4Interface) { $ipv4Interface.InterfaceMetric } else { $null }
            IPv6Metric          = if ($ipv6Interface) { $ipv6Interface.InterfaceMetric } else { $null }
            IPv4Mtu             = if ($ipv4Interface) { $ipv4Interface.NlMtu } else { $null }
            IPv6Mtu             = if ($ipv6Interface) { $ipv6Interface.NlMtu } else { $null }
            AutomaticMetric     = if ($ipv4Interface) { [bool]$ipv4Interface.AutomaticMetric } else { $null }
            NeighborCandidates  = @($peerCandidates)
            RecommendedActions  = @($recommendations)
        }
    }

    return @($results | Sort-Object InterfaceAlias, Role)
}
