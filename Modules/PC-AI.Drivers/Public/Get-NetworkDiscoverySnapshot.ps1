#Requires -Version 7.0
function Get-NetworkDiscoverySnapshot {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ComputerName,

        [Parameter()]
        [System.Management.Automation.PSCredential]$Credential,

        [Parameter()]
        [switch]$IncludeRawCommands
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    $scriptBlock = {
        param([bool]$IncludeRaw)

        $adapters = @(Get-CimInstance Win32_NetworkAdapter -ErrorAction SilentlyContinue |
            Where-Object { $_.NetConnectionID -or $_.PhysicalAdapter })

        $configs = @{}
        foreach ($cfg in @(Get-CimInstance Win32_NetworkAdapterConfiguration -ErrorAction SilentlyContinue)) {
            $configs[$cfg.Index] = $cfg
        }

        $ipInterfaces = @{}
        foreach ($iface in @(Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetIPInterface -ErrorAction SilentlyContinue)) {
            if (-not $iface.InterfaceAlias) { continue }
            if (-not $ipInterfaces.ContainsKey($iface.InterfaceAlias)) {
                $ipInterfaces[$iface.InterfaceAlias] = @()
            }
            $ipInterfaces[$iface.InterfaceAlias] += $iface
        }

        $neighbors = @{}
        foreach ($neighbor in @(Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -ErrorAction SilentlyContinue)) {
            if (-not $neighbor.InterfaceAlias) { continue }
            if (-not $neighbor.IPAddress) { continue }
            if ($neighbor.State -in @(0, '0', 'Unreachable', 'Invalid')) { continue }
            if ($neighbor.IPAddress -in @('255.255.255.255', '169.254.255.255')) { continue }
            if ($neighbor.IPAddress -match '^(22[4-9]|23\\d)\\.' -or $neighbor.IPAddress -like 'ff*') { continue }
            if ($neighbor.LinkLayerAddress -in @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff')) { continue }
            if (-not $neighbors.ContainsKey($neighbor.InterfaceAlias)) {
                $neighbors[$neighbor.InterfaceAlias] = @()
            }
            $neighbors[$neighbor.InterfaceAlias] += $neighbor
        }

        $routes = @()
        try {
            $routes = @(Get-CimInstance Win32_IP4RouteTable -ErrorAction Stop |
                Select-Object Destination, Mask, NextHop, Metric1, InterfaceIndex)
        } catch {
            $routes = @()
        }

        $adapterRows = foreach ($adapter in $adapters | Sort-Object NetConnectionID, Name) {
            $cfg = $configs[$adapter.Index]
            $ifaceRows = @($ipInterfaces[$adapter.NetConnectionID])
            $ipv4Metric = ($ifaceRows | Where-Object { $_.AddressFamily -eq 2 } | Select-Object -First 1 -ExpandProperty InterfaceMetric)
            $ipv6Metric = ($ifaceRows | Where-Object { $_.AddressFamily -eq 23 } | Select-Object -First 1 -ExpandProperty InterfaceMetric)
            $ipv4Mtu = ($ifaceRows | Where-Object { $_.AddressFamily -eq 2 } | Select-Object -First 1 -ExpandProperty NlMtu)
            $ipv6Mtu = ($ifaceRows | Where-Object { $_.AddressFamily -eq 23 } | Select-Object -First 1 -ExpandProperty NlMtu)

            [PSCustomObject]@{
                Name             = $adapter.Name
                InterfaceAlias   = $adapter.NetConnectionID
                Description      = $adapter.Description
                PhysicalAdapter  = [bool]$adapter.PhysicalAdapter
                NetEnabled       = [bool]$adapter.NetEnabled
                MacAddress       = $adapter.MACAddress
                Speed            = $adapter.Speed
                PnpDeviceId      = $adapter.PNPDeviceID
                IPv4Addresses    = @($cfg.IPAddress | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' })
                IPv6Addresses    = @($cfg.IPAddress | Where-Object { $_ -match ':' })
                Gateways         = @($cfg.DefaultIPGateway)
                DnsServers       = @($cfg.DNSServerSearchOrder)
                DhcpEnabled      = if ($null -ne $cfg.DHCPEnabled) { [bool]$cfg.DHCPEnabled } else { $null }
                DhcpServer       = $cfg.DHCPServer
                IPv4Metric       = $ipv4Metric
                IPv6Metric       = $ipv6Metric
                IPv4Mtu          = $ipv4Mtu
                IPv6Mtu          = $ipv6Mtu
                NeighborCount    = @($neighbors[$adapter.NetConnectionID]).Count
            }
        }

        $result = [PSCustomObject]@{
            ComputerName = $env:COMPUTERNAME
            AdapterCount = @($adapterRows).Count
            Adapters     = @($adapterRows)
            IPv4Routes   = @($routes)
        }

        if ($IncludeRaw) {
            $result | Add-Member -NotePropertyName RawIpconfig -NotePropertyValue ((ipconfig /all) | Out-String)
            $result | Add-Member -NotePropertyName RawArp -NotePropertyValue ((arp -a) | Out-String)
            $result | Add-Member -NotePropertyName RawRoute -NotePropertyValue ((route print) | Out-String)
        }

        return $result
    }

    if ($ComputerName) {
        $invokeParams = @{
            ComputerName = $ComputerName
            ScriptBlock  = $scriptBlock
            ArgumentList = @([bool]$IncludeRawCommands)
            ErrorAction  = 'Stop'
        }
        if ($Credential) {
            $invokeParams['Credential'] = $Credential
        }
        return Invoke-Command @invokeParams
    }

    return & $scriptBlock ([bool]$IncludeRawCommands)
}
