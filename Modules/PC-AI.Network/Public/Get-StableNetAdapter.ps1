#Requires -Version 5.1
<#
.SYNOPSIS
    Resolves a network adapter by a handle that survives re-enumeration.

.DESCRIPTION
    On a laptop that roams between docks, Thunderbolt hubs and USB-C dongles, an
    adapter's Name ("Ethernet 15") and ifIndex both change when the device is
    re-enumerated. Neither is a safe key to write into a script or a config file:
    the script keeps running and quietly reconfigures a different NIC.

    InterfaceGuid does not drift, so that is the handle this resolves on. It
    refuses to guess: zero matches throws, and more than one match throws rather
    than silently picking the first. That refusal is the point -- a "picked the
    first one" default is how the internet uplink gets reconfigured when the fleet
    link was intended.

    MAC lookup is deliberately NOT offered. Windows reports the same MacAddress on
    adapters that do not share one: internal Hyper-V vSwitches, IP-HTTPS, 6to4,
    Tailscale and the kernel-debugger adapter were all observed reporting this
    host's USB NIC MAC, and the same query returned 1 match on one call and 10 on
    the next. A lookup that intermittently resolves to the wrong device is worse
    than no lookup, so use InterfaceGuid.

    Use -Detailed to also report the receive-path features that decide throughput
    on cheap USB NICs (RSS queue count, RSC operational state and its failure
    reason, jumbo capability, and the bound NDIS components).

.PARAMETER InterfaceGuid
    The adapter's InterfaceGuid, in registry brace form, e.g.
    '{2222E1D5-9E2E-4B08-B751-3A92F8079347}'.

.PARAMETER Name
    An adapter name. Provided for completeness and interactive use; it is the
    unstable handle this function exists to replace, so it warns when used.

.PARAMETER Detailed
    Include RSS/RSC/offload state and bound NDIS components.

.EXAMPLE
    Get-StableNetAdapter -InterfaceGuid '{2222E1D5-9E2E-4B08-B751-3A92F8079347}'

.EXAMPLE
    Get-StableNetAdapter -InterfaceGuid $guid -Detailed
    Also report RSS/RSC state and why receive throughput may be capped.

.EXAMPLE
    $nic = Get-StableNetAdapter -InterfaceGuid $guid
    Set-NetAdapterAdvancedProperty -Name $nic.Name -RegistryKeyword '*JumboPacket' -RegistryValue '9014'
    Resolve first, then act on the resolved Name -- never on a hardcoded one.

.OUTPUTS
    PSCustomObject describing the single matched adapter.
#>
function Get-StableNetAdapter {
    [CmdletBinding(DefaultParameterSetName = 'ByGuid')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, ParameterSetName = 'ByGuid')]
        [ValidatePattern('^\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}$')]
        [string]$InterfaceGuid,

        [Parameter(Mandatory, ParameterSetName = 'ByName')]
        [string]$Name,

        [Parameter()]
        [switch]$Detailed
    )

    $all = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop)

    # Plain foreach accumulation rather than a Where-Object predicate, so the match
    # set is built by code that is trivially readable at a glance.
    $matched = @()
    $key = ''

    if ($PSCmdlet.ParameterSetName -eq 'ByGuid') {
        $key = $InterfaceGuid
        foreach ($cand in $all) {
            if ($cand.InterfaceGuid -eq $InterfaceGuid) { $matched += $cand }
        }
    }
    else {
        Write-Warning ("Resolving by Name is unstable: '{0}' can be reassigned to a different device on re-enumeration. Prefer -InterfaceGuid or -MacAddress." -f $Name)
        $key = $Name
        foreach ($cand in $all) {
            if ($cand.Name -eq $Name) { $matched += $cand }
        }
    }

    if ($matched.Count -eq 0) {
        throw ("No adapter matches {0} '{1}'. The device may be unplugged, or this may be the wrong host. " -f
               $PSCmdlet.ParameterSetName, $key) +
              'Enumerate with: Get-NetAdapter -IncludeHidden | Select-Object Name, ifIndex, MacAddress, InterfaceGuid'
    }
    if ($matched.Count -gt 1) {
        $desc = ($matched | ForEach-Object { '{0} [ifIndex {1}]' -f $_.Name, $_.ifIndex }) -join ', '
        throw ("{0} '{1}' matched {2} adapters ({3}). Refusing to guess which one you meant." -f
               $PSCmdlet.ParameterSetName, $key, $matched.Count, $desc)
    }

    $a = $matched[0]

    # netsh is authoritative for the IPv4 subinterface MTU. Get-NetAdapter's MtuSize
    # and the *JumboPacket keyword can BOTH read 9000/9014 while IPv4 still runs at
    # 1500, so a check that consults only Get-NetAdapter can report success wrongly.
    $ipv4Mtu = $null
    try {
        $ipv4Mtu = netsh interface ipv4 show subinterfaces |
            Select-String -SimpleMatch $a.Name |
            ForEach-Object { ($_ -split '\s+' | Where-Object { $_ })[0] } |
            Select-Object -First 1
    } catch { Write-Verbose "netsh subinterface query failed: $_" }

    $out = [ordered]@{
        Name                 = $a.Name
        ifIndex              = $a.ifIndex
        InterfaceGuid        = $a.InterfaceGuid
        MacAddress           = $a.MacAddress
        InterfaceDescription = $a.InterfaceDescription
        Status               = $a.Status
        LinkSpeed            = $a.LinkSpeed
        AdapterMtu           = $a.MtuSize
        IPv4Mtu              = $ipv4Mtu
        PnPDeviceID          = $a.PnPDeviceID
        IPv4Address          = (Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                                Where-Object { $_.IPAddress -notlike '169.254.*' } |
                                ForEach-Object { '{0}/{1}' -f $_.IPAddress, $_.PrefixLength }) -join ', '
    }

    if ($Detailed) {
        $rss = Get-NetAdapterRss -Name $a.Name -ErrorAction SilentlyContinue
        $rsc = Get-NetAdapterRsc -Name $a.Name -ErrorAction SilentlyContinue
        $jumbo = Get-NetAdapterAdvancedProperty -Name $a.Name -RegistryKeyword '*JumboPacket' -ErrorAction SilentlyContinue

        $out['RssEnabled']      = if ($rss) { $rss.Enabled } else { $false }
        $out['RssQueues']       = if ($rss) { $rss.NumberOfReceiveQueues } else { 0 }
        $out['RscEnabled']      = if ($rsc) { $rsc.IPv4Enabled } else { $null }
        $out['RscOperational']  = if ($rsc) { $rsc.IPv4OperationalState } else { $null }
        $out['RscFailureReason'] = if ($rsc) { $rsc.IPv4FailureReason } else { $null }
        $out['JumboValue']      = if ($jumbo) { $jumbo.DisplayValue } else { '<not exposed>' }
        $out['JumboSupported']  = if ($jumbo) { ($jumbo.ValidDisplayValues -join ' | ') } else { '' }
        $out['BoundComponents'] = @(Get-NetAdapterBinding -Name $a.Name -ErrorAction SilentlyContinue |
                                    Where-Object Enabled | Select-Object -ExpandProperty ComponentID)

        # An adapter with no RSS is packet-rate bound on receive: every interrupt
        # lands on one core. RSC is then the only coalescing left, so a non-operational
        # RSC on a no-RSS adapter is the single most useful thing to surface here.
        # Note NDISCompatibility does NOT prove a filter is at fault -- it can equally
        # be the miniport, so this reports the fact and does not diagnose a culprit.
        $out['ReceiveRisk'] = if (-not $out['RssEnabled'] -and $out['RscOperational'] -eq $false) {
            'HIGH: no RSS and RSC not operational ({0}) -- receive is packet-rate bound on one core' -f $out['RscFailureReason']
        } elseif (-not $out['RssEnabled']) {
            'MEDIUM: no RSS; receive interrupts are single-core, RSC is carrying coalescing'
        } else {
            'LOW'
        }
    }

    [PSCustomObject]$out
}
