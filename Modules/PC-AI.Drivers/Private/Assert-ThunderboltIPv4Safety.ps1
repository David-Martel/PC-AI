#Requires -Version 7.0
function Assert-ThunderboltIPv4Safety {
    <#
    .SYNOPSIS
        Validates peer identity and preserves Windows IPv4 address/route state.
    .DESCRIPTION
        Takes fresh OS observations before mutation, rechecks before static
        assignment, and verifies actual prefix/address and unrelated state after
        mutation. It never changes an adapter, address, route or host policy.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$InterfaceAlias,
        [string]$IPv4Address,
        [ValidateRange(8, 30)][int]$PrefixLength = 30,
        [ValidateSet('Preflight', 'Revalidate', 'Postcondition')][string]$Mode = 'Preflight',
        [object]$Baseline
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    function Get-IPv4Interval {
        param([string]$Address, [int]$Prefix)
        $parsed = $null
        if ($Prefix -lt 0 -or $Prefix -gt 32 -or -not [Net.IPAddress]::TryParse($Address, [ref]$parsed) -or
            $parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
            throw 'Cannot validate an IPv4 address or route prefix.'
        }
        $bytes = $parsed.GetAddressBytes()
        $value = [uint64]$bytes[0] * 16777216 + [uint64]$bytes[1] * 65536 + [uint64]$bytes[2] * 256 + [uint64]$bytes[3]
        $size = [uint64][Math]::Pow(2, 32 - $Prefix)
        $first = [uint64]([Math]::Floor($value / $size) * $size)
        [pscustomobject]@{ First = $first; Last = $first + $size - 1; Value = $value }
    }
    function Get-RouteInterval {
        param([object]$Route)
        if ($Route.DestinationPrefix -notmatch '^([0-9.]+)/([0-9]+)$') {
            throw 'Cannot validate an IPv4 route destination prefix.'
        }
        Get-IPv4Interval -Address $Matches[1] -Prefix ([int]$Matches[2])
    }
    function Get-OptionalValue {
        param([object]$Row, [string]$Name)
        $property = $Row.PSObject.Properties[$Name]
        if ($property) { return [string]$property.Value }
        return ''
    }
    function Get-AddressRows {
        param([object[]]$Rows)
        @($Rows | Where-Object { $null -ne $_ } | ForEach-Object {
                '{0}|{1}|{2}|{3}|{4}|{5}' -f $_.InterfaceIndex, $_.IPAddress, $_.PrefixLength,
                (Get-OptionalValue $_ 'SkipAsSource'), (Get-OptionalValue $_ 'PrefixOrigin'), (Get-OptionalValue $_ 'SuffixOrigin')
            } | Sort-Object)
    }
    function Get-RouteRows {
        param([object[]]$Rows)
        @($Rows | Where-Object { $null -ne $_ } | ForEach-Object {
                '{0}|{1}|{2}|{3}|{4}' -f $_.InterfaceIndex, $_.DestinationPrefix, $_.NextHop,
                (Get-OptionalValue $_ 'RouteMetric'), (Get-OptionalValue $_ 'Protocol')
            } | Sort-Object)
    }
    function Assert-SameRows {
        param([object[]]$Before, [object[]]$After, [string]$Description)
        if ((ConvertTo-Json -InputObject @($Before | Where-Object { $null -ne $_ }) -Compress) -cne
            (ConvertTo-Json -InputObject @($After | Where-Object { $null -ne $_ }) -Compress)) {
            throw "Thunderbolt IPv4 preservation failed: $Description changed. No success is reported and no automatic rollback is attempted."
        }
    }

    $requested = $null
    if ($IPv4Address) {
        $requested = Get-IPv4Interval -Address $IPv4Address -Prefix $PrefixLength
        $firstByte = [Net.IPAddress]::Parse($IPv4Address).GetAddressBytes()[0]
        if ($firstByte -eq 0 -or $firstByte -eq 127 -or $firstByte -ge 224 -or
            $requested.Value -eq $requested.First -or $requested.Value -eq $requested.Last) {
            throw 'Static IPv4 requires a usable unicast host address in the requested prefix.'
        }
    }
    if ($Mode -ne 'Preflight' -and -not $Baseline) {
        throw 'IPv4 revalidation and postconditions require the original preflight snapshot.'
    }
    $matches = @(Get-NetAdapter -Name $InterfaceAlias -IncludeHidden -ErrorAction Stop |
            Where-Object { $_.Name -eq $InterfaceAlias })
    if ($matches.Count -ne 1) { throw 'Expected exactly one current adapter matching the exact InterfaceAlias.' }
    $adapter = $matches[0]
    $pnpId = Get-OptionalValue $adapter 'PnPDeviceID'
    if ($adapter.InterfaceDescription -notmatch '(?i)Thunderbolt.*(Network|Ethernet)|USB4.*(P2P|Peer).*Network' -and
        $pnpId -notmatch '(?i)PROT_USB4NET') {
        throw 'Selected interface is not a genuine Thunderbolt / USB4 peer network adapter.'
    }
    if ($adapter.Status -ne 'Up' -or [int]$adapter.ifIndex -le 0) {
        throw 'Selected Thunderbolt / USB4 peer adapter must be Up with a valid interface index.'
    }
    $index = [int]$adapter.ifIndex
    $identity = '{0}|{1}|{2}|{3}' -f $index, $pnpId, (Get-OptionalValue $adapter 'InterfaceGuid'), (Get-OptionalValue $adapter 'MacAddress')
    $addresses = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop)
    $routes = @(Get-NetRoute -AddressFamily IPv4 -ErrorAction Stop)
    $selected = @($addresses | Where-Object InterfaceIndex -EQ $index)
    $existing = @($selected | Where-Object IPAddress -EQ $IPv4Address)
    if ($requested) {
        foreach ($address in $addresses) {
            $interval = Get-IPv4Interval -Address $address.IPAddress -Prefix ([int]$address.PrefixLength)
            if ($address.InterfaceIndex -eq $index) {
                if ($address.IPAddress -ne $IPv4Address -or $address.PrefixLength -ne $PrefixLength) {
                    throw 'Unrelated IPv4 address or mismatched prefix on selected interface; preserve it and resolve explicitly.'
                }
            }
            elseif ($interval.First -le $requested.Last -and $requested.First -le $interval.Last) {
                throw 'Requested IPv4 subnet overlaps an address on another interface; preserve foreign address state.'
            }
        }
        if ($existing.Count -gt 1) { throw 'Requested IPv4 address is ambiguous on the selected interface.' }
        if ($existing.Count -eq 1 -and ((Get-OptionalValue $existing[0] 'SkipAsSource') -eq 'True' -or
                (Get-OptionalValue $existing[0] 'AddressState') -ne 'Preferred')) {
            throw 'Requested IPv4 address is not Preferred and usable as a source.'
        }
        if ($existing.Count -eq 1 -and (Get-OptionalValue $existing[0] 'PrefixOrigin') -ne 'Manual') {
            throw 'Requested static IPv4 address is not manually configured; preserve its existing policy and resolve explicitly.'
        }
        foreach ($route in $routes) {
            $interval = Get-RouteInterval $route
            if ($route.InterfaceIndex -eq $index -and $route.DestinationPrefix -eq '0.0.0.0/0') {
                throw 'Selected peer interface has a default route; preserve it and resolve explicitly.'
            }
            # The LAN default remains the recovery path; specific foreign routes
            # must not cover any part of the requested peer subnet.
            if ($route.InterfaceIndex -ne $index -and $route.DestinationPrefix -ne '0.0.0.0/0' -and
                $interval.First -le $requested.Last -and $requested.First -le $interval.Last) {
                throw 'Requested IPv4 subnet overlaps a route on another interface; preserve foreign route state.'
            }
        }
        if ($existing.Count -eq 1) {
            $existingSubnetRoutes = @($routes | Where-Object {
                    $interval = Get-RouteInterval $_
                    $_.InterfaceIndex -eq $index -and $_.NextHop -eq '0.0.0.0' -and
                    $interval.First -eq $requested.First -and $interval.Last -eq $requested.Last
                })
            if ($existingSubnetRoutes.Count -ne 1) {
                throw 'Requested static IPv4 address requires exactly one current selected-interface on-link subnet route.'
            }
        }
    }
    $snapshot = [pscustomobject]@{
        InterfaceAlias = $adapter.Name; InterfaceIndex = $index; Identity = $identity
        Addresses = @($addresses); Routes = @($routes)
        AddressRows = @(Get-AddressRows $addresses); RouteRows = @(Get-RouteRows $routes)
        AlreadyAssigned = ($requested -and $existing.Count -eq 1)
    }
    if ($Mode -eq 'Preflight') { return $snapshot }
    if ($identity -cne $Baseline.Identity) { throw 'Selected adapter identity changed after preflight.' }
    if ($Mode -eq 'Revalidate' -or -not $requested -or $Baseline.AlreadyAssigned) {
        Assert-SameRows -Before $Baseline.AddressRows -After $snapshot.AddressRows -Description 'IPv4 address state'
        Assert-SameRows -Before $Baseline.RouteRows -After $snapshot.RouteRows -Description 'IPv4 route state'
    }
    else {
        if ($existing.Count -ne 1 -or $existing[0].PrefixLength -ne $PrefixLength) {
            throw "Thunderbolt IPv4 postcondition failed: requested $IPv4Address/$PrefixLength was not observed on the selected interface."
        }
        if ((Get-OptionalValue $existing[0] 'SkipAsSource') -eq 'True' -or
            (Get-OptionalValue $existing[0] 'AddressState') -ne 'Preferred') {
            throw 'Thunderbolt IPv4 postcondition failed: requested address is not Preferred and usable as a source.'
        }
        Assert-SameRows -Before $Baseline.AddressRows -After (Get-AddressRows @($addresses | Where-Object InterfaceIndex -NE $index)) -Description 'unrelated IPv4 address state'
        # Compare a multiset: every baseline route must remain exactly present.
        # Only newly generated on-link subnet/host/broadcast routes are admitted.
        $remaining = [Collections.Generic.List[object]]::new()
        foreach ($route in $routes) { $remaining.Add($route) }
        foreach ($row in $Baseline.RouteRows) {
            $matchIndex = -1
            for ($i = 0; $i -lt $remaining.Count; $i++) {
                if (@(Get-RouteRows @($remaining[$i]))[0] -ceq $row) { $matchIndex = $i; break }
            }
            if ($matchIndex -lt 0) { throw 'Thunderbolt IPv4 preservation failed: an existing IPv4 route changed or disappeared.' }
            $remaining.RemoveAt($matchIndex)
        }
        foreach ($route in $remaining) {
            $interval = Get-RouteInterval $route
            $allowedInterval = ($interval.First -eq $requested.First -and $interval.Last -eq $requested.Last) -or
            ($interval.First -eq $interval.Last -and ($interval.First -eq $requested.Value -or $interval.First -eq $requested.Last))
            if ($route.InterfaceIndex -ne $index -or $route.NextHop -ne '0.0.0.0' -or -not $allowedInterval) {
                throw 'Thunderbolt IPv4 preservation failed: an unrelated IPv4 route appeared.'
            }
        }
        $networkRoute = @($routes | Where-Object {
                $interval = Get-RouteInterval $_
                $_.InterfaceIndex -eq $index -and $_.NextHop -eq '0.0.0.0' -and
                $interval.First -eq $requested.First -and $interval.Last -eq $requested.Last
            })
        if ($networkRoute.Count -ne 1) { throw 'Thunderbolt IPv4 postcondition failed: one selected-interface on-link subnet route was not observed.' }
    }
    return $snapshot
}
