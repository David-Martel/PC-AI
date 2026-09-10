#Requires -Version 5.1
<#
.SYNOPSIS
    Internal: waits for a re-enumerating adapter to become usable again.

.DESCRIPTION
    After a PnP cycle or a driver-property change, Status -eq 'Up' arrives before
    the adapter is actually usable: the IP configuration is reapplied afterwards,
    and on a USB NIC the neighbour entry for the peer is gone until something
    provokes ARP. Treating 'Up' as ready is why post-change measurements come back
    as spurious failures.

    This waits for Up AND for the expected IPv4 address to be present, then warms
    the neighbour cache with a short ping before returning.

    Takes a script block for resolution rather than a name, because the adapter's
    Name and ifIndex may both have changed during the cycle -- which is the whole
    reason the caller is keyed on GUID or MAC.
#>
function Wait-PcaiAdapterReady {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [scriptblock]$Resolve,

        [Parameter()]
        [string]$ExpectedIPv4,

        [Parameter()]
        [ValidateRange(5, 600)]
        [int]$TimeoutSeconds = 90,

        [Parameter()]
        [string]$WarmPingTarget
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $nic = $null
        try { $nic = & $Resolve } catch {
            # Mid-cycle the device is genuinely absent; that is expected, keep waiting.
            Write-Verbose "adapter not resolvable yet: $($_.Exception.Message)"
            continue
        }
        if (-not $nic -or $nic.Status -ne 'Up') { continue }

        if ($ExpectedIPv4) {
            $has = Get-NetIPAddress -InterfaceIndex $nic.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                   Where-Object { $_.IPAddress -eq $ExpectedIPv4 }
            if (-not $has) { continue }
        }
        $ready = $true
        break
    }

    if (-not $ready) { return $false }

    # Provoke ARP/ND toward the PEER so the first real measurement is not paying for
    # address resolution. Pinging our own address would warm nothing, so when no peer
    # is supplied we simply skip the warm-up rather than perform a useless one.
    if ($WarmPingTarget) { $null = & ping.exe -n 2 -w 1500 $WarmPingTarget 2>&1 }

    $true
}
