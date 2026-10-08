#Requires -Version 7.0
<#
.SYNOPSIS
    Builds or applies a conservative optimization plan for a Thunderbolt / USB4 peer link.

.DESCRIPTION
    The function is intentionally narrow. It avoids broad stack changes and instead
    focuses on the settings that matter most for a dedicated peer-to-peer Windows link:

      - Interface metric
      - MTU
      - Optional static IPv4 assignment

    Without -Apply, the function returns the planned commands and the current adapter
    state. With -Apply, it stops at the first failed netsh command and verifies
    the resulting metrics, MTU and optional IPv4 address before returning status.
    An explicit alias must match exactly; automatic selection requires one adapter.

.PARAMETER InterfaceAlias
    USB4 / Thunderbolt interface alias, typically 'Ethernet 11'.

.PARAMETER InterfaceMetric
    Metric to assign to the interface for both IPv4 and IPv6.

.PARAMETER MtuBytes
    MTU to assign to the interface. The observed USB4 default on this host is 62000.

.PARAMETER IPv4Address
    Optional static IPv4 address to configure.

.PARAMETER PrefixLength
    Prefix length for the optional static IPv4 address. Defaults to /30 for a
    dedicated two-host link.

.PARAMETER Apply
    Execute the generated plan instead of returning it.
#>
function Set-ThunderboltNetworkOptimization {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter()]
        [string]$InterfaceAlias,

        [Parameter()]
        [ValidateRange(1, 9999)]
        [int]$InterfaceMetric = 15,

        [Parameter()]
        [ValidateRange(1280, 65535)]
        [int]$MtuBytes = 62000,

        [Parameter()]
        [string]$IPv4Address,

        [Parameter()]
        [ValidateRange(8, 30)]
        [int]$PrefixLength = 30,

        [Parameter()]
        [switch]$Apply
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    function ConvertTo-IPv4Mask {
        param([Parameter(Mandatory)][int]$Length)

        $mask = [uint32]0
        for ($i = 0; $i -lt $Length; $i++) {
            $mask = $mask -bor (1 -shl (31 - $i))
        }

        $bytes = [BitConverter]::GetBytes([uint32]$mask)
        [Array]::Reverse($bytes)
        return ($bytes | ForEach-Object { [int]$_ }) -join '.'
    }

    if ($IPv4Address) {
        $parsedAddress = $null
        if (-not [System.Net.IPAddress]::TryParse($IPv4Address, [ref]$parsedAddress) -or
            $parsedAddress.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
            throw 'IPv4Address must be a valid IPv4 address.'
        }
        $IPv4Address = $parsedAddress.ToString()
    }

    $status = @(if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
        Get-ThunderboltNetworkStatus
    } else {
        Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias |
            Where-Object { $_.InterfaceAlias -eq $InterfaceAlias }
    })
    if ($status.Count -eq 0) {
        throw 'No Thunderbolt / USB4 interface matched the requested adapter.'
    }
    if ($status.Count -ne 1) {
        throw 'Multiple Thunderbolt / USB4 interfaces matched. Specify one exact InterfaceAlias.'
    }

    $current = $status[0]
    $InterfaceAlias = [string]$current.InterfaceAlias
    if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
        throw 'The selected Thunderbolt / USB4 device has no network interface alias.'
    }
    $plan = [System.Collections.Generic.List[PSCustomObject]]::new()

    $plan.Add([PSCustomObject]@{
        Step        = 'SetIPv4Metric'
        Description = "Set IPv4 metric on $InterfaceAlias to $InterfaceMetric"
        Command     = "netsh interface ipv4 set interface name=$InterfaceAlias metric=$InterfaceMetric"
        Arguments   = @('interface', 'ipv4', 'set', 'interface', "name=$InterfaceAlias", "metric=$InterfaceMetric")
    })
    $plan.Add([PSCustomObject]@{
        Step        = 'SetIPv6Metric'
        Description = "Set IPv6 metric on $InterfaceAlias to $InterfaceMetric"
        Command     = "netsh interface ipv6 set interface $InterfaceAlias metric=$InterfaceMetric"
        Arguments   = @('interface', 'ipv6', 'set', 'interface', $InterfaceAlias, "metric=$InterfaceMetric")
    })
    $plan.Add([PSCustomObject]@{
        Step        = 'SetMtu'
        Description = "Set interface MTU on $InterfaceAlias to $MtuBytes"
        Command     = "netsh interface ipv4 set subinterface $InterfaceAlias mtu=$MtuBytes store=persistent"
        Arguments   = @('interface', 'ipv4', 'set', 'subinterface', $InterfaceAlias, "mtu=$MtuBytes", 'store=persistent')
    })

    if ($IPv4Address) {
        $mask = ConvertTo-IPv4Mask -Length $PrefixLength
        $plan.Add([PSCustomObject]@{
            Step        = 'SetStaticIPv4'
            Description = "Assign static IPv4 $IPv4Address/$PrefixLength to $InterfaceAlias"
            Command     = "netsh interface ipv4 set address name=$InterfaceAlias source=static address=$IPv4Address mask=$mask gateway=none store=persistent"
            Arguments   = @('interface', 'ipv4', 'set', 'address', "name=$InterfaceAlias", 'source=static', "address=$IPv4Address", "mask=$mask", 'gateway=none', 'store=persistent')
        })
    }

    if (-not $Apply) {
        return [PSCustomObject]@{
            InterfaceAlias    = $InterfaceAlias
            CurrentStatus     = $current
            PlannedActions    = @($plan)
            RecommendedNotes  = @(
                'Prefer a dedicated /30 for SMB and WinRM on a direct USB4 peer link when both ends are under your control.',
                'Keep the Thunderbolt / USB4 link isolated from Internet routing. The direct peer interface does not need a default gateway.',
                'If WinRM remains unavailable after addressing, set the remote USB4 connection profile to Private and enable PSRemoting on that machine.'
            )
        }
    }

    if (-not $PSCmdlet.ShouldProcess($InterfaceAlias, 'Apply Thunderbolt / USB4 optimization plan')) {
        return
    }

    # Inspect each native exit explicitly, even in shells that otherwise throw on
    # nonzero native exits. Earlier successful steps are not implicitly rolled back.
    $PSNativeCommandUseErrorActionPreference = $false
    foreach ($action in $plan) {
        $arguments = [string[]]$action.Arguments
        $global:LASTEXITCODE = 0
        $output = @(& netsh @arguments 2>&1)
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Thunderbolt optimization failed at $($action.Step) (netsh exit $exitCode). Remaining steps were not attempted. $($output -join [Environment]::NewLine)"
        }
    }

    $updated = @(Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias |
        Where-Object { $_.InterfaceAlias -eq $InterfaceAlias })
    if ($updated.Count -ne 1) {
        throw 'Thunderbolt optimization commands succeeded, but the selected adapter could not be uniquely verified.'
    }
    $verified = $updated[0]
    $mismatches = @(
        if ($verified.IPv4Metric -ne $InterfaceMetric) { "IPv4 metric expected $InterfaceMetric, observed $($verified.IPv4Metric)" }
        if ($verified.IPv6Metric -ne $InterfaceMetric) { "IPv6 metric expected $InterfaceMetric, observed $($verified.IPv6Metric)" }
        if ($verified.IPv4Mtu -ne $MtuBytes) { "IPv4 MTU expected $MtuBytes, observed $($verified.IPv4Mtu)" }
        if ($IPv4Address -and $IPv4Address -notin @($verified.IPv4Addresses)) { "IPv4 address $IPv4Address was not observed" }
    )
    if ($mismatches.Count -gt 0) {
        throw "Thunderbolt optimization postcondition failed: $($mismatches -join '; ')."
    }
    return $updated
}
