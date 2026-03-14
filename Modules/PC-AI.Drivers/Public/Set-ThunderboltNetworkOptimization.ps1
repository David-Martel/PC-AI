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
    state. With -Apply, it executes the corresponding netsh commands.

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

    $status = if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
        @(Get-ThunderboltNetworkStatus)
    } else {
        @(Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias)
    }
    if ($status.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($InterfaceAlias)) {
        $status = @(Get-ThunderboltNetworkStatus)
    }
    if ($status.Count -eq 0) {
        throw 'No Thunderbolt / USB4 interface matched the requested adapter.'
    }

    $current = $status | Select-Object -First 1
    $InterfaceAlias = [string]$current.InterfaceAlias
    $plan = [System.Collections.Generic.List[PSCustomObject]]::new()

    $plan.Add([PSCustomObject]@{
        Step        = 'SetIPv4Metric'
        Description = "Set IPv4 metric on $InterfaceAlias to $InterfaceMetric"
        Command     = "netsh interface ipv4 set interface name=$InterfaceAlias metric=$InterfaceMetric"
    })
    $plan.Add([PSCustomObject]@{
        Step        = 'SetIPv6Metric'
        Description = "Set IPv6 metric on $InterfaceAlias to $InterfaceMetric"
        Command     = "netsh interface ipv6 set interface $InterfaceAlias metric=$InterfaceMetric"
    })
    $plan.Add([PSCustomObject]@{
        Step        = 'SetMtu'
        Description = "Set interface MTU on $InterfaceAlias to $MtuBytes"
        Command     = "netsh interface ipv4 set subinterface $InterfaceAlias mtu=$MtuBytes store=persistent"
    })

    if ($IPv4Address) {
        $mask = ConvertTo-IPv4Mask -Length $PrefixLength
        $plan.Add([PSCustomObject]@{
            Step        = 'SetStaticIPv4'
            Description = "Assign static IPv4 $IPv4Address/$PrefixLength to $InterfaceAlias"
            Command     = "netsh interface ipv4 set address name=$InterfaceAlias source=static address=$IPv4Address mask=$mask gateway=none store=persistent"
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

    & netsh interface ipv4 set interface ("name=$InterfaceAlias") ("metric=$InterfaceMetric") | Out-Null
    & netsh interface ipv6 set interface $InterfaceAlias ("metric=$InterfaceMetric") | Out-Null
    & netsh interface ipv4 set subinterface $InterfaceAlias ("mtu=$MtuBytes") 'store=persistent' | Out-Null

    if ($IPv4Address) {
        $mask = ConvertTo-IPv4Mask -Length $PrefixLength
        & netsh interface ipv4 set address ("name=$InterfaceAlias") 'source=static' ("address=$IPv4Address") ("mask=$mask") 'gateway=none' 'store=persistent' | Out-Null
    }

    return @(Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias)
}
