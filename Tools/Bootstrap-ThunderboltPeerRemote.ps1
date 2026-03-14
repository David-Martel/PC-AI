#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter()]
    [string]$InterfaceAlias,

    [Parameter()]
    [string]$IPv4Address = '172.31.240.2',

    [Parameter()]
    [ValidateRange(8, 30)]
    [int]$PrefixLength = 30,

    [Parameter()]
    [ValidateRange(1, 9999)]
    [int]$InterfaceMetric = 15,

    [Parameter()]
    [ValidateRange(1280, 65535)]
    [int]$MtuBytes = 62000,

    [Parameter()]
    [switch]$SetPrivateProfile = $true,

    [Parameter()]
    [switch]$EnablePsRemoting = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
    $adapter = Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Status -eq 'Up' -and (
                $_.InterfaceDescription -match 'USB4' -or
                $_.InterfaceDescription -match 'Thunderbolt' -or
                $_.InterfaceDescription -match 'P2P'
            )
        } |
        Sort-Object ifIndex |
        Select-Object -First 1

    if (-not $adapter) {
        throw 'No active USB4/Thunderbolt/P2P adapter found.'
    }

    $InterfaceAlias = $adapter.Name
}

$existing = @(Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue)
foreach ($row in $existing) {
    if ($row.IPAddress -ne $IPv4Address) {
        Remove-NetIPAddress -InputObject $row -Confirm:$false -ErrorAction SilentlyContinue
    }
}

$current = @(Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -eq $IPv4Address })
if ($current.Count -eq 0) {
    New-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $IPv4Address -PrefixLength $PrefixLength -Type Unicast -ErrorAction Stop | Out-Null
}

if ($SetPrivateProfile) {
    $profile = Get-NetConnectionProfile -InterfaceAlias $InterfaceAlias -ErrorAction SilentlyContinue
    if ($profile) {
        Set-NetConnectionProfile -InterfaceAlias $InterfaceAlias -NetworkCategory Private -ErrorAction SilentlyContinue | Out-Null
    }
}

netsh interface ipv4 set interface "name=$InterfaceAlias" "metric=$InterfaceMetric" | Out-Null
netsh interface ipv6 set interface $InterfaceAlias "metric=$InterfaceMetric" | Out-Null
netsh interface ipv4 set subinterface $InterfaceAlias "mtu=$MtuBytes" store=persistent | Out-Null

if ($EnablePsRemoting) {
    Enable-PSRemoting -Force -SkipNetworkProfileCheck | Out-Null
}

Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop |
    Select-Object Name, InterfaceDescription, Status, MacAddress, ifIndex
Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Select-Object IPAddress, PrefixLength
Get-NetConnectionProfile -InterfaceAlias $InterfaceAlias -ErrorAction SilentlyContinue |
    Select-Object InterfaceAlias, NetworkCategory, IPv4Connectivity
