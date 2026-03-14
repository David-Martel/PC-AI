#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [string]$InterfaceAlias,

    [Parameter()]
    [string]$IPv4Address = '172.31.240.1',

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
    [switch]$SetPrivateProfile = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-LegacyScript {
    param([Parameter(Mandatory)][string]$Script)

    $wrapped = "`$ProgressPreference = 'SilentlyContinue'`r`n$Script"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrapped))
    $output = & powershell.exe -NoLogo -NoProfile -EncodedCommand $encoded
    if ($LASTEXITCODE -ne 0) {
        throw 'Windows PowerShell Thunderbolt link initialization failed.'
    }
    return ($output -join [Environment]::NewLine).Trim()
}

if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
    $detectedAlias = Invoke-LegacyScript -Script @"
`$adapter = Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue |
    Where-Object {
        `$_.Status -eq 'Up' -and (
            `$_.InterfaceDescription -match 'USB4' -or
            `$_.InterfaceDescription -match 'Thunderbolt' -or
            `$_.InterfaceDescription -match 'P2P'
        )
    } |
    Sort-Object ifIndex |
    Select-Object -First 1 -ExpandProperty Name
if (-not `$adapter) { throw 'No active USB4/Thunderbolt/P2P adapter found.' }
`$adapter
"@
    $InterfaceAlias = $detectedAlias.Trim()
}

$null = Invoke-LegacyScript -Script @"
`$alias = '$($InterfaceAlias.Replace("'", "''"))'
`$ip = '$IPv4Address'
`$prefix = $PrefixLength
`$metric = $InterfaceMetric
`$mtu = $MtuBytes
`$setPrivate = $(if ($SetPrivateProfile) { '$true' } else { '$false' })

`$existing = @(Get-NetIPAddress -InterfaceAlias `$alias -AddressFamily IPv4 -ErrorAction SilentlyContinue)
foreach (`$row in `$existing) {
    if (`$row.IPAddress -ne `$ip) {
        Remove-NetIPAddress -InputObject `$row -Confirm:`$false -ErrorAction SilentlyContinue
    }
}

`$current = @(Get-NetIPAddress -InterfaceAlias `$alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { `$_.IPAddress -eq `$ip })
if (`$current.Count -eq 0) {
    New-NetIPAddress -InterfaceAlias `$alias -IPAddress `$ip -PrefixLength `$prefix -Type Unicast -ErrorAction Stop | Out-Null
}

if (`$setPrivate) {
    `$profile = Get-NetConnectionProfile -InterfaceAlias `$alias -ErrorAction SilentlyContinue
    if (`$profile) {
        Set-NetConnectionProfile -InterfaceAlias `$alias -NetworkCategory Private -ErrorAction SilentlyContinue | Out-Null
    }
}
"@

& netsh interface ipv4 set interface ("name=$InterfaceAlias") ("metric=$InterfaceMetric") | Out-Null
& netsh interface ipv6 set interface $InterfaceAlias ("metric=$InterfaceMetric") | Out-Null
& netsh interface ipv4 set subinterface $InterfaceAlias ("mtu=$MtuBytes") 'store=persistent' | Out-Null

$status = Invoke-LegacyScript -Script @"
Get-NetAdapter -Name '$($InterfaceAlias.Replace("'", "''"))' -ErrorAction Stop |
    Select-Object Name, InterfaceDescription, Status, MacAddress, ifIndex |
    ConvertTo-Json -Depth 4
"@ | ConvertFrom-Json

$ipv4 = Invoke-LegacyScript -Script @"
Get-NetIPAddress -InterfaceAlias '$($InterfaceAlias.Replace("'", "''"))' -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Select-Object IPAddress, PrefixLength |
    ConvertTo-Json -Depth 4
"@ | ConvertFrom-Json

$profileJson = Invoke-LegacyScript -Script @"
Get-NetConnectionProfile -InterfaceAlias '$($InterfaceAlias.Replace("'", "''"))' -ErrorAction SilentlyContinue |
    Select-Object InterfaceAlias, NetworkCategory, IPv4Connectivity |
    ConvertTo-Json -Depth 4
"@
$profile = if ($profileJson) { $profileJson | ConvertFrom-Json } else { $null }

[pscustomobject]@{
    InterfaceAlias    = $InterfaceAlias
    Adapter           = $status
    IPv4              = @($ipv4)
    ConnectionProfile = $profile
    SuggestedPeerIP   = '172.31.240.2'
    RemoteBootstrap   = 'C:\codedev\PC_AI\Tools\Bootstrap-ThunderboltPeerRemote.ps1'
}
