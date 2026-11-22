#Requires -Version 7.2

<#
.SYNOPSIS
Inspect a Windows USB4 NetworkDirect preview host and an optional direct link.
.DESCRIPTION
Reports Windows build, Secure Boot, the preview feature flag and the local
path selected for one peer. The operator supplies the exact Microsoft-published
preview build once available and can pass a fresh USB4 route plan for the peer.
This command does not change firmware, registry, BCD, provider or network state,
run nd tools, or establish that the peer has the same build or working RDMA.
See docs/usb4/rdma-preview.md.
.EXAMPLE
./Get-PcaiUsb4RdmaReadiness.ps1
.EXAMPLE
./Get-PcaiUsb4RdmaReadiness.ps1 -PlanPath ./fresh-route-plan.json -PeerHost dtm-super-nuc
#>
[CmdletBinding(DefaultParameterSetName = 'Inventory')]
param(
    [Parameter(Mandatory, ParameterSetName = 'DirectLink')][string]$PlanPath,
    [Parameter(Mandatory, ParameterSetName = 'DirectLink')][string]$PeerHost,
    [ValidatePattern('^[0-9]{5}\.[0-9]{1,6}$')][string]$PublishedBuild,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments, DontShow)][string[]]$RemainingArguments = @()
)

function ConvertFrom-PcaiRdmaPlanTime {
    param($Value)
    if ($Value -is [DateTimeOffset]) { return $Value }
    if ($Value -is [DateTime] -and $Value.Kind -ne [DateTimeKind]::Unspecified) { return [DateTimeOffset]$Value }
    $parsed = [DateTimeOffset]::MinValue
    if ($Value -is [string] -and $Value -match '(?:Z|[+-][0-9]{2}:[0-9]{2})$' -and
        [DateTimeOffset]::TryParse($Value, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$parsed)) {
        return $parsed
    }
}

function Get-PcaiUsb4RdmaDirectPair {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$LocalHost,
        [Parameter(Mandatory)][string]$PeerHost,
        [DateTimeOffset]$Now = [DateTimeOffset]::UtcNow
    )

    $evaluated = ConvertFrom-PcaiRdmaPlanTime $Plan['evaluated_at_utc']
    $expires = ConvertFrom-PcaiRdmaPlanTime $Plan['expires_at_utc']
    if ($Plan['schema'] -cne 'pcai.usb4-route-plan/v1' -or -not $evaluated -or -not $expires -or
        $evaluated -gt $Now -or $expires -le $Now -or ($expires - $evaluated).TotalSeconds -gt 300) {
        throw 'A current pcai.usb4-route-plan/v1 with valid observation times is required.'
    }
    if ($LocalHost -eq $PeerHost) { throw 'A different directly connected peer is required.' }
    $paths = @($Plan['paths'] | Where-Object { $_.from -eq $LocalHost -and $_.to -eq $PeerHost })
    if ($paths.Count -ne 1 -or $paths[0].status -cne 'Direct' -or @($paths[0].hops).Count -ne 2 -or
        $paths[0].hops[0] -ne $LocalHost -or $paths[0].hops[1] -ne $PeerHost) {
        throw 'The route plan must name a single direct path to the requested peer.'
    }
    $edges = @($Plan['edges'] | Where-Object { $LocalHost -in $_.endpoints.host -and $PeerHost -in $_.endpoints.host })
    if ($edges.Count -ne 1 -or @($edges[0].endpoints).Count -ne 2) {
        throw 'The direct path must have one unambiguous physical edge with exactly two endpoints.'
    }
    $local = $edges[0].endpoints | Where-Object host -eq $LocalHost
    $peer = $edges[0].endpoints | Where-Object host -eq $PeerHost
    $addresses = @($local.ipv4_address, $peer.ipv4_address)
    $ips = foreach ($address in $addresses) {
        $ip = $null
        if (-not [Net.IPAddress]::TryParse([string]$address, [ref]$ip) -or
            $ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
            $ip.ToString() -cne $address -or $address.StartsWith('169.254.') -or [Net.IPAddress]::IsLoopback($ip)) {
            throw 'The direct edge requires canonical, non-link-local IPv4 addresses.'
        }
        $ip.GetAddressBytes() -join '.'
    }
    $netPrefix = $ips[0] -replace '\.[0-9]+$', ''
    $first = [int]($ips[0].Split('.')[-1])
    $second = [int]($ips[1].Split('.')[-1])
    $subnet = "$netPrefix.$($first -band 252)/30"
    if ($ips[1] -notlike "$netPrefix.*" -or ($first -band 252) -ne ($second -band 252) -or
        ($first -band 3) -notin 1,2 -or ($second -band 3) -notin 1,2 -or $first -eq $second -or
        $subnet -cne $edges[0].subnet -or $peer.ipv4_address -cne $paths[0].destination_ipv4) {
        throw 'The plan must give the direct link two distinct usable addresses in its /30.'
    }
    $interfaceGuid = [guid]::Empty
    $interfaceIndex = 0
    $mac = ([string]$local.mac) -replace '[:-]', ''
    if (-not [guid]::TryParse([string]$local.interface_guid, [ref]$interfaceGuid) -or $interfaceGuid -eq [guid]::Empty -or
        -not [int]::TryParse([string]$local.interface_index, [ref]$interfaceIndex) -or $interfaceIndex -le 0 -or
        $mac -notmatch '^[0-9A-Fa-f]{12}$') {
        throw 'The local edge endpoint lacks a usable USB4 interface identity.'
    }
    [pscustomobject]@{
        PeerHost = $PeerHost; SourceAddress = $ips[0]; PeerAddress = $ips[1]; Subnet = $subnet
        InterfaceGuid = $interfaceGuid; InterfaceIndex = $interfaceIndex; Mac = $mac
        ExpiresAtUtc = $expires.UtcDateTime.ToString('o')
    }
}

function Test-PcaiUsb4RdmaDirectRoute {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]$Pair,
        [AllowEmptyCollection()][object[]]$Adapter = @(),
        [AllowEmptyCollection()][object[]]$RouteSelection = @()
    )
    $live = @($Adapter | Where-Object InterfaceIndex -eq $Pair.InterfaceIndex)
    $address = @($RouteSelection | Where-Object { $_.PSObject.Properties['IPAddress'] })
    $route = @($RouteSelection | Where-Object { $_.PSObject.Properties['NextHop'] })
    if ($live.Count -ne 1 -or $address.Count -ne 1 -or $route.Count -ne 1) { return $false }
    $nic = $live[0]
    $source = $address[0]
    $egress = $route[0]
    $nicGuid = [guid]::Empty
    return ([guid]::TryParse([string]$nic.InterfaceGuid, [ref]$nicGuid) -and $nicGuid -eq $Pair.InterfaceGuid -and
        ($nic.MacAddress -replace '[:-]', '') -eq $Pair.Mac -and $nic.Status -eq 'Up' -and
        $nic.PnPDeviceID -match '^USB4\\[^\\]+&PROT_USB4NET\\' -and
        $source.IPAddress -eq $Pair.SourceAddress -and $source.InterfaceIndex -eq $Pair.InterfaceIndex -and
        $source.PrefixLength -eq 30 -and $source.PrefixOrigin -eq 'Manual' -and $source.AddressState -eq 'Preferred' -and
        $egress.InterfaceIndex -eq $Pair.InterfaceIndex -and $egress.NextHop -eq '0.0.0.0' -and
        $egress.DestinationPrefix -eq $Pair.Subnet)
}

function Get-PcaiUsb4RdmaBlocker {
    [CmdletBinding()]
    param([string]$Build, [string]$PublishedBuild, [string]$SecureBoot, [bool]$FeatureEnabled, [string]$RouteStatus)
    if (-not $PublishedBuild) { 'PublishedBuildRequired' }
    elseif ($PublishedBuild -ne $Build) { 'BuildMismatch' }
    if ($SecureBoot -eq 'Enabled') { 'SecureBootEnabled' }
    elseif ($SecureBoot -ne 'Disabled') { 'SecureBootUnknown' }
    if (-not $FeatureEnabled) { 'InterDomainRdmaDisabledOrUnknown' }
    if ($RouteStatus -ne 'Matched') { 'DirectRouteNotMatched' }
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($Help -or '--help' -in $RemainingArguments) { Get-Help $PSCommandPath -Detailed; return }
if ($RemainingArguments.Count -gt 0) { throw "Unexpected arguments: $($RemainingArguments -join ' ')" }
$hostName = ([Net.Dns]::GetHostName() -split '\.')[0].ToLowerInvariant()
$report = [ordered]@{
    schema = 'pcai.usb4-networkdirect-preflight/v1'
    host = $hostName
    observedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
    state = 'UnsupportedPlatform'
    qualification = 'PreflightOnly'
    authenticated = $false
    rdmaVerified = $false
}
if (-not $IsWindows) { [pscustomobject]$report; return }

$os = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop
$report.build = "$($os.CurrentBuild).$($os.UBR)"
$report.publishedBuild = $PublishedBuild
$report.secureBoot = try { if (Confirm-SecureBootUEFI -ErrorAction Stop) { 'Enabled' } else { 'Disabled' } } catch { 'Unknown' }
$report.interDomainRdmaEnabled = try {
    $flag = Get-ItemPropertyValue -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Services\Usb4HostRouter\Parameters' -Name EnableInterDomainRdma -ErrorAction Stop
    $flag -is [int] -and $flag -eq 1
} catch { $false }
$report.routeStatus = 'NotRequested'
if ($PSCmdlet.ParameterSetName -eq 'DirectLink') {
    try {
        $fullPath = [IO.Path]::GetFullPath($PlanPath, (Get-Location).ProviderPath)
        if ($fullPath -notmatch '^[A-Za-z]:\\' -or [IO.DriveInfo]::new($fullPath.Substring(0, 3)).DriveType -ne 'Fixed') {
            throw 'The route plan must be on a fixed local drive.'
        }
        $drive = [IO.Path]::GetPathRoot($fullPath)
        $cursor = $drive
        foreach ($component in $fullPath.Substring($drive.Length).Split('\')) {
            $cursor = Join-Path $cursor $component
            if ([IO.File]::GetAttributes($cursor).HasFlag([IO.FileAttributes]::ReparsePoint)) {
                throw 'The route plan path cannot traverse a reparse point.'
            }
        }
        $stream = [IO.File]::Open($fullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            if ($stream.Length -gt 1MB) { throw 'The route plan must be a local file of at most 1 MiB.' }
            $reader = [IO.StreamReader]::new($stream)
            $plan = $reader.ReadToEnd() | ConvertFrom-Json -AsHashtable -ErrorAction Stop
        } finally { $stream.Dispose() }
        $report.directPair = Get-PcaiUsb4RdmaDirectPair -Plan $plan -LocalHost $hostName -PeerHost $PeerHost
        $adapters = @(Get-NetAdapter -InterfaceIndex $report.directPair.InterfaceIndex -ErrorAction Stop)
        $selection = @(Find-NetRoute -RemoteIPAddress $report.directPair.PeerAddress -ErrorAction Stop)
        $report.routeStatus = if (Test-PcaiUsb4RdmaDirectRoute -Pair $report.directPair -Adapter $adapters -RouteSelection $selection) { 'Matched' } else { 'Mismatch' }
    } catch {
        $report.routeStatus = 'Rejected'
        $report.routeError = $_.Exception.Message
    }
}
$report.blockers = @(Get-PcaiUsb4RdmaBlocker -Build $report.build -PublishedBuild $PublishedBuild -SecureBoot $report.secureBoot -FeatureEnabled $report.interDomainRdmaEnabled -RouteStatus $report.routeStatus)
$report.state = if ($report.blockers.Count) { 'BlockedPrerequisite' } else { 'OperatorReviewRequired' }
[pscustomobject]$report
