<#
.SYNOPSIS
Run one bound endpoint of the Windows USB4 TCP hardware test.
.DESCRIPTION
Use the hardware test coordinator to supply the endpoint request and trusted
bounded process runner. The ready role only polls for the selected listener.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][hashtable]$Request,
    [Parameter(Mandatory)][string]$ProcessRunnerSource
)

$ErrorActionPreference = 'Stop'

function Get-LinkLocalAddress([string]$Address) {
    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$parsed) -or -not $parsed.IsIPv6LinkLocal) {
        throw "Expected an IPv6 link-local address, got '$Address'."
    }
    $parsed
}

function ConvertTo-MacKey([string]$Address) {
    $Address.Replace('-', '').Replace(':', '').ToUpperInvariant()
}

function Assert-DirectRoute {
    $rows = @(Find-NetRoute -LocalIPAddress $source -RemoteIPAddress $target -InterfaceIndex $index)
    $routes = @($rows | Where-Object { $_.DestinationPrefix })
    $addresses = @($rows | Where-Object { $_.IPAddress })
    if ($routes.Count -ne 1 -or $addresses.Count -ne 1 -or
        @($rows | Where-Object InterfaceIndex -NE $index).Count -or
        $routes[0].NextHop -ne '::' -or $routes[0].DestinationPrefix -eq '::/0' -or
        ($addresses[0].IPAddress -split '%', 2)[0] -ne "$localAddress") {
        throw 'Source routing must select the USB4 interface and an on-link peer.'
    }
}

function Get-Usb4Counters {
    $current = @(Get-NetAdapter -IncludeHidden | Where-Object InterfaceIndex -EQ $index)
    if ($current.Count -ne 1 -or $current[0].Status -ne 'Up' -or
        [guid]$current[0].InterfaceGuid -ne $adapterGuid -or
        (ConvertTo-MacKey $current[0].MacAddress) -ne $localMac) {
        throw 'The USB4 adapter identity or link state changed during the measurement.'
    }
    $statistics = @(Get-NetAdapterStatistics -Name $current[0].Name)
    if ($statistics.Count -ne 1 -or $statistics[0].Name -ne $current[0].Name -or
        $null -eq $statistics[0].ReceivedBytes -or $null -eq $statistics[0].SentBytes) {
        throw 'Expected byte counters for the selected USB4 adapter.'
    }
    $statistics[0]
}

function Get-TemporaryFirewallRule([string]$Name) {
    try {
        Get-NetFirewallRule -PolicyStore PersistentStore -Name $Name
    } catch {
        if ($_.FullyQualifiedErrorId -ne 'CmdletizationQuery_NotFound_InstanceID,Get-NetFirewallRule') { throw }
    }
}

if (-not $IsWindows -or $Request.role -notin @('listen', 'client', 'ready')) { throw 'Expected a Windows USB4 endpoint role.' }
if ($Request.local.host -ne [Environment]::MachineName -or -not $Request.peer.host -or
    $Request.peer.host -eq $Request.local.host) { throw 'Endpoint host identity does not match this machine and a distinct peer.' }
$port = [int]$Request.tcp_port
$seconds = [int]$Request.seconds
$streams = [int]$Request.streams
if ($port -lt 1024 -or $port -gt 65535 -or $seconds -lt 1 -or $seconds -gt 60 -or
    $streams -lt 1 -or $streams -gt 16 -or $Request.allow_temporary_firewall -isnot [bool]) {
    throw 'Expected a non-privileged TCP port, 1–60 seconds, 1–16 streams, and an explicit firewall choice.'
}
$adapterGuid = [guid]$Request.local.adapter_guid
$localMac = ConvertTo-MacKey $Request.local.mac
$peerMac = ConvertTo-MacKey $Request.peer.mac
if ($localMac -notmatch '^[A-F0-9]{12}$' -or $peerMac -notmatch '^[A-F0-9]{12}$' -or
    $localMac -eq '000000000000' -or $peerMac -eq '000000000000' -or $localMac -eq $peerMac) {
    throw 'Expected distinct USB4 adapter MAC addresses.'
}
$localAddress = Get-LinkLocalAddress $Request.local.ip
$peerAddress = Get-LinkLocalAddress $Request.peer.ip
if ($Request.local.ip.Contains('%') -or $Request.peer.ip.Contains('%') -or $localAddress.Equals($peerAddress)) {
    throw 'Expected distinct unscoped USB4 addresses.'
}
$adapters = @(Get-NetAdapter -IncludeHidden | Where-Object { [guid]$_.InterfaceGuid -eq $adapterGuid })
if ($adapters.Count -ne 1 -or $adapters[0].Status -ne 'Up' -or
    $adapters[0].PnPDeviceID -notmatch '^USB4\\[^\\]+&PROT_USB4NET\\' -or
    (ConvertTo-MacKey $adapters[0].MacAddress) -ne $localMac) {
    throw 'Expected one Up USB4NET adapter matching the configured GUID and MAC.'
}
$adapter = $adapters[0]
$index = [int]$adapter.InterfaceIndex
if ($index -le 0) { throw 'The USB4 adapter has no valid interface index.' }
$source = "$localAddress%$index"
$target = "$peerAddress%$index"
$addresses = @(Get-NetIPAddress -InterfaceIndex $index -AddressFamily IPv6 | Where-Object {
    $_.AddressState -eq 'Preferred' -and ($_.IPAddress -split '%', 2)[0] -eq "$localAddress"
})
if ($addresses.Count -ne 1) { throw 'The configured local address is not preferred on the selected USB4 adapter.' }

$path = [string]$Request.local.iperf_path
if ($path -notmatch '^[A-Za-z]:[\\/]' -or [IO.Path]::GetExtension($path) -ne '.exe' -or
    $path.IndexOfAny([char[]]'*?[]') -ge 0) {
    throw 'Expected the absolute local path of an iperf3 executable.'
}
$file = Get-Item -LiteralPath $path
if ($file -isnot [IO.FileInfo] -or $file.PSDrive.DisplayRoot -or
    [IO.DriveInfo]::new($file.Directory.Root.FullName).DriveType -eq [IO.DriveType]::Network) {
    throw 'Expected an executable on a local filesystem drive.'
}
$executable = $file.FullName

if ($Request.role -eq 'ready') {
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    $scopedAddress = [Net.IPAddress]::Parse($source)
    do {
        $listeners = @([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object { $_.Port -eq $port -and ($_.Address.Equals($scopedAddress) -or $_.Address.Equals($localAddress)) })
        if ($listeners.Count -eq 1) {
            $connections = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
                Where-Object { $_.LocalAddress -in @($source, "$localAddress") })
            if ($connections.Count -eq 1) {
                $process = Get-Process -Id $connections[0].OwningProcess -ErrorAction SilentlyContinue
                $ready = $process.Path -eq $executable
            }
        }
        if (-not $ready) { Start-Sleep -Milliseconds 150 }
    } until ($ready -or $clock.Elapsed.TotalSeconds -ge 8)
    @{ ready = $ready } | ConvertTo-Json -Compress
    return
}

$neighbors = @(Get-NetNeighbor -InterfaceIndex $index -AddressFamily IPv6 | Where-Object {
    ($_.IPAddress -split '%', 2)[0] -eq "$peerAddress" -and
    (ConvertTo-MacKey $_.LinkLayerAddress) -eq $peerMac -and
    $_.State -in @('Reachable', 'Stale', 'Delay', 'Probe', 'Permanent')
})
if ($neighbors.Count -ne 1) { throw 'The selected interface does not have the expected USB4 peer IP/MAC neighbor.' }
Assert-DirectRoute
$interfaces = @(Get-NetIPInterface -InterfaceIndex $index -AddressFamily IPv6)
if ($interfaces.Count -ne 1 -or $interfaces[0].NlMtu -lt 1280) { throw 'Expected one usable IPv6 link MTU.' }
$mtu = [int]$interfaces[0].NlMtu

. ([scriptblock]::Create($ProcessRunnerSource))
$binaryHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
$versionProcess = Invoke-VigilBoundedProcess -FilePath $executable -Arguments @('--version') -TimeoutSeconds 4
if ($versionProcess.ExitCode -ne 0 -or $versionProcess.Stdout -notmatch '(?m)^iperf (\d+\.\d+(?:\.\d+)?)\b' -or
    [version]$Matches[1] -lt [version]'3.11') { throw 'The USB4 TCP hardware test requires iperf3 3.11 or newer.' }
$version = $Matches[1]

$arguments = @('-6', '-B', $source, '-p', "$port", '--json')
if ($Request.role -eq 'listen') {
    $arguments += @('-s', '-1')
} else {
    $arguments += @('-c', $target, '--connect-timeout', '4000', '--bidir', '-P', "$streams", '-t', "$seconds")
}
$ruleName = $null
try {
    if ($Request.role -eq 'listen' -and $Request.allow_temporary_firewall) {
        if ($adapter.Name.IndexOfAny([char[]]'*?[]') -ge 0) { throw 'An exact firewall interface alias is required.' }
        $ruleName = 'PC-AI USB4 TCP HIL ' + [guid]::NewGuid().ToString('N')
        New-NetFirewallRule -PolicyStore PersistentStore -Name $ruleName -DisplayName $ruleName -Direction Inbound `
            -Action Allow -Protocol TCP -LocalPort $port -LocalAddress "$localAddress" -RemoteAddress "$peerAddress" `
            -InterfaceAlias $adapter.Name -Program $executable -Profile Any | Out-Null
    }
    $before = Get-Usb4Counters
    $processResult = Invoke-VigilBoundedProcess -FilePath $executable -Arguments $arguments -TimeoutSeconds ($seconds + 25)
    if ($processResult.ExitCode -ne 0 -and -not $processResult.Stdout.Trim()) {
        throw "iperf3 $($Request.role) failed (exit $($processResult.ExitCode)): $($processResult.Stderr.Trim())"
    }
    $result = ConvertFrom-Json -InputObject $processResult.Stdout -AsHashtable
    if ($processResult.ExitCode -ne 0 -or $result -isnot [Collections.IDictionary] -or $result.error) {
        throw "iperf3 $($Request.role) failed (exit $($processResult.ExitCode)): $($result.error) $($processResult.Stderr.Trim())"
    }
    $after = Get-Usb4Counters
    Assert-DirectRoute
    if ($after.ReceivedBytes -le $before.ReceivedBytes -or $after.SentBytes -le $before.SentBytes) {
        throw 'The selected adapter did not report bidirectional counter growth.'
    }
} finally {
    if ($ruleName -and (Get-TemporaryFirewallRule $ruleName)) {
        Remove-NetFirewallRule -PolicyStore PersistentStore -Name $ruleName
        if (Get-TemporaryFirewallRule $ruleName) {
            throw 'The temporary USB4 test firewall rule still exists after cleanup.'
        }
    }
}

@{
    role = $Request.role
    host = [Environment]::MachineName
    index = $index
    source = $source
    peer = $target
    version = $version
    binary_sha256 = $binaryHash
    mtu = $mtu
    rx_delta = $after.ReceivedBytes - $before.ReceivedBytes
    tx_delta = $after.SentBytes - $before.SentBytes
    result = $result
} | ConvertTo-Json -Compress -Depth 25
