#Requires -Version 7.0
function Find-ThunderboltPeer {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string[]]$ComputerNameCandidates = @(
            'DESKTOP-86FC90B',
            'dtm-supernuc',
            'dtm-work',
            'dtm-work-mobo'
        ),

        [Parameter()]
        [int]$WinRmPort = 5985,

        [Parameter()]
        [ValidateRange(250, 10000)]
        [int]$TcpTimeoutMs = 1500,

        [Parameter()]
        [switch]$IncludeRaw
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    # ── Phase 1: Discover Thunderbolt/USB4 adapters via CIM ──────────────
    $adapters = @(Get-CimInstance -ClassName Win32_NetworkAdapter -ErrorAction SilentlyContinue |
        Where-Object {
            $_.NetConnectionID -and (
                $_.Description -match 'USB4|Thunderbolt|P2P' -or
                $_.Name -match 'USB4|Thunderbolt|P2P' -or
                $_.PNPDeviceID -match 'PROT_USB4NET|THUNDERBOLT|USB4'
            )
        } |
        Sort-Object Index |
        ForEach-Object {
            [PSCustomObject]@{
                Name                 = $_.Name
                InterfaceDescription = $_.Description
                InterfaceAlias       = $_.NetConnectionID
                Status               = if ($_.NetEnabled) { 'Up' } else { 'Disconnected' }
                MacAddress           = $_.MACAddress
                LinkSpeed            = $_.Speed
                ifIndex              = $_.Index
            }
        })

    # ── Phase 2: Gather IP addresses and neighbors for matched adapters ──
    $interfaceIndexes = @($adapters | ForEach-Object { $_.ifIndex })
    $localAddresses = @()
    $neighbors = @()

    if ($interfaceIndexes.Count -gt 0) {
        $indexSet = [System.Collections.Generic.HashSet[int]]::new()
        foreach ($idx in $interfaceIndexes) { [void]$indexSet.Add($idx) }

        # IP addresses via StandardCimv2
        $localAddresses = @(
            Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetIPAddress -ErrorAction SilentlyContinue |
            Where-Object { $indexSet.Contains([int]$_.InterfaceIndex) } |
            ForEach-Object {
                $family = switch ([int]$_.AddressFamily) { 2 { 'IPv4' }; 23 { 'IPv6' }; default { 'Other' } }
                [PSCustomObject]@{
                    InterfaceIndex  = $_.InterfaceIndex
                    InterfaceAlias  = $_.InterfaceAlias
                    AddressFamily   = $family
                    IPAddress       = $_.IPAddress
                    PrefixLength    = $_.PrefixLength
                }
            })

        # Neighbors via StandardCimv2
        $neighbors = @(
            Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -ErrorAction SilentlyContinue |
            Where-Object {
                $indexSet.Contains([int]$_.InterfaceIndex) -and
                $_.IPAddress -and
                $_.State -notin @(0, '0', 'Unreachable', 'Invalid') -and
                $_.IPAddress -notin @('255.255.255.255', '169.254.255.255') -and
                $_.IPAddress -notmatch '^(22[4-9]|23\d)\.' -and
                $_.IPAddress -notlike 'ff*' -and
                $_.LinkLayerAddress -and
                $_.LinkLayerAddress -notin @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff') -and
                $_.IPAddress -ne 'ff02::1'
            } |
            ForEach-Object {
                [PSCustomObject]@{
                    InterfaceIndex   = $_.InterfaceIndex
                    IPAddress        = $_.IPAddress
                    LinkLayerAddress = $_.LinkLayerAddress
                    State            = [string]$_.State
                }
            })
    }

    # ── Phase 3: Build candidate target set ──────────────────────────────
    $candidateTargets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $ComputerNameCandidates) {
        if (-not [string]::IsNullOrWhiteSpace($name)) {
            [void]$candidateTargets.Add($name.Trim())
        }
    }
    foreach ($neighbor in $neighbors) {
        if (-not [string]::IsNullOrWhiteSpace($neighbor.IPAddress)) {
            [void]$candidateTargets.Add($neighbor.IPAddress)
        }
    }

    # ── Phase 4: DNS resolution (async with timeout) ────────────────────
    $dnsTimeoutMs = [Math]::Min($TcpTimeoutMs, 2000)
    $dnsTasks = [System.Collections.Generic.Dictionary[string, System.Threading.Tasks.Task]]::new()
    foreach ($target in $candidateTargets) {
        $dnsTasks[$target] = [System.Net.Dns]::GetHostAddressesAsync($target)
    }

    $nameResolution = @(foreach ($target in @($candidateTargets)) {
        $task = $dnsTasks[$target]
        $resolved = @()
        try {
            if ($task.Wait($dnsTimeoutMs)) {
                $resolved = @($task.Result | ForEach-Object { $_.IPAddressToString })
            }
        } catch { }
        [PSCustomObject]@{
            Target          = $target
            Resolved        = $resolved.Count -gt 0
            ResolvedAddress = $resolved
        }
    })

    # ── Phase 5: TCP WinRM probe (parallel) ──────────────────────────────
    $portCapture = $WinRmPort
    $timeoutCapture = $TcpTimeoutMs
    $winRmCandidates = @($candidateTargets | ForEach-Object -ThrottleLimit 8 -Parallel {
        $target = $_
        $tcpReachable = $false
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $async = $client.BeginConnect($target, $using:portCapture, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne($using:timeoutCapture, $false)) {
                try {
                    $client.EndConnect($async)
                    $tcpReachable = $client.Connected
                } catch {
                    $tcpReachable = $false
                }
            }
        } catch {
            $tcpReachable = $false
        } finally {
            $client.Close()
        }
        [PSCustomObject]@{
            Target       = $target
            WinRmPort    = $using:portCapture
            TcpTimeoutMs = $using:timeoutCapture
            TcpReachable = [bool]$tcpReachable
        }
    })

    # ── Phase 6: Optional raw diagnostics ────────────────────────────────
    $raw = $null
    if ($IncludeRaw) {
        $raw = [PSCustomObject]@{
            IpConfig = (ipconfig /all | Out-String).Trim()
            Arp      = (arp -a | Out-String).Trim()
            Route    = (route print | Out-String).Trim()
        }
    }

    # ── Result ───────────────────────────────────────────────────────────
    return [PSCustomObject]@{
        Timestamp       = Get-Date
        LocalComputer   = $env:COMPUTERNAME
        AdapterCount    = $adapters.Count
        Adapters        = @($adapters)
        LocalAddresses  = @($localAddresses)
        Neighbors       = @($neighbors)
        NameResolution  = @($nameResolution)
        WinRmCandidates = @($winRmCandidates)
        Raw             = $raw
    }
}
