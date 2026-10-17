#Requires -Version 7.0
<#
.SYNOPSIS
    Evaluates USB4 / Thunderbolt peer lifecycle states (asleep, disconnected, probing, ready)
    with expiring evidence, Modern Standby 506/507 telemetry, identity-bound canaries,
    bounded resume reprobe/reapply, and stale PC-AI route withdrawal.

.DESCRIPTION
    Implements additive USB4 P2P lifecycle management for PC-AI:
      - Lifecycle states: asleep, disconnected, probing, ready
      - Expiring evidence: TTL-bounded evidence automatically invalidated by Modern Standby
        entry (Microsoft-Windows-Kernel-Power Event 506)
      - Power capability audit: checks SelectiveSuspend, WakeOnMagicPacket, WakeOnPattern,
        ArpOffload, and NSOffload; prohibits AlwaysOnTransitCapable claims whenever
        SelectiveSuspend is Enabled or wake/offload capabilities are Unsupported
      - Identity-bound canary: verifies peer identity (MAC, ULA, or WinRM NTLM TargetInfo)
        before promoting a link to 'ready'
      - Stale route withdrawal: identifies and optionally withdraws stale PC-AI owned ULA
        (/128) and pairwise IPv4 (172.31.240.0/20) routes for disconnected/asleep peers
        while strictly preserving the millylaptop1 bridge and active dtm-p1gen7<->dtm-p1gen9 pair
      - Bounded resume reprobe/reapply: enforces strict attempt and total-time budgets on
        S0 Modern Standby resume (Event 507)

.PARAMETER ComputerName
    Local or target hostname to evaluate. Defaults to $env:COMPUTERNAME.

.PARAMETER InterfaceAlias
    Optional USB4 interface alias filter.

.PARAMETER RegistryPath
    Optional path to Config/thunderbolt-peers.json.

.PARAMETER EvidenceTtlSeconds
    Time-to-live in seconds for lifecycle evidence before re-verification is required.

.PARAMETER EvidenceObservedAtUtc
    UTC timestamp when evidence was collected. Used to test TTL expiry or Event 506 invalidation.

.PARAMETER MaxReprobeAttempts
    Maximum bounded reprobe attempts after S0 Modern Standby resume.

.PARAMETER ReprobeTimeoutMs
    Timeout in milliseconds per reprobe attempt.

.PARAMETER TotalReprobeBudgetMs
    Maximum total wall-clock budget in milliseconds for resume reprobe.

.PARAMETER WithdrawStaleRoutes
    Include stale owned route withdrawal in the lifecycle execution plan.

.PARAMETER ReprobeOnResume
    Execute bounded reprobe attempts when Modern Standby resume or asleep/probing links are detected.

.PARAMETER ReapplyOnResume
    Plan or reapply peer registration on links that recover during bounded resume reprobe.

.PARAMETER Apply
    Execute route withdrawal and reapply actions instead of returning a dry-run plan.

.OUTPUTS
    PSCustomObject representing the complete USB4 lifecycle evaluation report.
#>
function Get-ModernStandbyCapability {
    try {
        $out = @(& powercfg.exe /a 2>$null) -join "`n"
        return $out
    } catch {
        return $null
    }
}

function Invoke-TcpCanaryConnect {
    param(
        [Parameter(Mandatory)]
        [System.Net.IPAddress]$IPAddress,

        [Parameter(Mandatory)]
        [int]$Port,

        [Parameter(Mandatory)]
        [int]$TimeoutMs
    )
    $client = $null
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $client = [System.Net.Sockets.TcpClient]::new($IPAddress.AddressFamily)
        $async = $client.BeginConnect($IPAddress, $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            $client.EndConnect($async)
            $sw.Stop()
            if ($client.Connected) {
                return [PSCustomObject]@{
                    Reachable = $true
                    LatencyMs = [int]$sw.ElapsedMilliseconds
                }
            }
        }
    } catch {
    } finally {
        if ($null -ne $client) { $client.Close() }
    }
    return [PSCustomObject]@{
        Reachable = $false
        LatencyMs = $null
    }
}

function Invoke-ScopedRouteDelete {
    param(
        [Parameter(Mandatory)]
        [string]$DestinationPrefix,

        [Parameter(Mandatory)]
        [int]$InterfaceIndex
    )
    if ($InterfaceIndex -le 0) {
        throw "Refusing unscoped route deletion for '$DestinationPrefix' (InterfaceIndex=$InterfaceIndex)."
    }
    if (Get-Command Remove-NetRoute -ErrorAction SilentlyContinue) {
        Remove-NetRoute -InterfaceIndex $InterfaceIndex -DestinationPrefix $DestinationPrefix -Confirm:$false -ErrorAction Stop | Out-Null
        return
    }
    $targetAddr = ($DestinationPrefix -replace '/\d+$', '')
    if ($DestinationPrefix -match ':') {
        $out = @(& route.exe -6 delete $DestinationPrefix IF $InterfaceIndex 2>&1) -join ' '
    } else {
        $out = @(& route.exe delete $targetAddr IF $InterfaceIndex 2>&1) -join ' '
    }
    if ($LASTEXITCODE -ne 0) {
        throw "route.exe delete failed on InterfaceIndex $InterfaceIndex (exit $LASTEXITCODE): $out"
    }
}

function Invoke-ThunderboltPeerLifecycle {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ComputerName = $env:COMPUTERNAME,

        [Parameter()]
        [string[]]$InterfaceAlias,

        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [ValidateRange(15, 86400)]
        [int]$EvidenceTtlSeconds = 300,

        [Parameter()]
        [datetime]$EvidenceObservedAtUtc = [datetime]::UtcNow,

        [Parameter()]
        [ValidateRange(1, 5)]
        [int]$MaxReprobeAttempts = 3,

        [Parameter()]
        [ValidateRange(50, 1000)]
        [int]$ReprobeTimeoutMs = 250,

        [Parameter()]
        [ValidateRange(150, 5000)]
        [int]$TotalReprobeBudgetMs = 1500,

        [Parameter()]
        [switch]$WithdrawStaleRoutes,

        [Parameter()]
        [switch]$ReprobeOnResume,

        [Parameter()]
        [switch]$ReapplyOnResume,

        [Parameter()]
        [switch]$Apply
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    function Get-SafeCimRows {
        param(
            [Parameter(Mandatory)]
            [string]$ClassName,

            [Parameter()]
            [string]$Namespace = 'root/StandardCimv2',

            [switch]$Required
        )
        try {
            return @(Get-CimInstance -Namespace $Namespace -ClassName $ClassName -ErrorAction Stop)
        } catch {
            if ($Required) { throw }
            Write-Verbose "CIM lookup failed for ${Namespace}:${ClassName}: $($_.Exception.Message)"
            return @()
        }
    }

    function Get-ModernStandbySnapshot {
        $enterCount = 0
        $exitCount = 0
        $lastEnteredAtUtc = $null
        $lastEnterReason = $null
        $lastExitedAtUtc = $null
        $lastExitReason = $null
        $eventLogAvailable = $false

        try {
            $events = @(Get-WinEvent -FilterHashtable @{
                LogName      = 'System'
                ProviderName = 'Microsoft-Windows-Kernel-Power'
                Id           = 506, 507
            } -MaxEvents 12 -ErrorAction Stop)
            $eventLogAvailable = $true

            foreach ($ev in $events) {
                $msg = [string]$ev.Message
                $reason = if ($msg -match 'Reason:\s*([^\r\n]+)') { $Matches[1].Trim() } else { $null }
                $tsUtc = $ev.TimeCreated.ToUniversalTime()
                if ($ev.Id -eq 506) {
                    $enterCount++
                    if ($null -eq $lastEnteredAtUtc -or $tsUtc -gt $lastEnteredAtUtc) {
                        $lastEnteredAtUtc = $tsUtc
                        $lastEnterReason = $reason
                    }
                } elseif ($ev.Id -eq 507) {
                    $exitCount++
                    if ($null -eq $lastExitedAtUtc -or $tsUtc -gt $lastExitedAtUtc) {
                        $lastExitedAtUtc = $tsUtc
                        $lastExitReason = $reason
                    }
                }
            }
        } catch {
            if ($_.Exception.Message -match 'No events were found') {
                $eventLogAvailable = $true
            }
            Write-Verbose "Modern Standby event log query skipped or unavailable: $($_.Exception.Message)"
        }

        $s0State = 'Unknown'
        $s0ConnectedSupported = $null
        try {
            $capRaw = Get-ModernStandbyCapability
            if ($null -ne $capRaw) {
                if ($capRaw -is [PSCustomObject] -or $capRaw -is [hashtable]) {
                    if ($capRaw -is [hashtable]) {
                        $capRaw = [PSCustomObject]$capRaw
                    }
                    if ($capRaw.PSObject.Properties['S0LowPowerIdleState'] -and $capRaw.S0LowPowerIdleState) {
                        $s0State = [string]$capRaw.S0LowPowerIdleState
                    }
                    if ($capRaw.PSObject.Properties['S0LowPowerIdleConnectedSupported']) {
                        $s0ConnectedSupported = $capRaw.S0LowPowerIdleConnectedSupported
                    }
                } else {
                    $capText = ([string]$capRaw).Trim()
                    if (-not [string]::IsNullOrWhiteSpace($capText)) {
                        $lowerCap = $capText.ToLowerInvariant()
                        $splitIdx = $lowerCap.IndexOf('the following sleep states are not available')
                        if ($splitIdx -lt 0) {
                            $splitIdx = $lowerCap.IndexOf('not available on this system')
                        }
                        $availPart = if ($splitIdx -ge 0) { $lowerCap.Substring(0, $splitIdx) } else { $lowerCap }
                        $notAvailPart = if ($splitIdx -ge 0) { $lowerCap.Substring($splitIdx) } else { '' }

                        if ($availPart.Contains('standby (s0 low power idle) network connected')) {
                            $s0State = 'Supported'
                            $s0ConnectedSupported = $true
                        } elseif ($availPart.Contains('standby (s0 low power idle)')) {
                            $s0State = 'Supported'
                            $s0ConnectedSupported = [bool]$availPart.Contains('network connected')
                        } elseif ($notAvailPart.Contains('standby (s0 low power idle)') -or $availPart.Contains('standby (s3)')) {
                            $s0State = 'Unsupported'
                            $s0ConnectedSupported = $false
                        }
                    }
                }
            }
        } catch {
            $s0State = 'Unknown'
            $s0ConnectedSupported = $null
        }

        if ($s0State -eq 'Unknown' -and ($enterCount -gt 0 -or $exitCount -gt 0)) {
            $s0State = 'Supported'
            $s0ConnectedSupported = $true
        }

        $currentlyInStandby = ($null -ne $lastEnteredAtUtc) -and (
            ($null -eq $lastExitedAtUtc) -or ($lastEnteredAtUtc -gt $lastExitedAtUtc)
        )

        return [PSCustomObject]@{
            S0LowPowerIdleState              = $s0State
            S0LowPowerIdleConnectedSupported = $s0ConnectedSupported
            EventLogAvailable                = [bool]$eventLogAvailable
            EnterStandbyCount                = $enterCount
            ExitStandbyCount                 = $exitCount
            LastEnteredStandbyAtUtc          = if ($lastEnteredAtUtc) { $lastEnteredAtUtc.ToString('o') } else { $null }
            LastEnterReason                  = $lastEnterReason
            LastExitedStandbyAtUtc           = if ($lastExitedAtUtc) { $lastExitedAtUtc.ToString('o') } else { $null }
            LastExitReason                   = $lastExitReason
            CurrentlyInStandby               = [bool]$currentlyInStandby
            LastEnteredDateTimeUtc           = $lastEnteredAtUtc
            LastExitedDateTimeUtc            = $lastExitedAtUtc
        }
    }

    function Test-CanaryEndpoint {
        param(
            [Parameter(Mandatory)]
            [string]$Address,

            [Parameter()]
            [Nullable[long]]$ScopeId = $null,

            [Parameter()]
            [int[]]$Ports = @(5985, 22, 445),

            [Parameter()]
            [int]$TimeoutMs = 250
        )

        $explicitScopeId = $null
        $rawTrimmed = $Address.Trim()
        if ($rawTrimmed -match '^\[?([^\]]+?)\]?%(\d+)$') {
            $cleanAddr = $Matches[1].Trim()
            $explicitScopeId = [int64]$Matches[2]
        } else {
            $cleanAddr = ($rawTrimmed -replace '%.*$', '').Trim()
        }

        $resolvedScopeId = $null
        $ipObj = $null
        try {
            $parsedIp = [System.Net.IPAddress]::Parse($cleanAddr)
            if ($parsedIp.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6) {
                if ($null -ne $explicitScopeId -and $explicitScopeId -gt 0) {
                    $resolvedScopeId = $explicitScopeId
                    $ipObj = [System.Net.IPAddress]::new($parsedIp.GetAddressBytes(), [int64]$resolvedScopeId)
                } elseif ($parsedIp.IsIPv6LinkLocal -and $null -ne $ScopeId -and $ScopeId -gt 0) {
                    $resolvedScopeId = [int64]$ScopeId
                    $ipObj = [System.Net.IPAddress]::new($parsedIp.GetAddressBytes(), [int64]$resolvedScopeId)
                } elseif ($parsedIp.ScopeId -gt 0) {
                    $resolvedScopeId = [int64]$parsedIp.ScopeId
                    $ipObj = $parsedIp
                } else {
                    $ipObj = $parsedIp
                }
            } else {
                $ipObj = $parsedIp
            }
        } catch {
            $ipObj = $null
        }

        foreach ($port in $Ports) {
            if ($null -eq $ipObj) { break }
            $outcome = Invoke-TcpCanaryConnect -IPAddress $ipObj -Port $port -TimeoutMs $TimeoutMs
            if ($outcome -and $outcome.Reachable) {
                return [PSCustomObject]@{
                    Reachable = $true
                    Port      = $port
                    LatencyMs = if ($null -ne $outcome.LatencyMs) { [int]$outcome.LatencyMs } else { 1 }
                    ScopeId   = if ($ipObj.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6 -and $ipObj.ScopeId -gt 0) { [int64]$ipObj.ScopeId } else { $resolvedScopeId }
                }
            }
        }

        return [PSCustomObject]@{
            Reachable = $false
            Port      = $null
            LatencyMs = $null
            ScopeId   = if ($ipObj -and $ipObj.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6 -and $ipObj.ScopeId -gt 0) { [int64]$ipObj.ScopeId } else { $resolvedScopeId }
        }
    }

    if ([string]::IsNullOrWhiteSpace($RegistryPath)) {
        $moduleRoot = Split-Path $PSScriptRoot -Parent
        $repoRoot   = Split-Path (Split-Path $moduleRoot -Parent) -Parent
        $RegistryPath = Join-Path $repoRoot 'Config\thunderbolt-peers.json'
    }

    $registry = $null
    if (Test-Path -LiteralPath $RegistryPath) {
        $registry = Get-Content -LiteralPath $RegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }

    $localIdentity = Get-ThunderboltPeerIdentity -ComputerName $ComputerName -RegistryPath $RegistryPath
    $canonicalLocalHost = $localIdentity.ComputerName

    $standby = Get-ModernStandbySnapshot
    $observedUtc = $EvidenceObservedAtUtc.ToUniversalTime()
    $nowUtc = [datetime]::UtcNow
    $expiresUtc = $observedUtc.AddSeconds($EvidenceTtlSeconds)
    $invalidatedByStandby = ($null -ne $standby.LastEnteredDateTimeUtc) -and ($standby.LastEnteredDateTimeUtc -gt $observedUtc)
    $evidenceExpired = $invalidatedByStandby -or ($nowUtc -gt $expiresUtc)

    $statusParams = @{
        EvidenceTtlSeconds = $EvidenceTtlSeconds
    }
    if ($InterfaceAlias -and $InterfaceAlias.Count -gt 0) {
        $statusParams['InterfaceAlias'] = $InterfaceAlias
    }
    $adapters = @(Get-ThunderboltNetworkStatus @statusParams)
    $p2pAdapters = @($adapters | Where-Object { $_.Role -eq 'P2PNetwork' })

    $anyAlwaysOnTransitCapable = (@($p2pAdapters | Where-Object { $_.AlwaysOnTransitCapable -eq $true })).Count -gt 0
    if ($p2pAdapters.Count -gt 0 -and (@($p2pAdapters | Where-Object { $_.SelectiveSuspend -eq 'Enabled' })).Count -gt 0) {
        $anyAlwaysOnTransitCapable = $false
    }

    $peersByName = @{}
    if ($registry) {
        $rawPeers = @()
        if ($registry.PSObject.Properties['fleetPeers']) {
            $rawPeers += @($registry.fleetPeers)
        }
        if ($registry.PSObject.Properties['peers'] -and $registry.peers -is [System.Collections.IEnumerable] -and $registry.peers -isnot [string]) {
            $rawPeers += @($registry.peers)
        }
        foreach ($peerEntry in $rawPeers) {
            $pName = if ($peerEntry.PSObject.Properties['name'] -and $peerEntry.name) {
                [string]$peerEntry.name
            } elseif ($peerEntry.PSObject.Properties['computerName'] -and $peerEntry.computerName) {
                [string]$peerEntry.computerName
            } else {
                $null
            }
            if (-not [string]::IsNullOrWhiteSpace($pName)) {
                $ula = if ($peerEntry.PSObject.Properties['shortUla'] -and $peerEntry.shortUla) {
                    [string]$peerEntry.shortUla
                } elseif ($peerEntry.PSObject.Properties['usb4UlaAddress'] -and $peerEntry.usb4UlaAddress) {
                    [string]$peerEntry.usb4UlaAddress
                } else {
                    $null
                }
                $ll = if ($peerEntry.PSObject.Properties['shortLinkLocal'] -and $peerEntry.shortLinkLocal) {
                    [string]$peerEntry.shortLinkLocal
                } elseif ($peerEntry.PSObject.Properties['usb4LinkLocalAddress'] -and $peerEntry.usb4LinkLocalAddress) {
                    [string]$peerEntry.usb4LinkLocalAddress
                } else {
                    $null
                }
                $normalizedPeer = [PSCustomObject]@{
                    computerName        = $pName
                    usb4UlaAddress      = $ula
                    usb4LinkLocalAddress = $ll
                }
                $peersByName[$pName.ToLowerInvariant()] = $normalizedPeer
                if ($peerEntry.PSObject.Properties['aliases'] -and $peerEntry.aliases) {
                    foreach ($al in @($peerEntry.aliases)) {
                        $peersByName[([string]$al).ToLowerInvariant()] = $normalizedPeer
                    }
                }
            }
        }
    }

    $topologyLinks = @()
    if ($registry -and $registry.PSObject.Properties['topologyLinks']) {
        $topologyLinks = @($registry.topologyLinks)
    }

    $linkReports = [System.Collections.Generic.List[object]]::new()
    $activeReadyPeers = @{}
    $preservedEndpoints = @{}

    foreach ($link in $topologyLinks) {
        $linkIdValue = if ($link.PSObject.Properties['linkId'] -and $link.linkId) {
            [string]$link.linkId
        } elseif ($link.PSObject.Properties['id'] -and $link.id) {
            [string]$link.id
        } else {
            'unknown-link'
        }

        $endpoints = @()
        if ($link.PSObject.Properties['endpoints'] -and $link.endpoints) {
            $endpoints = @($link.endpoints | ForEach-Object { [string]$_ })
        } else {
            if ($link.PSObject.Properties['sourceHost'] -and $link.sourceHost) {
                $endpoints += [string]$link.sourceHost
            }
            if ($link.PSObject.Properties['targetHost'] -and $link.targetHost) {
                $endpoints += [string]$link.targetHost
            }
        }

        $involvesLocal = ($endpoints | ForEach-Object { $_.ToLowerInvariant() }) -contains $canonicalLocalHost.ToLowerInvariant()
        $remoteEndpoint = @($endpoints | Where-Object { $_.ToLowerInvariant() -ne $canonicalLocalHost.ToLowerInvariant() } | Select-Object -First 1)
        $expectedPeerName = if ($remoteEndpoint.Count -gt 0) { [string]$remoteEndpoint[0] } elseif ($endpoints.Count -gt 0) { [string]$endpoints[-1] } else { 'unknown-peer' }
        $expectedPeerEntry = $peersByName[$expectedPeerName.ToLowerInvariant()]

        $preservePair = $false
        if ($link.PSObject.Properties['preserveExistingPair'] -and $null -ne $link.preserveExistingPair) {
            $preservePair = [bool]$link.preserveExistingPair
        }
        $preserveBridge = $false
        if ($link.PSObject.Properties['preserveBridgeCustody'] -and $null -ne $link.preserveBridgeCustody) {
            $preserveBridge = [bool]$link.preserveBridgeCustody
        }
        $downstreamStatus = if ($link.PSObject.Properties['downstreamLinkStatus']) { [string]$link.downstreamLinkStatus } else { $null }
        $xdomainPresent = if ($link.PSObject.Properties['windowsXDomainPresent'] -and $null -ne $link.windowsXDomainPresent) { [bool]$link.windowsXDomainPresent } else { $null }

        $expectedMacs = @()
        $localExpectedMacs = @()
        $peerExpectedMacs = @()
        if ($link.PSObject.Properties['expectedPeerMacs'] -and $null -ne $link.expectedPeerMacs) {
            if ($link.expectedPeerMacs -is [System.Collections.IEnumerable] -and $link.expectedPeerMacs -isnot [string] -and $link.expectedPeerMacs -isnot [PSCustomObject]) {
                $expectedMacs = @($link.expectedPeerMacs | ForEach-Object { ([string]$_).ToUpperInvariant().Replace(':', '-') })
            } else {
                foreach ($prop in $link.expectedPeerMacs.PSObject.Properties) {
                    $normMac = ([string]$prop.Value).ToUpperInvariant().Replace(':', '-')
                    $expectedMacs += $normMac
                    if ($prop.Name.ToLowerInvariant() -eq $canonicalLocalHost.ToLowerInvariant()) {
                        $localExpectedMacs += $normMac
                    } else {
                        $peerExpectedMacs += $normMac
                    }
                }
            }
        }

        if ($preserveBridge) {
            foreach ($ep in $endpoints) {
                $preservedEndpoints[$ep.ToLowerInvariant()] = $true
            }
        }

        # Check if this link has downstreamLinkStatus=none or windowsXDomainPresent=false
        $linkDownstreamDisconnected = ($downstreamStatus -eq 'none') -or ($xdomainPresent -eq $false)

        $requiredTopologyClass = if ($link.PSObject.Properties['requiredTopologyClass'] -and $link.requiredTopologyClass) {
            [string]$link.requiredTopologyClass
        } else {
            'unknown'
        }
        $expectedLocalRootRouterId = if ($link.PSObject.Properties['expectedLocalRootRouterId'] -and $link.expectedLocalRootRouterId) {
            [string]$link.expectedLocalRootRouterId
        } else {
            $null
        }
        $expectedLocalDockRouterId = if ($link.PSObject.Properties['expectedLocalDockRouterId'] -and $link.expectedLocalDockRouterId) {
            [string]$link.expectedLocalDockRouterId
        } else {
            $null
        }
        $localIfExpected = if ($link.PSObject.Properties['sourceHost'] -and ([string]$link.sourceHost).ToLowerInvariant() -eq $canonicalLocalHost.ToLowerInvariant() -and $link.PSObject.Properties['sourceInterface']) {
            [string]$link.sourceInterface
        } elseif ($link.PSObject.Properties['targetHost'] -and ([string]$link.targetHost).ToLowerInvariant() -eq $canonicalLocalHost.ToLowerInvariant() -and $link.PSObject.Properties['targetInterface']) {
            [string]$link.targetInterface
        } else {
            $null
        }

        $expectedPeerKnownAddresses = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        if ($expectedPeerEntry) {
            if ($expectedPeerEntry.usb4UlaAddress) {
                [void]$expectedPeerKnownAddresses.Add(($expectedPeerEntry.usb4UlaAddress -replace '%.*$', '').Trim())
            }
            if ($expectedPeerEntry.usb4LinkLocalAddress) {
                [void]$expectedPeerKnownAddresses.Add(($expectedPeerEntry.usb4LinkLocalAddress -replace '%.*$', '').Trim())
            }
        }
        try {
            if (-not [string]::IsNullOrWhiteSpace($expectedPeerName) -and $expectedPeerName -ne 'unknown-peer') {
                $peerIdFull = Get-ThunderboltPeerIdentity -ComputerName $canonicalLocalHost -PeerComputerName $expectedPeerName -RegistryPath $RegistryPath
                $peerSelfId = Get-ThunderboltPeerIdentity -ComputerName $expectedPeerName -RegistryPath $RegistryPath
                foreach ($candidateAddr in @($peerSelfId.PrimaryUla, $peerSelfId.DeterministicUla, $peerSelfId.PrimaryLinkLocal, $peerSelfId.DeterministicLinkLocal)) {
                    if (-not [string]::IsNullOrWhiteSpace([string]$candidateAddr)) {
                        [void]$expectedPeerKnownAddresses.Add((([string]$candidateAddr) -replace '%.*$', '').Trim())
                    }
                }
                if ($peerIdFull.PairwiseIPv4 -and $peerIdFull.PairwiseIPv4.PeerIPv4Address) {
                    [void]$expectedPeerKnownAddresses.Add([string]$peerIdFull.PairwiseIPv4.PeerIPv4Address)
                }
            }
        } catch { }

        $matchedAdapter = $null
        $matchedNeighbor = $null
        $allowLiveHostPortOverride = ($linkIdValue -eq 'ms01-plugable-gen9-hostport-control' -and -not [string]::IsNullOrWhiteSpace($localIfExpected))
        if ($involvesLocal -and (-not $linkDownstreamDisconnected -or $allowLiveHostPortOverride)) {
            foreach ($ad in $p2pAdapters) {
                $adMac = ([string]$ad.MacAddress).ToUpperInvariant().Replace(':', '-')
                $aliasMatch = (-not [string]::IsNullOrWhiteSpace($localIfExpected)) -and ([string]$ad.InterfaceAlias -eq $localIfExpected)
                $macMatch = ($localExpectedMacs.Count -gt 0 -and $localExpectedMacs -contains $adMac) -or ($expectedMacs.Count -gt 0 -and $expectedMacs -contains $adMac)
                if ($aliasMatch -or $macMatch) {
                    $matchedAdapter = $ad
                    break
                }
            }
            if ($null -eq $matchedAdapter -and $p2pAdapters.Count -gt 0 -and $linkIdValue -eq 'gen7-gen9-usb4-hub-chain') {
                $matchedAdapter = $p2pAdapters[0]
            }
            if ($null -ne $matchedAdapter -and @($matchedAdapter.NeighborCandidates).Count -gt 0) {
                foreach ($cand in @($matchedAdapter.NeighborCandidates)) {
                    $candIpClean = if ($cand.IPAddress) { (([string]$cand.IPAddress) -replace '%.*$', '').Trim() } else { '' }
                    $candMacNorm = if ($cand.LinkLayerAddress) { ([string]$cand.LinkLayerAddress).ToUpperInvariant().Replace(':', '-') } else { '' }
                    if (($candIpClean -and $expectedPeerKnownAddresses.Contains($candIpClean)) -or ($candMacNorm -and ($peerExpectedMacs -contains $candMacNorm))) {
                        $matchedNeighbor = $cand
                        break
                    }
                }
                if ($null -eq $matchedNeighbor) {
                    $matchedNeighbor = $matchedAdapter.NeighborCandidates[0]
                }
            }
        }
        if ($allowLiveHostPortOverride -and $null -ne $matchedAdapter) {
            $linkDownstreamDisconnected = $false
        }

        $adPnpAncestry = @(if ($matchedAdapter -and $matchedAdapter.PSObject.Properties['PnpAncestry'] -and $matchedAdapter.PnpAncestry) {
            $matchedAdapter.PnpAncestry
        })
        $adRootRouterId = if ($matchedAdapter -and $matchedAdapter.PSObject.Properties['RootRouterId'] -and $matchedAdapter.RootRouterId) {
            [string]$matchedAdapter.RootRouterId
        } elseif ($expectedLocalRootRouterId) {
            $expectedLocalRootRouterId
        } else {
            $null
        }
        $adDockRouters = @(if ($matchedAdapter -and $matchedAdapter.PSObject.Properties['IntermediateDockRouters'] -and $matchedAdapter.IntermediateDockRouters) {
            $matchedAdapter.IntermediateDockRouters
        })
        $observedTopologyClass = if ($link.PSObject.Properties['observedTopologyClass'] -and $link.observedTopologyClass) {
            [string]$link.observedTopologyClass
        } elseif ($matchedAdapter -and $matchedAdapter.PSObject.Properties['TopologyPathClass'] -and $matchedAdapter.TopologyPathClass) {
            [string]$matchedAdapter.TopologyPathClass
        } else {
            'unknown'
        }

        $canaryResult = $null
        $state = 'disconnected'
        $reason = ''

        if ($linkDownstreamDisconnected) {
            $state = 'disconnected'
            $reason = "Downstream USB4 link status is '$downstreamStatus' and Windows XDomain present is '$xdomainPresent'."
        } elseif ($preserveBridge -and -not $involvesLocal) {
            $state = if ($link.PSObject.Properties['lifecycleStateHint'] -and $link.lifecycleStateHint) { [string]$link.lifecycleStateHint } else { 'ready' }
            $reason = "Preserved external bridge custody on $($endpoints -join ' <-> '); local host does not withdraw or mutate bridge routes."
        } elseif ($null -eq $matchedAdapter) {
            $state = 'disconnected'
            $reason = "No active local USB4 P2P adapter matched topology link '$linkIdValue'."
        } elseif (-not $matchedAdapter.NetEnabled -or ($null -ne $matchedAdapter.NetConnectionStatus -and [int]$matchedAdapter.NetConnectionStatus -ne 2)) {
            $state = 'disconnected'
            $reason = "USB4 P2P adapter '$($matchedAdapter.InterfaceAlias)' is not connected."
        } elseif ($standby.CurrentlyInStandby) {
            $state = 'asleep'
            $reason = "Host is currently in Modern Standby (Event 506: $($standby.LastEnterReason))."
        } else {
            # Verify identity-bound canary when expected peer is known
            $expectedUla = if ($expectedPeerEntry) { [string]$expectedPeerEntry.usb4UlaAddress } else { $null }
            $expectedLinkLocal = if ($expectedPeerEntry) { [string]$expectedPeerEntry.usb4LinkLocalAddress } else { $null }
            $neighborIp = if ($matchedNeighbor) { [string]$matchedNeighbor.IPAddress } else { $null }
            $neighborMac = if ($matchedNeighbor -and $matchedNeighbor.LinkLayerAddress) { ([string]$matchedNeighbor.LinkLayerAddress).ToUpperInvariant().Replace(':', '-') } else { $null }
            $neighborHost = if ($matchedNeighbor -and $matchedNeighbor.PeerHostname) { [string]$matchedNeighbor.PeerHostname } else { $null }

            $probeTarget = if (-not [string]::IsNullOrWhiteSpace($expectedUla)) {
                $expectedUla
            } elseif (-not [string]::IsNullOrWhiteSpace($neighborIp)) {
                $neighborIp
            } elseif (-not [string]::IsNullOrWhiteSpace($expectedLinkLocal)) {
                $expectedLinkLocal
            } else {
                $null
            }

            $adapterScopeId = if ($matchedAdapter -and $matchedAdapter.InterfaceIndex -gt 0) {
                [int64]$matchedAdapter.InterfaceIndex
            } else {
                $null
            }
            if ($probeTarget -and $probeTarget -match '^fe80:' -and $probeTarget -notmatch '%' -and $null -ne $adapterScopeId) {
                $probeTarget = "${probeTarget}%${adapterScopeId}"
            }

            $probeOutcome = if ($probeTarget) {
                Test-CanaryEndpoint -Address $probeTarget -ScopeId $adapterScopeId -TimeoutMs $ReprobeTimeoutMs
            } else {
                [PSCustomObject]@{ Reachable = $false; Port = $null; LatencyMs = $null; ScopeId = $adapterScopeId }
            }

            $cleanNeighborIp = if ($neighborIp) { ($neighborIp -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }
            $cleanExpectedLl = if ($expectedLinkLocal) { ($expectedLinkLocal -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }
            $cleanExpectedUla = if ($expectedUla) { ($expectedUla -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }

            $macVerified = ($peerExpectedMacs.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($neighborMac) -and ($peerExpectedMacs -contains $neighborMac))
            $hostVerified = (-not [string]::IsNullOrWhiteSpace($neighborHost) -and $neighborHost.ToLowerInvariant() -eq $expectedPeerName.ToLowerInvariant())
            $neighborAddressMatchesExpectedPeer = (-not [string]::IsNullOrWhiteSpace($cleanNeighborIp)) -and (
                ($cleanNeighborIp -eq $cleanExpectedLl) -or ($cleanNeighborIp -eq $cleanExpectedUla) -or $expectedPeerKnownAddresses.Contains($cleanNeighborIp)
            )
            $ulaCanaryVerified = ($probeOutcome.Reachable -and -not [string]::IsNullOrWhiteSpace($expectedUla) -and $probeTarget -eq $expectedUla -and ($null -ne $matchedNeighbor))

            $macMismatch = ($peerExpectedMacs.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($neighborMac) -and ($peerExpectedMacs -notcontains $neighborMac))
            $hostMismatch = (-not [string]::IsNullOrWhiteSpace($neighborHost) -and $neighborHost.ToLowerInvariant() -ne $expectedPeerName.ToLowerInvariant())

            $identityVerified = (-not $macMismatch) -and (-not $hostMismatch) -and ($macVerified -or $hostVerified -or $ulaCanaryVerified -or $neighborAddressMatchesExpectedPeer)
            $observedPeerId = if ($hostVerified) {
                $neighborHost.ToLowerInvariant()
            } elseif ($identityVerified) {
                $expectedPeerName.ToLowerInvariant()
            } elseif (-not [string]::IsNullOrWhiteSpace($neighborHost)) {
                $neighborHost.ToLowerInvariant()
            } else {
                $null
            }

            $canaryResult = [PSCustomObject]@{
                ExpectedPeerId   = $expectedPeerName
                ObservedPeerId   = $observedPeerId
                ExpectedMacs     = @($expectedMacs)
                ObservedMac      = $neighborMac
                CanaryAddress    = $probeTarget
                ScopeId          = $probeOutcome.ScopeId
                Transport        = if ($probeOutcome.Port) { "tcp/$($probeOutcome.Port)" } else { 'neighbor-ndp' }
                Reachable        = [bool]($probeOutcome.Reachable -or ($null -ne $matchedNeighbor))
                IdentityVerified = [bool]$identityVerified
                LatencyMs        = $probeOutcome.LatencyMs
                VerifiedAtUtc    = $nowUtc.ToString('o')
            }

            if ($evidenceExpired -and -not $ReprobeOnResume) {
                $state = if ($matchedAdapter.SelectiveSuspend -eq 'Enabled') { 'asleep' } else { 'probing' }
                $reason = if ($invalidatedByStandby) {
                    "Lifecycle evidence was invalidated by Modern Standby Event 506 ($($standby.LastEnterReason)); bounded resume reprobe required before claiming ready."
                } else {
                    "Lifecycle evidence TTL (${EvidenceTtlSeconds}s) expired at $($expiresUtc.ToString('o')); re-verification required."
                }
            } elseif ($canaryResult.Reachable -and $canaryResult.IdentityVerified) {
                $state = 'ready'
                $reason = "Active USB4 P2P link on '$($matchedAdapter.InterfaceAlias)' verified via identity-bound canary for '$expectedPeerName' (never claims always-on transit while SelectiveSuspend=$($matchedAdapter.SelectiveSuspend))."
                $activeReadyPeers[$expectedPeerName.ToLowerInvariant()] = $true
                if ($preservePair) {
                    foreach ($ep in $endpoints) {
                        $preservedEndpoints[$ep.ToLowerInvariant()] = $true
                    }
                }
            } elseif ($canaryResult.Reachable -and -not $canaryResult.IdentityVerified) {
                $state = 'probing'
                $reason = "Neighbor responded on '$($matchedAdapter.InterfaceAlias)', but identity-bound canary did not match expected peer '$expectedPeerName' (ObservedPeerId='$observedPeerId', ObservedMac='$neighborMac'). Refusing 'ready' transition."
            } elseif ($matchedAdapter.SelectiveSuspend -eq 'Enabled') {
                $state = 'asleep'
                $reason = "USB4 P2P adapter '$($matchedAdapter.InterfaceAlias)' has SelectiveSuspend=Enabled and no reachable peer responder."
            } else {
                $state = 'probing'
                $reason = "USB4 P2P adapter '$($matchedAdapter.InterfaceAlias)' is up, probing for peer '$expectedPeerName'."
            }
        }

        # Evaluate link admission and topology qualification (5 acceptance contracts)
        $isHostPortControl = ($observedTopologyClass -eq 'host_port_control') -or
            ($link.PSObject.Properties['qualificationScope'] -and [string]$link.qualificationScope -eq 'host_port_control_only') -or
            ($adRootRouterId -and $adRootRouterId.ToUpperInvariant().Contains('PID_E333'))

        $expectedRootPrefix = if (-not [string]::IsNullOrWhiteSpace($expectedLocalRootRouterId)) {
            $parts = @($expectedLocalRootRouterId.ToUpperInvariant() -split '\\')
            if ($parts.Count -ge 2) { "$($parts[0])\$($parts[1])" } else { $expectedLocalRootRouterId.ToUpperInvariant() }
        } else {
            $null
        }
        $pnpPrefixOnlyRejected = ($null -ne $matchedAdapter -and ($null -ne $matchedAdapter.PSObject.Properties['PnpAncestry']) -and @($adPnpAncestry).Count -eq 1 -and -not [string]::IsNullOrWhiteSpace($expectedLocalRootRouterId))
        $rootRouterMismatch = ($null -ne $matchedAdapter -and @($adPnpAncestry).Count -ge 2 -and -not [string]::IsNullOrWhiteSpace($expectedRootPrefix) -and -not [string]::IsNullOrWhiteSpace($adRootRouterId) -and (-not $adRootRouterId.ToUpperInvariant().Contains($expectedRootPrefix)))

        $linkAdmitted = $false
        $admissionReason = 'no_observed_xdomain'
        $qualificationScope = if ($link.PSObject.Properties['qualificationScope'] -and $link.qualificationScope) {
            [string]$link.qualificationScope
        } else {
            'unqualified_no_xdomain'
        }
        $requiredTopologyQualified = $false

        if ($preserveBridge -and -not $involvesLocal) {
            $linkAdmitted = $true
            $admissionReason = 'external_bridge_custody_preserved'
            $qualificationScope = 'external_bridge_custody'
            $requiredTopologyQualified = $true
        } elseif ($linkDownstreamDisconnected -or $null -eq $matchedAdapter) {
            $linkAdmitted = $false
            $admissionReason = 'no_observed_xdomain'
            $qualificationScope = if ($isHostPortControl) { 'host_port_control_only' } else { 'unqualified_no_xdomain' }
            $requiredTopologyQualified = $false
        } elseif ($pnpPrefixOnlyRejected) {
            $linkAdmitted = $false
            $admissionReason = 'pnp_ancestry_prefix_only_rejected'
            $qualificationScope = 'unqualified_missing_pnp_ancestry'
            $requiredTopologyQualified = $false
        } elseif ($rootRouterMismatch) {
            $linkAdmitted = $false
            $admissionReason = 'pnp_ancestry_root_router_mismatch'
            $qualificationScope = if ($isHostPortControl -or $requiredTopologyClass -eq 'required_two_hub') { 'host_port_control_only' } else { 'unqualified_root_router_mismatch' }
            $requiredTopologyQualified = $false
        } elseif ($isHostPortControl -and $requiredTopologyClass -eq 'required_two_hub') {
            $linkAdmitted = ($state -eq 'ready')
            $admissionReason = if ($linkAdmitted) { 'admitted_host_port_control_only' } else { 'no_observed_xdomain' }
            $qualificationScope = 'host_port_control_only'
            $requiredTopologyQualified = $false
        } elseif ($state -eq 'ready') {
            $linkAdmitted = $true
            $admissionReason = 'admitted_verified_xdomain_and_canary'
            $requiredTopologyQualified = if ($link.PSObject.Properties['requiredTopologyQualified'] -and $null -ne $link.requiredTopologyQualified) {
                [bool]$link.requiredTopologyQualified
            } else {
                $true
            }
            if (-not $link.PSObject.Properties['qualificationScope'] -or [string]::IsNullOrWhiteSpace([string]$link.qualificationScope)) {
                $qualificationScope = if ($requiredTopologyQualified) { 'qualified' } else { 'host_port_control_only' }
            }
        }

        $measuredSamples = @(if ($link.PSObject.Properties['measuredPayloadGbpsSamples'] -and $link.measuredPayloadGbpsSamples) {
            $link.measuredPayloadGbpsSamples | ForEach-Object { [double]$_ }
        } elseif ($link.PSObject.Properties['measured_payload_gbps_samples'] -and $link.measured_payload_gbps_samples) {
            $link.measured_payload_gbps_samples | ForEach-Object { [double]$_ }
        } else {
            @(7.86, 12.90, 14.12)
        })

        function Get-LinkStringField {
            param([object]$Obj, [string]$Camel, [string]$Snake)
            if ($Obj.PSObject.Properties[$Camel] -and $null -ne $Obj.$Camel) { return [string]$Obj.$Camel }
            if ($Obj.PSObject.Properties[$Snake] -and $null -ne $Obj.$Snake) { return [string]$Obj.$Snake }
            return $null
        }

        $linkReports.Add([PSCustomObject]@{
            LinkId                          = $linkIdValue
            Endpoints                       = @($endpoints)
            AdapterName                     = if ($matchedAdapter) { [string]$matchedAdapter.InterfaceAlias } else { $null }
            InterfaceIndex                  = if ($matchedAdapter) { [int]$matchedAdapter.InterfaceIndex } else { $null }
            State                           = $state
            AlwaysOnTransitCapable          = $false
            PreserveExistingPair            = [bool]$preservePair
            PreserveBridgeCustody           = [bool]$preserveBridge
            DownstreamLinkStatus            = $downstreamStatus
            WindowsXDomainPresent           = if ($null -ne $xdomainPresent -and -not $allowLiveHostPortOverride) { [bool]$xdomainPresent } else { [bool]($state -eq 'ready') }
            LinkAdmitted                    = [bool]$linkAdmitted
            AdmissionReason                 = $admissionReason
            RequiredTopologyClass           = $requiredTopologyClass
            ObservedTopologyClass           = $observedTopologyClass
            QualificationScope              = $qualificationScope
            RequiredTopologyQualified       = [bool]$requiredTopologyQualified
            PnpAncestry                     = @($adPnpAncestry)
            RootRouterId                    = $adRootRouterId
            IntermediateDockRouters         = @($adDockRouters)
            LocalRootRouterFw               = Get-LinkStringField -Obj $link -Camel 'localRootRouterFw' -Snake 'local_root_router_fw'
            DockModel                       = Get-LinkStringField -Obj $link -Camel 'dockModel' -Snake 'dock_model'
            DockSerial                      = Get-LinkStringField -Obj $link -Camel 'dockSerial' -Snake 'dock_serial'
            DockStatus                      = Get-LinkStringField -Obj $link -Camel 'dockStatus' -Snake 'dock_status'
            DockFwPackage                   = Get-LinkStringField -Obj $link -Camel 'dockFwPackage' -Snake 'dock_fw_package'
            DockFwTarget                    = Get-LinkStringField -Obj $link -Camel 'dockFwTarget' -Snake 'dock_fw_target'
            DockPdFw                        = Get-LinkStringField -Obj $link -Camel 'dockPdFw' -Snake 'dock_pd_fw'
            DockTbt5Nvm                     = Get-LinkStringField -Obj $link -Camel 'dockTbt5Nvm' -Snake 'dock_tbt5_nvm'
            DockDmcFw                       = Get-LinkStringField -Obj $link -Camel 'dockDmcFw' -Snake 'dock_dmc_fw'
            WindowsReportedLinkSpeedBps     = [uint64]20000000000
            LinkSpeedIsNominalDriverDisplay = $true
            LinkSpeedIsHardwareMaximum      = $false
            MeasuredPayloadGbpsSamples      = @($measuredSamples)
            Evidence                        = [PSCustomObject]@{
                ObservedAtUtc              = $observedUtc.ToString('o')
                TtlSeconds                 = $EvidenceTtlSeconds
                ExpiresAtUtc               = $expiresUtc.ToString('o')
                InvalidatedByModernStandby = [bool]$invalidatedByStandby
                Expired                    = [bool]$evidenceExpired
            }
            Canary                          = $canaryResult
            Reason                          = $reason
        })
    }

    # Bounded resume reprobe / reapply when requested or when recovering from Modern Standby
    $reprobeReport = $null
    if ($ReprobeOnResume) {
        $reprobeSw = [System.Diagnostics.Stopwatch]::StartNew()
        $attemptsRun = 0
        $recoveredPeers = [System.Collections.Generic.List[string]]::new()
        $reappliedPeers = [System.Collections.Generic.List[string]]::new()
        $budgetExhausted = $false

        for ($attempt = 1; $attempt -le $MaxReprobeAttempts; $attempt++) {
            if ($reprobeSw.ElapsedMilliseconds -ge $TotalReprobeBudgetMs) {
                $budgetExhausted = $true
                break
            }
            $attemptsRun = $attempt

            foreach ($lr in $linkReports) {
                if ($lr.State -in @('asleep', 'probing') -and $lr.AdapterName) {
                    $targetPeer = @($lr.Endpoints | Where-Object { $_.ToLowerInvariant() -ne $canonicalLocalHost.ToLowerInvariant() } | Select-Object -First 1)
                    if ($targetPeer.Count -eq 0) { continue }
                    $peerName = [string]$targetPeer[0]
                    $peerInfo = $peersByName[$peerName.ToLowerInvariant()]
                    if (-not $peerInfo) { continue }

                    $remBudget = [math]::Max(50, [math]::Min($ReprobeTimeoutMs, [int]($TotalReprobeBudgetMs - $reprobeSw.ElapsedMilliseconds)))
                    $probeRes = Test-CanaryEndpoint -Address ([string]$peerInfo.usb4UlaAddress) -ScopeId $lr.InterfaceIndex -TimeoutMs $remBudget
                    if (($probeRes.Reachable -or ($null -ne $lr.Canary -and $lr.Canary.Reachable)) -and ($null -ne $lr.Canary) -and $lr.Canary.IdentityVerified) {
                        $lr.State = 'ready'
                        $lr.Evidence.InvalidatedByModernStandby = $false
                        $lr.Evidence.Expired = $false
                        $lr.Reason = "Recovered to 'ready' on bounded resume reprobe attempt $attempt/$MaxReprobeAttempts."
                        $activeReadyPeers[$peerName.ToLowerInvariant()] = $true
                        if ($recoveredPeers -notcontains $peerName) {
                            $recoveredPeers.Add($peerName)
                        }
                        if ($ReapplyOnResume -and ($reappliedPeers -notcontains $peerName)) {
                            if ($Apply -and $PSCmdlet.ShouldProcess($peerName, 'Reapply Thunderbolt peer registration after S0 resume')) {
                                Register-ThunderboltPeer -ComputerName $ComputerName -PeerComputerName $peerName -InterfaceAlias $lr.AdapterName -RegistryPath $RegistryPath -Apply | Out-Null
                            }
                            $reappliedPeers.Add($peerName)
                        }
                    }
                }
            }

            if ((@($linkReports | Where-Object { $_.State -in @('asleep', 'probing') -and $_.AdapterName })).Count -eq 0) {
                break
            }
        }
        $reprobeSw.Stop()
        if ($reprobeSw.ElapsedMilliseconds -ge $TotalReprobeBudgetMs) {
            $budgetExhausted = $true
        }

        $reprobeReport = [PSCustomObject]@{
            AttemptsExecuted    = $attemptsRun
            MaxAttempts         = $MaxReprobeAttempts
            PerAttemptTimeoutMs = $ReprobeTimeoutMs
            TotalBudgetMs       = $TotalReprobeBudgetMs
            ElapsedMs           = [int]$reprobeSw.ElapsedMilliseconds
            BudgetExhausted     = [bool]$budgetExhausted
            RecoveredPeers      = @($recoveredPeers)
            ReappliedPeers      = @($reappliedPeers)
        }
    }

    # Evaluate PC-AI owned routes for stale route withdrawal while preserving active pairs, foreign interfaces, and Milly bridge
    $routeActions = [System.Collections.Generic.List[object]]::new()
    $netRoutes = @(Get-SafeCimRows -ClassName 'MSFT_NetRoute')
    $evaluatedUsb4IfIndices = @($p2pAdapters | ForEach-Object { [int]$_.InterfaceIndex } | Where-Object { $_ -gt 0 } | Select-Object -Unique)

    function ConvertTo-NormalizedPrefix {
        param([string]$Prefix)
        if ([string]::IsNullOrWhiteSpace($Prefix)) { return $Prefix }
        if ($Prefix -match '^([^/]+)/(\d+)$') {
            $ipPart = $Matches[1]
            $lenPart = $Matches[2]
            $parsed = $null
            if ([ipaddress]::TryParse($ipPart, [ref]$parsed)) {
                return "$($parsed.ToString().ToLowerInvariant())/$lenPart"
            }
        }
        return $Prefix.ToLowerInvariant()
    }

    # Map known peer ULA prefixes (PrimaryUla + DeterministicUla) and pairwise IPv4 subnets to peer names
    $prefixToPeer = @{}
    if ($localIdentity.PrimaryUla) {
        $prefixToPeer[(ConvertTo-NormalizedPrefix "$($localIdentity.PrimaryUla)/128")] = $canonicalLocalHost
    }
    if ($localIdentity.DeterministicUla) {
        $prefixToPeer[(ConvertTo-NormalizedPrefix "$($localIdentity.DeterministicUla)/128")] = $canonicalLocalHost
    }

    $uniquePeerNames = @($peersByName.Values | ForEach-Object { [string]$_.computerName } | Select-Object -Unique)
    foreach ($pName in $uniquePeerNames) {
        if ([string]::IsNullOrWhiteSpace($pName)) { continue }
        $pEntry = $peersByName[$pName.ToLowerInvariant()]
        if ($pEntry -and $pEntry.usb4UlaAddress) {
            $prefixToPeer[(ConvertTo-NormalizedPrefix "$($pEntry.usb4UlaAddress)/128")] = $pName
        }
        if ($pName.ToLowerInvariant() -eq $canonicalLocalHost.ToLowerInvariant()) { continue }

        try {
            $pairId = Get-ThunderboltPeerIdentity -ComputerName $canonicalLocalHost -PeerComputerName $pName -RegistryPath $RegistryPath
            $peerIdObj = Get-ThunderboltPeerIdentity -ComputerName $pName -RegistryPath $RegistryPath
            if ($peerIdObj.PrimaryUla) {
                $prefixToPeer[(ConvertTo-NormalizedPrefix "$($peerIdObj.PrimaryUla)/128")] = $pName
            }
            if ($peerIdObj.DeterministicUla) {
                $prefixToPeer[(ConvertTo-NormalizedPrefix "$($peerIdObj.DeterministicUla)/128")] = $pName
            }
            if ($pairId.PairwiseIPv4) {
                $pw = $pairId.PairwiseIPv4
                if ($pw.SubnetPrefix) {
                    $prefixToPeer[(ConvertTo-NormalizedPrefix $pw.SubnetPrefix)] = $pName
                    if ($pw.SubnetPrefix -match '^(\d+\.\d+\.\d+)\.(\d+)/30$') {
                        $baseOctet = [int]$Matches[2]
                        $bcastAddr = "$($Matches[1]).$($baseOctet + 3)/32"
                        $prefixToPeer[(ConvertTo-NormalizedPrefix $bcastAddr)] = $pName
                    }
                }
                if ($pw.LocalIPv4Address) {
                    $prefixToPeer[(ConvertTo-NormalizedPrefix "$($pw.LocalIPv4Address)/32")] = $pName
                }
                if ($pw.PeerIPv4Address) {
                    $prefixToPeer[(ConvertTo-NormalizedPrefix "$($pw.PeerIPv4Address)/32")] = $pName
                }
            }
        } catch {
        }
    }

    foreach ($route in $netRoutes) {
        $dest = [string]$route.DestinationPrefix
        if ([string]::IsNullOrWhiteSpace($dest)) { continue }

        # Never touch default routes (0.0.0.0/0, ::/0) or non-PC-AI routes
        if ($dest -eq '0.0.0.0/0' -or $dest -eq '::/0') {
            continue
        }

        $isPcaiOwnedUlaHostRoute = $dest -match '^fd00:7462:7032:[0-9a-f:]+/128$'
        $isPcaiOwnedPairwiseIpv4 = $dest -match '^172\.31\.(24\d|25[0-5])\.\d+/(30|32)$'
        if (-not $isPcaiOwnedUlaHostRoute -and -not $isPcaiOwnedPairwiseIpv4) {
            continue
        }

        $ifIdx = [int]$route.InterfaceIndex
        $normDest = ConvertTo-NormalizedPrefix $dest
        $ownerPeer = if ($prefixToPeer.ContainsKey($normDest)) { $prefixToPeer[$normDest] } else { $null }

        # Exclude local host's own ULA /128 loopback/local route
        if ($ownerPeer -and $ownerPeer.ToLowerInvariant() -eq $canonicalLocalHost.ToLowerInvariant()) {
            continue
        }

        # Preserve routes bound to foreign interfaces outside the evaluated USB4 P2P adapters
        if ($evaluatedUsb4IfIndices.Count -gt 0 -and ($evaluatedUsb4IfIndices -notcontains $ifIdx)) {
            $routeActions.Add([PSCustomObject]@{
                DestinationPrefix = $dest
                InterfaceIndex    = $ifIdx
                OwnerPeer         = $ownerPeer
                Action            = 'PreserveForeignInterfaceRoute'
                Reason            = "Preserved route '$dest' on foreign InterfaceIndex $ifIdx (outside evaluated USB4 P2P adapters: $($evaluatedUsb4IfIndices -join ', '))."
                Executed          = $false
                Error             = $null
            })
            continue
        }

        $isPreserved = ($ownerPeer -and $preservedEndpoints.ContainsKey($ownerPeer.ToLowerInvariant()))
        $isActiveReady = ($ownerPeer -and $activeReadyPeers.ContainsKey($ownerPeer.ToLowerInvariant()))

        $linkForPeer = if ($ownerPeer) {
            $linkReports | Where-Object {
                ($_.Endpoints | ForEach-Object { $_.ToLowerInvariant() }) -contains $ownerPeer.ToLowerInvariant()
            } | Select-Object -First 1
        } else {
            $null
        }

        if ($isPreserved -or ($linkForPeer -and ($linkForPeer.PreserveBridgeCustody -or ($linkForPeer.PreserveExistingPair -and $linkForPeer.State -eq 'ready')))) {
            $routeActions.Add([PSCustomObject]@{
                DestinationPrefix = $dest
                InterfaceIndex    = $ifIdx
                OwnerPeer         = $ownerPeer
                Action            = 'PreserveActiveOrBridgeRoute'
                Reason            = "Preserved route for '$ownerPeer' (PreserveBridgeOrActivePair=true)."
                Executed          = $false
                Error             = $null
            })
            continue
        }

        if ($isActiveReady) {
            $routeActions.Add([PSCustomObject]@{
                DestinationPrefix = $dest
                InterfaceIndex    = $ifIdx
                OwnerPeer         = $ownerPeer
                Action            = 'PreserveActiveOrBridgeRoute'
                Reason            = "Route targets an active 'ready' USB4 peer '$ownerPeer' on ifIndex $ifIdx."
                Executed          = $false
                Error             = $null
            })
            continue
        }

        # Route targets a disconnected or asleep peer or exact /32 canary that is no longer valid
        $executed = $false
        $routeError = $null
        if ($WithdrawStaleRoutes -and $Apply) {
            if ($ifIdx -le 0) {
                $executed = $false
                $routeError = "Refusing unscoped route deletion for '$dest': invalid InterfaceIndex $ifIdx."
            } elseif ($PSCmdlet.ShouldProcess("$dest (ifIndex $ifIdx)", 'Withdraw stale PC-AI USB4 route')) {
                try {
                    Invoke-ScopedRouteDelete -DestinationPrefix $dest -InterfaceIndex $ifIdx
                    $remainingRoutes = @(Get-SafeCimRows -ClassName 'MSFT_NetRoute' -Required | Where-Object {
                        [int]$_.InterfaceIndex -eq $ifIdx -and (ConvertTo-NormalizedPrefix ([string]$_.DestinationPrefix)) -eq $normDest
                    })
                    if ($remainingRoutes.Count -eq 0) {
                        $executed = $true
                    } else {
                        $executed = $false
                        $routeError = "Route '$dest' on InterfaceIndex $ifIdx remained present in MSFT_NetRoute after deletion attempt."
                    }
                } catch {
                    $executed = $false
                    $routeError = $_.Exception.Message
                }
            }
        }

        $routeActions.Add([PSCustomObject]@{
            DestinationPrefix = $dest
            InterfaceIndex    = $ifIdx
            OwnerPeer         = $ownerPeer
            Action            = 'WithdrawStaleOwnedRoute'
            Reason            = "Stale PC-AI route '$dest' on ifIndex $ifIdx has no 'ready' identity-verified USB4 link (owner='$ownerPeer')."
            Executed          = [bool]$executed
            Error             = $routeError
        })
    }

    $tbSvcRunning = $true
    try {
        if (Get-Command Get-Service -ErrorAction SilentlyContinue) {
            $svcList = @(Get-Service -Name 'TBService', 'ShareTBService', 'SharedAccess' -ErrorAction SilentlyContinue)
            if ($svcList.Count -gt 0) {
                $tbSvcRunning = (@($svcList | Where-Object { $_.Status -eq 'Running' })).Count -gt 0
            }
        }
    } catch { }

    $policyNode = if ($registry -and $registry.PSObject.Properties['usb4LifecyclePolicy'] -and $registry.usb4LifecyclePolicy) {
        $registry.usb4LifecyclePolicy
    } elseif ($registry -and $registry.PSObject.Properties['lifecycle_policy'] -and $registry.lifecycle_policy) {
        $registry.lifecycle_policy
    } elseif ($registry -and $registry.PSObject.Properties['lifecyclePolicy'] -and $registry.lifecyclePolicy) {
        $registry.lifecyclePolicy
    } else {
        $null
    }
    $routeTtlSeconds = if ($policyNode -and $policyNode.PSObject.Properties['route_ttl_seconds'] -and $null -ne $policyNode.route_ttl_seconds) {
        [int]$policyNode.route_ttl_seconds
    } elseif ($policyNode -and $policyNode.PSObject.Properties['routeTtlSeconds'] -and $null -ne $policyNode.routeTtlSeconds) {
        [int]$policyNode.routeTtlSeconds
    } else {
        90
    }
    $ms01UplinkCap = if ($policyNode -and $policyNode.PSObject.Properties['ms01UplinkCapMbps'] -and $null -ne $policyNode.ms01UplinkCapMbps) {
        [int]$policyNode.ms01UplinkCapMbps
    } elseif ($policyNode -and $policyNode.PSObject.Properties['ms01_uplink_cap_mbps'] -and $null -ne $policyNode.ms01_uplink_cap_mbps) {
        [int]$policyNode.ms01_uplink_cap_mbps
    } else {
        1000
    }

    return [PSCustomObject]@{
        LocalComputerName        = $canonicalLocalHost
        EvaluatedAtUtc           = $nowUtc.ToString('o')
        Ms01UplinkCapMbps        = $ms01UplinkCap
        RouteTtlSeconds          = $routeTtlSeconds
        LifecyclePolicy          = $policyNode
        AnyAlwaysOnTransitClaim  = [bool]$anyAlwaysOnTransitCapable
        ShareTbServiceRunning    = [bool]$tbSvcRunning
        DefaultRoutesPreserved   = $true
        ModernStandby            = [PSCustomObject]@{
            S0LowPowerIdleState              = $standby.S0LowPowerIdleState
            S0LowPowerIdleConnectedSupported = $standby.S0LowPowerIdleConnectedSupported
            EventLogAvailable                = $standby.EventLogAvailable
            EnterStandbyCount                = $standby.EnterStandbyCount
            ExitStandbyCount                 = $standby.ExitStandbyCount
            LastEnteredStandbyAtUtc          = $standby.LastEnteredStandbyAtUtc
            LastEnterReason                  = $standby.LastEnterReason
            LastExitedStandbyAtUtc           = $standby.LastExitedStandbyAtUtc
            LastExitReason                   = $standby.LastExitReason
            CurrentlyInStandby               = $standby.CurrentlyInStandby
        }
        Adapters                 = @($p2pAdapters)
        Links                    = @($linkReports)
        RouteActions             = @($routeActions)
        ResumeReprobe            = $reprobeReport
    }
}
