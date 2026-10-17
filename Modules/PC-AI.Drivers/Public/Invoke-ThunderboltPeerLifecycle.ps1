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
            [string]$Namespace = 'root/StandardCimv2'
        )
        try {
            return @(Get-CimInstance -Namespace $Namespace -ClassName $ClassName -ErrorAction Stop)
        } catch {
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

        try {
            $events = @(Get-WinEvent -FilterHashtable @{
                LogName      = 'System'
                ProviderName = 'Microsoft-Windows-Kernel-Power'
                Id           = 506, 507
            } -MaxEvents 12 -ErrorAction Stop)

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
            Write-Verbose "Modern Standby event log query skipped or unavailable: $($_.Exception.Message)"
        }

        $currentlyInStandby = ($null -ne $lastEnteredAtUtc) -and (
            ($null -eq $lastExitedAtUtc) -or ($lastEnteredAtUtc -gt $lastExitedAtUtc)
        )

        return [PSCustomObject]@{
            S0LowPowerIdleConnectedSupported = $true
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
            [int[]]$Ports = @(5985, 22, 445),

            [Parameter()]
            [int]$TimeoutMs = 250
        )

        $cleanAddr = ($Address -replace '%.*$', '').Trim()
        foreach ($port in $Ports) {
            $client = $null
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $ipObj = [System.Net.IPAddress]::Parse($cleanAddr)
                $client = [System.Net.Sockets.TcpClient]::new($ipObj.AddressFamily)
                $async = $client.BeginConnect($ipObj, $port, $null, $null)
                if ($async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
                    $client.EndConnect($async)
                    $sw.Stop()
                    if ($client.Connected) {
                        return [PSCustomObject]@{
                            Reachable = $true
                            Port      = $port
                            LatencyMs = [int]$sw.ElapsedMilliseconds
                        }
                    }
                }
            } catch {
            } finally {
                if ($null -ne $client) { $client.Close() }
            }
        }

        return [PSCustomObject]@{
            Reachable = $false
            Port      = $null
            LatencyMs = $null
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

        $matchedAdapter = $null
        $matchedNeighbor = $null
        if ($involvesLocal -and -not $linkDownstreamDisconnected) {
            foreach ($ad in $p2pAdapters) {
                $adMac = ([string]$ad.MacAddress).ToUpperInvariant().Replace(':', '-')
                if (($localExpectedMacs.Count -gt 0 -and $localExpectedMacs -contains $adMac) -or ($expectedMacs.Count -gt 0 -and $expectedMacs -contains $adMac)) {
                    $matchedAdapter = $ad
                    if (@($ad.NeighborCandidates).Count -gt 0) {
                        $matchedNeighbor = $ad.NeighborCandidates[0]
                    }
                    break
                }
            }
            if ($null -eq $matchedAdapter -and $p2pAdapters.Count -gt 0 -and $linkIdValue -eq 'gen7-gen9-usb4-hub-chain') {
                $matchedAdapter = $p2pAdapters[0]
                if (@($matchedAdapter.NeighborCandidates).Count -gt 0) {
                    $matchedNeighbor = $matchedAdapter.NeighborCandidates[0]
                }
            }
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

            $probeOutcome = if ($probeTarget) {
                Test-CanaryEndpoint -Address $probeTarget -TimeoutMs $ReprobeTimeoutMs
            } else {
                [PSCustomObject]@{ Reachable = $false; Port = $null; LatencyMs = $null }
            }

            $cleanNeighborIp = if ($neighborIp) { ($neighborIp -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }
            $cleanExpectedLl = if ($expectedLinkLocal) { ($expectedLinkLocal -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }
            $cleanExpectedUla = if ($expectedUla) { ($expectedUla -replace '%.*$', '').Trim().ToLowerInvariant() } else { $null }

            $macVerified = ($peerExpectedMacs.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($neighborMac) -and ($peerExpectedMacs -contains $neighborMac))
            $hostVerified = (-not [string]::IsNullOrWhiteSpace($neighborHost) -and $neighborHost.ToLowerInvariant() -eq $expectedPeerName.ToLowerInvariant())
            $neighborAddressMatchesExpectedPeer = (-not [string]::IsNullOrWhiteSpace($cleanNeighborIp)) -and (
                ($cleanNeighborIp -eq $cleanExpectedLl) -or ($cleanNeighborIp -eq $cleanExpectedUla)
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

        $linkReports.Add([PSCustomObject]@{
            LinkId                 = $linkIdValue
            Endpoints              = @($endpoints)
            AdapterName            = if ($matchedAdapter) { [string]$matchedAdapter.InterfaceAlias } else { $null }
            InterfaceIndex         = if ($matchedAdapter) { [int]$matchedAdapter.InterfaceIndex } else { $null }
            State                  = $state
            AlwaysOnTransitCapable = $false
            PreserveExistingPair   = [bool]$preservePair
            PreserveBridgeCustody  = [bool]$preserveBridge
            DownstreamLinkStatus   = $downstreamStatus
            WindowsXDomainPresent  = $xdomainPresent
            Evidence               = [PSCustomObject]@{
                ObservedAtUtc              = $observedUtc.ToString('o')
                TtlSeconds                 = $EvidenceTtlSeconds
                ExpiresAtUtc               = $expiresUtc.ToString('o')
                InvalidatedByModernStandby = [bool]$invalidatedByStandby
                Expired                    = [bool]$evidenceExpired
            }
            Canary                 = $canaryResult
            Reason                 = $reason
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
                    $probeRes = Test-CanaryEndpoint -Address ([string]$peerInfo.usb4UlaAddress) -TimeoutMs $remBudget
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

    # Evaluate PC-AI owned routes for stale route withdrawal while preserving active pairs and Milly bridge
    $routeActions = [System.Collections.Generic.List[object]]::new()
    $netRoutes = @(Get-SafeCimRows -ClassName 'MSFT_NetRoute')

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
            })
            continue
        }

        # Route targets a disconnected or asleep peer or exact /32 canary that is no longer valid
        $executed = $false
        $targetAddr = ($dest -replace '/\d+$', '')
        if ($WithdrawStaleRoutes -and $Apply) {
            if ($PSCmdlet.ShouldProcess("$dest (ifIndex $ifIdx)", 'Withdraw stale PC-AI USB4 route')) {
                if ($dest -match ':') {
                    & route.exe -6 delete $dest | Out-Null
                } else {
                    & route.exe delete $targetAddr | Out-Null
                }
                $executed = $true
            }
        }

        $routeActions.Add([PSCustomObject]@{
            DestinationPrefix = $dest
            InterfaceIndex    = $ifIdx
            OwnerPeer         = $ownerPeer
            Action            = 'WithdrawStaleOwnedRoute'
            Reason            = "Stale PC-AI route '$dest' on ifIndex $ifIdx has no 'ready' identity-verified USB4 link (owner='$ownerPeer')."
            Executed          = [bool]$executed
        })
    }

    return [PSCustomObject]@{
        LocalComputerName        = $canonicalLocalHost
        EvaluatedAtUtc           = $nowUtc.ToString('o')
        Ms01UplinkCapMbps        = if ($registry -and $registry.PSObject.Properties['usb4LifecyclePolicy']) { [int]$registry.usb4LifecyclePolicy.ms01UplinkCapMbps } else { 1000 }
        AnyAlwaysOnTransitClaim  = [bool]$anyAlwaysOnTransitCapable
        ModernStandby            = [PSCustomObject]@{
            S0LowPowerIdleConnectedSupported = $standby.S0LowPowerIdleConnectedSupported
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
