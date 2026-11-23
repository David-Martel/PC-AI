#Requires -Version 5.1
function Connect-PcaiEndpoint {
    <#
    .SYNOPSIS
        Opens a TCP connection to the first candidate that answers, racing address families per RFC 8305.

    .DESCRIPTION
        Happy Eyeballs v2 (RFC 8305) for peers that have several addresses, including scoped IPv6 link-local.

        Order (Get-PcaiConnectionOrder): scoped link-local candidates first (registry and neighbour-observed ahead of
        EUI-64 guesses, MAC-matched ahead of unmatched, a Reachable neighbour interface ahead of the rest), then
        IPv6 and IPv4 interleaved starting with IPv6.

        Race: attempt 1 starts at once. The next starts after -AttemptDelayMs, or IMMEDIATELY when the attempt in
        flight fails. The first success wins; every other client is disposed (outcome Cancelled), and no socket is
        left open. Attempts still pending at -TimeoutMs are disposed and reported as Timeout.

        Error convention: failure is DATA. When nothing answers the command returns Connected = $false with every
        attempt listed and does not throw. Pass -ErrorAction Stop explicitly to get a terminating error that carries
        the result as TargetObject. (A global $ErrorActionPreference of Stop does not change this, so a script can
        still read the attempt list.)

        A connect to a located address proves only that something listens there. Authenticate the peer with its SSH
        host key; identifiers never authenticate.

    .PARAMETER Candidate
        Candidates from Get-PcaiEndpointCandidate (pipeline), or any objects with an Address (zone allowed,
        fe80::1%25) or an IPAddress. Optional members: Source, NeighborState, MacMatched, Port.

    .PARAMETER HostName
        Resolve instead of supplying candidates: passed to Get-PcaiEndpointCandidate (as are PeerId, MacAddress,
        InterfaceAlias, InterfaceIndex, Provider, IncludeIPv4, IncludeIPv6 and Wake).

    .PARAMETER Port
        TCP port. Required unless every candidate carries a Port.

    .PARAMETER AttemptDelayMs
        Connection Attempt Delay, clamped to 10..2000. Default 250 (RFC 8305 recommends 250 ms).

    .PARAMETER TimeoutMs
        Overall deadline in milliseconds. Default 5000.

    .PARAMETER PreferredFamilyCount
        RFC 8305 First Address Family Count: how many IPv6 candidates are tried before the families alternate
        (IPv4, IPv6, IPv4, ...). Default 1. Link-local stays first within IPv6 but cannot starve IPv4.

    .PARAMETER PassThru
        Return the winning open TcpClient in the result's Client member; the caller owns and must dispose it.
        Without it the winner is closed again (a reachability test).

    .PARAMETER Connector
        Test seam. Script block (IPAddress, Port, Candidate) returning an object with Task (a
        System.Threading.Tasks.Task) and Client (IDisposable or $null). Default: TcpClient.ConnectAsync.

    .PARAMETER ResolverParameter
        Hashtable splatted into Get-PcaiEndpointCandidate (the injectable provider sources).

    .OUTPUTS
        Connected, Winner (normalised candidate), Attempts (Address, Class, Zone, Family, Source, StartMs, EndMs,
        Outcome Connected/Refused/Timeout/Unreachable/Cancelled, Error), ElapsedMs, TimedOut, Client.

    .EXAMPLE
        Connect-PcaiEndpoint -PeerId dtm-super-nuc -Port 22 -Wake | Format-List Connected, ElapsedMs

    .EXAMPLE
        $r = Get-PcaiEndpointCandidate -PeerId dtm-super-nuc | Connect-PcaiEndpoint -Port 22 -PassThru
        if ($r.Connected) { $r.Client.Dispose() }
    #>
    [CmdletBinding(DefaultParameterSetName = 'Candidate')]
    [OutputType([pscustomobject])]
    param(
        [Parameter(ParameterSetName = 'Candidate', Mandatory, ValueFromPipeline)][AllowNull()][AllowEmptyCollection()]$Candidate,
        [Parameter(ParameterSetName = 'Resolve')][string]$HostName,
        [Parameter(ParameterSetName = 'Resolve')][string]$PeerId,
        [Parameter(ParameterSetName = 'Resolve')][string]$MacAddress,
        [Parameter(ParameterSetName = 'Resolve')][string]$InterfaceAlias,
        [Parameter(ParameterSetName = 'Resolve')][int]$InterfaceIndex,
        [Parameter(ParameterSetName = 'Resolve')][ValidateSet('KnownPeer', 'Eui64', 'Neighbor', 'Dns')][string[]]$Provider,
        [Parameter(ParameterSetName = 'Resolve')][switch]$IncludeIPv4,
        [Parameter(ParameterSetName = 'Resolve')][switch]$IncludeIPv6,
        [Parameter(ParameterSetName = 'Resolve')][switch]$Wake,
        [Parameter(ParameterSetName = 'Resolve')][hashtable]$ResolverParameter,
        [ValidateRange(1, 65535)][int]$Port,
        [int]$AttemptDelayMs = 250,
        [int]$TimeoutMs = 5000,
        [ValidateRange(1, 64)][int]$PreferredFamilyCount = 1,
        [switch]$PassThru,
        [scriptblock]$Connector
    )

    begin {
        $collected = New-Object 'System.Collections.Generic.List[object]'
    }
    process {
        if ($PSCmdlet.ParameterSetName -eq 'Candidate' -and $null -ne $Candidate) {
            foreach ($item in @($Candidate)) { if ($null -ne $item) { $collected.Add($item) } }
        }
    }
    end {
        if ($AttemptDelayMs -lt 10) { $AttemptDelayMs = 10 }
        if ($AttemptDelayMs -gt 2000) { $AttemptDelayMs = 2000 }
        if ($TimeoutMs -lt 1) { $TimeoutMs = 1 }
        if (-not $Connector) { $Connector = { param($IPAddress, $Port, $Candidate) New-PcaiTcpConnection -IPAddress $IPAddress -Port $Port -Candidate $Candidate } }

        if ($PSCmdlet.ParameterSetName -eq 'Resolve') {
            $lookup = @{}
            foreach ($name in 'HostName', 'PeerId', 'MacAddress', 'InterfaceAlias', 'InterfaceIndex', 'Provider') {
                if ($PSBoundParameters.ContainsKey($name)) { $lookup[$name] = $PSBoundParameters[$name] }
            }
            if ($IncludeIPv4) { $lookup['IncludeIPv4'] = $true }
            if ($IncludeIPv6) { $lookup['IncludeIPv6'] = $true }
            if ($Wake) { $lookup['Wake'] = $true }
            if ($PSBoundParameters.ContainsKey('Port')) { $lookup['Port'] = $Port }
            if ($ResolverParameter) { foreach ($key in $ResolverParameter.Keys) { $lookup[$key] = $ResolverParameter[$key] } }
            foreach ($item in @(Get-PcaiEndpointCandidate @lookup)) { $collected.Add($item) }
        }

        # Normalise, then order. Unparseable candidates are reported as attempts that failed at once.
        $normalised = New-Object 'System.Collections.Generic.List[object]'
        $unparseable = New-Object 'System.Collections.Generic.List[object]'
        foreach ($item in $collected) {
            $n = ConvertTo-PcaiConnectCandidate -InputObject $item
            if ($null -eq $n) { $unparseable.Add($item) } else { $normalised.Add($n) }
        }
        # A candidate the egress proof omitted (Admitted = $false) is never dialled; it is reported as skipped.
        $dialable = New-Object 'System.Collections.Generic.List[object]'
        $skipped = New-Object 'System.Collections.Generic.List[object]'
        foreach ($n in $normalised) {
            if ($n.Admitted) { $dialable.Add($n) }
            else { $skipped.Add([pscustomobject]@{ Address = $n.Address; Class = $n.Class; Source = $n.Source; Reason = $n.Evidence }) }
        }
        $ordered = @(Get-PcaiConnectionOrder -Candidate $dialable.ToArray() -PreferredFamilyCount $PreferredFamilyCount)

        # Leave room for one attempt per family: if the stagger would start the second family after half the deadline,
        # compress it (never below the RFC 8305 floor of 10 ms).
        $firstV6 = -1; $firstV4 = -1
        for ($k = 0; $k -lt $ordered.Count; $k++) {
            if ($ordered[$k].Family -eq 'InterNetworkV6') { if ($firstV6 -lt 0) { $firstV6 = $k } }
            elseif ($firstV4 -lt 0) { $firstV4 = $k }
        }
        $needed = 0
        if ($firstV6 -ge 0 -and $firstV4 -ge 0) { $needed = [Math]::Max($firstV6, $firstV4) }
        if ($needed -gt 0 -and ($needed * $AttemptDelayMs) -gt ($TimeoutMs / 2)) {
            $AttemptDelayMs = [Math]::Max(10, [int][Math]::Floor($TimeoutMs / 2 / $needed))
        }

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $attempts = New-Object 'System.Collections.Generic.List[object]'
        $live = New-Object 'System.Collections.Generic.List[object]'
        $winner = $null
        $timedOut = $false
        $elapsedAtWin = 0
        $keepClient = $null
        $nextIndex = 0
        $nextStartAt = 0
        $disposeClient = {
            param($Client)
            if ($null -ne $Client) { try { $Client.Dispose() } catch { Write-Verbose "Dispose failed: $($_.Exception.Message)" } }
        }

        foreach ($bad in $unparseable) {
            $text = [string](Get-PcaiProperty $bad 'Address' $bad)
            $attempts.Add([pscustomobject]@{ Candidate = $null; Address = $text; Class = $null; Zone = $null; Family = $null; Source = [string](Get-PcaiProperty $bad 'Source' ''); StartMs = 0; EndMs = 0; Outcome = 'Unreachable'; Error = "'$text' is not a parseable IP address"; Task = $null; Client = $null })
        }

        try {
            while ($true) {
                # Harvest finished attempts, in candidate order so that a tie goes to the better candidate.
                $now = $sw.ElapsedMilliseconds
                foreach ($a in $live.ToArray()) {
                    if (-not $a.Task.IsCompleted) { continue }
                    $live.Remove($a) | Out-Null
                    $a.EndMs = $now
                    if ($a.Task.Status -eq [System.Threading.Tasks.TaskStatus]::RanToCompletion) {
                        if ($null -eq $winner) {
                            $winner = $a
                            $a.Outcome = 'Connected'
                            $elapsedAtWin = $now
                        }
                        else {
                            # Connected in the same wake as the winner: a loser. Close it.
                            $a.Outcome = 'Cancelled'
                            $a.Error = 'connected after the winner'
                            & $disposeClient $a.Client
                        }
                    }
                    else {
                        $failure = $null
                        if ($a.Task.IsCanceled) { $failure = [pscustomobject]@{ Outcome = 'Cancelled'; Message = 'connect was cancelled' } }
                        else { $failure = Get-PcaiAttemptOutcome -Exception $a.Task.Exception }
                        $a.Outcome = $failure.Outcome
                        $a.Error = $failure.Message
                        & $disposeClient $a.Client
                        # RFC 8305 section 5: a failure starts the next attempt now, not after the delay.
                        $nextStartAt = $now
                    }
                }
                if ($null -ne $winner) { break }
                $now = $sw.ElapsedMilliseconds
                if ($now -ge $TimeoutMs) { $timedOut = $true; break }
                if ($nextIndex -ge $ordered.Count -and $live.Count -eq 0) { break }

                if ($nextIndex -lt $ordered.Count -and $now -ge $nextStartAt) {
                    $cand = $ordered[$nextIndex]
                    $nextIndex++
                    $targetPort = 0
                    if ($PSBoundParameters.ContainsKey('Port')) { $targetPort = $Port }
                    elseif ($null -ne $cand.Port) { $targetPort = [int]$cand.Port }
                    $zone = $null
                    if ($null -ne $cand.InterfaceIndex) { $zone = [int]$cand.InterfaceIndex }
                    $a = [pscustomobject]@{
                        Candidate = $cand; Address = $cand.Address; Class = $cand.Class; Zone = $zone; Family = $cand.Family
                        Source = $cand.Source; StartMs = $now; EndMs = $null; Outcome = $null; Error = $null; Task = $null; Client = $null
                    }
                    $attempts.Add($a)
                    $nextStartAt = $now + $AttemptDelayMs
                    try {
                        if ($targetPort -le 0) { throw [System.ArgumentException]::new('no port: pass -Port or give the candidate a Port') }
                        $connection = & $Connector $cand.IPAddress $targetPort $cand
                        $a.Task = $connection.Task
                        $a.Client = $connection.Client
                        if ($a.Task -isnot [System.Threading.Tasks.Task]) { throw [System.InvalidOperationException]::new('connector did not return a Task') }
                        $live.Add($a)
                    }
                    catch {
                        $a.Outcome = 'Unreachable'
                        $a.Error = $_.Exception.Message
                        $a.EndMs = $sw.ElapsedMilliseconds
                        & $disposeClient $a.Client
                        $nextStartAt = $sw.ElapsedMilliseconds
                    }
                    continue
                }

                # Sleep until the next event: a task finishing, the next start, or the deadline.
                $wait = $TimeoutMs - $now
                if ($nextIndex -lt $ordered.Count) { $wait = [Math]::Min($wait, $nextStartAt - $now) }
                if ($wait -lt 0) { $wait = 0 }
                if ($live.Count -gt 0) {
                    $tasks = [System.Threading.Tasks.Task[]]@($live | ForEach-Object { $_.Task })
                    [void][System.Threading.Tasks.Task]::WaitAny($tasks, [int]$wait)
                }
                elseif ($wait -gt 0) { [System.Threading.Thread]::Sleep([int]$wait) }
            }
        }
        finally {
            $endMs = $sw.ElapsedMilliseconds
            foreach ($a in $live.ToArray()) {
                if ($null -ne $a.Task -and $a.Task.IsFaulted) { $null = $a.Task.Exception }
                $a.EndMs = $endMs
                if ($timedOut) { $a.Outcome = 'Timeout'; $a.Error = "no answer within ${TimeoutMs} ms" }
                else { $a.Outcome = 'Cancelled'; $a.Error = 'lost the race' }
                & $disposeClient $a.Client
            }
            $live.Clear()
            if ($null -ne $winner) {
                if ($PassThru) { $keepClient = $winner.Client } else { & $disposeClient $winner.Client }
            }
        }

        $elapsed = $elapsedAtWin
        if ($null -eq $winner) { $elapsed = $sw.ElapsedMilliseconds }
        $winningCandidate = $null
        if ($null -ne $winner) { $winningCandidate = $winner.Candidate }
        $rows = @($attempts | ForEach-Object {
                [pscustomobject]@{
                    Address = $_.Address; Class = $_.Class; Zone = $_.Zone; Family = $_.Family; Source = $_.Source
                    StartMs = $_.StartMs; EndMs = $_.EndMs; Outcome = $_.Outcome; Error = $_.Error
                }
            })
        $result = [pscustomobject]@{
            PSTypeName = 'PC-AI.Network.ConnectResult'
            Connected = ($null -ne $winner)
            Winner = $winningCandidate
            Attempts = $rows
            ElapsedMs = [int]$elapsed
            TimedOut = $timedOut
            EffectiveDelayMs = $AttemptDelayMs
            Skipped = @($skipped.ToArray())
            Client = $keepClient
        }
        if ($null -eq $winner -and $PSBoundParameters.ContainsKey('ErrorAction') -and "$($PSBoundParameters['ErrorAction'])" -eq 'Stop') {
            $summary = ($rows | ForEach-Object { "$($_.Address) $($_.Outcome)" }) -join '; '
            if (-not $summary) { $summary = 'no candidates' }
            $failure = [System.IO.IOException]::new("No endpoint answered within $TimeoutMs ms ($summary).")
            $record = New-Object System.Management.Automation.ErrorRecord $failure, 'PcaiEndpointUnreachable', ([System.Management.Automation.ErrorCategory]::ConnectionError), $result
            $PSCmdlet.ThrowTerminatingError($record)
        }
        $result
    }
}
