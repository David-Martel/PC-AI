#Requires -Version 5.1
<#
.SYNOPSIS
    Measures a link in both directions and localises any asymmetry to an endpoint.

.DESCRIPTION
    A single-direction throughput number cannot tell you which end is slow. Measuring
    both directions tells you the link is asymmetric but still not who is responsible:
    "A sends to B slowly" is equally consistent with A transmitting badly and B
    receiving badly.

    The discriminating measurement is a second peer. Testing the SAME sender against a
    DIFFERENT receiver holds the transmit path constant and varies only the receiver:

        A -> B slow  and  A -> C fast   =>  B's receive path is the problem
        A -> B slow  and  A -> C slow   =>  A's transmit path is the problem

    Without -ControlPeer this function reports the asymmetry and explicitly declines
    to attribute a cause, because attributing one from two numbers is a guess.

    Requires iperf3 on PATH here and an iperf3 server on each peer.

.PARAMETER Peer
    Primary peer running an iperf3 server.

.PARAMETER ControlPeer
    Second peer, also running an iperf3 server, used to localise the deficit.
    Choose one that is idle and reached over the same local interface.

.PARAMETER Port
    iperf3 port on both peers. Default 5201.

.PARAMETER Streams
    Parallel streams. Default 4. Compare single-stream and multi-stream to separate
    a per-connection limit from a per-interface one.

.PARAMETER Seconds
    Duration of each individual test. Default 8.

.PARAMETER AsymmetryFactor
    Ratio above which the link is called asymmetric. Default 2.0.

.EXAMPLE
    Test-NetPathHealth -Peer 10.60.4.1 -Port 5301

.EXAMPLE
    Test-NetPathHealth -Peer 10.60.4.1 -ControlPeer 10.60.4.2 -Port 5301
    Measures both directions to the peer, then holds the peer's transmit path constant
    against a second receiver to say which end is actually slow.

.OUTPUTS
    PSCustomObject with per-direction throughput and a localisation verdict.
#>
function Test-NetPathHealth {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$Peer,

        [Parameter()]
        [string]$ControlPeer,

        [Parameter()]
        [int]$Port = 5201,

        [Parameter()]
        [ValidateRange(1, 128)]
        [int]$Streams = 4,

        [Parameter()]
        [ValidateRange(1, 300)]
        [int]$Seconds = 8,

        [Parameter()]
        [ValidateRange(1.1, 100)]
        [double]$AsymmetryFactor = 2.0
    )

    if (-not (Get-Command iperf3 -ErrorAction SilentlyContinue)) {
        throw 'iperf3 was not found on PATH. Test-NetPathHealth measures with iperf3 and will not estimate throughput by any other means.'
    }

    function Invoke-Iperf {
        param([string]$Target, [switch]$Reverse)
        $iperfArgs = @('-c', $Target, '-p', $Port, '-P', $Streams, '-t', $Seconds, '-f', 'm')
        if ($Reverse) { $iperfArgs += '-R' }
        $raw = & iperf3 @iperfArgs 2>&1 | Out-String
        $line = ($raw -split "`n" | Where-Object { $_ -match '\[SUM\].*receiver' } | Select-Object -Last 1)
        if (-not $line) {
            # With -P 1 iperf3 prints no [SUM]; fall back to the per-stream receiver line.
            $line = ($raw -split "`n" | Where-Object { $_ -match 'receiver\s*$' } | Select-Object -Last 1)
        }
        if ($line -match '([\d.]+)\s+Mbits/sec') {
            [pscustomobject]@{ Mbits = [math]::Round([double]$Matches[1]); Error = $null }
        } else {
            $first = ($raw -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -First 2) -join ' / '
            [pscustomobject]@{ Mbits = $null; Error = $first }
        }
    }

    Write-Verbose "measuring local -> $Peer"
    $tx = Invoke-Iperf -Target $Peer
    Start-Sleep -Seconds 2
    Write-Verbose "measuring $Peer -> local"
    $rx = Invoke-Iperf -Target $Peer -Reverse

    $result = [ordered]@{
        Peer            = $Peer
        ControlPeer     = $ControlPeer
        Streams         = $Streams
        LocalToPeerMbits = $tx.Mbits
        PeerToLocalMbits = $rx.Mbits
        ControlMbits    = $null
        Asymmetric      = $false
        Ratio           = $null
        Verdict         = ''
        Errors          = @($tx.Error, $rx.Error | Where-Object { $_ })
    }

    if ($null -eq $tx.Mbits -or $null -eq $rx.Mbits) {
        $result.Verdict = 'INCOMPLETE: at least one direction did not produce a result. Check that an iperf3 server is listening on both ends and that the port is permitted by the firewall.'
        return [PSCustomObject]$result
    }

    $hi = [math]::Max($tx.Mbits, $rx.Mbits)
    $lo = [math]::Min($tx.Mbits, $rx.Mbits)
    $result.Ratio = if ($lo -gt 0) { [math]::Round($hi / $lo, 2) } else { [double]::PositiveInfinity }
    $result.Asymmetric = ($result.Ratio -ge $AsymmetryFactor)

    if (-not $result.Asymmetric) {
        $result.Verdict = 'SYMMETRIC: both directions within {0}x (ratio {1}).' -f $AsymmetryFactor, $result.Ratio
        return [PSCustomObject]$result
    }

    $slowDirection = if ($rx.Mbits -lt $tx.Mbits) { 'peer-to-local' } else { 'local-to-peer' }

    if (-not $ControlPeer) {
        $result.Verdict = ('ASYMMETRIC {0}x, slow direction is {1} ({2} vs {3} Mbit/s). ' -f
                           $result.Ratio, $slowDirection, $lo, $hi) +
                          'CAUSE NOT ESTABLISHED: two numbers cannot say which endpoint is responsible. ' +
                          'Re-run with -ControlPeer <another host on this segment> to localise it.'
        return [PSCustomObject]$result
    }

    # Hold the slow direction's SENDER constant, vary the receiver.
    if ($slowDirection -eq 'peer-to-local') {
        # Peer was sending. Have the peer send to the control host instead: we do that
        # by measuring control -> local is NOT equivalent, so instead compare this host's
        # receive against a different sender. Both isolate the local receive path.
        Write-Verbose "control: $ControlPeer -> local"
        $ctl = Invoke-Iperf -Target $ControlPeer -Reverse
        $result.ControlMbits = $ctl.Mbits
        if ($null -eq $ctl.Mbits) {
            $result.Verdict = "ASYMMETRIC $($result.Ratio)x but the control test against $ControlPeer failed: $($ctl.Error)"
            return [PSCustomObject]$result
        }
        $result.Verdict = if ($ctl.Mbits -lt ($hi / $AsymmetryFactor)) {
            ('LOCALISED TO THIS HOST''S RECEIVE PATH: {0} -> local is {1} Mbit/s and {2} -> local is {3} Mbit/s. ' -f
             $Peer, $lo, $ControlPeer, $ctl.Mbits) +
            'Two different senders are both slow into this machine, so the receiver is the constraint. ' +
            'Check RSS queue count and RSC operational state with Get-StableNetAdapter -Detailed.'
        } else {
            ('LOCALISED TO THE PEER OR THE PATH: {0} -> local is {1} Mbit/s but {2} -> local is {3} Mbit/s. ' -f
             $Peer, $lo, $ControlPeer, $ctl.Mbits) +
            'This host receives fine from another sender, so the deficit is not local.'
        }
    } else {
        Write-Verbose "control: local -> $ControlPeer"
        $ctl = Invoke-Iperf -Target $ControlPeer
        $result.ControlMbits = $ctl.Mbits
        if ($null -eq $ctl.Mbits) {
            $result.Verdict = "ASYMMETRIC but the control test against $ControlPeer failed: $($ctl.Error)"
            return [PSCustomObject]$result
        }
        $result.Verdict = if ($ctl.Mbits -lt ($hi / $AsymmetryFactor)) {
            ('LOCALISED TO THIS HOST''S TRANSMIT PATH: local -> {0} is {1} Mbit/s and local -> {2} is {3} Mbit/s. ' -f
             $Peer, $lo, $ControlPeer, $ctl.Mbits)
        } else {
            ('LOCALISED TO THE PEER''S RECEIVE PATH: local -> {0} is {1} Mbit/s but local -> {2} is {3} Mbit/s over the same transmit path. ' -f
             $Peer, $lo, $ControlPeer, $ctl.Mbits)
        }
    }

    [PSCustomObject]$result
}
