#Requires -Version 5.1
<#
.SYNOPSIS
    Recovers a USB/Thunderbolt network adapter stuck in a degraded receive state.

.DESCRIPTION
    USB-attached NICs (Realtek RTL815x, common in USB-C dongles and Thunderbolt
    docks) can enter a state where the link is Up at full negotiated speed, the
    IP configuration is intact, ping succeeds at normal latency, and the interface
    error and discard counters are all ZERO -- yet receive throughput has collapsed
    by one to two orders of magnitude. Nothing in the usual health checks reports a
    problem, because by every one of those measures the adapter is healthy.

    Two things do NOT recover it, both verified:
      * Restart-NetAdapter
      * restoring whatever advanced property triggered it

    What does recover it is a full PnP device cycle (Disable-PnpDevice then
    Enable-PnpDevice), which is the software equivalent of unplugging the cable.
    A single cycle is often only a partial recovery; measured recovery on the case
    this was written from went 13 -> 255 -> 754 Mbit/s across two cycles, so this
    function cycles until a throughput target is met or -MaxCycles is exhausted.

    Because "healthy" link state cannot distinguish the degraded case, a throughput
    measurement is the only real verification. Supply -TestPeer (needs iperf3 on
    PATH here and an iperf3 server on the peer) or your own -ThroughputTest script
    block. With neither, the function still performs the cycle but can only verify
    link and IP, and says so in the result rather than claiming success.

.PARAMETER InterfaceGuid
    Target adapter's InterfaceGuid. Stable across re-enumeration; prefer this.

.PARAMETER TestPeer
    Host running an iperf3 server, used to measure receive throughput.

.PARAMETER TestPort
    Port of that iperf3 server. Default 5201.

.PARAMETER MinAcceptableMbits
    Receive throughput at or above which the adapter is considered healthy.
    Required when a throughput test is available.

.PARAMETER ThroughputTest
    Script block returning receive throughput in Mbit/s as a number. Overrides the
    built-in iperf3 path; use it when the peer speaks something else.

.PARAMETER MaxCycles
    Maximum PnP cycles to attempt. Default 3.

.PARAMETER SettleSeconds
    Seconds to wait after the adapter returns before measuring. USB NICs need
    longer than PCIe ones. Default 10.

.EXAMPLE
    Repair-UsbNetAdapter -InterfaceGuid '{2222E1D5-9E2E-4B08-B751-3A92F8079347}' `
                         -TestPeer 10.60.4.1 -TestPort 5301 -MinAcceptableMbits 700

.EXAMPLE
    Repair-UsbNetAdapter -InterfaceGuid $guid -WhatIf
    Show what would happen without touching the device.

.OUTPUTS
    PSCustomObject with per-cycle measurements and a Recovered flag.

.NOTES
    Requires an elevated session. The cycle drops the link for several seconds --
    do not run it over the link being repaired.
#>
function Repair-UsbNetAdapter {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$InterfaceGuid,

        [Parameter()]
        [string]$TestPeer,

        [Parameter()]
        [int]$TestPort = 5201,

        [Parameter()]
        [double]$MinAcceptableMbits,

        [Parameter()]
        [scriptblock]$ThroughputTest,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$MaxCycles = 3,

        [Parameter()]
        [ValidateRange(1, 120)]
        [int]$SettleSeconds = 10
    )

    $isAdmin = ([Security.Principal.WindowsPrincipal] `
                [Security.Principal.WindowsIdentity]::GetCurrent()
               ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        throw 'Repair-UsbNetAdapter requires an elevated session: Disable-PnpDevice/Enable-PnpDevice are admin-only.'
    }

    # Resolve through a closure, not a captured name: the adapter's Name and ifIndex
    # can both differ after the PnP cycle, which is exactly why the caller keys on GUID.
    $resolve = { Get-StableNetAdapter -InterfaceGuid $InterfaceGuid }.GetNewClosure()

    $nic = & $resolve
    $expectedIp = ($nic.IPv4Address -split ',')[0].Split('/')[0]

    # Decide, once, how throughput will be measured -- and be explicit when it cannot be.
    $measure = $null
    if ($ThroughputTest) {
        $measure = $ThroughputTest
    } elseif ($TestPeer) {
        if (-not (Get-Command iperf3 -ErrorAction SilentlyContinue)) {
            Write-Warning 'iperf3 is not on PATH; falling back to link-state verification only.'
        } else {
            $peer = $TestPeer; $port = $TestPort
            $measure = {
                # -R makes the SERVER send, which is the receive direction under test.
                $raw = & iperf3 -c $peer -p $port -P 4 -t 8 -f m -R 2>&1 | Out-String
                $line = ($raw -split "`n" | Where-Object { $_ -match '\[SUM\].*receiver' } | Select-Object -Last 1)
                if ($line -match '([\d.]+)\s+Mbits/sec') { [double]$Matches[1] } else { $null }
            }.GetNewClosure()
        }
    }
    if ($measure -and -not $PSBoundParameters.ContainsKey('MinAcceptableMbits')) {
        throw 'A throughput test was supplied but -MinAcceptableMbits was not. Without a threshold there is no pass condition.'
    }

    $result = [ordered]@{
        Adapter          = $nic.Name
        InterfaceGuid    = $nic.InterfaceGuid
        PnPDeviceID      = $nic.PnPDeviceID
        MeasurementUsed  = [bool]$measure
        Baseline         = $null
        Cycles           = @()
        Recovered        = $false
        Verified         = $false
        Message          = ''
    }

    if ($measure) {
        $result.Baseline = & $measure
        Write-Verbose ("baseline receive: {0} Mbit/s" -f $result.Baseline)
        if ($null -ne $result.Baseline -and $result.Baseline -ge $MinAcceptableMbits) {
            $result.Recovered = $true; $result.Verified = $true
            $result.Message = ('No repair needed: receive is {0} Mbit/s, at or above the {1} Mbit/s threshold.' -f
                               [math]::Round($result.Baseline), $MinAcceptableMbits)
            return [PSCustomObject]$result
        }
    }

    for ($i = 1; $i -le $MaxCycles; $i++) {
        $nic = & $resolve
        $target = ('{0} ({1})' -f $nic.Name, $nic.PnPDeviceID)
        if (-not $PSCmdlet.ShouldProcess($target, "PnP disable/enable cycle $i of $MaxCycles")) {
            $result.Message = 'Skipped: -WhatIf or declined at the confirmation prompt.'
            return [PSCustomObject]$result
        }

        Write-Verbose "cycle ${i}: disabling $($nic.PnPDeviceID)"
        Disable-PnpDevice -InstanceId $nic.PnPDeviceID -Confirm:$false -ErrorAction Stop
        Start-Sleep -Seconds 6
        Enable-PnpDevice -InstanceId $nic.PnPDeviceID -Confirm:$false -ErrorAction Stop

        if (-not (Wait-PcaiAdapterReady -Resolve $resolve -ExpectedIPv4 $expectedIp `
                                        -WarmPingTarget $TestPeer -TimeoutSeconds 90)) {
            $result.Cycles += [PSCustomObject]@{ Cycle = $i; Mbits = $null; Note = 'adapter did not return' }
            $result.Message = "Adapter did not come back after cycle $i. Investigate before cycling again."
            return [PSCustomObject]$result
        }
        Start-Sleep -Seconds $SettleSeconds

        $mbits = if ($measure) { & $measure } else { $null }
        $result.Cycles += [PSCustomObject]@{ Cycle = $i; Mbits = if ($null -ne $mbits) { [math]::Round($mbits) } else { $null } }
        Write-Verbose ("cycle {0}: receive {1} Mbit/s" -f $i, $mbits)

        if (-not $measure) {
            $result.Recovered = $true
            $result.Message = 'Cycle completed and the adapter returned with its address. ' +
                              'NOT verified: no throughput test was available, and a degraded USB NIC ' +
                              'presents as fully healthy by link state alone.'
            return [PSCustomObject]$result
        }
        if ($null -ne $mbits -and $mbits -ge $MinAcceptableMbits) {
            $result.Recovered = $true; $result.Verified = $true
            $result.Message = ('Recovered after {0} cycle(s): {1} Mbit/s, at or above the {2} Mbit/s threshold.' -f
                               $i, [math]::Round($mbits), $MinAcceptableMbits)
            return [PSCustomObject]$result
        }
    }

    $best = ($result.Cycles | Where-Object { $null -ne $_.Mbits } | Measure-Object -Property Mbits -Maximum).Maximum
    $result.Message = ('Still below threshold after {0} cycle(s). Best observed {1} Mbit/s against a {2} Mbit/s target. ' -f
                       $MaxCycles, $best, $MinAcceptableMbits) +
                      'Next step is a physical replug or a reboot; the device may need a true bus reset.'
    [PSCustomObject]$result
}
