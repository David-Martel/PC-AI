#Requires -Version 5.1
<#
.SYNOPSIS
    Real-time VSock and network interface performance monitoring

.DESCRIPTION
    Monitors VSock and network interface performance in real-time with configurable
    refresh intervals. Displays throughput, packet rates, errors, and TCP statistics.

    Monitors:
    - Network interface throughput (bytes/sec)
    - Packet rates (packets/sec)
    - Error and discard rates
    - TCP connection statistics
    - WSL-specific network metrics

.PARAMETER RefreshInterval
    Seconds between updates (default: 2)

.PARAMETER Duration
    Total monitoring duration in seconds (default: unlimited, press Ctrl+C to stop)

.PARAMETER InterfaceFilter
    Filter adapters by name pattern (supports wildcards)

.PARAMETER IncludeVirtual
    Include virtual adapters in monitoring (default: true)

.PARAMETER OutputFile
    Path to save monitoring data as CSV

.PARAMETER Quiet
    Suppress console output (use with -OutputFile)

.EXAMPLE
    Watch-VSockPerformance
    Monitor all interfaces with default 2-second interval

.EXAMPLE
    Watch-VSockPerformance -RefreshInterval 1 -Duration 60
    Monitor for 60 seconds with 1-second updates

.EXAMPLE
    Watch-VSockPerformance -InterfaceFilter "*WSL*" -OutputFile "wsl-perf.csv"
    Monitor WSL interfaces and save to CSV

.OUTPUTS
    Real-time console display or PSCustomObject collection
#>
function Watch-VSockPerformance {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(1, 60)]
        [int]$RefreshInterval = 2,

        [Parameter()]
        [ValidateRange(0, 86400)]
        [int]$Duration = 0,

        [Parameter()]
        [string]$InterfaceFilter = '*',

        [Parameter()]
        [switch]$IncludeVirtual = $true,

        [Parameter()]
        [string]$OutputFile,

        [Parameter()]
        [switch]$Quiet
    )

    # Initialize collection for output
    $monitoringData = @()
    $startTime = Get-Date
    $endTime = if ($Duration -gt 0) { $startTime.AddSeconds($Duration) } else { $null }
    $loopCount = 0
    $maxIterations = $null
    if ($Duration -gt 0) {
        $safeInterval = [math]::Max($RefreshInterval, 1)
        $maxIterations = [math]::Ceiling($Duration / $safeInterval) + 1
    }

    # Previous statistics for delta calculation
    $previousStats = @{}

    # Display header
    if (-not $Quiet) {
        Clear-Host
        Write-Host "=======================================" -ForegroundColor Cyan
        Write-Host "  VSock Performance Monitor" -ForegroundColor Cyan
        Write-Host "  Refresh: ${RefreshInterval}s | Filter: $InterfaceFilter" -ForegroundColor Gray
        if ($Duration -gt 0) {
            Write-Host "  Duration: ${Duration}s" -ForegroundColor Gray
        }
        Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host "=======================================" -ForegroundColor Cyan
        Write-Host ""
    }

    $canUseCursor = $false
    if (-not $Quiet -and $Host -and $Host.UI -and $Host.UI.RawUI) {
        try {
            $null = $Host.UI.RawUI.CursorPosition
            $canUseCursor = ($Host.Name -eq 'ConsoleHost')
        } catch {
            $canUseCursor = $false
        }
    }

    # Get TCP global parameters once
    $tcpGlobal = $null
    try {
        $tcpGlobal = Get-NetTCPSetting -SettingName Internet -ErrorAction SilentlyContinue
    }
    catch { }

    try {
        while ($true) {
            $loopCount++
            $currentTime = Get-Date

            # Get adapters
            $adapters = Get-NetAdapter | Where-Object {
                $_.Name -like $InterfaceFilter -and
                ($IncludeVirtual -or $_.Virtual -eq $false)
            }

            # Collect current statistics
            $currentStats = @{}
            $interfaceMetrics = @()

            foreach ($adapter in $adapters) {
                try {
                    $stats = Get-NetAdapterStatistics -Name $adapter.Name -ErrorAction SilentlyContinue

                    if ($stats) {
                        $key = $adapter.Name

                        $currentStats[$key] = @{
                            BytesSent = $stats.SentBytes
                            BytesReceived = $stats.ReceivedBytes
                            PacketsSent = $stats.SentUnicastPackets
                            PacketsReceived = $stats.ReceivedUnicastPackets
                            ErrorsSent = $stats.OutboundDiscardedPackets + $stats.OutboundPacketErrors
                            ErrorsReceived = $stats.InboundDiscardedPackets + $stats.InboundPacketErrors
                            Timestamp = $currentTime
                        }

                        # Calculate deltas if we have previous data
                        if ($previousStats.ContainsKey($key)) {
                            $prev = $previousStats[$key]
                            $curr = $currentStats[$key]
                            $timeDelta = ($currentTime - $prev.Timestamp).TotalSeconds

                            if ($timeDelta -gt 0) {
                                $bytesSentPerSec = [math]::Max(0, ($curr.BytesSent - $prev.BytesSent) / $timeDelta)
                                $bytesRecvPerSec = [math]::Max(0, ($curr.BytesReceived - $prev.BytesReceived) / $timeDelta)
                                $packetsSentPerSec = [math]::Max(0, ($curr.PacketsSent - $prev.PacketsSent) / $timeDelta)
                                $packetsRecvPerSec = [math]::Max(0, ($curr.PacketsReceived - $prev.PacketsReceived) / $timeDelta)
                                $errorRate = [math]::Max(0, (($curr.ErrorsSent + $curr.ErrorsReceived) - ($prev.ErrorsSent + $prev.ErrorsReceived)) / $timeDelta)

                                $metric = [PSCustomObject]@{
                                    Timestamp         = $currentTime.ToString('yyyy-MM-dd HH:mm:ss')
                                    Interface         = $adapter.Name
                                    Status            = $adapter.Status
                                    LinkSpeed         = $adapter.LinkSpeed
                                    TxRate            = Format-BytesPerSecond -BytesPerSecond $bytesSentPerSec
                                    RxRate            = Format-BytesPerSecond -BytesPerSecond $bytesRecvPerSec
                                    TxPPS             = [math]::Round($packetsSentPerSec, 0)
                                    RxPPS             = [math]::Round($packetsRecvPerSec, 0)
                                    ErrorRate         = [math]::Round($errorRate, 2)
                                    TxBytesPerSec     = [math]::Round($bytesSentPerSec, 0)
                                    RxBytesPerSec     = [math]::Round($bytesRecvPerSec, 0)
                                    TotalBytesSent    = $curr.BytesSent
                                    TotalBytesReceived = $curr.BytesReceived
                                }

                                $interfaceMetrics += $metric
                                $monitoringData += $metric
                            }
                        }
                    }
                }
                catch {
                    Write-Verbose "Error getting stats for $($adapter.Name): $_"
                }
            }

            # Update previous stats
            $previousStats = $currentStats

            # Get TCP statistics
            $tcpStats = $null
            try {
                $tcpConnections = Get-NetTCPConnection -ErrorAction SilentlyContinue
                $tcpStats = [PSCustomObject]@{
                    Established = ($tcpConnections | Where-Object { $_.State -eq 'Established' }).Count
                    TimeWait    = ($tcpConnections | Where-Object { $_.State -eq 'TimeWait' }).Count
                    CloseWait   = ($tcpConnections | Where-Object { $_.State -eq 'CloseWait' }).Count
                    Listen      = ($tcpConnections | Where-Object { $_.State -eq 'Listen' }).Count
                    Total       = $tcpConnections.Count
                }
            }
            catch { }

            # Display output
            if (-not $Quiet -and $interfaceMetrics.Count -gt 0) {
                if ($canUseCursor) {
                    try {
                        # Move cursor to top (after header)
                        $cursorPos = $Host.UI.RawUI.CursorPosition
                        $cursorPos.Y = 8
                        $Host.UI.RawUI.CursorPosition = $cursorPos

                        # Clear previous content
                        for ($i = 0; $i -lt 30; $i++) {
                            Write-Host (' ' * 100)
                        }

                        $cursorPos.Y = 8
                        $Host.UI.RawUI.CursorPosition = $cursorPos
                    } catch {
                        $canUseCursor = $false
                    }
                }

                # Display interface metrics
                Write-Host "Network Interface Performance" -ForegroundColor Yellow
                Write-Host ('-' * 80) -ForegroundColor Gray

                $displayTable = $interfaceMetrics | Select-Object Interface, Status, TxRate, RxRate, TxPPS, RxPPS, ErrorRate

                $displayTable | Format-Table -AutoSize | Out-String | Write-Host

                # TCP Statistics
                if ($tcpStats) {
                    Write-Host "TCP Connections" -ForegroundColor Yellow
                    Write-Host ('-' * 80) -ForegroundColor Gray
                    Write-Host ("  Established: {0} | TimeWait: {1} | CloseWait: {2} | Listen: {3} | Total: {4}" -f `
                        $tcpStats.Established, $tcpStats.TimeWait, $tcpStats.CloseWait, $tcpStats.Listen, $tcpStats.Total) -ForegroundColor White
                }

                # TCP Global Settings
                if ($tcpGlobal) {
                    Write-Host ""
                    Write-Host "TCP Settings" -ForegroundColor Yellow
                    Write-Host ('-' * 80) -ForegroundColor Gray
                    Write-Host ("  AutoTuning: {0} | CongestionProvider: {1} | ECN: {2}" -f `
                        $tcpGlobal.AutoTuningLevelLocal,
                        $tcpGlobal.CongestionProvider,
                        $tcpGlobal.EcnCapability) -ForegroundColor White
                }

                # Elapsed time
                $elapsed = ($currentTime - $startTime).TotalSeconds
                Write-Host ""
                Write-Host ("Elapsed: {0:N0}s | Last Update: {1}" -f $elapsed, $currentTime.ToString('HH:mm:ss')) -ForegroundColor Gray
            }

            # Check duration limit after collecting a sample
            if ($endTime -and $currentTime -ge $endTime) {
                if (-not $Quiet) {
                    Write-Host "`n[*] Monitoring duration completed" -ForegroundColor Yellow
                }
                break
            }
            if ($maxIterations -and $loopCount -ge $maxIterations) {
                if (-not $Quiet) {
                    Write-Host "`n[*] Monitoring iteration limit reached" -ForegroundColor Yellow
                }
                break
            }

            # Sleep for interval
            Start-Sleep -Seconds $RefreshInterval
        }
    }
    catch {
        if ($_.Exception.Message -notmatch 'canceled|stopped') {
            Write-Error "Monitoring error: $_"
        }
    }
    finally {
        # Save to file if requested
        if ($OutputFile -and $monitoringData.Count -gt 0) {
            try {
                $monitoringData | Export-Csv -Path $OutputFile -NoTypeInformation -Force
                Write-Host "`n[+] Monitoring data saved to: $OutputFile" -ForegroundColor Green
            }
            catch {
                Write-Error "Failed to save output file: $_"
            }
        }
    }

    # Return data for pipeline
    if ($monitoringData.Count -gt 0) {
        return $monitoringData
    }
}
