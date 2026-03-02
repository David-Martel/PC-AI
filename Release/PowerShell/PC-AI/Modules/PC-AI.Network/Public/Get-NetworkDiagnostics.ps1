#Requires -Version 5.1
<#
.SYNOPSIS
    Performs comprehensive network stack analysis

.DESCRIPTION
    Analyzes the entire network stack including physical adapters, virtual switches,
    DNS configuration, routing tables, and network connectivity. Identifies issues
    and provides actionable diagnostics.

.PARAMETER IncludeVirtual
    Include virtual network adapters in analysis (default: true)

.PARAMETER TestConnectivity
    Perform connectivity tests to common endpoints (default: true)

.PARAMETER TestDNS
    Perform DNS resolution tests (default: true)

.PARAMETER Detailed
    Include detailed adapter statistics and configuration

.EXAMPLE
    Get-NetworkDiagnostics
    Performs standard network diagnostics

.EXAMPLE
    Get-NetworkDiagnostics -Detailed
    Performs detailed diagnostics with extended statistics

.EXAMPLE
    Get-NetworkDiagnostics -IncludeVirtual:$false
    Analyze only physical adapters

.OUTPUTS
    PSCustomObject with network diagnostic results
#>
function Get-NetworkDiagnostics {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$IncludeVirtual = $true,

        [Parameter()]
        [switch]$TestConnectivity = $true,

        [Parameter()]
        [switch]$TestDNS = $true,

        [Parameter()]
        [switch]$Detailed
    )

    $result = [PSCustomObject]@{
        Timestamp         = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        ComputerName      = $env:COMPUTERNAME
        PhysicalAdapters  = @()
        VirtualAdapters   = @()
        VirtualSwitches   = @()
        DNSConfiguration  = $null
        RoutingTable      = @()
        ConnectivityTests = @()
        DNSTests          = @()
        Issues            = @()
        Summary           = $null
    }

    Write-Host "[*] Starting network diagnostics..." -ForegroundColor Cyan

    # Get physical adapters
    Write-Host "[*] Analyzing physical adapters..." -ForegroundColor Yellow
    try {
        $physicalAdapters = Get-NetAdapter | Where-Object {
            $_.Virtual -eq $false -or $IncludeVirtual -eq $false
        }

        foreach ($adapter in $physicalAdapters) {
            $ipConfig = Get-NetIPConfiguration -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue
            $ipAddresses = Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue

            $adapterInfo = [PSCustomObject]@{
                Name            = $adapter.Name
                Description     = $adapter.InterfaceDescription
                Status          = $adapter.Status
                LinkSpeed       = $adapter.LinkSpeed
                MacAddress      = $adapter.MacAddress
                MediaType       = $adapter.MediaType
                IPv4Address     = ($ipAddresses | Where-Object { $_.AddressFamily -eq 'IPv4' }).IPAddress -join ', '
                IPv6Address     = ($ipAddresses | Where-Object { $_.AddressFamily -eq 'IPv6' }).IPAddress | Select-Object -First 1
                DefaultGateway  = $ipConfig.IPv4DefaultGateway.NextHop
                DNSServers      = ($ipConfig.DNSServer.ServerAddresses -join ', ')
                DHCP            = $ipConfig.NetIPv4Interface.Dhcp
            }

            if ($Detailed) {
                $stats = Get-NetAdapterStatistics -Name $adapter.Name -ErrorAction SilentlyContinue
                if ($stats) {
                    Add-Member -InputObject $adapterInfo -NotePropertyName 'BytesSent' -NotePropertyValue $stats.SentBytes
                    Add-Member -InputObject $adapterInfo -NotePropertyName 'BytesReceived' -NotePropertyValue $stats.ReceivedBytes
                    Add-Member -InputObject $adapterInfo -NotePropertyName 'ErrorsSent' -NotePropertyValue $stats.OutboundDiscardedPackets
                    Add-Member -InputObject $adapterInfo -NotePropertyName 'ErrorsReceived' -NotePropertyValue $stats.InboundDiscardedPackets
                }
            }

            $result.PhysicalAdapters += $adapterInfo

            # Check for issues
            if ($adapter.Status -ne 'Up') {
                $result.Issues += [PSCustomObject]@{
                    Category = 'Adapter'
                    Severity = 'Warning'
                    Item     = $adapter.Name
                    Issue    = "Adapter status: $($adapter.Status)"
                    Recommendation = 'Check physical connection or enable the adapter'
                }
            }

            if (-not $adapterInfo.IPv4Address -and $adapter.Status -eq 'Up') {
                $result.Issues += [PSCustomObject]@{
                    Category = 'Adapter'
                    Severity = 'Critical'
                    Item     = $adapter.Name
                    Issue    = 'No IPv4 address assigned'
                    Recommendation = 'Check DHCP configuration or assign static IP'
                }
            }
        }

        Write-Host "  [+] Found $($result.PhysicalAdapters.Count) physical adapters" -ForegroundColor Green
    }
    catch {
        $result.Issues += [PSCustomObject]@{
            Category = 'System'
            Severity = 'Error'
            Item     = 'Physical Adapters'
            Issue    = "Failed to enumerate: $_"
            Recommendation = 'Check WMI service and network stack'
        }
    }

    # Get virtual adapters
    if ($IncludeVirtual) {
        Write-Host "[*] Analyzing virtual adapters..." -ForegroundColor Yellow
        try {
            $virtualAdapters = Get-NetAdapter | Where-Object { $_.Virtual -eq $true }

            foreach ($adapter in $virtualAdapters) {
                $ipConfig = Get-NetIPConfiguration -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue
                $ipAddresses = Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue

                $result.VirtualAdapters += [PSCustomObject]@{
                    Name        = $adapter.Name
                    Description = $adapter.InterfaceDescription
                    Status      = $adapter.Status
                    MacAddress  = $adapter.MacAddress
                    IPv4Address = ($ipAddresses | Where-Object { $_.AddressFamily -eq 'IPv4' }).IPAddress -join ', '
                    Type        = if ($adapter.Name -match 'WSL|vEthernet') { 'Hyper-V' } elseif ($adapter.Name -match 'VPN') { 'VPN' } else { 'Other' }
                }
            }

            Write-Host "  [+] Found $($result.VirtualAdapters.Count) virtual adapters" -ForegroundColor Green
        }
        catch {
            Write-Warning "Failed to enumerate virtual adapters: $_"
        }
    }

    # Get virtual switches (Hyper-V)
    Write-Host "[*] Checking virtual switches..." -ForegroundColor Yellow
    try {
        $vmSwitches = Get-VMSwitch -ErrorAction SilentlyContinue

        foreach ($switch in $vmSwitches) {
            $result.VirtualSwitches += [PSCustomObject]@{
                Name       = $switch.Name
                SwitchType = $switch.SwitchType
                NetAdapterInterfaceDescription = $switch.NetAdapterInterfaceDescription
                AllowManagementOS = $switch.AllowManagementOS
            }
        }

        # Check for WSL switch
        $wslSwitch = $vmSwitches | Where-Object { $_.Name -eq 'WSL' }
        if (-not $wslSwitch) {
            $result.Issues += [PSCustomObject]@{
                Category = 'VirtualSwitch'
                Severity = 'Info'
                Item     = 'WSL Switch'
                Issue    = 'No dedicated WSL virtual switch found'
                Recommendation = 'WSL uses default NAT networking. Consider creating dedicated switch for advanced scenarios.'
            }
        }

        Write-Host "  [+] Found $($result.VirtualSwitches.Count) virtual switches" -ForegroundColor Green
    }
    catch {
        Write-Host "  [=] Hyper-V not available or not running as admin" -ForegroundColor Gray
    }

    # DNS configuration
    Write-Host "[*] Analyzing DNS configuration..." -ForegroundColor Yellow
    try {
        $dnsServers = Get-DnsClientServerAddress -ErrorAction SilentlyContinue |
            Where-Object { $_.ServerAddresses.Count -gt 0 } |
            Select-Object InterfaceAlias, @{N='Servers';E={$_.ServerAddresses -join ', '}}

        $dnsCache = Get-DnsClientCache -ErrorAction SilentlyContinue | Select-Object -First 10
        $dnsSuffix = (Get-DnsClientGlobalSetting).SuffixSearchList

        $result.DNSConfiguration = [PSCustomObject]@{
            Servers      = $dnsServers
            CacheEntries = $dnsCache.Count
            SearchSuffix = $dnsSuffix -join ', '
        }

        Write-Host "  [+] DNS configuration retrieved" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to get DNS configuration: $_"
    }

    # Routing table
    Write-Host "[*] Analyzing routing table..." -ForegroundColor Yellow
    try {
        $routes = Get-NetRoute -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.DestinationPrefix -ne '255.255.255.255/32' } |
            Select-Object DestinationPrefix, NextHop, RouteMetric, InterfaceAlias |
            Sort-Object RouteMetric |
            Select-Object -First 20

        $result.RoutingTable = $routes

        # Check for default route
        $defaultRoute = $routes | Where-Object { $_.DestinationPrefix -eq '0.0.0.0/0' }
        if (-not $defaultRoute) {
            $result.Issues += [PSCustomObject]@{
                Category = 'Routing'
                Severity = 'Critical'
                Item     = 'Default Route'
                Issue    = 'No default route found'
                Recommendation = 'Check gateway configuration'
            }
        }

        Write-Host "  [+] Retrieved $($result.RoutingTable.Count) routes" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to get routing table: $_"
    }

    # Connectivity tests
    if ($TestConnectivity) {
        Write-Host "[*] Testing connectivity..." -ForegroundColor Yellow

        $testTargets = @(
            @{ Host = '8.8.8.8'; Description = 'Google DNS' },
            @{ Host = '1.1.1.1'; Description = 'Cloudflare DNS' },
            @{ Host = 'google.com'; Description = 'Google (ICMP)' },
            @{ Host = 'localhost'; Description = 'Loopback' }
        )

        foreach ($target in $testTargets) {
            $pingResult = Measure-NetworkLatency -Target $target.Host -Count 2
            $result.ConnectivityTests += [PSCustomObject]@{
                Target      = $target.Host
                Description = $target.Description
                Success     = $pingResult.Success
                AvgLatency  = if ($pingResult.Success) { "{0:N1} ms" -f $pingResult.AvgLatency } else { 'N/A' }
                PacketLoss  = "{0:N0}%" -f $pingResult.PacketLoss
            }

            if (-not $pingResult.Success) {
                $result.Issues += [PSCustomObject]@{
                    Category = 'Connectivity'
                    Severity = if ($target.Host -eq 'localhost') { 'Critical' } else { 'Warning' }
                    Item     = $target.Description
                    Issue    = "Cannot reach $($target.Host)"
                    Recommendation = 'Check network configuration and firewall rules'
                }
            }
        }

        Write-Host "  [+] Connectivity tests completed" -ForegroundColor Green
    }

    # DNS resolution tests
    if ($TestDNS) {
        Write-Host "[*] Testing DNS resolution..." -ForegroundColor Yellow

        $dnsTestTargets = @('google.com', 'microsoft.com', 'github.com')

        foreach ($target in $dnsTestTargets) {
            try {
                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $resolved = Resolve-DnsName -Name $target -Type A -ErrorAction Stop | Select-Object -First 1
                $sw.Stop()

                $result.DNSTests += [PSCustomObject]@{
                    Domain     = $target
                    Success    = $true
                    IPAddress  = $resolved.IPAddress
                    QueryTime  = "{0:N0} ms" -f $sw.ElapsedMilliseconds
                }
            }
            catch {
                $result.DNSTests += [PSCustomObject]@{
                    Domain    = $target
                    Success   = $false
                    IPAddress = 'N/A'
                    QueryTime = 'N/A'
                }

                $result.Issues += [PSCustomObject]@{
                    Category = 'DNS'
                    Severity = 'Critical'
                    Item     = $target
                    Issue    = "DNS resolution failed"
                    Recommendation = 'Check DNS server configuration'
                }
            }
        }

        Write-Host "  [+] DNS tests completed" -ForegroundColor Green
    }

    # Generate summary
    $criticalCount = ($result.Issues | Where-Object { $_.Severity -eq 'Critical' }).Count
    $warningCount = ($result.Issues | Where-Object { $_.Severity -eq 'Warning' }).Count
    $infoCount = ($result.Issues | Where-Object { $_.Severity -eq 'Info' }).Count

    $result.Summary = [PSCustomObject]@{
        TotalAdapters     = $result.PhysicalAdapters.Count + $result.VirtualAdapters.Count
        ActiveAdapters    = ($result.PhysicalAdapters | Where-Object { $_.Status -eq 'Up' }).Count
        VirtualSwitches   = $result.VirtualSwitches.Count
        CriticalIssues    = $criticalCount
        WarningIssues     = $warningCount
        InfoItems         = $infoCount
        OverallStatus     = if ($criticalCount -gt 0) { 'Critical' } elseif ($warningCount -gt 0) { 'Warning' } else { 'Healthy' }
    }

    # Display summary
    Write-Host ""
    Write-Host "== Network Diagnostics Summary ==" -ForegroundColor Cyan
    Write-Host "  Total Adapters: $($result.Summary.TotalAdapters) ($($result.Summary.ActiveAdapters) active)" -ForegroundColor White
    Write-Host "  Virtual Switches: $($result.Summary.VirtualSwitches)" -ForegroundColor White

    if ($criticalCount -gt 0) {
        Write-Host "  Critical Issues: $criticalCount" -ForegroundColor Red
    }
    if ($warningCount -gt 0) {
        Write-Host "  Warnings: $warningCount" -ForegroundColor Yellow
    }

    $statusColor = switch ($result.Summary.OverallStatus) {
        'Critical' { 'Red' }
        'Warning' { 'Yellow' }
        default { 'Green' }
    }
    Write-Host "  Overall Status: $($result.Summary.OverallStatus)" -ForegroundColor $statusColor

    return $result
}
