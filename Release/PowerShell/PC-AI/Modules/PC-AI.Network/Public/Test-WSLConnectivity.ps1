#Requires -Version 5.1
<#
.SYNOPSIS
    Tests connectivity between Windows and WSL

.DESCRIPTION
    Performs comprehensive connectivity testing between Windows host and WSL
    distributions including:
    - WSL service status
    - Network interface connectivity
    - DNS resolution from both sides
    - Port connectivity tests
    - VSock communication
    - Internet access through WSL

.PARAMETER Distribution
    Specific WSL distribution to test (default: default distribution)

.PARAMETER TestPorts
    Array of ports to test connectivity on (default: 22, 80, 443, 3000, 8080)

.PARAMETER DNSTargets
    Array of DNS targets to resolve (default: google.com, github.com, localhost)

.PARAMETER SkipInternetTest
    Skip internet connectivity tests

.PARAMETER Detailed
    Include detailed timing information

.EXAMPLE
    Test-WSLConnectivity
    Run standard connectivity tests

.EXAMPLE
    Test-WSLConnectivity -Distribution "Ubuntu-22.04" -Detailed
    Test specific distribution with detailed output

.EXAMPLE
    Test-WSLConnectivity -TestPorts 22,3000,5432
    Test with custom ports

.OUTPUTS
    PSCustomObject with comprehensive connectivity report
#>
function Test-WSLConnectivity {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$Distribution,

        [Parameter()]
        [int[]]$TestPorts = @(22, 80, 443, 3000, 8080),

        [Parameter()]
        [string[]]$DNSTargets = @('google.com', 'github.com', 'localhost'),

        [Parameter()]
        [switch]$SkipInternetTest,

        [Parameter()]
        [switch]$Detailed
    )

    $result = [PSCustomObject]@{
        Timestamp           = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        WSLStatus           = $null
        Distributions       = @()
        NetworkInterfaces   = @()
        DNSTests            = @()
        PortTests           = @()
        InternetAccess      = $null
        VSockStatus         = $null
        Issues              = @()
        Summary             = $null
    }

    Write-Host "[*] Starting WSL connectivity tests..." -ForegroundColor Cyan

    # Check WSL status
    Write-Host "[*] Checking WSL status..." -ForegroundColor Yellow
    try {
        $wslStatus = wsl --status 2>&1

        $result.WSLStatus = [PSCustomObject]@{
            Installed = $true
            Version = if ($wslStatus -match 'Default Version:\s*(\d+)') { $Matches[1] } else { 'Unknown' }
            DefaultDistro = if ($wslStatus -match 'Default Distribution:\s*(.+)$') { $Matches[1].Trim() } else { 'Unknown' }
            KernelVersion = if ($wslStatus -match 'Kernel version:\s*(.+)$') { $Matches[1].Trim() } else { 'Unknown' }
        }

        Write-Host "  [+] WSL Version: $($result.WSLStatus.Version)" -ForegroundColor Green
        Write-Host "  [+] Default Distro: $($result.WSLStatus.DefaultDistro)" -ForegroundColor Green
    }
    catch {
        $result.WSLStatus = [PSCustomObject]@{
            Installed = $false
            Error = $_.Exception.Message
        }
        $result.Issues += [PSCustomObject]@{
            Category = 'WSL'
            Severity = 'Critical'
            Issue = 'WSL not installed or not accessible'
            Recommendation = 'Install WSL using: wsl --install'
        }
        Write-Host "  [!] WSL not available: $_" -ForegroundColor Red
    }

    # Get distributions
    Write-Host "[*] Enumerating WSL distributions..." -ForegroundColor Yellow
    $distributions = Get-WSLDistributions

    if ($distributions.Count -gt 0) {
        $result.Distributions = $distributions
        foreach ($distro in $distributions) {
            $stateColor = switch ($distro.State) {
                'Running' { 'Green' }
                'Stopped' { 'Yellow' }
                default { 'Gray' }
            }
            Write-Host "  [+] $($distro.Name): $($distro.State) (WSL$($distro.Version))" -ForegroundColor $stateColor
        }
    }
    else {
        $result.Issues += [PSCustomObject]@{
            Category = 'WSL'
            Severity = 'Warning'
            Issue = 'No WSL distributions found'
            Recommendation = 'Install a distribution: wsl --install -d Ubuntu'
        }
        Write-Host "  [!] No distributions found" -ForegroundColor Yellow
    }

    # Select distribution to test
    $testDistro = if ($Distribution) {
        $Distribution
    }
    elseif ($result.WSLStatus.DefaultDistro -and $result.WSLStatus.DefaultDistro -ne 'Unknown') {
        $result.WSLStatus.DefaultDistro
    }
    elseif ($distributions.Count -gt 0) {
        $distributions[0].Name
    }
    else {
        $null
    }

    if (-not $testDistro) {
        Write-Host "[!] No distribution available for testing" -ForegroundColor Red
        $result.Summary = [PSCustomObject]@{
            OverallStatus = 'Failed'
            PassedTests = 0
            FailedTests = 1
            TotalTests = 1
        }
        return $result
    }

    Write-Host "[*] Testing distribution: $testDistro" -ForegroundColor Cyan

    # Network interface tests
    Write-Host "[*] Testing network interfaces..." -ForegroundColor Yellow

    # Windows side - check WSL adapter
    try {
        $wslAdapter = Get-NetAdapter | Where-Object { $_.Name -match 'WSL|vEthernet.*WSL' }
        if ($wslAdapter) {
            $wslIP = Get-NetIPAddress -InterfaceIndex $wslAdapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue

            $result.NetworkInterfaces += [PSCustomObject]@{
                Side = 'Windows'
                Interface = $wslAdapter.Name
                Status = $wslAdapter.Status
                IPAddress = $wslIP.IPAddress
                Test = 'Present'
            }
            Write-Host "  [+] Windows WSL adapter: $($wslAdapter.Name) - $($wslIP.IPAddress)" -ForegroundColor Green
        }
        else {
            $result.NetworkInterfaces += [PSCustomObject]@{
                Side = 'Windows'
                Interface = 'WSL Adapter'
                Status = 'Not Found'
                IPAddress = $null
                Test = 'Missing'
            }
            Write-Host "  [!] WSL adapter not found on Windows" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  [!] Error checking Windows adapter: $_" -ForegroundColor Yellow
    }

    # WSL side - check network
    try {
        $wslIP = wsl -d $testDistro -- ip -4 addr show eth0 2>&1 | Select-String -Pattern 'inet\s+(\d+\.\d+\.\d+\.\d+)'
        if ($wslIP) {
            $ipMatch = $wslIP.Matches.Groups[1].Value

            $result.NetworkInterfaces += [PSCustomObject]@{
                Side = 'WSL'
                Interface = 'eth0'
                Status = 'Up'
                IPAddress = $ipMatch
                Test = 'OK'
            }
            Write-Host "  [+] WSL eth0: $ipMatch" -ForegroundColor Green
        }
        else {
            $result.Issues += [PSCustomObject]@{
                Category = 'Network'
                Severity = 'Critical'
                Issue = 'WSL eth0 interface has no IP address'
                Recommendation = 'Check WSL networking configuration'
            }
            Write-Host "  [!] WSL eth0 has no IP" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  [!] Error checking WSL network: $_" -ForegroundColor Yellow
    }

    # DNS resolution tests
    Write-Host "[*] Testing DNS resolution..." -ForegroundColor Yellow

    foreach ($target in $DNSTargets) {
        # Windows DNS test
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $windowsDNS = Resolve-DnsName -Name $target -Type A -ErrorAction Stop | Select-Object -First 1
            $sw.Stop()

            $result.DNSTests += [PSCustomObject]@{
                Target = $target
                Side = 'Windows'
                Success = $true
                IPAddress = $windowsDNS.IPAddress
                QueryTime = "$($sw.ElapsedMilliseconds) ms"
            }
            Write-Host "  [+] Windows DNS: $target -> $($windowsDNS.IPAddress) ($($sw.ElapsedMilliseconds)ms)" -ForegroundColor Green
        }
        catch {
            $result.DNSTests += [PSCustomObject]@{
                Target = $target
                Side = 'Windows'
                Success = $false
                IPAddress = $null
                QueryTime = 'N/A'
            }
            Write-Host "  [!] Windows DNS failed: $target" -ForegroundColor Red
        }

        # WSL DNS test
        if ($target -ne 'localhost') {
            try {
                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $wslDNS = wsl -d $testDistro -- nslookup $target 2>&1 | Select-String -Pattern 'Address:\s*(\d+\.\d+\.\d+\.\d+)' | Select-Object -Last 1
                $sw.Stop()

                if ($wslDNS) {
                    $wslIP = $wslDNS.Matches.Groups[1].Value
                    $result.DNSTests += [PSCustomObject]@{
                        Target = $target
                        Side = 'WSL'
                        Success = $true
                        IPAddress = $wslIP
                        QueryTime = "$($sw.ElapsedMilliseconds) ms"
                    }
                    Write-Host "  [+] WSL DNS: $target -> $wslIP ($($sw.ElapsedMilliseconds)ms)" -ForegroundColor Green
                }
                else {
                    $result.DNSTests += [PSCustomObject]@{
                        Target = $target
                        Side = 'WSL'
                        Success = $false
                        IPAddress = $null
                        QueryTime = 'N/A'
                    }
                    $result.Issues += [PSCustomObject]@{
                        Category = 'DNS'
                        Severity = 'Critical'
                        Issue = "WSL cannot resolve: $target"
                        Recommendation = 'Check /etc/resolv.conf in WSL'
                    }
                    Write-Host "  [!] WSL DNS failed: $target" -ForegroundColor Red
                }
            }
            catch {
                Write-Host "  [!] WSL DNS error for $target`: $_" -ForegroundColor Yellow
            }
        }
    }

    # Port connectivity tests (Windows to WSL)
    Write-Host "[*] Testing port connectivity..." -ForegroundColor Yellow

    foreach ($port in $TestPorts) {
        try {
            # Get WSL IP
            $wslIPLine = wsl -d $testDistro -- hostname -I 2>&1
            $wslIP = ($wslIPLine -split '\s+')[0]

            if ($wslIP -match '\d+\.\d+\.\d+\.\d+') {
                $portTest = Test-PortConnectivity -Host $wslIP -Port $port -TimeoutMs 2000

                $result.PortTests += [PSCustomObject]@{
                    Port = $port
                    Host = $wslIP
                    Direction = 'Windows->WSL'
                    Success = $portTest.Success
                    Message = $portTest.Message
                }

                if ($portTest.Success) {
                    Write-Host "  [+] Port $port`: Open" -ForegroundColor Green
                }
                else {
                    Write-Host "  [-] Port $port`: $($portTest.Message)" -ForegroundColor Gray
                }
            }
        }
        catch {
            Write-Host "  [!] Port $port test error: $_" -ForegroundColor Yellow
        }
    }

    # Internet access test from WSL
    if (-not $SkipInternetTest) {
        Write-Host "[*] Testing internet access from WSL..." -ForegroundColor Yellow

        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $internetTest = wsl -d $testDistro -- curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 https://www.google.com 2>&1
            $sw.Stop()

            $httpCode = $internetTest.Trim()

            $result.InternetAccess = [PSCustomObject]@{
                Success = ($httpCode -eq '200' -or $httpCode -eq '301' -or $httpCode -eq '302')
                HTTPCode = $httpCode
                ResponseTime = "$($sw.ElapsedMilliseconds) ms"
                Target = 'https://www.google.com'
            }

            if ($result.InternetAccess.Success) {
                Write-Host "  [+] Internet access: OK (HTTP $httpCode, $($sw.ElapsedMilliseconds)ms)" -ForegroundColor Green
            }
            else {
                Write-Host "  [!] Internet access: Failed (HTTP $httpCode)" -ForegroundColor Red
                $result.Issues += [PSCustomObject]@{
                    Category = 'Internet'
                    Severity = 'Critical'
                    Issue = 'WSL cannot access internet'
                    Recommendation = 'Check DNS and network configuration in WSL'
                }
            }
        }
        catch {
            $result.InternetAccess = [PSCustomObject]@{
                Success = $false
                HTTPCode = 'N/A'
                ResponseTime = 'N/A'
                Error = $_.Exception.Message
            }
            Write-Host "  [!] Internet test failed: $_" -ForegroundColor Red
        }
    }

    # VSock status
    Write-Host "[*] Checking VSock status..." -ForegroundColor Yellow

    try {
        # Check if VSock driver is loaded
        $vsockDriver = Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'Hyper-V.*Socket|VSock' }

        if ($vsockDriver) {
            $result.VSockStatus = [PSCustomObject]@{
                Available = $true
                Driver = $vsockDriver.Name
                Status = $vsockDriver.Status
            }
            Write-Host "  [+] VSock driver: $($vsockDriver.Status)" -ForegroundColor Green
        }
        else {
            # Try alternative check
            $hvsocket = Get-Service hvsocketcontrol -ErrorAction SilentlyContinue
            if ($hvsocket) {
                $result.VSockStatus = [PSCustomObject]@{
                    Available = $true
                    Driver = 'hvsocketcontrol'
                    Status = $hvsocket.Status
                }
                Write-Host "  [+] HV Socket: $($hvsocket.Status)" -ForegroundColor Green
            }
            else {
                $result.VSockStatus = [PSCustomObject]@{
                    Available = $false
                    Driver = 'Not found'
                    Status = 'N/A'
                }
                Write-Host "  [=] VSock driver not detected (may be integrated)" -ForegroundColor Gray
            }
        }
    }
    catch {
        Write-Host "  [!] VSock check error: $_" -ForegroundColor Yellow
    }

    # Generate summary
    $internetTestCount = if ($result.InternetAccess) { 1 } else { 0 }
    $totalTests = $result.DNSTests.Count + $result.PortTests.Count + $internetTestCount

    $internetPassCount = if ($result.InternetAccess -and $result.InternetAccess.Success) { 1 } else { 0 }
    $passedTests = ($result.DNSTests | Where-Object { $_.Success }).Count +
                   ($result.PortTests | Where-Object { $_.Success }).Count +
                   $internetPassCount

    $criticalIssues = ($result.Issues | Where-Object { $_.Severity -eq 'Critical' }).Count
    $warningIssues = ($result.Issues | Where-Object { $_.Severity -eq 'Warning' }).Count

    $result.Summary = [PSCustomObject]@{
        Distribution = $testDistro
        OverallStatus = if ($criticalIssues -gt 0) { 'Critical' } elseif ($warningIssues -gt 0) { 'Warning' } else { 'Healthy' }
        PassedTests = $passedTests
        FailedTests = $totalTests - $passedTests
        TotalTests = $totalTests
        CriticalIssues = $criticalIssues
        WarningIssues = $warningIssues
    }

    # Display summary
    Write-Host ""
    Write-Host "== WSL Connectivity Summary ==" -ForegroundColor Cyan
    Write-Host "  Distribution: $testDistro" -ForegroundColor White
    Write-Host "  Tests Passed: $passedTests/$totalTests" -ForegroundColor White

    if ($criticalIssues -gt 0) {
        Write-Host "  Critical Issues: $criticalIssues" -ForegroundColor Red
    }
    if ($warningIssues -gt 0) {
        Write-Host "  Warnings: $warningIssues" -ForegroundColor Yellow
    }

    $statusColor = switch ($result.Summary.OverallStatus) {
        'Critical' { 'Red' }
        'Warning' { 'Yellow' }
        default { 'Green' }
    }
    Write-Host "  Overall Status: $($result.Summary.OverallStatus)" -ForegroundColor $statusColor

    # Show issues if any
    if ($result.Issues.Count -gt 0) {
        Write-Host ""
        Write-Host "== Issues Found ==" -ForegroundColor Yellow
        foreach ($issue in $result.Issues) {
            $issueColor = switch ($issue.Severity) {
                'Critical' { 'Red' }
                'Warning' { 'Yellow' }
                default { 'Gray' }
            }
            Write-Host "  [$($issue.Severity)] $($issue.Category): $($issue.Issue)" -ForegroundColor $issueColor
            Write-Host "    -> $($issue.Recommendation)" -ForegroundColor Gray
        }
    }

    return $result
}
