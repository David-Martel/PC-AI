<#
.SYNOPSIS
    Comprehensive connection testing for Windows remote management

.DESCRIPTION
    Tests multiple connection methods to a remote Windows host including:
    - ICMP Ping (IPv4 and IPv6)
    - DNS resolution
    - SMB (ports 445, 139)
    - WinRM HTTP/HTTPS (ports 5985, 5986, 80, 443)
    - SSH (configurable port)
    - Custom TCP ports

.PARAMETER Server
    Target server IP or hostname

.PARAMETER Credential
    PSCredential for authenticated tests. Will prompt if not provided.

.PARAMETER SkipAuth
    Skip tests that require authentication

.PARAMETER SSHPort
    SSH port to test (default: 31415)

.EXAMPLE
    .\test-connection-methods.ps1 -Server 10.10.20.214
    .\test-connection-methods.ps1 -Server dtm-work -SkipAuth
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$Server = "10.10.20.214",

    [pscredential]$Credential,

    [switch]$SkipAuth,

    [int]$SSHPort = 31415,

    [string]$OutputPath = "$env:USERPROFILE"
)

$ErrorActionPreference = 'Continue'

# ============================================================================
# CONFIGURATION
# ============================================================================
$script:Results = @()
$script:StartTime = Get-Date

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
function Write-TestResult {
    param(
        [string]$Test,
        [string]$Status,  # Success, Partial, Failed
        [string]$Details,
        [int]$LatencyMs = -1
    )

    $colors = @{ Success='Green'; Partial='Yellow'; Failed='Red' }
    $symbols = @{ Success='[+]'; Partial='[~]'; Failed='[-]' }

    $latencyStr = if ($LatencyMs -ge 0) { " (${LatencyMs}ms)" } else { "" }
    Write-Host "$($symbols[$Status]) $Test : $Status$latencyStr" -ForegroundColor $colors[$Status]
    if ($Details) {
        Write-Host "    $Details" -ForegroundColor Gray
    }

    $script:Results += [PSCustomObject]@{
        Test = $Test
        Status = $Status
        Details = $Details
        LatencyMs = $LatencyMs
        Timestamp = Get-Date -Format 'HH:mm:ss'
    }
}

function Test-TcpPort {
    param([string]$TargetHost, [int]$Port, [int]$TimeoutMs = 3000)

    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $async = $client.BeginConnect($TargetHost, $Port, $null, $null)
        $wait = $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        $sw.Stop()

        if ($wait -and $client.Connected) {
            $client.EndConnect($async)
            $client.Close()
            return @{ Success = $true; LatencyMs = $sw.ElapsedMilliseconds }
        }
        $client.Close()
        return @{ Success = $false; LatencyMs = -1; Error = "Timeout" }
    } catch {
        return @{ Success = $false; LatencyMs = -1; Error = $_.Exception.Message }
    }
}

# ============================================================================
# TEST FUNCTIONS
# ============================================================================
function Test-DNSResolution {
    param([string]$Hostname)

    Write-Host "`n=== DNS Resolution ===" -ForegroundColor Cyan

    # Skip if already an IP
    if ($Hostname -match '^\d+\.\d+\.\d+\.\d+$') {
        Write-TestResult -Test "DNS Resolution" -Status "Partial" -Details "Input is IP address, no DNS lookup needed"
        return $Hostname
    }

    try {
        $dns = Resolve-DnsName -Name $Hostname -ErrorAction Stop
        $ipv4 = $dns | Where-Object { $_.Type -eq 'A' } | Select-Object -First 1
        $ipv6 = $dns | Where-Object { $_.Type -eq 'AAAA' } | Select-Object -First 1

        if ($ipv4) {
            Write-TestResult -Test "DNS A Record" -Status "Success" -Details "$Hostname -> $($ipv4.IPAddress)"
        }
        if ($ipv6) {
            Write-TestResult -Test "DNS AAAA Record" -Status "Success" -Details "$Hostname -> $($ipv6.IPAddress)"
        }

        return $ipv4.IPAddress
    } catch {
        Write-TestResult -Test "DNS Resolution" -Status "Failed" -Details $_.Exception.Message
        return $Hostname  # Return original, might be resolvable by system
    }
}

function Test-ICMPConnectivity {
    param([string]$Target)

    Write-Host "`n=== ICMP Connectivity ===" -ForegroundColor Cyan

    # IPv4 Ping
    try {
        $ping = Test-Connection -ComputerName $Target -Count 3 -ErrorAction Stop
        $avg = [math]::Round(($ping | Measure-Object -Property Latency -Average).Average, 1)
        $loss = [math]::Round((1 - ($ping.Count / 3)) * 100, 0)
        Write-TestResult -Test "IPv4 Ping" -Status "Success" -Details "Latency: ${avg}ms, Loss: ${loss}%" -LatencyMs $avg
    } catch {
        Write-TestResult -Test "IPv4 Ping" -Status "Failed" -Details $_.Exception.Message
    }

    # IPv6 Ping (if available)
    try {
        $ping6 = Test-Connection -ComputerName $Target -Count 1 -IPv6 -ErrorAction SilentlyContinue
        if ($ping6) {
            Write-TestResult -Test "IPv6 Ping" -Status "Success" -LatencyMs $ping6.Latency
        }
    } catch {
        Write-TestResult -Test "IPv6 Ping" -Status "Partial" -Details "IPv6 not available or blocked"
    }
}

function Test-SMBConnectivity {
    param([string]$Target, [pscredential]$Cred)

    Write-Host "`n=== SMB Connectivity ===" -ForegroundColor Cyan

    # Test port 445
    $port445 = Test-TcpPort -TargetHost $Target -Port 445
    if ($port445.Success) {
        Write-TestResult -Test "SMB Port 445" -Status "Success" -LatencyMs $port445.LatencyMs
    } else {
        Write-TestResult -Test "SMB Port 445" -Status "Failed" -Details $port445.Error
    }

    # Test port 139 (NetBIOS)
    $port139 = Test-TcpPort -TargetHost $Target -Port 139
    if ($port139.Success) {
        Write-TestResult -Test "NetBIOS Port 139" -Status "Success" -LatencyMs $port139.LatencyMs
    } else {
        Write-TestResult -Test "NetBIOS Port 139" -Status "Partial" -Details "Legacy port, may be disabled"
    }

    # Test authenticated access
    if (-not $SkipAuth -and $Cred -and $port445.Success) {
        try {
            $testPath = "\\$Target\C$"
            # Use New-PSDrive with credentials
            $driveName = "TestSMB_$(Get-Random)"
            New-PSDrive -Name $driveName -PSProvider FileSystem -Root $testPath -Credential $Cred -ErrorAction Stop | Out-Null
            Remove-PSDrive -Name $driveName -ErrorAction SilentlyContinue
            Write-TestResult -Test "SMB Authentication" -Status "Success" -Details "Admin share accessible"
        } catch {
            if ($_ -match 'Access is denied') {
                Write-TestResult -Test "SMB Authentication" -Status "Partial" -Details "Credentials valid but admin share denied"
            } else {
                Write-TestResult -Test "SMB Authentication" -Status "Failed" -Details $_.Exception.Message
            }
        }
    }
}

function Test-WinRMConnectivity {
    param([string]$Target, [pscredential]$Cred)

    Write-Host "`n=== WinRM Connectivity ===" -ForegroundColor Cyan

    $winrmPorts = @(
        @{ Port = 5985; Name = "WinRM HTTP"; Protocol = "HTTP" },
        @{ Port = 5986; Name = "WinRM HTTPS"; Protocol = "HTTPS" },
        @{ Port = 80; Name = "WinRM Compat HTTP"; Protocol = "HTTP" },
        @{ Port = 443; Name = "WinRM Compat HTTPS"; Protocol = "HTTPS" }
    )

    foreach ($p in $winrmPorts) {
        $result = Test-TcpPort -TargetHost $Target -Port $p.Port
        if ($result.Success) {
            Write-TestResult -Test "$($p.Name) ($($p.Port))" -Status "Success" -LatencyMs $result.LatencyMs
        } else {
            Write-TestResult -Test "$($p.Name) ($($p.Port))" -Status "Failed" -Details $result.Error
        }
    }

    # Test WSMan protocol
    Write-Host ""
    try {
        $wsmanResult = Test-WSMan -ComputerName $Target -ErrorAction Stop
        Write-TestResult -Test "WSMan Protocol" -Status "Success" -Details "ProductVersion: $($wsmanResult.ProductVersion)"

        # Try authenticated session
        if (-not $SkipAuth -and $Cred) {
            try {
                $testResult = Invoke-Command -ComputerName $Target -ScriptBlock { $env:COMPUTERNAME } -Credential $Cred -ErrorAction Stop
                Write-TestResult -Test "WinRM Remote Execution" -Status "Success" -Details "Remote hostname: $testResult"
            } catch {
                Write-TestResult -Test "WinRM Remote Execution" -Status "Failed" -Details $_.Exception.Message
            }
        }
    } catch {
        $errorMsg = $_.Exception.Message
        if ($errorMsg -match 'Access is denied') {
            Write-TestResult -Test "WSMan Protocol" -Status "Partial" -Details "Protocol works but auth required"
        } elseif ($errorMsg -match 'IPv4Filter|IPv6Filter') {
            Write-TestResult -Test "WSMan Protocol" -Status "Failed" -Details "Blocked by server IPv4/IPv6 filter - check GPO"
        } elseif ($errorMsg -match 'firewall') {
            Write-TestResult -Test "WSMan Protocol" -Status "Failed" -Details "Firewall blocking - run setup-server-winrm.ps1 on server"
        } else {
            Write-TestResult -Test "WSMan Protocol" -Status "Failed" -Details $errorMsg
        }
    }
}

function Test-SSHConnectivity {
    param([string]$Target, [int]$Port)

    Write-Host "`n=== SSH Connectivity ===" -ForegroundColor Cyan

    $result = Test-TcpPort -TargetHost $Target -Port $Port
    if ($result.Success) {
        Write-TestResult -Test "SSH Port $Port" -Status "Partial" -Details "Port open (connection test only)" -LatencyMs $result.LatencyMs

        # Try to get SSH banner
        try {
            $client = New-Object System.Net.Sockets.TcpClient($Target, $Port)
            $stream = $client.GetStream()
            $stream.ReadTimeout = 3000
            $reader = New-Object System.IO.StreamReader($stream)
            $banner = $reader.ReadLine()
            $client.Close()

            if ($banner -match 'SSH') {
                Write-TestResult -Test "SSH Protocol" -Status "Success" -Details "Banner: $banner"
            } else {
                Write-TestResult -Test "SSH Protocol" -Status "Failed" -Details "Not SSH - received: $($banner.Substring(0, [Math]::Min(50, $banner.Length)))"
            }
        } catch {
            if ($_ -match 'reset|closed') {
                Write-TestResult -Test "SSH Protocol" -Status "Failed" -Details "Connection reset - IP may be banned or service not running"
            } else {
                Write-TestResult -Test "SSH Protocol" -Status "Partial" -Details "Could not verify protocol"
            }
        }
    } else {
        Write-TestResult -Test "SSH Port $Port" -Status "Failed" -Details $result.Error
    }

    # Also check standard SSH port if different
    if ($Port -ne 22) {
        $result22 = Test-TcpPort -TargetHost $Target -Port 22
        if ($result22.Success) {
            Write-TestResult -Test "SSH Port 22 (Standard)" -Status "Success" -LatencyMs $result22.LatencyMs
        } else {
            Write-TestResult -Test "SSH Port 22 (Standard)" -Status "Failed" -Details "Standard port not listening"
        }
    }
}

function Test-HTTPServices {
    param([string]$Target)

    Write-Host "`n=== HTTP Services ===" -ForegroundColor Cyan

    # Disable certificate validation for testing (PowerShell 5.1 compatible)
    $originalCallback = [System.Net.ServicePointManager]::ServerCertificateValidationCallback
    [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }

    try {
        foreach ($port in @(80, 8080, 443)) {
            $protocol = if ($port -eq 443) { "https" } else { "http" }
            $url = "${protocol}://${Target}:${port}/"

            try {
                # Use parameters compatible with both PS 5.1 and PS 7+
                $params = @{
                    Uri = $url
                    TimeoutSec = 5
                    UseBasicParsing = $true
                    ErrorAction = 'Stop'
                }

                # Add SkipCertificateCheck only if available (PS 7+)
                if ($PSVersionTable.PSVersion.Major -ge 7) {
                    $params['SkipCertificateCheck'] = $true
                }

                $response = Invoke-WebRequest @params
                Write-TestResult -Test "HTTP $port" -Status "Success" -Details "Status: $($response.StatusCode)"
            } catch {
                $statusCode = $_.Exception.Response.StatusCode.value__
                if ($statusCode) {
                    Write-TestResult -Test "HTTP $port" -Status "Partial" -Details "HTTP $statusCode response"
                } else {
                    Write-TestResult -Test "HTTP $port" -Status "Failed" -Details $_.Exception.Message
                }
            }
        }
    } finally {
        # Restore original callback
        [System.Net.ServicePointManager]::ServerCertificateValidationCallback = $originalCallback
    }
}

function Get-RouteInfo {
    param([string]$Target)

    Write-Host "`n=== Network Route ===" -ForegroundColor Cyan

    try {
        $route = Find-NetRoute -RemoteIPAddress $Target -ErrorAction Stop
        Write-TestResult -Test "Route Interface" -Status "Success" -Details "$($route[0].InterfaceAlias) via $($route[0].NextHop)"

        # Get interface details
        $iface = Get-NetIPAddress -InterfaceIndex $route[0].InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
        if ($iface) {
            Write-TestResult -Test "Source IP" -Status "Success" -Details $iface.IPAddress
        }
    } catch {
        Write-TestResult -Test "Route Info" -Status "Partial" -Details "Could not determine route"
    }
}

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
function Show-Summary {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "CONNECTION TEST SUMMARY" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan

    $successCount = ($script:Results | Where-Object Status -eq 'Success').Count
    $partialCount = ($script:Results | Where-Object Status -eq 'Partial').Count
    $failedCount = ($script:Results | Where-Object Status -eq 'Failed').Count

    Write-Host ""
    Write-Host "Results: " -NoNewline
    Write-Host "$successCount Success" -ForegroundColor Green -NoNewline
    Write-Host " | " -NoNewline
    Write-Host "$partialCount Partial" -ForegroundColor Yellow -NoNewline
    Write-Host " | " -NoNewline
    Write-Host "$failedCount Failed" -ForegroundColor Red

    # Recommendations based on results
    Write-Host "`nRecommendations:" -ForegroundColor Cyan

    $winrmFailed = $script:Results | Where-Object { $_.Test -match 'WinRM|WSMan' -and $_.Status -eq 'Failed' }
    $sshFailed = $script:Results | Where-Object { $_.Test -match 'SSH' -and $_.Status -eq 'Failed' }
    $smbSuccess = $script:Results | Where-Object { $_.Test -match 'SMB.*445' -and $_.Status -eq 'Success' }

    if ($winrmFailed -and $winrmFailed.Details -match 'filter|GPO') {
        Write-Host "  [!] WinRM blocked by IPv4Filter - edit GPO on server" -ForegroundColor Yellow
        Write-Host "      Run: gpedit.msc -> Computer Config -> Admin Templates ->" -ForegroundColor Gray
        Write-Host "           Windows Components -> WinRM -> WinRM Service" -ForegroundColor Gray
    }

    if ($winrmFailed -and $winrmFailed.Details -match 'firewall') {
        Write-Host "  [!] WinRM blocked by firewall - run setup-server-winrm.ps1 on server" -ForegroundColor Yellow
    }

    if ($sshFailed -and $sshFailed.Details -match 'reset|banned') {
        Write-Host "  [!] SSH connection reset - check if your IP is banned" -ForegroundColor Yellow
        Write-Host "      Or SSH service may not be listening on that port" -ForegroundColor Gray
    }

    if ($smbSuccess) {
        Write-Host "  [+] SMB is working - can use for file transfer" -ForegroundColor Green
        Write-Host "      robocopy C:\local \\$Server\share /MIR" -ForegroundColor Gray
    }

    if ($successCount -eq 0) {
        Write-Host "  [!] No connection methods working - check network/firewall" -ForegroundColor Red
    }

    # Export results
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $csvPath = Join-Path $OutputPath "connection-test-$timestamp.csv"
    $script:Results | Export-Csv -Path $csvPath -NoTypeInformation
    Write-Host "`nResults exported to: $csvPath" -ForegroundColor Gray
}

# ============================================================================
# MAIN
# ============================================================================
function Main {
    Write-Host ""
    Write-Host "Comprehensive Connection Test" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host "Target: $Server"
    Write-Host "SSH Port: $SSHPort"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""

    # Get credentials if needed
    if (-not $SkipAuth -and -not $Credential) {
        Write-Host "Enter credentials for authenticated tests (or press Cancel to skip):" -ForegroundColor Yellow
        try {
            $Credential = Get-Credential -Message "Credentials for $Server (Cancel to skip auth tests)"
        } catch {
            $SkipAuth = $true
            Write-Host "Skipping authenticated tests" -ForegroundColor Yellow
        }
    }

    # Run tests
    $resolvedIP = Test-DNSResolution -Hostname $Server
    Get-RouteInfo -Target $resolvedIP
    Test-ICMPConnectivity -Target $resolvedIP
    Test-SMBConnectivity -Target $resolvedIP -Cred $Credential
    Test-WinRMConnectivity -Target $resolvedIP -Cred $Credential
    Test-SSHConnectivity -Target $resolvedIP -Port $SSHPort
    Test-HTTPServices -Target $resolvedIP

    # Show summary
    Show-Summary
}

Main
