#Requires -RunAsAdministrator
<#
.SYNOPSIS
Lightweight DNS proxy for resolving .local service domains to localhost.

.DESCRIPTION
This script sets up DNS redirection for .local domains to 127.0.0.1 without
interfering with external DNS or authentication systems.

.PARAMETER Action
start    : Enable DNS proxy and set DNS to localhost
stop     : Disable DNS proxy and restore system DNS
status   : Show current DNS configuration
restart  : Restart the DNS proxy service

.EXAMPLE
.\LocalDNSProxy.ps1 -Action start
.\LocalDNSProxy.ps1 -Action status
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('start', 'stop', 'status', 'restart')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$serviceName = 'dnsproxy'
$dnsProxyPort = 5354
$dnsProxyConfig = @{
    listen = "127.0.0.1:$dnsProxyPort"
    upstream = "1.1.1.1:53"
    defaultUpstream = $true
}

# Check if running as admin
if (-not ([System.Security.Principal.WindowsIdentity]::GetCurrent().Groups -match 'S-1-5-32-544')) {
    Write-Error "This script must run as Administrator"
    exit 1
}

function Get-DNSProxyPath {
    $paths = @(
        "C:\Program Files\dnsproxy\dnsproxy.exe",
        "C:\Users\david\bin\dnsproxy.exe",
        (Get-Command dnsproxy.exe -ErrorAction SilentlyContinue).Source
    )

    foreach ($path in $paths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }

    return $null
}

function Start-DNSProxy {
    Write-Host "Starting DNS Proxy..." -ForegroundColor Cyan

    $proxyPath = Get-DNSProxyPath
    if (-not $proxyPath) {
        Write-Warning "dnsproxy.exe not found. Installing dnsproxy..."
        Install-DNSProxy
        $proxyPath = Get-DNSProxyPath
        if (-not $proxyPath) {
            throw "Failed to locate dnsproxy after installation"
        }
    }

    # Create Windows Service
    try {
        $existingService = Get-Service $serviceName -ErrorAction SilentlyContinue
        if ($existingService) {
            Stop-Service $serviceName -Force
            Remove-Service $serviceName -Force
            Start-Sleep -Seconds 1
        }
    } catch {
        Write-Verbose "Service didn't exist or couldn't be removed"
    }

    # Create service
    $serviceParams = @{
        Name = $serviceName
        BinaryPathName = "`"$proxyPath`" --listen 127.0.0.1:$dnsProxyPort --upstream 8.8.8.8:53 --upstream 1.1.1.1:53"
        DisplayName = "DNS Proxy for Local Services"
        StartupType = "Automatic"
    }

    New-Service @serviceParams -ErrorAction SilentlyContinue | Out-Null
    Start-Service $serviceName

    Write-Host "✓ DNS Proxy service started" -ForegroundColor Green

    # Pause to let service stabilize
    Start-Sleep -Seconds 2

    # Set DNS to localhost (gracefully)
    Set-LocalDNS
}

function Stop-DNSProxy {
    Write-Host "Stopping DNS Proxy..." -ForegroundColor Cyan

    try {
        Stop-Service $serviceName -Force -ErrorAction SilentlyContinue
        Remove-Service $serviceName -Force -ErrorAction SilentlyContinue
        Write-Host "✓ DNS Proxy service stopped" -ForegroundColor Green
    } catch {
        Write-Warning "Could not stop service: $_"
    }

    # Restore system DNS to automatic/DHCP
    Restore-SystemDNS
}

function Restart-DNSProxy {
    Stop-DNSProxy
    Start-Sleep -Seconds 2
    Start-DNSProxy
}

function Set-LocalDNS {
    Write-Host "Configuring DNS to use localhost..." -ForegroundColor Cyan

    try {
        # Get active network adapters (excluding virtual/disabled)
        $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }

        foreach ($adapter in $adapters) {
            Write-Host "Setting DNS for: $($adapter.Name)" -ForegroundColor Gray

            # Set DNS to localhost with fallback to 8.8.8.8
            Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex `
                -ServerAddresses @('127.0.0.1', '8.8.8.8', '1.1.1.1') `
                -ErrorAction SilentlyContinue
        }

        Write-Host "✓ DNS configured" -ForegroundColor Green
    } catch {
        Write-Warning "Could not set DNS: $_"
    }
}

function Restore-SystemDNS {
    Write-Host "Restoring system DNS settings..." -ForegroundColor Cyan

    try {
        $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }

        foreach ($adapter in $adapters) {
            # Reset to DHCP
            Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex `
                -ResetServerAddresses `
                -ErrorAction SilentlyContinue
        }

        Write-Host "✓ DNS restored to system defaults" -ForegroundColor Green
    } catch {
        Write-Warning "Could not restore DNS: $_"
    }
}

function Show-Status {
    Write-Host "`n=== DNS Proxy Status ===" -ForegroundColor Cyan

    # Service status
    $service = Get-Service $serviceName -ErrorAction SilentlyContinue
    if ($service) {
        $status = $service.Status
        $color = if ($status -eq 'Running') { 'Green' } else { 'Red' }
        Write-Host "Service Status: " -NoNewline
        Write-Host $status -ForegroundColor $color
    } else {
        Write-Host "Service Status: " -NoNewline
        Write-Host "Not installed" -ForegroundColor Yellow
    }

    # Current DNS settings
    Write-Host "`nDNS Servers (Active Adapters):" -ForegroundColor Cyan
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }
    foreach ($adapter in $adapters) {
        $dnsServers = (Get-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex).ServerAddresses -join ', '
        Write-Host "  $($adapter.Name): $dnsServers" -ForegroundColor Gray
    }

    # Test DNS resolution
    Write-Host "`nDNS Resolution Test:" -ForegroundColor Cyan
    $testDomains = @('vertex-code-reviewer.local', 'mcp.local', 'google.com')
    foreach ($domain in $testDomains) {
        try {
            $result = [System.Net.Dns]::GetHostAddresses($domain)[0].IPAddressToString
            $color = if ($result -eq '127.0.0.1') { 'Green' } else { 'Yellow' }
            Write-Host "  $domain -> " -NoNewline
            Write-Host $result -ForegroundColor $color
        } catch {
            Write-Host "  $domain -> " -NoNewline
            Write-Host "Failed to resolve" -ForegroundColor Red
        }
    }

    Write-Host "`n"
}

function Install-DNSProxy {
    Write-Host "Downloading dnsproxy..." -ForegroundColor Cyan

    $binPath = "C:\Users\david\bin"
    if (-not (Test-Path $binPath)) {
        New-Item -ItemType Directory -Path $binPath -Force | Out-Null
    }

    $dnsProxyExe = "$binPath\dnsproxy.exe"

    # Try multiple sources
    $sources = @(
        "https://github.com/AdguardTeam/dnsproxy/releases/download/v0.66.4/dnsproxy-windows-amd64.exe"
    )

    foreach ($source in $sources) {
        try {
            Write-Host "Trying: $source" -ForegroundColor Gray
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $source -OutFile $dnsProxyExe -TimeoutSec 60 -ErrorAction Stop

            if (Test-Path $dnsProxyExe) {
                Write-Host "✓ dnsproxy downloaded successfully" -ForegroundColor Green
                return $dnsProxyExe
            }
        } catch {
            Write-Verbose "Failed from $source : $_"
            continue
        }
    }

    throw "Failed to download dnsproxy from any source"
}

# Main execution
try {
    switch ($Action) {
        'start' {
            Start-DNSProxy
            Show-Status
        }
        'stop' {
            Stop-DNSProxy
            Show-Status
        }
        'restart' {
            Restart-DNSProxy
            Show-Status
        }
        'status' {
            Show-Status
        }
    }
} catch {
    Write-Error "Error: $_"
    exit 1
}
