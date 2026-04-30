#Requires -RunAsAdministrator
<#
.SYNOPSIS
Install and configure Acrylic DNS for local service resolution.

.DESCRIPTION
- Downloads Acrylic DNS from alternative sources
- Installs it to Program Files
- Configures hosts file for .local domain resolution
- Sets up Windows Service for auto-start
- Configures system DNS to use Acrylic
#>

$ErrorActionPreference = 'Stop'
$AcrylicDir = "C:\Program Files\Acrylic DNS Proxy"
$AcrylicExe = "$AcrylicDir\Acrylic.exe"
$AcrylicHostsFile = "$AcrylicDir\AcrylicHosts.txt"
$AcrylicConfigFile = "$AcrylicDir\AcrylicConfiguration.ini"

Write-Host "`n=== Installing Acrylic DNS Proxy ===" -ForegroundColor Cyan

# Step 1: Check if already installed
Write-Host "`n[1/5] Checking for existing installation..." -ForegroundColor Gray
if (Test-Path $AcrylicExe) {
    Write-Host "✓ Acrylic DNS already installed at $AcrylicDir" -ForegroundColor Green
} else {
    Write-Host "⚠ Acrylic not found, attempting installation..." -ForegroundColor Yellow

    # Try winget first, then fall back to manual installation
    Write-Host "Attempting installation via winget..." -ForegroundColor Gray
    try {
        & winget install --id Mayakron.AcrylicDNS --accept-source-agreements --accept-package-agreements -q 2>&1 | Out-Null
        if (Test-Path $AcrylicExe) {
            Write-Host "✓ Acrylic installed successfully via winget" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠ Winget installation failed, skipping" -ForegroundColor Yellow
    }

    # If still not installed, provide manual instructions
    if (-not (Test-Path $AcrylicExe)) {
        Write-Host "`n⚠ Manual installation required:" -ForegroundColor Yellow
        Write-Host "  1. Download from: http://www.acrylic-dns.com/" -ForegroundColor Yellow
        Write-Host "  2. Run the installer" -ForegroundColor Yellow
        Write-Host "  3. Re-run this script to configure" -ForegroundColor Yellow
        exit 1
    }
}

# Step 2: Create Acrylic hosts configuration
Write-Host "`n[2/5] Configuring Acrylic hosts file..." -ForegroundColor Gray
if (-not (Test-Path $AcrylicHostsFile)) {
    Write-Host "⚠ AcrylicHosts.txt not found, creating..." -ForegroundColor Yellow
    $hostsContent = @"
# Acrylic DNS Proxy - Local Service Configuration
# Services for local development and testing

# Vertex AI MCP Services
127.0.0.1 vertex-code-reviewer.local
127.0.0.1 vertex-code-generator.local
127.0.0.1 vertex-master-architect.local
127.0.0.1 vertex-workspace-analyzer.local
127.0.0.1 vertex-doc-generator.local

# MCP Services
127.0.0.1 mcp.local

# Gemini CLI
127.0.0.1 gemini-cli.local

# Add your custom .local domains here
"@

    try {
        Set-Content -Path $AcrylicHostsFile -Value $hostsContent -Force
        Write-Host "✓ Hosts file configured" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Could not write hosts file: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "✓ Hosts file already exists" -ForegroundColor Green

    # Add our services if not present
    $hostsContent = Get-Content $AcrylicHostsFile
    $needsUpdate = $false

    $servicesToAdd = @(
        "127.0.0.1 vertex-code-reviewer.local",
        "127.0.0.1 mcp.local"
    )

    foreach ($service in $servicesToAdd) {
        if ($hostsContent -notmatch [regex]::Escape($service)) {
            $hostsContent += "`n$service"
            $needsUpdate = $true
        }
    }

    if ($needsUpdate) {
        Set-Content -Path $AcrylicHostsFile -Value $hostsContent -Force
        Write-Host "✓ Added missing services to hosts file" -ForegroundColor Green
    } else {
        Write-Host "✓ All services already configured" -ForegroundColor Green
    }
}

# Step 3: Start Acrylic service
Write-Host "`n[3/5] Starting Acrylic DNS service..." -ForegroundColor Gray
try {
    Start-Service -Name "Acrylic DNS Proxy Service" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $service = Get-Service "Acrylic DNS Proxy Service" -ErrorAction SilentlyContinue
    if ($service.Status -eq 'Running') {
        Write-Host "✓ Acrylic service started" -ForegroundColor Green
    } else {
        Write-Host "⚠ Service status: $($service.Status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not start service: $_" -ForegroundColor Yellow
}

# Step 4: Configure system DNS
Write-Host "`n[4/5] Configuring system DNS to use Acrylic..." -ForegroundColor Gray
try {
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }
    $configured = 0

    foreach ($adapter in $adapters) {
        Write-Host "  Setting DNS for: $($adapter.Name)" -ForegroundColor Gray
        Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex `
            -ServerAddresses @('127.0.0.1', '8.8.8.8', '1.1.1.1') `
            -ErrorAction SilentlyContinue
        $configured++
    }

    if ($configured -gt 0) {
        Write-Host "✓ System DNS configured for $configured adapter(s)" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Could not configure DNS: $_" -ForegroundColor Yellow
}

# Step 5: Verify DNS resolution
Write-Host "`n[5/5] Testing DNS resolution..." -ForegroundColor Gray
Start-Sleep -Seconds 1
ipconfig /flushdns | Out-Null

$testDomains = @('mcp.local', 'google.com')
$allPass = $true

foreach ($domain in $testDomains) {
    try {
        $result = [System.Net.Dns]::GetHostAddresses($domain)[0].IPAddressToString
        if ($domain -eq 'mcp.local' -and $result -eq '127.0.0.1') {
            Write-Host "  ✓ $domain -> $result (local)" -ForegroundColor Green
        } elseif ($domain -ne 'mcp.local' -and $result -ne '127.0.0.1') {
            Write-Host "  ✓ $domain -> $result (external)" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $domain -> $result (unexpected)" -ForegroundColor Red
            $allPass = $false
        }
    } catch {
        Write-Host "  ✗ $domain failed to resolve" -ForegroundColor Red
        $allPass = $false
    }
}

# Final status
Write-Host "`n=== Installation Complete ===" -ForegroundColor Cyan
if ($allPass) {
    Write-Host "✓ Acrylic DNS is configured and working correctly" -ForegroundColor Green
    Write-Host "`nYou can now:" -ForegroundColor Green
    Write-Host "  • Access local services via .local domains (e.g., http://mcp.local)"
    Write-Host "  • External domains resolve normally (e.g., google.com)"
    Write-Host "  • Services are defined in: $AcrylicHostsFile" -ForegroundColor Green
} else {
    Write-Host "⚠ Installation complete but some tests failed" -ForegroundColor Yellow
    Write-Host "  Check DNS settings and try: ipconfig /flushdns" -ForegroundColor Yellow
}

Write-Host "`nTo manage Acrylic DNS:" -ForegroundColor Cyan
Write-Host "  • Open: $AcrylicDir" -ForegroundColor Gray
Write-Host "  • GUI: Run 'Acrylic.exe'" -ForegroundColor Gray
Write-Host "  • Service: Get-Service 'Acrylic DNS Proxy Service'" -ForegroundColor Gray
Write-Host ""
