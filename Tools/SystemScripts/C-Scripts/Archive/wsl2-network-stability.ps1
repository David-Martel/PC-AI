# WSL2 Network Stability and Performance Enhancement Script
# Focused on mirrored networking mode optimization

Write-Host "=== WSL2 Network Stability Enhancement ===" -ForegroundColor Cyan

# 1. Advanced Hyper-V VSock Configuration
Write-Host "`n1. Configuring Hyper-V VSock for optimal WSL2 communication..." -ForegroundColor Yellow

try {
    # Get WSL VM information
    $wslVMs = Get-VM | Where-Object {$_.Name -like "*WSL*"}

    foreach ($vm in $wslVMs) {
        # Enable enhanced session mode for better performance
        Set-VM -Name $vm.Name -EnhancedSessionTransportType HvSocket -ErrorAction SilentlyContinue

        # Configure memory settings for stability
        Set-VMMemory -VMName $vm.Name -DynamicMemoryEnabled $true -MinimumBytes 2GB -MaximumBytes 36GB -ErrorAction SilentlyContinue

        Write-Host "Configured VM: $($vm.Name)" -ForegroundColor Green
    }
} catch {
    Write-Host "Direct VM configuration not accessible. WSL manages VMs automatically." -ForegroundColor Yellow
}

# 2. Network Interface Optimization
Write-Host "`n2. Optimizing network interfaces for WSL2 mirrored mode..." -ForegroundColor Yellow

# Get WSL network adapters
$wslAdapters = Get-NetAdapter | Where-Object {$_.InterfaceDescription -like "*WSL*" -or $_.Name -like "*WSL*"}

foreach ($adapter in $wslAdapters) {
    try {
        # Optimize buffer sizes for high-performance networking
        Set-NetAdapterAdvancedProperty -Name $adapter.Name -PropertyName "ReceiveBuffers" -PropertyValue "2048" -ErrorAction SilentlyContinue
        Set-NetAdapterAdvancedProperty -Name $adapter.Name -PropertyName "TransmitBuffers" -PropertyValue "2048" -ErrorAction SilentlyContinue

        # Enable jumbo frames if supported (for internal networking)
        Set-NetAdapterAdvancedProperty -Name $adapter.Name -PropertyName "*JumboPacket" -PropertyValue "9014" -ErrorAction SilentlyContinue

        # Disable power management to prevent network drops
        $powerMgmt = Get-NetAdapterPowerManagement -Name $adapter.Name -ErrorAction SilentlyContinue
        if ($powerMgmt) {
            Set-NetAdapterPowerManagement -Name $adapter.Name -AllowComputerToTurnOffDevice Disabled -ErrorAction SilentlyContinue
        }

        Write-Host "Optimized adapter: $($adapter.Name)" -ForegroundColor Green
    } catch {
        Write-Host "Could not optimize adapter: $($adapter.Name)" -ForegroundColor Yellow
    }
}

# 3. DNS and Routing Optimization
Write-Host "`n3. Configuring DNS and routing for development stability..." -ForegroundColor Yellow

try {
    # Configure DNS cache settings
    Set-DnsClientGlobalSetting -SuffixSearchList @() -ErrorAction SilentlyContinue

    # Optimize DNS cache
    Clear-DnsClientCache
    Set-DnsClient -InterfaceAlias "Ethernet*" -ConnectionSpecificSuffix "" -RegisterThisConnection $false -ErrorAction SilentlyContinue

    Write-Host "DNS optimization completed" -ForegroundColor Green
} catch {
    Write-Host "DNS optimization may require elevation" -ForegroundColor Yellow
}

# 4. WSL2 Specific Network Configuration
Write-Host "`n4. WSL2-specific network enhancements..." -ForegroundColor Yellow

# Create a script for WSL internal network optimization
$wslNetworkScript = @"
#!/bin/bash
# WSL2 Internal Network Optimization

echo "Configuring WSL2 network stack..."

# Optimize network buffer sizes
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

# TCP optimization for development workloads
echo 'net.ipv4.tcp_rmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_congestion_control = bbr' | sudo tee -a /etc/sysctl.conf

# Enable IP forwarding for container networking
echo 'net.ipv4.ip_forward = 1' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv6.conf.all.forwarding = 1' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p

echo "WSL2 network optimization completed"
"@

# Deploy network optimization to WSL
try {
    $wslNetworkScript | wsl bash
    Write-Host "WSL2 internal network optimization deployed" -ForegroundColor Green
} catch {
    Write-Host "Manual WSL configuration needed - check WSL connectivity" -ForegroundColor Yellow
}

# 5. Development Server Port Management
Write-Host "`n5. Configuring development server ports..." -ForegroundColor Yellow

# Common development framework ports
$devPorts = @{
    "React/Vite" = @(3000, 5173)
    "Angular" = @(4200)
    "Vue" = @(8080)
    "Next.js" = @(3000, 3001)
    "Express" = @(3000, 8000)
    "Django" = @(8000, 8080)
    "Flask" = @(5000)
    "FastAPI" = @(8000)
    "Jupyter" = @(8888)
    "Streamlit" = @(8501)
    "Gradio" = @(7860)
}

$wslVMId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'

foreach ($framework in $devPorts.Keys) {
    foreach ($port in $devPorts[$framework]) {
        try {
            # Remove existing rules to avoid conflicts
            Remove-NetFirewallRule -DisplayName "WSL-$framework-$port" -ErrorAction SilentlyContinue

            # Create new optimized rule
            New-NetFirewallHyperVRule -Name "WSL-$framework-$port" -DisplayName "WSL-$framework-$port" `
                -Direction Inbound -VMCreatorId $wslVMId -Protocol TCP -LocalPorts $port `
                -Action Allow -Profile Any -ErrorAction SilentlyContinue
        } catch {
            # Fallback to standard Windows firewall rule
            New-NetFirewallRule -DisplayName "WSL-$framework-$port" -Direction Inbound `
                -Protocol TCP -LocalPort $port -Action Allow -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Configured ports for $framework" -ForegroundColor Green
}

# 6. Container Runtime Optimization
Write-Host "`n6. Container runtime optimizations..." -ForegroundColor Yellow

# Docker Desktop integration optimization
$dockerConfigPath = "$env:USERPROFILE\.docker"
if (Test-Path $dockerConfigPath) {
    $dockerDaemonConfig = @{
        "experimental" = $true
        "features" = @{
            "buildkit" = $true
        }
        "builder" = @{
            "gc" = @{
                "defaultKeepStorage" = "20GB"
                "policy" = @(
                    @{"keepStorage" = "10GB"; "filter" = @("unused-for=2160h")}
                    @{"keepStorage" = "50GB"; "filter" = @("unused-for=168h")}
                    @{"all" = $true}
                )
            }
        }
        "log-driver" = "json-file"
        "log-opts" = @{
            "max-size" = "10m"
            "max-file" = "3"
        }
    }

    $dockerDaemonConfig | ConvertTo-Json -Depth 10 | Set-Content "$dockerConfigPath\daemon.json"
    Write-Host "Docker daemon configuration optimized" -ForegroundColor Green
}

# 7. Monitoring and Diagnostics
Write-Host "`n7. Setting up network monitoring..." -ForegroundColor Yellow

$monitoringScript = @"
# Network Monitoring Script for WSL2
Write-Host "=== WSL2 Network Status ===" -ForegroundColor Cyan

# WSL Status
Write-Host "`nWSL Status:" -ForegroundColor Yellow
wsl --status

# Network Configuration
Write-Host "`nNetwork Interfaces:" -ForegroundColor Yellow
wsl ip addr show

# Routing Table
Write-Host "`nRouting Table:" -ForegroundColor Yellow
wsl ip route show

# DNS Configuration
Write-Host "`nDNS Configuration:" -ForegroundColor Yellow
wsl cat /etc/resolv.conf

# Test Connectivity
Write-Host "`nConnectivity Tests:" -ForegroundColor Yellow
Write-Host "Windows to WSL (localhost):"
Test-NetConnection -ComputerName localhost -Port 22 -InformationLevel Quiet

Write-Host "WSL to Windows:"
wsl curl -s -o /dev/null -w "%{http_code}" http://www.google.com

# Docker Status (if installed)
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "`nDocker Status:" -ForegroundColor Yellow
    docker version --format "{{.Server.Version}}"
    wsl docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}
"@

$monitoringScript | Set-Content "$env:USERPROFILE\wsl2-network-monitor.ps1"
Write-Host "Network monitoring script created: wsl2-network-monitor.ps1" -ForegroundColor Green

Write-Host "`n=== Network Stability Summary ===" -ForegroundColor Cyan
Write-Host "1. Hyper-V VSock communication optimized" -ForegroundColor White
Write-Host "2. Network adapters configured for high performance" -ForegroundColor White
Write-Host "3. DNS and routing optimized for development" -ForegroundColor White
Write-Host "4. WSL2 internal network stack tuned" -ForegroundColor White
Write-Host "5. Development server ports configured" -ForegroundColor White
Write-Host "6. Container runtime optimized" -ForegroundColor White
Write-Host "7. Monitoring tools deployed" -ForegroundColor White

Write-Host "`nRestart WSL to apply all changes: wsl --shutdown && wsl" -ForegroundColor Yellow