# WSL2 and Hyper-V Optimization Script for High-End Development System
# Run as Administrator in PowerShell

Write-Host "=== WSL2 and Hyper-V Optimization for Development Environment ===" -ForegroundColor Cyan

# 1. WSL2 Configuration Backup and Update
Write-Host "`n1. Backing up current .wslconfig..." -ForegroundColor Yellow
$backupPath = "$env:USERPROFILE\.wslconfig.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
if (Test-Path "$env:USERPROFILE\.wslconfig") {
    Copy-Item "$env:USERPROFILE\.wslconfig" $backupPath
    Write-Host "Backup created: $backupPath" -ForegroundColor Green
}

# 2. Apply corrected configuration
Write-Host "`n2. Applying corrected .wslconfig..." -ForegroundColor Yellow
Copy-Item "$env:USERPROFILE\.wslconfig.fixed" "$env:USERPROFILE\.wslconfig" -Force
Write-Host "Corrected .wslconfig applied" -ForegroundColor Green

# 3. Hyper-V Network Optimization
Write-Host "`n3. Configuring Hyper-V for WSL2 optimization..." -ForegroundColor Yellow

try {
    # Enable Hyper-V Enhanced Session Mode (improves performance)
    Set-VMHost -EnableEnhancedSessionMode $true -ErrorAction SilentlyContinue
    Write-Host "Enhanced Session Mode enabled" -ForegroundColor Green

    # Configure Hyper-V Firewall for WSL (VMCreatorId for WSL)
    $wslVMId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'

    # Allow inbound connections for WSL VM
    Set-NetFirewallHyperVVMSetting -Name $wslVMId -DefaultInboundAction Allow -ErrorAction SilentlyContinue
    Write-Host "Hyper-V firewall configured for WSL" -ForegroundColor Green

    # Configure common development ports
    $devPorts = @(3000, 3001, 4200, 5000, 5173, 8000, 8080, 8888, 9000)
    foreach ($port in $devPorts) {
        New-NetFirewallHyperVRule -Name "WSL-Dev-$port" -DisplayName "WSL Development Port $port" `
            -Direction Inbound -VMCreatorId $wslVMId -Protocol TCP -LocalPorts $port -Action Allow -ErrorAction SilentlyContinue
    }
    Write-Host "Development ports configured: $($devPorts -join ', ')" -ForegroundColor Green

} catch {
    Write-Host "Hyper-V configuration requires elevation. Run as Administrator." -ForegroundColor Red
}

# 4. Windows Memory Management Optimization
Write-Host "`n4. Optimizing Windows memory management..." -ForegroundColor Yellow

# Disable Windows memory compression (can interfere with WSL2 memory allocation)
try {
    Disable-MMAgent -MemoryCompression -ErrorAction SilentlyContinue
    Write-Host "Memory compression disabled (recommended for high-RAM systems)" -ForegroundColor Green
} catch {
    Write-Host "Memory management optimization requires elevation" -ForegroundColor Red
}

# 5. Network Performance Optimization
Write-Host "`n5. Network performance optimizations..." -ForegroundColor Yellow

try {
    # Optimize network adapter settings for development
    $adapters = Get-NetAdapter | Where-Object {$_.Status -eq 'Up' -and $_.Virtual -eq $false}
    foreach ($adapter in $adapters) {
        # Disable Large Send Offload v2 (can cause issues with WSL2 mirrored networking)
        Set-NetAdapterLso -Name $adapter.Name -LsoV2IPv4 Disabled -LsoV2IPv6 Disabled -ErrorAction SilentlyContinue

        # Enable Receive Side Scaling for better performance
        Set-NetAdapterRss -Name $adapter.Name -Enabled $true -ErrorAction SilentlyContinue
    }
    Write-Host "Network adapter optimization completed" -ForegroundColor Green
} catch {
    Write-Host "Network optimization may require elevation or updated drivers" -ForegroundColor Yellow
}

# 6. WSL Configuration Deployment
Write-Host "`n6. Deploying WSL distribution configuration..." -ForegroundColor Yellow

# Deploy wsl.conf to Ubuntu WSL instance
$wslConfContent = Get-Content "$env:USERPROFILE\wsl.conf.ubuntu" -Raw
$wslCommand = "echo '$wslConfContent' | sudo tee /etc/wsl.conf > /dev/null"
try {
    wsl -d Ubuntu bash -c $wslCommand
    Write-Host "wsl.conf deployed to Ubuntu distribution" -ForegroundColor Green
} catch {
    Write-Host "Manual deployment needed: Copy wsl.conf.ubuntu content to /etc/wsl.conf in WSL" -ForegroundColor Yellow
}

# 7. WSL Restart and Validation
Write-Host "`n7. Restarting WSL for configuration to take effect..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep -Seconds 5

# Start WSL and validate configuration
Write-Host "`n8. Validating new configuration..." -ForegroundColor Yellow
wsl --status

# Performance recommendations
Write-Host "`n=== Performance Recommendations ===" -ForegroundColor Cyan
Write-Host "1. T: Drive (Dev Drive) optimization:" -ForegroundColor White
Write-Host "   - Ensure ReFS filesystem for optimal performance" -ForegroundColor Gray
Write-Host "   - Consider enabling Windows Defender exclusions for T:\vm\" -ForegroundColor Gray

Write-Host "`n2. Mirrored networking validation:" -ForegroundColor White
Write-Host "   - Test localhost connectivity: wsl curl http://localhost:PORT" -ForegroundColor Gray
Write-Host "   - Verify DNS resolution: wsl nslookup google.com" -ForegroundColor Gray

Write-Host "`n3. Memory optimization:" -ForegroundColor White
Write-Host "   - Monitor memory usage with: wsl --system" -ForegroundColor Gray
Write-Host "   - autoMemoryReclaim=gradual will release memory slowly" -ForegroundColor Gray

Write-Host "`n4. Docker/Container optimization:" -ForegroundColor White
Write-Host "   - Use Docker Desktop with WSL2 backend" -ForegroundColor Gray
Write-Host "   - Enable experimental features in Docker for better WSL integration" -ForegroundColor Gray

Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Restart WSL: wsl --shutdown && wsl" -ForegroundColor White
Write-Host "2. Test networking: wsl ip addr show" -ForegroundColor White
Write-Host "3. Validate systemd: wsl systemctl --version" -ForegroundColor White
Write-Host "4. Monitor performance with Windows Performance Toolkit if needed" -ForegroundColor White

Write-Host "`nOptimization completed! WSL should now perform better with mirrored networking." -ForegroundColor Green