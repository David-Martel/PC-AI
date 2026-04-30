#!/bin/bash

# DNSmasq WSL Implementation Script
# Complete setup for DNS caching and custom domain resolution in WSL
# Created for WSL Ubuntu environment

set -e
set -u

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Run as regular user with sudo access."
    fi
}

# Backup existing configuration
backup_config() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="/home/$USER/dns_backup_$timestamp"

    log "Creating backup directory: $backup_dir"
    mkdir -p "$backup_dir"

    # Backup existing files
    if [[ -f /etc/resolv.conf ]]; then
        sudo cp /etc/resolv.conf "$backup_dir/resolv.conf.bak"
        log "Backed up /etc/resolv.conf"
    fi

    if [[ -f /etc/wsl.conf ]]; then
        sudo cp /etc/wsl.conf "$backup_dir/wsl.conf.bak"
        log "Backed up /etc/wsl.conf"
    fi

    if [[ -f /etc/dnsmasq.conf ]]; then
        sudo cp /etc/dnsmasq.conf "$backup_dir/dnsmasq.conf.bak"
        log "Backed up existing /etc/dnsmasq.conf"
    fi

    echo "$backup_dir" > /tmp/dnsmasq_backup_location
    log "Backup location saved to /tmp/dnsmasq_backup_location"
}

# Install dnsmasq
install_dnsmasq() {
    log "Updating package list..."
    sudo apt-get update -q

    log "Installing dnsmasq..."
    sudo apt-get install -y dnsmasq dnsutils

    # Stop dnsmasq for configuration
    sudo systemctl stop dnsmasq
    sudo systemctl disable dnsmasq
    log "DNSmasq installed and temporarily disabled for configuration"
}

# Configure WSL to not generate resolv.conf
configure_wsl() {
    log "Configuring WSL to disable automatic resolv.conf generation..."

    # Create or update /etc/wsl.conf
    sudo tee /etc/wsl.conf > /dev/null << 'EOF'
[boot]
systemd=true

[network]
generateResolvConf=false

[interop]
enabled=true
appendWindowsPath=true

[user]
default=david
EOF

    log "WSL configuration updated. WSL restart will be required."
}

# Disable systemd-resolved
disable_systemd_resolved() {
    log "Disabling systemd-resolved..."

    # Check if systemd-resolved is running
    if systemctl is-active --quiet systemd-resolved; then
        warn "systemd-resolved is currently running. Stopping and disabling..."
        sudo systemctl stop systemd-resolved
        sudo systemctl disable systemd-resolved

        # Remove the stub resolver symlink if it exists
        if [[ -L /etc/resolv.conf ]]; then
            sudo rm /etc/resolv.conf
            log "Removed systemd-resolved stub symlink"
        fi
    else
        info "systemd-resolved is not running"
    fi
}

# Configure dnsmasq
configure_dnsmasq() {
    log "Configuring dnsmasq..."

    # Create dnsmasq configuration
    sudo tee /etc/dnsmasq.conf > /dev/null << 'EOF'
# DNSmasq Configuration for WSL
# Listen only on localhost
listen-address=127.0.0.1
bind-interfaces

# DNS port
port=53

# Cache size (default 150, increased for better performance)
cache-size=1000

# DNS upstream servers (Cloudflare and Google)
server=1.1.1.1
server=1.0.0.1
server=8.8.8.8
server=8.8.4.4

# Enable DNSSEC validation
dnssec

# Log queries for debugging (comment out in production)
log-queries
log-facility=/var/log/dnsmasq.log

# Custom domain mappings for WSL and Docker
address=/.wsl.localhost/127.0.0.1
address=/.docker.internal/127.0.0.1
address=/.local.dev/127.0.0.1

# Expand hosts file
expand-hosts
domain=wsl.local

# Local domain resolution
local=/wsl.local/
domain-needed
bogus-priv

# DHCP range (disabled by default, uncomment if needed)
# dhcp-range=192.168.1.50,192.168.1.150,12h

# Read additional config files
conf-dir=/etc/dnsmasq.d/,*.conf

# Performance tuning
no-negcache
strict-order
EOF

    log "DNSmasq configuration created"
}

# Create custom domain configurations
create_custom_domains() {
    log "Creating custom domain configurations..."

    sudo mkdir -p /etc/dnsmasq.d

    # WSL-specific domains
    sudo tee /etc/dnsmasq.d/wsl-domains.conf > /dev/null << 'EOF'
# WSL-specific domain mappings
address=/localhost.wsl/127.0.0.1
address=/host.docker.internal/host-gateway
address=/gateway.docker.internal/host-gateway

# Development domains
address=/dev.local/127.0.0.1
address=/test.local/127.0.0.1
address=/staging.local/127.0.0.1

# Custom project domains
address=/myproject.wsl/127.0.0.1
address=/api.wsl/127.0.0.1
address=/web.wsl/127.0.0.1
EOF

    log "Custom domain configurations created"
}

# Configure resolv.conf
configure_resolv_conf() {
    log "Configuring /etc/resolv.conf..."

    # Create new resolv.conf pointing to local dnsmasq
    sudo tee /etc/resolv.conf > /dev/null << 'EOF'
# DNSmasq local resolver
nameserver 127.0.0.1

# Fallback DNS servers (in case dnsmasq fails)
nameserver 1.1.1.1
nameserver 8.8.8.8

# Search domains
search wsl.local dtmventures.com
EOF

    # Make resolv.conf immutable to prevent WSL from overwriting
    sudo chattr +i /etc/resolv.conf

    log "resolv.conf configured and protected from automatic changes"
}

# Create dnsmasq log directory and set permissions
setup_logging() {
    log "Setting up logging..."

    sudo touch /var/log/dnsmasq.log
    sudo chown dnsmasq:nogroup /var/log/dnsmasq.log
    sudo chmod 644 /var/log/dnsmasq.log

    # Setup log rotation
    sudo tee /etc/logrotate.d/dnsmasq > /dev/null << 'EOF'
/var/log/dnsmasq.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    postrotate
        systemctl reload dnsmasq
    endscript
}
EOF

    log "Logging configured with rotation"
}

# Start and enable dnsmasq
start_dnsmasq() {
    log "Starting and enabling dnsmasq..."

    # Test configuration first
    if ! sudo dnsmasq --test; then
        error "DNSmasq configuration test failed"
    fi

    sudo systemctl enable dnsmasq
    sudo systemctl start dnsmasq

    # Check status
    if systemctl is-active --quiet dnsmasq; then
        log "DNSmasq started successfully"
    else
        error "Failed to start DNSmasq"
    fi
}

# Test DNS resolution
test_dns() {
    log "Testing DNS resolution..."

    info "Testing external domain resolution..."
    if nslookup google.com 127.0.0.1 > /dev/null 2>&1; then
        log "✓ External domain resolution working"
    else
        warn "✗ External domain resolution failed"
    fi

    info "Testing custom domain resolution..."
    if nslookup test.wsl.localhost 127.0.0.1 > /dev/null 2>&1; then
        log "✓ Custom domain resolution working"
    else
        warn "✗ Custom domain resolution failed"
    fi

    info "Testing localhost resolution..."
    if ping -c 1 localhost > /dev/null 2>&1; then
        log "✓ Localhost resolution working"
    else
        warn "✗ Localhost resolution failed"
    fi
}

# Display status and next steps
show_status() {
    log "DNSmasq setup completed!"

    echo -e "\n${BLUE}=== DNS Configuration Status ===${NC}"
    echo "• DNSmasq: $(systemctl is-active dnsmasq)"
    echo "• systemd-resolved: $(systemctl is-active systemd-resolved || echo 'disabled')"
    echo "• DNS Server: 127.0.0.1 (local dnsmasq)"
    echo "• Upstream: 1.1.1.1, 8.8.8.8"
    echo "• Cache Size: 1000 entries"

    echo -e "\n${BLUE}=== Custom Domains ===${NC}"
    echo "• *.wsl.localhost → 127.0.0.1"
    echo "• *.docker.internal → 127.0.0.1"
    echo "• *.local.dev → 127.0.0.1"
    echo "• localhost.wsl → 127.0.0.1"

    echo -e "\n${BLUE}=== Useful Commands ===${NC}"
    echo "• Check DNS status: sudo systemctl status dnsmasq"
    echo "• View DNS logs: sudo tail -f /var/log/dnsmasq.log"
    echo "• Test resolution: nslookup domain.wsl.localhost 127.0.0.1"
    echo "• Reload config: sudo systemctl reload dnsmasq"
    echo "• Edit config: sudo nano /etc/dnsmasq.conf"

    if [[ -f /tmp/dnsmasq_backup_location ]]; then
        local backup_dir=$(cat /tmp/dnsmasq_backup_location)
        echo -e "\n${YELLOW}=== Backup Location ===${NC}"
        echo "Configuration backup saved to: $backup_dir"
    fi

    echo -e "\n${YELLOW}=== Important Notes ===${NC}"
    echo "• WSL restart recommended: 'wsl --shutdown' then restart"
    echo "• Test all custom domains after WSL restart"
    echo "• Monitor /var/log/dnsmasq.log for any issues"
    echo "• To rollback, restore files from backup directory"
}

# Rollback function
rollback() {
    if [[ ! -f /tmp/dnsmasq_backup_location ]]; then
        error "No backup location found. Cannot rollback."
    fi

    local backup_dir=$(cat /tmp/dnsmasq_backup_location)

    log "Rolling back DNS configuration..."

    # Stop dnsmasq
    sudo systemctl stop dnsmasq || true
    sudo systemctl disable dnsmasq || true

    # Restore files
    if [[ -f "$backup_dir/resolv.conf.bak" ]]; then
        sudo chattr -i /etc/resolv.conf 2>/dev/null || true
        sudo cp "$backup_dir/resolv.conf.bak" /etc/resolv.conf
        log "Restored /etc/resolv.conf"
    fi

    if [[ -f "$backup_dir/wsl.conf.bak" ]]; then
        sudo cp "$backup_dir/wsl.conf.bak" /etc/wsl.conf
        log "Restored /etc/wsl.conf"
    fi

    # Re-enable systemd-resolved
    sudo systemctl enable systemd-resolved
    sudo systemctl start systemd-resolved

    log "Rollback completed. WSL restart recommended."
}

# Main execution
main() {
    echo -e "${BLUE}DNSmasq WSL Setup Script${NC}"
    echo "=========================="

    case "${1:-install}" in
        "install")
            check_root
            backup_config
            install_dnsmasq
            configure_wsl
            disable_systemd_resolved
            configure_dnsmasq
            create_custom_domains
            configure_resolv_conf
            setup_logging
            start_dnsmasq
            test_dns
            show_status
            ;;
        "test")
            test_dns
            ;;
        "status")
            show_status
            ;;
        "rollback")
            rollback
            ;;
        *)
            echo "Usage: $0 [install|test|status|rollback]"
            echo "  install  - Complete DNSmasq setup (default)"
            echo "  test     - Test DNS resolution"
            echo "  status   - Show current status"
            echo "  rollback - Restore previous configuration"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"