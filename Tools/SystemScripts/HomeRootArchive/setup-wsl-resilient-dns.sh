#!/bin/bash
# WSL2 Resilient DNS Setup Script
# Run inside WSL: bash /mnt/c/Users/david/setup-wsl-resilient-dns.sh
#
# This script sets up dnsmasq as a local DNS resolver with caching
# and fallback capabilities for improved network resilience.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "=========================================="
echo "  WSL2 Resilient DNS Setup"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_error "Please do not run as root. Script will use sudo when needed."
    exit 1
fi

# Check if we're in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    log_error "This script should be run inside WSL"
    exit 1
fi

# Prompt for DNS resolver choice
echo "Select DNS resolver to install:"
echo "  1) dnsmasq (lightweight, recommended)"
echo "  2) unbound (full recursive resolver)"
echo "  3) systemd-resolved (built-in)"
echo ""
read -p "Choice [1]: " DNS_CHOICE
DNS_CHOICE=${DNS_CHOICE:-1}

case $DNS_CHOICE in
    1)
        DNS_RESOLVER="dnsmasq"
        ;;
    2)
        DNS_RESOLVER="unbound"
        ;;
    3)
        DNS_RESOLVER="systemd-resolved"
        ;;
    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

log_info "Setting up $DNS_RESOLVER..."

# Backup current configuration
log_info "Backing up current DNS configuration..."
sudo cp /etc/resolv.conf /etc/resolv.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# Remove immutable flag if set
sudo chattr -i /etc/resolv.conf 2>/dev/null || true

# Create wsl.conf if it doesn't exist
if [ ! -f /etc/wsl.conf ]; then
    log_info "Creating /etc/wsl.conf..."
    sudo tee /etc/wsl.conf > /dev/null << 'EOF'
[boot]
systemd=true

[network]
generateResolvConf=false
hostname=wsl-ubuntu

[automount]
enabled=true
options="metadata,umask=22,fmask=11"

[interop]
enabled=true
appendWindowsPath=true
EOF
    log_success "Created /etc/wsl.conf"
else
    # Update existing wsl.conf
    if ! grep -q "generateResolvConf=false" /etc/wsl.conf; then
        log_info "Updating /etc/wsl.conf..."
        if grep -q "\[network\]" /etc/wsl.conf; then
            sudo sed -i '/\[network\]/a generateResolvConf=false' /etc/wsl.conf
        else
            echo -e "\n[network]\ngenerateResolvConf=false" | sudo tee -a /etc/wsl.conf > /dev/null
        fi
        log_success "Updated /etc/wsl.conf"
    fi
fi

# Install and configure based on choice
case $DNS_RESOLVER in
    "dnsmasq")
        log_info "Installing dnsmasq..."
        sudo apt-get update
        sudo apt-get install -y dnsmasq

        # Stop systemd-resolved if running (conflicts on port 53)
        sudo systemctl stop systemd-resolved 2>/dev/null || true
        sudo systemctl disable systemd-resolved 2>/dev/null || true

        log_info "Configuring dnsmasq..."
        sudo tee /etc/dnsmasq.conf > /dev/null << 'EOF'
# WSL2 Resilient DNS Configuration

# Listen only on localhost
listen-address=127.0.0.1
bind-interfaces

# Port (default 53)
port=53

# Upstream DNS servers (ordered by preference)
# Google DNS
server=8.8.8.8
server=8.8.4.4

# Cloudflare DNS
server=1.1.1.1
server=1.0.0.1

# Quad9 DNS (with malware blocking)
server=9.9.9.9

# OpenDNS
server=208.67.222.222

# Cache settings for resilience
cache-size=10000
local-ttl=300
neg-ttl=60

# Don't poll /etc/resolv.conf for changes
no-poll

# Don't read /etc/resolv.conf
no-resolv

# Log queries (uncomment for debugging)
# log-queries
# log-facility=/var/log/dnsmasq.log

# DNSSEC validation (optional, may cause issues with some networks)
# dnssec
# trust-anchor=.,20326,8,2,E06D44B80B8F1D39A95C0B0D7C65D08458E880409BBC683457104237C7F8EC8D

# Expand simple hostnames
expand-hosts

# Don't forward plain names
domain-needed

# Never forward addresses in the non-routed address spaces
bogus-priv

# Add local-only domains (won't query upstream)
# local=/local/
# local=/internal/
EOF

        # Set up resolv.conf
        log_info "Setting up /etc/resolv.conf..."
        sudo rm -f /etc/resolv.conf
        echo "nameserver 127.0.0.1" | sudo tee /etc/resolv.conf > /dev/null
        echo "options edns0 trust-ad" | sudo tee -a /etc/resolv.conf > /dev/null
        sudo chattr +i /etc/resolv.conf

        # Enable and start dnsmasq
        sudo systemctl enable dnsmasq
        sudo systemctl restart dnsmasq

        log_success "dnsmasq configured and started"
        ;;

    "unbound")
        log_info "Installing unbound..."
        sudo apt-get update
        sudo apt-get install -y unbound unbound-anchor

        # Stop systemd-resolved if running
        sudo systemctl stop systemd-resolved 2>/dev/null || true
        sudo systemctl disable systemd-resolved 2>/dev/null || true

        log_info "Downloading root hints..."
        sudo curl -s -o /var/lib/unbound/root.hints https://www.internic.net/domain/named.cache

        log_info "Configuring unbound..."
        sudo tee /etc/unbound/unbound.conf.d/wsl-resilient.conf > /dev/null << 'EOF'
server:
    # Network interface
    interface: 127.0.0.1
    port: 53

    # Access control
    access-control: 127.0.0.0/8 allow
    access-control: ::1/128 allow

    # Performance tuning
    num-threads: 2
    msg-cache-size: 50m
    rrset-cache-size: 100m
    key-cache-size: 50m
    neg-cache-size: 10m

    # Cache TTL settings
    cache-min-ttl: 300
    cache-max-ttl: 86400
    infra-host-ttl: 900

    # DNSSEC validation
    auto-trust-anchor-file: "/var/lib/unbound/root.key"

    # Root hints for recursive resolution
    root-hints: "/var/lib/unbound/root.hints"

    # Security hardening
    hide-identity: yes
    hide-version: yes
    harden-glue: yes
    harden-dnssec-stripped: yes
    harden-below-nxdomain: yes
    harden-referral-path: yes

    # Private addresses (don't query upstream)
    private-address: 10.0.0.0/8
    private-address: 172.16.0.0/12
    private-address: 192.168.0.0/16
    private-address: 169.254.0.0/16
    private-address: fd00::/8
    private-address: fe80::/10

    # Performance optimizations
    prefetch: yes
    prefetch-key: yes
    minimal-responses: yes
    serve-expired: yes
    serve-expired-ttl: 86400

    # Logging (uncomment for debugging)
    # verbosity: 1
    # log-queries: yes
    # logfile: "/var/log/unbound.log"

# Forward zones for specific domains (optional)
# forward-zone:
#     name: "internal.corp"
#     forward-addr: 10.0.0.1
EOF

        # Initialize DNSSEC anchor
        log_info "Initializing DNSSEC trust anchor..."
        sudo unbound-anchor -a /var/lib/unbound/root.key 2>/dev/null || true

        # Set up resolv.conf
        log_info "Setting up /etc/resolv.conf..."
        sudo rm -f /etc/resolv.conf
        echo "nameserver 127.0.0.1" | sudo tee /etc/resolv.conf > /dev/null
        sudo chattr +i /etc/resolv.conf

        # Enable and start unbound
        sudo systemctl enable unbound
        sudo systemctl restart unbound

        log_success "unbound configured and started"
        ;;

    "systemd-resolved")
        log_info "Configuring systemd-resolved..."

        # Enable systemd-resolved
        sudo systemctl enable systemd-resolved
        sudo systemctl start systemd-resolved

        # Create custom configuration
        sudo mkdir -p /etc/systemd/resolved.conf.d
        sudo tee /etc/systemd/resolved.conf.d/wsl-resilient.conf > /dev/null << 'EOF'
[Resolve]
# Primary DNS servers
DNS=8.8.8.8 1.1.1.1 8.8.4.4 1.0.0.1

# Fallback DNS servers
FallbackDNS=9.9.9.9 149.112.112.112 208.67.222.222

# Search domains
Domains=~.

# DNSSEC validation
DNSSEC=allow-downgrade

# DNS over TLS (optional)
DNSOverTLS=opportunistic

# Cache settings
Cache=yes
CacheFromLocalhost=yes

# Multicast DNS
MulticastDNS=yes

# LLMNR
LLMNR=yes
EOF

        # Set up resolv.conf symlink
        log_info "Setting up /etc/resolv.conf..."
        sudo rm -f /etc/resolv.conf
        sudo ln -s /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

        sudo systemctl restart systemd-resolved

        log_success "systemd-resolved configured and started"
        ;;
esac

# Create network recovery script
log_info "Creating network recovery script..."
sudo tee /usr/local/bin/wsl-network-fix > /dev/null << 'EOF'
#!/bin/bash
# WSL Network Quick Fix Script

check_connectivity() {
    ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1
}

check_dns() {
    host -W 2 google.com > /dev/null 2>&1
}

echo "Checking network connectivity..."

if check_connectivity; then
    echo "[OK] Internet connectivity working"
else
    echo "[FAIL] No internet connectivity"
    echo "Try: wsl.exe --shutdown (from Windows) and restart WSL"
fi

if check_dns; then
    echo "[OK] DNS resolution working"
else
    echo "[FAIL] DNS resolution not working"

    # Try to fix DNS
    echo "Attempting DNS fix..."

    # Restart DNS service
    if systemctl is-active --quiet dnsmasq; then
        sudo systemctl restart dnsmasq
    elif systemctl is-active --quiet unbound; then
        sudo systemctl restart unbound
    elif systemctl is-active --quiet systemd-resolved; then
        sudo systemctl restart systemd-resolved
        resolvectl flush-caches
    fi

    sleep 2

    if check_dns; then
        echo "[OK] DNS fixed!"
    else
        echo "[FAIL] DNS still not working"
        echo ""
        echo "Manual fix options:"
        echo "1. Check /etc/resolv.conf"
        echo "2. Run: sudo chattr -i /etc/resolv.conf"
        echo "3. Run: echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf"
        echo "4. Restart WSL from Windows: wsl.exe --shutdown"
    fi
fi
EOF
sudo chmod +x /usr/local/bin/wsl-network-fix

# Create network monitoring script
log_info "Creating network monitoring script..."
sudo tee /usr/local/bin/wsl-network-monitor > /dev/null << 'EOF'
#!/bin/bash
# WSL Network Monitor - runs in background to detect and recover from network issues

LOG_FILE="/var/log/wsl-network-monitor.log"
CHECK_INTERVAL=${1:-60}  # Default: check every 60 seconds
FAIL_THRESHOLD=3

fail_count=0

log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

check_network() {
    ping -c 1 -W 3 8.8.8.8 > /dev/null 2>&1
}

recover_network() {
    log_msg "Attempting network recovery..."

    # Flush DNS cache
    if command -v resolvectl &> /dev/null; then
        resolvectl flush-caches 2>/dev/null
    fi

    # Restart local DNS
    if systemctl is-active --quiet dnsmasq; then
        systemctl restart dnsmasq 2>/dev/null
    elif systemctl is-active --quiet unbound; then
        systemctl restart unbound 2>/dev/null
    elif systemctl is-active --quiet systemd-resolved; then
        systemctl restart systemd-resolved 2>/dev/null
    fi
}

log_msg "Network monitor started (interval: ${CHECK_INTERVAL}s)"

while true; do
    if check_network; then
        if [ $fail_count -gt 0 ]; then
            log_msg "Network recovered after $fail_count failures"
        fi
        fail_count=0
    else
        ((fail_count++))
        log_msg "Network check failed (count: $fail_count/$FAIL_THRESHOLD)"

        if [ $fail_count -ge $FAIL_THRESHOLD ]; then
            recover_network
            fail_count=0
        fi
    fi

    sleep $CHECK_INTERVAL
done
EOF
sudo chmod +x /usr/local/bin/wsl-network-monitor

# Verify installation
echo ""
log_info "Verifying DNS configuration..."
echo ""

# Test DNS resolution
echo "Testing DNS resolution..."
if host -W 5 google.com > /dev/null 2>&1; then
    log_success "DNS resolution: Working"
else
    log_warn "DNS resolution: May need WSL restart"
fi

# Test internet connectivity
echo "Testing internet connectivity..."
if ping -c 1 -W 3 8.8.8.8 > /dev/null 2>&1; then
    log_success "Internet connectivity: Working"
else
    log_warn "Internet connectivity: May need WSL restart"
fi

# Show status
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration files created:"
echo "  - /etc/wsl.conf"
if [ "$DNS_RESOLVER" = "dnsmasq" ]; then
    echo "  - /etc/dnsmasq.conf"
elif [ "$DNS_RESOLVER" = "unbound" ]; then
    echo "  - /etc/unbound/unbound.conf.d/wsl-resilient.conf"
else
    echo "  - /etc/systemd/resolved.conf.d/wsl-resilient.conf"
fi
echo "  - /etc/resolv.conf (locked)"
echo ""
echo "Utility scripts created:"
echo "  - /usr/local/bin/wsl-network-fix"
echo "  - /usr/local/bin/wsl-network-monitor"
echo ""
echo "Next steps:"
echo "  1. Restart WSL: wsl.exe --shutdown (from Windows PowerShell)"
echo "  2. Start WSL again"
echo "  3. Test with: wsl-network-fix"
echo ""
echo "To monitor network: wsl-network-monitor &"
echo ""
