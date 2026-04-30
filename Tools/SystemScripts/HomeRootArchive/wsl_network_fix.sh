#!/bin/bash
# WSL Networking Workaround Script
# This manually configures networking when automatic configuration fails

echo "=== Current Network State ==="
ip addr show
echo ""
echo "=== Current Routes ==="
ip route show
echo ""

# Get the Windows host IP from resolv.conf or use WSL's typical gateway
WINDOWS_HOST=$(ip route show | grep -i 'default via' | awk '{print $3}' 2>/dev/null)

if [ -z "$WINDOWS_HOST" ]; then
    # Try to get from nameserver in resolv.conf
    WINDOWS_HOST=$(grep nameserver /etc/resolv.conf | head -1 | awk '{print $2}')
fi

if [ -z "$WINDOWS_HOST" ]; then
    echo "ERROR: Could not determine Windows host IP"
    echo "Attempting to use common WSL gateway: 172.17.0.1"
    WINDOWS_HOST="172.17.0.1"
fi

echo "Using Windows host IP: $WINDOWS_HOST"
echo ""

# Find the primary interface (usually eth0 or similar, excluding docker bridges)
PRIMARY_IFACE=$(ip link show | grep -E '^[0-9]+: (eth|enp)' | head -1 | awk -F': ' '{print $2}')

if [ -z "$PRIMARY_IFACE" ]; then
    echo "ERROR: Could not find primary network interface"
    exit 1
fi

echo "Primary interface: $PRIMARY_IFACE"

# Bring up the interface if it's down
echo "Bringing up $PRIMARY_IFACE..."
sudo ip link set $PRIMARY_IFACE up

# Check if we already have a default route
if ip route show | grep -q 'default'; then
    echo "Default route already exists, removing old route..."
    sudo ip route del default
fi

# Add default route through Windows host
echo "Adding default route via $WINDOWS_HOST..."
sudo ip route add default via $WINDOWS_HOST dev $PRIMARY_IFACE

# Update DNS to use Windows host if needed
if ! grep -q "$WINDOWS_HOST" /etc/resolv.conf; then
    echo "Updating /etc/resolv.conf to use Windows host as DNS..."
    echo "nameserver $WINDOWS_HOST" | sudo tee /etc/resolv.conf > /dev/null
    echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf > /dev/null
    echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf > /dev/null
fi

echo ""
echo "=== New Network Configuration ==="
ip addr show $PRIMARY_IFACE
echo ""
echo "=== New Routes ==="
ip route show
echo ""
echo "=== Testing Connectivity ==="
ping -c 2 $WINDOWS_HOST && echo "✓ Can reach Windows host" || echo "✗ Cannot reach Windows host"
ping -c 2 8.8.8.8 && echo "✓ Can reach internet" || echo "✗ Cannot reach internet"
ping -c 2 google.com && echo "✓ DNS resolution works" || echo "✗ DNS resolution failed"
