#!/bin/bash
# Manual WSL Network Configuration
# This requires sudo access

echo "=== Manual WSL Network Fix ==="
echo "This script will configure WSL networking manually."
echo "You will be prompted for your sudo password."
echo ""

# Get the Windows gateway from vEthernet adapter
echo "Detecting Windows network gateway..."
WIN_GATEWAY=$(powershell.exe -Command "(Get-NetIPAddress -InterfaceAlias 'vEthernet (Default Switch)' -AddressFamily IPv4).IPAddress" 2>/dev/null | tr -d '\r')

if [ -z "$WIN_GATEWAY" ]; then
    echo "Warning: Could not detect Default Switch gateway. Using fallback 172.21.16.1"
    WIN_GATEWAY="172.21.16.1"
else
    echo "Found Windows gateway: $WIN_GATEWAY"
fi

# Extract subnet (first 3 octets)
SUBNET=$(echo $WIN_GATEWAY | cut -d'.' -f1-3)
WSL_IP="${SUBNET}.100"

echo "Will configure WSL with:"
echo "  IP: $WSL_IP/24"
echo "  Gateway: $WIN_GATEWAY"
echo ""

# Get current interface name
INTERFACE=$(ip -o link show | grep -E 'enP[0-9]+p0s0' | awk -F': ' '{print $2}')

if [ -z "$INTERFACE" ]; then
    echo "ERROR: Could not find WSL network interface!"
    exit 1
fi

echo "Found interface: $INTERFACE"
echo ""
echo "Applying configuration (requires sudo)..."
echo ""

# Bring interface up
echo "1. Bringing interface UP..."
sudo ip link set "$INTERFACE" up
if [ $? -ne 0 ]; then
    echo "   ERROR: Failed to bring interface up"
    exit 1
fi
echo "   ✅ Interface is UP"

# Flush old IP if exists
echo "2. Flushing old IP addresses..."
sudo ip addr flush dev "$INTERFACE"
echo "   ✅ Flushed"

# Assign new IP
echo "3. Assigning IP $WSL_IP/24..."
sudo ip addr add "$WSL_IP/24" dev "$INTERFACE"
if [ $? -ne 0 ]; then
    echo "   ERROR: Failed to assign IP"
    exit 1
fi
echo "   ✅ IP assigned"

# Add default route
echo "4. Adding default route via $WIN_GATEWAY..."
sudo ip route add default via "$WIN_GATEWAY" dev "$INTERFACE"
if [ $? -ne 0 ]; then
    echo "   ERROR: Failed to add default route"
    exit 1
fi
echo "   ✅ Default route added"

# Update DNS
echo "5. Updating DNS configuration..."
echo "nameserver $WIN_GATEWAY" | sudo tee /etc/resolv.conf > /dev/null
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf > /dev/null
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf > /dev/null
echo "   ✅ DNS configured"

echo ""
echo "=== Testing Connectivity ==="
echo ""

echo "Testing gateway ($WIN_GATEWAY)..."
if ping -c 2 -W 2 "$WIN_GATEWAY" > /dev/null 2>&1; then
    echo "   ✅ Gateway reachable"
else
    echo "   ❌ Gateway unreachable"
fi

echo "Testing external IP (8.8.8.8)..."
if ping -c 2 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "   ✅ Internet reachable"
else
    echo "   ❌ Internet unreachable"
fi

echo "Testing DNS resolution (google.com)..."
if ping -c 2 -W 2 google.com > /dev/null 2>&1; then
    echo "   ✅ DNS working"
else
    echo "   ❌ DNS not working"
fi

echo ""
echo "=== Configuration Complete ==="
echo "Current network state:"
ip addr show "$INTERFACE"
echo ""
echo "Current routes:"
ip route show
echo ""
echo "Note: This configuration will be lost on WSL restart."
echo "If successful, we'll create a permanent solution."