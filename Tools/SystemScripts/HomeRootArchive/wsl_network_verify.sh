#!/bin/bash
# WSL Network Verification Script
# Run this after HNS network reset to verify connectivity

echo "=== WSL Network Verification ==="
echo ""

echo "1. Network Interface Status:"
ip addr show | grep -A 8 "^[0-9]:"
echo ""

echo "2. Default Route:"
ip route show
echo ""

echo "3. DNS Configuration:"
cat /etc/resolv.conf
echo ""

echo "4. Connectivity Tests:"
echo "   Testing gateway ping..."
if ip route show | grep -q default; then
    GATEWAY=$(ip route show | grep default | awk '{print $3}')
    ping -c 2 -W 2 "$GATEWAY" && echo "   ✅ Gateway reachable" || echo "   ❌ Gateway unreachable"
else
    echo "   ❌ No default gateway configured"
fi
echo ""

echo "   Testing external IP (8.8.8.8)..."
ping -c 2 -W 2 8.8.8.8 && echo "   ✅ Internet reachable" || echo "   ❌ Internet unreachable"
echo ""

echo "   Testing DNS resolution..."
ping -c 2 -W 2 google.com && echo "   ✅ DNS working" || echo "   ❌ DNS not working"
echo ""

echo "5. Docker Containers (if running):"
docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "   Docker not running or not accessible"
echo ""

echo "=== Verification Complete ==="