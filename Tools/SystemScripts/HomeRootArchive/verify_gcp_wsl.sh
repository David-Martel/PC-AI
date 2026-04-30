#!/bin/bash
# Quick verification script for GCP and WSL integration

echo "===================================="
echo "GCP & WSL Integration Verification"
echo "===================================="

echo -e "\n[GCP Authentication Status]"
echo "Current Profile: $(cat ~/.gcp/current-profile.txt 2>/dev/null || echo 'Not set')"
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Credentials: ${GOOGLE_APPLICATION_CREDENTIALS:-Not set}"

if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✓ Credentials file exists"
else
    # Try Windows path
    WIN_CREDS="${GOOGLE_APPLICATION_CREDENTIALS//\/home\/david/C:\/Users\/david}"
    if [ -f "$WIN_CREDS" ]; then
        echo "✓ Credentials accessible via Windows path"
    else
        echo "✗ Credentials file not found"
    fi
fi

echo -e "\n[WSL Integration]"
if [ -d "//wsl.localhost/Ubuntu/home/david" ]; then
    echo "✓ WSL filesystem accessible"
else
    echo "✗ WSL filesystem not accessible"
fi

echo -e "\n[MCP Servers]"
ls -1 ~/.local/bin/rust*.exe 2>/dev/null | wc -l | xargs -I {} echo "Rust servers: {}"
ls -1 ~/.local/bin/python*.exe 2>/dev/null | wc -l | xargs -I {} echo "Python executables: {}"

echo -e "\n[Environment Variables]"
env | grep -E "^(GOOGLE|GCP|VERTEX)" | head -5

echo -e "\nVerification complete!"
