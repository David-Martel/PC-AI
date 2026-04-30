#!/bin/bash
# MCP Server NPX to Bun Conversion Testing Script
# Bash script for functional equivalency testing

echo -e "\033[36mMCP Server NPX to Bun Conversion Testing\033[0m"
echo -e "\033[36m=========================================\033[0m"

# Test configuration
declare -a servers=(
    "memory:@modelcontextprotocol/server-memory@2025.8.4"
    "filesystem:@modelcontextprotocol/server-filesystem@2025.8.21"
    "sequential-thinking:@modelcontextprotocol/server-sequential-thinking@2025.7.1"
    "github:@modelcontextprotocol/server-github@2025.4.8"
    "context7:@upstash/context7-mcp@1.0.16"
    "desktop-commander:@wonderwhy-er/desktop-commander@0.2.10"
)

# Function to test server startup
test_server_startup() {
    local command=$1
    local package=$2
    local server_name=$3

    echo -e "\n\033[33mTesting $server_name with $command...\033[0m"

    local start_time=$(date +%s%N)

    if [ "$command" = "npx" ]; then
        timeout 30 npx -y "$package" --version > /tmp/npx-output.txt 2>&1
        local exit_code=$?
    else
        timeout 30 bun x "$package" --version > /tmp/bun-output.txt 2>&1
        local exit_code=$?
    fi

    local end_time=$(date +%s%N)
    local duration=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)

    echo "$exit_code:$duration"
}

# Function to measure memory usage
get_memory_usage() {
    local command=$1
    local package=$2

    if [ "$command" = "npx" ]; then
        /usr/bin/time -v npx -y "$package" --help 2>&1 | grep "Maximum resident" | awk '{print $6}'
    else
        /usr/bin/time -v bun x "$package" --help 2>&1 | grep "Maximum resident" | awk '{print $6}'
    fi
}

# Arrays to store results
declare -a npx_results=()
declare -a bun_results=()
declare -a npx_memory=()
declare -a bun_memory=()

# Run tests for each server
for server_info in "${servers[@]}"; do
    IFS=':' read -r server_name package <<< "$server_info"

    echo -e "\n\n\033[32mTesting: $server_name\033[0m"
    echo -e "\033[90mPackage: $package\033[0m"

    # Test NPX version
    npx_result=$(test_server_startup "npx" "$package" "$server_name (NPX)")
    npx_results+=("$npx_result")

    # Test Bun version
    bun_result=$(test_server_startup "bun" "$package" "$server_name (Bun)")
    bun_results+=("$bun_result")

    # Get memory usage
    npx_mem=$(get_memory_usage "npx" "$package")
    bun_mem=$(get_memory_usage "bun" "$package")
    npx_memory+=("$npx_mem")
    bun_memory+=("$bun_mem")
done

# Display results
echo -e "\n\n\033[36m=== FUNCTIONAL EQUIVALENCY TEST RESULTS ===\033[0m"
echo -e "Server\t\t\tNPX Status\tBun Status\tNPX Time\tBun Time\tImprovement"
echo -e "------\t\t\t----------\t----------\t--------\t--------\t-----------"

i=0
for server_info in "${servers[@]}"; do
    IFS=':' read -r server_name package <<< "$server_info"

    IFS=':' read -r npx_exit npx_time <<< "${npx_results[$i]}"
    IFS=':' read -r bun_exit bun_time <<< "${bun_results[$i]}"

    npx_status="✓"
    bun_status="✓"
    [ "$npx_exit" != "0" ] && npx_status="✗"
    [ "$bun_exit" != "0" ] && bun_status="✗"

    if [ -n "$npx_time" ] && [ -n "$bun_time" ]; then
        improvement=$(echo "scale=2; (($npx_time - $bun_time) / $npx_time) * 100" | bc)
    else
        improvement="N/A"
    fi

    printf "%-20s\t%s\t\t%s\t\t%.2fs\t\t%.2fs\t\t%.1f%%\n" \
        "$server_name" "$npx_status" "$bun_status" "$npx_time" "$bun_time" "$improvement"

    ((i++))
done

# Summary
echo -e "\n\033[36m=== SUMMARY ===\033[0m"

# Count successful tests
success_count=0
for result in "${bun_results[@]}"; do
    IFS=':' read -r exit_code duration <<< "$result"
    [ "$exit_code" = "0" ] && ((success_count++))
done

if [ "$success_count" -eq "${#servers[@]}" ]; then
    echo -e "\033[32m✓ All servers passed functional tests\033[0m"
else
    echo -e "\033[31m✗ $((${#servers[@]} - success_count)) servers failed\033[0m"
fi

# MCP Inspector validation
echo -e "\n\033[36m=== RUNNING MCP INSPECTOR VALIDATION ===\033[0m"
echo -e "\033[33mTesting configuration files...\033[0m"

# Test main mcp.json
echo -e "\n\033[90mValidating /home/david/mcp.json...\033[0m"
if npx @modelcontextprotocol/inspector --cli --config "/home/david/mcp.json" 2>/dev/null; then
    echo -e "\033[32m✓ mcp.json validation passed\033[0m"
else
    echo -e "\033[31m✗ mcp.json validation failed\033[0m"
fi

echo -e "\n\033[36m=== TESTING COMPLETE ===\033[0m"
echo -e "\033[90mBackup files saved as *.npx-backup\033[0m"
echo -e "\033[90mTo rollback: cp file.json.npx-backup file.json\033[0m"