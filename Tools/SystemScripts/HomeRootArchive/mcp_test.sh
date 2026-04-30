#!/bin/bash
# MCP Server Test Suite
# Tests all configured MCP servers for basic availability and functionality

set -euo pipefail

# Configuration paths
CONFIG1_PATH="/c/Users/david/mcp.json"
CONFIG2_PATH="/c/Users/david/.claude/mcp.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results
declare -A results
total_tests=0
passed_tests=0
failed_tests=0
warned_tests=0

print_header() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

test_result() {
    local server_name="$1"
    local status="$2"
    local message="$3"

    total_tests=$((total_tests + 1))
    results[$server_name]="$status: $message"

    case "$status" in
        "PASS")
            echo -e "[${GREEN}PASS${NC}] $server_name: $message"
            passed_tests=$((passed_tests + 1))
            ;;
        "FAIL")
            echo -e "[${RED}FAIL${NC}] $server_name: $message"
            failed_tests=$((failed_tests + 1))
            ;;
        "WARN")
            echo -e "[${YELLOW}WARN${NC}] $server_name: $message"
            warned_tests=$((warned_tests + 1))
            ;;
    esac
}

test_command_exists() {
    local cmd="$1"
    if [[ "$cmd" == *.exe ]] || [[ "$cmd" == *:* ]]; then
        if [[ -f "$cmd" ]]; then
            return 0
        else
            return 1
        fi
    elif command -v "$cmd" &> /dev/null; then
        return 0
    elif [[ -f "/c/Users/david/.local/bin/$cmd" ]]; then
        return 0
    elif [[ -f "/c/Users/david/.local/bin/$cmd.exe" ]]; then
        return 0
    else
        return 1
    fi
}

test_rust_binary() {
    local binary_path="$1"
    local server_name="$2"

    if [[ ! -f "$binary_path" ]]; then
        test_result "$server_name" "FAIL" "Binary not found: $binary_path"
        return
    fi

    # Try different help flags
    for flag in "--help" "--version" "-h" "-V"; do
        if timeout 5s "$binary_path" "$flag" &>/dev/null; then
            test_result "$server_name" "PASS" "Rust binary functional"
            return
        fi
    done

    test_result "$server_name" "WARN" "Binary exists but may not be functional"
}

test_python_script() {
    local script_path="$1"
    local server_name="$2"

    if [[ ! -f "$script_path" ]]; then
        test_result "$server_name" "FAIL" "Python script not found: $script_path"
        return
    fi

    # Test syntax
    if uv run python -m py_compile "$script_path" &>/dev/null; then
        test_result "$server_name" "PASS" "Python script syntax valid"
    else
        test_result "$server_name" "FAIL" "Python script syntax error"
    fi
}

test_mcp_server() {
    local server_name="$1"
    local command="$2"
    local args_json="$3"

    echo "Testing server: $server_name"
    echo "Command: $command"

    # Test command availability
    if ! test_command_exists "$command"; then
        test_result "$server_name" "FAIL" "Command not found: $command"
        return
    fi

    # Parse args if they exist
    local args=()
    if [[ -n "$args_json" ]] && [[ "$args_json" != "null" ]]; then
        while IFS= read -r line; do
            args+=("$line")
        done < <(echo "$args_json" | jq -r '.[]?' 2>/dev/null || echo "")
    fi

    # Server-specific tests
    case "$command" in
        "uv")
            if [[ "${args[0]}" == "run" && "${args[1]}" == "python" ]]; then
                if [[ "${args[2]}" == "-m" ]]; then
                    test_result "$server_name" "PASS" "Python module command ready"
                elif [[ "${args[2]}" == *.py ]]; then
                    test_python_script "${args[2]}" "$server_name"
                else
                    test_result "$server_name" "WARN" "Python command format unclear"
                fi
            else
                test_result "$server_name" "WARN" "Non-standard uv command"
            fi
            ;;
        *.exe)
            test_rust_binary "$command" "$server_name"
            ;;
        "bun"|"npx"|"node")
            test_result "$server_name" "PASS" "Node.js runtime available"
            ;;
        *)
            test_result "$server_name" "PASS" "Command available"
            ;;
    esac
}

# Main test execution
print_header "MCP Server Test Suite"

echo "Testing MCP configurations:"
echo "  - $CONFIG1_PATH"
echo "  - $CONFIG2_PATH"

# Test each configuration file
for config_path in "$CONFIG1_PATH" "$CONFIG2_PATH"; do
    if [[ ! -f "$config_path" ]]; then
        echo -e "${RED}Configuration not found: $config_path${NC}"
        continue
    fi

    config_name=$(basename "$config_path")
    print_header "Testing Configuration: $config_name"

    # Extract server configurations
    servers=$(jq -r '.mcpServers | keys[]' "$config_path" 2>/dev/null || echo "")

    if [[ -z "$servers" ]]; then
        echo -e "${YELLOW}No servers found in $config_path${NC}"
        continue
    fi

    while IFS= read -r server_name; do
        if [[ -z "$server_name" ]]; then
            continue
        fi

        command=$(jq -r ".mcpServers.\"$server_name\".command" "$config_path")
        args_json=$(jq -c ".mcpServers.\"$server_name\".args // null" "$config_path")

        echo ""
        test_mcp_server "$server_name" "$command" "$args_json"
    done <<< "$servers"
done

# Special integration tests
print_header "Special Integration Tests"

# Test ast-grep integration
echo "Testing ast-grep integration..."
AST_GREP_PATH="/c/Users/david/.local/bin/sg.exe"
if [[ -f "$AST_GREP_PATH" ]]; then
    if timeout 5s "$AST_GREP_PATH" --version &>/dev/null; then
        version=$("$AST_GREP_PATH" --version 2>/dev/null | head -1)
        test_result "ast-grep-integration" "PASS" "ast-grep ready - $version"
    else
        test_result "ast-grep-integration" "FAIL" "ast-grep binary not functional"
    fi
else
    test_result "ast-grep-integration" "FAIL" "ast-grep binary not found: $AST_GREP_PATH"
fi

# Test key Rust binaries
echo ""
echo "Testing Rust MCP binaries..."
rust_binaries=(
    "rust-fs:/c/Users/david/.local/bin/rust-fs.exe"
    "rust-fetch:/c/Users/david/.local/bin/rust-fetch.exe"
    "rust-link:/c/Users/david/.local/bin/rust-link.exe"
    "rust-sequential-thinking:/c/Users/david/.local/bin/rust-sequential-thinking.exe"
)

for entry in "${rust_binaries[@]}"; do
    IFS=':' read -r name path <<< "$entry"
    echo ""
    test_rust_binary "$path" "$name-binary"
done

# Test essential commands
echo ""
echo "Testing essential dependencies..."
essential_commands=("bun" "uv" "python" "node")
for cmd in "${essential_commands[@]}"; do
    if test_command_exists "$cmd"; then
        test_result "$cmd-dependency" "PASS" "Essential command available"
    else
        test_result "$cmd-dependency" "FAIL" "Essential command missing: $cmd"
    fi
done

# Final summary
print_header "Test Summary"
echo "Total tests: $total_tests"
echo -e "✓ Passed: ${GREEN}$passed_tests${NC}"
echo -e "✗ Failed: ${RED}$failed_tests${NC}"
echo -e "⚠ Warnings: ${YELLOW}$warned_tests${NC}"

success_rate=$(( passed_tests * 100 / total_tests ))
echo "Success rate: $success_rate%"

# Detailed results
echo ""
echo "Detailed Results:"
echo "=================="
for server in "${!results[@]}"; do
    status_msg="${results[$server]}"
    case "$status_msg" in
        PASS:*)
            echo -e "${GREEN}✓${NC} $server: ${status_msg#PASS: }"
            ;;
        FAIL:*)
            echo -e "${RED}✗${NC} $server: ${status_msg#FAIL: }"
            ;;
        WARN:*)
            echo -e "${YELLOW}⚠${NC} $server: ${status_msg#WARN: }"
            ;;
    esac
done

# Recommendations
if [[ $failed_tests -gt 0 ]]; then
    echo ""
    print_header "Recommendations"
    echo "Failed servers require attention:"
    for server in "${!results[@]}"; do
        if [[ "${results[$server]}" == FAIL:* ]]; then
            echo "  - $server: ${results[$server]#FAIL: }"
        fi
    done
fi

echo ""
echo "Test completed at $(date)"