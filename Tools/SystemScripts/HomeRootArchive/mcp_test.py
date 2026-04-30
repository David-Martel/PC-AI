#!/usr/bin/env python3
"""
MCP Server Test Suite
Tests all configured MCP servers for availability and basic functionality
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Configuration paths
CONFIG_PATHS = [
    Path("C:/Users/david/mcp.json"),
    Path("C:/Users/david/.claude/mcp.json")
]

class McpTester:
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warned_tests = 0

    def print_header(self, message: str):
        print(f"\n{'='*60}")
        print(f"\033[36m{message}\033[0m")
        print('='*60)

    def test_result(self, server_name: str, status: str, message: str, details: str = ""):
        self.total_tests += 1
        self.results[server_name] = {
            'status': status,
            'message': message,
            'details': details
        }

        colors = {
            'PASS': '\033[32m',
            'FAIL': '\033[31m',
            'WARN': '\033[33m'
        }

        color = colors.get(status, '\033[0m')
        print(f"[{color}{status}\033[0m] {server_name}: {message}")

        if details:
            print(f"    Details: {details}")

        if status == 'PASS':
            self.passed_tests += 1
        elif status == 'FAIL':
            self.failed_tests += 1
        elif status == 'WARN':
            self.warned_tests += 1

    def test_command_exists(self, command: str) -> bool:
        """Test if a command/binary exists and is executable"""
        try:
            # Handle absolute paths
            if '/' in command or '\\' in command or command.endswith('.exe'):
                return Path(command).exists()

            # Handle standard commands
            result = subprocess.run(['where', command] if os.name == 'nt' else ['which', command],
                                 capture_output=True, shell=True)
            return result.returncode == 0
        except:
            return False

    def test_rust_binary(self, server_name: str, binary_path: str):
        """Test a Rust binary for functionality"""
        if not Path(binary_path).exists():
            self.test_result(server_name, "FAIL", f"Binary not found: {binary_path}")
            return

        # Try different help flags
        help_flags = ['--help', '--version', '-h', '-V']
        for flag in help_flags:
            try:
                result = subprocess.run([binary_path, flag],
                                      capture_output=True,
                                      text=True,
                                      timeout=5)
                if result.returncode == 0:
                    self.test_result(server_name, "PASS", "Rust binary functional")
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue

        self.test_result(server_name, "WARN", "Binary exists but may not be functional")

    def test_python_script(self, server_name: str, script_path: str):
        """Test a Python script for syntax validity"""
        if not Path(script_path).exists():
            self.test_result(server_name, "FAIL", f"Python script not found: {script_path}")
            return

        try:
            # Test syntax
            result = subprocess.run(['uv', 'run', 'python', '-m', 'py_compile', script_path],
                                  capture_output=True, timeout=10)
            if result.returncode == 0:
                self.test_result(server_name, "PASS", "Python script syntax valid")
            else:
                self.test_result(server_name, "FAIL", "Python script syntax error",
                               result.stderr.decode() if result.stderr else "Unknown error")
        except Exception as e:
            self.test_result(server_name, "FAIL", f"Python test error: {e}")

    def test_mcp_server(self, server_name: str, server_config: dict):
        """Test an individual MCP server configuration"""
        command = server_config.get('command', '')
        args = server_config.get('args', [])

        print(f"\nTesting server: {server_name}")
        print(f"Command: {command}")
        if args:
            print(f"Args: {' '.join(args)}")

        # Test command availability
        if not self.test_command_exists(command):
            self.test_result(server_name, "FAIL", f"Command not found: {command}")
            return

        # Server-specific tests based on command type
        if command == "uv" and len(args) >= 3:
            if args[0] == "run" and args[1] == "python":
                if args[2] == "-m":
                    self.test_result(server_name, "PASS", "Python module command ready")
                elif args[2].endswith('.py'):
                    self.test_python_script(server_name, args[2])
                else:
                    self.test_result(server_name, "WARN", "Python command format unclear")
            else:
                self.test_result(server_name, "WARN", "Non-standard uv command")
        elif command.endswith('.exe'):
            self.test_rust_binary(server_name, command)
        elif command in ['bun', 'npx', 'node']:
            self.test_result(server_name, "PASS", "Node.js runtime available")
        else:
            self.test_result(server_name, "PASS", "Command available")

    def test_configurations(self):
        """Test all MCP configurations"""
        self.print_header("MCP Server Test Suite")

        print("Testing MCP configurations:")
        for path in CONFIG_PATHS:
            print(f"  - {path}")

        # Test each configuration file
        for config_path in CONFIG_PATHS:
            if not config_path.exists():
                print(f"\033[31mConfiguration not found: {config_path}\033[0m")
                continue

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"\033[31mFailed to load {config_path}: {e}\033[0m")
                continue

            self.print_header(f"Testing Configuration: {config_path.name}")

            servers = config.get('mcpServers', {})
            if not servers:
                print(f"\033[33mNo servers found in {config_path}\033[0m")
                continue

            print(f"Found {len(servers)} servers")

            for server_name, server_config in servers.items():
                self.test_mcp_server(server_name, server_config)

    def run_special_tests(self):
        """Run special integration tests"""
        self.print_header("Special Integration Tests")

        # Test ast-grep integration
        print("Testing ast-grep integration...")
        ast_grep_path = Path("C:/Users/david/.local/bin/sg.exe")
        if ast_grep_path.exists():
            try:
                result = subprocess.run([str(ast_grep_path), '--version'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
                    self.test_result("ast-grep-integration", "PASS", f"ast-grep ready - {version}")
                else:
                    self.test_result("ast-grep-integration", "FAIL", "ast-grep binary not functional")
            except Exception as e:
                self.test_result("ast-grep-integration", "FAIL", f"ast-grep test failed: {e}")
        else:
            self.test_result("ast-grep-integration", "FAIL", f"ast-grep binary not found: {ast_grep_path}")

        # Test key Rust binaries
        print("\nTesting Rust MCP binaries...")
        rust_binaries = {
            "rust-fs": "C:/Users/david/.local/bin/rust-fs.exe",
            "rust-fetch": "C:/Users/david/.local/bin/rust-fetch.exe",
            "rust-link": "C:/Users/david/.local/bin/rust-link.exe",
            "rust-sequential-thinking": "C:/Users/david/.local/bin/rust-sequential-thinking.exe"
        }

        for name, path in rust_binaries.items():
            print(f"\nTesting {name}...")
            self.test_rust_binary(f"{name}-binary", path)

        # Test essential commands
        print("\nTesting essential dependencies...")
        essential_commands = {
            "bun": "bun",
            "uv": "uv",
            "node": "node"
        }

        for name, cmd in essential_commands.items():
            if self.test_command_exists(cmd):
                self.test_result(f"{name}-dependency", "PASS", "Essential command available")
            else:
                self.test_result(f"{name}-dependency", "FAIL", f"Essential command missing: {cmd}")

    def print_summary(self):
        """Print test summary and results"""
        self.print_header("Test Summary")

        print(f"Total tests: {self.total_tests}")
        print(f"✓ Passed: \033[32m{self.passed_tests}\033[0m")
        print(f"✗ Failed: \033[31m{self.failed_tests}\033[0m")
        print(f"⚠ Warnings: \033[33m{self.warned_tests}\033[0m")

        success_rate = (self.passed_tests * 100) // self.total_tests if self.total_tests > 0 else 0
        color = '\033[32m' if success_rate >= 80 else '\033[33m' if success_rate >= 60 else '\033[31m'
        print(f"Success rate: {color}{success_rate}%\033[0m")

        # Detailed results
        print("\nDetailed Results:")
        print("=" * 20)
        for server, result in self.results.items():
            status = result['status']
            message = result['message']

            if status == 'PASS':
                print(f"\033[32m✓\033[0m {server}: {message}")
            elif status == 'FAIL':
                print(f"\033[31m✗\033[0m {server}: {message}")
            elif status == 'WARN':
                print(f"\033[33m⚠\033[0m {server}: {message}")

        # Recommendations
        failed_servers = {k: v for k, v in self.results.items() if v['status'] == 'FAIL'}
        if failed_servers:
            print("\n" + "="*60)
            print("\033[36mRecommendations\033[0m")
            print("="*60)
            print("Failed servers requiring attention:")
            for server, result in failed_servers.items():
                print(f"  - {server}: {result['message']}")

    def run_all_tests(self):
        """Run complete test suite"""
        print("MCP Server Comprehensive Test Suite")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.test_configurations()
        self.run_special_tests()
        self.print_summary()

        print(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Export results as JSON
        results_file = Path("C:/Users/david/mcp_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total': self.total_tests,
                    'passed': self.passed_tests,
                    'failed': self.failed_tests,
                    'warned': self.warned_tests,
                    'success_rate': (self.passed_tests * 100) // self.total_tests if self.total_tests > 0 else 0
                },
                'results': self.results
            }, f, indent=2)

        print(f"Results exported to: {results_file}")

if __name__ == "__main__":
    tester = McpTester()
    tester.run_all_tests()