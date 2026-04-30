#!/usr/bin/env python3
"""
Comprehensive test script for GCP authentication and WSL integration with MCP servers.
Tests all critical components of the cross-platform setup.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class IntegrationTester:
    def __init__(self):
        self.results = {
            'gcp_auth': [],
            'wsl_integration': [],
            'mcp_servers': [],
            'cross_platform': []
        }
        self.passed = 0
        self.failed = 0

    def test(self, category: str, name: str, condition: bool, details: str = ""):
        """Record a test result."""
        status = "✓" if condition else "✗"
        self.results[category].append({
            'name': name,
            'passed': condition,
            'status': status,
            'details': details
        })
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"{status} {name}: {details}")

    def test_gcp_authentication(self):
        """Test Google Cloud Platform authentication setup."""
        print("\n=== GOOGLE CLOUD PLATFORM AUTHENTICATION ===")

        # Check environment variables
        required_vars = [
            'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_APPLICATION_CREDENTIALS',
            'VERTEXAI_PROJECT',
            'VERTEXAI_LOCATION'
        ]

        for var in required_vars:
            value = os.getenv(var)
            self.test('gcp_auth', f'Environment: {var}',
                     bool(value), value or "NOT SET")

        # Check service account key
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path:
            # Try both Windows and WSL paths
            paths_to_try = [
                creds_path,
                creds_path.replace('/home/david', 'C:/Users/david'),
                creds_path.replace('/home/david', '//wsl.localhost/Ubuntu/home/david')
            ]

            found = False
            for path in paths_to_try:
                if os.path.exists(path):
                    found = True
                    try:
                        with open(path, 'r') as f:
                            creds = json.load(f)
                            email = creds.get('client_email', 'unknown')
                            self.test('gcp_auth', 'Service Account Key', True,
                                     f"Valid for {email}")
                            break
                    except Exception as e:
                        self.test('gcp_auth', 'Service Account Key', False,
                                 f"Error reading: {e}")
                        break

            if not found:
                self.test('gcp_auth', 'Service Account Key', False,
                         f"File not found at any path variant")

        # Check GCP profile system
        profile_file = Path('C:/Users/david/.gcp/current-profile.txt')
        if profile_file.exists():
            current_profile = profile_file.read_text().strip()
            self.test('gcp_auth', 'GCP Profile System', True,
                     f"Current profile: {current_profile}")
        else:
            self.test('gcp_auth', 'GCP Profile System', False,
                     "Profile file not found")

        # Check profiles.json
        profiles_file = Path('C:/Users/david/.gcp/profiles.json')
        if profiles_file.exists():
            try:
                profiles = json.loads(profiles_file.read_text())
                profile_names = [p['Name'] for p in profiles]
                self.test('gcp_auth', 'Available Profiles', True,
                         f"Found: {', '.join(profile_names)}")
            except Exception as e:
                self.test('gcp_auth', 'Available Profiles', False,
                         f"Error reading profiles: {e}")

    def test_wsl_integration(self):
        """Test Windows Subsystem for Linux integration."""
        print("\n=== WSL INTEGRATION ===")

        # Check WSL installation
        try:
            result = subprocess.run(['cmd.exe', '/c', 'wsl', '--list', '--verbose'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                distros = [line.strip() for line in result.stdout.split('\n')
                          if 'Ubuntu' in line]
                self.test('wsl_integration', 'WSL Installed', True,
                         f"Found Ubuntu distribution")
            else:
                self.test('wsl_integration', 'WSL Installed', False,
                         "WSL not available")
        except Exception as e:
            self.test('wsl_integration', 'WSL Installed', False,
                     f"Cannot check WSL: {e}")

        # Check WSL filesystem access
        wsl_home = Path('//wsl.localhost/Ubuntu/home/david')
        self.test('wsl_integration', 'WSL Filesystem Access',
                 wsl_home.exists(), str(wsl_home))

        # Check WSL GCP directory
        wsl_gcp = Path('//wsl.localhost/Ubuntu/home/david/.gcp')
        self.test('wsl_integration', 'WSL GCP Directory',
                 wsl_gcp.exists(), str(wsl_gcp))

        # Check symlinks/shared auth
        win_auth = Path('C:/Users/david/.auth')
        wsl_auth = Path('//wsl.localhost/Ubuntu/home/david/.auth')

        if win_auth.exists() and wsl_auth.exists():
            # Check if they have the same content structure
            win_files = set(p.name for p in win_auth.rglob('*.json'))
            wsl_files = set(p.name for p in wsl_auth.rglob('*.json'))
            shared = win_files.intersection(wsl_files)
            self.test('wsl_integration', 'Shared Auth Directory',
                     len(shared) > 0, f"Shared files: {len(shared)}")
        else:
            self.test('wsl_integration', 'Shared Auth Directory', False,
                     "Auth directories not accessible")

        # Check WSLENV configuration
        wslenv = os.getenv('WSLENV')
        self.test('wsl_integration', 'WSLENV Configuration',
                 bool(wslenv), wslenv or "NOT SET")

    def test_mcp_servers(self):
        """Test MCP server configurations and executables."""
        print("\n=== MCP SERVER CONFIGURATION ===")

        # Check mcp.json exists
        mcp_config = Path('C:/Users/david/.claude/mcp.json')
        if mcp_config.exists():
            try:
                with open(mcp_config, 'r') as f:
                    config = json.load(f)
                    servers = list(config.get('mcpServers', {}).keys())
                    self.test('mcp_servers', 'MCP Configuration', True,
                             f"Found {len(servers)} servers")

                    # Check specific important servers
                    important_servers = ['gemini-cli', 'rust-fs', 'rust-fetch', 'rust-link']
                    for server in important_servers:
                        self.test('mcp_servers', f'Server: {server}',
                                 server in servers,
                                 "Configured" if server in servers else "Missing")
            except Exception as e:
                self.test('mcp_servers', 'MCP Configuration', False,
                         f"Error reading config: {e}")
        else:
            self.test('mcp_servers', 'MCP Configuration', False,
                     "mcp.json not found")

        # Check MCP server executables
        bin_dir = Path('C:/users/david/.local/bin')
        if bin_dir.exists():
            rust_servers = list(bin_dir.glob('rust*.exe'))
            self.test('mcp_servers', 'Rust MCP Servers',
                     len(rust_servers) > 0,
                     f"Found {len(rust_servers)} Rust servers")

            # Check Python executable for gemini-cli
            python_exe = bin_dir / 'python3.11.exe'
            self.test('mcp_servers', 'Python 3.11',
                     python_exe.exists(), str(python_exe))

    def test_cross_platform_compatibility(self):
        """Test cross-platform path translation and environment sharing."""
        print("\n=== CROSS-PLATFORM COMPATIBILITY ===")

        # Test path translation patterns
        test_paths = [
            ('/home/david/.auth/business/key.json', 'C:/Users/david/.auth/business/key.json'),
            ('/mnt/c/Users/david/file.txt', 'C:/Users/david/file.txt'),
            ('/mnt/t/projects/test', 'T:/projects/test')
        ]

        for wsl_path, win_path in test_paths:
            # Convert and check if logic would work
            converted = wsl_path.replace('/home/david', 'C:/Users/david')
            converted = converted.replace('/mnt/c', 'C:')
            converted = converted.replace('/mnt/t', 'T:')
            converted = converted.replace('/mnt/f', 'F:')

            matches = converted == win_path
            self.test('cross_platform', f'Path Translation', matches,
                     f"{wsl_path} -> {win_path}")

        # Check mount points configuration
        mount_env = os.getenv('RUST_FS_MOUNT_POINTS', '')
        if mount_env:
            mounts = mount_env.split(',')
            self.test('cross_platform', 'Mount Points', True,
                     f"Configured: {', '.join(mounts)}")
        else:
            # Check if they're configured in MCP config
            mcp_config = Path('C:/Users/david/.claude/mcp.json')
            if mcp_config.exists():
                try:
                    with open(mcp_config, 'r') as f:
                        config = json.load(f)
                        rust_fs = config.get('mcpServers', {}).get('rust-fs', {})
                        mount_points = rust_fs.get('env', {}).get('RUST_FS_MOUNT_POINTS', '')
                        self.test('cross_platform', 'Mount Points',
                                 bool(mount_points), mount_points or "Not configured")
                except:
                    self.test('cross_platform', 'Mount Points', False,
                             "Cannot read configuration")

        # Check credential accessibility from both sides
        cred_locations = [
            'C:/Users/david/.auth/business/service-account-key.json',
            '//wsl.localhost/Ubuntu/home/david/.auth/business/service-account-key.json'
        ]

        accessible = sum(1 for path in cred_locations if Path(path).exists())
        self.test('cross_platform', 'Credential Access',
                 accessible == len(cred_locations),
                 f"Accessible from {accessible}/{len(cred_locations)} locations")

    def generate_report(self):
        """Generate a summary report of all tests."""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        for category, tests in self.results.items():
            if tests:
                category_passed = sum(1 for t in tests if t['passed'])
                category_failed = len(tests) - category_passed

                print(f"\n{category.upper().replace('_', ' ')}:")
                print(f"  Passed: {category_passed}/{len(tests)}")

                # Show failed tests
                failed_tests = [t for t in tests if not t['passed']]
                if failed_tests:
                    print("  Failed tests:")
                    for test in failed_tests:
                        print(f"    {test['status']} {test['name']}: {test['details']}")

        print(f"\n" + "=" * 60)
        print(f"TOTAL: {self.passed} passed, {self.failed} failed")

        if self.failed == 0:
            print("✅ ALL TESTS PASSED - System is properly configured!")
        else:
            print(f"⚠️  {self.failed} tests failed - Review configuration above")

        return self.failed == 0

    def run_all_tests(self):
        """Run all integration tests."""
        print("=" * 60)
        print("GCP & WSL INTEGRATION TEST SUITE")
        print("=" * 60)

        self.test_gcp_authentication()
        self.test_wsl_integration()
        self.test_mcp_servers()
        self.test_cross_platform_compatibility()

        return self.generate_report()

def main():
    """Main entry point."""
    tester = IntegrationTester()
    success = tester.run_all_tests()

    # Save results to file
    results_file = Path('C:/Users/david/gcp_wsl_integration_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'results': tester.results,
            'summary': {
                'passed': tester.passed,
                'failed': tester.failed,
                'success': success
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())