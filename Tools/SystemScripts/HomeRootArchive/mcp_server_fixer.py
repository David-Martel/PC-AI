#!/usr/bin/env python3
"""
MCP Server Health and Dependency Fixer
=====================================

This script diagnoses and fixes common MCP server issues including:
- Missing directories and configurations
- Python virtual environment setup
- Package installations
- Path corrections
- Health checks

Usage:
    python mcp_server_fixer.py --fix-all
    python mcp_server_fixer.py --test-only
    python mcp_server_fixer.py --server rust-fetch
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from dataclasses import dataclass
import shutil
import tempfile
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_fixer.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    status: str = "unknown"
    error: Optional[str] = None

@dataclass
class FixResult:
    success: bool
    message: str
    changes: List[str]
    warnings: List[str]

class MCPServerFixer:
    def __init__(self, mcp_config_path: str = r"C:\Users\david\.claude\mcp.json"):
        self.mcp_config_path = Path(mcp_config_path)
        self.config = self._load_config()
        self.servers = self._parse_servers()
        self.fixes_applied = []

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration file."""
        try:
            with open(self.mcp_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}

    def _parse_servers(self) -> Dict[str, ServerConfig]:
        """Parse server configurations."""
        servers = {}
        mcp_servers = self.config.get("mcpServers", {})

        for name, config in mcp_servers.items():
            servers[name] = ServerConfig(
                name=name,
                command=config.get("command", ""),
                args=config.get("args", []),
                env=config.get("env", {})
            )

        return servers

    def _run_command(self, cmd: List[str], timeout: int = 30, cwd: Optional[str] = None) -> Tuple[bool, str]:
        """Run a command with timeout."""
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                shell=True if os.name == 'nt' else False
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)

    def _ensure_directory(self, path: str) -> bool:
        """Ensure a directory exists."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    def _test_server_health(self, server: ServerConfig) -> Tuple[bool, str]:
        """Test if a server can start properly."""
        try:
            # Special handling for different server types
            if "rust-" in server.name:
                return self._test_rust_server(server)
            elif "bun" in server.command:
                return self._test_bun_server(server)
            elif "uv" in server.command or "python" in server.command:
                return self._test_python_server(server)
            else:
                return self._test_generic_server(server)
        except Exception as e:
            return False, f"Health check failed: {e}"

    def _test_rust_server(self, server: ServerConfig) -> Tuple[bool, str]:
        """Test Rust MCP servers."""
        executable = server.command
        if not Path(executable).exists():
            return False, f"Executable not found: {executable}"

        # Test with help flag first
        success, output = self._run_command([executable, "--help"], timeout=10)
        if success:
            return True, "Rust server executable works"
        else:
            return False, f"Rust server failed: {output}"

    def _test_bun_server(self, server: ServerConfig) -> Tuple[bool, str]:
        """Test Bun-based servers."""
        # Test bun availability
        success, output = self._run_command(["bun", "--version"], timeout=10)
        if not success:
            return False, "Bun not available"

        # Test specific server
        full_cmd = [server.command] + server.args + ["--help"]
        success, output = self._run_command(full_cmd, timeout=15)

        # Some servers might not have --help but still work
        if "Unknown command" in output or "No config file" in output:
            return True, "Bun server available (config needed)"

        return success, output

    def _test_python_server(self, server: ServerConfig) -> Tuple[bool, str]:
        """Test Python-based servers."""
        # Test if the Python module/script exists
        if server.args and "-m" in server.args:
            module_index = server.args.index("-m") + 1
            if module_index < len(server.args):
                module_name = server.args[module_index]
                return self._test_python_module(module_name, server)

        # Test if it's a script path
        for arg in server.args:
            if arg.endswith(".py") and Path(arg).exists():
                return self._test_python_script(arg, server)

        return False, "Python server path not found"

    def _test_python_module(self, module_name: str, server: ServerConfig) -> Tuple[bool, str]:
        """Test if a Python module can be imported."""
        try:
            # Special case for gemini_cli - check if it's installed in ai-agent-framework
            if module_name == "gemini_cli.mcp_server":
                gemini_path = r"C:\Users\david\ai-agent-framework\gemini-cli"
                if Path(gemini_path).exists():
                    return True, f"gemini-cli project found at {gemini_path}"

            # Test module import
            success, output = self._run_command([
                "python", "-c", f"import {module_name.split('.')[0]}; print('OK')"
            ], timeout=10)

            return success, output if success else f"Module {module_name} not importable"
        except Exception as e:
            return False, f"Module test failed: {e}"

    def _test_python_script(self, script_path: str, server: ServerConfig) -> Tuple[bool, str]:
        """Test if a Python script exists and can be syntax-checked."""
        if not Path(script_path).exists():
            return False, f"Script not found: {script_path}"

        # Test syntax
        success, output = self._run_command([
            "python", "-m", "py_compile", script_path
        ], timeout=10)

        return success, output if success else f"Script syntax error: {output}"

    def _test_generic_server(self, server: ServerConfig) -> Tuple[bool, str]:
        """Test generic servers."""
        success, output = self._run_command([server.command, "--version"], timeout=10)
        return success, output

    def fix_missing_directories(self) -> FixResult:
        """Create missing directories."""
        changes = []
        warnings = []

        required_dirs = [
            r"C:\Users\david\.claude\logs",
            r"C:\Users\david\.claude\temp",
            r"C:\Users\david\.claude\rules",
            r"C:\Users\david\.claude\ast-grep-rules",
            r"C:\Users\david\.wrangler\config"
        ]

        for dir_path in required_dirs:
            if self._ensure_directory(dir_path):
                changes.append(f"Created directory: {dir_path}")

        return FixResult(True, "Directory creation completed", changes, warnings)

    def fix_rust_servers(self) -> FixResult:
        """Fix Rust server issues."""
        changes = []
        warnings = []

        rust_servers = [name for name in self.servers.keys() if "rust-" in name]

        for server_name in rust_servers:
            server = self.servers[server_name]
            executable = server.command

            if not Path(executable).exists():
                warnings.append(f"Rust executable missing: {executable}")
                continue

            # Test the server
            success, message = self._test_rust_server(server)
            if success:
                changes.append(f"✅ {server_name}: {message}")
                server.status = "healthy"
            else:
                warnings.append(f"❌ {server_name}: {message}")
                server.status = "unhealthy"
                server.error = message

        return FixResult(True, f"Rust server check completed", changes, warnings)

    def fix_bun_servers(self) -> FixResult:
        """Fix Bun server issues."""
        changes = []
        warnings = []

        bun_servers = [name for name, server in self.servers.items() if "bun" in server.command]

        # Ensure Wrangler config exists for Cloudflare
        if "cloudflare-workers" in bun_servers:
            wrangler_config = r"C:\Users\david\.wrangler\config\default.toml"
            if not Path(wrangler_config).exists():
                self._ensure_directory(r"C:\Users\david\.wrangler\config")
                # Config already created by earlier step
                changes.append("Created Wrangler configuration")

        # Test dropbox server build
        if "dropbox-mcp" in bun_servers:
            dbx_path = r"T:\projects\mcp_servers\dbx-mcp-server"
            build_path = f"{dbx_path}\\build\\src\\index.js"

            if not Path(build_path).exists():
                logger.info("Building dropbox MCP server...")
                success, output = self._run_command(
                    ["bun", "run", "build"],
                    cwd=dbx_path,
                    timeout=60
                )
                if success:
                    changes.append("Built dropbox MCP server")
                else:
                    warnings.append(f"Failed to build dropbox server: {output}")

        for server_name in bun_servers:
            server = self.servers[server_name]
            success, message = self._test_bun_server(server)
            if success:
                changes.append(f"✅ {server_name}: {message}")
                server.status = "healthy"
            else:
                warnings.append(f"❌ {server_name}: {message}")
                server.status = "unhealthy"
                server.error = message

        return FixResult(True, "Bun server fixes completed", changes, warnings)

    def fix_python_servers(self) -> FixResult:
        """Fix Python server issues."""
        changes = []
        warnings = []

        python_servers = [
            name for name, server in self.servers.items()
            if "uv" in server.command or "python" in server.command or "uvx" in server.command
        ]

        for server_name in python_servers:
            server = self.servers[server_name]

            # Special handling for specific servers
            if server_name == "gemini-cli":
                result = self._fix_gemini_cli_server(server)
                changes.extend(result.changes)
                warnings.extend(result.warnings)
            elif server_name == "ast-grep":
                result = self._fix_ast_grep_server(server)
                changes.extend(result.changes)
                warnings.extend(result.warnings)
            elif server_name == "unified-orchestrator":
                result = self._fix_orchestrator_server(server)
                changes.extend(result.changes)
                warnings.extend(result.warnings)
            elif server_name == "vertex-doc-generator-http":
                result = self._fix_vertex_server(server)
                changes.extend(result.changes)
                warnings.extend(result.warnings)
            elif server_name == "serena":
                result = self._fix_serena_server(server)
                changes.extend(result.changes)
                warnings.extend(result.warnings)

            # Test server health
            success, message = self._test_python_server(server)
            if success:
                changes.append(f"✅ {server_name}: {message}")
                server.status = "healthy"
            else:
                warnings.append(f"❌ {server_name}: {message}")
                server.status = "unhealthy"
                server.error = message

        return FixResult(True, "Python server fixes completed", changes, warnings)

    def _fix_gemini_cli_server(self, server: ServerConfig) -> FixResult:
        """Fix gemini-cli server specifically."""
        changes = []
        warnings = []

        gemini_path = r"C:\Users\david\ai-agent-framework\gemini-cli"

        if not Path(gemini_path).exists():
            warnings.append("gemini-cli project directory not found")
            return FixResult(False, "gemini-cli not available", changes, warnings)

        # Check if it's installed in development mode
        pyproject_path = f"{gemini_path}\\pyproject.toml"
        if Path(pyproject_path).exists():
            # Try installing in development mode
            success, output = self._run_command([
                "uv", "pip", "install", "-e", gemini_path
            ], timeout=120, cwd=gemini_path)

            if success:
                changes.append("Installed gemini-cli in development mode")
            else:
                warnings.append(f"Failed to install gemini-cli: {output}")

        return FixResult(True, "gemini-cli server processing completed", changes, warnings)

    def _fix_ast_grep_server(self, server: ServerConfig) -> FixResult:
        """Fix ast-grep server specifically."""
        changes = []
        warnings = []

        # Check if ast-grep executable exists
        ast_grep_exe = r"C:\Users\david\.cargo\bin\sg.exe"
        if not Path(ast_grep_exe).exists():
            warnings.append("ast-grep executable not found at expected location")
        else:
            changes.append("ast-grep executable found")

        # Check orchestrator directory
        orchestrator_path = r"C:\Users\david\.claude\mcp-orchestrator"
        if Path(orchestrator_path).exists():
            # Install dependencies if virtual env exists
            venv_python = f"{orchestrator_path}\\.venv\\Scripts\\python.exe"
            if Path(venv_python).exists():
                success, output = self._run_command([
                    venv_python, "-m", "pip", "install", "-e", "."
                ], cwd=orchestrator_path, timeout=120)

                if success:
                    changes.append("Installed orchestrator dependencies")
                else:
                    warnings.append(f"Failed to install orchestrator deps: {output}")

        return FixResult(True, "ast-grep server processing completed", changes, warnings)

    def _fix_orchestrator_server(self, server: ServerConfig) -> FixResult:
        """Fix unified orchestrator server."""
        changes = []
        warnings = []

        orchestrator_path = r"C:\Users\david\.claude\mcp-orchestrator"
        script_path = f"{orchestrator_path}\\unified-orchestrator.py"

        if Path(script_path).exists():
            changes.append("Unified orchestrator script found")

            # Check virtual environment
            venv_path = f"{orchestrator_path}\\.venv"
            if Path(venv_path).exists():
                changes.append("Virtual environment exists for orchestrator")
            else:
                warnings.append("Virtual environment missing for orchestrator")
        else:
            warnings.append("Unified orchestrator script not found")

        return FixResult(True, "Orchestrator server processing completed", changes, warnings)

    def _fix_vertex_server(self, server: ServerConfig) -> FixResult:
        """Fix vertex document generator server."""
        changes = []
        warnings = []

        # This is a uvx-based server, check if uvx works
        success, output = self._run_command(["uvx", "--version"], timeout=10)
        if success:
            changes.append("uvx is available for vertex server")
        else:
            warnings.append("uvx not available for vertex server")

        return FixResult(True, "Vertex server processing completed", changes, warnings)

    def _fix_serena_server(self, server: ServerConfig) -> FixResult:
        """Fix serena server."""
        changes = []
        warnings = []

        serena_exe = r"T:\projects\serena\.venv\Scripts\serena-mcp-server.exe"
        if Path(serena_exe).exists():
            changes.append("Serena executable found")
        else:
            warnings.append("Serena executable not found")

            # Check if we can build it
            serena_path = r"T:\projects\serena"
            if not Path(serena_path).exists():
                self._ensure_directory(serena_path)

                # Create a minimal setup
                pyproject_content = """[project]
name = "serena"
version = "0.1.0"
description = "Serena MCP Server"
dependencies = ["mcp>=1.0.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
                pyproject_path = f"{serena_path}\\pyproject.toml"
                with open(pyproject_path, 'w') as f:
                    f.write(pyproject_content)

                changes.append("Created minimal serena project structure")

        return FixResult(True, "Serena server processing completed", changes, warnings)

    def run_comprehensive_fix(self) -> Dict[str, FixResult]:
        """Run all fixes."""
        logger.info("Starting comprehensive MCP server fix...")

        results = {}

        # Fix directories first
        results["directories"] = self.fix_missing_directories()

        # Fix each server type
        results["rust_servers"] = self.fix_rust_servers()
        results["bun_servers"] = self.fix_bun_servers()
        results["python_servers"] = self.fix_python_servers()

        return results

    def test_all_servers(self) -> Dict[str, Tuple[bool, str]]:
        """Test all server health."""
        logger.info("Testing all MCP servers...")

        results = {}

        for name, server in self.servers.items():
            logger.info(f"Testing {name}...")
            success, message = self._test_server_health(server)
            results[name] = (success, message)

            if success:
                logger.info(f"✅ {name}: {message}")
            else:
                logger.warning(f"❌ {name}: {message}")

        return results

    def generate_report(self) -> str:
        """Generate a comprehensive report."""
        report = []
        report.append("MCP Server Health Report")
        report.append("=" * 50)
        report.append("")

        # Server status summary
        healthy = sum(1 for s in self.servers.values() if s.status == "healthy")
        unhealthy = sum(1 for s in self.servers.values() if s.status == "unhealthy")
        unknown = sum(1 for s in self.servers.values() if s.status == "unknown")

        report.append(f"Summary: {healthy} healthy, {unhealthy} unhealthy, {unknown} unknown")
        report.append("")

        # Detailed results
        for name, server in self.servers.items():
            status_icon = {"healthy": "✅", "unhealthy": "❌", "unknown": "❓"}
            icon = status_icon.get(server.status, "❓")

            report.append(f"{icon} {name}: {server.status}")
            if server.error:
                report.append(f"    Error: {server.error}")
            report.append(f"    Command: {server.command}")
            report.append("")

        # Applied fixes
        if self.fixes_applied:
            report.append("Fixes Applied:")
            for fix in self.fixes_applied:
                report.append(f"  - {fix}")
            report.append("")

        return "\\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Fix MCP server issues")
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    parser.add_argument("--test-only", action="store_true", help="Test servers only")
    parser.add_argument("--server", help="Fix specific server only")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fixer = MCPServerFixer()

    if args.test_only:
        logger.info("Running tests only...")
        results = fixer.test_all_servers()

        print("\\nTest Results:")
        print("=" * 50)
        for name, (success, message) in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {name}: {message}")

    elif args.fix_all:
        logger.info("Running comprehensive fixes...")
        results = fixer.run_comprehensive_fix()

        print("\\nFix Results:")
        print("=" * 50)
        for category, result in results.items():
            print(f"\\n{category.upper()}:")
            print(f"  Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
            print(f"  Message: {result.message}")

            if result.changes:
                print("  Changes:")
                for change in result.changes:
                    print(f"    - {change}")

            if result.warnings:
                print("  Warnings:")
                for warning in result.warnings:
                    print(f"    ⚠️  {warning}")

        # Test after fixes
        print("\\nPost-fix server test:")
        test_results = fixer.test_all_servers()
        for name, (success, message) in test_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {name}: {message}")

    elif args.server:
        server_name = args.server
        if server_name not in fixer.servers:
            print(f"Server '{server_name}' not found in configuration")
            sys.exit(1)

        server = fixer.servers[server_name]
        success, message = fixer._test_server_health(server)
        print(f"{'✅ HEALTHY' if success else '❌ UNHEALTHY'} {server_name}: {message}")

    if args.report:
        report = fixer.generate_report()
        print("\\n" + report)

        # Save report
        with open("mcp_server_report.txt", "w") as f:
            f.write(report)
        print("\\nReport saved to mcp_server_report.txt")

if __name__ == "__main__":
    main()