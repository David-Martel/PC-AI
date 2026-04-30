#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Troubleshoot and fix npm installation issues on Windows

.DESCRIPTION
    This script helps diagnose and fix common npm installation issues,
    particularly those related to node-gyp and native module compilation.

.EXAMPLE
    .\Fix-NPM-Issues.ps1
#>

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-ColoredOutput "=== NPM Issues Troubleshooting Script ===" "Cyan"

# Check current environment
Write-ColoredOutput "`nChecking current environment..." "Yellow"

$nodeVersion = & node --version 2>$null
$npmVersion = & npm --version 2>$null
$pythonPath = $env:PYTHON
$nodegypVersion = & node-gyp --version 2>$null

Write-ColoredOutput "  Node.js: $nodeVersion" "Green"
Write-ColoredOutput "  npm: $npmVersion" "Green"
Write-ColoredOutput "  PYTHON env var: $pythonPath" $(if($pythonPath) {"Green"} else {"Red"})
Write-ColoredOutput "  node-gyp: $nodegypVersion" $(if($nodegypVersion) {"Green"} else {"Yellow"})

# Check Python installation
Write-ColoredOutput "`nChecking Python installation..." "Yellow"
if ($pythonPath -and (Test-Path $pythonPath)) {
    $pythonVersion = & $pythonPath --version 2>$null
    Write-ColoredOutput "  Python version: $pythonVersion" "Green"
} else {
    Write-ColoredOutput "  Python not found or not configured" "Red"

    # Try to find Python installations
    $pythonPaths = @(
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python313\python.exe"
    )

    foreach ($path in $pythonPaths) {
        if (Test-Path $path) {
            Write-ColoredOutput "  Found Python at: $path" "White"
            try {
                $version = & $path --version 2>$null
                Write-ColoredOutput "    Version: $version" "Green"

                # Set environment variable
                [System.Environment]::SetEnvironmentVariable("PYTHON", $path, "User")
                $env:PYTHON = $path
                Write-ColoredOutput "    Set PYTHON environment variable" "Green"
                break
            }
            catch {
                Write-ColoredOutput "    Failed to get version" "Red"
            }
        }
    }
}

# Check Visual Studio Build Tools
Write-ColoredOutput "`nChecking Visual Studio Build Tools..." "Yellow"
$vcvarsPath = Get-Command "vcvars64.bat" -ErrorAction SilentlyContinue
$clPath = Get-Command "cl.exe" -ErrorAction SilentlyContinue

if ($vcvarsPath -or $clPath) {
    Write-ColoredOutput "  Build tools found" "Green"
    if ($clPath) {
        Write-ColoredOutput "    cl.exe: $($clPath.Source)" "White"
    }
} else {
    Write-ColoredOutput "  Build tools not found" "Red"
    Write-ColoredOutput "  You may need to install Visual Studio Build Tools" "Yellow"
    Write-ColoredOutput "  Run: choco install visualstudio2022buildtools" "White"
}

# Common npm fixes
Write-ColoredOutput "`nApplying common npm fixes..." "Yellow"

# Clear npm cache
Write-ColoredOutput "  Clearing npm cache..." "White"
try {
    & npm cache clean --force 2>$null
    Write-ColoredOutput "    Cache cleared" "Green"
} catch {
    Write-ColoredOutput "    Failed to clear cache" "Red"
}

# Install/update node-gyp globally
Write-ColoredOutput "  Installing/updating node-gyp..." "White"
try {
    & npm install -g node-gyp --silent
    Write-ColoredOutput "    node-gyp updated" "Green"
} catch {
    Write-ColoredOutput "    Failed to update node-gyp" "Red"
}

# Set npm registry (in case of network issues)
Write-ColoredOutput "  Checking npm registry..." "White"
$registry = & npm config get registry 2>$null
Write-ColoredOutput "    Current registry: $registry" "White"

# Suggestions
Write-ColoredOutput "`nSuggestions for problematic packages:" "Cyan"
Write-ColoredOutput "  1. Try installing with --ignore-scripts flag:" "White"
Write-ColoredOutput "     npm install -g @google/gemini-cli --ignore-scripts" "Gray"
Write-ColoredOutput "  2. Use alternative package manager:" "White"
Write-ColoredOutput "     yarn global add @google/gemini-cli" "Gray"
Write-ColoredOutput "  3. Use precompiled binaries when available:" "White"
Write-ColoredOutput "     npm install -g @google/gemini-cli --prefer-offline" "Gray"
Write-ColoredOutput "  4. Install with specific Python:" "White"
Write-ColoredOutput "     npm install -g @google/gemini-cli --python='$env:PYTHON'" "Gray"

Write-ColoredOutput "`n=== Troubleshooting completed ===" "Cyan"
Write-ColoredOutput "If issues persist, check the detailed logs in:" "White"
Write-ColoredOutput "C:\Users\$env:USERNAME\AppData\Local\npm-cache\_logs" "Gray"
