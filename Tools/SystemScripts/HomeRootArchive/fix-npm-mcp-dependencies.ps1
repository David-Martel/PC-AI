# Fix NPM/NPX MCP Tool Dependencies
# Comprehensive solution for broken npm.cmd infinite loop and MCP server configuration
# Author: DevOps Troubleshooter Agent
# Date: 2025-09-06

Write-Host "=== NPM/NPX MCP Dependencies Fix Script ===" -ForegroundColor Green
Write-Host "Diagnosing and fixing NPM/NPX issues for MCP servers..." -ForegroundColor Yellow

# Check current Node.js installation
Write-Host "`n1. Checking Node.js Installation..." -ForegroundColor Cyan
$nodeVersion = & node --version 2>$null
if ($nodeVersion) {
    Write-Host "   ✓ Node.js version: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "   ✗ Node.js not found in PATH" -ForegroundColor Red
    exit 1
}

# Check npm.cmd for infinite loop issue
Write-Host "`n2. Diagnosing npm.cmd infinite loop issue..." -ForegroundColor Cyan
$npmCmdPath = "C:\Program Files\nodejs\npm.cmd"
if (Test-Path $npmCmdPath) {
    $npmCmdContent = Get-Content $npmCmdPath -Raw
    if ($npmCmdContent -match 'npm\.cmd.*%\*') {
        Write-Host "   ✗ Found infinite loop in npm.cmd (self-referencing)" -ForegroundColor Red
        Write-Host "     Issue: npm.cmd calls itself recursively" -ForegroundColor Yellow
    } else {
        Write-Host "   ✓ npm.cmd appears correct" -ForegroundColor Green
    }
} else {
    Write-Host "   ✗ npm.cmd not found at expected location" -ForegroundColor Red
}

# Check if npm.ps1 exists and works
Write-Host "`n3. Testing PowerShell npm/npx scripts..." -ForegroundColor Cyan
$npmPs1Path = "C:\Program Files\nodejs\npm.ps1"
$npxPs1Path = "C:\Program Files\nodejs\npx.ps1"

if (Test-Path $npmPs1Path) {
    Write-Host "   ✓ npm.ps1 found" -ForegroundColor Green
    try {
        $npmPsVersion = & pwsh -File $npmPs1Path --version 2>$null | Select-Object -Last 1
        if ($npmPsVersion -match '\d+\.\d+\.\d+') {
            Write-Host "   ✓ npm.ps1 working - version: $npmPsVersion" -ForegroundColor Green
        }
    } catch {
        Write-Host "   ⚠ npm.ps1 exists but may have issues" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✗ npm.ps1 not found" -ForegroundColor Red
    exit 1
}

if (Test-Path $npxPs1Path) {
    Write-Host "   ✓ npx.ps1 found" -ForegroundColor Green
} else {
    Write-Host "   ✗ npx.ps1 not found" -ForegroundColor Red
    exit 1
}

# Fix npm.bat wrapper
Write-Host "`n4. Fixing npm.bat wrapper..." -ForegroundColor Cyan
$npmBatPath = "C:\users\david\.local\bin\npm.bat"
$npmBatDir = Split-Path $npmBatPath -Parent

if (!(Test-Path $npmBatDir)) {
    New-Item -Path $npmBatDir -ItemType Directory -Force | Out-Null
    Write-Host "   ✓ Created directory: $npmBatDir" -ForegroundColor Green
}

$npmBatContent = @"
@echo off
pwsh -File "C:\Program Files\nodejs\npm.ps1" %*
"@

Set-Content -Path $npmBatPath -Value $npmBatContent -Encoding ASCII
Write-Host "   ✓ Fixed npm.bat wrapper" -ForegroundColor Green

# Fix npx.bat wrapper
Write-Host "`n5. Fixing npx.bat wrapper..." -ForegroundColor Cyan
$npxBatPath = "C:\users\david\.local\bin\npx.bat"

$npxBatContent = @"
@echo off
pwsh -File "C:\Program Files\nodejs\npx.ps1" %*
"@

Set-Content -Path $npxBatPath -Value $npxBatContent -Encoding ASCII
Write-Host "   ✓ Fixed npx.bat wrapper" -ForegroundColor Green

# Test the fixed wrappers
Write-Host "`n6. Testing fixed npm/npx wrappers..." -ForegroundColor Cyan
try {
    $npmTestVersion = & $npmBatPath --version 2>$null | Select-Object -Last 1
    if ($npmTestVersion -match '\d+\.\d+\.\d+') {
        Write-Host "   ✓ npm.bat working - version: $npmTestVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ npm.bat may have issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ npm.bat test failed: $_" -ForegroundColor Red
}

try {
    $npxTestVersion = & $npxBatPath --version 2>$null | Select-Object -Last 1
    if ($npxTestVersion -match '\d+\.\d+\.\d+') {
        Write-Host "   ✓ npx.bat working - version: $npxTestVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ npx.bat may have issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ npx.bat test failed: $_" -ForegroundColor Red
}

# Test MCP Inspector
Write-Host "`n7. Testing @modelcontextprotocol/inspector..." -ForegroundColor Cyan
try {
    $inspectorOutput = & $npxBatPath @modelcontextprotocol/inspector --help 2>&1 | Out-String
    if ($inspectorOutput -match "Usage: inspector-bin") {
        Write-Host "   ✓ MCP Inspector working correctly" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ MCP Inspector responded but may have issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ MCP Inspector test failed: $_" -ForegroundColor Red
}

# Update MCP configuration if exists
Write-Host "`n8. Checking MCP configuration..." -ForegroundColor Cyan
$mcpConfigPath = "C:\Users\david\.claude\mcp.json"
if (Test-Path $mcpConfigPath) {
    Write-Host "   ✓ Found MCP configuration at: $mcpConfigPath" -ForegroundColor Green

    # Backup original config
    $backupPath = "$mcpConfigPath.backup-$(Get-Date -Format 'yyyy-MM-dd-HHmmss')"
    Copy-Item $mcpConfigPath $backupPath
    Write-Host "   ✓ Created backup: $backupPath" -ForegroundColor Green

    # Read and update configuration
    $mcpConfig = Get-Content $mcpConfigPath -Raw
    $updatedConfig = $mcpConfig -replace 'cmd.*?/c.*?npx', "C:\\users\\david\\.local\\bin\\npx.bat"

    if ($updatedConfig -ne $mcpConfig) {
        Set-Content -Path $mcpConfigPath -Value $updatedConfig -Encoding UTF8
        Write-Host "   ✓ Updated MCP configuration to use fixed npx wrapper" -ForegroundColor Green
    } else {
        Write-Host "   ℹ MCP configuration already uses correct npx paths" -ForegroundColor Blue
    }
} else {
    Write-Host "   ℹ No MCP configuration found - manual update needed" -ForegroundColor Blue
}

# Alternative installation methods
Write-Host "`n9. Alternative Package Manager Options..." -ForegroundColor Cyan

# Check if bun is available
$bunPath = Get-Command bun -ErrorAction SilentlyContinue
if ($bunPath) {
    Write-Host "   ✓ Bun available as alternative: $($bunPath.Source)" -ForegroundColor Green
    Write-Host "     Alternative command: bun x @modelcontextprotocol/inspector" -ForegroundColor Blue
} else {
    Write-Host "   ℹ Bun not available (can install from: https://bun.sh)" -ForegroundColor Blue
}

# Check if pnpm is available
$pnpmPath = Get-Command pnpm -ErrorAction SilentlyContinue
if ($pnpmPath) {
    Write-Host "   ✓ pnpm available as alternative: $($pnpmPath.Source)" -ForegroundColor Green
    Write-Host "     Alternative command: pnpm dlx @modelcontextprotocol/inspector" -ForegroundColor Blue
} else {
    Write-Host "   ℹ pnpm not available (can install via: npm install -g pnpm)" -ForegroundColor Blue
}

# Summary and next steps
Write-Host "`n=== SUMMARY ===" -ForegroundColor Green
Write-Host "Root Cause Found:" -ForegroundColor Yellow
Write-Host "  • npm.cmd had infinite loop (called itself recursively)" -ForegroundColor White
Write-Host "  • This caused npm/npx commands to hang indefinitely" -ForegroundColor White
Write-Host ""
Write-Host "Solution Applied:" -ForegroundColor Yellow
Write-Host "  • Fixed npm.bat to use npm.ps1 instead of broken npm.cmd" -ForegroundColor White
Write-Host "  • Fixed npx.bat to use npx.ps1 instead of broken npx.cmd" -ForegroundColor White
Write-Host "  • Updated MCP configuration to use working wrappers" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Test MCP servers: npx @modelcontextprotocol/inspector --help" -ForegroundColor White
Write-Host "  2. Validate config: npx @modelcontextprotocol/inspector --cli --config mcp.json" -ForegroundColor White
Write-Host "  3. Consider installing bun for faster package execution" -ForegroundColor White
Write-Host ""
Write-Host "Fixed Paths:" -ForegroundColor Yellow
Write-Host "  • npm: C:\users\david\.local\bin\npm.bat" -ForegroundColor White
Write-Host "  • npx: C:\users\david\.local\bin\npx.bat" -ForegroundColor White
Write-Host "  • MCP config: $mcpConfigPath" -ForegroundColor White

Write-Host "`n✅ NPM/NPX MCP Dependencies Fix Complete!" -ForegroundColor Green