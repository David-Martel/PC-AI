# Fix Bun MCP Text Rendering Issues on Windows
# This script addresses code page, encoding, and Bun-specific issues

param(
    [switch]$Apply = $false,
    [switch]$Test = $false
)

Write-Host "=== Bun MCP Text Rendering Fix Script ===" -ForegroundColor Cyan

# 1. Set UTF-8 Code Page
Write-Host "`n[1] Setting UTF-8 Code Page (65001)..." -ForegroundColor Yellow
if ($Apply) {
    # Set for current session
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null

    # Set in registry for persistence
    Set-ItemProperty -Path "HKCU:\Console" -Name "CodePage" -Value 65001 -Type DWORD -Force
    Write-Host "  ✓ Code page set to UTF-8" -ForegroundColor Green
} else {
    Write-Host "  → Would set code page to UTF-8 (65001)" -ForegroundColor Gray
}

# 2. Enable Virtual Terminal Processing
Write-Host "`n[2] Enabling Virtual Terminal Processing..." -ForegroundColor Yellow
if ($Apply) {
    Set-ItemProperty -Path "HKCU:\Console" -Name "VirtualTerminalLevel" -Value 1 -Type DWORD -Force
    Write-Host "  ✓ Virtual Terminal Processing enabled" -ForegroundColor Green
} else {
    Write-Host "  → Would enable Virtual Terminal Processing" -ForegroundColor Gray
}

# 3. Set Environment Variables for Bun
Write-Host "`n[3] Setting Bun-specific environment variables..." -ForegroundColor Yellow
$envVars = @{
    "PYTHONIOENCODING" = "utf-8"
    "LANG" = "en_US.UTF-8"
    "BUN_RUNTIME_DISABLE_COLORS" = "0"
    "NODE_NO_WARNINGS" = "1"
}

foreach ($key in $envVars.Keys) {
    if ($Apply) {
        [System.Environment]::SetEnvironmentVariable($key, $envVars[$key], [System.EnvironmentVariableTarget]::User)
        Write-Host "  ✓ Set $key = $($envVars[$key])" -ForegroundColor Green
    } else {
        Write-Host "  → Would set $key = $($envVars[$key])" -ForegroundColor Gray
    }
}

# 4. Create MCP wrapper script
Write-Host "`n[4] Creating MCP wrapper script..." -ForegroundColor Yellow
$wrapperPath = "$env:USERPROFILE\.claude\scripts\mcp-wrapper.cmd"
$wrapperContent = @'
@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8
set BUN_RUNTIME_DISABLE_COLORS=0
set FORCE_COLOR=1
set COLORTERM=truecolor
set TERM=xterm-256color
bun %*
'@

if ($Apply) {
    $scriptDir = Split-Path $wrapperPath -Parent
    if (!(Test-Path $scriptDir)) {
        New-Item -ItemType Directory -Path $scriptDir -Force | Out-Null
    }
    Set-Content -Path $wrapperPath -Value $wrapperContent -Encoding UTF8
    Write-Host "  ✓ Created MCP wrapper at $wrapperPath" -ForegroundColor Green
} else {
    Write-Host "  → Would create MCP wrapper script" -ForegroundColor Gray
}

# 5. Test rendering
if ($Test -or $Apply) {
    Write-Host "`n[5] Testing text rendering..." -ForegroundColor Yellow
    Write-Host "  Basic ASCII: ABCDEFGHIJKLMNOPQRSTUVWXYZ" -ForegroundColor White
    Write-Host "  Unicode: ✓ ✗ → ← ↑ ↓" -ForegroundColor White
    Write-Host "  Box Drawing: ┌─┐ │ └─┘" -ForegroundColor White
    Write-Host "  Colors: " -NoNewline
    Write-Host "Red " -ForegroundColor Red -NoNewline
    Write-Host "Green " -ForegroundColor Green -NoNewline
    Write-Host "Blue" -ForegroundColor Blue
}

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($Apply) {
    Write-Host "✓ All fixes have been applied!" -ForegroundColor Green
    Write-Host "Please restart your terminal and Claude Code for changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "This was a dry run. Use -Apply to make changes:" -ForegroundColor Yellow
    Write-Host "  .\fix-bun-mcp-rendering.ps1 -Apply" -ForegroundColor White
}