# Comprehensive Warp MCP Integration Test
# This script tests all aspects of the Warp MCP integration

Write-Host "🧪 Testing Warp MCP Integration" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Test 1: Environment Variables
Write-Host "`n1. Testing Environment Variables:" -ForegroundColor Yellow
$requiredVars = @("GOOGLE_CLOUD_PROJECT", "AIDER_MODEL", "VERTEXAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS")
foreach ($var in $requiredVars) {
    $value = [Environment]::GetEnvironmentVariable($var)
    if ($value) {
        $displayValue = if ($var -like "*CREDENTIALS") { "$(Split-Path $value -Leaf)" } else { $value }
        Write-Host "  ✓ $var = $displayValue" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $var = (not set)" -ForegroundColor Red
    }
}

# Test 2: MCP Configuration File
Write-Host "`n2. Testing MCP Configuration:" -ForegroundColor Yellow
$mcpConfigPath = "$HOME\.warp_mcp_config.json"
if (Test-Path $mcpConfigPath) {
    try {
        $mcpConfig = Get-Content $mcpConfigPath | ConvertFrom-Json
        $serverCount = @($mcpConfig.servers.PSObject.Properties).Count
        $envCount = @($mcpConfig.environment.PSObject.Properties).Count
        $aliasCount = @($mcpConfig.aliases.PSObject.Properties).Count

        Write-Host "  ✓ MCP config file exists" -ForegroundColor Green
        Write-Host "  ✓ Servers configured: $serverCount" -ForegroundColor Green
        Write-Host "  ✓ Environment variables: $envCount" -ForegroundColor Green
        Write-Host "  ✓ Aliases defined: $aliasCount" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to parse MCP config: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  ✗ MCP config file not found" -ForegroundColor Red
}

# Test 3: Warp Profile Functions
Write-Host "`n3. Testing Warp Profile Functions:" -ForegroundColor Yellow
$warpFunctions = @("mcp-list", "mcp-start", "Show-WarpStatus", "Get-WarpHelp")
foreach ($func in $warpFunctions) {
    if (Get-Command $func -ErrorAction SilentlyContinue) {
        Write-Host "  ✓ $func function available" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $func function not found" -ForegroundColor Red
    }
}

# Test 4: Sample MCP Server Configurations
Write-Host "`n4. Testing Sample MCP Server Configurations:" -ForegroundColor Yellow
if ($mcpConfig) {
    $testServers = @("filesystem", "github", "memory", "gcp")
    foreach ($serverName in $testServers) {
        $server = $mcpConfig.servers.$serverName
        if ($server) {
            Write-Host "  ✓ $serverName server configured" -ForegroundColor Green
            Write-Host "    Command: $($server.command)" -ForegroundColor Gray
            if ($server.args) {
                Write-Host "    Args: $($server.args -join ' ')" -ForegroundColor Gray
            }
        } else {
            Write-Host "  ⚠ $serverName server not found" -ForegroundColor Yellow
        }
    }
}

# Test 5: Windows Terminal Integration
Write-Host "`n5. Testing Windows Terminal Integration:" -ForegroundColor Yellow
$wtSettingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
if (Test-Path $wtSettingsPath) {
    try {
        $wtSettings = Get-Content $wtSettingsPath | ConvertFrom-Json
        $warpProfiles = $wtSettings.profiles.list | Where-Object { $_.name -like "*Warp*" }
        if ($warpProfiles) {
            Write-Host "  ✓ Windows Terminal integration active" -ForegroundColor Green
            $warpProfiles | ForEach-Object { Write-Host "    Profile: $($_.name)" -ForegroundColor Gray }
        } else {
            Write-Host "  ⚠ Warp profiles not found in Windows Terminal" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ✗ Failed to parse Windows Terminal settings" -ForegroundColor Red
    }
} else {
    Write-Host "  ⚠ Windows Terminal not found" -ForegroundColor Yellow
}

# Test 6: File Permissions and Accessibility
Write-Host "`n6. Testing File Accessibility:" -ForegroundColor Yellow
$testFiles = @(
    "$HOME\.warp_startup.ps1",
    "$HOME\.warp_mcp_config.json",
    "$([System.IO.Path]::GetDirectoryName($PROFILE))\Warp_profile.ps1"
)

foreach ($file in $testFiles) {
    if (Test-Path $file) {
        try {
            $content = Get-Content $file -TotalCount 1
            Write-Host "  ✓ $(Split-Path $file -Leaf) is accessible" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ $(Split-Path $file -Leaf) access error: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  ✗ $(Split-Path $file -Leaf) not found" -ForegroundColor Red
    }
}

# Test 7: Command Execution Test (Safe)
Write-Host "`n7. Testing Safe Command Execution:" -ForegroundColor Yellow
try {
    # Test mcp-list (safe command)
    $mcpListOutput = & { mcp-list } 2>&1
    if ($mcpListOutput -match "Available MCP Servers") {
        Write-Host "  ✓ mcp-list command works correctly" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ mcp-list output unexpected" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ✗ Command execution test failed: $_" -ForegroundColor Red
}

# Summary
Write-Host "`n🎯 Integration Test Summary:" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

if ($mcpConfig) {
    $successfulIntegration = $true

    # Count successful components
    $components = @(
        ($env:GOOGLE_CLOUD_PROJECT -and $env:AIDER_MODEL),
        (Test-Path $mcpConfigPath),
        (Get-Command mcp-list -ErrorAction SilentlyContinue),
        (@($mcpConfig.servers.PSObject.Properties).Count -gt 0)
    )

    $successCount = ($components | Where-Object { $_ }).Count
    $totalCount = $components.Count

    Write-Host "✓ Integration Status: $successCount/$totalCount components working" -ForegroundColor Green
    Write-Host "✓ MCP Servers Available: $(@($mcpConfig.servers.PSObject.Properties).Count)" -ForegroundColor Green
    Write-Host "✓ Environment Variables: $(@($mcpConfig.environment.PSObject.Properties).Count)" -ForegroundColor Green

    Write-Host "`nNext Steps:" -ForegroundColor White
    Write-Host "1. Test in Warp terminal: . '$HOME\.warp_startup.ps1'" -ForegroundColor Gray
    Write-Host "2. Try MCP commands: mcp-list, mcp-start memory" -ForegroundColor Gray
    Write-Host "3. Use AI integration: ai, code-review, explain" -ForegroundColor Gray
} else {
    Write-Host "✗ Integration failed - MCP configuration not found" -ForegroundColor Red
}

Write-Host ""
