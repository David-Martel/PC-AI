# MCP Configuration Validator Script
# Validates both .claude.json and mcp.json for syntax and schema compliance

Write-Host "=== MCP Configuration Validator ===" -ForegroundColor Cyan
Write-Host "Analyzing MCP server configurations..." -ForegroundColor Gray
Write-Host ""

$totalIssues = 0
$configs = @(
    @{Path = "C:\Users\david\.claude.json"; Name = "Main Config (.claude.json)"},
    @{Path = "C:\Users\david\.claude\mcp.json"; Name = "MCP Config (mcp.json)"}
)

foreach ($config in $configs) {
    Write-Host "Checking: $($config.Name)" -ForegroundColor Yellow
    Write-Host "Path: $($config.Path)" -ForegroundColor Gray

    if (!(Test-Path $config.Path)) {
        Write-Host "  [ERROR] File not found!" -ForegroundColor Red
        $totalIssues++
        continue
    }

    # Test JSON validity
    try {
        $json = Get-Content $config.Path -Raw | ConvertFrom-Json
        Write-Host "  [OK] Valid JSON syntax" -ForegroundColor Green
    } catch {
        Write-Host "  [ERROR] Invalid JSON: $_" -ForegroundColor Red
        $totalIssues++
        continue
    }

    # Check for MCP servers
    $mcpServers = $null
    if ($json.mcpServers) {
        $mcpServers = $json.mcpServers
        Write-Host "  [OK] Found mcpServers section" -ForegroundColor Green
    } elseif ($json.PSObject.Properties.Name -contains 'mcpServers') {
        $mcpServers = $json.mcpServers
        Write-Host "  [OK] Found mcpServers section" -ForegroundColor Green
    } else {
        # Check if it's the root-level MCP config
        if ($json.PSObject.Properties | Where-Object { $_.Value.command }) {
            $mcpServers = $json
            Write-Host "  [OK] Root-level MCP server definitions" -ForegroundColor Green
        } else {
            Write-Host "  [WARNING] No MCP servers found" -ForegroundColor Yellow
        }
    }

    if ($mcpServers) {
        $serverCount = 0
        $serverIssues = 0

        # Iterate through servers
        $mcpServers.PSObject.Properties | ForEach-Object {
            $serverName = $_.Name
            $server = $_.Value
            $serverCount++

            # Skip non-server properties
            if ($serverName -in @('globalSettings', 'security', 'version')) {
                return
            }

            Write-Host "    Server: $serverName" -ForegroundColor Cyan

            # Check required fields
            if (!$server.command) {
                Write-Host "      [ERROR] Missing 'command' field" -ForegroundColor Red
                $serverIssues++
            } else {
                # Check command validity
                $cmd = $server.command

                # Check Python commands
                if ($cmd -eq "python" -or $cmd -eq "python3") {
                    Write-Host "      [WARNING] Using bare Python command (should be 'uv run python')" -ForegroundColor Yellow
                    $serverIssues++
                }

                # Check if command exists (for non-WSL commands)
                if ($cmd -notlike "wsl*" -and $cmd -ne "cmd" -and $cmd -ne "uv") {
                    if ($cmd -like "*.exe" -or $cmd -like "*.cmd" -or $cmd -like "*.bat") {
                        if (!(Test-Path $cmd)) {
                            Write-Host "      [WARNING] Command not found: $cmd" -ForegroundColor Yellow
                        }
                    } else {
                        $cmdPath = (Get-Command $cmd -ErrorAction SilentlyContinue).Source
                        if (!$cmdPath -and !(Test-Path $cmd)) {
                            Write-Host "      [WARNING] Command not in PATH: $cmd" -ForegroundColor Yellow
                        }
                    }
                }
            }

            # Check args field
            if ($server.args -and $server.args -isnot [array]) {
                Write-Host "      [ERROR] 'args' must be an array" -ForegroundColor Red
                $serverIssues++
            }

            # Check for invalid fields (MCP schema compliance)
            $validFields = @('command', 'args', 'env', 'cwd')
            $server.PSObject.Properties | ForEach-Object {
                if ($_.Name -notin $validFields) {
                    Write-Host "      [WARNING] Non-standard field: $($_.Name)" -ForegroundColor Yellow
                }
            }

            # Check environment variables
            if ($server.env) {
                $server.env.PSObject.Properties | ForEach-Object {
                    if ($_.Value -match '\$\{([^}]+)\}') {
                        $varName = $matches[1]
                        if (!(Get-Item "env:$varName" -ErrorAction SilentlyContinue)) {
                            Write-Host "      [INFO] Environment variable not set: $varName" -ForegroundColor Gray
                        }
                    }
                }
            }
        }

        Write-Host "  Servers found: $serverCount" -ForegroundColor Gray
        if ($serverIssues -gt 0) {
            Write-Host "  Issues in servers: $serverIssues" -ForegroundColor Yellow
            $totalIssues += $serverIssues
        }
    }

    Write-Host ""
}

# Summary
Write-Host "=== Validation Summary ===" -ForegroundColor Cyan
if ($totalIssues -eq 0) {
    Write-Host "Status: ALL CONFIGURATIONS VALID" -ForegroundColor Green
    Write-Host "Both MCP configuration files are properly formatted and ready to use." -ForegroundColor Green
} else {
    Write-Host "Status: ISSUES FOUND" -ForegroundColor Yellow
    Write-Host "Total issues: $totalIssues" -ForegroundColor Yellow
    Write-Host "Please review the warnings and errors above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Quick Test Commands:" -ForegroundColor Cyan
Write-Host "  npx @modelcontextprotocol/inspector --config `"C:\Users\david\.claude.json`"" -ForegroundColor White
Write-Host "  npx @modelcontextprotocol/inspector --config `"C:\Users\david\.claude\mcp.json`"" -ForegroundColor White