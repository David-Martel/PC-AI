# Quick MCP Server Availability Test
$ConfigPath = "C:/Users/david/.warp_mcp_config.json"

Write-Host "=== Quick MCP Server Test ===" -ForegroundColor Green

# Load config
$config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
$servers = $config.mcpServers

$working = @()
$failed = @()

# Test basic command availability
foreach ($serverName in $servers.PSObject.Properties.Name) {
    $server = $servers.$serverName
    $cmd = $server.command

    Write-Host "Testing $serverName... " -NoNewline

    try {
        $exists = $false
        if ($cmd -like "*.exe") {
            $exists = Test-Path $cmd
        } elseif ($cmd -in @("bun", "npx", "uv", "node", "docker", "uvx")) {
            $check = & where.exe $cmd 2>$null
            $exists = $check -ne $null
        } else {
            $exists = Test-Path $cmd
        }

        if ($exists) {
            Write-Host "✓" -ForegroundColor Green
            $working += $serverName
        } else {
            Write-Host "✗ (command not found)" -ForegroundColor Red
            $failed += "$serverName (command not found: $cmd)"
        }
    } catch {
        Write-Host "✗ (error)" -ForegroundColor Red
        $failed += "$serverName (error: $_)"
    }
}

Write-Host "`n=== Results ===" -ForegroundColor Cyan
Write-Host "Working servers ($($working.Count)): $($working -join ', ')" -ForegroundColor Green
Write-Host "Failed servers ($($failed.Count)):" -ForegroundColor Red
$failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
