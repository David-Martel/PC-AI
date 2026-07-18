# MCP Server Testing Script
# Tests all servers in .warp_mcp_config.json systematically

$ErrorActionPreference = "Continue"
$ConfigPath = "C:/Users/david/.warp_mcp_config.json"
$ResultsFile = "C:/Users/david/.claude/MCP_TEST_RESULTS.md"

Write-Host "=== MCP Server Comprehensive Testing ===" -ForegroundColor Green
Write-Host "Config: $ConfigPath" -ForegroundColor Yellow
Write-Host "Results will be saved to: $ResultsFile" -ForegroundColor Yellow

# Initialize results file
@"
# MCP Server Test Results
Generated: $(Get-Date)
Config: $ConfigPath

## Test Summary
"@ | Out-File -FilePath $ResultsFile -Encoding UTF8

# Load configuration
try {
    $config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
    $servers = $config.mcpServers
    Write-Host "Found $($servers.PSObject.Properties.Count) servers to test" -ForegroundColor Cyan
} catch {
    Write-Error "Failed to load MCP config: $_"
    exit 1
}

$successCount = 0
$failureCount = 0
$results = @()

foreach ($serverName in $servers.PSObject.Properties.Name) {
    $server = $servers.$serverName
    Write-Host "`n--- Testing Server: $serverName ---" -ForegroundColor Magenta

    $testResult = @{
        Name = $serverName
        Command = $server.command
        Status = "Unknown"
        Issues = @()
        Tools = @()
    }

    # Test 1: Check if command exists
    $commandExists = $false
    try {
        if ($server.command -like "*.exe") {
            $commandExists = Test-Path $server.command
        } elseif ($server.command -eq "bun") {
            $bunCheck = & where.exe bun 2>$null
            $commandExists = $bunCheck -ne $null
        } elseif ($server.command -eq "npx") {
            $npxCheck = & where.exe npx 2>$null
            $commandExists = $npxCheck -ne $null
        } elseif ($server.command -eq "uv") {
            $uvCheck = & where.exe uv 2>$null
            $commandExists = $uvCheck -ne $null
        } elseif ($server.command -eq "node") {
            $nodeCheck = & where.exe node 2>$null
            $commandExists = $nodeCheck -ne $null
        } elseif ($server.command -eq "docker") {
            $dockerCheck = & where.exe docker 2>$null
            $commandExists = $dockerCheck -ne $null
        } else {
            $commandExists = Test-Path $server.command
        }

        if ($commandExists) {
            Write-Host "✓ Command exists: $($server.command)" -ForegroundColor Green
        } else {
            Write-Host "✗ Command not found: $($server.command)" -ForegroundColor Red
            $testResult.Issues += "Command not found: $($server.command)"
        }
    } catch {
        Write-Host "✗ Error checking command: $_" -ForegroundColor Red
        $testResult.Issues += "Error checking command: $_"
        $commandExists = $false
    }
    # Test 2: Check working directory if specified
    if ($server.cwd) {
        if (Test-Path $server.cwd) {
            Write-Host "✓ Working directory exists: $($server.cwd)" -ForegroundColor Green
        } else {
            Write-Host "✗ Working directory not found: $($server.cwd)" -ForegroundColor Red
            $testResult.Issues += "Working directory not found: $($server.cwd)"
        }
    }

    # Test 3: Test server startup (if command exists)
    if ($commandExists) {
        Write-Host "Testing server startup..." -ForegroundColor Yellow
        try {
            # Test with timeout using Claude CLI
            $testOutput = & claude --mcp-config $ConfigPath --timeout 10 -p "Test connectivity to $serverName server only. List available tools." 2>&1

            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Server startup test passed" -ForegroundColor Green
                $testResult.Status = "Working"
                $successCount++

                # Try to extract tool information from output
                if ($testOutput -match "tools|functions|capabilities") {
                    $testResult.Tools += "Tools available (see detailed output)"
                }
            } else {
                Write-Host "✗ Server startup test failed" -ForegroundColor Red
                $testResult.Status = "Failed"
                $testResult.Issues += "Server startup failed: $testOutput"
                $failureCount++
            }
        } catch {
            Write-Host "✗ Error during server test: $_" -ForegroundColor Red
            $testResult.Status = "Error"
            $testResult.Issues += "Test error: $_"
            $failureCount++
        }
    } else {
        $testResult.Status = "Command Not Found"
        $failureCount++
    }

    $results += $testResult
    Write-Host "Status: $($testResult.Status)" -ForegroundColor $(if ($testResult.Status -eq "Working") { "Green" } else { "Red" })
}

# Generate comprehensive report
$report = @"
# MCP Server Test Results
Generated: $(Get-Date)
Config: $ConfigPath

## Summary
- **Total Servers**: $($results.Count)
- **Working**: $successCount
- **Failed**: $failureCount
- **Success Rate**: $([math]::Round($successCount / $results.Count * 100, 1))%

## Detailed Results

"@

foreach ($result in $results) {
    $report += @"
### $($result.Name)
- **Command**: ``$($result.Command)``
- **Status**: $($result.Status)
- **Issues**: $($result.Issues.Count)

"@

    if ($result.Issues.Count -gt 0) {
        $report += "**Issues Found:**`n"
        foreach ($issue in $result.Issues) {
            $report += "- $issue`n"
        }
        $report += "`n"
    }

    if ($result.Tools.Count -gt 0) {
        $report += "**Tools Available:**`n"
        foreach ($tool in $result.Tools) {
            $report += "- $tool`n"
        }
        $report += "`n"
    }
}

$report += @"

## Recommendations

### Servers Needing Attention
"@

$brokenServers = $results | Where-Object { $_.Status -ne "Working" }
foreach ($server in $brokenServers) {
    $report += "- **$($server.Name)**: $($server.Status) - $($server.Issues -join '; ')`n"
}

$report += @"

### Next Steps
1. Install missing dependencies for failed servers
2. Fix configuration issues identified above
3. Re-run tests after fixes
4. Document working tool capabilities

"@

$report | Out-File -FilePath $ResultsFile -Encoding UTF8

Write-Host "`n=== Testing Complete ===" -ForegroundColor Green
Write-Host "Results saved to: $ResultsFile" -ForegroundColor Yellow
Write-Host "Working servers: $successCount/$($results.Count)" -ForegroundColor Cyan

# Display summary
Write-Host "`nWorking Servers:" -ForegroundColor Green
$results | Where-Object { $_.Status -eq "Working" } | ForEach-Object { Write-Host "  ✓ $($_.Name)" -ForegroundColor Green }

Write-Host "`nFailed Servers:" -ForegroundColor Red
$results | Where-Object { $_.Status -ne "Working" } | ForEach-Object { Write-Host "  ✗ $($_.Name) - $($_.Status)" -ForegroundColor Red }
