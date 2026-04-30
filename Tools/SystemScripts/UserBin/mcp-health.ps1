#Requires -Version 5.1
<#
.SYNOPSIS
    MCP Server health monitor
.DESCRIPTION
    Validates all configured MCP servers can start and respond
    Part of System Configuration Improvement Plan - Phase 5
.EXAMPLE
    .\mcp-health.ps1
    .\mcp-health.ps1 -ConfigPath "C:\Users\david\.claude\mcp.json" -Verbose
#>

[CmdletBinding()]
param(
    [string]$ConfigPath = "$env:USERPROFILE\.claude\mcp.json",
    [string[]]$ServerNames,
    [int]$TimeoutSeconds = 10,
    [switch]$Json,
    [switch]$Quick
)

function Test-McpServer {
    param(
        [string]$Name,
        [hashtable]$Config,
        [int]$Timeout
    )

    $result = @{
        Name = $Name
        Status = "unknown"
        Command = $Config.command
        StartTime = $null
        ResponseTime = $null
        Error = $null
    }

    try {
        $command = $Config.command

        # Check if command exists
        if (-not (Test-Path $command -PathType Leaf)) {
            $result.Status = "command_not_found"
            $result.Error = "Command not found: $command"
            return $result
        }

        # Test startup (quick check)
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = $command
        $pinfo.Arguments = ($Config.args -join " ")
        $pinfo.UseShellExecute = $false
        $pinfo.RedirectStandardOutput = $true
        $pinfo.RedirectStandardError = $true
        $pinfo.CreateNoWindow = $true

        # Set environment with safe variable expansion
        if ($Config.env) {
            foreach ($key in $Config.env.Keys) {
                # Validate key name to prevent injection
                if ($key -notmatch '^[A-Z_][A-Z0-9_]*$') {
                    Write-Verbose "Skipping invalid environment variable name: $key"
                    continue
                }

                $value = $Config.env[$key]
                # Expand environment variables safely
                if ($value -match '\$\{env:(\w+)\}') {
                    $envVar = $Matches[1]
                    $envValue = [Environment]::GetEnvironmentVariable($envVar)
                    if ($null -eq $envValue) {
                        Write-Verbose "Environment variable '$envVar' not found for server $Name"
                        $envValue = ""
                    }
                    $value = $value -replace '\$\{env:\w+\}', $envValue
                }
                $pinfo.EnvironmentVariables[$key] = $value
            }
        }

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $pinfo

        $started = $process.Start()
        if (-not $started) {
            $result.Status = "failed_to_start"
            return $result
        }

        # Wait briefly to see if process stays alive
        Start-Sleep -Milliseconds 500

        if ($process.HasExited) {
            $result.Status = "exited_immediately"
            $result.Error = $process.StandardError.ReadToEnd()
        }
        else {
            $result.Status = "running"
            $result.StartTime = $stopwatch.ElapsedMilliseconds
            $process.Kill()
        }

        $stopwatch.Stop()
        $result.ResponseTime = $stopwatch.ElapsedMilliseconds
    }
    catch {
        $result.Status = "error"
        $result.Error = $_.Exception.Message
    }

    return $result
}

# Load MCP configuration
if (-not (Test-Path $ConfigPath)) {
    Write-Error "MCP config not found: $ConfigPath"
    exit 1
}

try {
    $jsonContent = Get-Content $ConfigPath -Raw
    # PowerShell 5.1 compatible JSON parsing (no -AsHashtable)
    $configObj = $jsonContent | ConvertFrom-Json

    # Convert PSCustomObject to hashtable for iteration
    $config = @{}
    if ($configObj.mcpServers) {
        $config['mcpServers'] = @{}
        foreach ($prop in $configObj.mcpServers.PSObject.Properties) {
            $serverConfig = @{}
            foreach ($p in $prop.Value.PSObject.Properties) {
                $serverConfig[$p.Name] = $p.Value
            }
            $config['mcpServers'][$prop.Name] = $serverConfig
        }
    }
}
catch {
    Write-Error "Failed to parse MCP config: $_"
    exit 1
}

$servers = $config.mcpServers

# Filter servers if specified
if ($ServerNames) {
    $servers = @{}
    foreach ($name in $ServerNames) {
        if ($config.mcpServers.ContainsKey($name)) {
            $servers[$name] = $config.mcpServers[$name]
        }
    }
}

if ($servers.Count -eq 0) {
    Write-Warning "No MCP servers to check"
    exit 0
}

# Test each server
$results = @()
$totalCount = $servers.Count
$currentIndex = 0

foreach ($serverName in $servers.Keys) {
    $currentIndex++
    $serverConfig = $servers[$serverName]

    if (-not $Quick) {
        Write-Progress -Activity "Testing MCP Servers" -Status $serverName -PercentComplete (($currentIndex / $totalCount) * 100)
    }

    Write-Verbose "Testing server: $serverName"
    $result = Test-McpServer -Name $serverName -Config $serverConfig -Timeout $TimeoutSeconds
    $results += $result
}

if (-not $Quick) {
    Write-Progress -Activity "Testing MCP Servers" -Completed
}

# Output results
if ($Json) {
    @{
        timestamp = (Get-Date -Format "o")
        config_path = $ConfigPath
        servers_tested = $results.Count
        servers_healthy = ($results | Where-Object { $_.Status -eq "running" }).Count
        results = $results
    } | ConvertTo-Json -Depth 4
}
else {
    Write-Host ""
    Write-Host "MCP Server Health Check" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkGray
    Write-Host "Config: $ConfigPath" -ForegroundColor DarkGray
    Write-Host ""

    foreach ($result in $results) {
        $statusColor = switch ($result.Status) {
            "running" { "Green" }
            "command_not_found" { "Red" }
            "failed_to_start" { "Red" }
            "exited_immediately" { "Yellow" }
            default { "DarkGray" }
        }

        Write-Host "$($result.Name): " -NoNewline
        Write-Host $result.Status -ForegroundColor $statusColor

        if ($result.Error -and $VerbosePreference -eq 'Continue') {
            Write-Host "  Error: $($result.Error)" -ForegroundColor DarkGray
        }

        if ($result.ResponseTime) {
            Write-Host "  Startup: $($result.ResponseTime)ms" -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    $runningCount = ($results | Where-Object { $_.Status -eq "running" }).Count
    Write-Host "$runningCount/$($results.Count) servers healthy" -ForegroundColor $(if ($runningCount -eq $results.Count) { "Green" } else { "Yellow" })
}

# Return exit code
$unhealthyCount = ($results | Where-Object { $_.Status -ne "running" }).Count
exit $unhealthyCount
