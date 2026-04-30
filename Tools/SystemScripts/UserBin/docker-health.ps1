#Requires -Version 5.1
<#
.SYNOPSIS
    Docker container health check script
.DESCRIPTION
    Monitors Docker containers health status and reports status
    Part of System Configuration Improvement Plan - Phase 5
.EXAMPLE
    .\docker-health.ps1
    .\docker-health.ps1 -ContainerNames "rag-redis-backend","rag-mcp-server"
#>

[CmdletBinding()]
param(
    [string[]]$ContainerNames = @("rag-redis-backend", "rag-mcp-server"),
    [switch]$Json,
    [switch]$Quiet
)

function Get-ContainerHealth {
    param(
        [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')]
        [string]$Name
    )

    # Security: Validate container name to prevent command injection
    if ($Name -match '[;&|`$<>]') {
        return @{
            Name = $Name
            Status = "invalid_name"
            Health = "N/A"
            Running = $false
            Error = "Container name contains invalid characters"
        }
    }

    try {
        # Use -- to separate options from arguments (prevents option injection)
        $inspect = docker inspect --format='{{json .State}}' -- $Name 2>&1
        if ($LASTEXITCODE -ne 0) {
            return @{
                Name = $Name
                Status = "not_found"
                Health = "N/A"
                Running = $false
            }
        }

        $state = $inspect | ConvertFrom-Json
        $health = if ($state.Health) { $state.Health.Status } else { "no_healthcheck" }

        return @{
            Name = $Name
            Status = $state.Status
            Health = $health
            Running = $state.Running
            StartedAt = $state.StartedAt
            Pid = $state.Pid
        }
    }
    catch {
        return @{
            Name = $Name
            Status = "error"
            Health = "N/A"
            Running = $false
            Error = $_.Exception.Message
        }
    }
}

function Write-StatusLine {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Health
    )

    $statusColor = switch ($Status) {
        "running" { "Green" }
        "exited" { "Red" }
        "not_found" { "DarkGray" }
        default { "Yellow" }
    }

    $healthColor = switch ($Health) {
        "healthy" { "Green" }
        "unhealthy" { "Red" }
        "starting" { "Yellow" }
        default { "DarkGray" }
    }

    Write-Host "$Name`: " -NoNewline
    Write-Host $Status -ForegroundColor $statusColor -NoNewline
    Write-Host " | Health: " -NoNewline
    Write-Host $Health -ForegroundColor $healthColor
}

# Check if Docker is running
$dockerRunning = docker info 2>&1 | Out-Null; $LASTEXITCODE -eq 0
if (-not $dockerRunning) {
    if ($Json) {
        @{
            timestamp = (Get-Date -Format "o")
            docker_running = $false
            error = "Docker daemon not running"
        } | ConvertTo-Json
    }
    else {
        Write-Host "Docker daemon is not running" -ForegroundColor Red
    }
    exit 1
}

# Get health for all containers
$results = @()
foreach ($container in $ContainerNames) {
    $health = Get-ContainerHealth -Name $container
    $results += $health
}

# Output results
if ($Json) {
    @{
        timestamp = (Get-Date -Format "o")
        docker_running = $true
        containers = $results
        all_healthy = ($results | Where-Object { $_.Health -eq "healthy" }).Count -eq $results.Count
    } | ConvertTo-Json -Depth 3
}
elseif (-not $Quiet) {
    Write-Host ""
    Write-Host "Docker Container Health Check" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkGray

    foreach ($result in $results) {
        Write-StatusLine -Name $result.Name -Status $result.Status -Health $result.Health
    }

    Write-Host ""
    $healthyCount = ($results | Where-Object { $_.Health -eq "healthy" }).Count
    $totalCount = $results.Count

    if ($healthyCount -eq $totalCount) {
        Write-Host "All $totalCount containers healthy" -ForegroundColor Green
    }
    else {
        Write-Host "$healthyCount/$totalCount containers healthy" -ForegroundColor Yellow
    }
}

# Return exit code based on health
$unhealthyCount = ($results | Where-Object { $_.Health -ne "healthy" -and $_.Health -ne "no_healthcheck" }).Count
exit $unhealthyCount
