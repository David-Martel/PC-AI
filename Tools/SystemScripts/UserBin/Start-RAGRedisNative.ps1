#Requires -Version 5.1
<#
.SYNOPSIS
    Starts RAG-Redis system using native Windows binaries.
.DESCRIPTION
    Part of System Configuration Improvement Plan - Phase 5
    Supports both Docker-based and native Redis modes.
    Uses pre-built binaries from W:\dropbox-local\rag-redis-build.
.EXAMPLE
    .\Start-RAGRedisNative.ps1 -Mode Native
    Start using native Windows Redis on port 6380
.EXAMPLE
    .\Start-RAGRedisNative.ps1 -Mode Docker
    Start using Docker containers on port 6379
.EXAMPLE
    .\Start-RAGRedisNative.ps1 -Status
    Check status of all RAG-Redis components
#>

[CmdletBinding()]
param(
    [ValidateSet("Native", "Docker", "Auto")]
    [string]$Mode = "Auto",
    [switch]$Status,
    [switch]$Stop,
    [string]$LogPath = "$env:USERPROFILE\.machine\logs\rag-redis-native.log"
)

# Configuration
# Port mapping:
#   6379 - Docker Redis (Redis Stack with RediSearch, ReJSON, VectorSet modules) - RECOMMENDED
#   6380 - Native Windows Redis (basic, no modules)
# The RAG MCP server requires RediSearch module, so Docker Redis (6379) is required

$NativeRedisPath = "W:\dropbox-local\rag-redis-build\bin"
$NativeRagPath = "W:\dropbox-local\rag-redis-build\rag-binaries\bin"
$McpServerPath = "C:\Users\david\bin\rag-redis-mcp-server.exe"
$DockerComposePath = "C:\codedev\llm\rag-redis"
$NativeRedisPort = 6380  # Windows-only, no modules
$DockerRedisPort = 6379  # Docker Redis Stack with modules (REQUIRED for RAG)
$RagServerPort = 8080

# Ensure log directory exists
$logDir = Split-Path $LogPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LogPath -Value $logEntry

    $color = switch ($Level) {
        "INFO" { "Cyan" }
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host $logEntry -ForegroundColor $color
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Test-RedisConnection {
    param([int]$Port)
    try {
        $result = & redis-cli -p $Port ping 2>&1
        return $result -eq "PONG"
    } catch {
        return $false
    }
}

function Get-SystemStatus {
    Write-Host ""
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkGray
    Write-Host "  RAG-Redis System Status" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkGray

    # Check native Redis binary availability
    $nativeRedisAvailable = Test-Path "$NativeRedisPath\redis-server.exe"
    Write-Host "Native Redis binaries: " -NoNewline
    if ($nativeRedisAvailable) {
        Write-Host "Available" -ForegroundColor Green
    } else {
        Write-Host "Not Found" -ForegroundColor Red
    }

    # Check native RAG binaries
    $nativeRagAvailable = (Test-Path "$NativeRagPath\rag-cli.exe") -and (Test-Path "$NativeRagPath\rag-server.exe")
    Write-Host "Native RAG binaries: " -NoNewline
    if ($nativeRagAvailable) {
        Write-Host "Available" -ForegroundColor Green
    } else {
        Write-Host "Not Found" -ForegroundColor Red
    }

    # Check Docker availability
    $dockerRunning = $false
    try {
        $null = docker info 2>&1
        $dockerRunning = $LASTEXITCODE -eq 0
    } catch {}
    Write-Host "Docker daemon: " -NoNewline
    if ($dockerRunning) {
        Write-Host "Running" -ForegroundColor Green
    } else {
        Write-Host "Not Running" -ForegroundColor Red
    }

    # Check ports
    Write-Host ""
    Write-Host "Port Status:" -ForegroundColor Cyan

    # Native Redis port (6380)
    $port6380InUse = Test-PortInUse -Port $NativeRedisPort
    $redis6380Alive = Test-RedisConnection -Port $NativeRedisPort
    Write-Host "  Port $NativeRedisPort (native): " -NoNewline
    if ($redis6380Alive) {
        Write-Host "Redis RUNNING" -ForegroundColor Green
    } elseif ($port6380InUse) {
        Write-Host "In use (not Redis)" -ForegroundColor Yellow
    } else {
        Write-Host "Available" -ForegroundColor DarkGray
    }

    # Docker Redis port (6379)
    $port6379InUse = Test-PortInUse -Port $DockerRedisPort
    $redis6379Alive = Test-RedisConnection -Port $DockerRedisPort
    Write-Host "  Port $DockerRedisPort (docker): " -NoNewline
    if ($redis6379Alive) {
        Write-Host "Redis RUNNING" -ForegroundColor Green
    } elseif ($port6379InUse) {
        Write-Host "In use (not Redis)" -ForegroundColor Yellow
    } else {
        Write-Host "Available" -ForegroundColor DarkGray
    }

    # RAG Server port
    $port8080InUse = Test-PortInUse -Port $RagServerPort
    Write-Host "  Port $RagServerPort (rag-server): " -NoNewline
    if ($port8080InUse) {
        Write-Host "In use" -ForegroundColor Green
    } else {
        Write-Host "Available" -ForegroundColor DarkGray
    }

    # Docker containers
    if ($dockerRunning) {
        Write-Host ""
        Write-Host "Docker Containers:" -ForegroundColor Cyan
        try {
            $redisContainer = docker inspect --format='{{.State.Status}}' rag-redis-backend 2>&1
            Write-Host "  rag-redis-backend: " -NoNewline
            if ($redisContainer -eq "running") {
                Write-Host $redisContainer -ForegroundColor Green
            } else {
                Write-Host $redisContainer -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  rag-redis-backend: not found" -ForegroundColor DarkGray
        }

        try {
            $mcpContainer = docker inspect --format='{{.State.Status}}' rag-mcp-server 2>&1
            Write-Host "  rag-mcp-server: " -NoNewline
            if ($mcpContainer -eq "running") {
                Write-Host $mcpContainer -ForegroundColor Green
            } else {
                Write-Host $mcpContainer -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  rag-mcp-server: not found" -ForegroundColor DarkGray
        }
    }

    # RAG CLI status
    Write-Host ""
    Write-Host "RAG CLI Status:" -ForegroundColor Cyan
    if ($nativeRagAvailable) {
        try {
            $ragStatus = & "$NativeRagPath\rag-cli.exe" status --output json 2>&1 | ConvertFrom-Json
            Write-Host "  Mode: $($ragStatus.components.mode)" -ForegroundColor $(if ($ragStatus.components.mode -eq "docker") { "Green" } else { "Yellow" })
            Write-Host "  Redis: $($ragStatus.components.redis)" -ForegroundColor $(if ($ragStatus.components.redis -eq "connected") { "Green" } else { "Red" })
            Write-Host "  RAG System: $($ragStatus.components.rag_system)" -ForegroundColor $(if ($ragStatus.components.rag_system -eq "initialized") { "Green" } else { "Yellow" })
        } catch {
            Write-Host "  Unable to get status" -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkGray

    # Recommendation
    Write-Host ""
    if ($redis6380Alive) {
        Write-Host "Recommendation: Use Native mode (Redis running on $NativeRedisPort)" -ForegroundColor Cyan
    } elseif ($redis6379Alive) {
        Write-Host "Recommendation: Update config to use port 6379 (Docker Redis running)" -ForegroundColor Yellow
    } else {
        Write-Host "Recommendation: Start Native Redis with -Mode Native" -ForegroundColor Yellow
    }
}

function Stop-AllRAGServices {
    Write-Log "Stopping all RAG-Redis services..." "INFO"

    # Stop native Redis if running
    $redisProc = Get-Process -Name "redis-server" -ErrorAction SilentlyContinue
    if ($redisProc) {
        Write-Log "Stopping native Redis server..." "INFO"
        & redis-cli -p $NativeRedisPort shutdown nosave 2>&1 | Out-Null
        Start-Sleep -Seconds 2
    }

    # Stop rag-server if running
    $ragProc = Get-Process -Name "rag-server" -ErrorAction SilentlyContinue
    if ($ragProc) {
        Write-Log "Stopping RAG server..." "INFO"
        Stop-Process -Name "rag-server" -Force -ErrorAction SilentlyContinue
    }

    # Stop Docker containers
    $dockerRunning = $false
    try {
        $null = docker info 2>&1
        $dockerRunning = $LASTEXITCODE -eq 0
    } catch {}

    if ($dockerRunning) {
        Write-Log "Stopping Docker containers..." "INFO"
        docker stop rag-redis-backend rag-mcp-server 2>&1 | Out-Null
    }

    Write-Log "All services stopped" "SUCCESS"
}

function Start-NativeMode {
    Write-Log "Starting RAG-Redis in Native mode..." "INFO"

    # Verify binaries exist
    if (-not (Test-Path "$NativeRedisPath\redis-server.exe")) {
        Write-Log "Native Redis not found at $NativeRedisPath" "ERROR"
        return $false
    }

    # Check if Redis already running on target port
    if (Test-RedisConnection -Port $NativeRedisPort) {
        Write-Log "Redis already running on port $NativeRedisPort" "INFO"
    } else {
        # Start native Redis
        Write-Log "Starting Redis on port $NativeRedisPort..." "INFO"

        $redisArgs = @(
            "--port", $NativeRedisPort,
            "--daemonize", "no",
            "--loglevel", "notice"
        )

        Start-Process -FilePath "$NativeRedisPath\redis-server.exe" `
            -ArgumentList $redisArgs `
            -WindowStyle Hidden `
            -PassThru | Out-Null

        # Wait for Redis to start
        $maxWait = 30
        $waited = 0
        while (-not (Test-RedisConnection -Port $NativeRedisPort) -and $waited -lt $maxWait) {
            Start-Sleep -Seconds 1
            $waited++
        }

        if (Test-RedisConnection -Port $NativeRedisPort) {
            Write-Log "Redis started successfully on port $NativeRedisPort" "SUCCESS"
        } else {
            Write-Log "Failed to start Redis after ${waited}s" "ERROR"
            return $false
        }
    }

    return $true
}

function Start-DockerMode {
    Write-Log "Starting RAG-Redis in Docker mode..." "INFO"

    # Check Docker
    $dockerRunning = $false
    try {
        $null = docker info 2>&1
        $dockerRunning = $LASTEXITCODE -eq 0
    } catch {}

    if (-not $dockerRunning) {
        Write-Log "Docker daemon is not running" "ERROR"
        return $false
    }

    # Start containers
    Push-Location $DockerComposePath
    try {
        $result = docker-compose up -d redis rag-mcp-server 2>&1
        Write-Log "docker-compose: $result" "INFO"

        # Wait for containers
        Start-Sleep -Seconds 10

        if (Test-RedisConnection -Port $DockerRedisPort) {
            Write-Log "Docker Redis started on port $DockerRedisPort" "SUCCESS"
            return $true
        } else {
            Write-Log "Docker Redis failed to start" "ERROR"
            return $false
        }
    } finally {
        Pop-Location
    }
}

function Start-AutoMode {
    Write-Log "Auto-detecting best startup mode..." "INFO"

    # Check if native Redis already running
    if (Test-RedisConnection -Port $NativeRedisPort) {
        Write-Log "Native Redis already running on $NativeRedisPort - using Native mode" "INFO"
        return $true
    }

    # Check if Docker Redis already running
    if (Test-RedisConnection -Port $DockerRedisPort) {
        Write-Log "Docker Redis already running on $DockerRedisPort" "WARNING"
        Write-Log "Note: Config expects port $NativeRedisPort - consider updating config or starting native Redis" "WARNING"
        return $true
    }

    # Check if native binaries available
    if (Test-Path "$NativeRedisPath\redis-server.exe") {
        Write-Log "Native binaries available - using Native mode" "INFO"
        return Start-NativeMode
    }

    # Fall back to Docker
    Write-Log "Falling back to Docker mode" "INFO"
    return Start-DockerMode
}

# Main execution
Write-Log "═══════════════════════════════════════" "INFO"
Write-Log "RAG-Redis Native Startup Script" "INFO"

if ($Status) {
    Get-SystemStatus
    exit 0
}

if ($Stop) {
    Stop-AllRAGServices
    exit 0
}

$success = switch ($Mode) {
    "Native" { Start-NativeMode }
    "Docker" { Start-DockerMode }
    "Auto"   { Start-AutoMode }
}

if ($success) {
    Write-Log "RAG-Redis system started successfully" "SUCCESS"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  Check status: .\Start-RAGRedisNative.ps1 -Status" -ForegroundColor DarkGray
    Write-Host "  Test CLI:     W:\dropbox-local\rag-redis-build\rag-binaries\bin\rag-cli.exe status" -ForegroundColor DarkGray
    exit 0
} else {
    Write-Log "Failed to start RAG-Redis system" "ERROR"
    exit 1
}
