#Requires -Version 5.1
<#
.SYNOPSIS
    Comprehensive RAG-Redis system health check.
.DESCRIPTION
    Part of System Configuration Improvement Plan - Phase 5
    Validates all RAG-Redis components: Docker, WSL, Redis, MCP Server
.EXAMPLE
    .\Test-RAGRedisHealth.ps1
    .\Test-RAGRedisHealth.ps1 -Json
    .\Test-RAGRedisHealth.ps1 -Fix
#>

[CmdletBinding()]
param(
    [switch]$Json,
    [switch]$Fix,
    [switch]$Quick
)

$results = @{
    timestamp = (Get-Date -Format "o")
    overall_status = "unknown"
    components = @{}
}

function Test-ComponentHealth {
    param([string]$Name, [scriptblock]$Test, [scriptblock]$FixAction)

    $component = @{
        name = $Name
        status = "unknown"
        message = $null
        details = $null
    }

    try {
        $result = & $Test
        if ($result.success) {
            $component.status = "healthy"
            $component.details = $result.details
        } else {
            $component.status = "unhealthy"
            $component.message = $result.message

            if ($Fix -and $FixAction) {
                Write-Host "  Attempting fix for $Name..." -ForegroundColor Yellow
                try {
                    & $FixAction
                    Start-Sleep -Seconds 3
                    $retest = & $Test
                    if ($retest.success) {
                        $component.status = "fixed"
                        $component.message = "Auto-repaired"
                    }
                } catch {
                    $component.message = "Fix failed: $_"
                }
            }
        }
    } catch {
        $component.status = "error"
        $component.message = $_.Exception.Message
    }

    return $component
}

# Test 1: Redis Windows Service on 6380 (PRIMARY)
$results.components["redis_service"] = Test-ComponentHealth -Name "Redis Windows Service (6380)" -Test {
    $svc = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if (-not $svc) {
        @{ success = $false; message = "Redis Windows Service not installed" }
    } elseif ($svc.Status -ne "Running") {
        @{ success = $false; message = "Redis Windows Service exists but not running (Status: $($svc.Status))" }
    } else {
        $ping = redis-cli -p 6380 ping 2>&1
        if ($ping -eq "PONG") {
            $info = redis-cli -p 6380 info server 2>&1
            $version = ($info | Select-String "redis_version:(.+)").Matches.Groups[1].Value.Trim()
            @{
                success = $true
                details = @{
                    port = 6380
                    type = "Windows Service"
                    service_status = "Running"
                    redis_version = $version
                }
            }
        } else {
            @{ success = $false; message = "Redis service running but not responding on port 6380" }
        }
    }
} -FixAction {
    $svc = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($svc) {
        Start-Service -Name "Redis" -ErrorAction SilentlyContinue
    }
}

# Test 2: Docker Redis on 6379 (optional, for RediSearch modules)
if (-not $Quick) {
    $results.components["redis_docker"] = Test-ComponentHealth -Name "Docker Redis (6379)" -Test {
        $ping = redis-cli -p 6379 ping 2>&1
        if ($ping -eq "PONG") {
            $modules = redis-cli -p 6379 module list 2>&1
            $hasSearch = $modules -match "search"
            $hasReJSON = $modules -match "ReJSON"

            @{
                success = $true
                details = @{
                    port = 6379
                    has_redisearch = [bool]$hasSearch
                    has_rejson = [bool]$hasReJSON
                    modules = "RediSearch" + $(if ($hasReJSON) { ", ReJSON" } else { "" })
                }
            }
        } else {
            @{ success = $false; message = "Docker Redis not running on port 6379 (optional)" }
        }
    } -FixAction {
        Push-Location "C:\codedev\llm\rag-redis"
        docker-compose up -d redis
        Pop-Location
    }
}

# Test 4: MCP Server Binary
$results.components["mcp_binary"] = Test-ComponentHealth -Name "MCP Server Binary" -Test {
    $path = "C:\Users\david\bin\rag-redis-mcp-server.exe"
    if (Test-Path $path) {
        $info = Get-Item $path
        @{
            success = $true
            details = @{
                path = $path
                size_mb = [math]::Round($info.Length / 1MB, 2)
                modified = $info.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
            }
        }
    } else {
        @{ success = $false; message = "MCP server binary not found" }
    }
} -FixAction {
    Copy-Item "W:\dropbox-local\rag-redis-build\target\release\mcp-server.exe" "C:\Users\david\bin\rag-redis-mcp-server.exe" -Force
}

# Test 5: MCP Server Startup
if (-not $Quick) {
    $results.components["mcp_startup"] = Test-ComponentHealth -Name "MCP Server Startup" -Test {
        $env:REDIS_URL = "redis://127.0.0.1:6380"
        $proc = Start-Process -FilePath "C:\Users\david\bin\rag-redis-mcp-server.exe" `
            -RedirectStandardError "$env:TEMP\mcp-test-err.txt" `
            -PassThru -NoNewWindow

        Start-Sleep -Milliseconds 500

        if (-not $proc.HasExited) {
            $proc.Kill()
            @{ success = $true; details = "MCP server starts successfully" }
        } else {
            $err = Get-Content "$env:TEMP\mcp-test-err.txt" -Raw -ErrorAction SilentlyContinue
            if ($err -match "Redis connection pool initialized") {
                @{ success = $true; details = "MCP server initialized successfully" }
            } else {
                @{ success = $false; message = "MCP server failed to start: $err" }
            }
        }
    }
}

# Test 6: WSL Status (if Docker uses WSL2)
if (-not $Quick) {
    $results.components["wsl"] = Test-ComponentHealth -Name "WSL Status" -Test {
        $status = wsl -l -v 2>&1
        if ($LASTEXITCODE -eq 0) {
            $ubuntu = $status | Select-String "Ubuntu"
            @{
                success = $true
                details = @{
                    distributions = ($status | Select-String "^\*?\s+\w+" | ForEach-Object { $_.Line.Trim() })
                }
            }
        } else {
            @{ success = $false; message = "WSL not available" }
        }
    }
}

# Test 7: RAG Data Directories
$results.components["data_dirs"] = Test-ComponentHealth -Name "Data Directories" -Test {
    $dirs = @(
        "C:\codedev\llm\rag-redis\data\rag",
        "C:\codedev\llm\rag-redis\cache\embeddings",
        "C:\codedev\llm\rag-redis\logs"
    )
    $missing = $dirs | Where-Object { -not (Test-Path $_) }
    if ($missing.Count -eq 0) {
        @{ success = $true; details = @{ directories = $dirs.Count } }
    } else {
        @{ success = $false; message = "Missing directories: $($missing -join ', ')" }
    }
} -FixAction {
    foreach ($dir in @(
        "C:\codedev\llm\rag-redis\data\rag",
        "C:\codedev\llm\rag-redis\cache\embeddings",
        "C:\codedev\llm\rag-redis\logs"
    )) {
        New-Item -ItemType Directory -Path $dir -Force -ErrorAction SilentlyContinue | Out-Null
    }
}

# Calculate overall status
$statuses = $results.components.Values | ForEach-Object { $_.status }
$unhealthyCount = ($statuses | Where-Object { $_ -eq "unhealthy" -or $_ -eq "error" }).Count
$healthyCount = ($statuses | Where-Object { $_ -eq "healthy" -or $_ -eq "fixed" }).Count

if ($unhealthyCount -eq 0) {
    $results.overall_status = "healthy"
} elseif ($healthyCount -gt $unhealthyCount) {
    $results.overall_status = "degraded"
} else {
    $results.overall_status = "unhealthy"
}

# Output
if ($Json) {
    $results | ConvertTo-Json -Depth 5
} else {
    Write-Host ""
    Write-Host "RAG-Redis System Health Check" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor DarkGray

    foreach ($key in $results.components.Keys) {
        $comp = $results.components[$key]
        $statusColor = switch ($comp.status) {
            "healthy" { "Green" }
            "fixed" { "Yellow" }
            "unhealthy" { "Red" }
            "error" { "Red" }
            default { "DarkGray" }
        }

        Write-Host "  $($comp.name): " -NoNewline
        Write-Host $comp.status -ForegroundColor $statusColor

        if ($comp.message) {
            Write-Host "    $($comp.message)" -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor DarkGray

    $overallColor = switch ($results.overall_status) {
        "healthy" { "Green" }
        "degraded" { "Yellow" }
        default { "Red" }
    }
    Write-Host "Overall: " -NoNewline
    Write-Host $results.overall_status.ToUpper() -ForegroundColor $overallColor

    if ($results.overall_status -ne "healthy") {
        Write-Host ""
        Write-Host "Run with -Fix to attempt automatic repairs" -ForegroundColor Yellow
    }
}

# Exit with appropriate code
if ($results.overall_status -eq "healthy") {
    exit 0
} elseif ($results.overall_status -eq "degraded") {
    exit 1
} else {
    exit 2
}
