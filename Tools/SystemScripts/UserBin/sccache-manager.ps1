<#
.SYNOPSIS
    Robust sccache server management with connection pooling and auto-recovery

.DESCRIPTION
    Provides health monitoring, automatic restart, and connection management
    for sccache to prevent build failures from server overload.

.EXAMPLE
    # Start sccache with health monitoring
    .\sccache-manager.ps1 -Start

    # Check health and restart if needed
    .\sccache-manager.ps1 -HealthCheck

    # Run a build with sccache management
    .\sccache-manager.ps1 -Build "cargo build --release"
#>

param(
    [switch]$Start,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$HealthCheck,
    [switch]$Status,
    [switch]$ClearCache,
    [string]$Build,
    [int]$MaxRetries = 3,
    [int]$RetryDelayMs = 2000
)

$ErrorActionPreference = "Continue"

function Ensure-Directory {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Resolve-CacheRoot {
    param([string]$PreferredRoot)
    if ($PreferredRoot -and (Test-Path $PreferredRoot)) {
        return $PreferredRoot
    }

    $tDrive = 'T:\'
    if (Test-Path $tDrive) {
        $root = Join-Path $tDrive 'RustCache'
        Ensure-Directory -Path $root
        return $root
    }

    $fallback = $env:LOCALAPPDATA
    if (-not $fallback) { $fallback = $env:TEMP }
    if (-not $fallback) { $fallback = $env:USERPROFILE }
    $root = Join-Path $fallback 'RustCache'
    Ensure-Directory -Path $root
    return $root
}

function Resolve-PathOrFallback {
    param(
        [string]$PathValue,
        [string]$Fallback
    )
    if (-not $PathValue) { return $Fallback }
    try {
        $root = [System.IO.Path]::GetPathRoot($PathValue)
        if ($root -and (Test-Path $root)) { return $PathValue }
    } catch {}
    return $Fallback
}

$cacheRoot = Resolve-CacheRoot -PreferredRoot (Split-Path -Parent $env:SCCACHE_DIR)
$defaultCacheDir = Join-Path $cacheRoot 'sccache'
$defaultLogFile = Join-Path $defaultCacheDir 'manager.log'
$resolvedCacheDir = Resolve-PathOrFallback -PathValue $env:SCCACHE_DIR -Fallback $defaultCacheDir
$resolvedLogFile = Resolve-PathOrFallback -PathValue $env:SCCACHE_ERROR_LOG -Fallback $defaultLogFile

# Configuration
$script:Config = @{
    ServerPort = if ($env:SCCACHE_SERVER_PORT) { [int]$env:SCCACHE_SERVER_PORT } else { 4226 }
    MaxConnections = if ($env:SCCACHE_MAX_CONNECTIONS) { [int]$env:SCCACHE_MAX_CONNECTIONS } else { 8 }
    IdleTimeout = if ($env:SCCACHE_IDLE_TIMEOUT) { [int]$env:SCCACHE_IDLE_TIMEOUT } else { 600 }
    RequestTimeout = if ($env:SCCACHE_REQUEST_TIMEOUT) { [int]$env:SCCACHE_REQUEST_TIMEOUT } else { 180 }
    StartupTimeout = if ($env:SCCACHE_STARTUP_TIMEOUT) { [int]$env:SCCACHE_STARTUP_TIMEOUT } else { 30 }
    CacheDir = $resolvedCacheDir
    LogFile = $resolvedLogFile
}

function Join-Arguments {
    param([string[]]$InputArgs)
    $parts = foreach ($arg in $InputArgs) {
        if ($null -eq $arg) {
            '""'
        } elseif ($arg -match '[\s"]') {
            '"' + ($arg -replace '"', '\"') + '"'
        } else {
            $arg
        }
    }
    return ($parts -join ' ')
}

function Invoke-Process {
    param(
        [Parameter(Mandatory = $true)]
        [string]$File,
        [string[]]$Arguments = @(),
        [string]$ArgumentString = '',
        [switch]$CaptureOutput
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $File
    $psi.UseShellExecute = $false
    if ($CaptureOutput) {
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
    }
    if ($ArgumentString) {
        $psi.Arguments = $ArgumentString
    } elseif ($Arguments.Count -gt 0) {
        $psi.Arguments = Join-Arguments -InputArgs $Arguments
    }

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    try {
        $null = $proc.Start()
    } catch {
        return [PSCustomObject]@{
            ExitCode = 1
            Stdout = ''
            Stderr = $_.Exception.Message
        }
    }

    $stdout = ''
    $stderr = ''
    if ($CaptureOutput) {
        $stdout = $proc.StandardOutput.ReadToEnd()
        $stderr = $proc.StandardError.ReadToEnd()
    }

    $proc.WaitForExit()
    return [PSCustomObject]@{
        ExitCode = $proc.ExitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Resolve-Sccache {
    $cmd = Get-Command sccache -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    $logDir = Split-Path -Parent $script:Config.LogFile
    Ensure-Directory -Path $logDir
    Add-Content -Path $script:Config.LogFile -Value $logEntry -ErrorAction SilentlyContinue
}

function Test-SccacheHealth {
    <#
    .SYNOPSIS
        Check if sccache server is healthy and responding
    #>
    try {
        $sccacheExe = Resolve-Sccache
        if (-not $sccacheExe) {
            return @{ Healthy = $false; Reason = "sccache not found in PATH" }
        }

        $result = Invoke-Process -File $sccacheExe -Arguments @('--show-stats') -CaptureOutput
        $stats = ($result.Stdout + $result.Stderr)
        if ($result.ExitCode -ne 0) {
            return @{ Healthy = $false; Reason = "Stats command failed" }
        }

        # Check for valid stats output
        if ($stats -match "Cache location") {
            # Parse stats for potential issues
            $cacheErrors = 0
            $timeouts = 0
            $totalRequests = 0

            if ($stats -match "Cache errors\s+(\d+)") {
                $cacheErrors = [int]$matches[1]
            }
            if ($stats -match "Cache timeouts\s+(\d+)") {
                $timeouts = [int]$matches[1]
            }
            if ($stats -match "Compile requests\s+(\d+)") {
                $totalRequests = [int]$matches[1]
            }

            # If error rate is high (>10%), consider unhealthy
            if ($totalRequests -gt 100 -and ($cacheErrors + $timeouts) / $totalRequests -gt 0.1) {
                return @{
                    Healthy = $false
                    Reason = "High error rate: $cacheErrors errors, $timeouts timeouts out of $totalRequests requests"
                    Stats = $stats
                }
            }

            return @{ Healthy = $true; Stats = $stats }
        }

        return @{ Healthy = $false; Reason = "Invalid stats output" }
    }
    catch {
        return @{ Healthy = $false; Reason = $_.Exception.Message }
    }
}

function Start-SccacheServer {
    <#
    .SYNOPSIS
        Start sccache server with optimized settings
    #>
    Write-Log "Starting sccache server..."

    # Set environment for optimal performance
    $env:SCCACHE_CACHE_SIZE = "30G"
    $env:SCCACHE_DIR = $script:Config.CacheDir
    $env:SCCACHE_IDLE_TIMEOUT = $script:Config.IdleTimeout
    $env:SCCACHE_REQUEST_TIMEOUT = $script:Config.RequestTimeout
    $env:SCCACHE_STARTUP_TIMEOUT = $script:Config.StartupTimeout
    $env:SCCACHE_SERVER_PORT = $script:Config.ServerPort
    $env:SCCACHE_MAX_CONNECTIONS = $script:Config.MaxConnections
    $env:SCCACHE_CACHE_COMPRESSION = "zstd"
    $env:SCCACHE_DIRECT = "true"
    $env:SCCACHE_LOG = "warn"
    $env:SCCACHE_ERROR_LOG = $script:Config.LogFile

    # Start the server
    Ensure-Directory -Path $script:Config.CacheDir
    $logDir = Split-Path -Parent $script:Config.LogFile
    Ensure-Directory -Path $logDir
    $sccacheExe = Resolve-Sccache
    if (-not $sccacheExe) {
        Write-Log "sccache not found in PATH" -Level "ERROR"
        return $false
    }

    $result = Invoke-Process -File $sccacheExe -Arguments @('--start-server') -CaptureOutput

    $combined = ($result.Stdout + $result.Stderr)
    if ($result.ExitCode -eq 0 -or $combined -match "already running") {
        Write-Log "sccache server started successfully"

        # Wait for server to be ready
        $maxWait = $script:Config.StartupTimeout
        $waited = 0
        while ($waited -lt $maxWait) {
            Start-Sleep -Seconds 1
            $waited++
            $health = Test-SccacheHealth
            if ($health.Healthy) {
                Write-Log "sccache server is healthy after ${waited}s"
                return $true
            }
        }
        Write-Log "sccache server started but health check failed" -Level "WARN"
        return $false
    }
    else {
        Write-Log "Failed to start sccache server: $combined" -Level "ERROR"
        return $false
    }
}

function Stop-SccacheServer {
    <#
    .SYNOPSIS
        Gracefully stop sccache server
    #>
    Write-Log "Stopping sccache server..."

    # First try graceful stop
    $sccacheExe = Resolve-Sccache
    if (-not $sccacheExe) { return $true }
    $result = Invoke-Process -File $sccacheExe -Arguments @('--stop-server') -CaptureOutput

    if ($result.ExitCode -eq 0) {
        Write-Log "sccache server stopped gracefully"
        return $true
    }

    # Force kill if graceful stop fails
    Write-Log "Graceful stop failed, force killing..." -Level "WARN"
    Get-Process sccache*, sccache-dist* -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 1

    return $true
}

function Restart-SccacheServer {
    <#
    .SYNOPSIS
        Restart sccache server with fresh state
    #>
    Write-Log "Restarting sccache server..."
    Stop-SccacheServer | Out-Null
    Start-Sleep -Seconds 2
    return Start-SccacheServer
}

function Invoke-BuildWithRetry {
    <#
    .SYNOPSIS
        Run a cargo build with automatic sccache recovery on failure
    #>
    param(
        [string]$BuildCommand,
        [int]$MaxRetries = 3,
        [int]$RetryDelayMs = 2000
    )

    Write-Log "Starting build: $BuildCommand"

    # Ensure sccache is healthy before starting
    $health = Test-SccacheHealth
    if (-not $health.Healthy) {
        Write-Log "sccache unhealthy before build: $($health.Reason)" -Level "WARN"
        if (-not (Restart-SccacheServer)) {
            Write-Log "Failed to restart sccache, proceeding anyway" -Level "ERROR"
        }
    }

    $attempt = 0
    $success = $false
    $lastOutput = ""

    while ($attempt -lt $MaxRetries -and -not $success) {
        $attempt++
        Write-Log "Build attempt $attempt of $MaxRetries"

        # Run the build and capture output
        $result = Invoke-Process -File "cmd.exe" -ArgumentString "/d /s /c $BuildCommand" -CaptureOutput
        $exitCode = $result.ExitCode
        $lastOutput = ($result.Stdout + $result.Stderr)

        if ($exitCode -eq 0) {
            $success = $true
            Write-Log "Build succeeded on attempt $attempt"
        }
        else {
            # Check if failure was sccache-related
            $isSccacheError = $lastOutput -match "sccache" -and
                ($lastOutput -match "error" -or
                 $lastOutput -match "10054" -or
                 $lastOutput -match "connection.*closed" -or
                 $lastOutput -match "0xffffffff")

            if ($isSccacheError -and $attempt -lt $MaxRetries) {
                Write-Log "sccache-related build failure detected" -Level "WARN"

                # Restart sccache
                Write-Log "Restarting sccache server..."
                if (Restart-SccacheServer) {
                    Write-Log "sccache restarted, retrying build in ${RetryDelayMs}ms"
                    Start-Sleep -Milliseconds $RetryDelayMs
                }
                else {
                    Write-Log "Failed to restart sccache" -Level "ERROR"
                }
            }
            else {
                # Non-sccache error or max retries reached
                Write-Log "Build failed (exit code: $exitCode)" -Level "ERROR"
                Write-Output $lastOutput
                return $exitCode
            }
        }
    }

    if ($success) {
        Write-Output $lastOutput
        return 0
    }
    else {
        Write-Log "Build failed after $MaxRetries attempts" -Level "ERROR"
        Write-Output $lastOutput
        return 1
    }
}

function Clear-SccacheData {
    <#
    .SYNOPSIS
        Clear sccache cache and reset statistics
    #>
    Write-Log "Clearing sccache cache..."

    # Stop server first
    Stop-SccacheServer | Out-Null

    # Zero statistics
    $sccacheExe = Resolve-Sccache
    if ($sccacheExe) {
        $null = Invoke-Process -File $sccacheExe -Arguments @('--zero-stats')
    }

    Write-Log "Cache statistics cleared"

    # Restart server
    return Start-SccacheServer
}

function Show-SccacheStatus {
    <#
    .SYNOPSIS
        Display detailed sccache status information
    #>
    Write-Host "`n=== sccache Status ===" -ForegroundColor Cyan

    $health = Test-SccacheHealth

    if ($health.Healthy) {
        Write-Host "Server Status: HEALTHY" -ForegroundColor Green
        Write-Host "`nStatistics:"
        Write-Host $health.Stats
    }
    else {
        Write-Host "Server Status: UNHEALTHY" -ForegroundColor Red
        Write-Host "Reason: $($health.Reason)"
    }

    # Show configuration
    Write-Host "`n=== Configuration ===" -ForegroundColor Cyan
    Write-Host "Cache Directory: $($script:Config.CacheDir)"
    Write-Host "Max Connections: $($script:Config.MaxConnections)"
    Write-Host "Request Timeout: $($script:Config.RequestTimeout)s"
    Write-Host "Idle Timeout: $($script:Config.IdleTimeout)s"

    # Show cache size
    if (Test-Path $script:Config.CacheDir) {
        $cacheSize = (Get-ChildItem $script:Config.CacheDir -Recurse -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum / 1GB
        Write-Host ("Current Cache Size: {0:N2} GB" -f $cacheSize)
    }
}

# Main execution
if ($Start) {
    Start-SccacheServer
}
elseif ($Stop) {
    Stop-SccacheServer
}
elseif ($Restart) {
    Restart-SccacheServer
}
elseif ($HealthCheck) {
    $health = Test-SccacheHealth
    if ($health.Healthy) {
        Write-Host "sccache is healthy" -ForegroundColor Green
        exit 0
    }
    else {
        Write-Host "sccache is unhealthy: $($health.Reason)" -ForegroundColor Red
        Write-Host "Attempting restart..."
        if (Restart-SccacheServer) {
            Write-Host "sccache recovered" -ForegroundColor Green
            exit 0
        }
        exit 1
    }
}
elseif ($Status) {
    Show-SccacheStatus
}
elseif ($ClearCache) {
    Clear-SccacheData
}
elseif ($Build) {
    $result = Invoke-BuildWithRetry -BuildCommand $Build -MaxRetries $MaxRetries -RetryDelayMs $RetryDelayMs
    exit $result
}
else {
    Write-Host @"

sccache-manager.ps1 - Robust sccache server management

Usage:
    .\sccache-manager.ps1 -Start          Start sccache server
    .\sccache-manager.ps1 -Stop           Stop sccache server
    .\sccache-manager.ps1 -Restart        Restart sccache server
    .\sccache-manager.ps1 -HealthCheck    Check health and auto-recover
    .\sccache-manager.ps1 -Status         Show detailed status
    .\sccache-manager.ps1 -ClearCache     Clear cache and reset stats
    .\sccache-manager.ps1 -Build "cmd"    Run build with auto-recovery

Options:
    -MaxRetries N       Maximum retry attempts (default: 3)
    -RetryDelayMs N     Delay between retries in ms (default: 2000)

Examples:
    .\sccache-manager.ps1 -Build "cargo build --release"
    .\sccache-manager.ps1 -Build "cargo check --package mcp-core --tests" -MaxRetries 5

"@
}
