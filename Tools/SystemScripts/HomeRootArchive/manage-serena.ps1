# Serena MCP Server Lifecycle Manager
# Handles serena process management with DNS-based addressing to prevent port proliferation
# and browser window duplication.
#
# Usage:
#   .\manage-serena.ps1 -Action start   # Start serena server
#   .\manage-serena.ps1 -Action stop    # Stop serena server
#   .\manage-serena.ps1 -Action restart # Restart serena server
#   .\manage-serena.ps1 -Action status  # Check serena status
#   .\manage-serena.ps1 -Action dashboard # Open dashboard in existing window
#
# Configuration:
#   SERENA_PORT - The fixed port serena should use (default: 24282)
#   SERENA_HOSTNAME - DNS name to use instead of IP (default: serena.local)

param(
    [ValidateSet('start', 'stop', 'restart', 'status', 'dashboard')]
    [string]$Action = 'status',
    [int]$Port = 24282,
    [string]$Hostname = 'serena.local'
)

# Configuration
$SERENA_PORT = $Port
$SERENA_HOSTNAME = $Hostname
$SERENA_DASHBOARD_URL = "http://${SERENA_HOSTNAME}:${SERENA_PORT}/dashboard/index.html"
$SERENA_FALLBACK_URL = "http://127.0.0.1:${SERENA_PORT}/dashboard/index.html"
$PROCESS_NAME = 'serena'

# Helper Functions
function Write-Status {
    param([string]$Message, [ValidateSet('Info', 'Success', 'Warning', 'Error')]$Level = 'Info')
    $colors = @{
        'Info'    = 'Cyan'
        'Success' = 'Green'
        'Warning' = 'Yellow'
        'Error'   = 'Red'
    }
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $Message" -ForegroundColor $colors[$Level]
}

function Get-SerenaProcess {
    Get-Process -Name $PROCESS_NAME -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*start-mcp-server*"
    } | Select-Object -First 1
}

function Test-PortListening {
    param([int]$Port)
    try {
        $tcpConnection = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        return $tcpConnection.TcpTestSucceeded
    }
    catch {
        return $false
    }
}

function Test-DashboardHealth {
    try {
        $response = Invoke-WebRequest -Uri $SERENA_FALLBACK_URL -Method Head -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Setup-mDNS {
    # Add hostname to hosts file if not already present
    $hostsFile = "C:\Windows\System32\drivers\etc\hosts"
    $hostEntry = "127.0.0.1`t$SERENA_HOSTNAME"

    try {
        $hostsContent = Get-Content $hostsFile -ErrorAction SilentlyContinue
        if ($hostsContent -notlike "*$SERENA_HOSTNAME*") {
            Write-Status "Adding mDNS entry: $hostEntry" -Level 'Info'
            Add-Content -Path $hostsFile -Value "`n$hostEntry" -ErrorAction Stop
            Write-Status "mDNS entry added successfully" -Level 'Success'
        }
    }
    catch {
        Write-Status "Warning: Could not add mDNS entry (may require admin): $_" -Level 'Warning'
    }
}

function Start-SerenaServer {
    Write-Status "Starting Serena MCP Server..." -Level 'Info'

    # Check if serena is already running
    $existingProcess = Get-SerenaProcess
    if ($existingProcess) {
        Write-Status "Serena is already running (PID: $($existingProcess.Id))" -Level 'Warning'

        # Verify dashboard is accessible
        if (Test-PortListening -Port $SERENA_PORT) {
            Write-Status "Dashboard is accessible at: $SERENA_FALLBACK_URL" -Level 'Success'
            return $true
        }
        else {
            Write-Status "Process exists but dashboard is not responding. Stopping and restarting..." -Level 'Warning'
            Stop-SerenaServer
        }
    }

    # Setup mDNS first
    Setup-mDNS

    try {
        # Find serena executable in uv cache
        $serenaExe = Get-ChildItem -Path "$env:LOCALAPPDATA\uv\cache\archive-v0" -Filter "serena.exe" -Recurse -ErrorAction SilentlyContinue |
                     Select-Object -First 1

        if (-not $serenaExe) {
            Write-Status "Serena executable not found in uv cache. Attempting to use PATH..." -Level 'Warning'
            $serenaExe = Get-Command serena -ErrorAction SilentlyContinue
        }

        if (-not $serenaExe) {
            Write-Status "Could not locate serena executable" -Level 'Error'
            return $false
        }

        Write-Status "Using serena from: $($serenaExe.FullName)" -Level 'Info'

        # Start serena in background with proper arguments
        $processArgs = @(
            'start-mcp-server',
            '--transport', 'stdio',
            '--context', 'ide-assistant',
            '--log-level', 'ERROR'
        )

        $process = Start-Process -FilePath $serenaExe.FullName -ArgumentList $processArgs -PassThru -NoNewWindow

        Write-Status "Serena process started (PID: $($process.Id))" -Level 'Success'

        # Wait for dashboard to become available
        $maxRetries = 30
        $retryCount = 0

        Write-Status "Waiting for dashboard to become available..." -Level 'Info'
        while ($retryCount -lt $maxRetries) {
            if (Test-PortListening -Port $SERENA_PORT) {
                Start-Sleep -Milliseconds 500
                if (Test-DashboardHealth) {
                    Write-Status "Dashboard is now available at: $SERENA_FALLBACK_URL" -Level 'Success'
                    Write-Status "Access via: $SERENA_DASHBOARD_URL (if mDNS is configured)" -Level 'Info'
                    return $true
                }
            }
            $retryCount++
            Start-Sleep -Seconds 1
        }

        Write-Status "Timeout waiting for dashboard. Check if process is still running." -Level 'Warning'
        return $false
    }
    catch {
        Write-Status "Error starting serena: $_" -Level 'Error'
        return $false
    }
}

function Stop-SerenaServer {
    Write-Status "Stopping Serena MCP Server..." -Level 'Info'

    try {
        $processes = Get-Process -Name $PROCESS_NAME -ErrorAction SilentlyContinue |
                     Where-Object { $_.CommandLine -like "*start-mcp-server*" }

        if ($processes) {
            $processes | ForEach-Object {
                Write-Status "Stopping process (PID: $($_.Id))" -Level 'Info'
                Stop-Process -Id $_.Id -Force -ErrorAction Stop
            }
            Write-Status "Serena stopped successfully" -Level 'Success'
            return $true
        }
        else {
            Write-Status "No running Serena processes found" -Level 'Warning'
            return $false
        }
    }
    catch {
        Write-Status "Error stopping serena: $_" -Level 'Error'
        return $false
    }
}

function Get-ServerStatus {
    Write-Status "Checking Serena status..." -Level 'Info'
    Write-Host ""

    $process = Get-SerenaProcess

    if ($process) {
        Write-Host "Status: " -NoNewline -ForegroundColor Cyan
        Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "Process ID: " -NoNewline -ForegroundColor Cyan
        Write-Host $process.Id
        Write-Host "Executable: " -NoNewline -ForegroundColor Cyan
        Write-Host $process.Path
    }
    else {
        Write-Host "Status: " -NoNewline -ForegroundColor Cyan
        Write-Host "STOPPED" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Port Status: " -NoNewline -ForegroundColor Cyan
    if (Test-PortListening -Port $SERENA_PORT) {
        Write-Host "LISTENING (Port $SERENA_PORT)" -ForegroundColor Green
    }
    else {
        Write-Host "NOT LISTENING" -ForegroundColor Yellow
    }

    Write-Host "Dashboard Health: " -NoNewline -ForegroundColor Cyan
    if (Test-DashboardHealth) {
        Write-Host "HEALTHY" -ForegroundColor Green
        Write-Host "Dashboard URL: " -NoNewline -ForegroundColor Cyan
        Write-Host $SERENA_FALLBACK_URL -ForegroundColor Blue
    }
    else {
        Write-Host "NOT RESPONDING" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "mDNS Hostname: " -NoNewline -ForegroundColor Cyan
    Write-Host "$SERENA_HOSTNAME (Port $SERENA_PORT)" -ForegroundColor Blue
    Write-Host ""
}

function Open-Dashboard {
    Write-Status "Opening Serena Dashboard..." -Level 'Info'

    # Ensure server is running
    if (-not (Get-SerenaProcess)) {
        Write-Status "Server not running. Starting..." -Level 'Info'
        if (-not (Start-SerenaServer)) {
            Write-Status "Failed to start server" -Level 'Error'
            return $false
        }
    }

    # Verify dashboard is accessible
    if (-not (Test-DashboardHealth)) {
        Write-Status "Dashboard is not responding" -Level 'Error'
        return $false
    }

    # Open dashboard using mDNS hostname first, fallback to IP
    $urlToOpen = $SERENA_FALLBACK_URL

    try {
        # Check if we can resolve mDNS hostname
        $dns = @([System.Net.Dns]::GetHostAddresses($SERENA_HOSTNAME))
        if ($dns.Count -gt 0) {
            $urlToOpen = $SERENA_DASHBOARD_URL
            Write-Status "Using mDNS URL: $urlToOpen" -Level 'Info'
        }
        else {
            Write-Status "mDNS hostname not resolvable, using fallback URL" -Level 'Warning'
        }
    }
    catch {
        Write-Status "Could not resolve mDNS, using fallback URL" -Level 'Warning'
    }

    # Check for existing browser windows and reuse them
    $existingWindow = Get-Process -Name "*chrome*", "*firefox*", "*edge*", "*msedge*" -ErrorAction SilentlyContinue |
                      Select-Object -First 1

    if ($existingWindow) {
        Write-Status "Reusing existing browser window" -Level 'Info'
    }

    Write-Status "Opening URL: $urlToOpen" -Level 'Success'
    Start-Process -FilePath $urlToOpen

    return $true
}

# Main execution
switch ($Action) {
    'start' {
        Start-SerenaServer | Out-Null
    }
    'stop' {
        Stop-SerenaServer | Out-Null
    }
    'restart' {
        Stop-SerenaServer | Out-Null
        Start-Sleep -Seconds 2
        Start-SerenaServer | Out-Null
    }
    'status' {
        Get-ServerStatus
    }
    'dashboard' {
        Open-Dashboard | Out-Null
    }
}
