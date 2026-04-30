# Start-DevEnvironment.ps1
# Master Startup Script for Development Environment

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

$LogPath = Join-Path ([System.Environment]::GetFolderPath("MyDocuments")) "PowerShell\Logs"
if (-not (Test-Path $LogPath)) { New-Item -Path $LogPath -ItemType Directory -Force | Out-Null }
$LogFile = Join-Path $LogPath "DevStartup.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append
}

Write-Log "Starting Dev Environment Initialization..."

# 1. Start Docker Engine
Write-Log "Checking Docker Engine..."
try {
    $DockerScript = Join-Path $ScriptRoot "Start-DockerEngine.ps1"
    if (Test-Path $DockerScript) {
        Start-Process pwsh -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$DockerScript" -WindowStyle Hidden -Wait
        Write-Log "Docker startup check complete."
    } else {
        Write-Log "Error: Docker script not found at $DockerScript"
    }
} catch {
    Write-Log "Error starting Docker: $_"
}

# 2. Start WSL VSock Bridge
Write-Log "Starting WSL VSock Bridge..."
try {
    # Kill existing instances to avoid port conflicts
    Get-Process pwsh | Where-Object { $_.MainWindowTitle -like "*Start-WslBridge*" } | Stop-Process -Force -ErrorAction SilentlyContinue

    $BridgeScript = Join-Path $ScriptRoot "Start-WslBridge.ps1"
    if (Test-Path $BridgeScript) {
        # Start as a detached background job/process
        # We use -WindowStyle Hidden so it doesn't clutter the desktop
        Start-Process pwsh -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$BridgeScript", "-Port", "5000" -WindowStyle Hidden
        Write-Log "WSL Bridge started."
    } else {
        Write-Log "Error: Bridge script not found at $BridgeScript"
    }
} catch {
    Write-Log "Error starting Bridge: $_"
}

# 3. Warm-up WSL
Write-Log "Warming up WSL..."
try {
    wsl -d Ubuntu --exec echo "WSL Ready" | Out-Null
    Write-Log "WSL Warm-up complete."
} catch {
    Write-Log "Error warming up WSL: $_"
}

Write-Log "Initialization Complete."
