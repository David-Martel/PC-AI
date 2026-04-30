#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Safely repairs Docker Desktop by gracefully resetting WSL and Hyper-V state.

.DESCRIPTION
    This script fixes Docker Desktop startup issues caused by:
    - Hung WSL processes
    - VHDX mount failures
    - Docker service issues

    SAFETY: This script prioritizes graceful shutdown over force operations
    to prevent kernel crashes (BSOD). It avoids restarting vmcompute unless
    absolutely necessary.

.NOTES
    Run this script as Administrator.
    Version: 2.0 - Safe version (no vmcompute restart)
#>

[CmdletBinding()]
param(
    [switch]$ForceVmComputeRestart  # Only use if explicitly requested
)

$ErrorActionPreference = 'Continue'

function Write-Step {
    param([string]$Step, [string]$Message)
    Write-Host "[$Step] $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "  OK: $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor Gray
}

function Wait-ProcessExit {
    param(
        [string]$ProcessName,
        [int]$TimeoutSeconds = 30
    )

    $elapsed = 0
    while ($elapsed -lt $TimeoutSeconds) {
        $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
        if (-not $proc) {
            return $true
        }
        Start-Sleep -Seconds 1
        $elapsed++
    }
    return $false
}

Write-Host ""
Write-Host "=== Docker Desktop Safe Repair Script v2.0 ===" -ForegroundColor Cyan
Write-Host "This script uses graceful shutdown to prevent kernel crashes." -ForegroundColor Gray
Write-Host ""

# Phase 1: Stop Docker Desktop application gracefully
Write-Step "1/7" "Requesting Docker Desktop graceful shutdown..."

# Try graceful quit first via Docker CLI
$dockerDesktopRunning = Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue
if ($dockerDesktopRunning) {
    try {
        # Docker Desktop has a quit command
        & "C:\Program Files\Docker\Docker\resources\bin\docker.exe" desktop stop 2>$null
        Start-Sleep -Seconds 5
    } catch {
        Write-Info "Docker desktop stop command not available, using fallback"
    }
}

# Phase 2: Stop Docker service gracefully (not force)
Write-Step "2/7" "Stopping Docker service gracefully..."
$dockerService = Get-Service 'com.docker.service' -ErrorAction SilentlyContinue
if ($dockerService -and $dockerService.Status -eq 'Running') {
    Stop-Service 'com.docker.service' -ErrorAction SilentlyContinue
    # Wait for service to stop (up to 30 seconds)
    $waited = 0
    while ($waited -lt 30 -and (Get-Service 'com.docker.service').Status -ne 'Stopped') {
        Start-Sleep -Seconds 1
        $waited++
    }
    if ((Get-Service 'com.docker.service').Status -eq 'Stopped') {
        Write-Success "Docker service stopped"
    } else {
        Write-Info "Docker service still stopping..."
    }
} else {
    Write-Info "Docker service not running"
}

# Phase 3: Request graceful WSL shutdown FIRST (before any process killing)
Write-Step "3/7" "Requesting graceful WSL shutdown..."
Write-Info "This may take up to 30 seconds..."

$wslJob = Start-Job -ScriptBlock { wsl --shutdown 2>&1 }
$completed = Wait-Job -Job $wslJob -Timeout 30
if ($completed) {
    $result = Receive-Job -Job $wslJob
    Write-Success "WSL shutdown completed"
} else {
    Stop-Job -Job $wslJob -ErrorAction SilentlyContinue
    Write-Info "WSL shutdown timed out (will wait for processes to exit)"
}
Remove-Job -Job $wslJob -Force -ErrorAction SilentlyContinue

# Phase 4: Wait for WSL processes to exit naturally (DO NOT force kill immediately)
Write-Step "4/7" "Waiting for WSL processes to exit gracefully..."
$wslProcesses = @('wsl', 'wslhost')
$allExited = $true

foreach ($procName in $wslProcesses) {
    $proc = Get-Process -Name $procName -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Info "Waiting for $procName to exit (up to 30 seconds)..."
        if (-not (Wait-ProcessExit -ProcessName $procName -TimeoutSeconds 30)) {
            Write-Info "$procName did not exit gracefully"
            $allExited = $false
        } else {
            Write-Success "$procName exited"
        }
    }
}

# Phase 5: Only if graceful exit failed, try gentle termination
Write-Step "5/7" "Cleaning up remaining processes..."
$remainingWsl = Get-Process -Name 'wsl' -ErrorAction SilentlyContinue
$remainingDocker = Get-Process -Name 'Docker Desktop','com.docker.backend' -ErrorAction SilentlyContinue

if ($remainingWsl -or $remainingDocker) {
    Write-Info "Some processes remain, requesting termination..."

    # Stop Docker processes first
    Stop-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue
    Stop-Process -Name 'com.docker.backend' -ErrorAction SilentlyContinue
    Stop-Process -Name 'com.docker.diagnose' -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3

    # Then WSL (gentler approach - send close signal, not force kill)
    $wslProcs = Get-Process -Name 'wsl' -ErrorAction SilentlyContinue
    if ($wslProcs) {
        foreach ($p in $wslProcs) {
            try {
                $p.CloseMainWindow() | Out-Null
            } catch {}
        }
        Start-Sleep -Seconds 5

        # Only force if still running
        $stillRunning = Get-Process -Name 'wsl' -ErrorAction SilentlyContinue
        if ($stillRunning) {
            Write-Info "Force stopping remaining WSL processes..."
            Stop-Process -Name 'wsl' -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 5
        }
    }
    Write-Success "Cleanup completed"
} else {
    Write-Success "All processes exited gracefully"
}

# Phase 6: DO NOT restart vmcompute unless explicitly requested
# This is what caused the kernel crash
Write-Step "6/7" "Verifying system state..."

if ($ForceVmComputeRestart) {
    Write-Host ""
    Write-Host "  WARNING: -ForceVmComputeRestart was specified." -ForegroundColor Red
    Write-Host "  This can cause kernel crashes if WSL state is inconsistent." -ForegroundColor Red
    Write-Host "  Waiting 15 seconds for system to stabilize first..." -ForegroundColor Red
    Start-Sleep -Seconds 15

    # Check if any WSL processes remain
    $anyWsl = Get-Process -Name 'wsl','wslhost','wslservice' -ErrorAction SilentlyContinue
    if ($anyWsl) {
        Write-Host "  ABORT: WSL processes still running. Cannot safely restart vmcompute." -ForegroundColor Red
        Write-Host "  Please reboot instead." -ForegroundColor Yellow
    } else {
        Write-Info "Restarting vmcompute service..."
        Restart-Service vmcompute -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 10
        Write-Success "vmcompute restarted"
    }
} else {
    # Just verify vmcompute is running
    $vmcompute = Get-Service vmcompute -ErrorAction SilentlyContinue
    if ($vmcompute.Status -eq 'Running') {
        Write-Success "vmcompute service is running (not restarted)"
    } else {
        Write-Info "Starting vmcompute service..."
        Start-Service vmcompute -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
    }
}

# Extra stabilization wait
Write-Info "Waiting for system to stabilize..."
Start-Sleep -Seconds 5

# Phase 7: Start Docker
Write-Step "7/7" "Starting Docker..."

# Start service first
Start-Service 'com.docker.service' -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Normal
Write-Host ""

# Wait for Docker to become ready
Write-Host "Waiting for Docker to become ready..." -ForegroundColor Cyan
$maxWait = 120
$waited = 0
$interval = 5

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $waited += $interval

    if (Test-Path '\\.\pipe\docker_engine') {
        Write-Host ""
        Write-Host "SUCCESS: Docker is now running!" -ForegroundColor Green
        Write-Host ""
        try {
            docker info 2>&1 | Select-String -Pattern "Server Version|Storage Driver|Containers|Images" | ForEach-Object { Write-Host $_.Line }
        } catch {}
        exit 0
    }

    Write-Host "  Waiting... ($waited/$maxWait seconds)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Docker did not become ready within $maxWait seconds." -ForegroundColor Yellow
Write-Host ""
Write-Host "Recommended next steps:" -ForegroundColor Cyan
Write-Host "1. Check Docker Desktop for error dialogs" -ForegroundColor White
Write-Host "2. If errors persist, try: Reboot computer" -ForegroundColor White
Write-Host "3. If still failing after reboot, run:" -ForegroundColor White
Write-Host "   .\Repair-DockerDesktop.ps1 -ForceVmComputeRestart" -ForegroundColor Gray
Write-Host "4. Last resort: Reset Docker Desktop via Settings > Troubleshoot" -ForegroundColor White
exit 1
