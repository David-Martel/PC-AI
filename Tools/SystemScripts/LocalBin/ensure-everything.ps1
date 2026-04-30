<#
.SYNOPSIS
    Ensures Everything is running in user mode with IPC available

.DESCRIPTION
    Checks if Everything is running with proper IPC window.
    If not, starts it correctly. Returns exit code 0 if ready, 1 if not.

.EXAMPLE
    ensure-everything
    # Ensures Everything is ready for CLI use
#>

$ES_PATH = "C:\Program Files\Everything\es.exe"
$EVERYTHING_PATH = "C:\Program Files\Everything\Everything.exe"

function Test-EverythingIPC {
    if (-not (Test-Path $ES_PATH)) {
        return $false
    }

    $proc = Get-Process -Name "Everything" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $false
    }

    # Check for IPC window
    if ($proc.MainWindowHandle -eq 0) {
        return $false
    }

    # Quick IPC test
    $job = Start-Job -ScriptBlock {
        param($es)
        & $es -get-everything-version 2>&1
    } -ArgumentList $ES_PATH

    $result = Wait-Job $job -Timeout 2 | Receive-Job
    Remove-Job $job -Force -ErrorAction SilentlyContinue

    return $null -ne $result -and $result -match '^\d+\.\d+\.\d+\.\d+$'
}

function Start-EverythingWithIPC {
    # Stop existing instances
    $procs = Get-Process -Name "Everything" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "Stopping existing Everything processes..." -ForegroundColor Yellow
        $procs | Stop-Process -Force
        Start-Sleep -Seconds 2
    }

    # Start with GUI mode
    Write-Host "Starting Everything with IPC enabled..." -ForegroundColor Yellow
    Start-Process $EVERYTHING_PATH
    Start-Sleep -Seconds 3

    # Wait for IPC to be ready
    $maxWait = 30
    $waited = 0
    while ($waited -lt $maxWait) {
        if (Test-EverythingIPC) {
            Write-Host "Everything IPC ready!" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 1
        $waited++
        Write-Host "Waiting for IPC... ($waited/$maxWait)" -ForegroundColor Gray
    }

    Write-Host "Warning: IPC not ready after ${maxWait}s (database may still be indexing)" -ForegroundColor Yellow
    return $false
}

# Main
if (Test-EverythingIPC) {
    Write-Host "Everything is ready!" -ForegroundColor Green
    exit 0
}
else {
    if (Start-EverythingWithIPC) {
        exit 0
    }
    else {
        exit 1
    }
}
