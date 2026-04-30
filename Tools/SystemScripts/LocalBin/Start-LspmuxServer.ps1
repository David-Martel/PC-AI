<#
.SYNOPSIS
    Starts the lspmux server as a background process.
    Designed to be run at login via Task Scheduler or manually.
.DESCRIPTION
    Ensures only one lspmux server is running. Sets memory-optimization
    env vars that pass through to rust-analyzer instances.
    Delegates to CargoTools Start-LspmuxServer if available.
#>
[CmdletBinding()]
param(
    [switch]$Force
)

# Try CargoTools module first (has full health checking and memory management)
$cargoTools = Get-Module CargoTools -ListAvailable -ErrorAction SilentlyContinue
if ($cargoTools) {
    Import-Module CargoTools -ErrorAction SilentlyContinue
    if (Get-Command Start-LspmuxServer -Module CargoTools -ErrorAction SilentlyContinue) {
        Start-LspmuxServer -Force:$Force
        return
    }
}

# Fallback: standalone implementation
$serverProcess = Get-Process -Name 'lspmux' -ErrorAction SilentlyContinue |
    Where-Object { $_.MainModule.FileName -match 'lspmux' }

if ($serverProcess -and -not $Force) {
    Write-Host "lspmux server already running (PID $($serverProcess.Id))" -ForegroundColor Yellow
    return
}

if ($serverProcess -and $Force) {
    Write-Host "Stopping existing lspmux server (PID $($serverProcess.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $serverProcess.Id -Force
    Start-Sleep -Milliseconds 500
}

# Set env vars that lspmux passes through to RA via pass_environment config
$env:RA_LRU_CAPACITY = '64'
$env:CHALK_SOLVER_MAX_SIZE = '10'
$env:RA_PROC_MACRO_WORKERS = '1'
$env:RA_LOG = 'error'
$env:CARGO_HOME = 'T:\RustCache\cargo-home'

$lspmuxExe = (Get-Command lspmux -ErrorAction SilentlyContinue).Source
if (-not $lspmuxExe) {
    $lspmuxExe = "$env:USERPROFILE\.cargo\bin\lspmux.exe"
}

if (-not (Test-Path $lspmuxExe)) {
    Write-Error "lspmux not found. Install: cargo install lspmux"
    return
}

Write-Host "Starting lspmux server ($lspmuxExe)..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $lspmuxExe -ArgumentList 'server' `
    -WindowStyle Hidden -PassThru

# Lower priority
try { $proc.PriorityClass = 'BelowNormal' } catch {}

Write-Host "lspmux server started (PID $($proc.Id))" -ForegroundColor Green

# Verify it's listening
Start-Sleep -Seconds 1
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect('127.0.0.1', 27631)
    $tcp.Close()
    Write-Host "Verified: listening on 127.0.0.1:27631" -ForegroundColor Green
} catch {
    Write-Warning "Server started but port 27631 not yet accepting connections"
}
