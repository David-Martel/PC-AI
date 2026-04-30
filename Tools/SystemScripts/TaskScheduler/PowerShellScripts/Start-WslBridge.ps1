# Start-WslBridge.ps1
# Starts the VSock Bridge for WSL Interop
param(
    [int]$Port = 5000
)

# Import WslExtensions
$ModulePath = Join-Path ([System.Environment]::GetFolderPath("MyDocuments")) "PowerShell\Modules\WslExtensions\WslExtensions.psm1"
if (Test-Path $ModulePath) {
    Import-Module $ModulePath -Force
} else {
    Write-Error "WslExtensions module not found at $ModulePath"
    exit 1
}

Write-Host "Starting WSL VSock Bridge on Port $Port..." -ForegroundColor Cyan
Write-Host "Connect from WSL using: wsl-connect $Port" -ForegroundColor Gray

# Start listener (Blocking call)
try {
    Start-WslVSockListener -Port $Port
} catch {
    Write-Error "Failed to start listener: $_"
    Read-Host "Press Enter to exit..."
}
