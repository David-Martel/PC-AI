# Start-DockerEngine.ps1
# Ensures Docker Desktop is running and ready

Write-Host "Checking Docker Desktop status..." -ForegroundColor Cyan

$dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerProcess) {
    Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan
    if (Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe") {
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    } else {
        Write-Error "Docker Desktop executable not found."
        exit 1
    }

    # Wait for backend to start
    Write-Host "Waiting for Docker Engine..." -NoNewline
    $retries = 60
    while ($retries -gt 0) {
        Start-Sleep -Seconds 1
        if (Get-Process "com.docker.backend" -ErrorAction SilentlyContinue) {
            Write-Host " Ready!" -ForegroundColor Green
            break
        }
        Write-Host "." -NoNewline
        $retries--
    }
} else {
    Write-Host "Docker Desktop is already running." -ForegroundColor Green
}
