# Fix Scoop Environment Variables
$env:SCOOP = "C:\Users\david\scoop"
$env:SCOOP_GLOBAL = "C:\ProgramData\scoop"

# Set user environment variables permanently
[Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User')

# Add Scoop to PATH permanently
$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
$scoopPath = "$env:SCOOP\shims"
if ($currentPath -notlike "*$scoopPath*") {
    [Environment]::SetEnvironmentVariable('PATH', "$scoopPath;$currentPath", 'User')
}

Write-Host "Environment variables set:"
Write-Host "SCOOP = $env:SCOOP"
Write-Host "PATH includes: $scoopPath"

# Refresh environment for current session
$env:PATH = "$scoopPath;$env:PATH"

Write-Host "Scoop environment fixed successfully"