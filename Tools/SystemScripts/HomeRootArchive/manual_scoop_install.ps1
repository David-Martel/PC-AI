# Manual Scoop Installation Script
$env:PATH = "C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0;" + $env:PATH

# Set Scoop directory
$env:SCOOP = "C:\Users\david\scoop"
[Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User')

# Create directory
New-Item -ItemType Directory -Path $env:SCOOP -Force | Out-Null

# Download Scoop
$scoopZip = "$env:SCOOP\scoop.zip"
Invoke-WebRequest -Uri "https://github.com/ScoopInstaller/Scoop/archive/master.zip" -OutFile $scoopZip

# Extract Scoop
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($scoopZip, $env:SCOOP)

# Setup Scoop
Move-Item "$env:SCOOP\Scoop-master\*" $env:SCOOP -Force
Remove-Item "$env:SCOOP\Scoop-master" -Recurse -Force
Remove-Item $scoopZip -Force

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
$scoopPath = "$env:SCOOP\shims"
if ($currentPath -notlike "*$scoopPath*") {
    [Environment]::SetEnvironmentVariable('PATH', "$scoopPath;$currentPath", 'User')
}

Write-Host "Scoop installed successfully to $env:SCOOP"