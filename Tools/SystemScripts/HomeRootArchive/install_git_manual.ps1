# Manual Git Installation for Scoop
$gitVersion = "2.47.1"
$gitUrl = "https://github.com/git-for-windows/git/releases/download/v$gitVersion.windows.1/PortableGit-$gitVersion-64-bit.7z.exe"
$gitPath = "$env:SCOOP\apps\git\$gitVersion"
$gitCurrent = "$env:SCOOP\apps\git\current"

Write-Host "Creating git directory structure..."
New-Item -ItemType Directory -Path $gitPath -Force | Out-Null

Write-Host "Downloading Git portable..."
$webClient = New-Object System.Net.WebClient
$webClient.DownloadFile($gitUrl, "$gitPath\git-portable.exe")

Write-Host "Extracting Git..."
& "$gitPath\git-portable.exe" -o"$gitPath" -y

# Create current symlink equivalent
if (Test-Path $gitCurrent) {
    Remove-Item $gitCurrent -Force -Recurse
}
New-Item -ItemType Junction -Path $gitCurrent -Target $gitPath | Out-Null

# Create shim for git
$shimPath = "$env:SCOOP\shims\git.exe"
$gitExePath = "$gitCurrent\mingw64\bin\git.exe"

if (Test-Path $gitExePath) {
    Write-Host "Creating git shim..."
    $shimContent = @"
@echo off
"$gitExePath" %*
"@
    $shimContent | Out-File -FilePath "$env:SCOOP\shims\git.cmd" -Encoding ASCII

    Write-Host "Git installed successfully at: $gitExePath"
    return $true
} else {
    Write-Host "Git installation failed - executable not found"
    return $false
}