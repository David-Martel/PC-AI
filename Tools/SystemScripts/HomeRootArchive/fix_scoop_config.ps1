# Fix Scoop Configuration
$env:SCOOP = "C:\Users\david\scoop"
$env:SCOOP_GLOBAL = "C:\ProgramData\scoop"

# Set environment variables
[Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User')

# Create required directories
$directories = @(
    "$env:SCOOP\apps",
    "$env:SCOOP\buckets",
    "$env:SCOOP\cache",
    "$env:SCOOP\persist",
    "$env:SCOOP\shims"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Clone main bucket manually since we don't have git yet
$mainBucketPath = "$env:SCOOP\buckets\main"
if (!(Test-Path $mainBucketPath)) {
    Write-Host "Downloading main bucket..."
    $mainBucketZip = "$env:SCOOP\main-bucket.zip"
    Invoke-WebRequest -Uri "https://github.com/ScoopInstaller/Main/archive/master.zip" -OutFile $mainBucketZip

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($mainBucketZip, "$env:SCOOP\buckets")

    Move-Item "$env:SCOOP\buckets\Main-master" $mainBucketPath -Force
    Remove-Item $mainBucketZip -Force
    Write-Host "Main bucket installed successfully"
}

Write-Host "Scoop configuration fixed successfully"