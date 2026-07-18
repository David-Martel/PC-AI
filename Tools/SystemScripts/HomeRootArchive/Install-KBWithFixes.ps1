param(
    [string]$TargetKB = "KB5062553",
    [string]$ServicingStackKB = "KB5043080",
    [string]$DownloadDir = "$env:TEMP\WinUpdateTemp",
    [switch]$EnableLogging
)

$logPath = "$DownloadDir\update-log.txt"
function Log { param($msg)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] $msg"
    if ($EnableLogging) { Add-Content $logPath $line }
    Write-Host $line
}

function Ensure-Directory {
    param ($Path)
    if (-Not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

function Get-KBStatus {
    param ($KB)
    $pkg = Get-WindowsPackage -Online | Where-Object { $_.PackageName -like "*$KB*" }
    return ($pkg.Count -gt 0)
}

function Download-Update {
    param ($KB, $Uri)
    $FilePath = Join-Path $DownloadDir "$KB.msu"
    if (-Not (Test-Path $FilePath)) {
        Log "Downloading $KB from $Uri..."
        Invoke-WebRequest -Uri $Uri -OutFile $FilePath
    } else {
        Log "$KB already exists locally."
    }
    return $FilePath
}

function Install-Update {
    param ($FilePath)
    Log "Installing update from $FilePath..."
    Start-Process "wusa.exe" "$FilePath /quiet /norestart" -Wait
    Log "Finished installing $FilePath."
}

function AutoFix-WindowsUpdate {
    Log "Attempting auto-fix: Resetting Windows Update services..."
    $cmds = @(
        "net stop wuauserv",
        "net stop cryptSvc",
        "net stop bits",
        "net stop msiserver",
        "ren C:\Windows\SoftwareDistribution SoftwareDistribution.old",
        "ren C:\Windows\System32\catroot2 Catroot2.old",
        "net start wuauserv",
        "net start cryptSvc",
        "net start bits",
        "net start msiserver"
    )
    foreach ($cmd in $cmds) {
        Log $cmd
        & cmd.exe /c $cmd
    }
}

function AutoFix-ComponentStore {
    Log "Running DISM to repair component store..."
    DISM /Online /Cleanup-Image /RestoreHealth
    Log "DISM completed."
}

Ensure-Directory -Path $DownloadDir

$updateLinks = @{
    $ServicingStackKB = "https://catalog.s.download.windowsupdate.com/d/msuc/2025/07/ssu_KB5043080.msu"
    $TargetKB = "https://catalog.s.download.windowsupdate.com/d/msuc/2025/07/cu_KB5062553.msu"
}

Log "=== Starting Windows Update Utility ==="
AutoFix-WindowsUpdate
AutoFix-ComponentStore

foreach ($KB in @($ServicingStackKB, $TargetKB)) {
    if (-Not (Get-KBStatus -KB $KB)) {
        $File = Download-Update -KB $KB -Uri $updateLinks[$KB]
        Install-Update -FilePath $File
    } else {
        Log "$KB is already installed."
    }
}

Log "=== Update flow complete ==="