#Requires -Version 5.1
<#
.SYNOPSIS
Repoints the Google Drive (DriveFS) content cache to the F: cloud-cache-disk.

.DESCRIPTION
DriveFS stores cached file content under <ContentCachePath>\<account_id>\content_cache.
This script restores that path to F:\Google (the cloud-cache-disk), leaving the
DriveFS application state - databases, logs, pid - on C: where it belongs.

The content cache is a cache: relocating it loses no data, DriveFS re-fetches on
demand. The client is stopped first because DriveFS reads the setting at startup
and rewrites the key while running.

.PARAMETER CachePath
Target content cache directory.

.PARAMETER Restart
Restart DriveFS through the gated launcher after the change.

.EXAMPLE
pwsh -File .\Tools\Set-DriveFsContentCache.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$CachePath = 'F:\Google',
    [string]$RegPath = 'HKCU:\SOFTWARE\Google\DriveFS',
    [string]$ExpectedLabel = 'cloud-cache-disk',
    [switch]$Restart,
    [switch]$Force,
    [string]$Launcher = 'C:\codedev\PC_AI\Tools\Start-CloudClientsAfterVHDX.ps1'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# --- Guard: target must live on the real cloud-cache-disk ---
$drive = ($CachePath -split ':')[0]
$vol = Get-Volume -DriveLetter $drive -ErrorAction SilentlyContinue
if (-not $vol) { throw "${drive}: is not mounted - refusing to point the cache at a missing volume." }
if ($vol.FileSystemLabel -ne $ExpectedLabel) {
    throw "${drive}: label is '$($vol.FileSystemLabel)', expected '$ExpectedLabel' - refusing."
}
Write-Host "Target volume verified: ${drive}: '$($vol.FileSystemLabel)'" -ForegroundColor Green

$current = (Get-ItemProperty -Path $RegPath -Name ContentCachePath -ErrorAction SilentlyContinue).ContentCachePath
Write-Host "  current ContentCachePath : $current"
Write-Host "  new     ContentCachePath : $CachePath"
if ($current -eq $CachePath) { Write-Host 'Already set - nothing to do.' -ForegroundColor Green; return }

# --- Refuse if a CONTENT transfer looks in flight ---
# Only writes inside content_cache indicate file data moving. Log files and the
# metadata/root-preference SQLite DBs are written continuously during idle
# operation and must not be mistaken for a transfer.
$state = 'C:\Users\david\AppData\Local\Google\DriveFS'
$busy = Get-ChildItem $state -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-2) -and $_.FullName -match '\\content_cache\\' }
if ($busy) {
    Write-Warning 'DriveFS wrote content_cache files in the last 2 minutes - a transfer may be in flight.'
    $busy | Select-Object -First 5 FullName, LastWriteTime | Format-Table -AutoSize | Out-String | Write-Host
    if (-not $Force) {
        Write-Warning 'Refusing to stop DriveFS mid-transfer. Re-run with -Force to override.'
        return
    }
    Write-Warning '-Force specified: proceeding despite possible in-flight transfer.'
}

if (-not (Test-Path $CachePath)) {
    if ($PSCmdlet.ShouldProcess($CachePath, 'Create cache directory')) {
        New-Item -ItemType Directory -Path $CachePath -Force | Out-Null
    }
}

# --- Stop DriveFS (it rewrites the key while running) ---
if (Get-Process GoogleDriveFS -ErrorAction SilentlyContinue) {
    if ($PSCmdlet.ShouldProcess('GoogleDriveFS', 'Stop client')) {
        $exe = (Get-Process GoogleDriveFS | Select-Object -First 1).Path
        if ($exe) { Start-Process $exe -ArgumentList '--quit' -ErrorAction SilentlyContinue; Start-Sleep -Seconds 8 }
        $still = Get-Process GoogleDriveFS -ErrorAction SilentlyContinue
        if ($still) { $still | Stop-Process -Force; Start-Sleep -Seconds 3 }
        Write-Host '  DriveFS stopped' -ForegroundColor Yellow
    }
}

if ($PSCmdlet.ShouldProcess($RegPath, "Set ContentCachePath=$CachePath")) {
    Set-ItemProperty -Path $RegPath -Name ContentCachePath -Value $CachePath -Type String
    $verify = (Get-ItemProperty -Path $RegPath -Name ContentCachePath).ContentCachePath
    if ($verify -ne $CachePath) { throw "Registry write did not stick: got '$verify'" }
    Write-Host "  registry set and verified: $verify" -ForegroundColor Green
    Write-Host "  previous value was: $current  (restore with -CachePath '$current')"
}

if ($Restart -and $PSCmdlet.ShouldProcess('GoogleDriveFS', 'Restart via gated launcher')) {
    & $Launcher -RequireMountLog
    Write-Host "  launcher exit code: $LASTEXITCODE"
}
