# Step 2.1 - Graceful sync pause (OneDrive + Google Drive).
# Strategy: graceful shutdown via OneDrive.exe /shutdown (cleanly closes session,
# no in-flight sync corruption), then suspend GoogleDriveFS process.
# Resume: relaunch OneDrive.exe and resume GoogleDriveFS.
# This is REVERSIBLE - normal sync resumes after relaunch.
#
# Run with -Resume to undo (relaunches OneDrive, resumes Google Drive).
[CmdletBinding()]
param(
    [switch]$Resume,
    [int]$ObserveMinutes = 15
)
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$log  = Join-Path $root 'step2_1-sync-pause.log'
function Log { param($msg) "$(Get-Date -Format 'HH:mm:ss.fff') | $msg" | Tee-Object -FilePath $log -Append }

if ($Resume) {
    Log "==== Step 2.1 RESUME ===="
    # Relaunch OneDrive
    $odPath = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\OneDrive.exe'
    if (Test-Path $odPath) {
        Log "Relaunching OneDrive: $odPath /background"
        Start-Process -FilePath $odPath -ArgumentList '/background' -ErrorAction SilentlyContinue
    }
    # Relaunch Google Drive (typical install path)
    $gdPaths = @(
        "$env:ProgramFiles\Google\Drive File Stream\launch.bat",
        "$env:ProgramFiles\Google\Drive File Stream\GoogleDriveFS.exe"
    )
    foreach ($p in $gdPaths) {
        if (Test-Path $p) {
            Log "Relaunching Google Drive: $p"
            Start-Process -FilePath $p -ErrorAction SilentlyContinue
            break
        }
    }
    Start-Sleep -Seconds 3
    Log "Verify: OneDrive procs:"
    Get-Process OneDrive,GoogleDriveFS -ErrorAction SilentlyContinue | ForEach-Object {
        Log "  $($_.ProcessName) PID=$($_.Id) CPU=$($_.CPU)"
    }
    Log "==== RESUME COMPLETE ===="
    return
}

Log "==== Step 2.1 PAUSE START ===="
Log "Pre-state OneDrive/GoogleDrive process inventory:"
Get-Process OneDrive,GoogleDriveFS,FileSyncHelper,'OneDrive.Sync.Service' -ErrorAction SilentlyContinue | ForEach-Object {
    Log "  $($_.ProcessName) PID=$($_.Id) CPU=$($_.CPU)"
}

# 1. Graceful OneDrive shutdown
$odExe = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\OneDrive.exe'
if (-not (Test-Path $odExe)) {
    $odExe = (Get-Process OneDrive -ErrorAction SilentlyContinue | Select-Object -First 1).Path
}
if ($odExe) {
    Log "Issuing OneDrive /shutdown"
    & $odExe /shutdown 2>&1 | Out-String | ForEach-Object { Log "OneDrive: $_" }
    Start-Sleep -Seconds 5
    $stillUp = Get-Process OneDrive -ErrorAction SilentlyContinue
    if ($stillUp) {
        Log "OneDrive still running after /shutdown; attempting Stop-Process"
        $stillUp | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
} else {
    Log "OneDrive.exe path not found"
}

# 2. Stop OneDrive sidecar processes
foreach ($name in 'OneDrive.Sync.Service','FileSyncHelper') {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        Log "Stopping $($p.ProcessName) PID=$($p.Id)"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}

# 3. Stop Google Drive (no graceful shutdown command - direct stop)
foreach ($name in 'GoogleDriveFS') {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        Log "Stopping $($p.ProcessName) PID=$($p.Id)"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 3
Log "Post-pause state:"
$remaining = Get-Process OneDrive,GoogleDriveFS,FileSyncHelper,'OneDrive.Sync.Service' -ErrorAction SilentlyContinue
if ($remaining) {
    foreach ($p in $remaining) {
        Log "  STILL RUNNING: $($p.ProcessName) PID=$($p.Id)"
    }
} else {
    Log "  All sync processes stopped"
}

Log "Sync paused. Test touchpad for $ObserveMinutes minutes."
Log "To resume sync: & '$PSCommandPath' -Resume"
Log "==== Step 2.1 PAUSE COMPLETE ===="
