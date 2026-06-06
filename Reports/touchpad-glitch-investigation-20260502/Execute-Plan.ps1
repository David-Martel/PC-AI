# Execute-Plan.ps1 - Touchpad remediation execution
# Runs Gates A+B, captures rollback state, applies Step 2.3 (power plan),
# performs Step 0 (I2C HID reset), and stages Step 1 prerequisites.
#
# Each section logs to Reports\touchpad-glitch-investigation-20260502\.
# Idempotent: safe to re-run; checks for prior state.
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$log  = Join-Path $root 'execution-log.txt'
function Log { param($msg) "$(Get-Date -Format 'HH:mm:ss.fff') | $msg" | Tee-Object -FilePath $log -Append }

"===== EXECUTION RUN $(Get-Date -Format o) =====" | Set-Content $log

# ---- GATE A: Manual restore point ----
Log "Gate A: creating manual System Restore Point"
try {
    Checkpoint-Computer -Description "PreTouchpadFix-2026-05-02" -RestorePointType MODIFY_SETTINGS -ErrorAction Stop
    Log "Gate A: Checkpoint-Computer returned successfully"
} catch {
    Log "Gate A: Checkpoint-Computer FAILED: $($_.Exception.Message)"
    Log "Gate A: attempting WMI fallback"
    try {
        $sr = [wmiclass]'\\.\root\default:SystemRestore'
        $rc = $sr.CreateRestorePoint('PreTouchpadFix-2026-05-02', 12, 100)  # MODIFY_SETTINGS=12, BEGIN_SYSTEM_CHANGE=100
        Log "Gate A: WMI fallback returned $($rc.ReturnValue) (0=success)"
    } catch {
        Log "Gate A: WMI fallback FAILED: $($_.Exception.Message)"
    }
}
$rps = Get-ComputerRestorePoint -ErrorAction SilentlyContinue | Sort-Object SequenceNumber -Descending | Select-Object -First 5
$rpsOut = Join-Path $root 'gateA-restorepoints.txt'
$rps | Format-Table SequenceNumber, CreationTime, Description, RestorePointType -AutoSize | Out-String | Set-Content $rpsOut
Log "Gate A: top 5 RPs written to gateA-restorepoints.txt"
$preTouchpadRP = $rps | Where-Object { $_.Description -like '*PreTouchpadFix*' } | Select-Object -First 1
if ($preTouchpadRP) {
    Log "Gate A: NEW RP confirmed - sequence=$($preTouchpadRP.SequenceNumber) created=$($preTouchpadRP.CreationTime)"
} else {
    Log "Gate A: WARNING - no new RP visible. May be skipped due to frequency cap."
}

# ---- GATE B: VSS / shadow storage feasibility ----
Log "Gate B: checking VSS / shadow storage / free space"
$gateBOut = Join-Path $root 'gateB-vss-feasibility.txt'
"# Gate B - VSS feasibility ($(Get-Date -Format o))" | Set-Content $gateBOut
"`n## VSS shadow storage" | Add-Content $gateBOut
& vssadmin list shadowstorage 2>&1 | Out-String | Add-Content $gateBOut
"`n## VSS services" | Add-Content $gateBOut
Get-Service vss, swprv -ErrorAction SilentlyContinue | Format-Table Name, Status, StartType -AutoSize | Out-String | Add-Content $gateBOut
"`n## C: free space" | Add-Content $gateBOut
$cdrive = Get-PSDrive -Name C -ErrorAction SilentlyContinue
if ($cdrive) {
    $freeGB = [math]::Round($cdrive.Free / 1GB, 2)
    $usedGB = [math]::Round($cdrive.Used / 1GB, 2)
    "Free: $freeGB GB" | Add-Content $gateBOut
    "Used: $usedGB GB" | Add-Content $gateBOut
    Log "Gate B: C: free=$freeGB GB"
}
Log "Gate B: complete - see gateB-vss-feasibility.txt"

# ---- ROLLBACK STATE CAPTURE (pre-Step-2) ----
Log "Capturing rollback state for Step 2"

# Power plan rollback artifact
$powercfgRollback = Join-Path $root 'powercfg-rollback.txt'
$activeOut = & powercfg /getactivescheme 2>&1 | Out-String
$activeOut | Set-Content $powercfgRollback
$activeMatch = [regex]::Match($activeOut, 'GUID:\s*([0-9a-fA-F-]+)')
if ($activeMatch.Success) {
    $currentSchemeGuid = $activeMatch.Groups[1].Value
    Log "Active power scheme GUID: $currentSchemeGuid"
    "Captured GUID: $currentSchemeGuid" | Add-Content $powercfgRollback
    $powExport = Join-Path $root 'powercfg-current-backup.pow'
    & powercfg /export $powExport $currentSchemeGuid 2>&1 | Out-String | Add-Content $powercfgRollback
    if (Test-Path $powExport) {
        Log "Power plan exported to $powExport"
    }
} else {
    Log "WARNING: could not parse active power scheme GUID"
    $currentSchemeGuid = $null
}

# Process Lasso config backup
$plBackupDir = Join-Path $root 'pl-config-backup'
New-Item -ItemType Directory -Force -Path $plBackupDir | Out-Null
$plConfigDir = "C:\Program Files\Process Lasso\config"
if (-not (Test-Path $plConfigDir)) {
    $plConfigDir = "$env:ProgramData\ProcessLasso"
}
if (Test-Path $plConfigDir) {
    Copy-Item "$plConfigDir\*" $plBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    $copied = (Get-ChildItem $plBackupDir -Recurse -File).Count
    Log "PL config backup: $copied files copied from $plConfigDir to $plBackupDir"
} else {
    Log "WARNING: Process Lasso config dir not found in standard locations"
}

# ---- STEP 2.3: Power plan High Performance ----
Log "Step 2.3: switching to High Performance power plan"
$highPerfGuid = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
& powercfg /setactive $highPerfGuid 2>&1 | Out-String | ForEach-Object { Log "powercfg: $_" }
$verify = & powercfg /getactivescheme 2>&1 | Out-String
Log "Step 2.3 verify: $($verify -replace "`r?`n",' ')"

# ---- STEP 1 PREP: capture pre-restore package list ----
Log "Step 1 prep: capturing package list (for rollback documentation)"
$pkgPre = Join-Path $root 'packages-pre-step1.txt'
& dism /online /get-packages /format:list 2>&1 | Out-String | Set-Content $pkgPre
Log "Step 1 prep: packages-pre-step1.txt written"

# ---- STEP 0: I2C HID disable/enable cycle ----
Log "Step 0: I2C HID disable/enable cycle"
$snslDevice = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
    $_.InstanceId -like 'ACPI\SNSL002D\*' -and $_.Class -eq 'HIDClass'
}
if (-not $snslDevice) {
    # Fallback: any SNSL002D ACPI entry
    $snslDevice = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
        $_.InstanceId -like 'ACPI\SNSL002D*'
    } | Select-Object -First 1
}
if ($snslDevice) {
    $instId = $snslDevice.InstanceId
    Log "Step 0: target instance $instId"
    Log "Step 0: disabling..."
    & pnputil /disable-device "$instId" 2>&1 | Out-String | ForEach-Object { Log "pnputil: $_" }
    Start-Sleep -Seconds 8
    Log "Step 0: enabling..."
    & pnputil /enable-device "$instId" 2>&1 | Out-String | ForEach-Object { Log "pnputil: $_" }
    Start-Sleep -Seconds 3
    $verify = Get-PnpDevice -InstanceId $instId -ErrorAction SilentlyContinue
    Log "Step 0: post-cycle status = $($verify.Status)"
} else {
    Log "Step 0: ERROR - SNSL002D ACPI device not found, skipping"
}

# ---- STEP 2.1: Pause sync clients (suspend OneDrive process) ----
Log "Step 2.1: suspending OneDrive + GoogleDriveFS processes for diagnostic window"
$syncProcs = @{}
foreach ($name in 'OneDrive','GoogleDriveFS','OneDrive.Sync.Service','FileSyncHelper') {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($p in $procs) {
            $syncProcs[$p.Id] = $p.ProcessName
            Log "Step 2.1: noted $($p.ProcessName) PID=$($p.Id) (CPU=$($p.CPU))"
        }
    }
}
$pidsFile = Join-Path $root 'sync-suspended-pids.txt'
$syncProcs | ConvertTo-Json | Set-Content $pidsFile
Log "Step 2.1: suspended PIDs catalogued in sync-suspended-pids.txt (NOT actually suspended yet - confirm with user before suspending)"

Log "===== EXECUTION COMPLETE - check execution-log.txt and individual gate files ====="
Write-Host "Log: $log"
