#Requires -Version 7.0
<#
.SYNOPSIS
    Reports the 57-entry login-storm that causes post-login input lag and Shift+trackpad
    glitches on DTM-P1GEN7 (Lenovo ThinkPad P1 Gen 7, Windows 11), and optionally
    disables a curated set of DEFERRABLE apps from HKCU StartupApproved.

.DESCRIPTION
    Modern Windows tracks startup-app ENABLED/DISABLED state in three StartupApproved
    registry keys in addition to the classic Run/RunOnce/StartupFolder sources.  The first
    byte of each binary value controls state:
        0x02  = enabled   (first byte)
        0x03  = disabled  (first byte)
    The remaining 7 bytes are a FILETIME timestamp that Windows writes when the user
    toggles the state in Task Manager — always preserve the full 8-byte value.

    DEFAULT BEHAVIOUR (no switches): enumerates all three sources and categorises each
    entry as ESSENTIAL or DEFERRABLE, then prints a table.  NO registry writes occur.

    -Apply: disables the curated DEFERRABLE list (HKCU only, no admin required) by
    writing the 'disabled' byte (0x03) into StartupApproved.  Before any write the
    current bytes are saved to a timestamped JSON backup under .\backups\.  Idempotent:
    already-disabled entries are skipped.  Requires user confirmation (ConfirmImpact=High)
    unless -Confirm:$false is added.

    -Revert: restores the exact original byte arrays from the most recent backup (or the
    backup specified by -BackupFile).  Also HKCU only; no admin required.

    HKLM StartupApproved\Run is read for the report but NEVER written, even with -Apply.
    The script does not enumerate or touch HKLM.

    IMPORTANT: disabling these TRAY APPLICATIONS does not stop the ASIO/USB-audio
    DRIVERS (RME, Focusrite, Topping, miniDSP, PreSonus, EPOS) from loading — those
    load as kernel drivers regardless.  DPC-latency spikes from pro-audio drivers are
    unaffected.  This script only reduces userspace app-contention at login, which
    addresses the trackpad-scroll and Shift-ghost specifically.

    Curated NEVER-DISABLE entries (always left alone regardless of -Apply):
        OneDrive, Bitwarden, SecurityHealth, Tailscale, CloudflareWARP,
        any RME / Focusrite / Topping / miniDSP / PreSonus / EPOS panel,
        Lenovo Vantage, SynTP*, any HID/input driver tray app.

    Curated DEFERRABLE entries disabled by -Apply:
        Docker Desktop, Ollama, LM Studio (electron.app.LM Studio),
        GoogleDriveFS (2nd+ instances only — first is kept enabled),
        GoPro Webcam, MATLAB Connector, Mathworks Service Host,
        SOLIDWORKS 2024 Fast Start, SOLIDWORKS Background Downloader,
        AdobeBridge, Adobe Acrobat Synchronizer,
        WingetUI, NIRegistrationWizard.

.PARAMETER Apply
    Disable the curated DEFERRABLE startup entries in HKCU StartupApproved.
    No admin required.  Backs up current bytes first.  Requires confirmation unless
    -Confirm:$false is passed.

.PARAMETER Revert
    Restore HKCU StartupApproved values from the most recent backup JSON (or
    the backup specified by -BackupFile).

.PARAMETER BackupDir
    Directory for backup JSON files.
    Default: the 'backups' folder next to this script.

.PARAMETER BackupFile
    Explicit backup JSON path.  Used with -Revert to target a specific snapshot
    rather than the most recent one.

.EXAMPLE
    # Report only (default) — safe, no writes:
    pwsh -File .\Optimize-StartupLoad.ps1

.EXAMPLE
    # Apply with confirmation prompt:
    pwsh -File .\Optimize-StartupLoad.ps1 -Apply

.EXAMPLE
    # Apply without prompt (e.g. in a remediation pipeline):
    pwsh -File .\Optimize-StartupLoad.ps1 -Apply -Confirm:$false

.EXAMPLE
    # Revert from the most recent backup:
    pwsh -File .\Optimize-StartupLoad.ps1 -Revert

.EXAMPLE
    # Revert from a specific backup file:
    pwsh -File .\Optimize-StartupLoad.ps1 -Revert -BackupFile .\backups\startup-approved-20260530-143000.json

.NOTES
    Author  : input-stack investigation (Claude Code) - 2026-05-30
    Target  : DTM-P1GEN7 / Windows 11 26200
    Safety  : HKCU only, full byte-level backup/restore, idempotent, ShouldProcess.
              HKLM entries are enumerated for report but NEVER modified.
    Ref     : https://learn.microsoft.com/windows/win32/setupapi/run-and-runonce-registry-keys
              Task Manager "Startup Apps" tab reads StartupApproved\Run to determine
              the Enabled/Disabled state it displays and persists.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [switch]$Apply,
    [switch]$Revert,
    [string]$BackupDir = (Join-Path $PSScriptRoot 'backups'),
    [string]$BackupFile
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Registry key paths
# ---------------------------------------------------------------------------
$HkcuApprovedRun     = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'
$HkcuApprovedFolder  = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\StartupFolder'
$HklmApprovedRun     = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'

# ---------------------------------------------------------------------------
# Byte-state constants (first byte of the 8-byte binary value)
# ---------------------------------------------------------------------------
$BYTE_ENABLED  = [byte]0x02
$BYTE_DISABLED = [byte]0x03

# ---------------------------------------------------------------------------
# Curated lists
# ---------------------------------------------------------------------------
# Exact names as they appear in StartupApproved (or Win32_StartupCommand.Name).
# These are the entries -Apply will disable.
$CURATED_DEFERRABLE = [System.Collections.Generic.HashSet[string]]([System.StringComparer]::OrdinalIgnoreCase)
@(
    'Docker Desktop'
    'Ollama'
    'electron.app.LM Studio'
    'GoogleDriveFS'          # Special: 2nd+ instances only; first kept
    'GoPro Webcam'
    'MATLAB Connector'
    'Mathworks Service Host'
    'SOLIDWORKS 2024 Fast Start'
    'SOLIDWORKS Background Downloader'
    'AdobeBridge'
    'Adobe Acrobat Synchronizer'
    'WingetUI'
    'NIRegistrationWizard'
) | ForEach-Object { [void]$CURATED_DEFERRABLE.Add($_) }

# Regex for entries that must NEVER be disabled, regardless of categorisation.
# This is a hard veto applied before any write.
$NEVER_DISABLE_REGEX = [System.Text.RegularExpressions.Regex]::new(
    'OneDrive|Bitwarden|SecurityHealth|Tailscale|CloudflareWARP|' +
    'RME|Focusrite|Topping|miniDSP|PreSonus|EPOS|' +
    'Lenovo|SynTP|Synaptics|HID|ThinkPad',
    [System.Text.RegularExpressions.RegexOptions]::IgnoreCase -bor
    [System.Text.RegularExpressions.RegexOptions]::Compiled
)

# ---------------------------------------------------------------------------
# Helper: read all values from a registry key as @{ Name -> byte[] }
# Returns an empty hashtable when the key does not exist.
# ---------------------------------------------------------------------------
function Get-StartupApprovedBytes {
    param([string]$KeyPath)
    $result = @{}
    if (-not (Test-Path $KeyPath)) { return $result }
    $props = Get-ItemProperty -Path $KeyPath -ErrorAction SilentlyContinue
    foreach ($prop in $props.PSObject.Properties) {
        if ($prop.Name -like 'PS*') { continue }   # skip PS built-in properties
        $val = $prop.Value
        if ($val -is [byte[]]) {
            $result[$prop.Name] = $val
        }
    }
    return $result
}

# ---------------------------------------------------------------------------
# Helper: decode enabled/disabled from first byte
# ---------------------------------------------------------------------------
function Get-ApprovedState {
    param([byte[]]$Bytes)
    if ($null -eq $Bytes -or $Bytes.Count -eq 0) { return 'Unknown' }
    if ($Bytes[0] -eq $BYTE_ENABLED)  { return 'Enabled'  }
    if ($Bytes[0] -eq $BYTE_DISABLED) { return 'Disabled' }
    return ("Unknown(0x{0:X2})" -f $Bytes[0])
}

# ---------------------------------------------------------------------------
# Helper: categorise an entry by name
# ---------------------------------------------------------------------------
function Get-EntryCategory {
    param([string]$Name, [string]$Command)
    if ($NEVER_DISABLE_REGEX.IsMatch($Name) -or $NEVER_DISABLE_REGEX.IsMatch([string]$Command)) {
        return 'ESSENTIAL'
    }
    if ($CURATED_DEFERRABLE.Contains($Name)) {
        return 'DEFERRABLE'
    }
    return 'REVIEW'   # present but not in curated list — user should evaluate
}

# ---------------------------------------------------------------------------
# REVERT path
# ---------------------------------------------------------------------------
if ($Revert) {
    if (-not $BackupFile) {
        $BackupFile = Get-ChildItem -Path $BackupDir -Filter 'startup-approved-*.json' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $BackupFile -or -not (Test-Path $BackupFile)) {
        throw "No startup-approved backup found in '$BackupDir'. Specify -BackupFile."
    }

    Write-Host "Reverting StartupApproved from backup: $BackupFile" -ForegroundColor Cyan
    $backup = Get-Content $BackupFile -Raw | ConvertFrom-Json

    foreach ($section in @('HkcuRun', 'HkcuFolder')) {
        $keyPath = if ($section -eq 'HkcuRun') { $HkcuApprovedRun } else { $HkcuApprovedFolder }
        if (-not $backup.$section) { continue }
        foreach ($entry in $backup.$section.PSObject.Properties) {
            $name  = $entry.Name
            $bytes = [byte[]]($entry.Value)
            if ($PSCmdlet.ShouldProcess("$keyPath\$name", "restore $($bytes.Count) bytes")) {
                if (Test-Path $keyPath) {
                    Set-ItemProperty -Path $keyPath -Name $name -Value $bytes -Type Binary
                    Write-Host ("  [RESTORED] {0,-40} -> {1}" -f $name, (Get-ApprovedState $bytes))
                }
            }
        }
    }
    Write-Host "Revert complete. Sign out/in for startup changes to take effect." -ForegroundColor Green
    return
}

# ---------------------------------------------------------------------------
# ENUMERATE startup entries from Win32_StartupCommand + StartupApproved
# ---------------------------------------------------------------------------
Write-Host "Enumerating startup load..." -ForegroundColor Cyan

# Win32_StartupCommand gives us Name, Location, Command (all locations)
$startupCommands = Get-CimInstance Win32_StartupCommand -ErrorAction SilentlyContinue |
    Select-Object Name, Location, Command |
    Sort-Object Location, Name

# Read the binary state from StartupApproved keys
$hkcuRunBytes    = Get-StartupApprovedBytes -KeyPath $HkcuApprovedRun
$hkcuFolderBytes = Get-StartupApprovedBytes -KeyPath $HkcuApprovedFolder

# Read HKLM for reporting only (no admin check — silently empty if access denied)
$hklmRunBytes    = Get-StartupApprovedBytes -KeyPath $HklmApprovedRun

# Build unified report combining both sources
$entries = [System.Collections.Generic.List[pscustomobject]]::new()

foreach ($cmd in $startupCommands) {
    $bytes = $null
    $approvedKey = '(not in StartupApproved)'

    if ($hkcuRunBytes.ContainsKey($cmd.Name)) {
        $bytes = $hkcuRunBytes[$cmd.Name]
        $approvedKey = 'HKCU\Run'
    } elseif ($hkcuFolderBytes.ContainsKey($cmd.Name)) {
        $bytes = $hkcuFolderBytes[$cmd.Name]
        $approvedKey = 'HKCU\StartupFolder'
    } elseif ($hklmRunBytes.ContainsKey($cmd.Name)) {
        $bytes = $hklmRunBytes[$cmd.Name]
        $approvedKey = 'HKLM\Run (read-only)'
    }

    $state    = Get-ApprovedState -Bytes $bytes
    $category = Get-EntryCategory -Name $cmd.Name -Command $cmd.Command

    $entries.Add([pscustomobject]@{
        Category    = $category
        State       = $state
        Name        = $cmd.Name
        Location    = $cmd.Location
        ApprovedKey = $approvedKey
        Command     = $cmd.Command
    })
}

# Also capture any StartupApproved entries NOT present in Win32_StartupCommand
# (orphaned entries that WMI misses but Task Manager still sees)
$allApprovedNames = ($hkcuRunBytes.Keys + $hkcuFolderBytes.Keys) | Select-Object -Unique
foreach ($name in $allApprovedNames) {
    if ($entries.Name -contains $name) { continue }
    $bytes = if ($hkcuRunBytes.ContainsKey($name)) { $hkcuRunBytes[$name] } else { $hkcuFolderBytes[$name] }
    $approvedKey = if ($hkcuRunBytes.ContainsKey($name)) { 'HKCU\Run' } else { 'HKCU\StartupFolder' }
    $state    = Get-ApprovedState -Bytes $bytes
    $category = Get-EntryCategory -Name $name -Command ''
    $entries.Add([pscustomobject]@{
        Category    = $category
        State       = $state
        Name        = $name
        Location    = '(StartupApproved only)'
        ApprovedKey = $approvedKey
        Command     = ''
    })
}

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
$essential  = $entries | Where-Object { $_.Category -eq 'ESSENTIAL' }
$deferrable = $entries | Where-Object { $_.Category -eq 'DEFERRABLE' }
$review     = $entries | Where-Object { $_.Category -eq 'REVIEW' }

Write-Host ""
Write-Host "===== STARTUP LOAD REPORT — DTM-P1GEN7 =====" -ForegroundColor Cyan
Write-Host ("Total entries : {0}  |  ESSENTIAL: {1}  DEFERRABLE: {2}  REVIEW: {3}" -f
    $entries.Count, $essential.Count, $deferrable.Count, $review.Count)
Write-Host ""

Write-Host "--- ESSENTIAL (keep as-is) ---" -ForegroundColor Green
$essential | Format-Table -AutoSize -Wrap -Property Category, State, Name, ApprovedKey |
    Out-String | Write-Host

Write-Host "--- DEFERRABLE (candidates for -Apply disable) ---" -ForegroundColor Yellow
$deferrable | Format-Table -AutoSize -Wrap -Property Category, State, Name, ApprovedKey |
    Out-String | Write-Host

if ($review.Count -gt 0) {
    Write-Host "--- REVIEW (not in curated list — evaluate manually) ---" -ForegroundColor DarkYellow
    $review | Format-Table -AutoSize -Wrap -Property Category, State, Name, ApprovedKey, Command |
        Out-String | Write-Host
}

Write-Host "NOTE: ASIO/USB-audio DRIVERS (RME, Focusrite, Topping, miniDSP, PreSonus, EPOS)" -ForegroundColor DarkCyan
Write-Host "      load as kernel drivers regardless of tray-app startup state." -ForegroundColor DarkCyan
Write-Host "      Disabling tray apps reduces login-storm contention but does NOT reduce DPC latency." -ForegroundColor DarkCyan
Write-Host "      Use LatencyMon during a loaded session to identify DPC offenders directly." -ForegroundColor DarkCyan

# ---------------------------------------------------------------------------
# APPLY path
# ---------------------------------------------------------------------------
if (-not $Apply) {
    Write-Host ""
    Write-Host "Run with -Apply to disable the DEFERRABLE entries above (HKCU only, no admin needed)." -ForegroundColor DarkGray
    return
}

# --- Backup first ---
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$stamp      = (Get-Date).ToString('yyyyMMdd-HHmmss')
$backupPath = if ($BackupFile) { $BackupFile } else { Join-Path $BackupDir "startup-approved-$stamp.json" }

# Capture current bytes for all HKCU entries (before any change)
$hkcuRunSnapshot    = @{}
$hkcuFolderSnapshot = @{}
foreach ($k in $hkcuRunBytes.Keys)    { $hkcuRunSnapshot[$k]    = [int[]]$hkcuRunBytes[$k] }
foreach ($k in $hkcuFolderBytes.Keys) { $hkcuFolderSnapshot[$k] = [int[]]$hkcuFolderBytes[$k] }

$backupObj = [ordered]@{
    Timestamp   = (Get-Date).ToString('o')
    Machine     = $env:COMPUTERNAME
    HkcuRun     = $hkcuRunSnapshot
    HkcuFolder  = $hkcuFolderSnapshot
}
$backupObj | ConvertTo-Json -Depth 5 | Set-Content -Path $backupPath -Encoding utf8
Write-Host "Backup written: $backupPath" -ForegroundColor Cyan
Write-Host ""

# --- Disable curated deferrable entries ---
# Track GoogleDriveFS occurrences to keep the first instance enabled.
$googleDriveFsCount = 0

$totalDisabled  = 0
$totalSkipped   = 0

foreach ($entry in $entries) {
    if ($entry.Category -ne 'DEFERRABLE') { continue }
    $name = $entry.Name

    # Hard veto: never-disable regex (belt-and-suspenders)
    if ($NEVER_DISABLE_REGEX.IsMatch($name)) {
        Write-Warning "VETO: '$name' matched never-disable list — skipping despite DEFERRABLE category."
        $totalSkipped++
        continue
    }

    # GoogleDriveFS: keep first instance, disable subsequent ones
    if ($name -eq 'GoogleDriveFS') {
        $googleDriveFsCount++
        if ($googleDriveFsCount -eq 1) {
            Write-Host ("  [KEEP-1st] {0,-40} (first GoogleDriveFS instance retained)" -f $name) -ForegroundColor DarkGray
            $totalSkipped++
            continue
        }
    }

    # Determine which HKCU sub-key contains this entry
    $keyPath = $null
    $bytes   = $null
    if ($hkcuRunBytes.ContainsKey($name)) {
        $keyPath = $HkcuApprovedRun
        $bytes   = $hkcuRunBytes[$name]
    } elseif ($hkcuFolderBytes.ContainsKey($name)) {
        $keyPath = $HkcuApprovedFolder
        $bytes   = $hkcuFolderBytes[$name]
    } else {
        # Entry appears in HKLM only — skip (never write HKLM)
        Write-Host ("  [SKIP-HKLM] {0,-40} (HKLM entry — not modified; admin required)" -f $name) -ForegroundColor DarkGray
        $totalSkipped++
        continue
    }

    # Idempotency: already disabled?
    if ($bytes.Count -gt 0 -and $bytes[0] -eq $BYTE_DISABLED) {
        Write-Host ("  [ALREADY-OFF] {0,-40}" -f $name) -ForegroundColor DarkGray
        $totalSkipped++
        continue
    }

    # Write the change: set first byte to 0x03, preserve the rest
    $newBytes = [byte[]]($bytes)
    if ($newBytes.Count -lt 8) {
        # Pad to minimum 8 bytes (disabled FILETIME = all zeros is acceptable)
        $newBytes = [byte[]](New-Object byte[] 8)
    }
    $newBytes[0] = $BYTE_DISABLED

    if ($PSCmdlet.ShouldProcess("$keyPath : $name", "disable (first byte 0x02 -> 0x03)")) {
        Set-ItemProperty -Path $keyPath -Name $name -Value $newBytes -Type Binary
        Write-Host ("  [DISABLED] {0,-40} in {1}" -f $name, $keyPath) -ForegroundColor Yellow
        $totalDisabled++
    }
}

Write-Host ""
Write-Host ("Startup optimisation complete: {0} disabled, {1} skipped." -f $totalDisabled, $totalSkipped) -ForegroundColor Green
Write-Host "Sign out and back in (or reboot) for changes to take effect." -ForegroundColor Green
Write-Host ("Revert anytime:  pwsh -File `"$PSCommandPath`" -Revert") -ForegroundColor Green
