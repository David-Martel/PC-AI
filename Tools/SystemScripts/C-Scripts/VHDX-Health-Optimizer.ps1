#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Comprehensive VHDX and Virtual Disk Health Optimization for Windows with WSL/Docker.

.DESCRIPTION
  Production-ready script that:
  - Performs VHDX health checks (access, locks, fragmentation, sparse file status)
  - Compacts VHDX files to reclaim unused space
  - Integrates with Optimize-Disks.ps1 for physical disk TRIM/optimization
  - Manages WSL2 and Docker Desktop VHDX files safely
  - Configures scheduled maintenance tasks
  - Maintains detailed logs for troubleshooting

.PARAMETER Action
  Optimize   - Full optimization cycle (default)
  Check      - Health check only, no modifications
  Compact    - VHDX compaction only
  Cleanup    - Remove temp files and reclaim space
  Schedule   - Configure scheduled task
  Unschedule - Remove scheduled task
  Report     - Generate detailed health report

.PARAMETER Force
  Skip confirmation prompts.

.PARAMETER IncludeDocker
  Include Docker Desktop VHDX files (requires Docker stopped).

.PARAMETER IncludeWSL
  Include WSL distribution VHDX files (requires WSL shutdown).

.PARAMETER LogPath
  Custom log file path. Default: ~\logs\vhdx-health-optimizer.log

.PARAMETER ScheduleTime
  Time for scheduled task (24h format). Default: 04:00

.PARAMETER ScheduleDay
  Day for scheduled task. Default: Sunday

.NOTES
  Author: VHDX Health Optimizer
  Version: 1.0.0
  Requires: Windows 10/11 with Hyper-V, WSL2, or Docker Desktop

  SAFETY: This script uses graceful shutdown procedures to prevent kernel crashes.
          It will NOT force-restart vmcompute or other critical services.
#>

[CmdletBinding()]
param(
    [ValidateSet('Optimize','Check','Compact','Cleanup','Schedule','Unschedule','Report')]
    [string]$Action = 'Optimize',

    [switch]$Force,
    [switch]$IncludeDocker,
    [switch]$IncludeWSL,
    [string]$LogPath = "$env:USERPROFILE\logs\vhdx-health-optimizer.log",
    [string]$ScheduleTime = '04:00',
    [string]$ScheduleDay = 'Sunday'
)

#region Configuration
$Script:Config = @{
    Version = '1.0.0'
    LogPath = $LogPath

    # Known VHDX locations
    VHDXPaths = @{
        DockerDesktop = @(
            "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx",
            "$env:LOCALAPPDATA\Docker\wsl\main\ext4.vhdx",
            "T:\vm\docker\DockerDesktopWSL\main\ext4.vhdx",
            "T:\vm\docker\DockerDesktopWSL\disk\docker_data.vhdx"
        )
        WSLDistributions = "$env:LOCALAPPDATA\Packages\*\LocalState\ext4.vhdx"
        Custom = @(
            "T:\vm\*.vhdx",
            "T:\vm\**\*.vhdx",
            "$env:USERPROFILE\*.vhdx"
        )
    }

    # Safety thresholds
    Thresholds = @{
        MinFreeDiskGB = 10           # Minimum free disk space to proceed
        MaxVHDXSizeGB = 256          # Warning threshold for VHDX size
        MaxCompactionTimeMinutes = 30 # Timeout for compaction operations
        MinVHDXFreePercent = 20      # Minimum free space inside VHDX to skip compaction
    }

    # Related scripts to integrate (consolidated in C:\Scripts\)
    RelatedScripts = @{
        OptimizeDisks = "C:\Scripts\Optimize-Disks.ps1"
        WSLOptimizer  = "C:\Scripts\wsl2-comprehensive-optimizer.ps1"
        CheckVHDX     = "C:\Scripts\check-vhdx-access.ps1"
    }

    # Scheduled task configuration
    ScheduledTask = @{
        Name = 'VHDX Health Optimizer - Weekly Maintenance'
        Description = 'Performs routine VHDX health checks, compaction, and disk optimization'
    }
}
#endregion

#region Logging Functions
function Initialize-Logging {
    $logDir = Split-Path $Script:Config.LogPath -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    # Rotate log if > 5MB
    if (Test-Path $Script:Config.LogPath) {
        $logSize = (Get-Item $Script:Config.LogPath).Length / 1MB
        if ($logSize -gt 5) {
            $archivePath = $Script:Config.LogPath -replace '\.log$', "-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
            Move-Item $Script:Config.LogPath $archivePath -Force
        }
    }
}

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('INFO','SUCCESS','WARNING','ERROR','DEBUG')]
        [string]$Level = 'INFO',
        [string]$Color = 'White'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logEntry = "[$timestamp] [$Level] $Message"

    $colorMap = @{
        'INFO' = 'White'
        'SUCCESS' = 'Green'
        'WARNING' = 'Yellow'
        'ERROR' = 'Red'
        'DEBUG' = 'Gray'
    }

    if (-not $Color -or $Color -eq 'White') {
        $Color = $colorMap[$Level]
    }

    Write-Host $logEntry -ForegroundColor $Color
    Add-Content -Path $Script:Config.LogPath -Value $logEntry -Encoding UTF8
}
#endregion

#region Safety Functions
function Test-SafeToOperate {
    <#
    .SYNOPSIS
      Validates system state before potentially disruptive operations.
    #>
    param([switch]$RequireWSLDown, [switch]$RequireDockerDown)

    $issues = @()

    # Check disk space
    $systemDrive = Get-Volume -DriveLetter C -ErrorAction SilentlyContinue
    if ($systemDrive) {
        $freeGB = [math]::Round($systemDrive.SizeRemaining / 1GB, 2)
        if ($freeGB -lt $Script:Config.Thresholds.MinFreeDiskGB) {
            $issues += "System drive has only ${freeGB}GB free (minimum: $($Script:Config.Thresholds.MinFreeDiskGB)GB)"
        }
    }

    # Check WSL state if required
    if ($RequireWSLDown) {
        $wslState = wsl --list --running 2>&1
        if ($wslState -notmatch 'no running distributions|Windows Subsystem for Linux has no') {
            $issues += "WSL distributions are running. Use 'wsl --shutdown' first."
        }
    }

    # Check Docker state if required
    if ($RequireDockerDown) {
        $dockerProcess = Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue
        if ($dockerProcess) {
            $issues += "Docker Desktop is running. Stop it before VHDX operations."
        }
    }

    # Check vmcompute state (informational only - we don't force restart)
    $vmcompute = Get-Service -Name vmcompute -ErrorAction SilentlyContinue
    if ($vmcompute -and $vmcompute.Status -ne 'Running') {
        Write-Log "vmcompute service is not running (Status: $($vmcompute.Status))" "WARNING"
    }

    if ($issues.Count -gt 0) {
        foreach ($issue in $issues) {
            Write-Log $issue "ERROR"
        }
        return $false
    }

    return $true
}

function Stop-WSLGracefully {
    <#
    .SYNOPSIS
      Gracefully shuts down WSL without risking kernel crash.
    #>
    Write-Log "Gracefully shutting down WSL..." "INFO"

    try {
        # First, try graceful shutdown
        $result = wsl --shutdown 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "WSL shutdown initiated" "SUCCESS"

            # Wait for WSL to fully stop (up to 30 seconds)
            $timeout = 30
            $elapsed = 0
            while ($elapsed -lt $timeout) {
                Start-Sleep -Seconds 2
                $elapsed += 2

                $running = wsl --list --running 2>&1
                if ($running -match 'no running distributions|Windows Subsystem for Linux has no') {
                    Write-Log "WSL fully stopped after ${elapsed}s" "SUCCESS"
                    return $true
                }
            }

            Write-Log "WSL shutdown timed out after ${timeout}s" "WARNING"
            return $false
        }
        else {
            Write-Log "WSL shutdown command failed: $result" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "WSL shutdown error: $($_.Exception.Message)" "ERROR"
        return $false
    }
}
#endregion

#region VHDX Discovery Functions
function Get-AllVHDXFiles {
    <#
    .SYNOPSIS
      Discovers all VHDX files on the system.
    #>
    param(
        [switch]$IncludeDocker,
        [switch]$IncludeWSL
    )

    Write-Log "Discovering VHDX files..." "INFO"

    $allVHDX = @()

    # Docker Desktop VHDX files
    if ($IncludeDocker) {
        foreach ($path in $Script:Config.VHDXPaths.DockerDesktop) {
            if (Test-Path $path) {
                $allVHDX += Get-Item $path
                Write-Log "Found Docker VHDX: $path" "DEBUG"
            }
        }
    }

    # WSL Distribution VHDX files
    if ($IncludeWSL) {
        $wslPattern = $Script:Config.VHDXPaths.WSLDistributions
        $wslFiles = Get-ChildItem -Path $wslPattern -ErrorAction SilentlyContinue
        foreach ($file in $wslFiles) {
            $allVHDX += $file
            Write-Log "Found WSL VHDX: $($file.FullName)" "DEBUG"
        }
    }

    # Custom VHDX locations
    foreach ($pattern in $Script:Config.VHDXPaths.Custom) {
        try {
            $customFiles = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
            foreach ($file in $customFiles) {
                if ($allVHDX.FullName -notcontains $file.FullName) {
                    $allVHDX += $file
                    Write-Log "Found custom VHDX: $($file.FullName)" "DEBUG"
                }
            }
        }
        catch {
            Write-Log "Cannot search pattern $pattern" "DEBUG"
        }
    }

    Write-Log "Discovered $($allVHDX.Count) VHDX file(s)" "INFO"
    return $allVHDX
}

function Get-VHDXInfo {
    <#
    .SYNOPSIS
      Gets detailed information about a VHDX file.
    #>
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    $file = Get-Item $Path
    $info = @{
        Path = $file.FullName
        Name = $file.Name
        Directory = $file.DirectoryName
        SizeGB = [math]::Round($file.Length / 1GB, 2)
        LastWriteTime = $file.LastWriteTime
        IsLocked = $false
        IsSparse = $false
        VirtualSize = $null
        Type = 'Unknown'
        Health = 'Unknown'
    }

    # Determine VHDX type
    if ($info.Path -match 'Docker') {
        $info.Type = 'DockerDesktop'
    }
    elseif ($info.Path -match 'CanonicalGroupLimited|Ubuntu|Debian|SUSE') {
        $info.Type = 'WSLDistribution'
    }
    else {
        $info.Type = 'Custom'
    }

    # Check if file is locked
    try {
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        $stream.Close()
        $info.IsLocked = $false
    }
    catch {
        $info.IsLocked = $true
    }

    # Check sparse file attribute
    $attributes = [System.IO.File]::GetAttributes($Path)
    $info.IsSparse = $attributes -band [System.IO.FileAttributes]::SparseFile

    # Try to get VHDX metadata using Hyper-V cmdlet
    try {
        $vhdxMeta = Get-VHD -Path $Path -ErrorAction Stop
        $info.VirtualSize = [math]::Round($vhdxMeta.Size / 1GB, 2)
        $info.Health = if ($vhdxMeta.Attached) { 'Attached' } else { 'Detached' }
        $info.BlockSize = $vhdxMeta.BlockSize
        $info.VhdFormat = $vhdxMeta.VhdFormat
        $info.VhdType = $vhdxMeta.VhdType
        $info.FragmentationPercentage = $vhdxMeta.FragmentationPercentage
    }
    catch {
        Write-Log "Cannot get VHD metadata for $Path (Hyper-V module may not be available)" "DEBUG"
    }

    return [pscustomobject]$info
}
#endregion

#region Optimization Functions
function Optimize-VHDXFile {
    <#
    .SYNOPSIS
      Compacts a VHDX file to reclaim unused space.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [switch]$Force
    )

    Write-Log "Optimizing VHDX: $Path" "INFO"

    $vhdxInfo = Get-VHDXInfo -Path $Path
    if (-not $vhdxInfo) {
        Write-Log "VHDX file not found: $Path" "ERROR"
        return $false
    }

    # Check if locked
    if ($vhdxInfo.IsLocked) {
        Write-Log "VHDX is locked (in use): $Path" "WARNING"
        return $false
    }

    # Size before compaction
    $sizeBefore = $vhdxInfo.SizeGB

    # Perform compaction using Hyper-V cmdlet
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

        Write-Log "Starting compaction (this may take several minutes)..." "INFO"

        # Mount VHDX read-only first to allow internal optimization
        Mount-VHD -Path $Path -ReadOnly -ErrorAction Stop

        # Let Windows optimize the mounted disk
        Start-Sleep -Seconds 5

        # Dismount
        Dismount-VHD -Path $Path -ErrorAction Stop

        # Now compact
        Optimize-VHD -Path $Path -Mode Full -ErrorAction Stop

        $stopwatch.Stop()

        # Get new size
        $newInfo = Get-VHDXInfo -Path $Path
        $sizeAfter = $newInfo.SizeGB
        $savedGB = [math]::Round($sizeBefore - $sizeAfter, 2)

        if ($savedGB -gt 0) {
            Write-Log "Compaction complete: Saved ${savedGB}GB (${sizeBefore}GB -> ${sizeAfter}GB) in $([math]::Round($stopwatch.Elapsed.TotalMinutes, 1)) minutes" "SUCCESS"
        }
        else {
            Write-Log "Compaction complete: No space reclaimed (VHDX was already optimized)" "INFO"
        }

        return $true
    }
    catch {
        Write-Log "Compaction failed: $($_.Exception.Message)" "ERROR"

        # Ensure VHDX is dismounted on error
        try {
            Dismount-VHD -Path $Path -ErrorAction SilentlyContinue
        }
        catch {}

        return $false
    }
}

function Invoke-PhysicalDiskOptimization {
    <#
    .SYNOPSIS
      Runs the Optimize-Disks.ps1 script if available.
    #>
    $optimizeScript = $Script:Config.RelatedScripts.OptimizeDisks

    if (Test-Path $optimizeScript) {
        Write-Log "Running physical disk optimization via Optimize-Disks.ps1..." "INFO"

        try {
            & $optimizeScript -ShowOnly:$false
            Write-Log "Physical disk optimization complete" "SUCCESS"
            return $true
        }
        catch {
            Write-Log "Physical disk optimization failed: $($_.Exception.Message)" "ERROR"
            return $false
        }
    }
    else {
        Write-Log "Optimize-Disks.ps1 not found at $optimizeScript" "WARNING"

        # Fallback: Run built-in Optimize-Volume for SSDs
        Write-Log "Running built-in disk optimization..." "INFO"

        $volumes = Get-Volume | Where-Object {
            $_.DriveType -eq 'Fixed' -and $_.FileSystem -in @('NTFS','ReFS')
        }

        foreach ($vol in $volumes) {
            if ($vol.DriveLetter) {
                try {
                    Write-Log "Optimizing volume $($vol.DriveLetter):" "INFO"
                    Optimize-Volume -DriveLetter $vol.DriveLetter -ReTrim -ErrorAction SilentlyContinue
                }
                catch {
                    Write-Log "Could not optimize volume $($vol.DriveLetter): $($_.Exception.Message)" "WARNING"
                }
            }
        }

        return $true
    }
}

function Clear-TempFiles {
    <#
    .SYNOPSIS
      Clears temporary files to free up disk space.
    #>
    Write-Log "Cleaning up temporary files..." "INFO"

    $tempPaths = @(
        "$env:TEMP\*",
        "$env:LOCALAPPDATA\Temp\*",
        "$env:WINDIR\Temp\*",
        "$env:LOCALAPPDATA\Docker\log\*.log",
        "$env:LOCALAPPDATA\Docker\log\host\*.log"
    )

    $totalFreed = 0

    foreach ($path in $tempPaths) {
        try {
            $files = Get-ChildItem -Path $path -Recurse -Force -ErrorAction SilentlyContinue |
                     Where-Object { -not $_.PSIsContainer -and $_.LastWriteTime -lt (Get-Date).AddDays(-7) }

            foreach ($file in $files) {
                try {
                    $size = $file.Length
                    Remove-Item $file.FullName -Force -ErrorAction Stop
                    $totalFreed += $size
                }
                catch {}
            }
        }
        catch {}
    }

    $freedMB = [math]::Round($totalFreed / 1MB, 2)
    Write-Log "Cleaned up ${freedMB}MB of temporary files" "SUCCESS"
}
#endregion

#region Reporting Functions
function New-HealthReport {
    <#
    .SYNOPSIS
      Generates a comprehensive health report.
    #>
    Write-Log "`n========================================" "INFO" "Magenta"
    Write-Log "  VHDX HEALTH OPTIMIZER - REPORT" "INFO" "Magenta"
    Write-Log "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO" "Magenta"
    Write-Log "========================================`n" "INFO" "Magenta"

    # System Information
    Write-Log "--- SYSTEM INFORMATION ---" "INFO" "Cyan"
    $os = Get-CimInstance Win32_OperatingSystem
    $build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
    Write-Log "OS: $($os.Caption) Build $build" "INFO"
    Write-Log "Computer: $env:COMPUTERNAME" "INFO"

    # Disk Space Overview
    Write-Log "`n--- DISK SPACE OVERVIEW ---" "INFO" "Cyan"
    $volumes = Get-Volume | Where-Object { $_.DriveType -eq 'Fixed' -and $_.DriveLetter }
    foreach ($vol in $volumes) {
        $totalGB = [math]::Round($vol.Size / 1GB, 2)
        $freeGB = [math]::Round($vol.SizeRemaining / 1GB, 2)
        $usedPercent = [math]::Round((1 - ($vol.SizeRemaining / $vol.Size)) * 100, 1)

        $color = if ($usedPercent -gt 90) { 'Red' } elseif ($usedPercent -gt 75) { 'Yellow' } else { 'Green' }
        Write-Log "$($vol.DriveLetter): - ${freeGB}GB free of ${totalGB}GB (${usedPercent}% used) [$($vol.FileSystem)]" "INFO" $color
    }

    # WSL Status
    Write-Log "`n--- WSL STATUS ---" "INFO" "Cyan"
    try {
        $wslVersion = (wsl --version 2>&1) -join "`n"
        Write-Log "WSL Version Info:`n$wslVersion" "INFO"

        $wslDistros = wsl --list --verbose 2>&1
        Write-Log "`nWSL Distributions:" "INFO"
        $wslDistros | ForEach-Object {
            if ($_ -and $_ -notmatch '^$') { Write-Log "  $_" "INFO" "Gray" }
        }
    }
    catch {
        Write-Log "Cannot retrieve WSL information: $($_.Exception.Message)" "WARNING"
    }

    # Hyper-V Status
    Write-Log "`n--- HYPER-V STATUS ---" "INFO" "Cyan"
    try {
        $hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -ErrorAction SilentlyContinue
        if ($hyperv) {
            $color = if ($hyperv.State -eq 'Enabled') { 'Green' } else { 'Yellow' }
            Write-Log "Hyper-V State: $($hyperv.State)" "INFO" $color
        }

        $vmcompute = Get-Service -Name vmcompute -ErrorAction SilentlyContinue
        if ($vmcompute) {
            $color = if ($vmcompute.Status -eq 'Running') { 'Green' } else { 'Yellow' }
            Write-Log "vmcompute Service: $($vmcompute.Status)" "INFO" $color
        }
    }
    catch {
        Write-Log "Cannot retrieve Hyper-V information" "WARNING"
    }

    # VHDX Files
    Write-Log "`n--- VHDX FILES ---" "INFO" "Cyan"
    $allVHDX = Get-AllVHDXFiles -IncludeDocker -IncludeWSL

    if ($allVHDX.Count -eq 0) {
        Write-Log "No VHDX files found" "INFO"
    }
    else {
        foreach ($vhdx in $allVHDX) {
            $info = Get-VHDXInfo -Path $vhdx.FullName
            if ($info) {
                Write-Log "`n  File: $($info.Name)" "INFO" "Yellow"
                Write-Log "    Path: $($info.Path)" "INFO"
                Write-Log "    Type: $($info.Type)" "INFO"
                Write-Log "    Size: $($info.SizeGB) GB" "INFO"
                if ($info.VirtualSize) {
                    Write-Log "    Virtual Size: $($info.VirtualSize) GB" "INFO"
                }

                $lockColor = if ($info.IsLocked) { 'Red' } else { 'Green' }
                $lockStatus = if ($info.IsLocked) { 'LOCKED (in use)' } else { 'Unlocked' }
                Write-Log "    Status: $lockStatus" "INFO" $lockColor

                Write-Log "    Sparse: $($info.IsSparse)" "INFO"
                Write-Log "    Last Modified: $($info.LastWriteTime)" "INFO"

                if ($info.FragmentationPercentage) {
                    $fragColor = if ($info.FragmentationPercentage -gt 30) { 'Red' } elseif ($info.FragmentationPercentage -gt 10) { 'Yellow' } else { 'Green' }
                    Write-Log "    Fragmentation: $($info.FragmentationPercentage)%" "INFO" $fragColor
                }
            }
        }
    }

    # Docker Desktop Status
    Write-Log "`n--- DOCKER DESKTOP STATUS ---" "INFO" "Cyan"
    $dockerProcess = Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue
    if ($dockerProcess) {
        Write-Log "Docker Desktop: Running (PID: $($dockerProcess.Id))" "INFO" "Green"
    }
    else {
        Write-Log "Docker Desktop: Not running" "INFO" "Yellow"
    }

    # Recommendations
    Write-Log "`n--- RECOMMENDATIONS ---" "INFO" "Cyan"
    $recommendations = @()

    # Check for large VHDX files
    $largeVHDX = $allVHDX | ForEach-Object { Get-VHDXInfo $_.FullName } |
                 Where-Object { $_.SizeGB -gt $Script:Config.Thresholds.MaxVHDXSizeGB }
    if ($largeVHDX) {
        $recommendations += "Large VHDX files detected (>$($Script:Config.Thresholds.MaxVHDXSizeGB)GB). Consider running compaction."
    }

    # Check for unlocked VHDX that could be compacted
    $compactableVHDX = $allVHDX | ForEach-Object { Get-VHDXInfo $_.FullName } |
                       Where-Object { -not $_.IsLocked }
    if ($compactableVHDX) {
        $recommendations += "$($compactableVHDX.Count) VHDX file(s) can be compacted (stop WSL/Docker first for Docker/WSL files)."
    }

    # Check disk space
    $lowSpaceVolumes = Get-Volume | Where-Object {
        $_.DriveType -eq 'Fixed' -and
        $_.SizeRemaining -and
        ($_.SizeRemaining / $_.Size) -lt 0.15
    }
    if ($lowSpaceVolumes) {
        $recommendations += "Low disk space on: $($lowSpaceVolumes.DriveLetter -join ', ')"
    }

    if ($recommendations.Count -eq 0) {
        Write-Log "No issues detected. System is healthy." "SUCCESS" "Green"
    }
    else {
        foreach ($rec in $recommendations) {
            Write-Log "- $rec" "WARNING" "Yellow"
        }
    }

    Write-Log "`n========================================" "INFO" "Magenta"
    Write-Log "  Report saved to: $($Script:Config.LogPath)" "INFO"
    Write-Log "========================================`n" "INFO" "Magenta"
}
#endregion

#region Scheduled Task Functions
function Register-MaintenanceTask {
    param(
        [string]$Time = '04:00',
        [string]$Day = 'Sunday'
    )

    $taskName = $Script:Config.ScheduledTask.Name
    $scriptPath = $PSCommandPath

    if (-not $scriptPath -or -not (Test-Path $scriptPath)) {
        Write-Log "Cannot determine script path for scheduling. Save the script first." "ERROR"
        return $false
    }

    Write-Log "Configuring scheduled task: $taskName" "INFO"
    Write-Log "  Schedule: Every $Day at $Time" "INFO"

    try {
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument "-NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Action Optimize -IncludeWSL -IncludeDocker -Force"

        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At ([datetime]::ParseExact($Time, 'HH:mm', $null))

        $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

        $settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -RunOnlyIfNetworkAvailable:$false `
            -WakeToRun:$false `
            -ExecutionTimeLimit (New-TimeSpan -Hours 2)

        $task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
            -Description $Script:Config.ScheduledTask.Description

        Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null

        Write-Log "Scheduled task registered successfully" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "Failed to register scheduled task: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Unregister-MaintenanceTask {
    $taskName = $Script:Config.ScheduledTask.Name

    try {
        $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-Log "Scheduled task '$taskName' removed" "SUCCESS"
        }
        else {
            Write-Log "Scheduled task '$taskName' not found" "INFO"
        }
        return $true
    }
    catch {
        Write-Log "Failed to remove scheduled task: $($_.Exception.Message)" "ERROR"
        return $false
    }
}
#endregion

#region Main Execution
function Invoke-FullOptimization {
    Write-Log "Starting full VHDX and disk optimization..." "INFO"

    # Phase 1: Pre-flight checks
    Write-Log "`n[PHASE 1] Pre-flight checks" "INFO" "Cyan"

    $requireWSLDown = $IncludeWSL
    $requireDockerDown = $IncludeDocker

    if (-not (Test-SafeToOperate -RequireWSLDown:$requireWSLDown -RequireDockerDown:$requireDockerDown)) {
        if (-not $Force) {
            Write-Log "Pre-flight checks failed. Use -Force to override." "ERROR"
            return
        }
        Write-Log "Force mode enabled, proceeding despite warnings..." "WARNING"
    }

    # Phase 2: Stop services if needed
    Write-Log "`n[PHASE 2] Preparing environment" "INFO" "Cyan"

    if ($IncludeWSL) {
        if (-not (Stop-WSLGracefully)) {
            if (-not $Force) {
                Write-Log "Could not stop WSL gracefully. Aborting." "ERROR"
                return
            }
        }
    }

    # Phase 3: Cleanup
    Write-Log "`n[PHASE 3] Cleanup" "INFO" "Cyan"
    Clear-TempFiles

    # Phase 4: VHDX Optimization
    Write-Log "`n[PHASE 4] VHDX Optimization" "INFO" "Cyan"
    $allVHDX = Get-AllVHDXFiles -IncludeDocker:$IncludeDocker -IncludeWSL:$IncludeWSL

    foreach ($vhdx in $allVHDX) {
        $info = Get-VHDXInfo -Path $vhdx.FullName
        if ($info -and -not $info.IsLocked) {
            Optimize-VHDXFile -Path $vhdx.FullName -Force:$Force
        }
        elseif ($info -and $info.IsLocked) {
            Write-Log "Skipping locked VHDX: $($vhdx.FullName)" "WARNING"
        }
    }

    # Phase 5: Physical Disk Optimization
    Write-Log "`n[PHASE 5] Physical Disk Optimization" "INFO" "Cyan"
    Invoke-PhysicalDiskOptimization

    # Phase 6: Summary
    Write-Log "`n[PHASE 6] Summary" "INFO" "Cyan"
    Write-Log "Optimization complete!" "SUCCESS"
    Write-Log "Log file: $($Script:Config.LogPath)" "INFO"
}

# Main entry point
Initialize-Logging

Write-Log "VHDX Health Optimizer v$($Script:Config.Version)" "INFO" "Cyan"
Write-Log "Action: $Action" "INFO"

switch ($Action) {
    'Optimize' {
        Invoke-FullOptimization
    }

    'Check' {
        New-HealthReport
    }

    'Compact' {
        if (-not (Test-SafeToOperate -RequireWSLDown:$IncludeWSL -RequireDockerDown:$IncludeDocker)) {
            if (-not $Force) {
                Write-Log "Safety checks failed. Stop WSL/Docker first or use -Force." "ERROR"
                return
            }
        }

        $allVHDX = Get-AllVHDXFiles -IncludeDocker:$IncludeDocker -IncludeWSL:$IncludeWSL
        foreach ($vhdx in $allVHDX) {
            Optimize-VHDXFile -Path $vhdx.FullName -Force:$Force
        }
    }

    'Cleanup' {
        Clear-TempFiles
    }

    'Schedule' {
        Register-MaintenanceTask -Time $ScheduleTime -Day $ScheduleDay
    }

    'Unschedule' {
        Unregister-MaintenanceTask
    }

    'Report' {
        New-HealthReport
    }
}

Write-Log "`nDone." "INFO" "Cyan"
#endregion
