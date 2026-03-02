#Requires -Version 5.1
<#
.SYNOPSIS
    Optimizes VSock and TCP settings for WSL2 performance (Requires Administrator)

.DESCRIPTION
    Applies performance optimizations to VSock and TCP stack settings to improve
    WSL2 networking performance. Includes registry modifications with automatic
    backup and support for -WhatIf preview.

    Optimizations include:
    - TCP auto-tuning level adjustments
    - VSock buffer size optimization
    - RSS (Receive Side Scaling) settings
    - TCP timestamps and window scaling
    - Memory pressure thresholds

.PARAMETER Profile
    Optimization profile to apply
    - Balanced: Moderate optimizations suitable for most workloads
    - Performance: Aggressive optimizations for maximum throughput
    - Conservative: Minimal changes, prioritizes stability

.PARAMETER BackupPath
    Path to store registry backup (default: PC_AI Config directory)

.PARAMETER RestoreBackup
    Restore settings from backup file

.PARAMETER SkipWSLRestart
    Do not restart WSL after applying optimizations

.EXAMPLE
    Optimize-VSock
    Apply balanced optimizations

.EXAMPLE
    Optimize-VSock -Profile Performance
    Apply aggressive performance optimizations

.EXAMPLE
    Optimize-VSock -WhatIf
    Preview changes without applying them

.EXAMPLE
    Optimize-VSock -RestoreBackup
    Restore previous settings from backup

.OUTPUTS
    PSCustomObject with optimization results
#>
function Optimize-VSock {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('Balanced', 'Performance', 'Conservative')]
        [string]$Profile = 'Balanced',

        [Parameter()]
        [string]$BackupPath,

        [Parameter()]
        [switch]$RestoreBackup,

        [Parameter()]
        [switch]$SkipWSLRestart
    )

    # Check for Administrator privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error "This function requires Administrator privileges. Please run PowerShell as Administrator."
        return
    }

    # Set default backup path
    if (-not $BackupPath) {
        $configPath = Join-Path (Split-Path $script:ModuleRoot -Parent | Split-Path -Parent) 'Config'
        if (-not (Test-Path $configPath)) {
            New-Item -Path $configPath -ItemType Directory -Force | Out-Null
        }
        $BackupPath = Join-Path $configPath 'vsock-backup.json'
    }

    $result = [PSCustomObject]@{
        Timestamp       = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Profile         = $Profile
        ChangesApplied  = @()
        ChangesPending  = @()
        Errors          = @()
        WSLRestarted    = $false
        BackupCreated   = $false
    }

    # Handle restore
    if ($RestoreBackup) {
        if (-not (Test-Path $BackupPath)) {
            Write-Error "Backup file not found: $BackupPath"
            return
        }

        Write-Host "[*] Restoring VSock settings from backup..." -ForegroundColor Cyan

        try {
            $backupData = Get-Content -Path $BackupPath -Raw | ConvertFrom-Json

            foreach ($entry in $backupData) {
                if ($PSCmdlet.ShouldProcess("$($entry.Path)\$($entry.Name)", "Restore to $($entry.Value)")) {
                    Set-ItemProperty -Path $entry.Path -Name $entry.Name -Value $entry.Value -Force
                    Write-Host "  [+] Restored: $($entry.Name)" -ForegroundColor Green
                }
            }

            Write-Host "[+] Settings restored from backup" -ForegroundColor Green
            return
        }
        catch {
            Write-Error "Failed to restore backup: $_"
            return
        }
    }

    Write-Host "[*] Starting VSock optimization (Profile: $Profile)..." -ForegroundColor Cyan

    # Define optimization settings per profile
    $optimizations = @{
        # TCP Auto-tuning
        'TCP_AutoTuning' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
            Name = 'EnableAutoTuning'
            Balanced = 1
            Performance = 1
            Conservative = 0
            Type = 'DWord'
            Description = 'TCP auto-tuning for dynamic buffer sizing'
        }

        # TCP Window Scaling
        'TCP_WindowScaling' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
            Name = 'Tcp1323Opts'
            Balanced = 3
            Performance = 3
            Conservative = 1
            Type = 'DWord'
            Description = 'TCP RFC 1323 options (timestamps + window scaling)'
        }

        # Default TTL
        'TCP_DefaultTTL' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
            Name = 'DefaultTTL'
            Balanced = 128
            Performance = 128
            Conservative = 64
            Type = 'DWord'
            Description = 'Default TTL for outgoing packets'
        }

        # TCP Chimney Offload (deprecated but still affects some systems)
        'TCP_ChimneyOffload' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
            Name = 'EnableTCPChimney'
            Balanced = 0
            Performance = 0
            Conservative = 0
            Type = 'DWord'
            Description = 'TCP Chimney offload (disabled for compatibility)'
        }

        # RSS Processor Affinity
        'RSS_BaseProcNumber' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\NDIS\Parameters'
            Name = 'RssBaseCpu'
            Balanced = 1
            Performance = 0
            Conservative = 2
            Type = 'DWord'
            Description = 'RSS base processor (spread load across cores)'
        }

        # Network Throttling Index
        'Net_ThrottlingIndex' = @{
            Path = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile'
            Name = 'NetworkThrottlingIndex'
            Balanced = 10
            Performance = -1  # 0xFFFFFFFF (disabled)
            Conservative = 10
            Type = 'DWord'
            Description = 'Network throttling index for multimedia'
        }

        # System Responsiveness
        'System_Responsiveness' = @{
            Path = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile'
            Name = 'SystemResponsiveness'
            Balanced = 10
            Performance = 0
            Conservative = 20
            Type = 'DWord'
            Description = 'System responsiveness priority'
        }

        # TCP Max Data Retransmissions
        'TCP_MaxDataRetransmissions' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
            Name = 'TcpMaxDataRetransmissions'
            Balanced = 5
            Performance = 3
            Conservative = 5
            Type = 'DWord'
            Description = 'Max TCP data retransmission attempts'
        }

        # Memory Low/Medium/High thresholds (affects network buffer allocation)
        'Mem_LowThreshold' = @{
            Path = 'HKLM:\SYSTEM\CurrentControlSet\Services\LanmanWorkstation\Parameters'
            Name = 'MaxCmds'
            Balanced = 50
            Performance = 100
            Conservative = 30
            Type = 'DWord'
            Description = 'SMB max commands (affects network buffering)'
        }
    }

    # Create backup before making changes
    Write-Host "[*] Creating backup of current settings..." -ForegroundColor Yellow
    $backupEntries = @()

    foreach ($key in $optimizations.Keys) {
        $opt = $optimizations[$key]

        if (Test-Path $opt.Path) {
            $currentValue = Get-RegistryValueSafe -Path $opt.Path -Name $opt.Name
            if ($null -ne $currentValue) {
                $backupEntries += @{
                    Path = $opt.Path
                    Name = $opt.Name
                    Value = $currentValue
                    Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                }
            }
        }
    }

    if ($backupEntries.Count -gt 0) {
        try {
            $backupEntries | ConvertTo-Json -Depth 3 | Out-File -FilePath $BackupPath -Force
            $result.BackupCreated = $true
            Write-Host "  [+] Backup saved to: $BackupPath" -ForegroundColor Green
        }
        catch {
            Write-Warning "Failed to create backup: $_"
        }
    }

    # Apply optimizations
    Write-Host "[*] Applying optimizations..." -ForegroundColor Yellow

    foreach ($key in $optimizations.Keys) {
        $opt = $optimizations[$key]
        $targetValue = $opt.$Profile

        # Handle special case for -1 (0xFFFFFFFF)
        if ($targetValue -eq -1) {
            $targetValue = [uint32]::MaxValue
        }

        $currentValue = Get-RegistryValueSafe -Path $opt.Path -Name $opt.Name

        # Check if change is needed
        if ($currentValue -eq $targetValue) {
            Write-Host "  [=] $($opt.Description): Already optimal" -ForegroundColor Gray
            continue
        }

        $changeInfo = [PSCustomObject]@{
            Setting     = $key
            Description = $opt.Description
            OldValue    = $currentValue
            NewValue    = $targetValue
            Path        = "$($opt.Path)\$($opt.Name)"
        }

        if ($PSCmdlet.ShouldProcess("$($opt.Path)\$($opt.Name)", "Set to $targetValue ($($opt.Description))")) {
            try {
                $success = Set-RegistryValueSafe -Path $opt.Path -Name $opt.Name -Value $targetValue -PropertyType $opt.Type -WhatIf:$false

                if ($success) {
                    $result.ChangesApplied += $changeInfo
                    Write-Host "  [+] $($opt.Description): $currentValue -> $targetValue" -ForegroundColor Green
                }
                else {
                    $result.Errors += "Failed to set $key"
                    Write-Host "  [!] Failed: $($opt.Description)" -ForegroundColor Red
                }
            }
            catch {
                $result.Errors += "Error setting $key`: $_"
                Write-Host "  [!] Error: $($opt.Description) - $_" -ForegroundColor Red
            }
        }
        else {
            $result.ChangesPending += $changeInfo
            Write-Host "  [*] Would set $($opt.Description): $currentValue -> $targetValue" -ForegroundColor Cyan
        }
    }

    # Configure netsh settings
    Write-Host "[*] Configuring network shell settings..." -ForegroundColor Yellow

    $netshCommands = @{
        Balanced = @(
            @{ Cmd = 'netsh int tcp set global autotuninglevel=normal'; Desc = 'TCP auto-tuning: normal' },
            @{ Cmd = 'netsh int tcp set global chimney=disabled'; Desc = 'TCP Chimney: disabled' },
            @{ Cmd = 'netsh int tcp set global rss=enabled'; Desc = 'RSS: enabled' },
            @{ Cmd = 'netsh int tcp set global timestamps=enabled'; Desc = 'TCP timestamps: enabled' }
        )
        Performance = @(
            @{ Cmd = 'netsh int tcp set global autotuninglevel=experimental'; Desc = 'TCP auto-tuning: experimental' },
            @{ Cmd = 'netsh int tcp set global chimney=disabled'; Desc = 'TCP Chimney: disabled' },
            @{ Cmd = 'netsh int tcp set global rss=enabled'; Desc = 'RSS: enabled' },
            @{ Cmd = 'netsh int tcp set global timestamps=enabled'; Desc = 'TCP timestamps: enabled' },
            @{ Cmd = 'netsh int tcp set global ecncapability=enabled'; Desc = 'ECN: enabled' }
        )
        Conservative = @(
            @{ Cmd = 'netsh int tcp set global autotuninglevel=disabled'; Desc = 'TCP auto-tuning: disabled' },
            @{ Cmd = 'netsh int tcp set global chimney=disabled'; Desc = 'TCP Chimney: disabled' },
            @{ Cmd = 'netsh int tcp set global rss=enabled'; Desc = 'RSS: enabled' }
        )
    }

    foreach ($cmd in $netshCommands[$Profile]) {
        if ($PSCmdlet.ShouldProcess($cmd.Desc, 'Apply netsh setting')) {
            try {
                $output = Invoke-Expression $cmd.Cmd 2>&1
                if ($LASTEXITCODE -eq 0 -or $output -notmatch 'error|failed') {
                    Write-Host "  [+] $($cmd.Desc)" -ForegroundColor Green
                }
                else {
                    Write-Host "  [!] $($cmd.Desc): May require reboot" -ForegroundColor Yellow
                }
            }
            catch {
                Write-Host "  [!] $($cmd.Desc): $_" -ForegroundColor Yellow
            }
        }
    }

    # Restart WSL if requested
    if (-not $SkipWSLRestart -and $result.ChangesApplied.Count -gt 0) {
        if ($PSCmdlet.ShouldProcess('WSL', 'Restart to apply changes')) {
            Write-Host "[*] Restarting WSL to apply changes..." -ForegroundColor Yellow
            try {
                wsl --shutdown
                Start-Sleep -Seconds 3

                # Quick test
                $testResult = wsl -d Ubuntu -e echo "VSock test" 2>&1
                if ($testResult -match "VSock test") {
                    $result.WSLRestarted = $true
                    Write-Host "  [+] WSL restarted successfully" -ForegroundColor Green
                }
                else {
                    Write-Host "  [!] WSL restart may need manual verification" -ForegroundColor Yellow
                }
            }
            catch {
                Write-Host "  [!] WSL restart failed: $_" -ForegroundColor Red
                $result.Errors += "WSL restart: $_"
            }
        }
    }

    # Summary
    Write-Host ""
    Write-Host "== VSock Optimization Summary ==" -ForegroundColor Cyan
    Write-Host "  Profile: $Profile" -ForegroundColor White
    Write-Host "  Changes Applied: $($result.ChangesApplied.Count)" -ForegroundColor White

    if ($WhatIfPreference) {
        Write-Host "  Changes Pending (WhatIf): $($result.ChangesPending.Count)" -ForegroundColor Cyan
    }

    if ($result.Errors.Count -gt 0) {
        Write-Host "  Errors: $($result.Errors.Count)" -ForegroundColor Red
    }

    if ($result.BackupCreated) {
        Write-Host "  Backup: $BackupPath" -ForegroundColor Green
    }

    if ($result.ChangesApplied.Count -gt 0 -or $WhatIfPreference) {
        Write-Host ""
        Write-Host "[*] Note: Some changes may require a system reboot to take full effect" -ForegroundColor Yellow
    }

    return $result
}
