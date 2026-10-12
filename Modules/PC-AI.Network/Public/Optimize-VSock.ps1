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

    $PSNativeCommandUseErrorActionPreference = $false

    # Check for Administrator privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error "This function requires Administrator privileges. Please run PowerShell as Administrator."
        return
    }

    # Set default backup path
    $configPath = $null
    if (-not $BackupPath) {
        $configPath = Join-Path (Split-Path $script:ModuleRoot -Parent | Split-Path -Parent) 'Config'
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
            $result.Errors += "Backup file not found: $BackupPath"
            return $result
        }

        Write-Host "[*] Restoring VSock settings from backup..." -ForegroundColor Cyan

        try {
            $jsonParameters = @{}
            $jsonCommand = Get-Command Microsoft.PowerShell.Utility\ConvertFrom-Json -ErrorAction Stop
            if ($jsonCommand.Parameters.ContainsKey('DateKind')) { $jsonParameters.DateKind = 'String' }
            $backupData = Get-Content -Path $BackupPath -Raw | Microsoft.PowerShell.Utility\ConvertFrom-Json @jsonParameters
            if ($null -eq $backupData -or @($backupData).Count -eq 0) { throw 'Registry backup contains no entries.' }

            # Validate and prepare the entire retained ledger before any mutation or approval.
            $restoreEntries = @()
            foreach ($savedEntry in $backupData) {
                foreach ($required in @('Path', 'Name', 'Value')) {
                    if (-not $savedEntry.PSObject.Properties[$required]) { throw "Registry backup entry lacks $required." }
                }
                if ($savedEntry.Path -isnot [string] -or [string]::IsNullOrWhiteSpace($savedEntry.Path) -or
                    $savedEntry.Name -isnot [string] -or [string]::IsNullOrWhiteSpace($savedEntry.Name)) {
                    throw 'Registry backup entry requires a nonempty path and property name.'
                }
                $hasPresence = $null -ne $savedEntry.PSObject.Properties['Present']
                $value = $savedEntry.Value
                if ($hasPresence) {
                    if ($savedEntry.Present -isnot [bool] -or -not $savedEntry.PSObject.Properties['Kind']) { throw 'Invalid registry presence or kind metadata.' }
                    if (-not $savedEntry.Present) {
                        if ($null -ne $savedEntry.Kind -or $null -ne $value) { throw 'Absent registry property has unexpected value metadata.' }
                    } else {
                        if ($savedEntry.Kind -isnot [string]) { throw 'Invalid registry value kind metadata.' }
                        $value = switch ($savedEntry.Kind) {
                            'DWord' {
                                if ($value -isnot [int] -and $value -isnot [long]) { throw 'DWord backup requires an integer.' }
                                [int]$value
                            }
                            'QWord' {
                                if ($value -isnot [int] -and $value -isnot [long]) { throw 'QWord backup requires an integer.' }
                                [long]$value
                            }
                            'String' {
                                if ($value -isnot [string]) { throw 'String backup requires literal text.' }
                                [string]$value
                            }
                            'ExpandString' {
                                if ($value -isnot [string]) { throw 'ExpandString backup requires literal text.' }
                                [string]$value
                            }
                            'MultiString' {
                                if ($value -isnot [array] -or @($value | Where-Object { $_ -isnot [string] }).Count) { throw 'MultiString backup requires an array of literal strings.' }
                                ,([string[]]$value)
                            }
                            'Binary' {
                                if ($value -isnot [array] -or @($value | Where-Object { ($_ -isnot [int] -and $_ -isnot [long]) -or $_ -lt 0 -or $_ -gt 255 }).Count) { throw 'Binary backup requires an array of byte values.' }
                                ,([byte[]]$value)
                            }
                            default { throw 'Invalid registry value kind metadata.' }
                        }
                    }
                } elseif (@($value | Where-Object { $_ -is [datetime] }).Count) {
                    throw 'This JSON parser cannot preserve date-shaped registry text literally.'
                }
                $restoreEntries += [PSCustomObject]@{
                    Path = $savedEntry.Path; Name = $savedEntry.Name; Value = $value
                    HasPresence = $hasPresence
                    Present = if ($hasPresence) { $savedEntry.Present } else { $null }
                    Kind = if ($hasPresence) { $savedEntry.Kind } else { $null }
                }
            }

            foreach ($entry in $restoreEntries) {
                $changeInfo = [PSCustomObject]@{ Setting = $entry.Name; Path = "$($entry.Path)\$($entry.Name)"; NewValue = $entry.Value }
                $action = if ($entry.HasPresence -and -not $entry.Present) { 'Remove originally absent property' } else { "Restore to $($entry.Value)" }
                if ($PSCmdlet.ShouldProcess($changeInfo.Path, $action)) {
                    if ($entry.HasPresence) {
                        if (-not $entry.Present) {
                            $key = Get-Item -LiteralPath $entry.Path -ErrorAction Stop
                            try {
                                if ($key.GetValueNames() -contains $entry.Name) {
                                    Remove-ItemProperty -LiteralPath $entry.Path -Name $entry.Name -ErrorAction Stop
                                }
                            } finally { $key.Dispose() }
                        } else {
                            Set-ItemProperty -LiteralPath $entry.Path -Name $entry.Name -Value $entry.Value -Type $entry.Kind -Force -ErrorAction Stop
                        }
                    } else {
                        # Retained predecessor backups have no presence or type metadata.
                        Set-ItemProperty -Path $entry.Path -Name $entry.Name -Value $entry.Value -Force -ErrorAction Stop
                    }
                    $result.ChangesApplied += $changeInfo
                    Write-Host "  [+] Restored: $($entry.Name)" -ForegroundColor Green
                } else {
                    $result.ChangesPending += $changeInfo
                }
            }

            if ($result.ChangesPending.Count -eq 0) { Write-Host "[+] Settings restored from backup" -ForegroundColor Green }
            return $result
        }
        catch {
            $result.Errors += "Failed to restore backup: $_"
            return $result
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

    try {
        foreach ($setting in $optimizations.Keys) {
            $opt = $optimizations[$setting]
            $key = Get-Item -LiteralPath $opt.Path -ErrorAction Stop
            try {
                $present = $key.GetValueNames() -contains $opt.Name
                $value = $null
                $kind = $null
                if ($present) {
                    $value = $key.GetValue($opt.Name, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
                    if ($null -eq $value) { throw "Failed to capture $($opt.Name)." }
                    $kind = $key.GetValueKind($opt.Name).ToString()
                    if ($kind -notin @('DWord','QWord','String','ExpandString','MultiString','Binary')) { throw "Unsupported original registry kind $kind." }
                }
                $backupEntries += @{
                    Path = $opt.Path; Name = $opt.Name; Present = $present; Value = $value; Kind = $kind
                    Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                }
            } finally { $key.Dispose() }
        }
    } catch {
        $result.Errors += "Registry backup capture: $_"
        return $result
    }

    if ($configPath -and -not (Test-Path -LiteralPath $configPath)) {
        if ($PSCmdlet.ShouldProcess($configPath, 'Create backup directory')) {
            try { New-Item -Path $configPath -ItemType Directory -Force -ErrorAction Stop | Out-Null }
            catch { $result.Errors += "Backup directory: $_"; return $result }
        } elseif (-not $WhatIfPreference) {
            $result.Errors += 'Backup directory creation was declined; no optimization applied.'
            return $result
        }
    }

    if ($PSCmdlet.ShouldProcess($BackupPath, 'Create settings backup')) {
        try {
            $stream = [IO.File]::Open($BackupPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
                try { $writer.Write(($backupEntries | ConvertTo-Json -Depth 3)); $writer.Flush() }
                finally { $writer.Dispose() }
            } finally { $stream.Dispose() }
            $result.BackupCreated = $true
            Write-Host "  [+] Backup saved to: $BackupPath" -ForegroundColor Green
        }
        catch {
            $result.Errors += "Backup: $_"
            Write-Warning "Failed to create backup: $_"
            return $result
        }
    } elseif (-not $WhatIfPreference) {
        $result.Errors += 'Settings backup was declined; no optimization applied.'
        return $result
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
            @{ Args = @('int', 'tcp', 'set', 'global', 'autotuninglevel=normal'); Desc = 'TCP auto-tuning: normal' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'chimney=disabled'); Desc = 'TCP Chimney: disabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'rss=enabled'); Desc = 'RSS: enabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'timestamps=enabled'); Desc = 'TCP timestamps: enabled' }
        )
        Performance = @(
            @{ Args = @('int', 'tcp', 'set', 'global', 'autotuninglevel=experimental'); Desc = 'TCP auto-tuning: experimental' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'chimney=disabled'); Desc = 'TCP Chimney: disabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'rss=enabled'); Desc = 'RSS: enabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'timestamps=enabled'); Desc = 'TCP timestamps: enabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'ecncapability=enabled'); Desc = 'ECN: enabled' }
        )
        Conservative = @(
            @{ Args = @('int', 'tcp', 'set', 'global', 'autotuninglevel=disabled'); Desc = 'TCP auto-tuning: disabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'chimney=disabled'); Desc = 'TCP Chimney: disabled' },
            @{ Args = @('int', 'tcp', 'set', 'global', 'rss=enabled'); Desc = 'RSS: enabled' }
        )
    }

    foreach ($cmd in $netshCommands[$Profile]) {
        if ($PSCmdlet.ShouldProcess($cmd.Desc, 'Apply netsh setting')) {
            try {
                $null = & netsh @($cmd.Args) 2>&1
                $netshExit = $LASTEXITCODE
                if ($netshExit -eq 0) {
                    Write-Host "  [+] $($cmd.Desc)" -ForegroundColor Green
                }
                else {
                    $result.Errors += "netsh $($cmd.Desc) failed with exit code $netshExit."
                    Write-Host "  [!] $($cmd.Desc): May require reboot" -ForegroundColor Yellow
                }
            }
            catch {
                $result.Errors += "netsh $($cmd.Desc): $_"
                Write-Host "  [!] $($cmd.Desc): $_" -ForegroundColor Yellow
            }
        }
    }

    # Restart WSL if requested
    if (-not $SkipWSLRestart -and $result.ChangesApplied.Count -gt 0) {
        if ($PSCmdlet.ShouldProcess('WSL', 'Restart to apply changes')) {
            Write-Host "[*] Restarting WSL to apply changes..." -ForegroundColor Yellow
            try {
                $null = wsl --shutdown 2>&1
                $shutdownExit = $LASTEXITCODE
                if ($shutdownExit -ne 0) { throw "WSL shutdown failed with exit code $shutdownExit." }
                Start-Sleep -Seconds 3

                # Quick test
                $testResult = wsl -d Ubuntu -e echo "VSock test" 2>&1
                $testExit = $LASTEXITCODE
                if ($testExit -ne 0) { throw "WSL restart verification failed with exit code $testExit." }
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
