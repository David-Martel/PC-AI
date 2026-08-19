#Requires -RunAsAdministrator
<#
.SYNOPSIS
    WinRM Server Setup Script with GPO detection and comprehensive diagnostics

.DESCRIPTION
    Configures WinRM for remote PowerShell access. Detects GPO-controlled settings
    and provides remediation steps when direct configuration isn't possible.

.PARAMETER ClientIPs
    Array of client IP addresses to allow. Defaults to common DTM network IPs.

.PARAMETER Force
    Skip confirmation prompts

.PARAMETER DiagnoseOnly
    Only run diagnostics without making changes

.EXAMPLE
    .\setup-server-winrm.ps1
    .\setup-server-winrm.ps1 -ClientIPs @("192.168.1.100", "10.0.0.50")
    .\setup-server-winrm.ps1 -DiagnoseOnly
#>

param(
    [string[]]$ClientIPs = @("10.10.15.150", "10.10.20.199", "10.10.15.0/24", "10.10.20.0/24"),
    [switch]$Force,
    [switch]$DiagnoseOnly
)

$ErrorActionPreference = 'Continue'

# ============================================================================
# CONFIGURATION
# ============================================================================
$script:Config = @{
    WinRMPorts = @(5985, 5986)
    Hostname = $env:COMPUTERNAME
    LocalIPs = @()
    GPOIssues = @()
    FixesNeeded = @()
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
function Write-Status {
    param([string]$Message, [string]$Type = 'Info')
    $colors = @{ Success='Green'; Warning='Yellow'; Error='Red'; Info='Cyan'; Header='Magenta' }
    $symbols = @{ Success='[+]'; Warning='[!]'; Error='[-]'; Info='[*]'; Header='[=]' }
    Write-Host "$($symbols[$Type]) $Message" -ForegroundColor $colors[$Type]
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n$('=' * 60)" -ForegroundColor Magenta
    Write-Status $Title -Type Header
    Write-Host "$('=' * 60)" -ForegroundColor Magenta
}

function Test-IsGPOControlled {
    param([string]$SettingOutput)
    return $SettingOutput -match '\[Source="GPO"\]'
}

function Get-LocalIPAddresses {
    $ips = @()
    try {
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
            ForEach-Object { $ips += $_.IPAddress }
    } catch {
        # Fallback for module issues
        $ipconfig = ipconfig | Select-String 'IPv4 Address'
        $ipconfig | ForEach-Object {
            if ($_ -match '(\d+\.\d+\.\d+\.\d+)') {
                $ips += $Matches[1]
            }
        }
    }
    return $ips
}

function Test-IPInRange {
    param([string]$IP, [string]$Range)

    if ($Range -eq '*') { return $true }

    # Handle CIDR notation (e.g., 10.10.15.0/24)
    if ($Range -match '/') {
        # Simplified - just check if same subnet
        $rangeBase = ($Range -split '/')[0] -replace '\.\d+$', ''
        $ipBase = $IP -replace '\.\d+$', ''
        return $rangeBase -eq $ipBase
    }

    # Handle range notation (e.g., 10.10.15.1-10.10.15.100)
    if ($Range -match '^(\d+\.\d+\.\d+)\.(\d+)-\d+\.\d+\.\d+\.(\d+)$') {
        $base = $Matches[1]
        $start = [int]$Matches[2]
        $end = [int]$Matches[3]

        if ($IP -match "^$([regex]::Escape($base))\.(\d+)$") {
            $ipLast = [int]$Matches[1]
            return ($ipLast -ge $start -and $ipLast -le $end)
        }
    }

    # Handle wildcard (e.g., 10.10.15.*)
    if ($Range -match '\*') {
        $pattern = '^' + ($Range -replace '\*', '\d+' -replace '\.', '\.') + '$'
        return $IP -match $pattern
    }

    # Direct match
    return $IP -eq $Range
}

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================
function Get-WinRMConfiguration {
    Write-Section "WinRM Configuration Analysis"

    $config = @{
        ServiceRunning = $false
        ListenerConfigured = $false
        IPv4Filter = $null
        IPv4FilterGPO = $false
        TrustedHosts = $null
        BlockedClients = @()
    }

    # Check service
    $svc = Get-Service WinRM -ErrorAction SilentlyContinue
    $config.ServiceRunning = $svc.Status -eq 'Running'
    Write-Status "WinRM Service: $($svc.Status)" -Type $(if($config.ServiceRunning){'Success'}else{'Error'})

    # Get WinRM config
    try {
        $winrmOutput = winrm get winrm/config/service 2>&1 | Out-String

        # Check IPv4Filter
        if ($winrmOutput -match 'IPv4Filter\s*=\s*([^\r\n\[]+)') {
            $config.IPv4Filter = $Matches[1].Trim()
        }
        $config.IPv4FilterGPO = Test-IsGPOControlled $winrmOutput

        Write-Status "IPv4Filter: $($config.IPv4Filter)" -Type Info
        if ($config.IPv4FilterGPO) {
            Write-Status "IPv4Filter is GPO-controlled - cannot change via script" -Type Warning
            $script:Config.GPOIssues += "IPv4Filter"
        }

        # Check which client IPs would be blocked
        if ($config.IPv4Filter -and $config.IPv4Filter -ne '*') {
            $filterRanges = $config.IPv4Filter -split ',\s*'
            foreach ($clientIP in $ClientIPs) {
                # Skip CIDR notation for this check
                if ($clientIP -match '/') { continue }

                $allowed = $false
                foreach ($range in $filterRanges) {
                    if (Test-IPInRange -IP $clientIP -Range $range) {
                        $allowed = $true
                        break
                    }
                }
                if (-not $allowed) {
                    $config.BlockedClients += $clientIP
                    Write-Status "Client $clientIP is BLOCKED by IPv4Filter!" -Type Error
                }
            }
        }

        # Check listeners
        $listeners = winrm enumerate winrm/config/listener 2>&1 | Out-String
        $config.ListenerConfigured = $listeners -match 'Port\s*=\s*5985'
        Write-Status "HTTP Listener (5985): $(if($config.ListenerConfigured){'Configured'}else{'Missing'})" -Type $(if($config.ListenerConfigured){'Success'}else{'Warning'})

    } catch {
        Write-Status "Failed to get WinRM config: $_" -Type Error
    }

    # Check TrustedHosts on this machine (for outbound connections)
    try {
        $trustedHosts = (Get-Item WSMan:\localhost\Client\TrustedHosts -ErrorAction SilentlyContinue).Value
        $config.TrustedHosts = $trustedHosts
        Write-Status "TrustedHosts: $(if($trustedHosts){$trustedHosts}else{'(empty)'})" -Type Info
    } catch {}

    return $config
}

function Test-PortAccessibility {
    param([int]$Port)

    try {
        $listener = netstat -ano | Select-String ":$Port\s+.*LISTENING"
        return $null -ne $listener
    } catch {
        return $false
    }
}

function Get-GPORemediationSteps {
    param([string[]]$GPOIssues, [string[]]$BlockedClients)

    Write-Section "GPO Remediation Required"

    Write-Status "The following settings are controlled by Group Policy:" -Type Warning
    foreach ($issue in $GPOIssues) {
        Write-Host "  - $issue" -ForegroundColor Yellow
    }

    if ($BlockedClients.Count -gt 0) {
        Write-Host ""
        Write-Status "These client IPs are BLOCKED by the current IPv4Filter:" -Type Error
        foreach ($ip in $BlockedClients) {
            Write-Host "  - $ip" -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Status "To fix, run gpedit.msc on this machine and:" -Type Info
    Write-Host @"

  1. Navigate to:
     Computer Configuration > Administrative Templates >
     Windows Components > Windows Remote Management (WinRM) > WinRM Service

  2. Open "Allow remote server management through WinRM"

  3. Change the IPv4 filter to one of:
     - *                           (allow all - simplest)
     - 10.10.15.*, 10.10.20.*      (allow both subnets)
     - 10.10.15.0-10.10.15.255, 10.10.20.0-10.10.20.255

  4. Click OK, then run: gpupdate /force

  5. Verify with: winrm get winrm/config/service | findstr IPv4

"@ -ForegroundColor White

    # Also create a registry fix script
    $regFixPath = "$env:TEMP\fix-winrm-ipfilter.reg"
    @"
Windows Registry Editor Version 5.00

; This sets IPv4Filter to allow all IPs
; Run as Administrator, then: gpupdate /force

[HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\WinRM\Service]
"IPv4Filter"="*"
"@ | Set-Content $regFixPath -Encoding ASCII

    Write-Status "Registry fix exported to: $regFixPath" -Type Info
    Write-Host "  Run as admin: reg import `"$regFixPath`" && gpupdate /force" -ForegroundColor Cyan
}

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================
function Enable-WinRMService {
    Write-Section "Enabling WinRM Service"

    if ($DiagnoseOnly) {
        Write-Status "DiagnoseOnly mode - skipping changes" -Type Info
        return
    }

    try {
        # Enable PS Remoting (handles most setup)
        $result = Enable-PSRemoting -Force -SkipNetworkProfileCheck -ErrorAction Stop 2>&1
        Write-Status "PowerShell Remoting enabled" -Type Success
    } catch {
        if ($_ -match 'already') {
            Write-Status "PowerShell Remoting already enabled" -Type Info
        } else {
            Write-Status "Enable-PSRemoting warning: $_" -Type Warning
        }
    }

    # Ensure service is running and set to auto-start
    try {
        Set-Service WinRM -StartupType Automatic -ErrorAction SilentlyContinue
        Start-Service WinRM -ErrorAction SilentlyContinue
        Write-Status "WinRM service configured and started" -Type Success
    } catch {
        Write-Status "Service configuration: $_" -Type Warning
    }
}

function Set-TrustedHosts {
    Write-Section "Configuring Trusted Hosts"

    if ($DiagnoseOnly) {
        Write-Status "DiagnoseOnly mode - skipping changes" -Type Info
        return
    }

    $hostsToAdd = $ClientIPs | Where-Object { $_ -notmatch '/' }  # Exclude CIDR
    $hostString = $hostsToAdd -join ','

    try {
        Set-Item WSMan:\localhost\Client\TrustedHosts -Value $hostString -Force -ErrorAction Stop
        Write-Status "TrustedHosts set to: $hostString" -Type Success
    } catch {
        Write-Status "Failed to set TrustedHosts: $_" -Type Error
    }
}

function New-WinRMFirewallRules {
    Write-Section "Configuring Firewall Rules"

    if ($DiagnoseOnly) {
        Write-Status "DiagnoseOnly mode - skipping changes" -Type Info
        return
    }

    $ipsToAllow = $ClientIPs | Where-Object { $_ -notmatch '/' }  # Use individual IPs

    foreach ($ip in $ipsToAllow) {
        foreach ($port in @(5985, 5986)) {
            $protocol = if ($port -eq 5985) { 'HTTP' } else { 'HTTPS' }
            $ruleName = "WinRM-$protocol-$ip"

            # Check if rule exists
            $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

            if (-not $existing) {
                try {
                    New-NetFirewallRule -DisplayName $ruleName `
                        -Direction Inbound `
                        -LocalPort $port `
                        -Protocol TCP `
                        -RemoteAddress $ip `
                        -Action Allow `
                        -ErrorAction Stop | Out-Null
                    Write-Status "Created firewall rule: $ruleName" -Type Success
                } catch {
                    Write-Status "Failed to create rule $ruleName : $_" -Type Error
                }
            } else {
                Write-Status "Firewall rule exists: $ruleName" -Type Info
            }
        }
    }

    # Also ensure the default WinRM rules are enabled
    try {
        Enable-NetFirewallRule -DisplayGroup "Windows Remote Management" -ErrorAction SilentlyContinue
        Write-Status "Enabled Windows Remote Management firewall group" -Type Success
    } catch {}
}

function Set-WinRMSettings {
    Write-Section "Configuring WinRM Settings"

    if ($DiagnoseOnly) {
        Write-Status "DiagnoseOnly mode - skipping changes" -Type Info
        return
    }

    $settings = @(
        @{ Path = 'winrm/config/service'; Setting = 'MaxConcurrentOperationsPerUser'; Value = '100' },
        @{ Path = 'winrm/config/winrs'; Setting = 'MaxMemoryPerShellMB'; Value = '1024' }
    )

    foreach ($s in $settings) {
        try {
            $result = & winrm set $s.Path "@{$($s.Setting)=`"$($s.Value)`"}" 2>&1

            if ($result -match 'cannot be changed.*GPO|controlled by policies') {
                Write-Status "$($s.Setting) is GPO-controlled - skipped" -Type Warning
                $script:Config.GPOIssues += $s.Setting
            } else {
                Write-Status "Set $($s.Setting) = $($s.Value)" -Type Success
            }
        } catch {
            Write-Status "Failed to set $($s.Setting): $_" -Type Warning
        }
    }
}

function Test-WinRMListener {
    Write-Section "Testing WinRM Listener"

    # Check if listening on expected ports
    foreach ($port in $script:Config.WinRMPorts) {
        $listening = Test-PortAccessibility -Port $port
        Write-Status "Port $port listening: $listening" -Type $(if($listening){'Success'}else{'Warning'})
    }

    # Show listener config
    try {
        $listeners = winrm enumerate winrm/config/listener 2>&1
        Write-Status "Active listeners:" -Type Info
        $listeners | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } catch {
        Write-Status "Could not enumerate listeners" -Type Warning
    }
}

# ============================================================================
# MAIN
# ============================================================================
function Main {
    Write-Host ""
    Write-Host "WinRM Server Setup Script" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host "Hostname: $env:COMPUTERNAME"
    Write-Host "Mode: $(if($DiagnoseOnly){'Diagnose Only'}else{'Configure'})"
    Write-Host ""

    # Get local IPs
    $script:Config.LocalIPs = Get-LocalIPAddresses
    Write-Status "Local IPs: $($script:Config.LocalIPs -join ', ')" -Type Info
    Write-Status "Client IPs to allow: $($ClientIPs -join ', ')" -Type Info

    # Analyze current configuration
    $winrmConfig = Get-WinRMConfiguration

    # Setup steps (skipped if DiagnoseOnly)
    Enable-WinRMService
    Set-TrustedHosts
    New-WinRMFirewallRules
    Set-WinRMSettings
    Test-WinRMListener

    # If GPO issues found, show remediation
    if ($script:Config.GPOIssues.Count -gt 0 -or $winrmConfig.BlockedClients.Count -gt 0) {
        Get-GPORemediationSteps -GPOIssues $script:Config.GPOIssues -BlockedClients $winrmConfig.BlockedClients
    }

    # Final summary
    Write-Section "Summary"
    Write-Status "WinRM Service: $((Get-Service WinRM).Status)" -Type $(if((Get-Service WinRM).Status -eq 'Running'){'Success'}else{'Error'})

    $trustedHosts = (Get-Item WSMan:\localhost\Client\TrustedHosts -ErrorAction SilentlyContinue).Value
    Write-Status "TrustedHosts: $(if($trustedHosts){$trustedHosts}else{'(not set)'})" -Type Info

    if ($winrmConfig.BlockedClients.Count -gt 0) {
        Write-Status "ACTION REQUIRED: Fix GPO IPv4Filter to allow client IPs!" -Type Error
    } else {
        Write-Host ""
        Write-Status "Test from CLIENT with:" -Type Info
        Write-Host "  Test-WSMan -ComputerName $($script:Config.LocalIPs[0])" -ForegroundColor White
        Write-Host "  Enter-PSSession -ComputerName $($script:Config.LocalIPs[0]) -Credential (Get-Credential)" -ForegroundColor White
    }

    Write-Host ""
}

Main
