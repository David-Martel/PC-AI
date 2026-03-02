#Requires -Version 5.1
<#
.SYNOPSIS
    Repairs WSL networking issues

.DESCRIPTION
    Fixes common WSL networking problems including virtual switch configuration,
    NAT setup, and network stack reset.

.PARAMETER RestartWSL
    Restart WSL after applying fixes (default: true)

.PARAMETER Force
    Apply all fixes without prompting

.EXAMPLE
    Repair-WSLNetworking
    Repairs networking with default options

.EXAMPLE
    Repair-WSLNetworking -RestartWSL:$false
    Repairs networking without restarting WSL

.OUTPUTS
    PSCustomObject with repair results
#>
function Repair-WSLNetworking {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$RestartWSL = $true,

        [Parameter()]
        [switch]$Force
    )

    # Check for Administrator privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error "This function requires Administrator privileges. Please run PowerShell as Administrator."
        return
    }

    $result = [PSCustomObject]@{
        VirtualSwitchFixed = $false
        NATConfigured      = $false
        NetworkStackReset  = $false
        WSLRestarted       = $false
        Errors             = @()
    }

    Write-Host "[*] Starting WSL Networking Repair..." -ForegroundColor Cyan

    # Fix virtual switch
    if ($PSCmdlet.ShouldProcess('WSL Virtual Switch', 'Configure')) {
        try {
            Write-Host "[*] Checking WSL virtual switch..." -ForegroundColor Yellow

            $wslSwitch = Get-VMSwitch -Name "WSL" -ErrorAction SilentlyContinue
            if (-not $wslSwitch) {
                Write-Host "  [*] Creating WSL virtual switch..." -ForegroundColor Yellow
                New-VMSwitch -Name "WSL" -SwitchType Internal -ErrorAction Stop
                Start-Sleep -Seconds 2
                $result.VirtualSwitchFixed = $true
                Write-Host "  [+] WSL virtual switch created" -ForegroundColor Green
            }
            else {
                Write-Host "  [=] WSL virtual switch already exists" -ForegroundColor Green
                $result.VirtualSwitchFixed = $true
            }

            # Configure adapter
            $wslAdapter = Get-NetAdapter -Name "vEthernet (WSL)" -ErrorAction SilentlyContinue
            if ($wslAdapter) {
                Remove-NetIPAddress -InterfaceAlias "vEthernet (WSL)" -Confirm:$false -ErrorAction SilentlyContinue
                Remove-NetRoute -InterfaceAlias "vEthernet (WSL)" -Confirm:$false -ErrorAction SilentlyContinue
                New-NetIPAddress -InterfaceAlias "vEthernet (WSL)" -IPAddress "172.16.0.1" -PrefixLength 16 -ErrorAction SilentlyContinue

                # Configure NAT
                $existingNat = Get-NetNat -Name "WSL" -ErrorAction SilentlyContinue
                if (-not $existingNat) {
                    New-NetNat -Name "WSL" -InternalIPInterfaceAddressPrefix "172.16.0.0/16" -ErrorAction SilentlyContinue
                    $result.NATConfigured = $true
                    Write-Host "  [+] NAT configured for WSL" -ForegroundColor Green
                }
            }
        }
        catch {
            $result.Errors += "Virtual switch: $_"
            Write-Host "  [!] Error configuring virtual switch: $_" -ForegroundColor Red
        }
    }

    # Reset network stack
    if ($PSCmdlet.ShouldProcess('Network Stack', 'Reset')) {
        try {
            Write-Host "[*] Resetting network stack..." -ForegroundColor Yellow
            netsh winsock reset | Out-Null
            netsh int ip reset | Out-Null
            ipconfig /flushdns | Out-Null
            $result.NetworkStackReset = $true
            Write-Host "  [+] Network stack reset complete" -ForegroundColor Green
        }
        catch {
            $result.Errors += "Network stack reset: $_"
            Write-Host "  [!] Error resetting network stack: $_" -ForegroundColor Red
        }
    }

    # Restart WSL
    if ($RestartWSL -and $PSCmdlet.ShouldProcess('WSL', 'Restart')) {
        try {
            Write-Host "[*] Restarting WSL..." -ForegroundColor Yellow
            wsl --shutdown
            Start-Sleep -Seconds 5

            # Test WSL
            $testResult = wsl -d Ubuntu echo "WSL connectivity test" 2>&1
            if ($testResult -match "WSL connectivity test") {
                $result.WSLRestarted = $true
                Write-Host "  [+] WSL restarted and verified" -ForegroundColor Green
            }
            else {
                Write-Host "  [!] WSL restart may have issues" -ForegroundColor Yellow
            }
        }
        catch {
            $result.Errors += "WSL restart: $_"
            Write-Host "  [!] Error restarting WSL: $_" -ForegroundColor Red
        }
    }

    # Summary
    Write-Host ""
    if ($result.Errors.Count -eq 0) {
        Write-Host "[*] WSL networking repair completed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "[!] WSL networking repair completed with some errors" -ForegroundColor Yellow
    }

    return $result
}
