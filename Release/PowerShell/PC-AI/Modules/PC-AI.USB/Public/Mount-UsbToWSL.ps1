#Requires -Version 5.1
<#
.SYNOPSIS
    Attaches a USB device to WSL

.DESCRIPTION
    Uses usbipd-win to attach a USB device to a WSL distribution.
    Requires administrator privileges.

.PARAMETER BusId
    USB Bus ID (e.g., 1-3, 18-4)

.PARAMETER Distribution
    WSL distribution name (default: Ubuntu)

.PARAMETER AutoAttach
    Enable auto-attach on hotplug

.EXAMPLE
    Mount-UsbToWSL -BusId "18-4"
    Attaches device to default distribution

.EXAMPLE
    Mount-UsbToWSL -BusId "18-4" -Distribution "Ubuntu" -AutoAttach
    Attaches with auto-reattach enabled

.OUTPUTS
    PSCustomObject with operation result
#>
function Mount-UsbToWSL {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+-\d+$')]
        [string]$BusId,

        [Parameter()]
        [string]$Distribution = 'Ubuntu',

        [Parameter()]
        [switch]$AutoAttach
    )

    $result = [PSCustomObject]@{
        BusId        = $BusId
        Distribution = $Distribution
        Success      = $false
        AutoAttach   = $AutoAttach
        Message      = $null
    }

    # Check prerequisites
    if (-not (Test-UsbIpdInstalled)) {
        $result.Message = "usbipd-win is not installed. Install from: https://github.com/dorssel/usbipd-win"
        Write-Error $result.Message
        return $result
    }

    if (-not (Test-IsAdministrator)) {
        $result.Message = "Administrator privileges required for USB attach operation"
        Write-Error $result.Message
        return $result
    }

    if ($PSCmdlet.ShouldProcess("Device $BusId", "Attach to WSL distribution '$Distribution'")) {
        try {
            Write-Host "[*] Attaching device $BusId to $Distribution..." -ForegroundColor Cyan

            $args = @('attach', '--wsl', $Distribution, '--busid', $BusId)
            if ($AutoAttach) {
                $args += '--auto-attach'
            }

            $output = & usbipd @args 2>&1

            if ($LASTEXITCODE -eq 0) {
                $result.Success = $true
                $result.Message = "Device $BusId attached to $Distribution"
                Write-Host "[+] $($result.Message)" -ForegroundColor Green

                if ($AutoAttach) {
                    Write-Host "[*] Auto-attach enabled for this device" -ForegroundColor Cyan
                }

                Write-Host ""
                Write-Host "Verify in WSL with:" -ForegroundColor Yellow
                Write-Host "  wsl -d $Distribution -- lsusb" -ForegroundColor White
                Write-Host "  wsl -d $Distribution -- ls -l /dev/ttyUSB*" -ForegroundColor White
            }
            else {
                $result.Message = "Failed to attach device: $output"
                Write-Error $result.Message
            }
        }
        catch {
            $result.Message = "Attach operation failed: $_"
            Write-Error $result.Message
        }
    }

    return $result
}
