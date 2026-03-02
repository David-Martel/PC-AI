#Requires -Version 5.1
<#
.SYNOPSIS
    Detaches a USB device from WSL

.DESCRIPTION
    Uses usbipd-win to detach a USB device from WSL and optionally
    unbind it to restore the Windows driver.

.PARAMETER BusId
    USB Bus ID (e.g., 1-3, 18-4)

.PARAMETER Unbind
    Also unbind device to restore Windows driver

.EXAMPLE
    Dismount-UsbFromWSL -BusId "18-4"
    Detaches device from WSL

.EXAMPLE
    Dismount-UsbFromWSL -BusId "18-4" -Unbind
    Detaches and restores Windows driver

.OUTPUTS
    PSCustomObject with operation result
#>
function Dismount-UsbFromWSL {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+-\d+$')]
        [string]$BusId,

        [Parameter()]
        [switch]$Unbind
    )

    $result = [PSCustomObject]@{
        BusId    = $BusId
        Detached = $false
        Unbound  = $false
        Message  = $null
    }

    if (-not (Test-UsbIpdInstalled)) {
        $result.Message = "usbipd-win is not installed"
        Write-Error $result.Message
        return $result
    }

    if (-not (Test-IsAdministrator)) {
        $result.Message = "Administrator privileges required for USB detach operation"
        Write-Error $result.Message
        return $result
    }

    if ($PSCmdlet.ShouldProcess("Device $BusId", "Detach from WSL")) {
        try {
            Write-Host "[*] Detaching device $BusId from WSL..." -ForegroundColor Cyan

            $output = & usbipd detach --busid $BusId 2>&1

            if ($LASTEXITCODE -eq 0) {
                $result.Detached = $true
                Write-Host "[+] Device $BusId detached from WSL" -ForegroundColor Green

                if ($Unbind) {
                    Write-Host "[*] Unbinding device to restore Windows driver..." -ForegroundColor Cyan
                    $unbindOutput = & usbipd unbind --busid $BusId 2>&1

                    if ($LASTEXITCODE -eq 0) {
                        $result.Unbound = $true
                        Write-Host "[+] Device unbound - Windows driver restored" -ForegroundColor Green
                    }
                    else {
                        Write-Warning "Unbind failed: $unbindOutput"
                        Write-Host "[*] Device may need to be unplugged/replugged to restore COM port" -ForegroundColor Yellow
                    }
                }

                $result.Message = "Device $BusId successfully detached"
            }
            else {
                $result.Message = "Failed to detach device: $output"
                Write-Error $result.Message
            }
        }
        catch {
            $result.Message = "Detach operation failed: $_"
            Write-Error $result.Message
        }
    }

    return $result
}
