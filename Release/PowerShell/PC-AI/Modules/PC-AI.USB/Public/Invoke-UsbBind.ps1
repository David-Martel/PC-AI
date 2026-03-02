#Requires -Version 5.1
<#
.SYNOPSIS
    Binds a USB device for WSL sharing

.DESCRIPTION
    Uses usbipd-win to bind a USB device for sharing with WSL.
    This is required before the device can be attached.

.PARAMETER BusId
    USB Bus ID (e.g., 1-3, 18-4)

.PARAMETER Force
    Force binding even with incompatible USB filters

.EXAMPLE
    Invoke-UsbBind -BusId "18-4"
    Binds device for sharing

.EXAMPLE
    Invoke-UsbBind -BusId "18-4" -Force
    Force binds device

.OUTPUTS
    PSCustomObject with operation result
#>
function Invoke-UsbBind {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+-\d+$')]
        [string]$BusId,

        [Parameter()]
        [switch]$Force
    )

    $result = [PSCustomObject]@{
        BusId   = $BusId
        Success = $false
        Message = $null
    }

    if (-not (Test-UsbIpdInstalled)) {
        $result.Message = "usbipd-win is not installed"
        Write-Error $result.Message
        return $result
    }

    if (-not (Test-IsAdministrator)) {
        $result.Message = "Administrator privileges required for USB bind operation"
        Write-Error $result.Message
        return $result
    }

    if ($PSCmdlet.ShouldProcess("Device $BusId", "Bind for WSL sharing")) {
        try {
            Write-Host "[*] Binding device $BusId for sharing..." -ForegroundColor Cyan

            $args = @('bind', '--busid', $BusId)
            if ($Force) {
                $args += '--force'
            }

            $output = & usbipd @args 2>&1

            if ($LASTEXITCODE -eq 0) {
                $result.Success = $true
                $result.Message = "Device $BusId bound for sharing"
                Write-Host "[+] $($result.Message)" -ForegroundColor Green
                Write-Host "[*] Device can now be attached to WSL with: Mount-UsbToWSL -BusId $BusId" -ForegroundColor Cyan
            }
            else {
                $result.Message = "Failed to bind device: $output"

                if ($output -match 'incompatible.*filter' -and -not $Force) {
                    Write-Warning "Incompatible USB filter detected. Try with -Force parameter."
                }

                Write-Error $result.Message
            }
        }
        catch {
            $result.Message = "Bind operation failed: $_"
            Write-Error $result.Message
        }
    }

    return $result
}
