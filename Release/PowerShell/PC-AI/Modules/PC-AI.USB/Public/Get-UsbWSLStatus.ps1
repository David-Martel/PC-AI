#Requires -Version 5.1
<#
.SYNOPSIS
    Gets complete USB/WSL status

.DESCRIPTION
    Returns comprehensive status including WSL distributions, USB devices,
    usbipd service status, and WSL usbip client status.

.EXAMPLE
    Get-UsbWSLStatus
    Returns full status information

.OUTPUTS
    PSCustomObject with status information
#>
function Get-UsbWSLStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param()

    $result = [PSCustomObject]@{
        WSLDistributions = @()
        UsbDevices       = @()
        UsbIpdService    = $null
        UsbIpClient      = $null
        Severity         = 'OK'
    }

    Write-Host "=== USB/WSL Status ===" -ForegroundColor Cyan
    Write-Host ""

    # Get WSL distributions
    Write-Host "WSL Distributions:" -ForegroundColor Yellow
    try {
        $wslList = wsl -l -v 2>&1
        Write-Host $wslList

        $lines = $wslList -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch '^\s*NAME\s+STATE\s+VERSION' }
        foreach ($line in $lines) {
            if ($line -match '^\s*(\*?)\s*(\S+)\s+(\S+)\s+(\d+)') {
                $result.WSLDistributions += [PSCustomObject]@{
                    Name      = $matches[2]
                    State     = $matches[3]
                    Version   = [int]$matches[4]
                    IsDefault = $matches[1] -eq '*'
                }
            }
        }
    }
    catch {
        Write-Warning "Could not get WSL distributions: $_"
        $result.Severity = 'Warning'
    }

    Write-Host ""

    # Get USB devices
    Write-Host "USB Devices:" -ForegroundColor Yellow
    $result.UsbDevices = Get-UsbDeviceList
    if ($result.UsbDevices) {
        $result.UsbDevices | Format-Table -AutoSize | Out-String | Write-Host
    }
    else {
        Write-Host "  No USB devices found or enumeration failed"
    }

    Write-Host ""

    # Check usbipd service
    Write-Host "usbipd Service:" -ForegroundColor Yellow
    $service = Get-Service usbipd -ErrorAction SilentlyContinue
    if ($service) {
        $result.UsbIpdService = [PSCustomObject]@{
            Status    = $service.Status
            StartType = $service.StartType
        }
        Write-Host "  Status: $($service.Status)"
        Write-Host "  StartType: $($service.StartType)"

        if ($service.Status -ne 'Running') {
            $result.Severity = 'Warning'
        }
    }
    else {
        Write-Warning "  usbipd service not found - reinstall usbipd-win"
        $result.Severity = 'Warning'
    }

    Write-Host ""

    # Check WSL usbip client
    Write-Host "WSL usbip client:" -ForegroundColor Yellow
    $defaultDistro = $result.WSLDistributions | Where-Object { $_.IsDefault } | Select-Object -First 1
    if ($defaultDistro) {
        try {
            $usbipVersion = wsl -d $defaultDistro.Name -- bash -c "usbip version 2>&1" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $result.UsbIpClient = $usbipVersion
                Write-Host "  $usbipVersion" -ForegroundColor Green
            }
            else {
                Write-Warning "  usbip not found - run: wsl -- sudo apt install linux-tools-generic"
                $result.Severity = 'Warning'
            }
        }
        catch {
            Write-Warning "  Could not check usbip client: $_"
        }
    }

    Write-Host ""
    return $result
}
