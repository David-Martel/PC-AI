#Requires -Version 5.1
<#
.SYNOPSIS
    Lists USB devices available for WSL attachment and general diagnostics.
#>
function Get-UsbDeviceList {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string]$Filter
    )

    $results = @()

    # 1. Try PCAI Native Diagnostics (Phase 7)
    $nativeAvailable = $false
    try {
        if (Get-Module -Name 'PC-AI.Acceleration') {
            $nativeAvailable = Test-PcaiNativeAvailable
        }
    } catch {}

    if ($nativeAvailable) {
        try {
            $json = Get-PcaiNativeUsbDiagnostics
            if ($json) {
                $nativeDevices = $json | ConvertFrom-Json
                foreach ($dev in $nativeDevices) {
                    $device = [PSCustomObject]@{
                        BusId       = $null
                        VID_PID     = if ($dev.hardware_id -match 'VID_([0-9A-F]{4})&PID_([0-9A-F]{4})') { "$($Matches[1]):$($Matches[2])" } else { $null }
                        Description = $dev.name
                        State       = $dev.status
                        Source      = 'Native'
                        DeviceID    = $dev.hardware_id
                        NativeStatus = if ($dev.config_error_code -gt 0) {
                            [PSCustomObject]@{
                                Code        = $dev.config_error_code
                                Description = $dev.status
                                Summary     = $dev.error_summary
                                HelpUrl     = $dev.help_url
                            }
                        } else { $null }
                    }

                    if (-not $Filter -or $device.Description -like "*$Filter*" -or $device.VID_PID -like "*$Filter*") {
                        $results += $device
                    }
                }

                if ($results.Count -gt 0) {
                    return $results
                }
            }
        }
        catch {
            Write-Verbose "Native USB diagnostics failed: $_"
        }
    }

    # 2. Try usbipd
    if (Test-UsbIpdInstalled) {
        try {
            $output = & usbipd list 2>&1
            if ($LASTEXITCODE -eq 0 -and $output -notmatch 'Unhandled exception|ERROR_PATH_NOT_FOUND') {
                $lines = $output -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch '^BUSID\s+VID:PID' }

                foreach ($line in $lines) {
                    if ($line -match '^(\d+-\d+)\s+([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(.+?)\s+(Not\s+shared|Shared|Attached)') {
                        $device = [PSCustomObject]@{
                            BusId       = $matches[1]
                            VID_PID     = $matches[2]
                            Description = $matches[3].Trim()
                            State       = $matches[4]
                            Source      = 'usbipd'
                        }

                        if (-not $Filter -or $device.Description -like "*$Filter*" -or $device.VID_PID -like "*$Filter*") {
                            $results += $device
                        }
                    }
                }

                if ($results.Count -gt 0) {
                    return $results
                }
            }
        }
        catch {
            Write-Verbose "usbipd error: $_"
        }
    }

    # 3. Fallback to WMI
    $wmiDevices = Get-UsbDevicesFallback

    if ($Filter) {
        $wmiDevices = $wmiDevices | Where-Object {
            $_.Description -like "*$Filter*" -or $_.VID_PID -like "*$Filter*"
        }
    }

    return $wmiDevices
}
