#Requires -Version 5.1
<#
.SYNOPSIS
    Internal helper functions for USB module
#>

function Test-UsbIpdInstalled {
    [CmdletBinding()]
    [OutputType([bool])]
    param()
    $usbipd = Get-Command usbipd.exe -ErrorAction SilentlyContinue
    return $null -ne $usbipd
}

function Test-IsAdministrator {
    [CmdletBinding()]
    [OutputType([bool])]
    param()
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-UsbDevicesFallback {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param()

    $nativeAvailable = $false
    try {
        if (Get-Module -Name 'PC-AI.Acceleration') {
            $nativeAvailable = Test-PcaiNativeAvailable
        }
    } catch {}

    try {
        $devices = Get-CimInstance -ClassName Win32_PnPEntity | Where-Object {
            $_.Service -match 'usbser|ftdibus|usbccgp' -or
            $_.DeviceID -match '^USB\\VID_'
        } | ForEach-Object {
            $vidPid = $null
            if ($_.DeviceID -match 'VID_([0-9A-F]{4}).*PID_([0-9A-F]{4})') {
                $vidPid = "$($Matches[1]):$($Matches[2])"
            }

            $deviceObj = [PSCustomObject]@{
                DeviceID      = $_.DeviceID
                Description   = $_.Caption
                Status        = $_.Status
                VID_PID       = $vidPid
                Source        = 'WMI'
                ConfigManagerErrorCode = $_.ConfigManagerErrorCode
                NativeStatus  = $null
            }

            if ($nativeAvailable -and $_.ConfigManagerErrorCode -gt 0) {
                try {
                    $probInfo = [PcaiNative.PcaiCore]::GetUsbProblemInfo($_.ConfigManagerErrorCode)
                    if ($probInfo) {
                        $deviceObj.NativeStatus = [PSCustomObject]@{
                            Code        = $probInfo.Code
                            Description = $probInfo.Description
                            Summary     = $probInfo.Summary
                            HelpUrl     = $probInfo.HelpUrl
                        }
                        $deviceObj.Status = "Error ($($probInfo.Description))"
                    }
                } catch {
                    Write-Verbose "Failed to get native problem info for code $($_.ConfigManagerErrorCode): $_"
                }
            }
            $deviceObj
        }

        return $devices
    }
    catch {
        Write-Error "Failed to enumerate USB devices via WMI: $_"
        return @()
    }
}
