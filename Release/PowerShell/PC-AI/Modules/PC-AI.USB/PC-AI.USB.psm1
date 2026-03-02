#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI USB Management Module

.DESCRIPTION
    Provides USB device management for WSL integration using usbipd-win.

.NOTES
    Author: PC_AI Framework
    Version: 1.0.0
    Requires: usbipd-win (https://github.com/dorssel/usbipd-win)
#>

$script:ModuleRoot = $PSScriptRoot

# Import private functions
$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

# Import public functions
$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Get-UsbDeviceList',
    'Mount-UsbToWSL',
    'Dismount-UsbFromWSL',
    'Get-UsbWSLStatus',
    'Invoke-UsbBind',
    'Get-PcaiNativeUsbDiagnostics'
)
