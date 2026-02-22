#Requires -Version 5.1

<#
.SYNOPSIS
    PC-AI Hardware Diagnostics Module

.DESCRIPTION
    Provides read-only cmdlets for querying Windows hardware health via CIM/WMI.
    Replaces the legacy Get-PcDiagnostics.ps1 monolithic script with individual,
    composable functions that can be called independently or orchestrated via
    New-DiagnosticReport.

    Exported functions:
      Get-DeviceErrors      - Query Device Manager for PnP devices with non-zero
                              ConfigManagerErrorCode (equivalent to hardware errors
                              shown in Device Manager with yellow/red icons).
      Get-DiskHealth        - Query physical disk SMART status via CIM/wmic.
      Get-UsbStatus         - Enumerate USB controllers and connected USB devices,
                              including error codes for each device.
      Get-NetworkAdapters   - List physical network adapters with connection status,
                              speed, and enabled state.
      Get-SystemEvents      - Query the System event log for disk/USB/storage-related
                              warnings and errors over a configurable look-back window.
      New-DiagnosticReport  - Orchestrate all of the above into a structured report
                              object (or file) covering all hardware categories.

    Native acceleration:
      This module uses only built-in CIM/WMI cmdlets and wmic. It does NOT require
      PcaiNative.dll. High-performance native acceleration for file/process metrics
      is handled by the PC-AI.Performance module instead.

    Dependencies:
      - PowerShell 5.1 or later
      - Windows 10/11 (CIM provider)
      - Administrator rights recommended for full event log and device access
#>

$script:ModuleRoot = $PSScriptRoot

$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Get-DeviceErrors',
    'Get-DiskHealth',
    'Get-UsbStatus',
    'Get-NetworkAdapters',
    'Get-SystemEvents',
    'New-DiagnosticReport'
)
