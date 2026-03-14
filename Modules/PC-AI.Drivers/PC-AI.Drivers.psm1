#Requires -Version 7.0

<#
.SYNOPSIS
    PC-AI Driver Management Module

.DESCRIPTION
    Provides cmdlets for inventorying installed drivers, querying the curated driver
    registry, comparing installed versions against known-latest, and launching driver
    updates from trusted vendor sources.

    Exported functions:
      Get-PnpDeviceInventory  - Enumerate PnP devices and their installed driver
                                versions using CIM/PnpDevice, matched against registry
                                rules (VID/PID, friendly name, PCI class).
      Get-DriverRegistry      - Load and return the driver-registry.json as a structured
                                object. Accepts optional -DeviceId filter.
      Compare-DriverVersion   - Compare an installed driver version string against the
                                latestVersion in the registry entry; returns status
                                (Current, Outdated, Unknown, NoUpdate).
      Get-DriverReport        - Orchestrate inventory + registry + comparison into a
                                single structured report object covering all matched
                                devices, with per-device update status.
      Install-DriverUpdate    - Launch the appropriate update action for a registry
                                entry: open download URL, invoke installer exe, or
                                direct to Windows Update. Supports -WhatIf.
      Update-DriverRegistry   - Refresh the local driver-registry.json from a remote
                                source URL or update individual device entries.

    Dependencies:
      - PowerShell 5.1 or later
      - Windows 10/11 (PnP provider, CIM)
      - Administrator rights recommended for device enumeration and driver installation
      - driver-registry.json at Config\driver-registry.json relative to module root
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
    'Get-PnpDeviceInventory',
    'Get-DriverRegistry',
    'Compare-DriverVersion',
    'Get-DriverReport',
    'Install-DriverUpdate',
    'Update-DriverRegistry',
    'Get-NetworkDiscoverySnapshot',
    'Find-ThunderboltPeer',
    'Get-ThunderboltNetworkStatus',
    'Connect-ThunderboltPeer',
    'Set-ThunderboltNetworkOptimization'
)
