#Requires -Version 7.0
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-ThunderboltRcloneMount.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $SourcePath)) { throw 'USB4 mount gate is missing; legacy startup mount can follow the default LAN route.' }
. $SourcePath
$adapter = @{ InterfaceDescription = 'USB4(TM) P2P Network Adapter'; Status = $null; InterfaceOperationalStatus = 1; AdminStatus = 1; ifIndex = 28 }
$addresses = @(@{ IPAddress = '172.31.252.134'; InterfaceIndex = 28; AddressState = 'Preferred' })
$route = @(@{ InterfaceIndex = 28; DestinationPrefix = '172.31.252.132/30'; NextHop = '0.0.0.0' })
# Selector: OwnSourceAndOnLinkRouteAdmitMount. Protects useful USB4 mounts;
# detects an unconditional deny gate. Needs fixtures only. Breadcrumb: Test-ThunderboltMountPath.
if (-not (Test-ThunderboltMountPath $adapter $addresses $route '172.31.252.134')) { throw 'Valid USB4 source/route rejected.' }
Write-Output 'PASS: OwnSourceAndOnLinkRouteAdmitMount'
# Selector: FallbackAndWrongSourceRefuseMount. Protects roaming multihomed hosts;
# detects missing/foreign sources and default-LAN fallback. Needs no host APIs.
# Breadcrumb: Test-ThunderboltMountPath interface ownership and on-link checks.
$wrong = @(@{ InterfaceIndex = 31; DestinationPrefix = '0.0.0.0/0'; NextHop = '192.168.50.1' })
if (Test-ThunderboltMountPath $adapter $addresses $wrong '172.31.252.134') { throw 'Default LAN fallback admitted.' }
if (Test-ThunderboltMountPath $adapter @() $route '172.31.252.134') { throw 'Absent source admitted.' }
$foreign = @(@{ IPAddress = '172.31.252.134'; InterfaceIndex = 31; AddressState = 'Preferred' })
if (Test-ThunderboltMountPath $adapter $foreign $route '172.31.252.134') { throw 'Foreign source admitted.' }
Write-Output 'PASS: FallbackAndWrongSourceRefuseMount'
# Selector: AbsentOrNonUsb4RefuseMount. Protects interface custody; detects an
# alias alone admitting a wired/WiFi adapter or missing hardware. Needs fixtures.
# Breadcrumb: Test-ThunderboltMountPath actual driver description/status checks.
$down = $adapter.Clone(); $down.InterfaceOperationalStatus = 2; $down.Status = 'Up'
if (Test-ThunderboltMountPath $down $addresses $route '172.31.252.134') { throw 'Disconnected adapter admitted.' }
$lan = $adapter.Clone(); $lan.InterfaceDescription = 'Wired Ethernet'
if (Test-ThunderboltMountPath $lan $addresses $route '172.31.252.134') { throw 'Unrelated Ethernet adapter admitted.' }
$disabled = $adapter.Clone(); $disabled.AdminStatus = 2
if (Test-ThunderboltMountPath $disabled $addresses $route '172.31.252.134') { throw 'Administratively disabled USB4 adapter admitted.' }
Write-Output 'PASS: AbsentOrNonUsb4RefuseMount'
