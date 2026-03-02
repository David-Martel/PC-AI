#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI Virtualization Module

.DESCRIPTION
    Provides WSL2, Hyper-V, and Docker diagnostic and optimization functions.

.NOTES
    Author: PC_AI Framework
    Version: 1.0.0
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
    'Get-WSLStatus',
    'Optimize-WSLConfig',
    'Set-WSLDefenderExclusions',
    'Repair-WSLNetworking',
    'Get-HyperVStatus',
    'Get-DockerStatus',
    'Backup-WSLConfig',
    'Get-WSLEnvironmentHealth',
    'Enable-WSLSystemd',
    'Install-WSLVsockBridge',
    'Get-WSLVsockBridgeStatus',
    'Invoke-WSLNetworkToolkit',
    'Invoke-WSLDockerHealthCheck',
    'Install-HVSockProxy',
    'Register-HVSockServices',
    'Start-HVSockProxy',
    'Stop-HVSockProxy',
    'Get-HVSockProxyStatus',
    'Get-PcaiServiceHealth',
    'Invoke-PcaiServiceHost',
    'Invoke-PcaiDoctor',
    'Set-PCaiServiceState'
)
