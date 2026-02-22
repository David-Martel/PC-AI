@{
    RootModule = 'PC-AI.Virtualization.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'a4c7e2d1-8b3f-4a5e-9d6c-1f2e3b4a5c6d'
    Author = 'PC_AI Framework'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025 PC_AI Framework. All rights reserved.'
    Description = 'WSL2, Hyper-V, and Docker diagnostics and optimization for PC-AI framework.'
    PowerShellVersion = '5.1'
    FunctionsToExport = @(
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
    CmdletsToExport = @()
    VariablesToExport = @()
    AliasesToExport = @()
    PrivateData = @{
        PSData = @{
            Tags = @('WSL', 'HyperV', 'Docker', 'Virtualization', 'PC-AI')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
        PCAI = @{
            Commands = @('diagnose', 'optimize', 'doctor')
        }
    }
}
