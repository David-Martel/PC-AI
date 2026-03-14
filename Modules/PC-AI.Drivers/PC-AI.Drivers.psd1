@{
    RootModule        = 'PC-AI.Drivers.psm1'
    ModuleVersion     = '1.1.0'
    GUID              = 'a4c7f2e1-3b85-4d92-8e1f-9c2d4a6b7e3f'
    Author            = 'David Martel'
    CompanyName       = 'David Martel'
    Copyright         = '(c) 2026 David Martel. All rights reserved.'
    Description       = 'Driver inventory, version comparison, Thunderbolt/USB4 networking, and update management module for PC-AI.'
    FunctionsToExport = @(
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
    CmdletsToExport   = @()
    VariablesToExport = @()
    AliasesToExport   = @()
    PrivateData       = @{
        PSData = @{
            Tags = @('Drivers', 'PnP', 'Updates', 'Hardware', 'Inventory', 'Thunderbolt', 'USB4', 'Networking')
        }
        PCAI = @{
            Commands = @('drivers', 'thunderbolt', 'network')
        }
    }
}
