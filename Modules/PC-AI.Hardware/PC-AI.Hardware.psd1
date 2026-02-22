@{
    RootModule        = 'PC-AI.Hardware.psm1'
    ModuleVersion     = '1.0.0'
    GUID              = 'b9d3e1f2-06ce-48f1-9c6a-1a2b3c4d5e6f'
    Author            = 'David Martel'
    CompanyName       = 'David Martel'
    Copyright         = '(c) 2025 David Martel. All rights reserved.'
    Description       = 'Hardware diagnostics module for PC-AI (WMI/CIM based).'
    FunctionsToExport = @(
        'Get-DeviceErrors',
        'Get-DiskHealth',
        'Get-UsbStatus',
        'Get-NetworkAdapters',
        'Get-SystemEvents',
        'New-DiagnosticReport'
    )
    CmdletsToExport   = @()
    VariablesToExport = @()
    AliasesToExport   = @()
    PrivateData       = @{
        PSData = @{
            Tags = @('Hardware', 'Diagnostics', 'WMI', 'CIM')
        }
        PCAI = @{
            Commands = @('diagnose')
        }
    }
}
