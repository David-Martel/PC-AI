@{
    RootModule = 'PC-AI.USB.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'c5d9f3e2-7a4b-4c8d-9e1f-2a3b4c5d6e7f'
    Author = 'PC_AI Framework'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025 PC_AI Framework. All rights reserved.'
    Description = 'USB device management for WSL integration in PC-AI framework.'
    PowerShellVersion = '5.1'
    FunctionsToExport = @(
        'Get-UsbDeviceList',
        'Mount-UsbToWSL',
        'Dismount-UsbFromWSL',
        'Get-UsbWSLStatus',
        'Invoke-UsbBind',
        'Get-PcaiNativeUsbDiagnostics'
    )
    CmdletsToExport = @()
    VariablesToExport = @()
    AliasesToExport = @()
    PrivateData = @{
        PSData = @{
            Tags = @('USB', 'WSL', 'usbipd', 'PC-AI')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
        PCAI = @{
            Commands = @('usb')
        }
    }
}
