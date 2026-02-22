@{
    # Script module file associated with this manifest
    RootModule = 'PC-AI.Network.psm1'

    # Version number of this module
    ModuleVersion = '1.0.0'

    # ID used to uniquely identify this module
    GUID = 'd6e0f4a3-8b5c-4d9e-0f2a-3b4c5d6e7f8a'

    # Author of this module
    Author = 'PC_AI Framework'

    # Company or vendor of this module
    CompanyName = 'PC_AI'

    # Copyright statement for this module
    Copyright = '(c) 2025 PC_AI Framework. All rights reserved.'

    # Description of the functionality provided by this module
    Description = 'Network diagnostics and optimization module for PC-AI framework. Provides network stack analysis, VSock optimization, and WSL connectivity testing.'

    # Minimum version of PowerShell required by this module
    PowerShellVersion = '5.1'

    # Functions to export from this module
    FunctionsToExport = @(
        'Get-NetworkDiagnostics',
        'Optimize-VSock',
        'Watch-VSockPerformance',
        'Test-WSLConnectivity'
    )

    # Cmdlets to export from this module
    CmdletsToExport = @()

    # Variables to export from this module
    VariablesToExport = @()

    # Aliases to export from this module
    AliasesToExport = @()

    # Private data to pass to the module
    PrivateData = @{
        PSData = @{
            Tags = @('Network', 'Diagnostics', 'VSock', 'WSL', 'PC-AI')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
        PCAI = @{
            Commands = @('diagnose', 'optimize', 'perf')
        }
    }
}
