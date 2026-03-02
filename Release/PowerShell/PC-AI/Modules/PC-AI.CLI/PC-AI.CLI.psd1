@{
    RootModule = 'PC-AI.CLI.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'd3cf2a44-7c3b-4ad7-9db7-4a5b3b2d6d0a'
    Author = 'PC_AI Framework'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025-2026 PC_AI'
    Description = 'PC-AI CLI utilities for dynamic help and command metadata.'
    PowerShellVersion = '5.1'
    FunctionsToExport = @(
        'Get-PCCommandMap'
        'Get-PCCommandModules'
        'Get-PCCommandList'
        'Get-PCCommandSummary'
        'Get-PCModuleHelpIndex'
        'Get-PCModuleHelpEntry'
        'Parse-PCArguments'
        'Resolve-PCArguments'
    )
    PrivateData = @{
        PSData = @{
            Tags = @('PC-AI','CLI','Help')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
    }
}
