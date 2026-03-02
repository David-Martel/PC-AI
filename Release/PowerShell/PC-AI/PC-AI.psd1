@{
    RootModule = 'PC-AI.psm1'
    ModuleVersion = '1.0.0'
    GUID = '02a9dcaf-0ac8-49ca-bcdd-61f39a12c721'
    Author = 'PC_AI Project'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025-2026 PC_AI Project'
    Description = 'PC-AI unified release module bundling diagnostics, acceleration, and AI tooling components.'
    PowerShellVersion = '7.0'
    CompatiblePSEditions = @('Core', 'Desktop')
    NestedModules = @(
        'Modules\PC-AI.Common\PC-AI.Common.psm1',
        'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1',
        'Modules\PC-AI.Cleanup\PC-AI.Cleanup.psd1',
        'Modules\PC-AI.CLI\PC-AI.CLI.psd1',
        'Modules\PC-AI.Evaluation\PC-AI.Evaluation.psd1',
        'Modules\PC-AI.Hardware\PC-AI.Hardware.psd1',
        'Modules\PC-AI.LLM\PC-AI.LLM.psd1',
        'Modules\PC-AI.Network\PC-AI.Network.psd1',
        'Modules\PC-AI.Performance\PC-AI.Performance.psd1',
        'Modules\PC-AI.USB\PC-AI.USB.psd1',
        'Modules\PC-AI.Virtualization\PC-AI.Virtualization.psd1',
        'Modules\PcaiInference.psd1'
    )
    FunctionsToExport = '*'
    CmdletsToExport = '*'
    VariablesToExport = @()
    AliasesToExport = '*'
    PrivateData = @{
        PSData = @{
            Tags = @('pc-ai', 'diagnostics', 'acceleration', 'llm')
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
        }
    }
}
