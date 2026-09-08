@{
    # Module metadata
    RootModule = 'PC-AI.LLM.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'a9b3c7d6-1e8f-4a2b-3c5d-6e7f8a9b0c1d'
    Author = 'PC-AI Project'
    CompanyName = 'PC-AI'
    Copyright = '(c) 2026 PC-AI Project. All rights reserved.'
    Description = 'PowerShell module for integrating pcai-inference LLM with PC diagnostics and analysis'

    # PC-AI.LLM.psm1 opens with `#Requires -PSEdition Core`, so this module
    # cannot load on Windows PowerShell 5.1 at all. Advertising
    # PowerShellVersion = '5.1' was a contract the module could not honour: the
    # first real 5.1 compatibility run failed with "cannot be run because it
    # contained a #requires statement for PowerShell editions 'Core'".
    # CompatiblePSEditions requires PowerShellVersion >= 5.1 per the docs; 7.0
    # satisfies that and states the actual floor.
    PowerShellVersion = '7.0'
    CompatiblePSEditions = @('Core')

    # Functions to export
    FunctionsToExport = @(
        'Get-LLMStatus'
        'Send-OllamaRequest'
        'Invoke-LLMChat'
        'Invoke-LLMChatRouted'
        'Invoke-LLMChatTui'
        'Invoke-FunctionGemmaReAct'
        'Invoke-FunctionGemmaDataset'
        'Invoke-FunctionGemmaTokenCache'
        'Invoke-FunctionGemmaTrain'
        'Invoke-FunctionGemmaEval'
        'Invoke-FunctionGemmaTests'
        'Invoke-PCDiagnosis'
        'Set-LLMConfig'
        'Set-LLMProviderOrder'
        'Invoke-SmartDiagnosis'
        'Invoke-NativeSearch'
        'Invoke-DocSearch'
        'Get-SystemInfoTool'
        'Invoke-LogSearch'
    )

    # Cmdlets to export
    CmdletsToExport = @()

    # Variables to export
    VariablesToExport = @()

    # Aliases to export
    AliasesToExport = @()

    # Private data
    PrivateData = @{
        PSData = @{
            Tags = @('pcai-inference', 'LLM', 'AI', 'Diagnostics', 'PC-AI')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
            IconUri = ''
            ReleaseNotes = 'Updated to use pcai-inference as the primary local LLM backend'
        }
        PCAI = @{
            Commands = @('analyze', 'chat', 'llm')
        }
    }
}
