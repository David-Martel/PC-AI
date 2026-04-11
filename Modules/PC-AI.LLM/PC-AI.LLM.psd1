@{
    # Module metadata
    RootModule = 'PC-AI.LLM.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'a9b3c7d6-1e8f-4a2b-3c5d-6e7f8a9b0c1d'
    Author = 'PC-AI Project'
    CompanyName = 'PC-AI'
    Copyright = '(c) 2026 PC-AI Project. All rights reserved.'
    Description = 'PowerShell module for integrating pcai-inference LLM with PC diagnostics and analysis'

    # CompatiblePSEditions requires PowerShellVersion >= 5.1 per PowerShell docs;
    # Test-ModuleManifest fails on older PS without this pairing.
    PowerShellVersion = '5.1'
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
