@{
    RootModule = 'PcaiMedia.psm1'
    ModuleVersion = '0.1.0'
    GUID = 'a3f7b2e9-6c8d-4a1f-9e5b-7d0c2f8a4b6e'
    Author = 'PC-AI Team'
    CompanyName = 'PC-AI'
    Copyright = '(c) 2026 PC-AI Team. All rights reserved.'
    Description = 'PowerShell FFI bindings for pcai-media Rust native library (Janus-Pro multimodal)'
    PowerShellVersion = '5.1'

    FunctionsToExport = @(
        'Initialize-PcaiMedia'
        'Import-PcaiMediaModel'
        'New-PcaiImage'
        'Get-PcaiImageAnalysis'
        'Stop-PcaiMedia'
        'Get-PcaiMediaStatus'
    )

    VariablesToExport = @()
    AliasesToExport = @()

    PrivateData = @{
        PSData = @{
            Tags = @('LLM', 'Media', 'FFI', 'Native', 'AI', 'Janus-Pro', 'Image-Generation', 'VQA')
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
        }

        # Native DLL requirements
        NativeDependencies = @{
            DllName = 'pcai_media.dll'
            MinVersion = '0.1.0'
            Models = @('deepseek-ai/Janus-Pro-1B', 'deepseek-ai/Janus-Pro-7B')
        }
    }
}
