@{
    RootModule = 'PcaiInference.psm1'
    ModuleVersion = '1.0.0'
    GUID = '8e7c4f1a-3b9d-4c5e-8f0a-1b2c3d4e5f6a'
    Author = 'PC-AI Team'
    CompanyName = 'PC-AI'
    Copyright = '(c) 2026 PC-AI Team. All rights reserved.'
    Description = 'PowerShell FFI bindings for pcai-inference Rust native library'
    PowerShellVersion = '7.0'

    FunctionsToExport = @(
        'Initialize-PcaiInference'
        'Import-PcaiModel'
        'Invoke-PcaiGenerate'
        'Close-PcaiInference'
        'Get-PcaiInferenceStatus'
        'Test-PcaiInference'
        'Test-PcaiDllVersion'
    )

    VariablesToExport = @()
    AliasesToExport = @()

    PrivateData = @{
        PSData = @{
            Tags = @('LLM', 'Inference', 'FFI', 'Native', 'AI', 'llama.cpp', 'mistral.rs')
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
        }

        # Native DLL requirements
        NativeDependencies = @{
            DllName = 'pcai_inference.dll'
            MinVersion = '1.0.0'
            Backends = @('llamacpp', 'mistralrs')
        }
    }
}
