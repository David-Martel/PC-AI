@{
    RootModule = 'PC-AI.Acceleration.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'b0c1d2e3-4f5a-6b7c-8d9e-0f1a2b3c4d5e'
    Author = 'PC_AI Project'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025 PC_AI Project. All rights reserved.'
    Description = 'Rust and .NET performance acceleration layer for PC_AI framework'
    PowerShellVersion = '5.1'

    FunctionsToExport = @(
        # Tool Detection
        'Get-RustToolStatus'
        'Test-RustToolAvailable'

        # Accelerated Operations
        'Search-LogsFast'
        'Find-FilesFast'
        'Get-ProcessesFast'
        'Get-FileHashParallel'
        'Find-DuplicatesFast'
        'Get-DiskUsageFast'
        'Search-ContentFast'

        # Benchmarking
        'Measure-CommandPerformance'
        'Compare-ToolPerformance'

        # PCAI Native (Rust DLL + C# P/Invoke)
        'Initialize-PcaiNative'
        'Test-PcaiNativeAvailable'
        'Get-PcaiNativeStatus'
        'Get-PcaiCapabilities'
        'Invoke-PcaiNativeDuplicates'
        'Invoke-PcaiNativeFileSearch'
        'Invoke-PcaiNativeContentSearch'
    )

    PrivateData = @{
        PSData = @{
            Tags = @('Performance', 'Rust', 'Acceleration', 'Parallel', 'dotnet')
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
    }
}
