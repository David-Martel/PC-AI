@{
    RootModule        = 'PC-AI.Gpu.psm1'
    ModuleVersion     = '1.0.0'
    GUID              = 'b3e8d1a2-5f74-4c91-9d2e-7a3f5c8b0e4d'
    Author            = 'David Martel'
    CompanyName       = 'David Martel'
    Copyright         = '(c) 2026 David Martel. All rights reserved.'
    Description       = 'NVIDIA GPU inventory, software registry, utilization monitoring, environment initialization, and update management module for PC-AI.'
    FunctionsToExport = @(
        'Get-NvidiaGpuInventory',
        'Get-NvidiaSoftwareRegistry',
        'Get-NvidiaSoftwareStatus',
        'Get-NvidiaGpuUtilization',
        'Get-NvidiaCompatibilityMatrix',
        'Initialize-NvidiaEnvironment',
        'Install-NvidiaSoftware',
        'Update-NvidiaSoftwareRegistry'
    )
    CmdletsToExport   = @()
    VariablesToExport = @()
    AliasesToExport   = @()
    PrivateData       = @{
        PSData = @{
            Tags = @('NVIDIA', 'GPU', 'CUDA', 'cuDNN', 'TensorRT', 'Nsight', 'Hardware', 'Inventory', 'Performance')
        }
        PCAI = @{
            Commands = @('gpu', 'nvidia', 'cuda')
        }
    }
}
