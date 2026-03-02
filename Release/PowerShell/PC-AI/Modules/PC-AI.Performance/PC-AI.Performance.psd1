@{
    # Script module or binary module file associated with this manifest.
    RootModule        = 'PC-AI.Performance.psm1'

    # Version number of this module.
    ModuleVersion     = '1.0.0'

    # ID used to uniquely identify this module
    GUID              = 'a8c20b43-95bd-47d4-c8e5-83f8e6c49910'

    # Author of this module
    Author            = 'David Martel'

    # Company or vendor of this module
    CompanyName       = 'David Martel'

    # Copyright statement for this module
    Copyright         = '(c) 2025 David Martel. All rights reserved.'

    # Description of the functionality provided by this module
    Description       = 'High-performance native metrics collector for PC_AI using Rust/C#.'

    # Functions to export from this module
    FunctionsToExport = @(
        'Get-DiskSpace',
        'Get-ProcessPerformance',
        'Watch-SystemResources',
        'Optimize-Disks',
        'Get-PcaiDiskUsage',
        'Get-PcaiTopProcess',
        'Get-PcaiMemoryStat',
        'Test-PcaiNative'
    )

    # Struct mappings
    TypesToProcess    = @('PC-AI.Performance.types.ps1xml')

    # Cmdlets to export from this module
    CmdletsToExport   = @()

    # Variables to export from this module
    VariablesToExport = @()

    # Aliases to export from this module
    AliasesToExport   = @()

    # Private data to pass to the module specified in RootModule/ModuleToProcess.
    PrivateData       = @{
        PSData = @{
            # Tags applied to this module. These help with module discovery in online galleries.
            Tags = @('Performance', 'Rust', 'Diagnostics', 'AI')
        }
        PCAI = @{
            Commands = @('optimize', 'perf')
        }
    }
}
