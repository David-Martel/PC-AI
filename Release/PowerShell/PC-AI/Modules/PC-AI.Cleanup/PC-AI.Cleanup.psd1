@{
    # Script module or binary module file associated with this manifest.
    RootModule = 'PC-AI.Cleanup.psm1'

    # Version number of this module.
    ModuleVersion = '1.0.0'

    # Supported PSEditions
    CompatiblePSEditions = @('Desktop', 'Core')

    # ID used to uniquely identify this module
    GUID = 'f8a2b6c5-0d7e-4f1a-2b4c-5d6e7f8a9b0c'

    # Author of this module
    Author = 'PC_AI Project'

    # Company or vendor of this module
    CompanyName = 'PC_AI'

    # Copyright statement for this module
    Copyright = '(c) 2025 PC_AI Project. All rights reserved.'

    # Description of the functionality provided by this module
    Description = 'PC-AI Cleanup module for system maintenance tasks including PATH cleanup, duplicate file detection, and temp file management.'

    # Minimum version of the PowerShell engine required by this module
    PowerShellVersion = '5.1'

    # Functions to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no functions to export.
    FunctionsToExport = @(
        'Get-PathDuplicates',
        'Repair-MachinePath',
        'Find-DuplicateFiles',
        'Clear-TempFiles'
    )

    # Cmdlets to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no cmdlets to export.
    CmdletsToExport = @()

    # Variables to export from this module
    VariablesToExport = @()

    # Aliases to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no aliases to export.
    AliasesToExport = @()

    # Private data to pass to the module specified in RootModule/ModuleToProcess. This may also contain a PSData hashtable with additional module metadata used by PowerShell.
    PrivateData = @{
        PSData = @{
            # Tags applied to this module. These help with module discovery in online galleries.
            Tags = @('Cleanup', 'Maintenance', 'PATH', 'Duplicates', 'TempFiles', 'Windows')

            # A URL to the license for this module.
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'

            # A URL to the main website for this project.
            ProjectUri = 'https://github.com/David-Martel/PC-AI'

            # ReleaseNotes of this module
            ReleaseNotes = @'
## 1.0.0
- Initial release
- Get-PathDuplicates: Analyze PATH for duplicates and non-existent entries
- Repair-MachinePath: Clean up PATH with backup support
- Find-DuplicateFiles: Detect duplicate files by hash
- Clear-TempFiles: Safe temp file cleanup with space reclamation reporting
'@
        }
        PCAI = @{
            Commands = @('cleanup')
        }
    }
}
