#Requires -Version 5.1

<#
.SYNOPSIS
    PC-AI CLI Utilities Module — dynamic help extraction and command discovery.

.DESCRIPTION
    Provides cmdlets for introspecting all loaded PC-AI modules, building a
    structured command map, and resolving CLI arguments at runtime. Supports
    both pure-PowerShell help parsing and accelerated native extraction via
    PcaiNative.dll (HelpExtractor) when available.

    Exported functions:
      Get-PCCommandMap      - Build a hashtable mapping command names to their
                              module, synopsis, parameters, and help metadata.
      Get-PCCommandModules  - List all PC-AI module names currently in scope.
      Get-PCCommandList     - Return a flat list of all exported command names
                              across loaded PC-AI modules.
      Get-PCCommandSummary  - Return a condensed name+synopsis list suitable
                              for display in a CLI help screen.
      Get-PCModuleHelpIndex - Build a per-module index of all help entries
                              (function name -> synopsis).
      Get-PCModuleHelpEntry - Retrieve the full comment-based help block for a
                              single named function.
      ConvertTo-PCArgumentMap - Convert a raw argument string into a
                               structured parameter hashtable.
      Resolve-PCArguments   - Resolve parsed arguments against a command's
                              declared parameter set, applying defaults and
                              validation.

    Native acceleration (PcaiNative.dll — HelpExtractor):
      On import the module checks for bin\PcaiNative.dll at the project root.
      When present, [PcaiNative.HelpExtractor] is used for faster bulk help
      extraction; otherwise the module falls back to pure PowerShell parsing.
      The DLL is optional — all functions work without it.

    Dependencies:
      - PowerShell 5.1 or later
      - PC-AI modules imported into the session (for Get-PCCommandMap/List)
      - bin\PcaiNative.dll (optional, for native HelpExtractor acceleration)
#>

$script:ModuleRoot = $PSScriptRoot
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $script:ModuleRoot)
$script:HelpExtractorType = $null
$script:CommandMapCache = $null
$script:CommandMapCacheKey = $null

$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object { . $_.FullName }
}
$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object { . $_.FullName }
}

Export-ModuleMember -Function @(
    'Get-PCModuleHelpIndex'
    'Get-PCModuleHelpEntry'
    'Get-PCCommandMap'
    'Get-PCCommandModules'
    'Get-PCCommandList'
    'Get-PCCommandSummary'
    'ConvertTo-PCArgumentMap'
    'Resolve-PCArguments'
)
