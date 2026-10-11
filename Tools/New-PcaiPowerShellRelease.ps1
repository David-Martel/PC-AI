#Requires -Version 7.0

<#
.SYNOPSIS
Creates a unified PowerShell release with explicit component exports.
.DESCRIPTION
Packages component modules, configuration and two allowlisted diagnostic scripts
without importing or executing them. Component
manifests supply explicit public exports; script-only modules use literal
Export-ModuleMember declarations parsed with the PowerShell AST. Ambiguous or
wildcard declarations fail before writes rather than silently changing the API.
NestedModules performs initial component loading; the root module does not load
them a second time. DryRun and WhatIf perform no version lookup or writes.
.PARAMETER OutputRoot
Directory containing the generated PC-AI module folder.
.PARAMETER SourceRoot
Optional source checkout. Defaults to the repository containing this script.
.PARAMETER ModuleVersion
Optional explicit numeric version. Otherwise use repository build metadata.
.PARAMETER Clean
Remove the generated PC-AI output folder before packaging.
.PARAMETER DryRun
Describe packaging without reading component exports or changing files.
.PARAMETER Help
Show help, also available with -h and --help.
.EXAMPLE
.\Tools\New-PcaiPowerShellRelease.ps1 -DryRun
.EXAMPLE
.\Tools\New-PcaiPowerShellRelease.ps1 -OutputRoot C:\Temp\PcaiRelease -ModuleVersion 1.0.0
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\Release\PowerShell'),
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidatePattern('^\d+\.\d+\.\d+(?:\.\d+)?$')]
    [string]$ModuleVersion,
    [switch]$Clean,
    [switch]$DryRun,
    [Alias('h', '-help')] [switch]$Help,
    [Parameter(ValueFromRemainingArguments)] [string[]]$RemainingArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($RemainingArguments) {
    foreach ($argument in $RemainingArguments) {
        if ($argument -eq '--help') { $Help = $true }
        else { throw "Unsupported argument: $argument" }
    }
}
if ($Help) { Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed; return }
if ($DryRun -or $WhatIfPreference) {
    [pscustomobject]@{ Status = 'DryRun'; OutputRoot = $OutputRoot; ModuleVersion = $ModuleVersion; Clean = $Clean.IsPresent; WritesPerformed = $false }
    return
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if ($IsWindows) {
    foreach ($part in $OutputRoot.Split([char[]]'\/')) {
        $trimmed = $part.TrimEnd('.', ' ')
        if ($trimmed -eq '$null' -or $trimmed -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw 'OutputRoot contains a reserved Windows path component.'
        }
    }
    if ($OutputRoot.Substring(2).Contains(':')) { throw 'OutputRoot cannot contain alternate stream syntax.' }
}

function Get-PcaiDeclaredExport {
    param([string]$ModulePath, [string]$Kind)
    $scriptPath = $ModulePath
    if ([System.IO.Path]::GetExtension($ModulePath) -eq '.psd1') {
        $data = Microsoft.PowerShell.Utility\Import-PowerShellDataFile -LiteralPath $ModulePath
        $key = if ($Kind -eq 'Alias') { 'AliasesToExport' } else { "${Kind}sToExport" }
        if ($data.ContainsKey($key)) {
            $names = @($data[$key])
            foreach ($name in $names) {
                if ([System.Management.Automation.WildcardPattern]::ContainsWildcardCharacters([string]$name)) {
                    throw "Explicit $Kind exports are required in $ModulePath. Resolve wildcard exports before packaging."
                }
            }
            return $names
        }
        if (-not $data.ContainsKey('RootModule')) { throw "RootModule is required to resolve $Kind exports from $ModulePath." }
        $scriptPath = Join-Path (Split-Path -Parent $ModulePath) $data.RootModule
    }
    if ([System.IO.Path]::GetExtension($scriptPath) -ne '.psm1') {
        throw "Cannot statically resolve $Kind exports from $scriptPath; declare them in its manifest."
    }
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) { throw "Cannot parse component module $scriptPath." }
    $commands = @($ast.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.CommandAst] -and
        $node.GetCommandName() -match '(^|\\)Export-ModuleMember$'
    }, $true))
    if ($commands.Count -eq 0) { throw "Explicit Export-ModuleMember declarations are required in $scriptPath." }
    $names = [System.Collections.Generic.List[string]]::new()
    foreach ($command in $commands) {
        $elements = $command.CommandElements
        foreach ($element in $elements) {
            if ($element -is [System.Management.Automation.Language.VariableExpressionAst] -and $element.Splatted) {
                throw "Cannot statically resolve splatted exports in $scriptPath."
            }
        }
        for ($index = 1; $index -lt $elements.Count; $index++) {
            $parameter = $elements[$index]
            if ($parameter -isnot [System.Management.Automation.Language.CommandParameterAst] -or $parameter.ParameterName -ne $Kind) { continue }
            $argument = $parameter.Argument
            if ($null -eq $argument) {
                $index++
                if ($index -ge $elements.Count) { throw "Missing $Kind export declaration in $scriptPath." }
                $argument = $elements[$index]
            }
            try { $values = @($argument.SafeGetValue()) }
            catch { throw "Literal $Kind exports are required in $scriptPath; dynamic declarations cannot be packaged safely." }
            foreach ($value in $values) {
                if ($value -isnot [string] -or [System.Management.Automation.WildcardPattern]::ContainsWildcardCharacters($value)) {
                    throw "Explicit literal $Kind export names are required in $scriptPath."
                }
                $names.Add($value)
            }
        }
    }
    return $names.ToArray()
}

function ConvertTo-PcaiExportLiteral {
    param([string[]]$Names)
    if ($Names.Count -eq 0) { return '@()' }
    return "@($(($Names | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ', '))"
}

function Test-PcaiReleasePathWithin {
    param([string]$Path, [string]$Root)
    $separator = [System.IO.Path]::DirectorySeparatorChar
    $comparison = if ($IsWindows) { [System.StringComparison]::OrdinalIgnoreCase } else { [System.StringComparison]::Ordinal }
    $prefix = $Root.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + $separator
    return $Path.Equals($Root, $comparison) -or $Path.StartsWith($prefix, $comparison)
}

function Assert-PcaiReleaseDirectory {
    param([string]$Path, [switch]$InspectDescendants, [switch]$MustExist)
    if ($MustExist -and -not [System.IO.Directory]::Exists($Path)) {
        throw "Required release source directory not found: $Path"
    }
    $cursor = $Path
    while ($cursor) {
        $attributes = $null
        try { $attributes = [System.IO.File]::GetAttributes($cursor) }
        catch [System.IO.FileNotFoundException] { $attributes = $null }
        catch [System.IO.DirectoryNotFoundException] { $attributes = $null }
        if ($null -ne $attributes) {
            if (($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Release path contains a reparse point: $cursor"
            }
            if (($attributes -band [System.IO.FileAttributes]::Directory) -eq 0) {
                throw "Release directory path is an existing file: $cursor"
            }
        }
        $cursor = [System.IO.Path]::GetDirectoryName($cursor)
    }
    if ($InspectDescendants -and [System.IO.Directory]::Exists($Path)) {
        foreach ($entry in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $Path -Force -Recurse) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Release payload contains a reparse point: $($entry.FullName)"
            }
        }
    }
}

$projectRoot = [System.IO.Path]::GetFullPath($SourceRoot)
$buildVersionScript = Join-Path $projectRoot 'Tools\Get-BuildVersion.ps1'
$sourceModulesRoot = Join-Path $projectRoot 'Modules'
$sourceConfigRoot = Join-Path $projectRoot 'Config'
$sourceToolsRoot = Join-Path $projectRoot 'Tools'
$diagnosticScripts = @('Get-PcaiMachineCapabilities.ps1', 'Test-PcaiMediaToolchain.ps1')
$promptDocuments = @('CHAT.md', 'DIAGNOSE.md', 'DIAGNOSE_LOGIC.md')
$presentPromptDocuments = @()
$targetModuleRoot = Join-Path $OutputRoot 'PC-AI'
$targetModulesRoot = Join-Path $targetModuleRoot 'Modules'
$targetConfigRoot = Join-Path $targetModuleRoot 'Config'
$targetToolsRoot = Join-Path $targetModuleRoot 'Tools'
$buildVersion = $null

# A normal release lives below SourceRoot/Release, but must never replace the
# checkout itself or any source payload. Reject aliases before lexical checks.
Assert-PcaiReleaseDirectory -Path $projectRoot -MustExist
Assert-PcaiReleaseDirectory -Path $sourceModulesRoot -InspectDescendants -MustExist
Assert-PcaiReleaseDirectory -Path $sourceConfigRoot -InspectDescendants -MustExist
Assert-PcaiReleaseDirectory -Path $sourceToolsRoot -MustExist
Assert-PcaiReleaseDirectory -Path $OutputRoot
Assert-PcaiReleaseDirectory -Path $targetModuleRoot -InspectDescendants
Assert-PcaiReleaseDirectory -Path $targetToolsRoot
if (Test-PcaiReleasePathWithin -Path $projectRoot -Root $targetModuleRoot) {
    throw 'Generated release target overlaps the source checkout.'
}
foreach ($protectedRoot in @($sourceModulesRoot, $sourceConfigRoot, $sourceToolsRoot)) {
    if ((Test-PcaiReleasePathWithin -Path $targetModuleRoot -Root $protectedRoot) -or
        (Test-PcaiReleasePathWithin -Path $protectedRoot -Root $targetModuleRoot)) {
        throw "Generated release target overlaps a source payload: $protectedRoot"
    }
}

# Validate the fixed script payload before cleaning an existing release. Other
# source Tools files are intentionally excluded from this distribution.
foreach ($fileName in $diagnosticScripts) {
    $sourcePath = Join-Path $sourceToolsRoot $fileName
    if (-not [System.IO.File]::Exists($sourcePath)) {
        throw "Required release diagnostic script not found: $sourcePath"
    }
    if (([System.IO.File]::GetAttributes($sourcePath) -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Release diagnostic script contains a reparse point: $sourcePath"
    }
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($sourcePath, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) { throw "Cannot parse release diagnostic script: $sourcePath" }
}
foreach ($fileName in $promptDocuments) {
    $sourcePath = Join-Path $projectRoot $fileName
    if (-not (Test-Path -LiteralPath $sourcePath)) { continue }
    if (([System.IO.File]::GetAttributes($sourcePath) -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Release prompt document contains a reparse point: $sourcePath"
    }
    if (-not [System.IO.File]::Exists($sourcePath)) { throw "Release prompt document is not a file: $sourcePath" }
    $presentPromptDocuments += $fileName
}

if (-not $ModuleVersion) {
    $ModuleVersion = '1.0.0'
    if (Test-Path -LiteralPath $buildVersionScript) {
        try {
            $buildVersion = & $buildVersionScript -Quiet
            if ($buildVersion -and $buildVersion.SemVer) {
                $ModuleVersion = $buildVersion.SemVer
            }
        } catch {
            Write-Verbose "Optional build metadata unavailable; using fallback module version '$ModuleVersion': $($_.Exception.Message)"
        }
    }
}

$moduleFolders = @(
    'PC-AI.Acceleration',
    'PC-AI.Cleanup',
    'PC-AI.CLI',
    'PC-AI.Common',
    'PC-AI.Evaluation',
    'PC-AI.Hardware',
    'PC-AI.LLM',
    'PC-AI.Network',
    'PC-AI.Performance',
    'PC-AI.USB',
    'PC-AI.Virtualization'
)

$singleModuleFiles = @(
    'PcaiInference.psd1',
    'PcaiInference.psm1'
)

$nestedModules = @('Modules\PC-AI.Common\PC-AI.Common.psm1') + @(
    $moduleFolders | Where-Object { $_ -ne 'PC-AI.Common' } | ForEach-Object { "Modules\$_\$_.psd1" }
) + @('Modules\PcaiInference.psd1')
$exports = @{}
foreach ($kind in @('Function', 'Cmdlet', 'Alias')) {
    $names = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    if ($kind -eq 'Function') {
        [void]$names.Add('Get-PcaiReleaseInfo')
        [void]$names.Add('Import-PcaiComponentModules')
    }
    foreach ($relativePath in $nestedModules) {
        $sourcePath = Join-Path $sourceModulesRoot ($relativePath.Substring('Modules\'.Length))
        foreach ($name in @(Get-PcaiDeclaredExport -ModulePath $sourcePath -Kind $kind)) { [void]$names.Add($name) }
    }
    $exports[$kind] = @($names | Sort-Object)
}
$functionLiteral = ConvertTo-PcaiExportLiteral -Names $exports.Function
$cmdletLiteral = ConvertTo-PcaiExportLiteral -Names $exports.Cmdlet
$aliasLiteral = ConvertTo-PcaiExportLiteral -Names $exports.Alias

if (-not $PSCmdlet.ShouldProcess($targetModuleRoot, 'Generate unified PowerShell release')) { return }
# Every recursive deletion below stays inside this exact generated module folder.
$outputPrefix = $OutputRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not [System.IO.Path]::GetFullPath($targetModuleRoot).StartsWith($outputPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Generated module directory is outside OutputRoot.'
}

if ($Clean -and (Test-Path -LiteralPath $targetModuleRoot)) {
    Remove-Item -LiteralPath $targetModuleRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $targetModulesRoot -Force | Out-Null
New-Item -ItemType Directory -Path $targetConfigRoot -Force | Out-Null
New-Item -ItemType Directory -Path $targetToolsRoot -Force | Out-Null
foreach ($fileName in $diagnosticScripts) {
    Copy-Item -LiteralPath (Join-Path $sourceToolsRoot $fileName) -Destination (Join-Path $targetToolsRoot $fileName) -Force
}
foreach ($fileName in $presentPromptDocuments) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $fileName) -Destination (Join-Path $targetModuleRoot $fileName) -Force
}

foreach ($folder in $moduleFolders) {
    $sourcePath = Join-Path $sourceModulesRoot $folder
    $targetPath = Join-Path $targetModulesRoot $folder
    if (-not [System.IO.Path]::GetFullPath($targetPath).StartsWith($targetModuleRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Component output is outside the generated module directory.'
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required module folder not found: $sourcePath"
    }
    if (Test-Path -LiteralPath $targetPath) {
        Remove-Item -LiteralPath $targetPath -Recurse -Force
    }
    Copy-Item -LiteralPath $sourcePath -Destination $targetModulesRoot -Recurse -Force
}

foreach ($fileName in $singleModuleFiles) {
    $sourcePath = Join-Path $sourceModulesRoot $fileName
    if (Test-Path -LiteralPath $sourcePath) {
        Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $targetModulesRoot $fileName) -Force
    }
}

Get-ChildItem -LiteralPath $sourceConfigRoot -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $targetConfigRoot -Recurse -Force
}

$rootPsm1Path = Join-Path $targetModuleRoot 'PC-AI.psm1'
$rootPsm1 = @'
Set-StrictMode -Version Latest

$script:ModuleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:ComponentManifests = @(
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

function Import-PcaiComponentModules {
    [CmdletBinding()]
    param()

    foreach ($relativePath in $script:ComponentManifests) {
        $manifestPath = Join-Path $script:ModuleRoot $relativePath
        if (Test-Path -LiteralPath $manifestPath) {
            Import-Module -Name $manifestPath -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }
}

function Get-PcaiReleaseInfo {
    [CmdletBinding()]
    param()

    [PSCustomObject]@{
        ModuleRoot = $script:ModuleRoot
        ConfigPath = Join-Path $script:ModuleRoot 'Config'
        Components = $script:ComponentManifests
    }
}

Export-ModuleMember -Function __FUNCTION_EXPORTS__ -Cmdlet __CMDLET_EXPORTS__ -Alias __ALIAS_EXPORTS__
'@
$rootPsm1 = $rootPsm1.Replace('__FUNCTION_EXPORTS__', $functionLiteral).Replace('__CMDLET_EXPORTS__', $cmdletLiteral).Replace('__ALIAS_EXPORTS__', $aliasLiteral)
Set-Content -LiteralPath $rootPsm1Path -Value $rootPsm1 -Encoding UTF8

$rootPsd1Path = Join-Path $targetModuleRoot 'PC-AI.psd1'
$manifestBody = @"
@{
    RootModule = 'PC-AI.psm1'
    ModuleVersion = '$ModuleVersion'
    GUID = '02a9dcaf-0ac8-49ca-bcdd-61f39a12c721'
    Author = 'PC_AI Project'
    CompanyName = 'PC_AI'
    Copyright = '(c) 2025-2026 PC_AI Project'
    Description = 'PC-AI unified release module bundling diagnostics, acceleration, and AI tooling components.'
    PowerShellVersion = '7.0'
    CompatiblePSEditions = @('Core', 'Desktop')
    NestedModules = @(
$(($nestedModules | ForEach-Object { "        '$_'" }) -join ",`r`n")
    )
    FunctionsToExport = $functionLiteral
    CmdletsToExport = $cmdletLiteral
    VariablesToExport = @()
    AliasesToExport = $aliasLiteral
    PrivateData = @{
        PSData = @{
            Tags = @('pc-ai', 'diagnostics', 'acceleration', 'llm')
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
            LicenseUri = 'https://github.com/David-Martel/PC-AI/blob/main/LICENSE'
        }
    }
}
"@
Set-Content -LiteralPath $rootPsd1Path -Value $manifestBody -Encoding UTF8

$readmePath = Join-Path $targetModuleRoot 'README.md'
$readme = @"
# PC-AI PowerShell Release Module

This folder is a distribution-oriented module layout for PC-AI.

## Import

\`\`\`powershell
Import-Module .\PC-AI.psd1 -Force
\`\`\`

## Included Components

$(($moduleFolders + 'PcaiInference') | ForEach-Object { "- $_" } | Out-String)
"@
Set-Content -LiteralPath $readmePath -Value $readme -Encoding UTF8

$licenseSource = Join-Path $projectRoot 'LICENSE'
if (Test-Path -LiteralPath $licenseSource) {
    Copy-Item -LiteralPath $licenseSource -Destination (Join-Path $targetModuleRoot 'LICENSE') -Force
}

$releaseInfoPath = Join-Path $targetModuleRoot 'RELEASE_INFO.json'
$releaseInfo = [ordered]@{
    ModuleName = 'PC-AI'
    ModuleVersion = $ModuleVersion
    SourceRoot = $projectRoot
    GeneratedAtUtc = [DateTime]::UtcNow.ToString('o')
    DiagnosticScripts = @($diagnosticScripts | ForEach-Object { "Tools\$_" })
    PromptDocuments = $presentPromptDocuments
}
if ($buildVersion) {
    $releaseInfo.BuildVersion = $buildVersion
}
$releaseInfo | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $releaseInfoPath -Encoding UTF8

$result = [PSCustomObject]@{
    OutputRoot = $OutputRoot
    ModuleRoot = $targetModuleRoot
    ModuleManifest = $rootPsd1Path
    ModuleVersion = $ModuleVersion
}

$result
