[CmdletBinding()]
param(
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\Release\PowerShell'),
    [string]$ModuleVersion = '1.0.0',
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceModulesRoot = Join-Path $projectRoot 'Modules'
$sourceConfigRoot = Join-Path $projectRoot 'Config'
$targetModuleRoot = Join-Path $OutputRoot 'PC-AI'
$targetModulesRoot = Join-Path $targetModuleRoot 'Modules'
$targetConfigRoot = Join-Path $targetModuleRoot 'Config'

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

if ($Clean -and (Test-Path -LiteralPath $targetModuleRoot)) {
    Remove-Item -LiteralPath $targetModuleRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $targetModulesRoot -Force | Out-Null
New-Item -ItemType Directory -Path $targetConfigRoot -Force | Out-Null

foreach ($folder in $moduleFolders) {
    $sourcePath = Join-Path $sourceModulesRoot $folder
    $targetPath = Join-Path $targetModulesRoot $folder
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

Import-PcaiComponentModules

Export-ModuleMember -Function * -Alias *
'@
Set-Content -LiteralPath $rootPsm1Path -Value $rootPsm1 -Encoding UTF8

$rootPsd1Path = Join-Path $targetModuleRoot 'PC-AI.psd1'
$nestedModules = @(
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
    FunctionsToExport = '*'
    CmdletsToExport = '*'
    VariablesToExport = @()
    AliasesToExport = '*'
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

$result = [PSCustomObject]@{
    OutputRoot = $OutputRoot
    ModuleRoot = $targetModuleRoot
    ModuleManifest = $rootPsd1Path
    ModuleVersion = $ModuleVersion
}

$result
