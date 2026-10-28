Set-StrictMode -Version Latest

$script:AggregateRoot = $PSScriptRoot
$script:ModuleRoot = Split-Path -Parent $script:AggregateRoot
$script:AggregateContract = Microsoft.PowerShell.Utility\Import-PowerShellDataFile -LiteralPath (Join-Path $PSScriptRoot 'PC-AI.psd1')
$script:ComponentExports = $script:AggregateContract.PrivateData.PcaiAggregate.ExportsByComponent
$script:ComponentOrder = @(
    'PC-AI.Common', 'PC-AI.Acceleration', 'PC-AI.Cleanup', 'PC-AI.CLI',
    'PC-AI.Evaluation', 'PC-AI.Hardware', 'PC-AI.LLM', 'PC-AI.Network',
    'PC-AI.Performance', 'PC-AI.USB', 'PC-AI.Virtualization', 'PcaiInference'
)
$script:ComponentModules = @{}
$script:ComponentManifests = @()

function Test-PcaiAggregatePathIdentity {
    param([string]$Left, [string]$Right)
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    [string]::Equals([IO.Path]::GetFullPath($Left), [IO.Path]::GetFullPath($Right), $comparison)
}

function Get-PcaiAggregateProducerPlan {
    $sourceRepository = Split-Path -Parent $script:ModuleRoot
    $sourceLayout = (Test-Path -LiteralPath (Join-Path $sourceRepository 'PC-AI.ps1') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $sourceRepository 'Tools/Install-PcaiDevModules.ps1') -PathType Leaf)
    foreach ($name in $script:ComponentOrder) {
        $directory = Join-Path $script:ModuleRoot $name
        $leaf = if ($name -eq 'PC-AI.Common') { "$name.psm1" } else { "$name.psd1" }
        $path = Join-Path $directory $leaf
        if ($name -eq 'PcaiInference' -and $sourceLayout -and -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $directory = $script:ModuleRoot
            $path = Join-Path $directory $leaf
        }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "PC-AI requires its admitted sibling component '$name': $path"
        }
        if (((Get-Item -LiteralPath $directory -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "PC-AI component '$name' must not escape its admitted module root through a reparse point."
        }
        $rootScript = $path
        if ([IO.Path]::GetExtension($path) -eq '.psd1') {
            $metadata = Microsoft.PowerShell.Utility\Import-PowerShellDataFile -LiteralPath $path
            if (-not $metadata.ContainsKey('RootModule') -or $metadata.RootModule -cne "$name.psm1") {
                throw "PC-AI component '$name' selects a foreign RootModule."
            }
            $rootScript = Join-Path $directory "$name.psm1"
            if (-not (Test-Path -LiteralPath $rootScript -PathType Leaf) -or
                ((Get-Item -LiteralPath $rootScript -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "PC-AI component '$name' has no admitted root script."
            }
        }
        [pscustomobject]@{
            Name = $name
            Manifest = $path
            Directory = $directory
            RootScript = $rootScript
            RelativePath = [IO.Path]::GetRelativePath($script:AggregateRoot, $path)
        }
    }
}

# Both current initializers accept Force. Common delegates to Acceleration, so
# select that implementation explicitly instead of depending on import order.
$script:AggregateInitializer = {
    [CmdletBinding()]
    [OutputType([bool])]
    param([switch]$Force)

    $producer = $script:ComponentModules['PC-AI.Acceleration']
    & $producer.ExportedFunctions['Initialize-PcaiNative'] @PSBoundParameters
}

function Import-PcaiAggregateProducers {
    param([switch]$Force)

    # Resolve every required local producer before importing any component. A
    # same-named module elsewhere on PSModulePath cannot fill an incomplete set.
    $plan = @(Get-PcaiAggregateProducerPlan)
    $modules = @{}
    foreach ($component in $plan) {
        $module = Microsoft.PowerShell.Core\Import-Module -Name $component.Manifest -Scope Local -PassThru -Force:$Force -ErrorAction Stop
        if ($module.Name -cne $component.Name -or
            -not (Test-PcaiAggregatePathIdentity $module.ModuleBase $component.Directory) -or
            -not (Test-PcaiAggregatePathIdentity $module.Path $component.RootScript)) {
            throw "PC-AI imported a foreign producer for '$($component.Name)'."
        }
        foreach ($name in $script:ComponentExports[$component.Name]) {
            if (-not $module.ExportedFunctions.ContainsKey($name)) {
                throw "PC-AI component '$($component.Name)' does not export required function '$name'."
            }
            # Retain the producer-bound script block and its parameter contract.
            # Import-Module's local function scope alone would lose these exports
            # when this loader returns. No component body is stored in this package.
            Set-Item -LiteralPath "Function:\script:$name" -Value $module.ExportedFunctions[$name].ScriptBlock
        }
        $modules[$component.Name] = $module
    }
    $script:ComponentModules = $modules
    $script:ComponentManifests = @($plan.RelativePath)
    Set-Item -LiteralPath Function:\script:Initialize-PcaiNative -Value $script:AggregateInitializer
}

function Import-PcaiComponentModules {
    <# .SYNOPSIS Reload the aggregate's admitted standalone components. #>
    [CmdletBinding()]
    param()

    # Retain the legacy explicit reload behavior, but refuse missing/foreign
    # producers rather than silently accepting an incomplete compatibility API.
    Import-PcaiAggregateProducers -Force
    Export-ModuleMember -Function $script:AggregateContract.FunctionsToExport -Cmdlet @() -Alias @()
}

function Get-PcaiReleaseInfo {
    <# .SYNOPSIS Return aggregate and retained configuration locations. #>
    [CmdletBinding()]
    param()

    [pscustomobject]@{
        ModuleRoot = $script:AggregateRoot
        ConfigPath = Join-Path $script:AggregateRoot 'Config'
        Components = $script:ComponentManifests
    }
}

Import-PcaiAggregateProducers
Export-ModuleMember -Function $script:AggregateContract.FunctionsToExport -Cmdlet @() -Alias @()
