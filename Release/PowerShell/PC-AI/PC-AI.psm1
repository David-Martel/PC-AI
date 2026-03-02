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
