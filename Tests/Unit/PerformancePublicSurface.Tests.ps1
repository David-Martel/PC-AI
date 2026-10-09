param([string]$PerformanceManifest)

BeforeAll {
    if (-not $PerformanceManifest) {
        $PerformanceManifest = Join-Path $PSScriptRoot '../../Modules/PC-AI.Performance/PC-AI.Performance.psd1'
    }
    $script:ManifestContract = Import-PowerShellDataFile -LiteralPath $PerformanceManifest
    $script:SurfaceModule = Import-Module -Name $PerformanceManifest -Force -PassThru -ErrorAction Stop
}

Describe 'Declared performance consumer command availability' -Tag 'Unit', 'Performance', 'Portable' {
    It 'publishes the declared <Name> command to module consumers' -TestCases @(
        @{Name='Get-PcaiMemoryPressure'},
        @{Name='Get-PcaiProcessCategories'},
        @{Name='Get-PcaiOptimizationPlan'}
    ) {
        param($Name)
        $script:ManifestContract.FunctionsToExport | Should -Contain $Name
        $script:SurfaceModule.ExportedFunctions.Keys | Should -Contain $Name
        $command = Get-Command -Name "$($script:SurfaceModule.Name)\$Name" -ErrorAction Stop
        $command.Module.Path | Should -Be $script:SurfaceModule.Path
    }
}
