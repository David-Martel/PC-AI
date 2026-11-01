#Requires -Modules Pester

$moduleRoot = Split-Path -Parent $PSScriptRoot
Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force

Describe 'Find-FilesFast native manifest fallback' {
    It 'returns real directories for a native no-ignore scan when native search is available' {
        if (-not (Get-Command Test-PcaiNativeAvailable -ErrorAction SilentlyContinue)) {
            Set-ItResult -Skipped -Because 'PC-AI native bridge is not exposed in this session'
            return
        }

        if (-not (Test-PcaiNativeAvailable)) {
            Set-ItResult -Skipped -Because 'PC-AI native search is not available on this machine'
            return
        }

        $testRepoRoot = if ($env:PCAI_ROOT) { $env:PCAI_ROOT } else { Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) }
        $result = Find-FilesFast -Path $testRepoRoot -Type directory -MaxDepth 1 -NoIgnore -PreferNative

        $result | Should -Not -BeNullOrEmpty
        $result.Count | Should -BeGreaterThan 0
        $result[0] | Should -BeOfType ([System.IO.DirectoryInfo])
    }
}
