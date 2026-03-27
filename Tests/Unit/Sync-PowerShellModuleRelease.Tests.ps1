#Requires -Version 7.0

Describe 'Sync-PowerShellModuleRelease' -Tag 'Unit', 'Fast', 'Portable' {
    It 'mirrors a promoted release bundle into requested module roots' {
        $tempRoot = Join-Path $env:TEMP ('pcai-release-sync-' + [guid]::NewGuid().ToString('N'))
        $repoRoot = Join-Path $tempRoot 'repo'
        $releaseRoot = Join-Path $repoRoot 'Release\PowerShell'
        $releaseModuleRoot = Join-Path $releaseRoot 'Demo.Module'
        $destinationOne = Join-Path $tempRoot 'OneDrive\Documents\PowerShell\Modules'
        $destinationTwo = Join-Path $tempRoot 'Documents\PowerShell\Modules'
        $scriptPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Sync-PowerShellModuleRelease.ps1'

        New-Item -ItemType Directory -Path $releaseModuleRoot -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $releaseModuleRoot 'Demo.Module.psd1') -Value '@{ RootModule = ''Demo.Module.psm1'' }' -Encoding utf8NoBOM
        Set-Content -LiteralPath (Join-Path $releaseModuleRoot 'Demo.Module.psm1') -Value 'function Get-DemoModule { ''ok'' }' -Encoding utf8NoBOM

        try {
            $result = & $scriptPath `
                -RepoRoot $repoRoot `
                -ModuleName 'Demo.Module' `
                -ReleaseRoot $releaseRoot `
                -DestinationRoots @($destinationOne, $destinationTwo) `
                -BuildMode Never `
                -SkipValidation `
                -Quiet

            Test-Path -LiteralPath (Join-Path $destinationOne 'Demo.Module\Demo.Module.psd1') | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $destinationTwo 'Demo.Module\Demo.Module.psm1') | Should -BeTrue
            $result.SyncResults.Count | Should -Be 2
            ($result.SyncResults | Where-Object DestinationRoot -eq $destinationOne).Updated | Should -BeTrue
        } finally {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
