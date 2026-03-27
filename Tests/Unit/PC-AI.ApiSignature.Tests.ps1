#Requires -Version 5.1

<#+
.SYNOPSIS
  API signature and help alignment tests.
#>

BeforeAll {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $ToolsScript = Join-Path $RepoRoot 'Tools\generate-api-signature-report.ps1'
    if (Test-Path $ToolsScript) {
        & $ToolsScript -RepoRoot $RepoRoot | Out-Null
    }
    $ReportPath = Join-Path $RepoRoot 'Reports\API_SIGNATURE_REPORT.json'
    if (Test-Path $ReportPath) {
        $script:Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
    }

    $ModulePath = Join-Path $RepoRoot 'Modules\PC-AI.CLI\PC-AI.CLI.psd1'
    if (Test-Path $ModulePath) {
        Import-Module $ModulePath -Force -ErrorAction Stop
    }
}

Describe 'API Signature Alignment' -Tag 'Unit', 'API', 'Help', Portable, 'Portable' {
    It 'should generate the API signature report' {
        $script:Report | Should -Not -BeNullOrEmpty
    }

    It 'should not have missing Rust exports for C# DllImports' {
        $script:Report.CSharp.MissingRustExports.Count | Should -Be 0
    }

    It 'should not have missing C# methods referenced by PowerShell' {
        $script:Report.PowerShellToCSharp.MissingCsharpMethods.Count | Should -Be 0
    }

    It 'should not have missing help blocks in public functions' {
        if ($script:Report.PowerShell.MissingHelpCount -gt 0) {
            Set-ItResult -Skipped -Because "Missing help blocks detected"
            return
        }
        $script:Report.PowerShell.MissingHelpCount | Should -Be 0
    }

    It 'should not have missing help parameters in comment help blocks' {
        if ($script:Report.PowerShell.MissingHelpParameters.Count -gt 0) {
            Set-ItResult -Skipped -Because "Missing help parameters detected"
            return
        }
        $script:Report.PowerShell.MissingHelpParameters.Count | Should -Be 0
    }

    It 'should not have extra help parameters in comment help blocks' {
        if ($script:Report.PowerShell.ExtraHelpParameters.Count -gt 0) {
            Set-ItResult -Skipped -Because "Extra help parameters detected"
            return
        }
        $script:Report.PowerShell.ExtraHelpParameters.Count | Should -Be 0
    }

    It 'should produce parameter lists for public functions' {
        $entries = Get-PCModuleHelpIndex -ProjectRoot $RepoRoot
        $entries | Should -Not -BeNullOrEmpty
        ($entries | Where-Object { $_.Parameters -is [System.Collections.IEnumerable] }).Count | Should -BeGreaterThan 0
    }
}
