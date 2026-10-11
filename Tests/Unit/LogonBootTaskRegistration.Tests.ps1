#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# The logon task used to launch a bare "pwsh.exe", resolved through the
# user's PATH at every logon, where user folders such as ~\bin come early.
BeforeAll {
    $script:Path = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools' 'SystemScripts' 'Machine' 'Register-SecretsBootTask.ps1'
    $script:Ast = [System.Management.Automation.Language.Parser]::ParseFile($script:Path, [ref]$null, [ref]$null)
}

Describe 'Register-SecretsBootTask' -Tag 'Unit', 'Windows' {
    It 'passes an absolute pwsh path to New-ScheduledTaskAction' {
        $calls = $script:Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'New-ScheduledTaskAction' }, $true)
        @($calls).Count | Should -Be 1
        $calls[0].Extent.Text | Should -Not -Match '-Execute\s+"?pwsh\.exe"?\s'
        $calls[0].Extent.Text | Should -Match '-Execute\s+\$pwshPath'
        (Get-Content -Raw $script:Path) | Should -Match "\`$pwshPath = Join-Path \`$PSHOME 'pwsh\.exe'"
    }
}
