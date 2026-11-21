#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: the minimal private profile is deployed to hosts (the NUC) that previously
# had no private profile at all and paid for it with a 1.7-17 s startup and a thrown import. These
# tests pin the properties that make it safe to deploy: it parses, does only cheap local work,
# orders PATH deterministically without dead entries, and keeps secret access on demand.
# See Config/PowerShell/Profile/README.md and Tests/Boot/ProfileBootstrap.Tests.ps1.

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:MinimalProfile = Join-Path $script:RepoRoot 'Config\PowerShell\Profile\Microsoft.PowerShell_profile.minimal.ps1'
    $script:Text = Get-Content -LiteralPath $script:MinimalProfile -Raw
    $tokens = $null
    $parseErrors = $null
    $script:Ast = [System.Management.Automation.Language.Parser]::ParseFile($script:MinimalProfile, [ref]$tokens, [ref]$parseErrors)
    $script:ParseErrors = @($parseErrors)

    function Invoke-MinimalProfile {
        param([Parameter(Mandatory)][string]$HomePath)
        $probe = @'
. $env:PCAI_TEST_MINIMAL
[Console]::Write(([pscustomobject]@{
    Path          = $env:Path
    PsModulePath  = $env:PSModulePath
    HasVaultFn    = [bool](Get-Command Connect-MachineVault -CommandType Function -ErrorAction SilentlyContinue)
    HasAgentEnvFn = [bool](Get-Command Import-MachineAgentEnvironment -CommandType Function -ErrorAction SilentlyContinue)
    LeakedVars    = @(Get-Variable -Name minimal* -ErrorAction SilentlyContinue).Count
    Modules       = @(Get-Module | Where-Object { $_.Name -notmatch '^(Microsoft\.PowerShell|PSReadLine)' } | ForEach-Object Name)
} | ConvertTo-Json -Compress))
'@
        $start = [Diagnostics.ProcessStartInfo]::new('pwsh')
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.RedirectStandardInput = $true
        foreach ($arg in '-NoLogo', '-NoProfile', '-Command', $probe) { $start.ArgumentList.Add($arg) }
        $start.Environment['USERPROFILE'] = $HomePath
        $start.Environment['HOME'] = $HomePath
        $start.Environment['PCAI_TEST_MINIMAL'] = $script:MinimalProfile
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            $process.Start() | Should -BeTrue
            $process.StandardInput.Close()
            $out = $process.StandardOutput.ReadToEndAsync()
            $err = $process.StandardError.ReadToEndAsync()
            $process.WaitForExit(60000) | Should -BeTrue
            [pscustomobject]@{ Stdout = $out.GetAwaiter().GetResult(); Stderr = $err.GetAwaiter().GetResult() }
        } finally {
            if ($process.Id -and -not $process.HasExited) { $process.Kill() }
            $process.Dispose()
        }
    }
}

Describe 'Minimal private profile' -Tag 'Unit', 'Boot', 'Portable' {
    It 'parses without errors' {
        $script:ParseErrors.Count | Should -Be 0
    }

    # Why: the profile runs in every interactive shell, so it must stay free of module imports,
    # network or process launches, secret reads and OneDrive dependence.
    It 'does only cheap local work' {
        $script:Text | Should -Not -Match '(?i)Import-Module|Invoke-WebRequest|Invoke-RestMethod|Start-Process|Invoke-Expression|OneDrive|\.bwdata|bw\.exe|bw sync'
    }

    It 'keeps secret access behind the on-demand Connect-MachineVault function' {
        $script:Text | Should -Match 'function Connect-MachineVault'
        $outside = $script:Ast.EndBlock.Statements | Where-Object { $_ -isnot [System.Management.Automation.Language.FunctionDefinitionAst] }
        ($outside | ForEach-Object { $_.Extent.Text }) -join "`n" | Should -Not -Match 'BW_SESSION|Initialize-BitwardenSessionFromBackends'
    }

    It 'puts existing preferred directories first, skips missing ones and leaves no duplicates' {
        $homePath = Join-Path $TestDrive 'min-home'
        foreach ($sub in 'bin', '.cargo\bin', '.local\bin') { [IO.Directory]::CreateDirectory((Join-Path $homePath $sub)) | Out-Null }
        $r = Invoke-MinimalProfile -HomePath $homePath
        $r.Stderr | Should -BeExactly ''
        $state = $r.Stdout | ConvertFrom-Json
        $entries = @($state.Path -split ';')
        $entries[0] | Should -BeExactly (Join-Path $homePath '.local\bin')
        $entries[1] | Should -BeExactly (Join-Path $homePath 'bin')
        $entries[2] | Should -BeExactly (Join-Path $homePath '.cargo\bin')
        $state.Path | Should -Not -Match ([regex]::Escape((Join-Path $homePath '.dotnet\tools')))
        $normalized = $entries | ForEach-Object { $_.TrimEnd('\').ToLowerInvariant() }
        ($normalized | Select-Object -Unique).Count | Should -Be $normalized.Count
        $state.HasVaultFn | Should -BeTrue
        $state.LeakedVars | Should -Be 0
        $state.Modules | Should -BeNullOrEmpty
    }

    It 'defines Import-MachineAgentEnvironment only when the machine helper exists' {
        $withHelper = Join-Path $TestDrive 'with-helper'
        [IO.Directory]::CreateDirectory((Join-Path $withHelper '.machine')) | Out-Null
        [IO.File]::WriteAllText((Join-Path $withHelper '.machine\Import-AgentEnvironment.ps1'), 'function Import-MachineAgentEnvironment { }', [Text.UTF8Encoding]::new($false))
        $withoutHelper = Join-Path $TestDrive 'without-helper'
        [IO.Directory]::CreateDirectory($withoutHelper) | Out-Null
        ((Invoke-MinimalProfile -HomePath $withHelper).Stdout | ConvertFrom-Json).HasAgentEnvFn | Should -BeTrue
        ((Invoke-MinimalProfile -HomePath $withoutHelper).Stdout | ConvertFrom-Json).HasAgentEnvFn | Should -BeFalse
    }
}
