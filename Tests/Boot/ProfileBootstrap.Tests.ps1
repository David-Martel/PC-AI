#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ProfileSource = Join-Path $script:RepoRoot 'Config\PowerShell\Profile\Microsoft.PowerShell_profile.ps1'
    $script:SourceText = Get-Content -LiteralPath $script:ProfileSource -Raw
    $tokens = $null
    $parseErrors = $null
    $script:ProfileAst = [System.Management.Automation.Language.Parser]::ParseFile($script:ProfileSource, [ref]$tokens, [ref]$parseErrors)
    $script:ParseErrors = @($parseErrors)
}

Describe 'Public profile bootstrap command resolution' -Tag 'Unit', 'Boot', 'Portable' {
    It 'parses without errors' {
        $script:ParseErrors.Count | Should -Be 0
    }

    It 'qualifies every builtin command used before private profile delegation' {
        $commands = @($script:ProfileAst.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.CommandAst]
        }, $true) | ForEach-Object { $_.GetCommandName() } | Where-Object { $_ })
        foreach ($command in $commands) {
            $command | Should -Match '^Microsoft\.PowerShell\.(Core|Management|Utility)\\'
        }
        $commands | Should -Contain 'Microsoft.PowerShell.Management\Join-Path'
        $commands | Should -Contain 'Microsoft.PowerShell.Management\Test-Path'
        $commands | Should -Contain 'Microsoft.PowerShell.Utility\Write-Warning'
        $commands | Should -Contain 'Microsoft.PowerShell.Core\Get-Module'
        $commands | Should -Contain 'Microsoft.PowerShell.Core\Remove-Module'
    }

    It 'preserves guarded delegation and missing-profile diagnostics' {
        $delegations = @($script:ProfileAst.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.CommandAst] -and
            $node.InvocationOperator -eq [System.Management.Automation.Language.TokenKind]::Dot
        }, $true))
        $delegations.Count | Should -Be 1
        $delegations[0].Extent.Text | Should -BeExactly '. $localProfile'
        $guardedBlock = $delegations[0].Parent.Parent
        $guardedBlock.Parent | Should -BeOfType ([System.Management.Automation.Language.IfStatementAst])
        $guardedBlock.Parent.Clauses[0].Item1.Extent.Text | Should -BeExactly 'Microsoft.PowerShell.Management\Test-Path -LiteralPath $localProfile'
        [object]::ReferenceEquals($guardedBlock, $guardedBlock.Parent.Clauses[0].Item2) | Should -BeTrue
        $guardedBlock.Statements[-1] | Should -BeOfType ([System.Management.Automation.Language.ReturnStatementAst])
        $script:SourceText | Should -Match 'Canonical local PowerShell profile not found: \$localProfile'
        $script:SourceText | Should -Not -Match '(?i)Import-Module|SetEnvironmentVariable|BW_SESSION|Invoke-Expression'
    }

    It 'silences SSH protocol output in a real <Mode> child process' -ForEach @(
        @{ Mode = 'Command' }, @{ Mode = 'EncodedCommand' }, @{ Mode = 'File' }
    ) {
        param($Mode)
        $probe = ". '" + $script:ProfileSource.Replace("'", "''") + "'; [Console]::Write('PROBE:' + `$env:POWERSHELL_PROFILE_ROOT)"
        $start = [System.Diagnostics.ProcessStartInfo]::new('pwsh')
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.ArgumentList.Add('-NoLogo')
        $start.ArgumentList.Add('-NoProfile')
        $start.Environment['SSH_CONNECTION'] = 'protocol-test'
        $start.Environment['POWERSHELL_PROFILE_ROOT'] = 'guard-probe'
        if ($Mode -eq 'EncodedCommand') {
            $start.ArgumentList.Add('-EncodedCommand')
            $start.ArgumentList.Add([Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($probe)))
        } elseif ($Mode -eq 'File') {
            $fixture = Join-Path $TestDrive 'ssh-profile-probe.ps1'
            Set-Content -LiteralPath $fixture -Value $probe -Encoding utf8NoBOM
            $start.ArgumentList.Add('-File')
            $start.ArgumentList.Add($fixture)
        } else {
            $start.ArgumentList.Add('-Command')
            $start.ArgumentList.Add($probe)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            $process.Start() | Should -BeTrue
            $output = $process.StandardOutput.ReadToEndAsync()
            $errors = $process.StandardError.ReadToEndAsync()
            $process.WaitForExit(15000) | Should -BeTrue
            $process.ExitCode | Should -Be 0
            $output.GetAwaiter().GetResult() | Should -BeExactly 'PROBE:guard-probe'
            $errors.GetAwaiter().GetResult() | Should -BeExactly ''
        } finally {
            if ($process.Id -and -not $process.HasExited) { $process.Kill() }
            $process.Dispose()
        }
    }
}
