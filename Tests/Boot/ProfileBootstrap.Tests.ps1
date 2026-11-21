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
        $script:SourceText | Should -Match '(?s)if \(Microsoft\.PowerShell\.Management\\Test-Path -LiteralPath \$localProfile\) \{\s*\. \$localProfile\s*\}\s*else \{'
        $script:SourceText | Should -Match 'Canonical local PowerShell profile not found: \$localProfile'
        $script:SourceText | Should -Not -Match '(?i)SetEnvironmentVariable|BW_SESSION|Invoke-Expression'
    }

    # Why: the only module the public bootstrap may import is the WinGet feedback provider, and
    # only lazily from the command-not-found hook. An eager or unqualified import re-introduces
    # the 1.4-31 s cloud-module stall and the throw on hosts without winget.
    It 'imports only the WinGet feedback module, module-qualified, inside the lazy hook' {
        $imports = @($script:ProfileAst.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.CommandAst] -and
            $node.GetCommandName() -match '(?i)Import-Module$'
        }, $true))
        $imports.Count | Should -Be 1
        $imports[0].GetCommandName() | Should -BeExactly 'Microsoft.PowerShell.Core\Import-Module'
        $imports[0].Extent.Text | Should -Match 'Microsoft\.WinGet\.CommandNotFound'
        $hook = $script:ProfileAst.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
            $node.Left.Extent.Text -match 'CommandNotFoundAction'
        }, $true) | Select-Object -First 1
        $hook | Should -Not -BeNullOrEmpty
        $imports[0].Extent.StartOffset | Should -BeGreaterThan $hook.Extent.StartOffset
        $imports[0].Extent.EndOffset | Should -BeLessThan $hook.Extent.EndOffset
    }

    # Why: command discovery misses during private startup (a Get-Command probe on work measured about
    # six) fire a CommandNotFoundAction handler. Registering the WinGet hook before delegation made each
    # interactive start pay the cloud-module import, so it must come after the private profile and must
    # not replace a handler the private profile installed.
    It 'registers the lazy hook after private delegation and only when no handler exists' {
        $hook = $script:ProfileAst.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
            $node.Left.Extent.Text -match 'CommandNotFoundAction'
        }, $true) | Select-Object -First 1
        $delegation = $script:SourceText.IndexOf('. $localProfile')
        $delegation | Should -BeGreaterThan 0
        $hook.Extent.StartOffset | Should -BeGreaterThan $delegation
        $guard = $script:SourceText.Substring($delegation, $hook.Extent.StartOffset - $delegation)
        $guard | Should -Match '-and \$null -eq \$ExecutionContext\.InvokeCommand\.CommandNotFoundAction'
    }

    It 'ends every path with return so appended installer blocks are unreachable' {
        $last = $script:ProfileAst.EndBlock.Statements[-1]
        $last | Should -BeOfType ([System.Management.Automation.Language.ReturnStatementAst])
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

Describe 'Public profile bootstrap startup fast paths' -Tag 'Unit', 'Boot' {
    # Evaluated at discovery time so -Skip can use it.
    $wingetPresent = [bool](Get-Command winget -CommandType Application -ErrorAction SilentlyContinue) -and
        [bool](Get-Module -ListAvailable Microsoft.WinGet.CommandNotFound)

    BeforeAll {
        # PowerToys appends this eager import after whatever bootstrap it finds. The fixture
        # reproduces that layout so the tests prove the bootstrap makes it unreachable.
        $script:PowerToysBlock = @'

#f45873b3-b655-43a6-b217-97c00aa0db58 PowerToys CommandNotFound module

Import-Module -Name Microsoft.WinGet.CommandNotFound
#f45873b3-b655-43a6-b217-97c00aa0db58
'@
        $script:Fixture = Join-Path $TestDrive 'bootstrap-with-powertoys.ps1'
        [IO.File]::WriteAllText($script:Fixture, $script:SourceText + $script:PowerToysBlock, [Text.UTF8Encoding]::new($false))

        function Invoke-BootstrapChild {
            param(
                [Parameter(Mandatory)][string]$HomePath,
                [hashtable]$Environment = @{}
            )
            $probe = @'
. $env:PCAI_TEST_BOOTSTRAP
[Console]::Write(([pscustomobject]@{
    WinGetLoaded = [bool](Get-Module Microsoft.WinGet.CommandNotFound)
    ProbeLoaded  = [bool](Get-Variable -Name PCAI_BOOTSTRAP_PROBE -Scope Global -ErrorAction SilentlyContinue)
    ProfileRoot  = [string]$env:POWERSHELL_PROFILE_ROOT
} | ConvertTo-Json -Compress))
'@
            $start = [Diagnostics.ProcessStartInfo]::new('pwsh')
            $start.UseShellExecute = $false
            $start.RedirectStandardOutput = $true
            $start.RedirectStandardError = $true
            $start.RedirectStandardInput = $true
            foreach ($arg in '-NoLogo', '-NoProfile', '-Command', $probe) { $start.ArgumentList.Add($arg) }
            foreach ($name in 'SSH_CONNECTION', 'POWERSHELL_PROFILE_ROOT', 'PCAI_PROFILE_FAST', 'PCAI_DISABLE_PSREADLINE') {
                $start.Environment.Remove($name) | Out-Null
            }
            $start.Environment['USERPROFILE'] = $HomePath
            $start.Environment['HOME'] = $HomePath
            $start.Environment['PCAI_TEST_BOOTSTRAP'] = $script:Fixture
            foreach ($key in $Environment.Keys) { $start.Environment[$key] = $Environment[$key] }
            $process = [Diagnostics.Process]::new()
            $process.StartInfo = $start
            try {
                $process.Start() | Should -BeTrue
                $process.StandardInput.Close()
                $out = $process.StandardOutput.ReadToEndAsync()
                $err = $process.StandardError.ReadToEndAsync()
                $process.WaitForExit(60000) | Should -BeTrue
                [pscustomobject]@{
                    Stdout = $out.GetAwaiter().GetResult()
                    Stderr = $err.GetAwaiter().GetResult()
                }
            } finally {
                if ($process.Id -and -not $process.HasExited) { $process.Kill() }
                $process.Dispose()
            }
        }

        function New-HomeFixture {
            param([string]$Name, [switch]$WithPrivateProfile)
            $homePath = Join-Path $TestDrive $Name
            [IO.Directory]::CreateDirectory($homePath) | Out-Null
            if ($WithPrivateProfile) {
                $dir = Join-Path $homePath '.config\powershell'
                [IO.Directory]::CreateDirectory($dir) | Out-Null
                [IO.File]::WriteAllText((Join-Path $dir 'Microsoft.PowerShell_profile.ps1'), '$global:PCAI_BOOTSTRAP_PROBE = 1', [Text.UTF8Encoding]::new($false))
            }
            $homePath
        }
    }

    # Why: a Claude agent-home has no private profile, so the old bootstrap fell through to the
    # PowerToys import (31 s measured on the laptop, 17 s on the NUC). The agent-home path must
    # be silent and instant.
    It 'returns silently from an agent-home without importing WinGet or delegating' {
        $homePath = New-HomeFixture 'agent-homes\h1\home'
        $r = Invoke-BootstrapChild -HomePath $homePath
        $r.Stderr | Should -BeExactly ''
        $state = $r.Stdout | ConvertFrom-Json
        $state.WinGetLoaded | Should -BeFalse
        $state.ProbeLoaded | Should -BeFalse
        $state.ProfileRoot | Should -BeExactly ''
    }

    # Why: the regression behind the NUC failure. With no private profile the bootstrap warns and
    # stops; the trailing PowerToys import must never run (winget is absent there, so it throws).
    It 'warns once and never reaches an appended PowerToys import when the private profile is missing' {
        $homePath = New-HomeFixture 'plain-home'
        $r = Invoke-BootstrapChild -HomePath $homePath
        # The child host writes the warning stream to stdout, so inspect both pipes.
        $all = $r.Stdout + $r.Stderr
        ([regex]::Matches($all, 'Canonical local PowerShell profile not found')).Count | Should -Be 1
        $all | Should -Not -Match 'CommandNotFound'
        ($all.Substring($all.IndexOf('{')) | ConvertFrom-Json).WinGetLoaded | Should -BeFalse
    }

    It 'still delegates to the private profile and skips the appended import' {
        $homePath = New-HomeFixture 'delegating-home' -WithPrivateProfile
        $r = Invoke-BootstrapChild -HomePath $homePath
        $r.Stderr | Should -BeExactly ''
        $state = $r.Stdout | ConvertFrom-Json
        $state.ProbeLoaded | Should -BeTrue
        $state.WinGetLoaded | Should -BeFalse
    }

    # Why: an explicit, documented opt-out for tools that want a bare shell without -NoProfile.
    It 'honors PCAI_PROFILE_FAST=1 by skipping delegation silently' {
        $homePath = New-HomeFixture 'fast-home' -WithPrivateProfile
        $r = Invoke-BootstrapChild -HomePath $homePath -Environment @{ PCAI_PROFILE_FAST = '1' }
        $r.Stderr | Should -BeExactly ''
        ($r.Stdout | ConvertFrom-Json).ProbeLoaded | Should -BeFalse
    }

    Context 'lazy WinGet command-not-found hook' {
        BeforeAll {
            $hook = $script:ProfileAst.FindAll({ param($node)
                $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
                $node.Left.Extent.Text -match 'CommandNotFoundAction'
            }, $true) | Select-Object -First 1
            $script:HookText = $hook.Right.Expression.ScriptBlock.Extent.Text
            $script:HookRunner = Join-Path $TestDrive 'hook-runner.ps1'
            $runner = @'
param([string]$HookText)
$ExecutionContext.InvokeCommand.CommandNotFoundAction = [scriptblock]::Create($HookText.Trim().TrimStart('{').TrimEnd('}'))
try { & 'pcai-no-such-command-zzz' } catch {}
[Console]::Write(([pscustomobject]@{
    WinGetLoaded = [bool](Get-Module Microsoft.WinGet.CommandNotFound)
    HookCleared  = ($null -eq $ExecutionContext.InvokeCommand.CommandNotFoundAction)
} | ConvertTo-Json -Compress))
'@
            [IO.File]::WriteAllText($script:HookRunner, $runner, [Text.UTF8Encoding]::new($false))

            function Invoke-HookChild {
                param([string]$PathOverride)
                $start = [Diagnostics.ProcessStartInfo]::new('pwsh')
                $start.UseShellExecute = $false
                $start.RedirectStandardOutput = $true
                $start.RedirectStandardError = $true
                $start.RedirectStandardInput = $true
                foreach ($arg in '-NoLogo', '-NoProfile', '-File', $script:HookRunner, '-HookText', $script:HookText) { $start.ArgumentList.Add($arg) }
                if ($PathOverride) { $start.Environment['PATH'] = $PathOverride }
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

        # Why: the NUC has the module installed but no winget; importing there throws.
        It 'does not import the module when winget is not on PATH' {
            $r = Invoke-HookChild -PathOverride $PSHOME
            $r.Stderr | Should -BeExactly ''
            $state = $r.Stdout | ConvertFrom-Json
            $state.WinGetLoaded | Should -BeFalse
            $state.HookCleared | Should -BeTrue
        }

        It 'imports the module once on the first unresolved command when winget exists' -Skip:(-not $wingetPresent) {
            $r = Invoke-HookChild
            $r.Stderr | Should -BeExactly ''
            ($r.Stdout | ConvertFrom-Json).WinGetLoaded | Should -BeTrue
        }
    }
}
