#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ProfileSource = Join-Path $script:RepoRoot 'Config\PowerShell\Profile\Microsoft.PowerShell_profile.ps1'
    $script:BundledReadLine = Join-Path $PSHOME 'Modules\PSReadLine\PSReadLine.psd1'

    function Invoke-ProfileSafetyProbe {
        param(
            [AllowNull()][string]$OptOut,
            [switch]$MissingProcessSetting,
            [string]$ProfilePath = $script:ProfileSource
        )

        $probe = @'
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module $env:PCAI_TEST_PSREADLINE -ErrorAction Stop
$before = [bool](Get-Module PSReadLine)
$redirected = [Console]::IsInputRedirected
# Execute the exact public guard statements, stopping before private startup.
# Never dot-source the full bootstrap or change HOME/USERPROFILE for this probe.
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:PCAI_TEST_PROFILE, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Public profile did not parse' }
$boundary = $ast.EndBlock.Statements | Where-Object {
    $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $_.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
    $_.Left.VariablePath.UserPath -eq 'localProfileRoot'
} | Select-Object -First 1
if ($null -eq $boundary) { throw 'Private profile delegation boundary was not found' }
$guardStatements = @($ast.EndBlock.Statements | Where-Object { $_.Extent.EndOffset -le $boundary.Extent.StartOffset })
if ($guardStatements.Count -ne 4) { throw 'Unexpected public profile guard structure' }
$guardText = ($guardStatements | ForEach-Object { $_.Extent.Text }) -join [Environment]::NewLine
# A nested invocation proves the fallback survives outside the profile scope.
$effectiveOptOut = & { . ([scriptblock]::Create($guardText)); $psReadLineOptOut }
$after = [bool](Get-Module PSReadLine)
$reader = Get-Command PSConsoleHostReadLine -CommandType Function
$first = PSConsoleHostReadLine
$second = PSConsoleHostReadLine
$third = PSConsoleHostReadLine
$eof = PSConsoleHostReadLine
[Console]::Write(([pscustomobject]@{
    Before = $before; After = $after; Redirected = $redirected
    ReaderModule = $reader.ModuleName; EffectiveOptOut = $effectiveOptOut
    GuardStatementCount = $guardStatements.Count
    First = $first; Second = $second; Third = $third; Eof = $eof
} | ConvertTo-Json -Compress))
'@
        $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
        $start.UseShellExecute = $false
        $start.RedirectStandardInput = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.ArgumentList.Add('-NoLogo')
        $start.ArgumentList.Add('-NoProfile')
        $start.ArgumentList.Add('-NonInteractive')
        $start.ArgumentList.Add('-Command')
        $start.ArgumentList.Add($probe)
        $start.Environment.Remove('SSH_CONNECTION') | Out-Null
        if ($MissingProcessSetting) {
            $start.Environment.Remove('PCAI_DISABLE_PSREADLINE') | Out-Null
        } else {
            $start.Environment['PCAI_DISABLE_PSREADLINE'] = $OptOut
        }
        $start.Environment['PCAI_TEST_PROFILE'] = $ProfilePath
        $start.Environment['PCAI_TEST_PSREADLINE'] = $script:BundledReadLine
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            $process.Start() | Should -BeTrue
            $output = $process.StandardOutput.ReadToEndAsync()
            $errors = $process.StandardError.ReadToEndAsync()
            $process.StandardInput.WriteLine('first line')
            $process.StandardInput.WriteLine('')
            $process.StandardInput.WriteLine('Unicode café')
            $process.StandardInput.Close()
            $process.WaitForExit(15000) | Should -BeTrue
            $process.ExitCode | Should -Be 0
            $errors.GetAwaiter().GetResult() | Should -BeExactly ''
            $output.GetAwaiter().GetResult() | ConvertFrom-Json
        } finally {
            if ($process.Id -and -not $process.HasExited) { $process.Kill() }
            $process.Dispose()
        }
    }
}

Describe 'PowerShell crash profile console safety' -Tag 'Unit', 'Boot' {
    It 'removes a real preloaded PSReadLine module and preserves line input with opt-out <OptOut>' -ForEach @(
        @{ OptOut = '0' }, @{ OptOut = '1' }
    ) {
        param($OptOut)

        $result = Invoke-ProfileSafetyProbe -OptOut $OptOut
        $result.Before | Should -BeTrue
        $result.GuardStatementCount | Should -Be 4
        $result.Redirected | Should -BeTrue
        $result.After | Should -BeFalse
        $result.ReaderModule | Should -BeExactly ''
        $result.EffectiveOptOut | Should -BeExactly $OptOut
        $result.First | Should -BeExactly 'first line'
        $result.Second | Should -BeExactly ''
        $result.Third | Should -BeExactly 'Unicode café'
        $result.Eof | Should -BeNullOrEmpty
    }

    It 'reads the actual persisted user setting when the child process environment lacks an override' {
        # Reading current user state avoids temporarily changing real user settings.
        $persisted = [Environment]::GetEnvironmentVariable('PCAI_DISABLE_PSREADLINE', 'User')
        $result = Invoke-ProfileSafetyProbe -MissingProcessSetting
        $result.EffectiveOptOut | Should -Be $persisted
        $result.Before | Should -BeTrue
        $result.After | Should -BeFalse
        $result.First | Should -BeExactly 'first line'
        $result.Third | Should -BeExactly 'Unicode café'
    }

    It 'never executes private canonical startup after the public guard boundary' {
        # Keep the public guards intact and trap the first private startup
        # statement. This temporary fixture never writes or sources a real profile.
        $fixturePath = Join-Path $TestDrive 'private-startup-trap.ps1'
        $profileText = [System.IO.File]::ReadAllText($script:ProfileSource)
        $tokens = $null
        $parseErrors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseInput($profileText, [ref]$tokens, [ref]$parseErrors)
        $boundary = $ast.EndBlock.Statements | Where-Object {
            $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and
            $_.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
            $_.Left.VariablePath.UserPath -eq 'localProfileRoot'
        } | Select-Object -First 1
        $trap = '$localProfileRoot = $(throw ''PRIVATE_STARTUP_EXECUTED'')'
        $fixtureText = $profileText.Remove($boundary.Extent.StartOffset, $boundary.Extent.Text.Length).Insert($boundary.Extent.StartOffset, $trap)
        [System.IO.File]::WriteAllText($fixturePath, $fixtureText)
        $result = Invoke-ProfileSafetyProbe -OptOut '1' -ProfilePath $fixturePath
        $result.GuardStatementCount | Should -Be 4
        $result.Before | Should -BeTrue
        $result.After | Should -BeFalse
        $result.First | Should -BeExactly 'first line'
        $result.Third | Should -BeExactly 'Unicode café'
        $result.Eof | Should -BeNullOrEmpty
    }
}
