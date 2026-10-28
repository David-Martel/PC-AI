#Requires -Version 7.0
param([string]$SyntaxCheckerPath, [string]$SyntaxConfigPath)

BeforeAll {
    $script:SyntaxRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $configPath = if ($SyntaxConfigPath) { $SyntaxConfigPath } else {
        Join-Path $script:SyntaxRepo '.pre-commit-config.yaml'
    }
    $configLines = [IO.File]::ReadAllLines($configPath)
    $hookLine = [Array]::FindIndex($configLines, [Predicate[string]] {
            param($line)
            $line -match '^\s*- id: powershell-syntax\s*$'
        })
    if ($hookLine -lt 0) { throw 'Configured powershell-syntax hook is missing.' }
    $entryLine = $null
    for ($index = $hookLine + 1; $index -lt $configLines.Length; $index++) {
        if ($configLines[$index] -match '^\s*- id:') { break }
        if ($configLines[$index] -match '^\s*entry:') { $entryLine = $configLines[$index]; break }
    }
    # This is a bounded argv boundary check for the maintained single-line entry,
    # not a general YAML/shell parser. Refuse another shape before launching files.
    if ($entryLine -notmatch '^\s*entry:\s*(?<exe>pwsh)\s+(?<profile>-NoProfile)\s+(?<mode>-File)\s+(?<helper>Tools/[^\s]+\.ps1)\s*$') {
        throw 'Configured syntax hook must pass filenames through pwsh -NoProfile -File, without command evaluation.'
    }
    $script:SyntaxEntryArguments = @($Matches.profile, $Matches.mode)
    $script:SyntaxChecker = if ($SyntaxCheckerPath) { $SyntaxCheckerPath } else {
        Join-Path $script:SyntaxRepo $Matches.helper
    }
    $script:SyntaxPwsh = (Get-Command $Matches.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    $script:SyntaxFixtureRoot = Join-Path $TestDrive 'literal space [sources]'
    [void][IO.Directory]::CreateDirectory($script:SyntaxFixtureRoot)

    function New-SyntaxFixture {
        param([string]$Name, [string]$Content)
        $path = Join-Path $script:SyntaxFixtureRoot $Name
        [IO.File]::WriteAllText($path, $Content)
        return $path
    }

    function Invoke-SyntaxChecker {
        param([string[]]$SourcePath)
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = $script:SyntaxPwsh
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in $script:SyntaxEntryArguments + @($script:SyntaxChecker) + $SourcePath) {
            [void]$start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            if (-not $process.Start()) { throw 'Syntax-check child did not start.' }
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(15000)) {
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Syntax-check child exceeded 15 seconds.'
            }
            [pscustomobject]@{
                ExitCode = $process.ExitCode
                StdOut   = $stdout.GetAwaiter().GetResult()
                StdErr   = $stderr.GetAwaiter().GetResult()
            }
        }
        finally {
            $process.Dispose()
        }
    }
}

Describe 'PowerShell staged-source syntax safety' -Tag 'Unit', 'Portable', 'PreCommitSafety' {
    # Protects: A syntax check must not execute valid staged scripts.
    # Detects: The old -Command filename-appending boundary writes a marker instead of only parsing.
    # Needs: PowerShell 7, Pester 5 and disposable TestDrive; no modules, network or hardware.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1; .pre-commit-config.yaml powershell-syntax.
    It 'accepts valid staged source without executing its write-marker statements' {
        $marker = Join-Path $script:SyntaxFixtureRoot 'valid-source-executed.marker'
        $fixture = New-SyntaxFixture 'valid-marker.ps1' @'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'valid-source-executed.marker'), 'unexpected execution')
'@
        $result = Invoke-SyntaxChecker @($fixture)
        Test-Path -LiteralPath $marker | Should -BeFalse
        $result.ExitCode | Should -Be 0
    }

    # Protects: Tracked dot-prefixed source remains eligible for syntax validation.
    # Detects: Get-Item without Force treats existing hidden Unix source as absent.
    # Needs: PowerShell 7, Pester 5 and TestDrive; the dot-file form runs on every platform.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1 literal source discovery.
    It 'accepts dot-prefixed staged source without executing its statements' {
        $marker = Join-Path $script:SyntaxFixtureRoot 'dot-source-executed.marker'
        $fixture = New-SyntaxFixture '.staged-source.ps1' @'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'dot-source-executed.marker'), 'unexpected execution')
'@
        $result = Invoke-SyntaxChecker @($fixture)
        Test-Path -LiteralPath $marker | Should -BeFalse
        $result.ExitCode | Should -Be 0
    }

    # Protects: Windows hidden source receives the same syntax-only treatment as visible source.
    # Detects: Get-Item without Force rejects a staged file with the Hidden attribute.
    # Needs: Windows file attributes, PowerShell 7 and Pester 5; explicitly skipped on other platforms.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1 literal source discovery.
    It 'accepts Windows hidden staged source without executing its statements' -Skip:(-not $IsWindows) {
        $marker = Join-Path $script:SyntaxFixtureRoot 'hidden-source-executed.marker'
        $fixture = New-SyntaxFixture 'hidden-source.ps1' @'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'hidden-source-executed.marker'), 'unexpected execution')
'@
        $attributes = [IO.File]::GetAttributes($fixture)
        try {
            [IO.File]::SetAttributes($fixture, ($attributes -bor [IO.FileAttributes]::Hidden))
            $result = Invoke-SyntaxChecker @($fixture)
            Test-Path -LiteralPath $marker | Should -BeFalse
            $result.ExitCode | Should -Be 0
        }
        finally {
            [IO.File]::SetAttributes($fixture, $attributes)
        }
    }

    # Protects: Malformed source is refused before any staged statements can execute.
    # Detects: Success-only guards or execution-based checks miss/refuse source for the wrong reason.
    # Needs: PowerShell 7, Pester 5 and disposable TestDrive.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1.
    It 'rejects invalid script syntax without executing its leading write-marker statement' {
        $marker = Join-Path $script:SyntaxFixtureRoot 'invalid-source-executed.marker'
        $fixture = New-SyntaxFixture 'invalid-marker.ps1' @'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'invalid-source-executed.marker'), 'unexpected execution')
function Broken {
'@
        $result = Invoke-SyntaxChecker @($fixture)
        $result.ExitCode | Should -Not -Be 0
        $result.StdErr | Should -Match 'Missing closing'
        Test-Path -LiteralPath $marker | Should -BeFalse
    }

    # Protects: Removed or misspelled staged paths cannot silently pass the guard.
    # Detects: Ignoring absent files permits an unchecked source set.
    # Needs: PowerShell 7, Pester 5 and a nonexistent path under TestDrive.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1.
    It 'rejects a missing source file and names the refused path' {
        $missing = Join-Path $script:SyntaxFixtureRoot 'missing-source.ps1'
        $result = Invoke-SyntaxChecker @($missing)
        $result.ExitCode | Should -Not -Be 0
        $result.StdErr | Should -Match 'missing-source\.ps1'
    }

    # Protects: These explicit ps1/psm1/psd1 cases remain usable through literal path binding.
    # Detects: Splitting whitespace, treating brackets as wildcards, or dropping supported extensions.
    # Needs: PowerShell 7, Pester 5 and TestDrive paths containing spaces and brackets.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1; .pre-commit-config.yaml extension selector.
    It 'accepts valid <Extension> source at a literal space and bracket path' -TestCases @(
        @{ Extension = 'ps1'; Content = '$value = 1' }
        @{ Extension = 'psm1'; Content = 'function Get-Value { 1 }' }
        @{ Extension = 'psd1'; Content = "@{ Value = '[literal]' }" }
    ) {
        param($Extension, $Content)
        $fixture = New-SyntaxFixture ("valid-source.$Extension") $Content
        (Invoke-SyntaxChecker @($fixture)).ExitCode | Should -Be 0
    }

    # Protects: Explicit module/data cases receive the same refusal as malformed scripts.
    # Detects: An extension filter that skips psm1/psd1 or an always-success replacement.
    # Needs: PowerShell 7, Pester 5 and disposable TestDrive.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1.
    It 'rejects malformed <Extension> source' -TestCases @(
        @{ Extension = 'psm1'; Content = 'function Broken {' }
        @{ Extension = 'psd1'; Content = '@{' }
    ) {
        param($Extension, $Content)
        $fixture = New-SyntaxFixture ("invalid-source.$Extension") $Content
        $result = Invoke-SyntaxChecker @($fixture)
        $result.ExitCode | Should -Not -Be 0
        $result.StdErr | Should -Match 'Missing closing'
    }

    # Protects: One invalid staged file fails the batch even when a later file is valid.
    # Detects: Last-file exit-code overwrite, filename dropping, or early success.
    # Needs: PowerShell 7, Pester 5 and two disposable source fixtures.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1 foreach/aggregate failure boundary.
    It 'refuses a mixed batch rather than accepting the last valid file' {
        $invalid = New-SyntaxFixture 'batch-invalid.ps1' 'function Broken {'
        $valid = New-SyntaxFixture 'batch-valid.ps1' '$value = 1'
        $result = Invoke-SyntaxChecker @($invalid, $valid)
        $result.ExitCode | Should -Not -Be 0
        $result.StdErr | Should -Match 'batch-invalid\.ps1'
    }

    # Protects: Both declared help entrypoints work without source access or child evaluation.
    # Detects: Parameter binding consumes help as a missing file or requires interactive input.
    # Needs: PowerShell 7 and Pester 5; no source fixtures.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1 help contract.
    It 'shows usage for <Option> without requiring source files' -TestCases @(
        @{ Option = '-h' }
        @{ Option = '--help' }
    ) {
        param($Option)
        $result = Invoke-SyntaxChecker @($Option)
        $result.ExitCode | Should -Be 0
        $result.StdOut | Should -Match '^Usage:'
    }

    # Protects: An empty invocation cannot masquerade as checked source.
    # Detects: A missing filename handoff reports success while checking nothing.
    # Needs: PowerShell 7 and Pester 5; no source fixtures.
    # Breadcrumb: Tools/Test-PowerShellSyntax.ps1 empty-input refusal.
    It 'rejects an invocation with no source files' {
        (Invoke-SyntaxChecker @()).ExitCode | Should -Not -Be 0
    }
}
