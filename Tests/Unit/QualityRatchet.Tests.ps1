#Requires -Version 7.0
# Controls for the static-quality ratchet (Tools/Quality/Invoke-QualityRatchet.ps1 and
# Tools/Quality/PcaiPsAstChecks.psm1).
#
# Why this file exists: the ratchet is only worth having if it can fail. Every Describe below plants a
# known-bad pattern in a throwaway tree under $TestDrive and asserts that the tool counts it, and that a
# baseline lower than the planted count produces exit 1. Without these controls a scanner that silently
# stopped matching would read as a clean, improving repository.
# Code under test: Tools/Quality/Invoke-QualityRatchet.ps1, Tools/Quality/PcaiPsAstChecks.psm1,
# sgconfig.ratchet.yml, PSScriptAnalyzerSettings.ratchet.psd1, rules/rust/ffi/*.yml (rule self-tests in
# rules/tests/rust-ffi/).
# Skips: tests that need ast-grep or PSScriptAnalyzer skip (visibly) when the tool is not installed; the
# missing-tool controls do not need either tool and always run.
Set-StrictMode -Version Latest

# Tool detection runs at discovery time as well, because -Skip expressions are evaluated before BeforeAll.
$script:HaveAstGrep = [bool](Get-Command ast-grep -CommandType Application -ErrorAction SilentlyContinue)
if (-not $script:HaveAstGrep) {
    $sgDiscovery = Get-Command sg -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sgDiscovery) { $script:HaveAstGrep = [bool]((& $sgDiscovery.Source --version) -match '^\s*ast-grep') }
}
$script:HavePssa = [bool](Get-Module -ListAvailable PSScriptAnalyzer)

BeforeAll {
    $script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $script:RatchetScript = Join-Path $script:RepoRoot 'Tools/Quality/Invoke-QualityRatchet.ps1'
    $script:Module = Join-Path $script:RepoRoot 'Tools/Quality/PcaiPsAstChecks.psm1'
    Import-Module $script:Module -Force
    $script:Pwsh = (Get-Process -Id $PID).Path

    $script:HaveAstGrep = [bool](Get-Command ast-grep -CommandType Application -ErrorAction SilentlyContinue)
    if (-not $script:HaveAstGrep) {
        $sgCommand = Get-Command sg -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($sgCommand) { $script:HaveAstGrep = (& $sgCommand.Source --version) -match '^\s*ast-grep' }
    }
    $script:HavePssa = [bool](Get-Module -ListAvailable PSScriptAnalyzer)
    $script:SgPath = $null
    $script:SgVersion = $null
    foreach ($candidate in 'ast-grep', 'sg') {
        $found = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $found) { continue }
        $versionText = (& $found.Source --version) -join ' '
        if ($versionText -match '^\s*ast-grep\s+(\d+\.\d+\.\d+)') { $script:SgPath = $found.Source; $script:SgVersion = $Matches[1]; break }
    }

    function script:Write-FixtureFile {
        param([string]$Root, [string]$Relative, [string]$Text)
        $path = Join-Path $Root $Relative
        [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $path))
        [System.IO.File]::WriteAllText($path, $Text, [System.Text.UTF8Encoding]::new($false))
    }

    # One planted instance of every PowerShell check, one prod and one test Rust unwrap, one C# catch-all.
    function script:New-RatchetFixture {
        param([string]$Root)
        script:Write-FixtureFile $Root 'Modules/Planted.psm1' @'
function Invoke-Planted {
    try { Get-Item -Path 'C:\does-not-exist' } catch { }
    Set-Item -Path 'HKCU:\Software\PcaiFixture' -Value 1 -ErrorAction SilentlyContinue
    git status
}
function Write-Error { param($Message) }
function Invoke-Clean {
    git status
    if ($LASTEXITCODE -ne 0) { throw 'git failed' }
}
'@
        script:Write-FixtureFile $Root 'Tests/Unit/Planted.Tests.ps1' "try { 1 } catch { }`n"
        script:Write-FixtureFile $Root 'Native/lib/src/lib.rs' @'
pub fn prod(o: Option<u8>) -> u8 {
    o.unwrap()
}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        let v = Some(1u8).unwrap();
        assert_eq!(v, 1);
    }
}
'@
        script:Write-FixtureFile $Root 'Native/App/Planted.cs' @'
class Planted {
    int Parse(string s) {
        try { return int.Parse(s); }
        catch (Exception) { return 0; }
    }
}
'@
    }

    # An inert ast-grep stand-in. --version and the canary probe (run from a pcai-ratchet-probe temp directory)
    # are forwarded to the real binary; every other scan misbehaves as PCAI_FAKE_SG_MODE says.
    function script:New-FakeAstGrep {
        param([string]$Directory)
        [void][System.IO.Directory]::CreateDirectory($Directory)
        if ($IsWindows) {
            $path = Join-Path $Directory 'fake-ast-grep.cmd'
            $body = @(
                '@echo off',
                ('if "%1"=="--version" (echo ast-grep ' + $script:SgVersion + '& exit /b 0)'),
                ('echo %CD% | findstr /C:"pcai-ratchet-probe" >nul && ("' + $script:SgPath + '" %* & exit /b %ERRORLEVEL%)'),
                'if "%PCAI_FAKE_SG_MODE%"=="claim" (echo Error: 3 error^(s^) found in code. 1>&2 & exit /b 1)',
                'exit /b 1'
            ) -join "`r`n"
        } else {
            $path = Join-Path $Directory 'fake-ast-grep'
            $body = @(
                '#!/bin/sh',
                ('if [ "$1" = "--version" ]; then echo "ast-grep ' + $script:SgVersion + '"; exit 0; fi'),
                ('case "$PWD" in *pcai-ratchet-probe*) exec "' + $script:SgPath + '" "$@";; esac'),
                'if [ "$PCAI_FAKE_SG_MODE" = "claim" ]; then echo "Error: 3 error(s) found in code." >&2; fi',
                'exit 1'
            ) -join "`n"
        }
        [System.IO.File]::WriteAllText($path, $body + "`n")
        if (-not $IsWindows) { chmod +x $path }
        return $path
    }

    function script:Invoke-Ratchet {
        param([string]$Fixture, [string]$OutDir, [string[]]$Extra = @())
        $arguments = @('-NoProfile', '-File', $script:RatchetScript, '-RepoRoot', $Fixture,
            '-BaselinePath', (Join-Path $OutDir 'baseline.json'), '-OutputPath', (Join-Path $OutDir 'result.json')) + $Extra
        $output = @(& $script:Pwsh @arguments 2>&1)
        $exit = $LASTEXITCODE
        $resultPath = Join-Path $OutDir 'result.json'
        $json = if (Test-Path -LiteralPath $resultPath) { Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -AsHashtable } else { $null }
        return [pscustomobject]@{ ExitCode = $exit; Output = ($output -join "`n"); Result = $json }
    }

    function script:Get-TreeHash {
        param([string]$Root)
        Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName |
            ForEach-Object { '{0} {1}' -f $_.FullName.Substring($Root.Length), (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    }
}

# Protects: the PowerShell checks that PSScriptAnalyzer and ast-grep cannot express (swallowed errors,
# silenced state changes, unchecked native exit codes, shadowed built-ins, Invoke-Expression).
# Detects: a check that stops matching, or starts flagging the correct form.
Describe 'PowerShell AST checks' -Tag 'Unit', 'Quality', 'Portable' {
    BeforeAll {
        function script:Get-Rules {
            param([string]$Code)
            $path = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + '.ps1')
            [System.IO.File]::WriteAllText($path, $Code)
            @(Get-PcaiPsAstFinding -Path $path | ForEach-Object { $_.Rule })
        }
    }

    It 'counts an empty catch block' {
        script:Get-Rules 'try { 1 } catch { }' | Should -Contain 'empty-catch'
    }

    It 'does not count a catch block that handles the error' {
        script:Get-Rules 'try { 1 } catch { Write-Warning $_ }' | Should -Not -Contain 'empty-catch'
    }

    It 'counts -ErrorAction SilentlyContinue on a state-changing verb in every spelling' -TestCases @(
        @{ Code = "Set-Item -Path x -ErrorAction SilentlyContinue" }
        @{ Code = "Remove-Item x -EA SilentlyContinue" }
        @{ Code = "Stop-Service x -ErrorAction:SilentlyContinue" }
        @{ Code = "New-Item x -ErrorA 'SilentlyContinue'" }
    ) {
        param($Code)
        script:Get-Rules $Code | Should -Contain 'silent-continue-state-changing'
    }

    It 'does not count SilentlyContinue on a read-only verb or a different action' {
        script:Get-Rules 'Get-Item x -ErrorAction SilentlyContinue' | Should -Not -Contain 'silent-continue-state-changing'
        script:Get-Rules 'Set-Item x -ErrorAction Stop' | Should -Not -Contain 'silent-continue-state-changing'
        script:Get-Rules 'Set-Item x -ErrorVariable e' | Should -Not -Contain 'silent-continue-state-changing'
    }

    It 'counts a function that shadows a built-in cmdlet' {
        script:Get-Rules 'function Write-Error { param($m) }' | Should -Contain 'shadowed-builtin-command'
        script:Get-Rules 'function Get-PcaiSomethingUnique { }' | Should -Not -Contain 'shadowed-builtin-command'
    }

    It 'counts a native executable with no later exit-code check' {
        script:Get-Rules "function F { git status }" | Should -Contain 'native-exe-exit-code-unchecked'
        script:Get-Rules "function F { & 'C:\tools\thing.exe' --x }" | Should -Contain 'native-exe-exit-code-unchecked'
    }

    It 'accepts a native executable whose exit code is read afterwards in the same script block' {
        script:Get-Rules "function F { git status; if (`$LASTEXITCODE -ne 0) { throw 'x' } }" | Should -Not -Contain 'native-exe-exit-code-unchecked'
        script:Get-Rules "function F { git status; if (-not `$?) { throw 'x' } }" | Should -Not -Contain 'native-exe-exit-code-unchecked'
    }

    It 'does not accept an exit-code read that happens before the call or in another function' {
        script:Get-Rules "function F { if (`$LASTEXITCODE) { 1 }; git status }" | Should -Contain 'native-exe-exit-code-unchecked'
        script:Get-Rules "function A { git status }`nfunction B { `$LASTEXITCODE }" | Should -Contain 'native-exe-exit-code-unchecked'
    }

    It 'trusts an effective $PSNativeCommandUseErrorActionPreference = $true assignment before the call' {
        script:Get-Rules "`$PSNativeCommandUseErrorActionPreference = `$true`ngit status" | Should -Not -Contain 'native-exe-exit-code-unchecked'
        script:Get-Rules "`$global:PSNativeCommandUseErrorActionPreference = `$true`nfunction F { git status }" | Should -Not -Contain 'native-exe-exit-code-unchecked'
        script:Get-Rules "function F { `$PSNativeCommandUseErrorActionPreference = `$true; git status }" | Should -Not -Contain 'native-exe-exit-code-unchecked'
    }

    # Protects: the dominance rule. An assignment exempts a call only when it is an unconditional statement that
    # runs before the call: in the same statement list, or straight-line in an enclosing script block.
    It 'trusts an assignment that shares a statement list with the call, or is straight-line in an enclosing block' -TestCases @(
        @{ Name = 'same try block'; Code = "try { `$PSNativeCommandUseErrorActionPreference = `$true; git status } catch { Write-Warning `$_ }" }
        @{ Name = 'same if branch'; Code = "if (`$x) { `$PSNativeCommandUseErrorActionPreference = `$true`n git status }" }
        @{ Name = 'top-level, call nested in a branch later'; Code = "`$PSNativeCommandUseErrorActionPreference = `$true`nif (`$x) { git status }" }
        @{ Name = 'function body, call in a loop later'; Code = "function F { `$PSNativeCommandUseErrorActionPreference = `$true`n foreach (`$i in 1..2) { git status } }" }
    ) {
        param($Name, $Code)
        script:Get-Rules $Code | Should -Not -Contain 'native-exe-exit-code-unchecked'
    }

    # Protects: the exemption. A mere mention of the variable used to exempt every native call in the file.
    # Detects: $false assignments, plain reads, a later assignment, an overridden assignment and an assignment
    # in an unrelated scope each wrongly exempting an unchecked call.
    It 'does not let <Name> exempt an unchecked native call' -TestCases @(
        @{ Name = 'a $false assignment'; Code = "`$PSNativeCommandUseErrorActionPreference = `$false`ngit status" }
        @{ Name = 'a read-only reference'; Code = "`$null = `$PSNativeCommandUseErrorActionPreference`ngit status" }
        @{ Name = 'an assignment that comes after the call'; Code = "git status`n`$PSNativeCommandUseErrorActionPreference = `$true" }
        @{ Name = 'a $true assignment that is switched off before the call'; Code = "`$PSNativeCommandUseErrorActionPreference = `$true`n`$PSNativeCommandUseErrorActionPreference = `$false`ngit status" }
        @{ Name = 'an assignment in a different function'; Code = "function A { `$PSNativeCommandUseErrorActionPreference = `$true }`nfunction B { git status }" }
        @{ Name = 'a non-literal assignment'; Code = "`$PSNativeCommandUseErrorActionPreference = `$someValue`ngit status" }
        @{ Name = 'an assignment inside if ($x)'; Code = "if (`$x) { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit status" }
        @{ Name = 'an assignment inside if ($false)'; Code = "if (`$false) { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit --version" }
        @{ Name = 'an assignment inside else'; Code = "if (`$x) { 1 } else { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit status" }
        @{ Name = 'an assignment inside try'; Code = "try { `$PSNativeCommandUseErrorActionPreference = `$true } catch { }`ngit status" }
        @{ Name = 'an assignment inside catch'; Code = "try { 1 } catch { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit status" }
        @{ Name = 'an assignment inside foreach'; Code = "foreach (`$i in 1..3) { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit status" }
        @{ Name = 'an assignment inside while'; Code = "while (`$x) { `$PSNativeCommandUseErrorActionPreference = `$true }`ngit status" }
        @{ Name = 'an assignment inside a switch branch'; Code = "switch (`$x) { 1 { `$PSNativeCommandUseErrorActionPreference = `$true } }`ngit status" }
        @{ Name = 'a conditional $false between a $true assignment and the call'; Code = "`$PSNativeCommandUseErrorActionPreference = `$true`nif (`$x) { `$PSNativeCommandUseErrorActionPreference = `$false }`ngit status" }
    ) {
        param($Name, $Code)
        script:Get-Rules $Code | Should -Contain 'native-exe-exit-code-unchecked'
    }

    It 'counts Invoke-Expression and iex' {
        script:Get-Rules 'Invoke-Expression $x' | Should -Contain 'invoke-expression'
        script:Get-Rules 'iex $x' | Should -Contain 'invoke-expression'
    }
}

# Protects: the C# catch-all heuristic. Detects: missed swallowing catches, and false positives on
# rethrows, filtered catches and braces inside strings or comments.
Describe 'C# catch-all checks' -Tag 'Unit', 'Quality', 'Portable' {
    BeforeAll {
        function script:Get-CsRules {
            param([string]$Code)
            $path = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + '.cs')
            [System.IO.File]::WriteAllText($path, $Code)
            @(Get-PcaiCSharpCatchFinding -Path $path | ForEach-Object { $_.Rule })
        }
    }

    It 'counts empty catch-alls and catch-alls that return a default' {
        script:Get-CsRules 'class C { void M() { try { A(); } catch { } } }' | Should -Contain 'catch-all-empty'
        script:Get-CsRules 'class C { int M() { try { return A(); } catch (Exception ex) { Log(ex); return -1; } } }' | Should -Contain 'catch-all-returns-default'
        script:Get-CsRules 'class C { string M() { try { return A(); } catch (System.Exception) { return null; } } }' | Should -Contain 'catch-all-returns-default'
    }

    It 'ignores rethrows, filtered catches, specific exceptions and look-alikes in strings and comments' {
        script:Get-CsRules 'class C { void M() { try { A(); } catch (Exception) { Log(); throw; } } }' | Should -BeNullOrEmpty
        script:Get-CsRules 'class C { int M() { try { return A(); } catch (Exception ex) when (ex is IOException) { return 0; } } }' | Should -BeNullOrEmpty
        script:Get-CsRules 'class C { int M() { try { return A(); } catch (IOException) { return 0; } } }' | Should -BeNullOrEmpty
        script:Get-CsRules "class C { string S = `"catch { }`"; // catch (Exception) { return 0; }`n}" | Should -BeNullOrEmpty
    }
}

# Protects: prod-versus-test attribution, which decides which baseline counter a finding lands in.
# Detects: test code counted as prod (inflating the gate) or prod code hidden as test (hiding debt).
Describe 'Scope classification' -Tag 'Unit', 'Quality', 'Portable' {
    It 'classifies <Path> as <Expected>' -TestCases @(
        @{ Path = 'Tests/Unit/A.Tests.ps1'; Expected = 'test' }
        @{ Path = 'Native/PcaiNative.Tests/X.cs'; Expected = 'test' }
        @{ Path = 'Native/pcai_core/pcai_inference/tests/api.rs'; Expected = 'test' }
        @{ Path = 'Native/pcai_core/x/benches/b.rs'; Expected = 'test' }
        @{ Path = 'Native/PcaiNative/FooTests.cs'; Expected = 'test' }
        @{ Path = 'Modules/PC-AI.Common/Public/Get-Thing.ps1'; Expected = 'prod' }
        @{ Path = 'Native/pcai_core/pcai_inference/src/lib.rs'; Expected = 'prod' }
        @{ Path = 'Tools\Quality\Invoke-QualityRatchet.ps1'; Expected = 'prod' }
    ) {
        param($Path, $Expected)
        Get-PcaiQualityFileScope -RelativePath $Path | Should -Be $Expected
    }

    It 'starts the Rust test region at the first #[cfg(test)] mod, not at a cfg(test) fn' {
        $file = Join-Path $TestDrive 'scope.rs'
        [System.IO.File]::WriteAllText($file, "#[cfg(test)]`nfn helper() {}`n`nfn prod() {}`n`n#[cfg(test)]`nmod tests {}`n")
        Get-PcaiRustTestStartLine -Path $file | Should -Be 5
        $none = Join-Path $TestDrive 'none.rs'
        [System.IO.File]::WriteAllText($none, "fn only_prod() {}`n")
        Get-PcaiRustTestStartLine -Path $none | Should -Be ([int]::MaxValue)
    }
}

# Protects: the ratchet end to end - counting, baseline comparison, exit codes and fail-closed tool
# handling - on a fixture tree with exactly one planted instance of each pattern.
# Detects: a counter that reads zero on planted bad code, a baseline that cannot fail, and a missing tool
# that is treated as a pass.
Describe 'Quality ratchet end to end' -Tag 'Unit', 'Quality', 'Portable' {
    BeforeAll {
        $script:Fixture = Join-Path $TestDrive 'tree'
        $script:Out = Join-Path $TestDrive 'out'
        New-Item -ItemType Directory -Path $script:Out -Force | Out-Null
        script:New-RatchetFixture -Root $script:Fixture
        $script:HaveTools = $script:HaveAstGrep -and $script:HavePssa
        if ($script:HaveTools) {
            $script:TreeBefore = script:Get-TreeHash -Root $script:Fixture
            $script:Created = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-UpdateBaseline')
            $script:TreeAfter = script:Get-TreeHash -Root $script:Fixture
        }
    }

    It 'creates a baseline from the fixture and exits 0' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $script:Created.ExitCode | Should -Be 0 -Because $script:Created.Output
        $script:Created.Result.baseline_action | Should -Be 'created'
    }

    It 'counts each planted pattern exactly once' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $counts = $script:Created.Result.counts
        $counts['pcai-ps-ast|empty-catch|prod'] | Should -Be 1
        $counts['pcai-ps-ast|empty-catch|test'] | Should -Be 1
        $counts['pcai-ps-ast|silent-continue-state-changing|prod'] | Should -Be 1
        $counts['pcai-ps-ast|shadowed-builtin-command|prod'] | Should -Be 1
        $counts['pcai-ps-ast|native-exe-exit-code-unchecked|prod'] | Should -Be 1 -Because 'only the git call in Invoke-Planted lacks a check'
        $counts['pssa|PSAvoidUsingEmptyCatchBlock|prod'] | Should -Be 1
        $counts['pcai-cs-catch|catch-all-returns-default|prod'] | Should -Be 1
        $counts['ast-grep|unwrap-call|prod'] | Should -Be 1
        $counts['ast-grep|unwrap-call|test'] | Should -Be 1 -Because 'the unwrap after #[cfg(test)] mod is test code'
    }

    # Protects: shared build caches. Native/pcai_core/target/** is a junction to a shared cargo cache on some
    # hosts; the walker must neither enter excluded directories nor follow any junction or symlink.
    It 'does not walk target/ or follow a junction when scanning a non-git tree' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $outside = Join-Path $TestDrive 'outside-cache'
        script:Write-FixtureFile $outside 'deps/bad.rs' "pub fn x(o: Option<u8>) -> u8 { o.unwrap() }`n"
        script:Write-FixtureFile $script:Fixture 'target/debug/bad.rs' "pub fn x(o: Option<u8>) -> u8 { o.unwrap() }`n"
        $linkKind = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
        New-Item -ItemType $linkKind -Path (Join-Path $script:Fixture 'Native/linked') -Target $outside | Out-Null
        try {
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
            $run.Result.counts['ast-grep|unwrap-call|prod'] | Should -Be 1 -Because 'only Native/lib/src/lib.rs counts'
            (Test-Path -LiteralPath (Join-Path $outside 'deps/bad.rs')) | Should -BeTrue -Because 'the scan must not delete through the link'
        } finally {
            # Remove the link itself, never its contents.
            [System.IO.Directory]::Delete((Join-Path $script:Fixture 'Native/linked'), $false)
            Remove-Item -LiteralPath (Join-Path $script:Fixture 'target') -Recurse -Force
        }
    }

    # Protects: analyzer traversal. The analyzers used to be pointed at top-level directories, so they could
    # walk excluded or untracked descendants (a junction to the shared cargo cache under target/).
    # Detects: any analyzer input that is not an allowed file, or a PSScriptAnalyzer directory that is not
    # scanned non-recursively.
    It 'hands the analyzers only allowed files and directories' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        script:Write-FixtureFile $script:Fixture 'vendor/dep/Vendored.ps1' "try { 1 } catch { }`n"
        script:Write-FixtureFile $script:Fixture 'vendor/dep/vendored.rs' "pub fn v(o: Option<u8>) -> u8 { o.unwrap() }`n"
        script:Write-FixtureFile $script:Fixture 'Modules/target/Hidden.ps1' "try { 1 } catch { }`n"
        script:Write-FixtureFile $script:Fixture 'Native/node_modules/pkg/n.rs' "pub fn n(o: Option<u8>) -> u8 { o.unwrap() }`n"
        try {
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
            $inputs = $run.Result.scan_inputs
            $inputs | Should -Not -BeNullOrEmpty -Because 'the result must record what the analyzers were given'
            @($inputs.ast_grep_files) | Should -Be @('Native/App/Planted.cs', 'Native/lib/src/lib.rs')
            @($inputs.pssa_files) | Should -Be @('Modules/Planted.psm1', 'Tests/Unit/Planted.Tests.ps1')
            $run.Result.counts['ast-grep|unwrap-call|prod'] | Should -Be 1
            $run.Result.counts['pssa|PSAvoidUsingEmptyCatchBlock|prod'] | Should -Be 1
        } finally {
            Remove-Item -LiteralPath (Join-Path $script:Fixture 'vendor') -Recurse -Force
            Remove-Item -LiteralPath (Join-Path $script:Fixture 'Modules/target') -Recurse -Force
            Remove-Item -LiteralPath (Join-Path $script:Fixture 'Native/node_modules') -Recurse -Force
        }
    }

    # Protects: PSScriptAnalyzer inputs. It used to be given a directory, which makes it parse every .ps1/.psm1/.psd1
    # in it, including an untracked or malformed sibling that is outside the approved inventory.
    # Detects: any file that is not tracked being handed to (and parsed by) PSScriptAnalyzer.
    It 'never hands PSScriptAnalyzer an untracked sibling, so a malformed one causes no parse diagnostic' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa -and [bool](Get-Command git -CommandType Application -ErrorAction SilentlyContinue))) {
        $repo = Join-Path $TestDrive 'git-tree'
        script:Write-FixtureFile $repo 'Modules/Tracked.psm1' "try { 1 } catch { }`n"
        script:Write-FixtureFile $repo 'Modules/Untracked.ps1' "function Broken { if ( `n"
        script:Write-FixtureFile $repo 'Modules/Untracked.psd1' "@{ Broken = `n"
        $git = (Get-Command git -CommandType Application | Select-Object -First 1).Source
        & $git -C $repo init -q
        if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
        & $git -C $repo add Modules/Tracked.psm1
        if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
        $gitOut = Join-Path $TestDrive 'git-out'
        [void][System.IO.Directory]::CreateDirectory($gitOut)
        $data = [ordered]@{ schema_version = 1; tools = [ordered]@{ 'ast-grep' = $script:SgVersion; PSScriptAnalyzer = (Get-Module -ListAvailable PSScriptAnalyzer | Sort-Object Version -Descending | Select-Object -First 1).Version.ToString(); 'pcai-quality-checks' = (Get-PcaiChecksVersion) }; counts = [ordered]@{ 'pssa|PSAvoidUsingEmptyCatchBlock|prod' = 5; 'pcai-ps-ast|empty-catch|prod' = 5 } }
        [System.IO.File]::WriteAllText((Join-Path $gitOut 'baseline.json'), ($data | ConvertTo-Json -Depth 5))
        $run = script:Invoke-Ratchet -Fixture $repo -OutDir $gitOut
        @($run.Result.scan_inputs.pssa_files) | Should -Be @('Modules/Tracked.psm1')
        ($run.Result.counts.Keys -join ',') | Should -Not -Match 'ParseError'
        ($run.Result.reasons -join "`n") | Should -Not -Match 'PSScriptAnalyzer'
        $run.Result.counts['pssa|PSAvoidUsingEmptyCatchBlock|prod'] | Should -Be 1
        $run.ExitCode | Should -Be 0 -Because $run.Output
    }

    It 'splits ast-grep inputs into bounded batches without losing findings' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $many = Join-Path $TestDrive 'many'
        for ($i = 0; $i -lt 40; $i++) {
            script:Write-FixtureFile $many ('crate/src/module_with_a_long_name_{0:D3}.rs' -f $i) "pub fn f(o: Option<u8>) -> u8 { o.unwrap() }`n"
        }
        $manyOut = Join-Path $TestDrive 'many-out'
        $run = script:Invoke-Ratchet -Fixture $many -OutDir $manyOut -Extra @('-UpdateBaseline', '-AstGrepBatchChars', '400')
        $run.ExitCode | Should -Be 0 -Because $run.Output
        $run.Result.scan_inputs.ast_grep_batches | Should -BeGreaterThan 3
        $run.Result.counts['ast-grep|unwrap-call|prod'] | Should -Be 40
    }

    # Protects: the ast-grep exit-code contract. Exit 1 used to be accepted whenever stderr was empty, so a
    # scan that died quietly looked like a clean tree when the baseline had no findings for the files in it.
    # Detects: a scanner that passes the canary but fails (or lies) on the real scan.
    It 'fails the run when ast-grep passes the canary but exits 1 with no diagnostic on the real scan' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $shim = script:New-FakeAstGrep -Directory (Join-Path $TestDrive 'shim-silent')
        $env:PCAI_FAKE_SG_MODE = 'silent'
        try {
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-AstGrepCommand', $shim)
        } finally {
            Remove-Item Env:PCAI_FAKE_SG_MODE
        }
        $run.ExitCode | Should -Be 1
        ($run.Result.reasons -join "`n") | Should -Match 'ast-grep .*exit(ed)? 1'
    }

    It 'fails the run when ast-grep claims error diagnostics that it did not emit' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $shim = script:New-FakeAstGrep -Directory (Join-Path $TestDrive 'shim-claim')
        $env:PCAI_FAKE_SG_MODE = 'claim'
        try {
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-AstGrepCommand', $shim)
        } finally {
            Remove-Item Env:PCAI_FAKE_SG_MODE
        }
        $run.ExitCode | Should -Be 1
        ($run.Result.reasons -join "`n") | Should -Match 'ast-grep .*(error\(s\)|diagnostic)'
    }

    It 'passes against the baseline it just wrote' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
        $run.ExitCode | Should -Be 0 -Because $run.Output
        $run.Result.verdict | Should -Be 'pass'
    }

    It 'fails with exit 1 when the baseline is lower than the planted count' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $baselinePath = Join-Path $script:Out 'baseline.json'
        $original = Get-Content -LiteralPath $baselinePath -Raw
        try {
            $data = $original | ConvertFrom-Json -AsHashtable
            $data.counts['pcai-ps-ast|empty-catch|prod'] = 0
            $data.counts['ast-grep|unwrap-call|prod'] = 0
            [System.IO.File]::WriteAllText($baselinePath, ($data | ConvertTo-Json -Depth 5))
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
            $run.ExitCode | Should -Be 1
            $run.Result.verdict | Should -Be 'fail'
            ($run.Result.reasons -join "`n") | Should -Match 'pcai-ps-ast\|empty-catch\|prod rose from 0 to 1'
            ($run.Result.reasons -join "`n") | Should -Match 'ast-grep\|unwrap-call\|prod rose from 0 to 1'
        } finally {
            [System.IO.File]::WriteAllText($baselinePath, $original)
        }
    }

    It 'fails closed when ast-grep is missing' {
        $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-AstGrepCommand', 'pcai-no-such-ast-grep')
        $run.ExitCode | Should -Be 1
        ($run.Result.reasons -join "`n") | Should -Match 'ast-grep is not installed'
    }

    It 'fails closed when PSScriptAnalyzer is missing' {
        $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-PssaModuleName', 'PcaiNoSuchModule')
        $run.ExitCode | Should -Be 1
        ($run.Result.reasons -join "`n") | Should -Match "module 'PcaiNoSuchModule' is not installed"
    }

    It 'fails closed on a tool whose MAJOR.MINOR differs from the baseline' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $baselinePath = Join-Path $script:Out 'baseline.json'
        $original = Get-Content -LiteralPath $baselinePath -Raw
        try {
            $data = $original | ConvertFrom-Json -AsHashtable
            $data.tools['ast-grep'] = '0.1.0'
            [System.IO.File]::WriteAllText($baselinePath, ($data | ConvertTo-Json -Depth 5))
            $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
            $run.ExitCode | Should -Be 1
            ($run.Result.reasons -join "`n") | Should -Match 'ast-grep is .* but the baseline was recorded with 0\.1\.0'
        } finally {
            [System.IO.File]::WriteAllText($baselinePath, $original)
        }
    }

    It 'fails closed when the baseline is missing' {
        $run = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir (Join-Path $TestDrive 'empty-out') -Extra @('-AstGrepCommand', 'pcai-no-such-ast-grep')
        $run.ExitCode | Should -Be 1
        ($run.Result.reasons -join "`n") | Should -Match 'baseline not found'
    }

    It 'gives identical JSON on two runs except generated_at, and leaves the scanned tree untouched' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $first = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
        $firstText = (Get-Content -LiteralPath (Join-Path $script:Out 'result.json') -Raw) -replace '"generated_at":\s*"[^"]*"', ''
        $second = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out
        $secondText = (Get-Content -LiteralPath (Join-Path $script:Out 'result.json') -Raw) -replace '"generated_at":\s*"[^"]*"', ''
        $first.ExitCode | Should -Be 0
        $second.ExitCode | Should -Be 0
        $secondText | Should -Be $firstText
        (script:Get-TreeHash -Root $script:Fixture) | Should -Be $script:TreeBefore
        $script:TreeAfter | Should -Be $script:TreeBefore
    }

    It '-UpdateBaseline refuses to raise a count and lowers one that fell' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $baselinePath = Join-Path $script:Out 'baseline.json'
        $original = Get-Content -LiteralPath $baselinePath -Raw
        $catchFile = Join-Path $script:Fixture 'Tests/Unit/Planted.Tests.ps1'
        $catchText = Get-Content -LiteralPath $catchFile -Raw
        try {
            Add-Content -LiteralPath $catchFile -Value 'try { 2 } catch { }'
            $raise = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-UpdateBaseline')
            $raise.ExitCode | Should -Be 1
            (Get-Content -LiteralPath $baselinePath -Raw) | Should -Be $original -Because 'a failed update must not touch the baseline'

            [System.IO.File]::WriteAllText($catchFile, "1`n")
            $lower = script:Invoke-Ratchet -Fixture $script:Fixture -OutDir $script:Out -Extra @('-UpdateBaseline')
            $lower.ExitCode | Should -Be 0 -Because $lower.Output
            $lower.Result.baseline_action | Should -Be 'lowered'
            $after = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json -AsHashtable
            $after.counts.ContainsKey('pcai-ps-ast|empty-catch|test') | Should -BeFalse
            $after.counts['pcai-ps-ast|empty-catch|prod'] | Should -Be 1
        } finally {
            [System.IO.File]::WriteAllText($catchFile, $catchText)
            [System.IO.File]::WriteAllText($baselinePath, $original)
        }
    }
}

# Protects: the exit code of `Build.ps1 -Component lint -LintProfile ratchet`. The ratchet gate function used
# to leak the subprocess's stdout into its return value, so a failed ratchet returned summary strings plus
# $false, which is truthy, and the build exited 0.
# Detects: a failing ratchet that Build.ps1 reports as success. Runs a copy of Build.ps1 and the ratchet in a
# throwaway tree so the real baseline is never involved.
Describe 'Build.ps1 ratchet lint gate' -Tag 'Unit', 'Quality', 'Portable' {
    BeforeAll {
        function script:Invoke-MiniBuild {
            $output = @(& $script:Pwsh -NoProfile -File (Join-Path $script:MiniRepo 'Build.ps1') -Component lint -LintProfile ratchet 2>&1)
            $exit = $LASTEXITCODE
            return [pscustomobject]@{ ExitCode = $exit; Output = ($output -join "`n") }
        }
        $script:MiniRepo = Join-Path $TestDrive 'minirepo'
        $script:MiniOut = Join-Path $TestDrive 'mini-out'
        if ($script:HaveAstGrep -and $script:HavePssa) {
            [void][System.IO.Directory]::CreateDirectory($script:MiniRepo)
            Copy-Item -LiteralPath (Join-Path $script:RepoRoot 'Build.ps1') -Destination $script:MiniRepo
            foreach ($file in 'sgconfig.ratchet.yml', 'sgconfig.yml', 'PSScriptAnalyzerSettings.ratchet.psd1', 'PSScriptAnalyzerSettings.psd1') {
                Copy-Item -LiteralPath (Join-Path $script:RepoRoot $file) -Destination $script:MiniRepo
            }
            foreach ($dir in 'Tools/Quality', 'rules/core', 'rules/rust', 'rules/csharp', 'rules/tests') {
                $parent = Split-Path -Parent (Join-Path $script:MiniRepo $dir)
                [void][System.IO.Directory]::CreateDirectory($parent)
                Copy-Item -LiteralPath (Join-Path $script:RepoRoot $dir) -Destination $parent -Recurse
            }
            script:New-RatchetFixture -Root $script:MiniRepo
            $baseline = script:Invoke-Ratchet -Fixture $script:MiniRepo -OutDir $script:MiniOut -Extra @('-UpdateBaseline')
            if ($baseline.ExitCode -ne 0) { throw "could not create the mini-repo baseline: $($baseline.Output)" }
            $script:MiniBaseline = Join-Path $script:MiniRepo 'Config/quality-baseline.json'
            [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $script:MiniBaseline))
            Copy-Item -LiteralPath (Join-Path $script:MiniOut 'baseline.json') -Destination $script:MiniBaseline
        }
    }

    It 'exits 0 when the ratchet passes' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $run = script:Invoke-MiniBuild
        $run.ExitCode | Should -Be 0 -Because $run.Output
        $run.Output | Should -Match 'Quality ratchet passed'
    }

    It 'exits nonzero when the baseline is lower than the counts' -Skip:(-not ($script:HaveAstGrep -and $script:HavePssa)) {
        $original = Get-Content -LiteralPath $script:MiniBaseline -Raw
        try {
            $data = $original | ConvertFrom-Json -AsHashtable
            $data.counts['pcai-ps-ast|empty-catch|prod'] = 0
            [System.IO.File]::WriteAllText($script:MiniBaseline, ($data | ConvertTo-Json -Depth 5))
            $run = script:Invoke-MiniBuild
            $run.ExitCode | Should -Not -Be 0 -Because $run.Output
            $run.Output | Should -Match 'Quality ratchet failed'
        } finally {
            [System.IO.File]::WriteAllText($script:MiniBaseline, $original)
        }
    }

    It 'lets no lint gate function write a native call to its success stream' {
        # Static audit of Build.ps1: inside each gate function, a statement that starts with the call operator
        # must be assigned or end in Out-Null / Out-Host. Otherwise its output joins the function's return value.
        $text = Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Build.ps1') -Raw
        $ast = [System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$null, [ref]$null)
        $gates = 'Invoke-QualityRatchetCheck', 'Invoke-AstGrepCheck', 'Invoke-PowerShellQuality', 'Invoke-PrettierQuality', 'Invoke-DotnetFormatQuality', 'Invoke-TomlQuality', 'Invoke-RagQuality', 'Invoke-RustQuality', 'Invoke-RustInferenceQuality'
        $leaks = foreach ($name in $gates) {
            $function = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }, $true) | Select-Object -First 1
            if (-not $function) { "$name not found"; continue }
            $pipelines = $function.Body.FindAll({ param($n) $n -is [System.Management.Automation.Language.PipelineAst] }, $true)
            foreach ($pipeline in $pipelines) {
                $first = $pipeline.PipelineElements[0]
                if ($first -isnot [System.Management.Automation.Language.CommandAst]) { continue }
                if ($first.InvocationOperator -ne [System.Management.Automation.Language.TokenKind]::Ampersand) { continue }
                if ($pipeline.Parent -is [System.Management.Automation.Language.AssignmentStatementAst]) { continue }
                $last = $pipeline.PipelineElements[$pipeline.PipelineElements.Count - 1]
                $lastName = if ($last -is [System.Management.Automation.Language.CommandAst]) { $last.GetCommandName() } else { $null }
                if ($lastName -notin 'Out-Null', 'Out-Host') { "$name line $($pipeline.Extent.StartLineNumber): $($pipeline.Extent.Text)" }
            }
        }
        @($leaks) | Should -BeNullOrEmpty
    }
}

# Protects: the new Rust FFI rules and the ratchet's ast-grep configuration.
# Detects: a rule that never matches (the rules/powershell failure mode) or matches its own safe form.
Describe 'ast-grep ratchet rules' -Tag 'Unit', 'Quality', 'Portable' {
    It 'passes the rule self-tests in rules/tests (must-match and must-not-match snippets)' -Skip:(-not $script:HaveAstGrep) {
        $sgCommand = (Get-Command ast-grep -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1)
        if (-not $sgCommand) { $sgCommand = Get-Command sg -CommandType Application | Select-Object -First 1 }
        Push-Location $script:RepoRoot
        try {
            $output = @(& $sgCommand.Source test --config sgconfig.ratchet.yml --skip-snapshot-tests 2>&1)
            $exit = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $exit | Should -Be 0 -Because ($output -join "`n")
        ($output -join "`n") | Should -Match 'test result: ok\. 4 passed; 0 failed'
    }

    It 'keeps sgconfig.yml narrow so lefthook and the pre-build gate do not read the backlog' {
        (Get-Content -LiteralPath (Join-Path $script:RepoRoot 'sgconfig.yml') -Raw) | Should -Not -Match 'rules/rust'
        (Get-Content -LiteralPath (Join-Path $script:RepoRoot 'sgconfig.ratchet.yml') -Raw) | Should -Match 'rules/rust'
    }
}
