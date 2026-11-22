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

    It 'trusts $PSNativeCommandUseErrorActionPreference' {
        script:Get-Rules "`$PSNativeCommandUseErrorActionPreference = `$true`ngit status" | Should -Not -Contain 'native-exe-exit-code-unchecked'
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
