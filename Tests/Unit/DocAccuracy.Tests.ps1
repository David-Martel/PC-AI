#Requires -Version 7.0
# Contracts for the documentation-accuracy gate (Tools/validate-doc-accuracy.ps1) and the two scripts whose failures
# used to hide documentation drift (Tools/update-tool-coverage.ps1, Tools/Invoke-DocPipeline.ps1).
#
# Why this file exists: the previous validator printed mismatches and always exited 0, so CLAUDE.md, README.md and
# CI-CD-GUIDE.md drifted unnoticed (a CI table of disabled workflows, a /version route that never existed, crate and
# component lists that no longer matched the code). A gate that cannot fail is not a gate, so every check below has a
# positive control: a fixture repository built in TestDrive that carries exactly one injected defect, which the named
# check must report. The real-repository test then proves the checks also pass on the tree they protect.
# The fixture lives in TestDrive, not under Tests/Fixtures, so the real-repository scan never sees a deliberately broken
# link or an unexported function.
Set-StrictMode -Version Latest

BeforeAll {
    $script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $script:Gate = Join-Path $script:RepoRoot 'Tools/validate-doc-accuracy.ps1'
    $script:Coverage = Join-Path $script:RepoRoot 'Tools/update-tool-coverage.ps1'
    $script:Pipeline = Join-Path $script:RepoRoot 'Tools/Invoke-DocPipeline.ps1'
    $script:Utf8 = [System.Text.UTF8Encoding]::new($false)

    function Write-FixtureFile {
        param([string]$Root, [string]$Relative, [string]$Text)
        $full = Join-Path $Root $Relative
        $dir = Split-Path -Parent $full
        if (-not (Test-Path -LiteralPath $dir)) { $null = New-Item -ItemType Directory -Path $dir -Force }
        # Here-strings take the line endings of this file (CRLF in a Windows checkout); fixtures are always LF.
        [System.IO.File]::WriteAllText($full, $Text.Replace("`r`n", "`n"), $script:Utf8)
    }

    # A minimal repository on which every check passes. Each control mutates one file of it.
    function New-DocFixture {
        param([string]$Name, [scriptblock]$Mutate)
        $root = Join-Path $TestDrive $Name
        $null = New-Item -ItemType Directory -Path $root -Force
        Write-FixtureFile $root 'Build.ps1' @'
[CmdletBinding()]
param(
    [ValidateSet('alpha', 'beta', 'all')]
    [string]$Component = 'all',

    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release'
)
'@
        $tree = @'
```
FIX/
+-- Native/pcai_core/    # Rust workspace (2-crate monorepo)
|   +-- pcai_one/        # first crate
|   +-- pcai_two/        # second crate
+-- Tools/
```
'@
        Write-FixtureFile $root 'CLAUDE.md' @"
# Fixture CLAUDE

$tree

**Available ``-Component`` values:** ``all`` (default), ``alpha``, ``beta``

| Workflow file | Trigger | State |
|---|---|---|
| ``ci.yml`` | PR | active |
| ``release.yml`` | tag | disabled_manually |

### pcai-inference Endpoints

- Health check: ``GET http://127.0.0.1:8080/health``
- Chat completion: ``POST http://127.0.0.1:8080/v1/chat/completions``

See the [guide](docs/guide.md) and [the tools](Tools/).
"@
        Write-FixtureFile $root 'README.md' @"
# Fixture README

[![CI](https://example.invalid/actions/workflows/ci.yml/badge.svg)](https://example.invalid/actions/workflows/ci.yml)

$tree
"@
        Write-FixtureFile $root 'CI-CD-GUIDE.md' "# Guide`n`nSee .github/workflows/ci.yml and .github/workflows/release.yml.`n"
        Write-FixtureFile $root 'docs/guide.md' "# Guide`n"
        Write-FixtureFile $root 'Tools/README.md' "# Tools`n`nNothing here names a retired host.`n"
        Write-FixtureFile $root 'Tools/b.ps1' "# script`n"
        Write-FixtureFile $root '.github/workflows/ci.yml' "name: CI`non: push`njobs: {}`n"
        Write-FixtureFile $root '.github/workflows/release.yml' "name: Release`non: push`njobs: {}`n"
        Write-FixtureFile $root 'Native/pcai_core/Cargo.toml' "[workspace]`nmembers = [`n    `"pcai_one`",`n    `"pcai_two`",`n]`n"
        Write-FixtureFile $root 'Native/pcai_core/pcai_inference/src/http/mod.rs' @'
fn router() {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
}
'@
        Write-FixtureFile $root 'Config/pcai-tools.json' @'
{ "tools": [
  { "function": { "name": "tool_a" }, "pcai_mapping": { "module": "Fake", "cmdlet": "Get-Fake", "params": {} } },
  { "function": { "name": "tool_b" }, "pcai_mapping": { "script": "Tools/b.ps1", "params": {} } },
  { "function": { "name": "tool_c" }, "pcai_mapping": { "cmdlet": "wsl", "args": ["--shutdown"] } }
] }
'@
        Write-FixtureFile $root 'Deploy/rust-functiongemma/TOOLS.md' "# Catalog`n`n### tool_a`nA.`n`n### tool_b`nB.`n`n### tool_c`nC.`n"
        Write-FixtureFile $root 'Modules/Fake/Fake.psd1' "@{ RootModule = 'Fake.psm1'; FunctionsToExport = @('Get-Fake') }`n"
        Write-FixtureFile $root 'Modules/Fake/Fake.psm1' "function Get-Fake { 'fake' }`n"
        Write-FixtureFile $root 'Deploy/rust-functiongemma-train/data/training_data.jsonl' (
            '{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"tool_a"}}]}]}' + "`n" +
            '{"messages":[{"role":"assistant","content":"NO_TOOL"}]}' + "`n")
        if ($Mutate) { & $Mutate $root }
        return $root
    }

    function Add-FixtureText {
        param([string]$Root, [string]$Relative, [string]$Text)
        [System.IO.File]::AppendAllText((Join-Path $Root $Relative), $Text, $script:Utf8)
    }

    function Edit-FixtureText {
        param([string]$Root, [string]$Relative, [string]$Old, [string]$New)
        $full = Join-Path $Root $Relative
        $text = [System.IO.File]::ReadAllText($full)
        if (-not $text.Contains($Old)) { throw "fixture text not found in ${Relative}: $Old" }
        [System.IO.File]::WriteAllText($full, $text.Replace($Old, $New), $script:Utf8)
    }

    # Runs the gate in-process (its exit statement ends only the script) and returns the exit code and parsed report.
    function Invoke-Gate {
        param([string]$Root, [string[]]$Check, [hashtable]$Extra = @{})
        $out = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + '.json')
        $splat = @{ RepoRoot = $Root; OutputPath = $out } + $Extra
        if ($Check) { $splat.Check = $Check }
        $null = & $script:Gate @splat *>&1
        $code = $LASTEXITCODE
        [pscustomobject]@{ Exit = $code; Report = ([System.IO.File]::ReadAllText($out) | ConvertFrom-Json); Path = $out }
    }

    function Get-Findings {
        param($Result, [string]$Id, [string]$Status)
        @($Result.Report.checks | Where-Object { $_.id -eq $Id -and $_.status -eq $Status })
    }
}

Describe 'documentation accuracy gate on the real repository' -Tag 'Unit', 'Portable' {
    # Protects: CLAUDE.md, README.md, CI-CD-GUIDE.md and the tool catalog keep agreeing with Build.ps1, the Cargo
    #   workspace, the workflow files, the HTTP routes, the tool schema, the module manifests and every relative link.
    # Detects: any documentation drift in the tree under test; the message names the file, line, claim and the fix.
    # Needs: PowerShell 7 and the checkout; no network, gh, or module import. Retired-host mentions stay warnings.
    # Breadcrumb: Tools/validate-doc-accuracy.ps1; docs/reference/DOCUMENTATION_AND_TEST_STANDARD.md in the owner repo.
    BeforeAll {
        $script:Real = Invoke-Gate -Root $script:RepoRoot
    }

    It 'exits 0 with no failing check' {
        $failures = @($script:Real.Report.checks | Where-Object status -EQ 'fail' |
                ForEach-Object { "$($_.id) $($_.file):$($_.line) $($_.claim) -> $($_.actual)" })
        $failures | Should -BeNullOrEmpty
        $script:Real.Exit | Should -Be 0
    }

    It 'ran every check and each one produced a pass entry or a finding' {
        $ids = @($script:Real.Report.checks | ForEach-Object id | Select-Object -Unique)
        foreach ($id in 'build-components', 'rust-crates', 'workflows', 'endpoints', 'tool-schema', 'relative-links', 'module-exports', 'retired-hosts') {
            $ids | Should -Contain $id
        }
        $script:Real.Report.schema_version | Should -Be 1
        $script:Real.Report.summary.fail | Should -Be 0
    }

    It 'keeps retired-host findings at warn by default' {
        @($script:Real.Report.checks | Where-Object { $_.id -eq 'retired-hosts' -and $_.status -eq 'fail' }).Count | Should -Be 0
    }
}

Describe 'documentation accuracy gate positive controls' -Tag 'Unit', 'Portable' {
    # Protects: a check that cannot fail is not a check; each control proves its check reports a real defect.
    # Detects: a check that passes on a fixture carrying the defect it exists to catch (for example by reading the wrong
    #   file, by a regex that matches nothing, or by treating a missing input as success).
    # Needs: PowerShell 7 and TestDrive; no repository files are read except the gate script itself.
    # Breadcrumb: Tools/validate-doc-accuracy.ps1 Test-* functions.
    It 'passes every check on the clean fixture, so each control below changes one fact' {
        $root = New-DocFixture 'clean'
        $r = Invoke-Gate -Root $root
        $r.Exit | Should -Be 0
        @($r.Report.checks | Where-Object status -NE 'pass' | ForEach-Object { "$($_.id): $($_.claim) -> $($_.actual)" }) | Should -BeNullOrEmpty
        @($r.Report.checks | ForEach-Object id | Select-Object -Unique) | Should -Contain 'tool-schema'
    }

    Context 'build-components' {
        It 'fails when CLAUDE.md lists a component the ValidateSet does not accept' {
            $root = New-DocFixture 'bc-extra' { param($r) Edit-FixtureText $r 'CLAUDE.md' '`beta`' '`beta`, `ghost`' }
            $r = Invoke-Gate -Root $root -Check 'build-components'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'build-components' 'fail').actual | Should -Match 'ghost'
        }
        It 'fails when CLAUDE.md omits a component' {
            $root = New-DocFixture 'bc-missing' { param($r) Edit-FixtureText $r 'CLAUDE.md' ', `beta`' '' }
            $r = Invoke-Gate -Root $root -Check 'build-components'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'build-components' 'fail').actual | Should -Match 'beta'
        }
    }

    Context 'rust-crates' {
        It 'fails when the tree names a crate that is not a workspace member' {
            $root = New-DocFixture 'rc-extra' { param($r) Edit-FixtureText $r 'CLAUDE.md' '|   +-- pcai_two/        # second crate' "|   +-- pcai_two/        # second crate`n|   +-- pcai_ghost/      # not a member" }
            $r = Invoke-Gate -Root $root -Check 'rust-crates'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'rust-crates' 'fail').actual -join ' ' | Should -Match 'pcai_ghost'
        }
        It 'fails when a workspace member is missing from the README tree and when the stated count is wrong' {
            $root = New-DocFixture 'rc-missing' {
                param($r)
                Edit-FixtureText $r 'README.md' "|   +-- pcai_two/        # second crate`n" ''
                Edit-FixtureText $r 'README.md' '(2-crate monorepo)' '(5-crate monorepo)'
            }
            $r = Invoke-Gate -Root $root -Check 'rust-crates'
            $r.Exit | Should -Be 1
            $text = (Get-Findings $r 'rust-crates' 'fail' | ForEach-Object { "$($_.claim) $($_.actual)" }) -join ' | '
            $text | Should -Match 'pcai_two'
            $text | Should -Match '5 crates'
        }
    }

    Context 'workflows' {
        It 'fails on a workflow file with no row in the CLAUDE.md table and none in the guide' {
            $root = New-DocFixture 'wf-undocumented' { param($r) Write-FixtureFile $r '.github/workflows/new.yml' "name: New`non: push`njobs: {}`n" }
            $r = Invoke-Gate -Root $root -Check 'workflows'
            $r.Exit | Should -Be 1
            @(Get-Findings $r 'workflows' 'fail' | Where-Object claim -Match 'new\.yml').Count | Should -BeGreaterOrEqual 2
        }
        It 'fails on a table row, guide reference or README badge that names a workflow that does not exist' {
            $root = New-DocFixture 'wf-ghost' {
                param($r)
                Edit-FixtureText $r 'CLAUDE.md' '| `release.yml` | tag | disabled_manually |' "| ``release.yml`` | tag | disabled_manually |`n| ``ghost.yml`` | none | active |"
                Add-FixtureText $r 'CI-CD-GUIDE.md' "`nAlso .github/workflows/phantom.yml.`n"
                Add-FixtureText $r 'README.md' "`n[badge](https://example.invalid/actions/workflows/missing.yml/badge.svg)`n"
            }
            $r = Invoke-Gate -Root $root -Check 'workflows'
            $claims = (Get-Findings $r 'workflows' 'fail' | ForEach-Object claim) -join ' | '
            $claims | Should -Match 'ghost\.yml'
            $claims | Should -Match 'phantom\.yml'
            $claims | Should -Match 'missing\.yml'
        }
    }

    Context 'endpoints' {
        It 'fails on a documented /version endpoint that has no route' {
            $root = New-DocFixture 'ep-version' { param($r) Add-FixtureText $r 'CLAUDE.md' "`n- ``/version`` endpoint returns JSON with the git hash.`n" }
            $r = Invoke-Gate -Root $root -Check 'endpoints'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'endpoints' 'fail').claim | Should -Match '/version'
        }
        It 'fails on a documented route with the wrong HTTP verb' {
            $root = New-DocFixture 'ep-verb' { param($r) Edit-FixtureText $r 'CLAUDE.md' 'GET http://127.0.0.1:8080/health' 'POST http://127.0.0.1:8080/health' }
            $r = Invoke-Gate -Root $root -Check 'endpoints'
            $r.Exit | Should -Be 1
        }
    }

    Context 'tool-schema' {
        It 'fails when a schema tool is missing from TOOLS.md' {
            $root = New-DocFixture 'ts-missing' {
                param($r)
                Edit-FixtureText $r 'Config/pcai-tools.json' '{ "function": { "name": "tool_c" }' '{ "function": { "name": "tool_d" }, "pcai_mapping": { "script": "Tools/b.ps1" } },{ "function": { "name": "tool_c" }'
            }
            $r = Invoke-Gate -Root $root -Check 'tool-schema'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'tool-schema' 'fail').claim | Should -Match 'tool_d'
        }
        It 'fails when a mapping names a script that does not exist' {
            $root = New-DocFixture 'ts-script' { param($r) Edit-FixtureText $r 'Config/pcai-tools.json' 'Tools/b.ps1' 'Tools/gone.ps1' }
            $r = Invoke-Gate -Root $root -Check 'tool-schema'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'tool-schema' 'fail').claim | Should -Match 'gone\.ps1'
        }
        It 'fails when a mapping names a cmdlet that the module does not define' {
            $root = New-DocFixture 'ts-cmdlet' { param($r) Edit-FixtureText $r 'Config/pcai-tools.json' 'Get-Fake' 'Get-Ghost' }
            $r = Invoke-Gate -Root $root -Check 'tool-schema'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'tool-schema' 'fail').claim | Should -Match 'Get-Ghost'
        }
    }

    Context 'relative-links' {
        It 'fails on a link to a missing file' {
            $root = New-DocFixture 'rl-dead' { param($r) Add-FixtureText $r 'docs/guide.md' "`nSee [gone](missing.md).`n" }
            $r = Invoke-Gate -Root $root -Check 'relative-links'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'relative-links' 'fail').claim | Should -Be 'missing.md'
        }
        It 'fails on a case mismatch, which resolves on Windows and breaks on Linux' {
            $root = New-DocFixture 'rl-case' { param($r) Add-FixtureText $r 'docs/guide.md' "`nSee [x](../README.MD) and [y](../CLAUDE.md).`n" }
            $r = Invoke-Gate -Root $root -Check 'relative-links'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'relative-links' 'fail').fix_hint | Should -Match 'Case mismatch'
        }
        It 'ignores fenced code, inline code, URLs and pure anchors' {
            $root = New-DocFixture 'rl-ignored' {
                param($r)
                Add-FixtureText $r 'docs/guide.md' "`n``````text`n[a](nowhere.md)`n``````n`n``[b](nowhere.md)`` [c](https://example.invalid/x) [d](#top) [e](mailto:a@b.invalid)`n"
            }
            (Invoke-Gate -Root $root -Check 'relative-links').Exit | Should -Be 0
        }
    }

    Context 'module-exports' {
        It 'fails when a manifest exports a function the module does not define' {
            $root = New-DocFixture 'me-ghost' { param($r) Edit-FixtureText $r 'Modules/Fake/Fake.psd1' "@('Get-Fake')" "@('Get-Fake', 'Get-Unexported')" }
            $r = Invoke-Gate -Root $root -Check 'module-exports'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'module-exports' 'fail').claim | Should -Match 'Get-Unexported'
        }
    }

    Context 'training-data' {
        It 'fails when a training example calls a tool that is not in the schema' {
            $root = New-DocFixture 'td-unknown' {
                param($r)
                Write-FixtureFile $r 'Deploy/rust-functiongemma-train/data/training_data.jsonl' '{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"not_a_tool"}}]}]}'
            }
            $r = Invoke-Gate -Root $root -Check 'training-data'
            $r.Exit | Should -Be 1
            (Get-Findings $r 'training-data' 'fail').actual | Should -Match 'not_a_tool'
        }
    }

    Context 'retired-hosts' {
        It 'reports a retired host as warn by default and does not fail the gate' {
            $root = New-DocFixture 'rh-warn' { param($r) Add-FixtureText $r 'README.md' "`nThe runner lives on vigil1.`n" }
            $r = Invoke-Gate -Root $root -Check 'retired-hosts'
            $r.Exit | Should -Be 0
            (Get-Findings $r 'retired-hosts' 'warn').actual | Should -Match 'vigil1'
        }
        It 'fails the gate when -RetiredHostSeverity is fail' {
            $root = New-DocFixture 'rh-fail' { param($r) Add-FixtureText $r 'README.md' "`nThe runner lives on vigil1.`n" }
            $r = Invoke-Gate -Root $root -Check 'retired-hosts' -Extra @{ RetiredHostSeverity = 'fail' }
            $r.Exit | Should -Be 1
            (Get-Findings $r 'retired-hosts' 'fail').Count | Should -Be 1
        }
        It 'matches every denylisted name case-insensitively on word boundaries, in Tools READMEs too' {
            $root = New-DocFixture 'rh-names' {
                param($r)
                Add-FixtureText $r 'CLAUDE.md' "`nASUSPRO13 and spark-0060 and spark-3066 and MillyLaptop1 and Milly and localhost:8400 and DGX Sparks.`n"
                Add-FixtureText $r 'Tools/README.md' "`nNode vigil1 again.`n"
            }
            $r = Invoke-Gate -Root $root -Check 'retired-hosts' -Extra @{ RetiredHostSeverity = 'fail' }
            $names = (Get-Findings $r 'retired-hosts' 'fail' | ForEach-Object actual) -join ' | '
            foreach ($n in 'asuspro13', 'spark-NNNN', 'millylaptop1', 'milly', 'localhost:8400', 'DGX Spark', 'vigil1') { $names | Should -Match ([regex]::Escape($n)) }
            @(Get-Findings $r 'retired-hosts' 'fail' | Where-Object file -EQ 'Tools/README.md').Count | Should -Be 1
        }
        It 'does not flag look-alike words' {
            $root = New-DocFixture 'rh-lookalike' { param($r) Add-FixtureText $r 'README.md' "`nMillyard, vigil-utils, vigil10 and sparkle-1.`n" }
            (Invoke-Gate -Root $root -Check 'retired-hosts' -Extra @{ RetiredHostSeverity = 'fail' }).Exit | Should -Be 0
        }
        It 'allowlists history files (Reports, CHANGELOG.md) even when the scope is widened' {
            $root = New-DocFixture 'rh-allow' {
                param($r)
                Write-FixtureFile $r 'CHANGELOG.md' "Retired asuspro13 and vigil1.`n"
                Write-FixtureFile $r 'Reports/old/notes.md' "millylaptop1 was here.`n"
                Write-FixtureFile $r 'docs/live.md' "Do not name asuspro13.`n"
            }
            $extra = @{ RetiredHostSeverity = 'fail'; RetiredHostScope = @('**/*.md') }
            $r = Invoke-Gate -Root $root -Check 'retired-hosts' -Extra $extra
            @(Get-Findings $r 'retired-hosts' 'fail' | ForEach-Object file) | Should -Be @('docs/live.md')
            $none = Invoke-Gate -Root $root -Check 'retired-hosts' -Extra ($extra + @{ RetiredHostAllowlist = @('nothing/**') })
            @(Get-Findings $none 'retired-hosts' 'fail').Count | Should -BeGreaterThan 2
        }
    }

    Context 'file reading' {
        It 'survives one-line and empty markdown files in scope on a full run (cached reads stay arrays)' {
            $root = New-DocFixture 'rd-short' {
                param($r)
                Write-FixtureFile $r 'Tools/x/README.md' "single line`n"
                Write-FixtureFile $r 'Tools/y/README.md' ''
                Write-FixtureFile $r 'docs/oneline.md' '[guide](guide.md)'
            }
            $r = Invoke-Gate -Root $root
            $r.Exit | Should -Be 0
            @($r.Report.checks | Where-Object status -EQ 'fail').Count | Should -Be 0
        }
        It 'accepts a comma-separated -Check string, as pwsh -File passes it, and rejects an unknown id' {
            $root = New-DocFixture 'rd-check'
            $out = Join-Path $TestDrive 'rd-check.json'
            $null = & pwsh -NoLogo -NoProfile -File $script:Gate -RepoRoot $root -OutputPath $out -Check 'build-components,rust-crates' *>&1
            $LASTEXITCODE | Should -Be 0
            @(([System.IO.File]::ReadAllText($out) | ConvertFrom-Json).checks | ForEach-Object id | Select-Object -Unique) | Should -Be @('build-components', 'rust-crates')
            $null = & pwsh -NoLogo -NoProfile -File $script:Gate -RepoRoot $root -Check 'no-such-check' *>&1
            $LASTEXITCODE | Should -Not -Be 0
        }
    }

    Context 'skip handling' {
        It 'reports skip, not pass, when a check input is absent' {
            $root = New-DocFixture 'skip-inputs' { param($r) Remove-Item -LiteralPath (Join-Path $r 'Build.ps1') }
            $r = Invoke-Gate -Root $root -Check 'build-components'
            @(Get-Findings $r 'build-components' 'skip').Count | Should -Be 1
            @(Get-Findings $r 'build-components' 'pass').Count | Should -Be 0
        }
    }
}

Describe 'documentation accuracy gate is deterministic and idempotent' -Tag 'Unit', 'Portable' {
    # Protects: the gate is safe to run in any hook or CI job: same input gives the same report, and nothing but
    #   -OutputPath is written.
    # Detects: a timestamp, ordering or hashtable-enumeration leak into the report; a check that writes into the repo.
    # Needs: PowerShell 7 and TestDrive; git for the tracked-file comparison (skipped when absent).
    # Breadcrumb: Tools/validate-doc-accuracy.ps1 result sorting and "no repo writes" contract.
    It 'yields identical JSON apart from generated_at on two runs' {
        $root = New-DocFixture 'idem-json' { param($r) Add-FixtureText $r 'README.md' "`nvigil1`n" }
        $a = Invoke-Gate -Root $root
        $b = Invoke-Gate -Root $root
        $strip = { param($rep) $rep.PSObject.Properties.Remove('generated_at'); $rep | ConvertTo-Json -Depth 8 }
        (& $strip $a.Report) | Should -Be (& $strip $b.Report)
        $a.Exit | Should -Be $b.Exit
    }

    It 'writes nothing to the repository: the fixture tree is byte-identical after a run' {
        $root = New-DocFixture 'idem-tree'
        $snapshot = { Get-ChildItem -LiteralPath $root -Recurse -File -Force | Sort-Object FullName | ForEach-Object { "$($_.FullName)|$($_.Length)|$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)" } }
        $before = & $snapshot
        $null = Invoke-Gate -Root $root
        (& $snapshot) | Should -Be $before
    }

    It 'leaves the real working tree unchanged (git status before equals after)' {
        $git = Get-Command git -ErrorAction SilentlyContinue
        if (-not $git -or -not (Test-Path -LiteralPath (Join-Path $script:RepoRoot '.git'))) {
            Set-ItResult -Skipped -Because 'git or a .git entry is not available'
            return
        }
        $before = @(& git -C $script:RepoRoot status --porcelain)
        $r1 = Invoke-Gate -Root $script:RepoRoot
        $r2 = Invoke-Gate -Root $script:RepoRoot
        @(& git -C $script:RepoRoot status --porcelain) | Should -Be $before
        $r1.Report.checks.Count | Should -Be $r2.Report.checks.Count
        ($r1.Report.checks | ConvertTo-Json -Depth 8) | Should -Be ($r2.Report.checks | ConvertTo-Json -Depth 8)
    }
}

Describe 'update-tool-coverage.ps1 handles script mappings' -Tag 'Unit', 'Portable' {
    # Protects: `generate-auto-docs.ps1` runs update-tool-coverage.ps1; a tool mapped by `script` instead of `cmdlet`
    #   (pcai_get_machine_capabilities, pcai_test_media_toolchain) must not abort the run.
    # Detects: reading `.cmdlet` on a mapping that has only `script` under Set-StrictMode (PropertyNotFoundException,
    #   exit 1), counting a script mapping as unmapped, and counting a mapping whose script file is absent as mapped.
    # Needs: PowerShell 7 and TestDrive; the script writes Reports/ under -RepoRoot, so only a fixture root is used.
    # Breadcrumb: Tools/update-tool-coverage.ps1 mapping loop.
    BeforeAll {
        $script:CovRoot = Join-Path $TestDrive 'coverage'
        Write-FixtureFile $script:CovRoot 'Config/pcai-tools.json' @'
{ "tools": [
  { "function": { "name": "pcai_script_tool" }, "pcai_mapping": { "script": "Tools/Do-Thing.ps1", "params": {} } },
  { "function": { "name": "pcai_cmdlet_tool" }, "pcai_mapping": { "module": "M", "cmdlet": "Get-Thing", "params": {} } },
  { "function": { "name": "pcai_missing_script_tool" }, "pcai_mapping": { "script": "Tools/Not-There.ps1", "params": {} } },
  { "function": { "name": "pcai_unmapped_tool" } }
] }
'@
        Write-FixtureFile $script:CovRoot 'Tools/Do-Thing.ps1' "# script`n"
        Write-FixtureFile $script:CovRoot 'Modules/PC-AI.LLM/Public/Invoke-FunctionGemmaReAct.ps1' "# no tool names here`n"
        $script:CovError = $null
        try { $null = & $script:Coverage -RepoRoot $script:CovRoot *>&1 } catch { $script:CovError = $_ }
        $script:CovReport = $null
        $reportPath = Join-Path $script:CovRoot 'Reports/TOOL_SCHEMA_REPORT.json'
        if (Test-Path -LiteralPath $reportPath) { $script:CovReport = [System.IO.File]::ReadAllText($reportPath) | ConvertFrom-Json }
    }

    It 'does not throw on a mapping that has only a script' {
        $script:CovError | Should -BeNullOrEmpty
        $script:CovReport | Should -Not -BeNullOrEmpty
    }

    It 'counts a script mapping whose file exists, and a cmdlet mapping, as mapped' {
        @($script:CovReport.MissingInMapping) | Should -Not -Contain 'pcai_script_tool'
        @($script:CovReport.MissingInMapping) | Should -Not -Contain 'pcai_cmdlet_tool'
    }

    It 'does not count a script mapping whose file is absent, or a tool with no mapping' {
        @($script:CovReport.MissingInMapping) | Should -Contain 'pcai_missing_script_tool'
        @($script:CovReport.MissingInMapping) | Should -Contain 'pcai_unmapped_tool'
    }
}

Describe 'Invoke-DocPipeline.ps1 records litho failures' -Tag 'Unit', 'Portable' {
    # Protects: the pipeline report and exit code reflect whether litho actually worked.
    # Detects: litho extract or generate exiting nonzero while the step is recorded as Success (a native command's
    #   failure does not throw, so the exit code must be read), and a failed run overwriting a good report.
    # Needs: PowerShell 7 and TestDrive; a fake `litho` first on PATH (a .cmd on Windows, a shell script elsewhere).
    # Breadcrumb: Tools/Invoke-DocPipeline.ps1 Invoke-LithoExtraction and Invoke-LithoDocGeneration.
    BeforeAll {
        function Invoke-PipelineWithFakeLitho {
            param([string]$Name, [int]$LithoExit)
            $fixture = Join-Path $TestDrive $Name
            $tools = Join-Path $fixture 'Tools'
            $bin = Join-Path $fixture 'fakebin'
            $null = New-Item -ItemType Directory $tools, $bin, (Join-Path $fixture 'Reports') -Force
            Copy-Item -LiteralPath $script:Pipeline -Destination $tools
            # Stubs for the two steps that record an Error (not a Warning) when their script is absent, so the only
            # thing that can fail the fixture run is litho.
            [System.IO.File]::WriteAllText((Join-Path $tools 'update-doc-status.ps1'), "param([string]`$RepoRoot)`n", $script:Utf8)
            [System.IO.File]::WriteAllText((Join-Path $tools 'generate-functiongemma-tool-docs.ps1'), "# stub`n", $script:Utf8)
            [System.IO.File]::WriteAllText((Join-Path $fixture 'Reports/LITHO_EXTRACT.json'), 'GOOD-REPORT', $script:Utf8)
            if ($IsWindows) {
                [System.IO.File]::WriteAllText((Join-Path $bin 'litho.cmd'), "@echo off`r`necho partial output`r`nexit /b $LithoExit`r`n", $script:Utf8)
            } else {
                $sh = Join-Path $bin 'litho'
                [System.IO.File]::WriteAllText($sh, "#!/bin/sh`necho partial output`nexit $LithoExit`n", $script:Utf8)
                & chmod +x $sh
            }
            $savedPath = $env:PATH
            try {
                $env:PATH = $bin + [System.IO.Path]::PathSeparator + $env:PATH
                $output = & pwsh -NoLogo -NoProfile -File (Join-Path $tools 'Invoke-DocPipeline.ps1') -Mode Full -SkipRust -SkipTraining 2>&1
                $code = $LASTEXITCODE
            } finally {
                $env:PATH = $savedPath
            }
            $report = [System.IO.File]::ReadAllText((Join-Path $fixture 'Reports/DOC_PIPELINE_REPORT.json')) | ConvertFrom-Json
            [pscustomobject]@{ Exit = $code; Report = $report; Output = ($output | Out-String); Fixture = $fixture }
        }
        $script:LithoFail = Invoke-PipelineWithFakeLitho -Name 'litho-fail' -LithoExit 7
        $script:LithoOk = Invoke-PipelineWithFakeLitho -Name 'litho-ok' -LithoExit 0
    }

    It 'records LithoExtract as Error and exits 1 when litho extract exits nonzero' {
        @($script:LithoFail.Report.steps | Where-Object Name -EQ 'LithoExtract').Status | Should -Be 'Error'
        ($script:LithoFail.Report.errors -join ' ') | Should -Match 'litho extract.*exited 7'
        $script:LithoFail.Exit | Should -Be 1
    }

    It 'keeps the previous LITHO_EXTRACT.json when litho extract fails' {
        [System.IO.File]::ReadAllText((Join-Path $script:LithoFail.Fixture 'Reports/LITHO_EXTRACT.json')) | Should -Be 'GOOD-REPORT'
        Test-Path -LiteralPath (Join-Path $script:LithoFail.Fixture 'Reports/LITHO_EXTRACT.json.tmp') | Should -BeFalse
    }

    It 'records LithoDocs as Warning (the step is optional) when litho generate exits nonzero' {
        @($script:LithoFail.Report.steps | Where-Object Name -EQ 'LithoDocs').Status | Should -Be 'Warning'
    }

    It 'still records Success for both steps when litho exits 0 (positive control)' {
        @($script:LithoOk.Report.steps | Where-Object Name -EQ 'LithoExtract').Status | Should -Be 'Success'
        @($script:LithoOk.Report.steps | Where-Object Name -EQ 'LithoDocs').Status | Should -Be 'Success'
        $script:LithoOk.Exit | Should -Be 0
    }
}
