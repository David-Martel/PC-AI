# Static-quality ratchet

Counts risky patterns across Rust, C# and PowerShell and fails when any count is higher than
[`Config/quality-baseline.json`](../../Config/quality-baseline.json), so the existing backlog can only shrink.

```powershell
# Install: ast-grep 0.51.x and PSScriptAnalyzer 1.25.x (the versions the baseline records), PowerShell 7+
cargo install ast-grep --locked --version 0.51.0
Install-Module PSScriptAnalyzer -RequiredVersion 1.25.0 -Scope CurrentUser

# Run (about 20 s; writes .pcai/assessment/quality.json, which is gitignored)
pwsh -NoProfile -File Tools/Quality/Invoke-QualityRatchet.ps1

# Test (fixture trees under $TestDrive with planted bad code; rule self-tests via sg test)
Invoke-Pester -Path Tests/Unit/QualityRatchet.Tests.ps1

# Also reachable as a lint profile
.\Build.ps1 -Component lint -LintProfile ratchet
```

Exit 0 means no count rose. Exit 1 means a count rose, a tool is missing or at a different MAJOR.MINOR
than the baseline, a tool failed its canary probe, or the baseline is unreadable. The JSON result lists
`counts`, `delta`, `improvements`, `new_findings` (findings in files changed against `origin/main`) and
the `reasons` for a failure.

## What it counts

Every key is `<tool>|<rule>|<prod|test>`.

| Tool | Source | Notes |
| --- | --- | --- |
| `ast-grep` | `rules/core`, `rules/rust`, `rules/csharp` via [`sgconfig.ratchet.yml`](../../sgconfig.ratchet.yml) | Separate from `sgconfig.yml`: lefthook and the pre-build gate read that file, and most Rust rules are severity `error`. |
| `pssa` | [`PSScriptAnalyzerSettings.ratchet.psd1`](../../PSScriptAnalyzerSettings.ratchet.psd1) | Re-enables `PSAvoidUsingEmptyCatchBlock`, which the repo settings exclude, and `PSAvoidOverwritingBuiltInCmdlets`. The repo gate's slower rules stay in `-LintProfile powershell`; add `-IncludeRepoSettings` to run them here too. |
| `pcai-ps-ast` | [`PcaiPsAstChecks.psm1`](PcaiPsAstChecks.psm1) | Empty catch, `-ErrorAction SilentlyContinue` on state-changing verbs, native exe with no later `$LASTEXITCODE` read, shadowed built-ins, `Invoke-Expression`. |
| `pcai-cs-catch` | same module | Catch-all clauses that are empty or return a default without rethrowing. |

`vendor/`, `third_party/`, `worktrees/` and `Tests/Fixtures/` are not scanned. Scanned files are the
git-tracked ones, so untracked scratch files never move a count.

Prod versus test: any path segment `tests`/`test`/`benches`/`examples`/`*.Tests`, file names ending
`.Tests.ps1`, `Tests.cs`, `_test.rs`, and Rust code after the first `#[cfg(test)] mod` in a file.

## Heuristic limits

The PowerShell native-exit-code check and the C# catch-all check are heuristics; they are documented at
the top of `PcaiPsAstChecks.psm1`. The ast-grep Rust rules for `extern "C"` are approximations too (see
each rule's `note`). A count is a trend signal, not a proof of a bug.

## Lowering the baseline

When a count falls the run prints `IMPROVED:`. Lock it in from a clean tree and commit the result:

```powershell
pwsh -NoProfile -File Tools/Quality/Invoke-QualityRatchet.ps1 -UpdateBaseline
```

`-UpdateBaseline` never raises a count. Adding a rule, upgrading ast-grep/PSScriptAnalyzer, or changing a
check needs `-Rebaseline` in a change of its own; it prints every raised count so a reviewer sees them.

## Adding an ast-grep rule

Put the rule in `rules/rust/` (or `rules/csharp/`), add a must-match and a must-not-match snippet in
`rules/tests/`, and run `sg test --config sgconfig.ratchet.yml --skip-snapshot-tests`. Then
`-Rebaseline`, because the new rule's existing findings become part of the baseline.

## Related

- [`Test-ChangedPowerShellLint.ps1`](../Test-ChangedPowerShellLint.ps1): PSScriptAnalyzer on changed files only.
- [`Tests/Unit/QualityRatchet.Tests.ps1`](../../Tests/Unit/QualityRatchet.Tests.ps1): the controls for everything above.
- `Update-PowerShellBuiltinCommandList.ps1`: regenerates the static built-in command list used for the shadowing check.
