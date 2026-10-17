# PC-AI portable cache bootstrap independent review

Reviewed branch `fix/portable-cache-bootstrap` working changes, 2026-10-17 UTC. Reviewer: Codex portable_fix_review. No repository source edits; preserved other work. Read repository AGENTS.md and codex-code-review skill.

## Disposition

Approve bounded null-root fix and bootstrap CI; no critical/high finding in this diff. This is not full deployment, FFI, acceleration or USB4 qualification.

Actual implementation path is `Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1`. The two guards suppress optional Join-Path construction for missing/whitespace Windows roots; explicit CLI precedence and provider policy are unchanged. No runtime timeout, memory, concurrency or cache-size number changes. The new CI job's 10-minute timeout bounds disposable build/test execution, not production requests; no production-load claim applies. Rollback is revert of this source/CI change, not machine configuration restoration. Build timeout appears as a failed job rather than silent successful deployment.

## Behavior validation

Ran real `Tests/Portable/Test-CacheBootstrap.ps1` with `-SourcePath` pointing separately to the changed helper and temporary exact `git show HEAD:Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1` original. Both Ubuntu MS01 and Windows dtm-p1gen7, profile-free PowerShell 7.6.6:

| Selector | Changed | Original | Disposition |
|---|---|---|---|
| absent Windows roots preserve memory bootstrap | PASS | FAIL empty-string Join-Path, exit 1 | accept regression oracle |
| whitespace Windows roots do not break bootstrap | PASS | not reached after first failure | accept bounded assertion |
| explicit CLI path remains first | PASS | not reached after first failure | accept path selection; not exhaustive precedence proof |

Changed run exited 0; original exited 1 on both machines. Windows copied only test/helper/original into a random private temporary directory, ran profile-free child pwsh, then removed that directory in finally; cleanup reported OWN_TEMP_REMOVED. No repo, profile, deployment or registry edits on Windows.

## Hash-bound selector review

All three selectors reside in test SHA256 `6d7df4989c144cca3aed8a88b88278d7e51e5901440a2c340d6a08bfb60271ff` and exercise helper SHA256 `4f209a91c07c27a736cd83a909e5e707f55b418fcb74becd359b5c7ebedc78a1`.
Workflow SHA256 `a38cb3c8042e510c543a491b9f021887991a71e8a395fbba43cdf7e7c5cc00c0`; documentation SHA256 `892594071b6f6476addb8d4c342b22326b57d52b5dcde2342a49ef30955c7540`.

## CI and test limits

Workflow explicitly checks actual runner OS/architecture against linux-x64 or win-x64 and supplies that RID to dotnet build, overriding the project's Windows default. Permissions are read-only contents; no hardware, deployment, secrets or profiles are invoked. Standard GitHub pwsh shell uses profile-free invocation. Hosted CI itself was not run in this review. x64 only: ARM/macOS not qualified. Build succeeds without loading native libraries; current conditional payload includes Windows DLL names, so do not present build success as Linux native packaging.

Optional follow-ups: assert Enabled false in whitespace selector as well; add a competing real candidate to prove precedence rather than selected explicit file alone; independently exercise each original mutant so whitespace guard is not masked by the first null-root failure. Test intentionally clears provider selection and cannot establish Redis execution, cache roundtrip, native loader or malformed provider behavior. Fixture file is never executed. Full module import remains outside this test and may expose separate platform assumptions. Existing helper dot-source sets global cache variables; process isolation prevents persistent user session impact in CI and the reviewed commands.

## Extended discovery review, superseding workflow/docs hashes above

Approve the bounded Find-RustTool portability change. Actual host PATH separator, Windows-only .exe suffix, nonexistent-directory skipping, optional USERPROFILE root and Application-only fallback address genuine platform assumptions. Search precedence remains dynamic configured root, configured search paths, PATH, legacy fd folder, command lookup; no route, driver, power or deployment changes.

Ubuntu and gen7 Windows PowerShell Core 7.6.6 each ran the exact current `Test-RustToolDiscovery.ps1` against current source: both selectors PASS, process exit 0. Original HEAD source exits 1 on null USERPROFILE before first assertion. On Ubuntu a temporary original-plus-null-guard-only control still fails on nonexistent C drive, demonstrating null guard alone is insufficient. Windows private temp copies removed in finally, OWN_TEMP_REMOVED confirmed.

| Selector | Disposition | Limits |
|---|---|---|
| host PATH discovery works without Windows roots | accept | exact host-suffix temporary file selected; intentionally not executed; not executable permission/version qualification |
| absent tool remains unavailable | accept | improbable name reaches absence path; does not isolate every fallback branch |

Hash bindings: discovery test `8792011a1195825b6c80bbb067ad50abca4013daf2ea4cf64c7a4837e751dff8`; discovery implementation `a4b782173343188f5cda19b0d6630291830181a4d469b45f9ff434484fa50c0c`. Final workflow `546d84d1158787e4dd3c8f469bf31fa4cce42b776d2f32dc81b6451e920ba559`; final docs `253a5a3da8b96707322d5308c70bd6df9fda97844dd9d3d1ff29b163aef482dc`; README `4d9b0ba7929ecc03005beb1d4c4d63713052451257c4a9043fa3e31b9295b053`. Earlier cache helper/test hashes remain unchanged.

README now correctly conditions acceleration gains on equivalent-output validation on target platform instead of unqualified 5–40x claim. New workflow includes discovery source path triggers and both real behavior tests. Review has not executed hosted CI. Additional future tests should cover executable permission, stale cache invalidation, Application fallback in isolation, whitespace PCAI_ROOT and Windows PowerShell 5.1 compatibility if separately supported. Existing directory/file selection does not prove an executable is runnable; no acceleration gain follows from discovery success. Tests change script-local maps and process environment in disposable child processes, not profiles or production config.
