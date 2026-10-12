# Duplicate enumeration failure review

Reviewed October 10, 2026. Owner: Codex integration lane.

`Find-WithFdForDuplicates` previously accepted paths emitted by an fd command
that exited unsuccessfully. An inert adapter emitted an existing nested file
and exited with code 7; the actual shipped function returned that partial path
instead of applying the caller's nonrecursive fallback scope. The initial
targeted run had 13 passing cases and one failing case.
The later bracket-named root trial had 14 passing cases and one failing case
because fd output paths were interpreted as wildcards. These failed-first
counts come from the root agent's tool transcript; the independent review
accepts the retained final receipt rather than those unsaved baselines.

The function now captures the command exit immediately and uses the existing
warning and fallback route when it is nonzero. A successful empty enumeration
remains empty. Both Get-ChildItem fallback routes use LiteralPath, which also
fixes Include filtering against a nonrecursive directory root. Previously
Path with Include could yield zero matches for that root.
The fd output existence and item lookups also use LiteralPath so bracket-named
directories are accepted correctly.

The maintained `Tests/Unit/DuplicateFallbackContracts.Tests.ps1` suite has
15 actual passing cases, with zero failures, skips or unrun cases. It exercises
the shipped enumeration, grouping and .NET hash helper on eight owned files
totalling 27 bytes. The four added cases cover nonrecursive Include/Exclude,
failed partial output, successful output and successful empty output. Existing
SHA256, SHA1 and MD5 grouping, size boundaries, exclusion and recursion cases
remain in place. Source hashes remained unchanged during the final run;
ScriptAnalyzer reported zero findings using the repository settings.

Final evidence is retained outside Git at
`D:/pcai-relocation/duplicate-fd-exit-contract-r2/result.json` and `pester.xml`.
The JSON receipt SHA256 is
`D875920BD5D5EE3C2DED6D3500C96E8D7C04FB7EACE9F228917EA4BF26E5DD70`.
The source-bound independent review is recorded in
`Tests/Fixtures/DuplicateFallback.TestReview.json`; all eleven historical
assertion bodies remain unchanged.
The exit fixture is a PowerShell adapter, not a qualification of a particular
installed native fd binary. No scan-speed improvement or coverage percentage
is claimed. Native hash failures, native backend parity and full consumer
acceptance remain separate gaps. Current-head CI must be checked after this
change is published; the unchanged 85% coverage gate still applies.
