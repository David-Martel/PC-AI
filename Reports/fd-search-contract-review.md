# fd search failure and cache admission review

Reviewed on October 10, 2026. Root owns this repair; the independent reviewer
owns its separate receipts. The production change is limited to `Find-WithFd`
in `Modules/PC-AI.Acceleration/Public/Find-FilesFast.ps1`.

Previously, invalid fd regexes returned a warning and empty results. The public
dispatcher could cache those empty results as successful searches. A synthetic
invocation producing a plausible path followed by exit23 also escaped the
failure boundary. Invocation exceptions and PowerShell's native-error preference
could lose their original cause through the same catch block.

The repair captures the native exit immediately after invocation and admits
results only on exit0. Nonzero exits raise `PcaiFdSearchFailed` with the captured
exit code and tool target; a bare rethrow preserves invocation exceptions and
the caller's actual `NativeCommandExitException`. The dispatcher and cache
implementation remain unchanged, so failed calls never reach cache insertion.
Existing fd stderr suppression remains unchanged; this is not stderr capture.

## Validation

| Gate | Actual result | Scope |
|---|---|---|
| Original source controls | 3PASS / 5FAIL | Retained RED evidence, not operational success |
| Exact private candidate | 8PASS / 0FAIL | Independently reviewed physical fd and explicit synthetic seams |
| Maintained canonical suite | 8PASS / 0FAIL / 0SKIP / 0NOT_RUN | Byte-exact current canonical source/helpers/test copy on Carbon |
| ScriptAnalyzer | 0 errors / 0 warnings | Production file and maintained test with repo settings |
| Legacy PowerShell5.1 discovery | 8SKIP / 0PASS / 0FAIL / 0NOT_RUN | Actual5.1 discovery only; no native fd behavior acceptance |
| Current-head hosted CI | PENDING | Prior head1188 still failed the unchanged85% coverage gate |
| Performance/profile/full-module/fleet acceptance | NOT_TESTED | No speedup or deployment claim |

The maintained suite is `Tests/Unit/PcaiFdSearchContracts.Tests.ps1`. Six controls
use physical fd over three tiny owned files; two use deliberately synthetic
invocation scripts. Cache and backend routing are spies, not a live cache/native
DLL/profile qualification. Maintained path helpers are exact source AST extents.
Successful empty, filtered positive and direct FileInfo/path results remain
unchanged. Failures assert no escaped rows or cache insertion; the native-error
preference control checks the real underlying exception type and its own exit.

CI installs fd10.5.0 from the official release with archive SHA256
`A227701B8551C35A9931D9F6DA75503CF86D88E182D71FB849A70864C5D57CD7`,
then checks executable existence and its reported version. A missing fd on a
supported host fails discovery instead of skipping these contracts. On hosts
below PowerShell7.3, all eight are explicitly skipped because the suite includes
the newer native-error preference contract. No coverage floor, denominator,
exclusion or legacy failure baseline changed.

## Retained receipts

Original RED child99439880…/parentE6C7CECB… and private GREEN child8226A39A…/
parentC916BFA7… remain immutable under
`D:/pcai-relocation/find-files-fd-{baseline,candidate}-actual-r2/`.
Independent candidate review:
`.pcai/integration/find-files-fd-candidate-actual-peer-r2/review.json`,
SHA256 `1EE7AE3014D7ADDCEA36F521547D91F4F51A89522F2B3069AE58938515678C54`.

Maintained actual child:
`D:/pcai-relocation/find-files-fd-maintained-actual-r3/find-files-fd-maintained-r3/result.json`,
SHA256 `BFEC4E543642BC742A2BADE62DDF1F4D5DF66E443184E5A2AEB7DB9A869E3726`.
XML SHA256 `584A4C4A7C811843581082BB9C5DC1175B6E74A53BD061CDE4E7F5C871089C1F`.
Original parent receipt SHA256
`D05D4F956F366C5388B4E84C167FE7CB78E3333524FF9B0678354F637BE10F1A`.
Its original Root29192/session96290 retained the job through completion under
60s/2MiB stream caps; all ten closure flags are true and pending custody is zero.
Private source predecessors held for legacy-discovery and before/after JSON
pin defects remain preserved; neither was run.

The independent maintained actual review is sealed at
.pcai/integration/find-files-fd-maintained-actual-peer-r3/actual-review.json,
SHA256 4D51DABCB42DE987C3E547A09B76E78F01A6F716D9961715E4B7E135F26A25C0.

Actual Windows PowerShell5.1 discovery preserved all eight exact selectors as
XML Ignored/executedFalse. Child receipt SHA256
852223167B42DF2C1FBEF3B8848970A3F6C40B3F9B966725B84BEED996BDD2AB,
parent SHA256 084568604715A60B6997FB7800AC4649DD7CEEEF1EAD31C9065C1ED2A7E74282,
and XML SHA256 801EF293E50C667E37A9B573E99AF847C30EB1717ED94A0784C0B2932E6F6BF4
are retained under D:/pcai-relocation/find-files-fd-legacy-actual-r2/.
Original Root29192/session96290 retained child23260 through exit0 and all ten
closure flags; pending custody is zero. The first legacy launch failed before
test import because inherited PowerShell7 module paths hid Get-FileHash.
Its original4172 failure remains immutable under the legacy-actual-r1 folder.
The successful successor set only its child process module path to its own
Windows PowerShell built-in modules. No machine/profile environment changed.
