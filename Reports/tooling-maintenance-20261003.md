# PC_AI tooling and workstation maintenance, 2026-10-03

## Completed source repair and validation

`Get-PcaiAccelerationProbe` previously checked the acceleration directory's
parent (`Modules`) as the repository. With no `PCAI_ROOT` override, an explicit
repo-local Common import returned null repository/native roots. It now calls
the existing repository resolver and requires a `PC-AI.ps1` file before using
that checkout's native paths. A module checkout takes precedence over a
different environment checkout; existing environment fallbacks remain.

Five new tests use the real resolver and filesystem fixtures, mocking only
manifest discovery. They cover repository ancestry, checkout precedence,
native file discovery, rejection of unrelated AGENTS roots and missing-manifest
fallback. Before repair: three failed, two passed. After repair: eight executed
tests passed, zero failed, including three existing portable bootstrap tests.
One Windows-tagged bootstrap test was excluded from this bounded local run.

The live no-override probe found `C:\codedev\PC_AI\bin\pcai_core_lib.dll`;
the native ABI check passed, CPU count was 22 and sample token estimate was 19,
with no reported error. These installed binaries predate this change; no Rust
rebuild or source/binary equivalence claim is made.

PSScriptAnalyzer with the unchanged repository settings returned zero findings
for both changed scripts. The default unfiltered analyzer reports 13 inherited
warnings in Module-Common: nine Write-Host, one empty catch, one plural noun and
two built-in wrapper names. No rule was weakened for this repair.

## Bounded tooling benchmarks

Existing `Invoke-PcaiToolingBenchmarks.ps1` was run with explicit case selection,
private output roots, no coverage refresh and no recursive/network/GPU cases.
Both before/after runs had `PCAI_ROOT` unset. Import includes child PowerShell
startup; probes are warm in-process observations under heavy contention.

| Case | Before mean ms | After mean ms | Measured iterations |
|---|---:|---:|---:|
| Acceleration import | 1384.39 | 1746.51 | 3 each |
| Acceleration probe | 11.04 | 27.39 | 5 each |
| Direct native core probe | not measured | 33.94 | 5 |

No startup speed improvement is established. The repaired probe performs
successful repository/native discovery rather than the previous incomplete
path check, and workload contention varied between runs.

Token-estimate timings were 23.98 ms native versus 77.93 ms PowerShell, five
iterations each, mean ratio 3.25. The PowerShell standard deviation was 51.93 ms.
Native token estimation and the regex word-count baseline are different
heuristics; this is implementation timing, not output parity or a fleet SLA.

## Workstation health and package maintenance

### Interactive PowerShell profile repair

The existing real-console profile harness initially failed its native-state
assertion: PSReadLine reported `Jsonl` instead of `JsonlAccelerated`. Its first
three trace runs had a 2125 ms total median. Both profile shims defaulted
`PS_SKIP_PROFILE_ACCELERATOR` to `1` despite working installed libraries.

The installed ProfileAccelerator raw-tail guard also had an expression bug:

```powershell
# Before: the member access is not applied to the constructed type-name object.
if ([System.Management.Automation.PSTypeName]'ProfileAccelerator.Managed.RawJsonlAccelerator'.Type) {
# After:
if (([System.Management.Automation.PSTypeName]'ProfileAccelerator.Managed.RawJsonlAccelerator').Type) {
```

Three exact-byte backups were verified before applying this one-line guard fix
and changing only the two shims' default flag to `0`. Explicit opt-out remains
supported; existing agent sessions were not restarted. Installed module before
SHA256 `91f845e8dbffd87580b616c2ec466c5b7ebf11f98455c700471ca3170734ae0f`,
after `033a9e39338e2c54e0f9d652c5e15bea14e49e58977f63e10d9ac4e756379b91`.
Both shim before hashes were
`bc4527e55d3a144abf8576b61720d4ff63c8b21c94f1eef12ef8c97e5c7c5eea`,
after `55649611d05344ce728d65826656aac9a00cb4da2c3ccea1507c66a4f979c970`.

Private copied-module validation followed by installed-module validation passed
all five cases: Unicode history round trip, empty raw tail, malformed raw record
preservation, malformed history record skipping and last-record tail bounds.
Installed counters: three native writes, five native reads, zero fallback
writes/reads and zero errors. Synthetic tests did not read live history.
The original failing receipt is retained; its payload round trip succeeded,
but its all-native requirement failed.

Final default-path real-console startup: three samples, minimum 883 ms,
median 974 ms, mean 978 ms, maximum 1076 ms. The existing verifier then passed
its unchanged 1600 ms threshold (minimum 958 ms, median 974 ms), and correctly
rejected a deliberate 1600 ms delay (minimum 2692 ms). The observation shows
restored functionality and faster samples; contention also changed, so the
before/after difference is not an isolated causal speedup measurement.

The existing harness/verifier were copied privately with only child-window
visibility changed to Hidden and the verifier's harness filename adjusted.
Their full/interactive/native-state and negative-control assertions were
preserved. No native rebuild was needed. The installed module is an ordinary
unversioned directory; a tracked authoritative source was not found in the
bounded search. Future reinstallations must preserve or upstream this guard fix.

### System and package observations

At 08:05 UTC Windows reported 104,608,956,416 committed bytes against a
106,763,288,576-byte limit (about 98%), 977 MB available physical memory and
about 14,957 pages/sec. Earlier inventory found many concurrently active shells
and MCP server copies. These observations identify contention, not a proven
memory leak or authority to stop another owner's processes.

The pagefile already permits 32–128 GiB, WSL is capped at 32 GiB, and disks were
healthy with about 405.75 GiB free on C:. No blanket registry/storage tweak,
driver restart, WSL shutdown or heavy build was applied. Owners were asked to
pause new heavy Windows work and identify processes for graceful closure.
Oto reported completion of its local validation/browser/server jobs.

Cppcheck was upgraded through the official WinGet manifest from 2.21.0 to
2.22.0. WinGet verified the installer hash; installation succeeded and the
active command reports 2.22.0. A clean C fixture exits zero; a null-dereference
fixture emits `nullPointer` and the requested error exit code, proving the
updated tool detects an actual defect. The release includes tokenizer/template crash
repairs ([official release](https://github.com/cppcheck-opensource/cppcheck/releases/tag/2.22.0)).
Other updates remain scoped: Chocolatey's active age 1.1.0 shadows WinGet's
1.3.1; a WinGet-only update would not fix that command. Pandoc was deferred while
report work remains active. The inventory's larger update queue was not applied
in bulk under memory pressure.

## Evidence custody and remaining work

Private measurements and inventories are under
`C:\Users\david\.cache\pcai\broad-maintenance-20261003`; they contain process
and local-environment detail and are excluded from this public report.
Independent review found no implementation blockers. Exact committed-head
review, hosted CI and integration are recorded separately on the PR.

The existing Rust profiler requires fixes for crate names, absolute output
paths, environment restoration and command failures before its output can
qualify performance. Full import/startup acceleration and controlled repeat
benchmarks remain open. Foreign worktrees and ignored captures are preserved.
Milly backup, Thunderbolt physical-peer proof, fleet runtime/HIL and rendering
acceptance remain governed by their existing evidence gates.

At 08:18 UTC Milly reported Ubuntu 26.04.1 LTS, kernel 7.0.0-38, an active Cog
seat0 session and active SSH, GDM, NetworkManager and rsyslog. Package audit was
empty and upgrade simulation reported zero upgrades. Its RTX 5000 Ada Laptop
GPU reported driver 595.91.07 and 16376 MiB; 148 GiB was free on root. The
previously documented apport automatic-report timeout remains visible.

The follow-up owner/GitHub review verified TPM PR1597 and Spark PR2491/2493
merged with their checks passed. Spark PR2494 was still running checks, while
Oto draft51 passed both Python lanes and retained hardware/timing holds. No
owner-released foreign worktree or session was established for deletion.
