# Native optimizer terminal advice review

Reviewed October 10, 2026; owner: `codex-p1-pcai-integration`.

The native optimizer previously marked absent-parent terminal snapshots as
automatic-safe kill recommendations and estimated savings as count times 4 MB.
Source inspection establishes those expressions; no execution of the predecessor
or measured savings is claimed. A missing parent in a process snapshot does not
establish ownership, disposability or a leak.

The bounded repair in
`Native/pcai_core/pcai_core_lib/src/performance/optimizer.rs` retains the existing
threshold (more than ten), action identifier and JSON/ABI layout. Advice now sets
`safe_to_auto=false`, explicitly requires manual ownership review and labels
savings unmeasured. The existing unsigned savings field uses zero as a documented
compatibility placeholder, not an observed zero-memory result. This producer
does not execute process termination.

Exact terminal matching accepts `cmd`, `conhost`, `cmd.exe` and `conhost.exe`.
The current locked sysinfo 0.39.6 Windows source preserves the executable suffix;
the caller lowercases the name. Other names remain excluded. The collector
already uses the snapshot's `process.parent()` and does not launch a per-terminal
CIM query. This repair has no measured performance claim.

## Source and verification custody

- Committed predecessor raw SHA256:
  `4AB8482D2A55C12AEBA902F531BB3C35145CC8F0966EB9C6C10BBA9D973C9275`.
- Canonical formatted successor raw SHA256:
  `281D2173C55D6B4C6459668BB9C405016E4AEFCFB44299214B4C01F741710656`.
- Root read the complete four-hunk diff; physical rustfmt formatting and its
  check passed. No other native source or ABI changes belong to this repair.
- Four maintained tests call the real helper/matcher: the threshold boundary,
  serialized manual/unmeasured fields, maximum count without estimated gain,
  and exact accepted/rejected names. Existing tests remain intact.
- Canonical compilation and these four tests are **PENDING** exact-head hosted
  Windows Core CI. No private differential test or local Rust build has run.
  The private Cargo runner remains **SOURCE_HOLD**, unused, after independent
  review found source-binding and cleanup gaps.

Private source manifests, preceding proposals and independent review remain
under `.pcai/integration/native-orphan-*`. Whole-Core hosted CI is the selected
qualification route to avoid another local build during workstation paging.
Publication, CI, paired DLL/managed runtime, deployment and performance remain
separate claims. The PowerShell coverage floor remains 85%.

## Remaining functional gaps

Pool estimates derived from physical memory and working sets still lack true
kernel-pool provenance. Paging currently returns zero when unknown. Handle-query
failures also return zero. Leak/driver causal labels and other estimated savings
are not established by those snapshots. Repeated full sysinfo refreshes and the
fixed delay require a matched workload benchmark before optimization.

Current native/managed resolution, real consumer parity, cancellation and
`Tools/Test-Optimizer.ps1` report/dry-run behavior remain OPEN. None of those gaps is
qualified by the four pure contracts or by physical formatting.
