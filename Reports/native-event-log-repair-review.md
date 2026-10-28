# Native Event Log repair qualification

The Windows native collector returned fabricated event fields, ignored the
requested time window, capped results at 100 and closed Event Log handles with
the wrong API. Its JSON fields also differed from the public PowerShell contract.
Those findings came from the fleet agent-bus review and inspection of
[the collector](../Native/pcai_core/pcai_core_lib/src/telemetry/event_log.rs),
[its export](../Native/pcai_core/pcai_core_lib/src/lib.rs) and
[the consumer](../Modules/PC-AI.Hardware/Public/Get-SystemEvents.ps1).

The reviewed repair reads real System event values and publisher-rendered text,
filters the requested days and levels, pages owned batches and uses `EvtClose`.
It retains the two-argument C ABI and legacy JSON fields while supplying the
complete public fields. Ordinary API, rendering, bound and explicit reserve
failures return NULL; a successful empty result returns `[]`. PowerShell rejects
incompatible or incomplete arrays and uses its existing fallback. `IncludeInfo`
uses that fallback because the native ABI selects levels 1–3. CRLF first-line
handling matches the existing fallback, including the retained trailing CR.

The collector admits each synchronous API operation against a shared ten-second
budget. It cannot cancel an operation already admitted. Fields are bounded to
65,536 UTF-16 units including NUL, retained text to 8 MiB and serialized JSON to
16 MiB. Exceeding a limit rejects the complete native result. These are payload
bounds rather than an RSS ceiling or a guarantee of OOM recovery; release
`panic=abort` still aborts. Native fallback frequency and whole-call latency need
target-host measurement before any performance claim.

## Qualification scope

[Windows CI](../.github/workflows/ci.yml) now explicitly checks, lints and tests
`pcai_core_lib`; the existing inference-only profiles did not compile this code.
The 19 reviewed Event Log cases use inert variants, strings, clocks and close callbacks. They
cover rendering bounds, ownership, partial-result rejection, paging, text
accounting and escaped-output limits. At published `d664fa0`, current-head
[CI](https://github.com/David-Martel/PC-AI/actions/runs/38017583334) compiled the
Windows bindings but rejected their Clippy step: the alignment check used the
manual remainder form. The reviewed `is_multiple_of` correction preserves that
alignment rejection. The downstream Rust test job was skipped; native test
qualification remains pending a successful successor run.

The public contract suite uses synthetic fixtures and mocks both native sampling
and `Get-WinEvent`. The current canonical suite passed all 14 cases with zero failures, skips or
unexecuted cases, including CRLF equality. Its four source/fixture pins stayed
unchanged during the run (Pester 5.7.1; 11.99 seconds).
No real Event Log query or loaded-library replacement has occurred during review.

Actual Windows publisher availability, log access, localized rendering, real
handle growth and paired Rust/C#/PowerShell deployment remain **NOT TESTED**.
The repository's existing 85% PowerShell coverage gate remains unchanged. That
same published run passed 2,567 tests with zero failures and 39 explicit skips,
then failed the source-bound coverage gate at 13,299/21,105 commands (63.0135%).
Inert unit passes do not establish fleet runtime parity or complete repository
integration.
