# Independent USB4 publication and retirement review

Reviewed by `codex-p1gen9-zip-fix` at 2026-11-10T19:20:29.858638+00:00. Exact reviewed files and normalized LF hashes are in the [machine-readable receipt](usb4-journal-publication-review.json).

All 23 publication, 45 admission, 130 peer and 34 retirement assertions passed in fresh profile-free PowerShell processes. Two independent real-file regression assertions confirmed that the writer does not validate a separate generation and refuses a scalar captured generation without changing corrupt bytes or history.

The review found and reproduced two custody losses before correction: a validate/read race that archived `null`, and retirement accepting a live address from an in-flight unrecorded AddAddress. Both are corrected in the bound revision. Empty-input admission and optional policy ULA handling were also corrected. No remaining blocker was found within this narrow file-only recovery shape.

Recovery remains limited to the declared Consumer ExistingWinNat Rollback/Snapshot shape, a disabled exact task, pinned source/installed hashes, exact adapter and absent owned/potentially-created addresses. Unknown or nonaddress custody is refused. It archives exact evidence before clearing custody and never changes networking or enables the task.

The global mutexes coordinate cooperating privileged writers. The sequence is not an atomic compare-and-swap against arbitrary administrators. API flushing/readback does not prove hardware power-loss durability. Fixture execution does not approve a live retirement: the operator lane must independently bind the current plan, hashes, disabled task and observed address absence immediately before Apply.
