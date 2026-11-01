# Process and hash transport fallback contracts

Reviewed October 10, 2026; owner: Codex integration lane.

Process and hash helpers swallowed terminal worker errors. Process listing could
then start a managed scan despite unresolved original process custody. Hashing
also used an unbounded direct command after worker negotiation. Both helpers
now use the existing disk consumer's terminal exception policy, shared in the
private transport implementation. Deadlines, cancellation, malformed frames,
EOF and retained-custody causes propagate through exception wrappers without
replacing their original objects. Preexisting custody is checked before launch
and again before genuine availability or legacy-protocol fallback.

Hash legacy fallback now uses the existing bounded CLI transport and its argument
array. The disable-worker setting is honored. An empty worker response no longer
triggers a direct CLI request in the hash helper. Process listing preserves a
successful zero-row response without launching another backend. Genuine missing
tool availability remains eligible for managed fallback. The disk classifier's
executable statements are unchanged; its existing helper delegates to the shared
classifier.

A separate small-file defect referenced an uninitialized results variable under
PowerShell strict mode. The accumulator is initialized before processing. A real
three-byte abc fixture fails before the repair and passes afterward with its
known SHA256, unchanged bytes and no native transport calls.

The initial 27 maintained controls pass one and fail 26 on the original wrappers;
all 27 pass after repair. Seven additional controls cover empty responses, direct
CLI deadlines, failed legacy cleanup and the real strict-mode file. The final
three-suite run passes 112 tests with zero failures or NotRun. Three native-bundle
controls and one unavailable filesystem short-alias control are explicitly
skipped. All five source/test hashes match before and after that run.
The original control bodies and failed receipts remain preserved. ScriptAnalyzer
reports zero findings across the five changed sources/tests. Outputs and test
temporary files are on D:.

Actual receipts: `D:/pcai-relocation/process-hash-fallback-qa-r1/`;
current result: `full-regression-r2.json`. These are ordinary Pester/tool exit
receipts, not a new native supervisor closure qualification or a coverage run.
Inert lower transport boundaries test the actual maintained wrapper bodies;
existing worker tests separately exercise their retained subprocess fixtures.

Remaining gaps include native hash response path/cardinality/digest validation,
literal leading-dash path handling and native field/schema parity. The three
native-bundle skips do not qualify actual Rust/DLL pairing; short-path handling
also remains untested on this filesystem. Other users, real
backend deployment, shared-server reuse, lifecycle faults outside the executed
fixtures and measured responsiveness improvements remain open. Current-head
hosted validation is required after publication; the 85% coverage floor remains
unchanged.
