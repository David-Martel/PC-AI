# USB4 topology maintenance review

Build, test and run instructions live in the
[topology operator contract](../docs/usb4/fabric-topology.md). This report records
the scope and admission limits of the observation feature; the live backlog is in
[boot.TODO.md](../boot.TODO.md#usb4-topology-observation-and-hot-plug-planning).

## Result and evidence

The working Thunderbolt Share session with dtm-carbon-two provides a physical
connection reference. The new collector and native `topology-plan` command use
Windows USB4NET identity and neighbor observations without a Share installation,
license query or vendor API dependency. The
[collector](../Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1) and
[native model](../Native/pcai_core/pcai_tb_cli/src/fabric_plan.rs) perform no network
configuration. Their output explicitly calls a paired edge `ObservedUnverified`.

Actual independently collected Gen9 and Carbon observations produced one such
edge while retaining a second, unidentified Gen9 endpoint as blocked. Both local
USB4 interfaces had overlapping APIPA networks, which the planner reports. The
fresh pair was evaluated before its 90-second expiry; a separate expired pair
produced no segments. These historical captures are evidence of that window,
not authorization to use a peer now. Source-bound TCP 445 connections succeeded;
SSH over the edge timed out. SMB authentication and share permissions were not
qualified. Existing management SSH was used to collect Carbon evidence.

## Validation and review

Nine native model selectors cover reciprocal evidence, malformed or ambiguous
identity, scoped addresses, freshness and replacement. Independent mutation
review caught an alias-coverage gap and masked uniqueness checks; the corrected
selectors detected eleven compiling runtime faults. Three initial uniqueness
mutants did not compile and receive no detection credit. Six public CLI controls
cover argument refusal, input bounds, prior-edge withdrawal and FIFO refusal;
the FIFO control is explicitly unsupported on Windows. Eight PowerShell fixture
controls cover present-only discovery, unknown roots, errors, neighbor filtering,
bounds and side-effect-free help/DryRun. The initial seven controls detected seven
targeted faults. A successor addressless-adapter control reproduces the original
failure, retains a healthy second adapter and detects both ID-only and
category-only error guards. Its eight-selector suite passes in native PowerShell
under WSL and in the exact-head hosted Windows job; local Windows PowerShell
startup stalled before output. No adapter was disabled to test the native empty
query error ID, so real provider error-shape qualification remains separate.

Hash-bound review records:

- [Native model and CLI](../Tests/Portable/usb4-fabric-plan-review.json)
- [Windows collector](../Tests/Portable/usb4-fabric-observation-review.json)

The [cross-platform CI entrypoint](../.github/workflows/usb4-lifecycle.yml) runs
the actual Rust package, collector fixtures and explicit CLI contracts.

The separate [bounded-provider fix, PR286](https://github.com/David-Martel/PC-AI/pull/286)
qualifies optional Windows telemetry failures and requires observed ancestry
before required topology admission or configuration reapply. Its
[review record](../Tests/Portable/usb4-provider-bounds-review.json) distinguishes
the implementation cohorts and their exact test counts.

## Deployment boundaries

No address, MTU, route, firewall, bridge, task, Share process or service changed.
The collector's outer child uses the existing bounded runner; its JSON size check
is post-capture, not a streaming memory cap. No performance improvement or actual
hot-plug, sleep, reboot or throughput acceptance is claimed. A withdrawal list is
plan data, not an operating-system deletion command.

Accepted main also contains the [standalone Windows peer framework](../docs/usb4/peer-networking.md).
Remaining deployment work includes source/release provenance, per-edge task
isolation and qualified explicit enrollment. An installed release can be ahead
of main; compare manifests and preserve pending ownership journals before any
promotion. Work, QNAP and UniFi remain separate authenticated management/storage
edges until each route and service is qualified. The QNAP owner cleared
hash-checked transfers through Work; this change performs no NAS transfers or
gateway/cloud configuration.
