# Network adapter classification review

`Get-NetworkDiagnostics -IncludeVirtual:$false` previously counted virtual
adapters as physical adapters and collected their IP configuration. The physical
predicate now selects only adapters whose `Virtual` property is false. The
existing virtual branch continues to control optional virtual output.

`Tests/Unit/NetworkAdapterClassification.Tests.ps1` runs the actual producer with
two physical and two virtual owned adapters. Its 21 controls cover default,
explicit true and explicit false selection, physical and total counts, active
physical summary semantics, virtual IP calls and forbidden external boundaries.
Collectors are inert collaborators; these tests do not exercise a live network.

The original producer failed five of these controls (16 passed). The corrected
producer passes all 21. The maintained three-file Network selection passes
130 controls, with zero failures and one existing skip. Its fixture exports only
the producer so collaborator command metadata cannot affect later Pester mocks;
empty virtual output is also safe under inherited strict mode.

The three selected files are `NetworkAdapterClassification.Tests.ps1`,
`PcaiNetworkContracts.Tests.ps1` and `PC-AI.Network.Tests.ps1` under `Tests/Unit`.
These results qualify unit behavior and fixture isolation. They do not establish
live adapter readiness, network throughput, or the repository-wide 85% command
coverage requirement. The earlier full run at `0be1130` remains failed and its
64.786% diagnostic source-bound union remains a separate coverage backlog.
