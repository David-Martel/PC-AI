# Windows fleet capabilities

`Tools/Get-PcaiMachineCapabilities.ps1` inventories Windows hardware without loading
PowerShell profiles, PC-AI native libraries, vendor tools or models. It performs
12 CIM queries, each with a three-second operation timeout by default. CIM
timeouts are provider operation limits rather than a guarantee about total wall
time during a malfunctioning Windows management stack. No stress workload runs.

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiMachineCapabilities.ps1 -AsJson
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiMachineCapabilities.ps1 -OutputPath C:\Temp\machine-capabilities.json
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiMachineCapabilities.ps1 -DryRun
```

`-h` and `--help` display help. Dry-run reads no profile, queries no hardware and
writes nothing, even if an output path is supplied. Existing output files need
explicit `-Force` to replace them. No environment variables, tasks, power plans
or build configuration are modified. Reports omit instance suffixes, serials,
GPU UUIDs, IP addresses and MAC addresses; vendor/product IDs remain useful for
driver matching. The report includes hostname, hardware names and driver versions.

Device presence comes from `Win32_PnPEntity.Present`, with error 45 classified as
disconnected. An old registered GPU or installed DLL never establishes current
GPU availability. A present device with status OK, error 0 and a matching current
driver is a driver candidate. CUDA, Intel compute, vendor telemetry and real
workloads still require separate validation. Missing PnP information means
unknown presence; video-controller records alone cannot establish presence.

The inventory groups Intel, NVIDIA, USB, Thunderbolt/USB4, audio/video and network
devices. Thunderbolt grouping identifies controllers by their reported names;
it does not prove a particular peripheral is routed through Thunderbolt. NIC
connection status and speed are raw provider values, not link-quality tests.

Providers report `Supported`, `SupportedNoInstances`, `Unsupported` for explicit
invalid class/namespace/not-supported errors, or `Unknown` for failures such as
access denied. Successful enumeration only confirms the provider supplied
instances. It does not validate readings or diagnose sensor health. ACPI thermal
zones are not CPU core sensors. Missing temperatures and power measurements stay
unknown; the tool never translates missing readings into healthy zero values.

Profiles and schemas live in `Config/machine-profiles/`. Expectations remain
separate from live discovery and never cause a GPU to become available. The NUC
profile records the observed Intel NUC13L3Kv7/i7-1370P and Iris Xe, plus the user's
disconnected RTX 5060 Ti statement. The dtm-work profile records only the user's
NVIDIA A2000 statement; chassis, CPU, connection and power remain unknown.

The NUC user statement describes an Anker fixed 20 V, 5 A adapter budget up to
100 W, with no other loads, compared with a 120 W/6 A OEM reference. This is
manual configuration, not sensed input power, voltage or a validated USB-PD
negotiation. The tool reports input watts and voltage as null. Do not use CPU
package RAPL power to estimate whole-PC input power or adapter headroom.

Build policy recommends at most two jobs on the NUC while its power path is under investigation, and provisionally two on
dtm-work. It further limits jobs to one per four logical CPUs and one per 4 GiB
of currently available memory, with a minimum of one. Unknown CPU/memory data
returns one. These are conservative interactive policies, not benchmark results
or guarantees of stability. They are recommendations only and are never applied
by the inventory script.

Validation uses `Tests/Boot/MachineCapabilities.Tests.ps1`: realistic present,
disconnected and unknown GPU fixtures; driver disagreement; unavailable
providers; privacy exclusions; dry-run and output safety; schemas; and a bounded
real Windows smoke test. No added native component is required.

`Tools/New-PcaiPowerShellRelease.ps1` generates explicit function, cmdlet and
alias exports from component manifests and literal script-only export declarations,
while retaining the unified public API and `NestedModules`. Nested components
load once during import; the umbrella no longer forces a second eager import.
Packaging is separate from deployment: generate into a temporary `OutputRoot`,
then review the umbrella manifest/module before any installed-file update. When
updating only an installed umbrella, use `SourceRoot` pointing to that installed
module's existing component tree so exports reflect those components; copying a
new repository umbrella does not upgrade installed components. This helper
installs nothing; `-DryRun`, `-WhatIf`, `-h` and `--help` are available.
