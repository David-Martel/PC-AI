# USB4 NetworkDirect preview qualification

Run the [read-only preflight](../../Tools/SystemScripts/Networking/Get-PcaiUsb4RdmaReadiness.ps1)
on each Windows endpoint. It identifies the Microsoft preview prerequisites and
can check one direct route through Windows' selected interface. Its result is a
local observation. Repeat across both endpoints and keep application and
transport results with the exact source and binary hashes.

    pwsh -NoProfile -File Tools/SystemScripts/Networking/Get-PcaiUsb4RdmaReadiness.ps1
    pwsh -NoProfile -File Tools/SystemScripts/Networking/Get-PcaiUsb4RdmaReadiness.ps1 -PlanPath ./fresh-route-plan.json -PeerHost dtm-super-nuc

Microsoft's November 19, 2026 announcement still lists the exact Experimental
(Future Platforms) Windows build as forthcoming. The November 20 notes for build
29712.1013 make no USB4 RDMA claim. The preflight reports
**PublishedBuildRequired** until the operator supplies a Microsoft-published
build and revision through the optional PublishedBuild parameter. A matching
build, feature flag, Secure Boot observation and route can reach
**OperatorReviewRequired**. Review still requires test-signing state, the
matching NDv2 provider installation and a dedicated lab pair. The preflight
writes no configuration, installs no provider and runs no network probe.
[Microsoft USB4 RDMA announcement](https://techcommunity.microsoft.com/blog/microsoftusbblog/introducing-rdma-over-usb4%C2%AE-on-windows-run-larger-local-ai-models-across-pcs/4574602) ·
[Windows Insider notes](https://blogs.windows.com/windows-insider/2026/11/20/announcing-new-builds-for-20-november-2026/).

## Test sequence

1. Use matching builds, drivers and exact provider artifacts on dedicated test
   hosts. Microsoft requires test signing and Secure Boot changes for this
   preview. Keep these firmware and boot settings outside normal fleet
   deployment. Verify the upstream archive checksum and retain the installed
   provider hash separately from Git or Authenticode evidence. The preview
   ZIP for version 0.1.0 contains unsigned test binaries.
   [Microsoft provider source](https://github.com/microsoft/windows-usb4-rdma/tree/v0.1.0) ·
   [Provider release](https://github.com/microsoft/windows-usb4-rdma/releases/tag/v0.1.0).
2. Use a direct cable between each test pair. A three-host full mesh has three
   direct cables and one nonoverlapping /30 per cable. Recollect each endpoint
   through [fabric observation](fabric-topology.md) and produce a
   [current route plan](multi-hop-routing.md). The preflight rejects routed
   paths, expired observations, ambiguous edges, changed adapter GUID or MAC,
   a nonmanual source address and a Windows route choosing another interface.
   It evaluates the local path to the peer. Run it at the other endpoint to
   evaluate the return path.
3. Confirm each endpoint's provider and adapter with ndprov_install dump
   and ndadapterinfo. Use bounded ndping, one-shot ndlat and ndtput
   listeners, and test both directions. Record Windows build, provider SHA-256,
   host and adapter identity, test flags, payload sizes, latency distribution,
   receiver goodput, CPU and errors. Measure TCP over the same interfaces and
   verify counter deltas. A vendor example, IP reachability or SMB Direct check
   alone supplies a different evidence scope.
   [Microsoft benchmark reference](https://github.com/microsoft/windows-usb4-rdma/blob/v0.1.0/docs/test-suite-reference.md).
4. Repeat after cable removal and reconnection. Measure time to patched route
   observation and application reconnect separately. The preview transport
   leaves reconnection to the application. For distributed model tests keep
   identical branches and binaries, endpoint revision, model quantization,
   context and split fixed across the TCP and NetworkDirect runs.
   [Microsoft llama.cpp sample](https://github.com/microsoft/llama.cpp-nd-usb4).

## Linux and mixed fleets

Windows USB4NET and Linux ThunderboltIP remain the common TCP/UDP path. Linux
introduced USB4STREAM in 7.2. Test a maintained kernel or distribution backport
with `CONFIG_USB4_STREAM`. Run `pcai-tb stream-preflight --compact` on each
endpoint. It reports local visibility of the kernel control directory, module
entries, and `/dev/tbstreamN` character-device entries. `Absent` means the path
is absent in the current namespace. An unavailable or incomplete scan stays
explicit. Device entries can appear before a peer attaches. The probe avoids
opening a stream device because the first open can allocate DMA rings.

Qualify USB4STREAM on two Linux hosts. Use the configured stream's `index` for
its local device mapping and preserve enrolled peer identity independently of the
node number. A managed data path needs bounded nonblocking I/O, partial-transfer
handling, and requalification after a hangup before reopening. Compare concurrent
receiver goodput, latency, CPU and reconnection with authenticated USB4NET TCP on
the same physical link. For RXE or SIW, qualify software verbs on the observed
USB4NET interface. Experimental DMA-ring verbs drivers need matched endpoints and
platform qualification. Keep IP forwarding and internet passthrough on the
qualified network transport. Test NetworkDirect interoperability independently.
For a fleet with supported RDMA Ethernet NICs, validate conventional RoCE
separately with device-specific transport counters. The Microsoft preview
preflight returns **UnsupportedPlatform** on Linux.
[Linux 7.2 USB4 documentation](https://github.com/torvalds/linux/blob/v7.2/Documentation/admin-guide/thunderbolt.rst) ·
[USB4STREAM driver](https://github.com/torvalds/linux/blob/v7.2/drivers/thunderbolt/stream.c) ·
[Maintained Linux releases](https://www.kernel.org/) ·
[Linux RDMA userspace](https://github.com/linux-rdma/rdma-core) ·
[Linux USB4 verbs experiment](https://github.com/hellas-ai/thunderbolt-ibverbs) ·
[Strix Halo peer measurements](https://github.com/jyatesdotdev/strix-rdma#status).

The [CLI controls](../../Native/pcai_core/pcai_tb_cli/tests/stream_preflight_cli.rs)
exercise the report and option boundaries on Windows and Linux. The
[Linux fixtures](../../Native/pcai_core/pcai_tb_cli/src/stream_preflight.rs)
exercise kernel-path and device-entry classification without a live stream.
[Native test review](../../Native/pcai_core/pcai_tb_cli/tests/stream-preflight-review.json).

The [preflight unit tests](../../Tests/Unit/PcaiUsb4RdmaReadiness.Tests.ps1) use
a sanitized route-plan fixture and simulated Windows interfaces. They require
PowerShell 7.2 or later and can run on Windows or Linux. They exercise blocked
and matched direct routes without a live driver, network or provider.
[Test review](../../Tests/Unit/pcai-usb4-rdma-preflight-review.json).
