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
matching NDv2 provider registration and a dedicated lab pair. Record the
Microsoft publication URL and date alongside the exact build supplied to the
preflight. The preflight accepts the operator's build claim and observes the
registry flag. Confirm test signing, provider registration and a reboot after
the flag was set using separate evidence on each host. The preflight writes no
configuration, installs no provider and runs no network probe.
[Microsoft USB4 RDMA announcement](https://techcommunity.microsoft.com/blog/microsoftusbblog/introducing-rdma-over-usb4%C2%AE-on-windows-run-larger-local-ai-models-across-pcs/4574602) ·
[Windows Insider notes](https://blogs.windows.com/windows-insider/2026/11/20/announcing-new-builds-for-20-november-2026/).

## Test sequence

Start with the [USB4NET TCP baseline](peer-networking.md) on a direct, qualified
pair. Record both endpoint identities, adapter and driver versions, link rate,
MTU and tool versions. Run one-way tests in each direction and a simultaneous
bidirectional test. Capture receiver goodput per direction, latency under load,
CPU, retransmits and interface counter deltas. Repeat after a cable replug and
record time to link observation, route recovery and application reconnect
separately. Keep cable, ports, MTU and workload fixed when comparing transports.
[iperf3 options and result formats](https://software.es.net/iperf/invoking.html).

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
   and ndadapterinfo. The v0.1.0 provider implements two-sided send and receive.
   Use bounded ndping, one-shot ndlat and ndtput
   listeners, and test both directions. Record Windows build, provider SHA-256,
   host and adapter identity, test flags, payload sizes, latency distribution,
   receiver goodput, CPU and errors. Measure TCP over the same interfaces and
   verify counter deltas. A vendor example, IP reachability or SMB Direct check
   alone supplies a different evidence scope.
   [v0.1.0 provider scope](https://github.com/microsoft/windows-usb4-rdma/tree/v0.1.0) ·
   [Microsoft benchmark reference](https://github.com/microsoft/windows-usb4-rdma/blob/v0.1.0/docs/test-suite-reference.md).
4. Repeat after cable removal and reconnection. Measure time to patched route
   observation and application reconnect separately. The preview transport
   leaves reconnection to the application. For distributed model tests keep
   identical branches and binaries, endpoint revision, model quantization,
   context and split fixed across the TCP and NetworkDirect runs. Compare both
   against local-only inference when the model fits. Record time to first token,
   tokens per second and transport bytes per token. Record a larger model that
   only fits across the pair as a separate capacity result.
   [Microsoft llama.cpp sample](https://github.com/microsoft/llama.cpp-nd-usb4).

## Linux and mixed fleets

Windows USB4NET and Linux ThunderboltIP provide the common TCP/UDP path for
mixed peers. On Linux, choose experiments by what the application needs to test.

| Candidate | Qualification scope |
| --- | --- |
| [Upstream USB4STREAM](https://docs.kernel.org/admin-guide/thunderbolt.html#streaming-data-directly-over-thunderbolt-cable) | Raw Linux-to-Linux host streams via `/dev/tbstreamX`. Check the installed kernel's `CONFIG_USB4_STREAM` and match its userspace ABI on both endpoints. Verify payload hashes in both directions and repeat after reconnect. |
| [Soft-RoCE and Soft-iWARP](https://github.com/linux-rdma/rdma-core/blob/master/MAINTAINERS) | Linux verbs application compatibility over IP on USB4NET. Qualify the software provider on both ends and compare against the same-link TCP baseline. |
| [Hellas Thunderbolt verbs](https://github.com/hellas-ai/thunderbolt-ibverbs) | Out-of-tree Linux kernel modules and userspace. Pin matching module and userspace revisions on two lab hosts, then verify the selected device and actual verbs traffic. |
| [Strix Halo USB4 driver](https://github.com/hyukudan/strix-halo-usb4-driver) | Out-of-tree kernel patches for AMD Strix Halo, including an experimental DMA-BUF path. Use the project's exact kernel baseline and matching test hardware. |

The Microsoft preview preflight returns **UnsupportedPlatform** on Linux.
Windows NetworkDirect to Linux USB4STREAM or verbs remains unqualified. Keep
conventional RoCE over an RDMA-capable Ethernet NIC as a separate result.

The [preflight unit tests](../../Tests/Unit/PcaiUsb4RdmaReadiness.Tests.ps1) use
a sanitized route-plan fixture and simulated Windows interfaces. They require
PowerShell 7.2 or later and can run on Windows or Linux. They exercise blocked
and matched direct routes without a live driver, network or provider.
[Test review](../../Tests/Unit/pcai-usb4-rdma-preflight-review.json).
