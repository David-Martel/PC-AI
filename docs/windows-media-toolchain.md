# Windows media qualification

Run the [qualification tool](../Tools/Test-PcaiMediaToolchain.ps1) from a fresh `pwsh -NoLogo -NoProfile -NonInteractive`
session. Default operation lists installed FFmpeg/GStreamer capabilities and
current GPU inventory; it does not open a camera, microphone or network stream.
Compiled codec lists do not prove that a connected GPU can use them.

```powershell
.\Tools\Test-PcaiMediaToolchain.ps1 -DryRun
.\Tools\Test-PcaiMediaToolchain.ps1
# Optional: 16 synthetic black 320x240 frames, Intel QSV, discarded output.
.\Tools\Test-PcaiMediaToolchain.ps1 -ValidateIntelQsv
# Optional: DirectShow device metadata only; its expected nonzero exit is retained.
.\Tools\Test-PcaiMediaToolchain.ps1 -ListCaptureDevices
```

Use `-OutputPath` with a new absolute filename in an existing directory to save
JSON. Existing reports are never overwritten. Reserved Windows components and
alternate data streams are rejected. Each subprocess has a configurable
`-TimeoutSeconds` (default 15), except the first GStreamer invocation. Its
`--version` command can build a cold or stale plugin registry after installation;
`-GStreamerScanTimeoutSeconds` gives that one probe a separate bounded budget
(default 120 seconds, configurable from 1 through 120). Warm scans return as soon
as they finish. GStreamer element metadata probes retain `-TimeoutSeconds`.
Timeouts terminate the owned child process tree.
The result preserves stdout/stderr, exit codes and provider failures. Consult
individual statuses; `CompletedWithIssues` means qualification is incomplete or
a requested probe failed. GStreamer element failures remain in their individual
probe records because hardware-specific elements may be absent on this host.

Python inspection is opt-in and accepts only an explicit existing venv
`Scripts\python.exe` with `pyvenv.cfg` explicitly declaring
`include-system-site-packages = false`. It uses Python isolated mode and never
installs or restores packages:

```powershell
.\Tools\Test-PcaiMediaToolchain.ps1 -PythonPath C:\path\to\media-venv\Scripts\python.exe
```

The report lists installed OpenCV variants, its full build information and CUDA
device count, plus PyAV's actual bundled FFmpeg library versions. Multiple
OpenCV wheel variants are reported as a conflict and cv2 is not imported. A
stock CPU wheel's cv2.cuda namespace does not establish CUDA capability. Keep
the exact repository's selected dependencies in its own environment.

The current Windows FFmpeg build includes QSV/NVENC/D3D11 options. NVENC still
requires a supported connected NVIDIA GPU and driver; this tool never attempts
an NVIDIA runtime workload. The CUDA toolkit is not required merely to run a
prebuilt FFmpeg binary. RTX5060Ti (Blackwell, sm_120) supports AV1 encode; RTX
A2000 (Ampere, sm_86) does not, despite supporting AV1 decode.

GStreamer 1.28.8 uses a unified Inno Setup EXE. Select MSVC x86_64 runtime or
development installation to match the application. The tool searches PATH,
current-user/system Program Files defaults and the older C:\gstreamer path;
`-GStreamerPath` selects an explicit inspect executable. Installing GStreamer
does not enable its backend inside an already-built OpenCV wheel.

Primary references:

- [FFmpeg hardware options](https://ffmpeg.org/ffmpeg-all.html)
- [GStreamer current Windows downloads](https://gstreamer.freedesktop.org/download/)
- [OpenCV wheel variants and CPU-only builds](https://github.com/opencv/opencv-python/blob/4.x/README.md)
- [NVIDIA codec hardware matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)
- [NVIDIA FFmpeg setup](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)

Qualification checks do not replace physical device/HIL acceptance or prove a
remedy for whole-PC freezes. Review the investigation's power and firmware
evidence before adding sustained media workloads.

## Cold registry scan controls

The [metadata report](../Tools/Test-PcaiMediaToolchain.ps1) aggregates every
GStreamer inspection result. A crash, timeout or unexpected nonzero exit makes
GStreamer `Failed` and the overall report `CompletedWithIssues`, even when the
version probe succeeds. Explicitly absent optional elements appear in
`missingOptionalElements`; their original process result and exit code remain in
`probes`. Absence requires the exact `No such element or plugin '<name>'`
diagnostic and exit -1 (Windows) or 255 (POSIX). Other failures remain issues.
`Enumerated` describes completed metadata inspection and does not qualify a GPU,
capture device or media workload. The [regression controls](../Tests/Boot/MediaToolchain.Tests.ps1)
cover crashes, timeouts, unexpected failures and optional absence using process
fixtures, without loading plugins or starting workloads.

The [Pester controls](../Tests/Boot/MediaToolchain.Tests.ps1) reproduce a first
registry scan that needs more than the ordinary metadata budget. They also keep
caller-selected deadlines independent, retain an exhausted scan as a failure,
and prohibit work during dry-run. These rows cover the selectors changed by
the cold-scan correction in the [qualification tool](../Tools/Test-PcaiMediaToolchain.ps1).

Hash normalization: UTF-8 with CRLF converted to LF. Source SHA256:
`0dd6b38ac9010c97c7de7f23b82c3d48f8582f77f898b2f5bd807f863a9474a3`.
Test SHA256: `975a399de8ac335b66b0c9e545932fd018881330a7c949f08fb2fe2af3d4b7d5`.
Review base: `1fd892fdded08d4080337e044343a7626368983b`.
Reviewed on 2026-10-21 by `codex-gen9-fleet-resume`, an independent agent.
Scope: full source/test/document diff and 26 Windows mocked-host and native-process
tests, all passing, including the real hung-child timeout. The cold-scan fixture
fails against the preceding implementation; increasing only the first probe's
budget makes it pass. A custom first-scan budget leaves metadata deadlines
unchanged, and an exhausted scan remains a failure.

| Selector | Reviewer | Disposition |
| --- | --- | --- |
| does no process, provider, discovery or output work in DryRun | codex-gen9-fleet-resume | keep |
| allows a cold registry scan while retaining short metadata deadlines | codex-gen9-fleet-resume | keep |
| keeps custom registry and metadata deadlines independent | codex-gen9-fleet-resume | keep |
| retains a registry scan timeout as incomplete qualification | codex-gen9-fleet-resume | keep |

## Element metadata regression review

Reviewed 2026-11-02 against integration base `816038d4` and the exact hashes below.
Hashes are SHA256 of UTF-8 source with CRLF normalized to LF; selector
hashes cover each complete parameterized Pester `It` command, including its
literal cases. Code under test is the
[media toolchain script](../Tools/Test-PcaiMediaToolchain.ps1), hash
`aac67c7724944a091dd6a0bb4ce8e47c6fc8f8b71aa2678175f5a76e1b4183f8`.

| Selector group in [media tests](../Tests/Boot/MediaToolchain.Tests.ps1) | Selector SHA256 | Reviewer / type | Disposition |
| --- | --- | --- | --- |
| reports &lt;Failure&gt; element metadata as incomplete despite a successful version probe: crashed, timed-out, unexpectedly failed | `81bfd19cad58dd95f2c42bfaae9e70ee8df54956d276be2434a97b262fe75687` | pcai_review / independent agent | keep |
| retains an explicitly missing optional plugin with exit &lt;Code&gt; without claiming runtime validation: -1, 255 | `a91ab09653e520304c2e2bad2324144baa501e9214e2f34c6660bce53d18f23b` | pcai_review / independent agent | keep |

Independent review confirmed that these cases exercise shipping aggregation and
retain raw probe failures. The focused diagnostic run passed 47 tests; three
Integration tests were not run. Version-only aggregation fails the three negative
cases. The positive absence cases permit only an exact diagnostic with the
documented exit convention; timeout and error flags still refuse that exception.
No test assertion was weakened and no runtime qualification was added.

All five new media cases failed against the preceding report: three failures
were incorrectly `Enumerated`, and optional absence had no explicit inventory.
The corrected report retains every process result. The Windows missing-element
metadata control independently produced the exact absent-element diagnostic and
exit `-1`. No crash fixture executes a native plugin, encode or capture workload.
