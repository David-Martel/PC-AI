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
