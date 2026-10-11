# Windows media qualification

Run the qualification tool from a fresh `pwsh -NoLogo -NoProfile -NonInteractive`
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
`-TimeoutSeconds` (default 15); timeouts terminate the owned child process tree.
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

GStreamer 1.28.7 uses a unified Inno Setup EXE. Select MSVC x86_64 runtime or
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
