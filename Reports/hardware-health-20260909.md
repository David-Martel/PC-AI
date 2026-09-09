# Hardware health — dtm-p1gen7, 2026-09-09

Captured with `Tools\Test-HardwareHealth.ps1`. Machine-readable companion:
[`hardware-health-20260909.json`](hardware-health-20260909.json).

```
pwsh -File Tools\Test-HardwareHealth.ps1 -OutputJson Reports\hardware-health-20260909.json
```

606 devices present · **5 errors** · 1 warning.

## 1. The internal dGPU is dead, and it is a driver-version conflict

| GPU | Bus | Driver | Status |
|---|---|---|---|
| NVIDIA RTX 5060 Ti (eGPU, Razer Core X V2) | 38 | `32.0.15.9636` | OK |
| NVIDIA RTX 5060 Ti (stale enumeration) | 82 | `32.0.15.9636` | not present |
| **NVIDIA RTX 2000 Ada (internal laptop dGPU)** | 1 | **`32.0.16.1088`** | **Error, problem 31** |
| Intel Arc Pro Graphics | 0 | `32.0.101.8517` | OK |

Problem 31 is "Windows cannot load the drivers required for this device".
Windows ships **one** NVIDIA driver package per system, so two NVIDIA devices on
different versions is the ordinary cause: the second one cannot load.

Confirmed from the CUDA side — `nvidia-smi` (driver 596.36, CUDA 13.2) enumerates
**only** the 5060 Ti. The RTX 2000 Ada is invisible to CUDA, so every CUDA feature
in the Rust workspace (`pcai-media`, candle CUDA, llama.cpp CUDA) is running on
the eGPU only, and silently loses the internal GPU whenever the Core X is
detached.

Plausibly also a boot cost: `nvWmi64.exe` appears in the post-boot slow list at
10.9 s average across 9 hits, which is consistent with a driver retry loop.

**Remedy (needs a human, not a script):** a clean reinstall of a single NVIDIA
driver covering both GPUs. Not done here — it is a reboot-affecting change.

## 2. Windows Hello — three separate things, only one of them fixed

Readiness needs all three. Reporting only the sensors is how "the hardware is
fine so Hello should work" happens.

| Signal | State found | Action |
|---|---|---|
| Biometric devices | 2 present, both OK (Synaptics UWP WBDI, Facial Recognition) | none needed |
| `WbioSrvc` | **Stopped**, though `START_TYPE = 2 AUTO_START` | **started 2026-09-09**; auto-starts from now on |
| NGC credential store | **0 containers** | **needs interactive enrolment** |

`WbioSrvc` being stopped was a genuine fault, not a trigger-start service idling:
`sc qc` reports `AUTO_START`. It also carries RPC-interface start triggers, which
is why it can appear to work on demand while never running at boot.

The load-bearing finding is the NGC store: **zero containers means no Windows
Hello credential is enrolled at all.** No service or driver repair changes that.
It has to be re-enrolled by hand in Settings → Accounts → Sign-in options.

## 3. Keyboard and touchpad — no device-level fault today

Every input device reports **OK**, including `HID-compliant touch pad`,
`TrackPoint Device`, `ETD HSA Device` (Elan), and all five keyboard nodes.

The two USB `problem 43` errors are **not** input devices — their parents are hub
controllers (`VID_2188`, `VID_2230`) at `Port_#0005.Hub_#0016` and
`Port_#0004.Hub_#0006`, i.e. failed enumerations on a dock/hub, not on a keyboard
or touchpad.

So the previously recorded touchpad/keyboard complaints are not currently a
hardware fault. If glitches persist they are behavioural under I/O load, which
matches the existing hypothesis in [`boot.TODO.md`](../boot.TODO.md) (sync-client
I/O contention) and the measured post-boot contention below.

## 4. Stale virtual adapter

`Cisco AnyConnect Virtual Miniport Adapter` reports Error while exposing **no** CM
problem code — the signature of a virtual adapter left behind by an uninstalled
VPN client. Harmless, but it is one of the five errors and will stay until the
device is removed.

## Related measurement — post-boot, not the task chain

Captured the same day from `Microsoft-Windows-Diagnostics-Performance/Operational`
across three boots:

| Boot | MainPathBoot | PostBoot |
|---|---|---|
| 08:02 | 453 ms | 75.5 s |
| 09:28 | 454 ms | 91.1 s |
| 09:50 | 455 ms | **97.7 s** |

The kernel/driver path is excellent and flat; the cost is entirely post-boot, and
it is **growing**. Top named contributors by average: `chrome.exe` 51.2 s,
`explorer.exe` 48.7 s, `svchost.exe` 47.6 s (max 180 s), `HapticService.exe`
46.1 s, `OfficeClickToRun.exe` 34.6 s, **`MsMpEng.exe` 27.3 s across 19 hits**,
`PhoneExperienceHost.exe` 17.5 s across 17 hits.

**No PC_AI scheduled task appears in the top 18.** Re-sequencing the VHDX/cloud
task chain would not measurably improve boot; the cost is vendor startup software
and Defender. Defender's 19 hits are consistent with the recorded
exclusion-rot problem.
