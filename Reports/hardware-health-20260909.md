# Hardware health — dtm-p1gen7, 2026-09-09

Captured with `Tools\Test-HardwareHealth.ps1`. Machine-readable companion:
[`hardware-health-20260909.json`](hardware-health-20260909.json).

```
pwsh -File Tools\Test-HardwareHealth.ps1 -OutputJson Reports\hardware-health-20260909.json
```

608 devices present · **3 errors** · 0 warnings (regenerated 2026-09-09 after the GPU fix and the Hello correction; matches the companion JSON).

## 1. The internal dGPU is dead, and it is a driver-version conflict

> ## ✅ RESOLVED 2026-09-09 — both GPUs now work, on one driver
>
> ```
> index, name,                                      driver_version, memory.total
> 0,     NVIDIA RTX 2000 Ada Generation Laptop GPU, 610.88,          8188 MiB
> 1,     NVIDIA GeForce RTX 5060 Ti,                610.88,         16311 MiB
> ```
>
> Both report `status=OK, problem=0`. CUDA sees **both**, 24.5 GB total, up from
> 16 GB with the Ada dead.
>
> **What actually fixed it, and the rule worth keeping:** the fault was a
> *version* split, not a *branch* incompatibility. `nvlddmkm.sys` is one shared
> kernel driver, so two packages from different releases (596.36 and 610.88)
> cannot both load and the loser reports problem 31. Installing the single 610.88
> release put both GPUs on one kernel driver.
>
> **Different INFs are a red herring.** One NVIDIA release ships ~45 OEM-specific
> INFs and *no single one of them lists every device*: here `DEV_28B8` is served
> by `nvltsi.inf` and `DEV_2D04` by `nv_dispsi.inf`, from the **same** 610.88
> package, and both work. Chasing "one INF containing every device" is chasing
> something that does not exist.
>
> **Immediately after the install the eGPU reported problem 12** ("cannot find
> enough free resources") — PCIe MMIO/BAR allocation, not an unsupported device.
> Re-enumerating the enclosure cleared it with **no reboot required**.
>
> The earlier conclusion in this section — that the two GPUs were on incompatible
> GeForce-vs-professional branches and could never coexist — was **wrong**, and is
> kept below only as provenance.

| GPU | Bus | Driver | Status |
|---|---|---|---|
| NVIDIA RTX 5060 Ti (eGPU, Razer Core X V2) | 38 | `32.0.15.9636` | OK |
| NVIDIA RTX 5060 Ti (stale enumeration) | 82 | `32.0.15.9636` | not present |
| **NVIDIA RTX 2000 Ada (internal laptop dGPU)** | 1 | **`32.0.16.1088`** | **Error, problem 31** |
| Intel Arc Pro Graphics | 0 | `32.0.101.8517` | OK |

Problem 31 is "Windows cannot load the drivers required for this device".

> **⚠ Corrected 2026-09-09 18:35.** The first version of this section said the
> remedy was "reinstall a single driver covering every NVIDIA GPU". **That is not
> achievable on this machine**, and the correction matters because it sends you
> after something impossible. Verified by reading the driver-store INFs directly:
>
> | Package | `DEV_28B8` (RTX 2000 Ada) | `DEV_2D04` (RTX 5060 Ti) |
> |---|---|---|
> | `nv_dispsi.inf` 596.36 — GeForce branch | **absent** | present |
> | `nvltwi.inf` 610.88 — Lenovo/RTX professional branch | present | **absent** |
>
> **Neither installed package lists both devices**, and `nvlddmkm.sys` is a
> single shared kernel driver — so only one branch can ever be loaded and one GPU
> stays at problem 31 regardless of how many times either is reinstalled. This is
> not a botched install; it is a consequence of pairing a **professional** internal
> GPU (RTX 2000 Ada, served by the RTX/Quadro branch) with a **consumer** eGPU
> (GeForce RTX 5060 Ti, served by the GeForce branch).
>
> Real options, none of which a script should pick:
> 1. **Keep the eGPU** (status quo). CUDA already runs on it, and at 16 GB it is
>    the more capable card. Cost: no NVIDIA GPU when the Core X is detached — the
>    laptop falls back to Intel Arc. Disabling the Ada device stops it erroring
>    and should remove the `nvWmi64.exe` retry cost.
> 2. **Keep the internal Ada** — remove the GeForce package. Cost: the eGPU stops
>    working, and you lose 16 GB of VRAM for the Rust CUDA workloads.
> 3. **Find a single package listing both device IDs.** Not confirmed to exist;
>    would need checking against NVIDIA's driver catalogue.
>
> `Tools\Test-HardwareHealth.ps1` now determines this automatically — it scans the
> driver store and reports whether *any* installed package covers every NVIDIA
> device present, rather than assuming a reinstall will help.

Confirmed from the CUDA side — `nvidia-smi` (driver 596.36, CUDA 13.2) enumerates
**only** the 5060 Ti. The RTX 2000 Ada is invisible to CUDA, so every CUDA feature
in the Rust workspace (`pcai-media`, candle CUDA, llama.cpp CUDA) is running on
the eGPU only, and silently loses the internal GPU whenever the Core X is
detached.

Plausibly also a boot cost: `nvWmi64.exe` appears in the post-boot slow list at
10.9 s average across 9 hits, which is consistent with a driver retry loop.

**Remedy (needs a human, not a script):** a clean reinstall of a single NVIDIA
driver covering both GPUs. Not done here — it is a reboot-affecting change.

## 2. Windows Hello — CORRECTED: it was already working

> ## ⚠ The section below is WRONG and is kept only as provenance
>
> It claims zero Hello credentials are enrolled. **That verdict came from a
> swallowed access denial, not from an empty store.**
>
> The NGC credential directory is ACL'd to `SYSTEM` and `NgcCtnrSvc` **only** —
> Administrators are excluded — so enumerating it fails *even elevated*. With
> `-ErrorAction SilentlyContinue`, that denial returns an empty collection which
> is indistinguishable from "nothing is enrolled". The check could not fail
> loudly, so it failed quietly and confidently.
>
> **What is actually enrolled**, read from sources that are readable:
>
> | Factor | Source | State |
> |---|---|---|
> | PIN | Passport KSP — `uvkey-E9F2E36D…` | ✅ enrolled |
> | Face | `WinBio AccountInfo\<SID>\EnrolledFactors = 2` | ✅ enrolled |
> | Fingerprint | bit 8 of the same value, unset | ❌ not enrolled |
> | FIDO passkeys | Passport KSP | 4 registered, incl. `GOOGLE_ACCOUNT:107425477980575938280` |
>
> So Hello has been working for this account all along. **The only real gap is
> fingerprint**, and the Synaptics sensor is healthy (`status=OK`, driver
> 6.0.69.1136, 2026-05-24) — the two `1609` secure-connection errors were a
> single 40-second episode on 09-03, not a standing fault. Adding fingerprint is
> an interactive enrolment, nothing more.
>
> `WbioSrvc` being Stopped is **not** a fault either, and was misreported as one
> twice. It is `Start=2` but carries RPC start triggers: LogonUI starts it on
> demand at the lock screen and it idle-stops afterwards, so Stopped is its
> normal resting state mid-session. Starting it by hand does not stick — verified
> by doing exactly that and finding it stopped again hours later.
>
> The tool now reads enrolment from WinBio, reports service state as context
> only, and surfaces recent `1609` events instead.

### Original (incorrect) section, retained for provenance

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
