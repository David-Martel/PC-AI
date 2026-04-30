---
context_id: pcai-nvidia-option-b-20260410
created_at: 2026-04-10
branch_at_capture: main
session_id: deb85b54-c43f-48d7-bbb4-cb8540d9b277
topic: NVIDIA Option B execution — staged 595.97 Enterprise install for DEV_28B8
---

# NVIDIA Option B — Filtered 595.97 Enterprise install (DEV_28B8 only)

## Goal
Install newest available NVIDIA drivers on this Lenovo ThinkPad while keeping both
the RTX 5060 Ti (DEV_2D04) and RTX 2000 Ada Laptop (DEV_28B8) functional.
User selected **Option B** from the 2026-04-10 morning menu (filtered staged-package
install via repo tooling, no nvidia.com download required).

## Pre-install state
| Card | Status | Problem | Driver | INF |
|---|---|---|---|---|
| RTX 5060 Ti | OK (one instance) + Unknown ghost | 0 | 32.0.15.9186 (591.86 GRD) | oem387.inf |
| RTX 2000 Ada | Error | 31 (CM_PROB_FAILED_ADD) | 32.0.15.8241 (582.41 Enterprise) | oem242.inf |

## Execution
- Tool: `C:\codedev\PC_AI\Tools\Install-InfDriver.ps1` (pnputil wrapper)
- Wrapper script: `C:\codedev\PC_AI\.pcai\tmp\nvidia-install-optionb.ps1` (loop over 6 INFs)
- Source: `C:\ProgramData\NVIDIA Corporation\NVIDIA app\UpdateFramework\ota-artifacts\grd\post-processing\224337314adfbc25ca5e95bff03ee45f\Display.Driver`
- Target INFs: nvltwi, nvblwi, nvdmwi, nvfmwi, nvmiwi, nvmsowi (all touch DEV_28B8)
- Explicitly avoided: `nv_dispwi.inf` (per memory: would disturb 5060 Ti binding)

## Per-INF results
| Original | Published | Driver | Notes |
|---|---|---|---|
| nvltwi.inf | oem127.inf | 32.0.15.9597 | Exit 3010 (reboot required) |
| nvblwi.inf | oem188.inf | 32.0.15.9597 | OK |
| nvdmwi.inf | oem392.inf | 32.0.15.9597 | OK |
| nvfmwi.inf | oem393.inf | 32.0.15.9597 | OK |
| nvmiwi.inf | oem394.inf | 32.0.15.9597 | OK |
| nvmsowi.inf | oem395.inf | 32.0.15.9597 | OK |

In-loop verifier (`-VerifyHardwareId '*VEN_10DE*DEV_28B8*'`) reported `Status -eq OK`
on every INF, showing the 2000 Ada immediately picked up the new binding.

## Post-install state (after `pnputil /scan-devices`, no reboot yet)
| Card | Status | Problem | Driver | INF | Delta |
|---|---|---|---|---|---|
| RTX 2000 Ada | **OK** | **0** | **32.0.15.9597** | oem127.inf | **FIXED** |
| RTX 5060 Ti | **Error** | **31** | 32.0.15.9186 | oem387.inf | **REGRESSED** |
| RTX 5060 Ti (ghost) | Unknown | — | 32.0.15.9186 | oem387.inf | unchanged |

`nvidia-smi`:
```
0, NVIDIA RTX 2000 Ada Generation Laptop GPU, 595.97, 8188 MiB
```
(5060 Ti not visible — driver not loaded)

## Diagnosis
Same WDDM kernel-mismatch pattern as the morning's original Code 31 on the 2000 Ada,
just with the cards swapped. Newly installed 595.97 `nvlddmkm.sys` is now newer than
what the 591.86 GRD `oem387` binding can load on the 5060 Ti. The kernel-incompatibility
flag has migrated from DEV_28B8 to DEV_2D04 because the kernel binaries are shared
across the graphics subsystem.

## Local recovery search (read-only)
Searched for any INF on disk that owns DEV_2D04 newer than 591.86:
- `C:\ProgramData\NVIDIA Corporation\NVIDIA app\UpdateFramework` — only the 591.86 GRD package (already installed as oem387)
- `C:\NVIDIA`, `C:\Drivers`, `C:\Downloads`, `$env:USERPROFILE\Downloads` — none found
- `pnputil /enum-drivers` — only `nv_dispig.inf` (oem387 @ 591.86) covers DEV_2D04

**Local recovery is exhausted.** Enterprise packages don't include `nv_dispig.inf`
because consumer DEV_IDs aren't in the Enterprise product line.

## Driver store snapshot post-install
| Published | Original | Version | Notes |
|---|---|---|---|
| oem127.inf | nvltwi.inf | 32.0.15.9597 | NEW — bound to 2000 Ada |
| oem188.inf | nvblwi.inf | 32.0.15.9597 | NEW |
| oem242.inf | nvltwi.inf | 32.0.15.8241 | OLD 582.41, still in store, superseded |
| oem387.inf | nv_dispig.inf | 32.0.15.9186 | 591.86 GRD, still bound to 5060 Ti |
| oem392.inf | nvdmwi.inf | 32.0.15.9597 | NEW |
| oem393.inf | nvfmwi.inf | 32.0.15.9597 | NEW |
| oem394.inf | nvmiwi.inf | 32.0.15.9597 | NEW |
| oem395.inf | nvmsowi.inf | 32.0.15.9597 | NEW |

## Decision pending
Three options for next step (presented to user):
1. **Reboot first** — nvltwi reported 3010, official "complete install" step. Low cost,
   unlikely to recover 5060 Ti without matching kernel but worth trying before invasive options.
2. **Option C (manual GRD download)** — user fetches matching GRD from nvidia.com that
   pairs with 595.97 Enterprise WDDM (likely 597.xx or newer GRD). Then filtered install
   via Install-InfDriver against `nv_dispig.inf` only.
3. **Rollback** — `pnputil /delete-driver oem127,oem188,oem392,oem393,oem394,oem395 /uninstall /force`
   restores prior state (5060 Ti OK, 2000 Ada Code 31). Net loss but predictable recovery.

## Files written this session
- `C:\codedev\PC_AI\.pcai\tmp\nvidia-baseline.ps1` — pre-install state check
- `C:\codedev\PC_AI\.pcai\tmp\nvidia-install-optionb.ps1` — Install-InfDriver loop wrapper
- `C:\codedev\PC_AI\.pcai\tmp\nvidia-postinstall.ps1` — scan-devices + post-install verification
- `C:\codedev\PC_AI\.pcai\tmp\nvidia-find-grd.ps1` — DEV_2D04 INF search + driver store enum

## Recommendation given to user
**Option C (manual nvidia.com download)** is the only path to a 2-card-OK state.
Reboot is cheap to try first since nvltwi reported 3010 anyway. Rollback is the safe
escape via `pnputil /delete-driver oem127,oem188,oem392,oem393,oem394,oem395 /uninstall /force`.

## Hand-off state at save
- Tasks 1-4 all completed in tracker
- Memory updated: `project_nvidia_split_driver.md` reflects new post-Option-B state
- Branch: `main` @ `5e7d591` (clean, no commits made this session)
- Driver state changed but **nothing committed to git** — all artifacts in `.pcai/tmp/` (gitignored)
- Awaiting user choice between: reboot / Option C / rollback
