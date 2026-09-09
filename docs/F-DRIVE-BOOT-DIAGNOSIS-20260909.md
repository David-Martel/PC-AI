# Why F: did not mount at boot or logon — dtm-p1gen7, 2026-09-09

## ✅ RESOLVED 2026-09-09 11:13 — F: is internal again, cloud clients running

Design rule adopted (user directive): **external drives are ABSENT by default. Nothing in the
boot path or the cloud-sync path may depend on one.** Externals are opportunistic cache only.

What made this easy: **the only thing that ever needed F: was Google Drive's content cache**
(`HKCU:\SOFTWARE\Google\DriveFS\ContentCachePath = F:\Google`), which re-fetches on demand.
Dropbox's sync root is on **C:** (`C:\Users\david\University of Michigan Dropbox\...`) and never
needed F: at all. So no 2.2 TB re-sync was required — only a cache rebuild.

| | before | after |
|---|---|---|
| F: backing | `D:\vm\cloud-cache-disk.vhdx` on an **external** 8 TB Thunderbolt disk | **`T:\vm\cloud-cache-disk.vhdx`** on internal T: |
| F: size | 2,200 GB volume / 1,860 GB file | 500 GB dynamic, 0.16 GB on disk at creation |
| mount task | `PT5S`, no wait, path on D: | `PT1M30S`, `-WaitForVhdSeconds 15`, path on T: |
| cloud clients | not started (gate exit 2) | **GoogleDriveFS + Dropbox running**, G:/J: mounted |
| boot path | depended on a removable disk | **fully internal (C:/T:)** — verified |

Verified end-to-end through Task Scheduler, not by hand:
`AutoMount_VHDX_cloud-cache-disk` → `ExitCode 0`; launcher logged
`Mount log OK (ExitCode 0, this boot, ...)`; `F:\Google` populating (72 → 790 items in 2 min).

The old 1,860 GB cache VHDX is untouched on the 8 TB enclosure. It is **not** needed and should
not be moved back.

> **⚠ Measure T: free before quoting it.** This document contains two different figures
> (~1.37 TB early in the session, ~2.1 TB later) because T: free genuinely moved ~600 GB during
> the session with **no deletion and no `Optimize-Volume -ReTrim`**. At 2026-09-09 11:40 it is
> stable at **1,970.1 GB** (`Get-Volume` and `Get-PSDrive` agree; `fsutil` reports
> 2,115,371,925,504 bytes) and held flat over a 20 s sample. **The ~600 GB change is unexplained**
> — it is *not* the documented ReFS allocator behaviour (allocator pressure does not release
> space; ReTrim does) and *not* live writers (T:'s VHDXs total ~10 GB). Do not reason from a
> remembered number; re-measure.

### Hardening added 2026-09-09 11:28 — F: now self-heals

`Tools\Repair-CloudCacheMount.ps1` + the **`CloudCache-MountWatchdog`** task (SYSTEM/Highest,
boot+3 min and logon, then **every 15 minutes**).

The boot mount task is single-shot, and Task Scheduler's restart-on-failure does **not** re-fire
on a nonzero exit — measured: `ExitCode 51` with `RestartCount=3` and `LastRunTime` never
advancing. So one missed mount used to cost the whole session. Now the worst case is ~3 minutes.

> **⚠ Corrected 2026-09-09 12:40 — the "~3 minutes" claim above did NOT hold as first
> installed.** A repetition pattern only arms when its trigger *fires*. The watchdog was
> registered with boot and logon triggers only, after this session's boot had already
> happened, so it armed nothing: it ran once at 11:35 and at 12:26 still had `NextRunTime`
> empty on a nominal 15-minute interval. Between installing the watchdog and the next
> reboot the worst case was **unbounded**, not 3 minutes — the exact window the watchdog
> exists to cover. Fixed by adding a third, already-fired trigger so the cycle arms at
> install time; `NextRunTime` went from empty to 12:42:59 on re-registration. The ~3 minute
> figure is correct only from the next boot onward, or after that fix.
>
> Ruled out, so nobody re-litigates it: this was **not** `StopAtDurationEnd`. That element
> defaults to `true` while `Duration` is left empty, which looks wrong. An A/B on a
> throwaway task with a time trigger showed *both* `true` and `false` schedule a NextRun 15
> minutes out — empty `Duration` repeats indefinitely, as documented. Trigger type was the
> whole story.
>
> Found by `Tools\Test-ScheduledTaskHealth.ps1`, which flagged the watchdog as stalled on
> its first run.

Design rules it obeys:
- Identifies the volume by **label, never drive letter** (a removable can take F:).
- Only mounts from **internal** `T:\vm\cloud-cache-disk.vhdx`. If the label appears backed by
  anything else it reports and refuses (`exit 3`) rather than calling it healthy.
- **No-op when healthy**, and it restarts the cloud clients *only if it actually repaired the
  mount* — so a deliberately-quit Dropbox stays quit.

Verified by real fault injection, twice — dismounting the VHDX and, in the second run, also
killing Dropbox:

| run | result |
|---|---|
| direct script | volume gone → repaired in **14 s**, launcher started |
| **via the SYSTEM task** | volume gone + Dropbox killed → task `0x0` in **22 s**, volume back, **both clients running** |

The second run is the one that matters: it proves SYSTEM can start the *interactive* launcher
task and get clients back in the user's session.

**Sizing is settled, not guessed.** `HKLM:\SOFTWARE\Policies\Google\DriveFS\ContentCacheMaxKbytes`
= 157,286,400 KB = **150 GB hard policy cap**. F: at 500 GB is 3.3× the maximum the cache can
ever reach, so it cannot thrash and cannot fill T:.

Also fixed: `Set-VhdxMountDelay.ps1` still *defaulted* to `-DelaySeconds 5` — running it would
have silently re-broken the mount. Default is now 90.

### Hardening round 2 — 11:40

Three gaps found by review after the above was already "verified":

1. **The `WrongBacking` branch had never executed** — both fault-injection runs went
   `Missing → Healthy`, so the one branch that fires *when the enclosure comes back and its old
   copy grabs the label* was untested. It contained two strict-mode hazards: `.DriveLetter` on a
   null `Get-Partition` result, and `(... | Select-Object -First 1).Location` on an empty set.
   Under `Set-StrictMode -Version 2.0` those **throw**, so the watchdog would have died with an
   unhandled exception instead of the designed `exit 3`. Extracted a guarded `Get-DiskDriveLetter`
   helper and re-tested by pointing `-ExpectedVhd` at a path that does not back the volume:
   now logs the real backing and exits **3**, no crash. A healthy run still exits 0.

2. **Client restart is now gated on an interactive session existing.** The launcher runs as
   `david / Interactive` and `Start-Process`es GUI clients. Checked the principal:
   `LogonType = Interactive`, which means Windows itself will not run it with nobody logged on —
   so the session-0 hazard was already prevented by the OS. The gate makes that explicit and
   *logged* (a `WARN` naming why nothing started) instead of a silent no-op, and it is accurate
   about the recovery: the launcher carries its own **zero-delay logon trigger**, so the clients
   come up at the next logon. The mount half is unaffected and still runs headless as SYSTEM.

3. **The watchdog existed only because it was registered live** — it was in no install script, so
   a rebuild from the repo would have silently lost self-healing. Now registered by
   `Install-CloudClientBootGating.ps1` **STEP 1b** (and removed by `-Rollback`). Verified by
   registering a throwaway copy from the installer's own code and diffing it against the running
   task: principal and triggers match exactly. *(Re-verified 2026-09-09 12:45 after the trigger
   fix below — the task now carries **three** triggers, not two, and the live task and the
   installer's output are byte-identical once registration timestamps are normalised.)* Gotcha preserved in a comment — the repetition
   `Duration` must be `''` (empty = indefinite); `[TimeSpan]::MaxValue` produces
   `P99999999DT23H59M59S`, which `Register-ScheduledTask` rejects outright.

Full fault injection was re-run after these edits: repaired in **14 s**, task `0x0`, both clients
running, and the launcher correctly skipped Google Drive (already running) while starting only
Dropbox — the "don't fight the user" rule holding under a real repair.

### Still outstanding: W:
`shared-dev.vhdx` (~306 GB, 710,352 files) is **real data, not cache** — it was NOT recreated.
Its task stays armed at `T:\vm\shared-dev.vhdx` and returns `0x33` each boot until the file is
restored from the 8 TB enclosure, at which point it self-heals. Use
`Tools\Restore-DrivesToInternal.ps1` (locate the enclosure **by size, not by letter**).

---

## Verdict (original diagnosis)

The Task Scheduler startup/logon sequence is **not** broken. Every task fired, in order, with
the right delays, and the cloud-client gate did exactly what it was written to do.

**F: did not mount because its backing file lives on a Thunderbolt disk that is not connected
to this machine right now.** `AutoMount_VHDX_cloud-cache-disk` points at
`D:\vm\cloud-cache-disk.vhdx`. D: is an **8 TB WD_BLACK SN850X inside an ACASIS TBU405Pro
Thunderbolt 3 enclosure**. That enclosure enumerated on the 07:59 boot and F: mounted
normally; it has not enumerated on either boot since.

This is proven, not inferred: **F: mounted successfully at 08:00:22 today** (2.20 TB, label
`cloud-cache-disk`, NTFS, Healthy, `ExitCode: 0`).

## What actually happened — three boots today

| Time | Event | F: mount result |
|---|---|---|
| 07:59:36 | Boot #1 (after an unclean shutdown — `Kernel-Power 41`) | |
| 07:59:52 | TB router `Intel - Module 2 PD 85W` arrives | |
| 07:59:54 | TB router **`ACASIS - TBU405Pro` arrives** | |
| 07:59:57 | `WD_BLACK SN850X 8000GB` arrives (3 s behind its router) → D: | |
| 08:00:22 | `AutoMount_VHDX_cloud-cache-disk` runs | **0x0 SUCCESS** — F: mounted, 2.20 TB |
| 08:50:15 | Clean shutdown | |
| 09:26:26 | Boot #2 — **no ACASIS, no D:** | |
| 09:27:41 | mount runs | **0x33 (51)** `Missing VHDX file: D:\vm\cloud-cache-disk.vhdx` |
| 09:47:46 | Boot #3 — still no ACASIS | |
| 09:48:21 | `AutoMount_VHDX_cloud-cache-disk` (boot+5 s) | **0x33 (51)** same error |
| 09:48:21 | `CloudClients-AfterVHDX` (logon, no delay) starts gate | |
| 09:49:17 | `AutoMount_VHDX_shared-dev` (boot+1 m) | **0x33 (51)** `Missing D:\vm\shared-dev.vhdx` → no W: either |
| 09:49:47 | `AutoMount_VHDX_share-ext4` (boot+1 m 30 s) | **0x0** — its path is `T:\vm\share-ext4.vhdx`, internal |
| 09:51:55 | `CloudClients-AfterVHDX` times out | **0x2** — clients deliberately NOT started |

**The one mount task that succeeded is the only one whose VHDX lives on T:.** That is the
whole diagnosis in a sentence.

The Thunderbolt topology also changed between boot #1 and boot #2. At 07:59 the chain was
`Intel Module 2 PD 85W` + `ACASIS TBU405Pro`. Currently attached: `Plugable TBT4-UDZ`,
`CalDigit Element 5 Hub`, `Razer Core X V2`, `Focusrite`. Different dock, no ACASIS.

## Evidence

- `Get-PSDrive` / `Get-Volume`: only **C:** and **T:** exist. No D:, F:, or W:.
- `Get-Disk`: three disks — two internal NVMe `WD_BLACK SN850X 4000GB` (PCI Slot 8 = T:,
  Slot 10 = C:) plus one `File Backed Virtual` = `T:\vm\share-ext4.vhdx`. Both M.2 slots are
  accounted for; **no 8 TB disk is attached.**
- **Neither `cloud-cache-disk.vhdx` nor `shared-dev.vhdx` exists on any attached volume.**
  Verified with a reparse-point-skipping `[System.IO.Directory]::EnumerateFiles` walk
  (`IgnoreInaccessible`, `AttributesToSkip = ReparsePoint|Offline`) over C:\ and T:\, **with a
  positive control per root that passed in both cases** (`Mount-PersistentVHDX.ps1` on C:,
  `share-ext4.vhdx` on T:). Found 25 `.vhdx` on C: (largest 74.5 GB, Ubuntu WSL) and 11 on T:
  (largest 517.5 GB, Ubuntu ext4). Neither target is among them.
  Corroborating arithmetic: T: used is 1,883 GB while the F: volume is 2,362 GB — it cannot
  fit inside T:'s used space at all.
  *Caveat:* the walk skips `Offline`-attributed files, so a fully dehydrated cloud placeholder
  would not be seen — implausible for a multi-TB VHDX, but not formally excluded.

  ⚠ **Methodology note — two earlier searches silently produced nothing and were nearly believed.**
  A background `Get-ChildItem -Recurse` over C:\,T:\ aborted with `Win32Exception` and returned
  empty; a `cmd /c dir /s /b` emitted only a shell banner. A third attempt's control *passed on
  T: but aborted on C:* while still printing `(no hits)`. On this machine an empty search result
  is worthless without a positive control — see [[reference-windows-disk-survey-method]].
- Mount result JSON, verbatim: `"Errors": ["Missing VHDX file: D:\\vm\\cloud-cache-disk.vhdx"]`,
  `"ExitCode": 51`, `"MountAttempted": false`.
- **The disk is in the ACASIS — confirmed by 1:1 arrival/removal correlation, every time:**

  | ACASIS TBU405Pro router | SN850X 8000GB | Removed together |
  |---|---|---|
  | arr 09/03 13:08:19 | arr 09/03 13:08:22 | both 09/03 15:33:5x |
  | arr 09/03 16:14:58 | arr 09/03 16:15:02 | both 09/04 17:59:54 |
  | arr 09/07 09:12:47 | arr 09/07 09:12:51 | both 09/08 12:43:15 |
  | arr 09/08 20:53:14 | arr 09/08 20:53:19 | both 09/08 22:10:5x |
  | arr 09/09 07:59:54 | arr 09/09 07:59:57 | — |

- The disk's NVMe controller parent is `PCI\VEN_8086&DEV_15EF` (Intel Thunderbolt PCIe
  bridge), `LocationPaths` = `ACPI(_SB_)#ACPI(PC00)#ACPI(TRP0)#ACPI(PXSX)...` — a Thunderbolt
  **root port**, i.e. PCIe-tunnelled, not an internal slot.
- Six stale PnP instance IDs for the 8 TB disk, each at a different PCI bus address — the
  signature of repeated hot-plug at varying positions.
- `disk/157 "Disk 2 has been surprise removed"` + a `disk/153` I/O-retry storm on Disks 3 and 4
  at 09/08 12:43:15, when the ACASIS, the Intel Module 2 and a Cable Matters router all
  dropped simultaneously — an unclean yank of the whole chain, not a safe eject.

## Why this is possible at all

Per the 2026-08-29 topology work, `shared-dev.vhdx` (W:) was moved off T: onto D: and its task
was correctly updated. `cloud-cache-disk.vhdx` (F:) was later moved to D: too, and that task
was updated as well. Both moves were done correctly — the tasks point at the real current
locations.

The defect is architectural, not clerical: **two boot-critical VHDXs now live on a removable
Thunderbolt disk, while their mount tasks assume a fixed internal disk present a few seconds
after POST.** F: is the cache root for Dropbox and Google Drive, so every boot without that
enclosure attached takes the cloud clients down with it.

## The cloud-client gate is working correctly — do not loosen it

`Start-CloudClientsAfterVHDX.ps1` deliberately refuses to start Dropbox and Google Drive when
the `cloud-cache-disk` volume is absent, validating by **volume label**, not drive letter. Its
own comment gives the reason: starting a sync client whose cache root is missing silently
relocates the sync root onto C:. **Exit 0x2 is a successful safety trip, not a bug.** With F:
at 2.2 TB, letting the clients start would have been far worse than not starting them.

## Real defects found (all still latent — none caused today's failure)

1. **`-RequireMountLog` has a staleness race.** Lines 155–156 take the newest `*.result.json`
   by `LastWriteTime` with no check that it belongs to the current boot. Today that directory
   holds an 08:00:22 `ExitCode: 0` alongside two later failures — exactly the shape that lets
   a *previous* boot's success satisfy the gate on a boot that failed. The 180 s label poll
   caught it first this time, so it stayed latent, but the log gate is currently capable of
   the opposite of its documented purpose.
   **Fix:** require the chosen JSON's `StartedAt` to be later than
   `(Get-CimInstance Win32_OperatingSystem).LastBootUpTime`, and treat "no JSON for this boot"
   as a failure rather than a `WARN`.
2. **No wait, no effective retry on a missing file.** `Mount-PersistentVHDX.ps1` fails
   immediately (line 459) with no poll loop. The task carries
   `RestartCount=3 / RestartInterval=PT1M`, but `LastRunTime` never advanced past 09:48:21 —
   Task Scheduler's restart-on-failure did not re-fire on a nonzero exit. One missed
   enumeration window costs the whole session.
3. **boot+5 s is a thin margin for a Thunderbolt disk — untested at cold boot.** The F: task
   uses `PT5S`; the sibling that succeeds uses `PT1M30S` on an internal disk. On boot #1 the
   ACASIS took ~18 s from OS start to enumerate and the disk 3 s more. I have no measurement
   of the boot-trigger path with the enclosure attached (boot #1's successful mount ran at
   08:00:22, well after the boot+5 s window), so treat this as a plausible race rather than a
   proven one — but 5 s is clearly inside the risk zone for TB enumeration and authorization.
4. **No device-arrival trigger anywhere.** Nothing re-attempts the mount when D: is plugged in
   after boot. Today's 08:00:22 and 09:27:41 runs were on-demand, not triggered — the current
   recovery path is manual.

## Recommended fixes, in order

1. **Reconnect the ACASIS TBU405Pro — but do NOT expect `D:\vm\...` to come back.**
   ⚠ Superseded mid-session by a second finding: **the letter `D:` is now held by a different
   disk** — `FATDRIVE`, FAT32, 953.6 GB, on a Sabrent USB SSD (Disk 3), holding audiometric
   calibration data. FAT32 caps a file at 4 GB, so that volume can never hold these VHDXs, and
   the returning 8 TB enclosure will be assigned some *other* letter because D: is occupied.
   Locate it by size, never by letter:
   `Get-Disk | Where-Object Size -gt 7TB | Get-Partition | Select DriveLetter`
   Then run `Tools\Restore-DrivesToInternal.ps1 -DryRun` (its Step 0 refuses on a non-NTFS or
   undersized source, verified against the FAT32 disk) and adjust the source path to whatever
   letter the enclosure actually received.

   **The mount tasks have already been repointed to `T:\vm\`** (the intended destination), so
   they now fail fast and honestly until the files are moved there, rather than polling a disk
   that is present and wrong.
2. **Decide where these two VHDXs should live.** If F: is to remain boot-critical with cloud
   sync roots on it, it belongs on an internal disk — but note F: is now **2.2 TB** and T: has
   ~2.1 TB free, so it will not fit as-is; it would need compaction or a size reduction first.
   Otherwise, accept that F: is removable and stop treating it as a boot-time guarantee.
3. **If it stays on D:**, replace the boot trigger with a device/volume-arrival trigger (event
   trigger on `Microsoft-Windows-Kernel-PnP`, or a `Win32_VolumeChangeEvent` watcher), or add
   a bounded poll loop for the VHDX path in `Mount-PersistentVHDX.ps1` and push the delay well
   past Thunderbolt authorization. Add `-ExpectedDiskUniqueId` so a re-enumerated bus address
   cannot defeat identification. Give `CloudClients-AfterVHDX` the same arrival-driven retry.
4. **Fix the `-RequireMountLog` staleness race** as described above.

## Flagged — outside the scope of this question

- **`D:\repo-backups\` is recorded as holding sole copies of four repos** (rust-tkhm,
  litho-workspace, pixel, litho-qmd-meta-20260720), with
  `D:\staging-for-deletion\DO-NOT-WIPE.md` as that volume's authority. Those are offline while
  the enclosure is detached — and the 09/08 12:43:15 drop was a surprise removal with
  in-flight I/O retries, so check that volume is clean when it returns.
- `Kernel-Power 41` at 07:59:40 — the 07:59 boot followed an unclean shutdown. Unrelated to
  the mount failure, but worth knowing about.
- `DevEnvironmentStartup` is **Disabled** (last ran 2026-01-25, exit 0). Left alone.
- `UnifiUdmDriveStackStartup` Disabled (2026-04-30, 0x32); `iSCSIAgentAutoStartup` still
  Running (0x41301). Neither is implicated.

## Memory note to correct

`reference_dtm_p1gen7_disk_topology.md` is now stale in two ways: it records F: →
`T:\vm\cloud-cache-disk.vhdx` and warns the task "still points at T:", when in fact F: was
relocated to D: **and** the task was updated; and it implies D: is fixed internal storage when
it is a Thunderbolt enclosure. It also lists F: at 1400 GB; it is now 2.20 TB.
