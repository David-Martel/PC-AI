# Cloud cache layout on F: (cloud-cache-disk) and boot gating

- **Date:** 2026-09-08
- **Machine:** dtm-p1gen7
- **Scripts:** `Tools\Restore-CloudCacheLayout.ps1`, `Tools\Start-CloudClientsAfterVHDX.ps1`,
  `Tools\Install-CloudClientBootGating.ps1`, `Tools\Set-DriveFsContentCache.ps1`

## The F: drive

`F:` is a **dynamic VHDX** at `D:\vm\cloud-cache-disk.vhdx`, label `cloud-cache-disk`,
2,200 GB, NTFS. It is attached at boot by the scheduled task
`AutoMount_VHDX_cloud-cache-disk` -> `Tools\Mount-PersistentVHDX.ps1`
(boot trigger + 30s delay, SYSTEM, RunLevel Highest). That task is healthy and was
**not modified**; its last runs all report `ExitCode 0`.

## What was found

| Location | Logical | Real (materialized) | Verdict |
|---|---:|---:|---|
| `C:\Users\david\University of Michigan Dropbox` (ACTIVE) | 1,017 GB | **0 GB** | all 570,056 files online-only |
| `F:\University of Michigan Dropbox` (stale) | 7,313 GB | **1,166.6 GB** | orphaned - see below |
| `C:\...\AppData\Local\Google\DriveFS` (ACTIVE cache) | 5.6 GB | 5.6 GB | on C:, should be on F: |
| `F:\Google` (old cache) | 0 GB | 0 GB | correct structure, empty |
| `F:\OneDrive - Auricle Inc` | 24.5 GB | 0 GB | OneDrive Business account no longer configured |
| `F:\Proton-Drive` | 0 GB | 0 GB | Proton Drive has no configured root |

**The F: Dropbox tree was orphaned.** Its placeholders fail to open with
`The cloud file provider is not running` - the Dropbox client had been repointed to
`C:`, leaving 1.17 TB of real data plus ~393k dead reparse points stranded on F:.

**Google Drive mount points** (virtual, no local storage): `G:` = auricleinc,
`J:` = umich, `T:\cloud-cache\google` = a third account. Registry lists account
`105016...` as `H` but it actually mounts at `G:` - harmless drift, noted only.

## Changes made

### 1. Orphaned folders archived on F: (`Restore-CloudCacheLayout.ps1`)

In-volume NTFS renames - **metadata only, 0 bytes copied, completed in 0.2s total**:

| From | To |
|---|---|
| `F:\University of Michigan Dropbox` | `F:\_archive\UMich-Dropbox-stale-20260908` |
| `F:\OneDrive - Auricle Inc` | `F:\_archive\OneDrive-AuricleInc-stale-20260908` |
| `F:\Proton-Drive` | `F:\_archive\Proton-Drive-stale-20260908` |

Nothing deleted. Reverse with `Move-Item <dest> <source>`. The script refuses to run
unless F: carries the `cloud-cache-disk` label, and it reads the live Dropbox
`info.json` + OneDrive registry to build a protected list of **active** sync roots so
it can never archive a root a client is using.

### 2. Google Drive content cache moved back to F: (`Set-DriveFsContentCache.ps1`)

`HKCU:\SOFTWARE\Google\DriveFS\ContentCachePath`:
`C:\Users\david\AppData\Local\Google\DriveFS` -> **`F:\Google`**

DriveFS application state (SQLite DBs, logs, pid) correctly stays on C:; only the
content cache moves. Verified after restart: `G:` and `J:` mounted,
`T:\cloud-cache\google` browsable, and `F:\Google` repopulating (1,448 files / 1.72 GB).
Restore with `-CachePath 'C:\Users\david\AppData\Local\Google\DriveFS'`.

### 3. Boot gating installed (`Install-CloudClientBootGating.ps1`)

**The race this fixes:** the VHDX mounts at *boot*+30s as SYSTEM, but Dropbox and
Google Drive autostarted from *Run keys at logon*. On the last boot the margin was only
~70s (mount 09:13:29, logon 09:14:39). Losing that race starts a sync client whose
root/cache is missing - the most plausible cause of the C: relocation found above.

New task **`CloudClients-AfterVHDX`** (logon trigger, runs as `david`,
**RunLevel Limited**) -> `Start-CloudClientsAfterVHDX.ps1`, which:

- waits up to 180s for a volume whose **label** is `cloud-cache-disk`
  (never `Test-Path F:\` alone - a USB stick can take F:);
- with `-RequireMountLog`, also requires the newest `AutoMount_VHDX_cloud-cache-disk`
  `*.result.json` to report `ExitCode 0`, so a *degraded* mount does not pass;
- starts `GoogleDriveFS --startup_mode` and `Dropbox /systemstartup`, skipping any
  already running (idempotent), resolving version-numbered install dirs by glob;
- **if the volume never appears, starts nothing and exits 2.**

Racing Run keys were removed **only after** the task was proven to run
(`LastTaskResult 0`), and their values were backed up first:

| Hive | Name |
|---|---|
| `HKCU:\...\CurrentVersion\Run` | `GoogleDriveFS` |
| `HKLM:\SOFTWARE\WOW6432Node\...\CurrentVersion\Run` | `Dropbox` |

> Dropbox is 32-bit, so its machine Run key is under **WOW6432Node** - a 64-bit
> PowerShell does not see it at the plain HKLM Run path. The first dry-run reported
> "Dropbox: not present" because of exactly this; the script now checks both hives.

Backup: `C:\codedev\PC_AI\Logs\CloudClientStart\runkey-backup.json`
Undo: `pwsh -File Tools\Install-CloudClientBootGating.ps1 -Rollback`

> The backup **merges** with any existing file. An earlier revision overwrote it,
> and because a re-run sees already-removed keys as "not present", the Dropbox
> entry was silently dropped and `-Rollback` could not have restored it. Fixed,
> and the backup was repaired by hand; it now holds both entries.

### Gating is durable for Dropbox, best-effort for Google Drive

**Dropbox**: its `WOW6432Node` Run key stays removed. Fully gated.

**Google Drive**: re-creates its own HKCU Run key every time it starts, so the
removal does not stick. Setting the documented
`HKLM\SOFTWARE\Policies\Google\DriveFS\AutoStartOnLogin=0` override was tried and
**reverted** - measured on this machine:

| Attempt | Result |
|---|---|
| policy=0, launch with `--startup_mode` | process exits instantly (the flag declares a login start, which the policy forbids) - gated launcher could not start Drive |
| policy=0, launch without `--startup_mode` | process survives, but `G:` and `J:` never mounted |
| either way | Drive re-added its HKCU Run key regardless |

The policy delivered nothing and broke Google Drive, so it is not used; the
installer now refuses to set it and clears it on `-Rollback`. Drive therefore
still autostarts itself at logon and can in principle beat the mount. That is
tolerable: Drive's `ContentCachePath` is a *cache*, and it recreates the
directory when F: appears - unlike a sync **root**, it does not relocate. The
launcher remains useful for Drive as a restart-if-missing safety net.

### 4. Mount delay reduced 30s -> 5s (`Set-VhdxMountDelay.ps1`)

Because Google Drive re-adds its Run key and therefore cannot be held behind the
launcher, the only lever that actually widens its margin is mounting F: sooner.
`AutoMount_VHDX_cloud-cache-disk` now uses `PT5S` on the trigger **and**
`-StartupDelaySeconds 5` in the action - both, since changing one leaves the other
sleeping. The mount script keeps its own `MountTimeoutSeconds` and retry handling,
and the VHDX is on internal NVMe, so the long fixed delay bought nothing.

Expected margin: previously mount at boot+30s vs logon ~127s (~70s); now boot+5s
(~95s+). Re-verified after the edit: task `ExitCode 0`, `DegradedReasons` none,
F: healthy.

## Verification performed

| Check | Result |
|---|---|
| Archive renames | 3/3 moved, 0.2s, F: free unchanged at 1,031 GB |
| Active roots protected during archive | C: Dropbox + C: OneDrive correctly skipped |
| Launcher, volume present | detects in 3.7s, validates mount log, idempotent skip |
| **Launcher, volume ABSENT (negative control)** | **exit 2, clients NOT started** |
| **Launcher, run elevated (negative control)** | **exit 4, refuses** |
| Install ordering | task verified `LastTaskResult 0` before Run keys touched |
| DriveFS after cache move | running; `G:`/`J:`/`T:\cloud-cache\google` all mounted |
| End-to-end | DriveFS restarted *by the gated launcher itself*, exit 0 |
| Mount task after delay edit | `ExitCode 0`, no degraded reasons, F: healthy |
| Full health sweep 2026-09-08 | all clients running; G:/J:/T: mounted; cache on F:; both tasks `lastResult=0` |

## Gotcha worth remembering

**Cloud sync clients must not be started elevated.** Launching DriveFS from an elevated
shell makes it exit immediately (`G:`/`J:` never appear). The scheduled task runs
`RunLevel Limited`, which is correct; the launcher now detects elevation and exits 4
rather than silently failing. Override with `-AllowElevated` only if you know why.

## Not done (deliberate)

- **Dropbox sync root left on `C:`.** Moving it was offered and not selected. It costs
  ~0 bytes to move (the tree is entirely placeholders) and the F: path is now clear, so
  this can be done any time via the Dropbox client's *Preferences -> Sync -> Move* -
  never by moving the folder by hand.
- ~~`AutoMount_VHDX_cloud-cache-disk` left untouched~~ - **superseded 2026-09-08**, see
  "Mount delay reduced" below.
- The 1.17 TB in `F:\_archive\UMich-Dropbox-stale-20260908` is **kept**. Before ever
  deleting it, verify nothing is stranded locally:
  `rclone check --one-way --size-only "F:\_archive\UMich-Dropbox-stale-20260908\David Martel" "dropbox-umich:/David Martel"`
  (`--size-only` compares metadata, so unreadable placeholders do not break the check).
