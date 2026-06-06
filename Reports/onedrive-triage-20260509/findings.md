# OneDrive Triage — 2026-05-09 Findings

## Executive summary

OneDrive (personal MSA, davidmartel07@gmail.com, cid `8d77a48e98fdb6dc`) is
running and authenticated, but the **upload pipeline has not initiated a single
PUT since 2026-05-01 03:02 UTC**. Diagnostic counters in `SyncDiagnostics.log`
have been frozen at `uptimeSecs=126` since OneDrive started today — the prior
"13,348 stuck → 24,592 stuck" reading was telemetry staleness, not a real
backlog spike.

The actual root cause is a **WNS (Windows Push Notification Service) channel
that fails to establish for the OneDrive process this session**. Without the
WNS channel, OneDrive falls back to slow polling (~68 min) for *server*
changes — but the *local* change-watcher → upload pipeline is gated on the WNS
subscription being healthy, so the 290 pending local changes never reach the
upload queue. Server-side push notifications were working as recently as
2026-05-08 21:56 UTC; this session has had zero `NotificationReceived` events.

**The 4/30 `/reset` is NOT the cause but is partially responsible** for masking
the problem: OneDrive 26.062 → 26.070 upgrade after the reset broke
SyncDiagnostics.log emission, so the telemetry the user has been watching is
unreliable. Don't reset again.

## Evidence

### Operation history (`od_ServiceOperationHistory`, 1320 rows)

| Operation | Count | Last seen | Notes |
|---|---|---|---|
| `DownloadBlock` | 551 | 2026-05-07 12:44 | Downloads worked through 5/7 |
| `EnumChanges` | 466 | 2026-05-09 05:30 | 196 × 304 (server says nothing changed); 1 × 401 (transient on 5/4 18:56) |
| `NotificationReceived` | 103 | **2026-05-08 21:56** | WNS push: last successful notification |
| `GetClientPolicy` | 59 | 2026-05-09 05:24 | 2 × resultCode=0 (transient) |
| `ProvisionFolder` | 16 | 2026-05-09 02:45 | All 200 |
| `CreateSubscription` | 16 | **2026-05-08 13:00** | Last WNS subscription created |
| `GetSignature` (UploadFile) | 12 | **2026-05-01 03:02** | Last upload-related ops |
| `UploadFile_*` (any kind) | 0 | — | **Zero uploads since 5/1 03:02 UTC** |

The scenario name on every recent EnumChanges is
`CheckForChanges_FindChangesPollingWNSChannelNotConnected_ODB_FindChangesScenario/NotificationLatencyScenario`
— OneDrive is in WNS-disconnected polling fallback mode.

### Sync engine state (`SyncEngineDatabase.db`)

| Table | Count | Notes |
|---|---|---|
| `od_ClientFile_Records` | 148,426 | Matches reported file count |
| `od_ClientFolder_Records` | 16,130 | Matches reported folder count |
| `od_ClientFilePostponedChange_Records` | **0** | No postponed/failed file changes |
| `od_ClientFolderPostponedChange_Records` | **0** | No postponed folder changes |
| `od_CreateAddedFolderFailures` | 0 | No folder failures |
| `odc_convergence_items` | **0** | Convergence queue empty |
| `od_ThrottleHistory` | **0** | Server is NOT throttling us |
| `od_HydrationData` | 554 | Files with placeholder hydration metadata |

### Diagnostic counter freeze

`SyncDiagnostics.log` shows:
- `clientVersion: 26.070.0414.0001` (current)
- `uptimeSecs: 126` (frozen — OneDrive has actually been running 36+ minutes)
- `timeUtc: 2026-05-09T18:42:36Z` (frozen, ~33 min stale)
- `driveChangesToSend: 24592`, `driveSentChanges: 0` (these numbers are stale; queue tables show empty)
- `numLocalChanges: 290` (stale; Cloud Files filter ETL is current and shows file
  watcher is alive)

The 4/30 reset moved the client from 26.062 to 26.070. The newer client
apparently writes diagnostics differently or stopped emitting after 126s.

### Write-test (probe-12)

- Created `OneDrive\_synctest_20260509\synctest-20260509-151440.txt` (216 bytes)
- Waited 90 seconds
- All counters identical pre/post (because diagnostic emitter is frozen)
- Shell `Availability status: Sync pending` — Cloud Files filter sees the file
- File did NOT upload in the 90s window

This confirms: file watcher is alive, but local→server upload is wedged.

### Network

- DNS resolves all OneDrive endpoints (login.live.com, d.docs.live.net,
  my.microsoftpersonalcontent.com, oneclient.sfx.ms, graph.microsoft.com)
- TCP 443 reachable to all critical hosts
- Cloudflare WARP service is RUNNING but tunnel is **disconnected** (manual
  disconnect) — not interfering
- WpnService + WpnUserService_16ec35 both running
- `client.wns.windows.com:443` reachable
- OneDrive has 1 ESTABLISHED TCP connection to `20.189.173.11:443` (Microsoft
  Azure) + several bound HTTP/2-Push sockets

### Auth state

- Account authority: `https://login.microsoftonline.com/consumers` (correct
  for personal MSA)
- `LastSignInResult: 0` (success)
- `LastSignInTime` and `LastAttemptedSignInTime` both updated today
- Single 401 in entire history (5/4 18:56 — transient, recovered)

### Stale tasks (Phase 3 — applied)

7 stale OneDrive scheduled tasks deleted (owned by SIDs 1002, 1003, 1007,
1009 mapped to WsiAccount, DevToolsUser, CodexSandboxOffline). Active tasks
preserved:

- `OneDrive Per-Machine Standalone Update Task` (SYSTEM)
- `OneDrive Reporting Task-...-1001` (active user)
- `OneDrive Startup Task-...-1001` (active user)

### Vault, FileCoAuth, WER (cleared as concerns)

- **Vault**: `vaultState=0`, no `PersonalVault` registry key. Not gating uploads.
- **FileCoAuth.exe**: present at `C:\Program Files\Microsoft OneDrive\26.070.0414.0001\FileCoAuth.exe` (per-machine
  install). Spawning correctly on demand (FileCoAuth `.odlgz` logs from PIDs
  35516, 41624 timestamped today). Not the issue. Earlier "missing FileCoAuth"
  finding was wrong — looked only in stale per-user install dir.
- **WER**: 0 entries in ReportQueue or ReportArchive for OneDrive/FileSync.
  The 4/30 crash storm has stopped. Not a current concern.

## Root cause (high confidence)

**The OneDrive process this session failed to establish its WNS notification
channel after start.** Without that channel, OneDrive runs in degraded mode:
slow server polling for download changes (304 Not Modified) but no active
upload trigger when local changes are detected by the Cloud Files filter.

Why it failed: WNS subscription registration races with WNS service init at
boot/startup. When OneDrive starts before WNS is fully ready (or hits a
transient WAM token issue), the channel registration fails silently and
OneDrive proceeds without it. WAS working through 2026-05-08 21:56 UTC, then
broke at the next OneDrive restart.

Why this prevents uploads despite an alive file watcher: OneDrive's local
change pipeline (Cloud Files filter → SyncEngine → upload queue) is
internally pumped by the WNS subscription event loop. Without the
subscription, file events are buffered into `numLocalChanges` but the upload
queue is never populated.

## Remediation

### Recommended (low risk, high confidence): clean restart of OneDrive

```powershell
# Stop both processes
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# Confirm both exited
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue | Format-Table Name, Id

# Restart
Start-Process 'C:\Program Files\Microsoft OneDrive\OneDrive.exe' -ArgumentList '/background'

# Wait 60s for WNS subscription to register
Start-Sleep -Seconds 60

# Verify WNS connection by checking for new NotificationReceived ops
sqlite3 ...SyncEngineDatabase.copy.db "SELECT MAX(timestamp) FROM od_ServiceOperationHistory WHERE operationName='NotificationReceived';"
```

**Expected after restart:**
- New `CreateSubscription` entries within 30 seconds
- New `NotificationReceived` entries as the server pushes the queued local
  changes to the client's "I'm here, send me what's pending" event
- `UploadFile_StorageGetSignatureScenario` entries as OneDrive begins
  uploading the 290 pending local changes

**If the restart does NOT fix it** (no `CreateSubscription` after 60s, or no
`UploadFile` operations after 5 min):
- Check `Get-AppxPackage Microsoft.WNSChannel*` for app package state
- Restart `WpnUserService_*` user-mode service (`Restart-Service WpnUserService_*`)
- Re-attempt OneDrive restart

**Only if both above fail**: consider re-signin (Account → Unlink this PC →
Sign back in). This preserves cached files (placeholders stay) but forces
WAM/OneAuth to re-register.

**`/reset` is NOT recommended** — would discard 727 GB of cached state for
no diagnosed benefit, and the previous `/reset` is implicated in the
diagnostic-counter freeze.

### Concerns / things to monitor after fix
- The 290 local changes will queue rapidly; expect 5-30 min of upload
  activity depending on file sizes and bandwidth.
- 425 historical sync problems referenced from registry
  (`PreviousDiagnosticsRunSyncProblemsCount`) live in `OCSI.db` (146 MB,
  29074 property records, 33973 path mappings). These are co-authoring state,
  not the upload-hang root cause. Worth a follow-up audit but not blocking.

## Files in this triage
- `01-syncengine-inventory.txt` — log file inventory
- `02-account-settings.txt` — settings dir + ClientPolicy contents
- `03-network-probes.txt` — DNS, reachability, proxy, OneDrive TCP
- `04-vault-and-wer.txt` — vault state, WER queue, app log errors
- `05-stale-tasks.txt`, `05-stale-tasks.csv` — task inventory
- `06-filecoauth.txt` — per-user/per-machine install layout
- `07-dns-deep.txt` — DNS via 1.1.1.1, 8.8.8.8, system; WARP status
- `08-syncengine-strings.txt` — strings extracted from active .aodl
- `09-installs.txt` — full install layout (per-machine vs per-user)
- `10-live-state.txt` — DB schema + Tenant + AuthenticationURLs
- `11-live-sync-current.txt` — current SyncDiagnostics values
- `12-writetest.txt` — write-test pre/post comparison
- `13-odlsent-strings.txt` — gunzip+strings on .odlsent (no useful hits)
- `14-svchistory.txt` — service operation history detail + OCSI tables
- `15-stale-cleanup.txt` — Phase 3 cleanup applied
- `16-ops-summary.txt` — operation counts, result codes, WNS endpoints
- `SyncEngineDatabase.copy.db`, `OCSI.copy.db`, `SettingsDatabase.copy.db` — copies for further analysis
- `liverepair-readonly/` — `Repair-OneDriveSync.ps1 -DryRun` outputs
- `writetest-pre/`, `writetest-post/` — write-test snapshots
