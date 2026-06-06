# OneDrive Triage v2 — 2026-05-09 (post broader investigation)

## Executive summary (revised)

OneDrive (davidmartel07@gmail.com / personal MSA / cid `8d77a48e98fdb6dc`) is
running, signed-in, and reaches Microsoft's servers (12+ established TCP
connections to Azure/Office endpoints). But it **cannot register a Windows
Push Notification (WNS) channel** for any of its 4 notification handlers. As
a result, OneDrive runs in degraded `WNSChannelNotConnected` polling mode and
its **upload pipeline never fires** for the 290+ pending local changes.

### What was tried (and why each was insufficient)
| Action | Result |
|---|---|
| 1. Stale-SID OneDrive task cleanup (7 deleted) | ✅ Done. Independent value — removed registry/scheduler clutter. |
| 2. Process restart (orphan OneDrive.exe in elevated session) | ❌ Failed — wrong session integrity |
| 3. Process restart via user's Startup Task (correct session) | ❌ Sign-in completed but no WNS channel registered |
| 4. WpnUserService restart + Startup Task | ❌ Same — OneDrive starts cleanly but channel registration silently fails |

After step 4, OneDrive shows:
- Sign-in successful (`SyncEngineSignIn` scenario completed, GetQuotaInfo 200, ProvisionFolder 200)
- 12 ESTABLISHED TCP connections to Microsoft endpoints
- **0 `CreateSubscription` operations** (would happen if channel registration succeeded)
- **0 `NotificationReceived` operations**
- All `EnumChanges` ops still in `WNSChannelNotConnected` scenario
- WPN database shows OneDrive's NotificationHandlers exist (RecordIds 2951, 2952, 3163, 1060) but **none has a corresponding WNSPushChannel entry**

## Root cause (revised, high confidence)

**OneDrive's WAM-mediated PushNotificationChannelManager call is failing
silently to obtain a channel from the WNS service.** WNS service is alive,
`client.wns.windows.com:443` is TCP-reachable, OneAuth tokens for the MSA
account are fresh (rewritten today at 15:40:56). But the OneDrive UWP
package (`Microsoft.OneDriveSync_8wekyb3d8bbwe`) cannot complete the channel
handshake. This is a **post-update auth/identity state issue** that won't
clear by service or process restart alone — it requires either:
1. Clearing OneDrive's WAM token cache (forces fresh sign-in flow)
2. Unlink + re-link the account (cleanest)

The trigger date (last successful `NotificationReceived` = 2026-05-08
21:56:47 UTC) lines up with the OneDrive 26.062 → 26.070 update. The
upgrade left a token state that newer 26.070 client cannot use to complete
WNS channel handshake.

## Other findings (broader investigation)

### Process Lasso configuration (probe-19) — CONCERN

OneDrive ecosystem is configured with Below-Normal CPU + IO Priority 1 (Low):
- `OneDrive.exe`: BelowNormal CPU, IO=1 (Low)
- `OneDrive.Sync.Service.exe`: BelowNormal, IO=1
- `FileSyncHelper.exe`: BelowNormal, IO=1
- `FileCoAuth.exe`: not in DefaultPriorities (default Normal)

**Verdict**: Likely NOT the WNS root cause — OneDrive sustained 5 MB/s IO
during scan, so it isn't IO-starved. But Microsoft recommends OneDrive run
at default priority. Consider relaxing the IO=1 throttle to IO=2 (Normal-Low)
to prevent edge-case starvation when other heavy IO is concurrent.

### Startup interference (probe-20) — Multiple cloud sync providers

Currently running sync ecosystem:
- **OneDrive** (primary) ✓
- **Cloudflare WARP** Running but tunnel disconnected (org=auricleinc Zero Trust)
- **WireGuardManager** Running
- **Tailscale** Running
- **GoogleDriveFS** in HKCU Run (executable not running today)
- **Dropbox** in HKLM Run (not running)
- **Proton Drive** in HKCU Run (not running)
- **iCloud** sync root present but provider not running (per boot.TODO)

The tunnel software (WARP/WireGuard/Tailscale) is harmless but is a class
of software that can interfere with OneDrive when active. WARP-`exclude`
mode with `WarpWithDnsOverHttps` could affect OneDrive when reconnected;
keep WARP **disconnected** when OneDrive needs to recover.

### OneDrive configuration (probe-21) — issues identified

- **No HKLM/HKCU Group Policies** on OneDrive — no policy lockouts (good)
- **Files-On-Demand state**: registry value blank — likely default-on (placeholders enabled)
- **KFM**: 3 folders protected (`KfmFoldersProtectedNow=3584`)
- **Storage Sense**: enabled with cloud-file consent
- **148K files / 16K folders** in sync root (large but Microsoft-supported)
- **Cloud Files filter**: 180,451 instances tracked (matches file+folder count)
- **Defender exclusions**: only ONE OneDrive folder excluded (a MATLAB
  example dir). **Microsoft recommends excluding the OneDrive folder and 4
  process names from Defender real-time scanning** — currently 0/4 process
  exclusions for OneDrive/FileSyncHelper/FileCoAuth/OneDrive.Sync.Service.
  This is a measurable performance issue.
- **WebView2 Runtime**: 147.0.3912.98 installed ✓
- **OneDrive UWP package**: `Microsoft.OneDriveSync_8wekyb3d8bbwe` v26070.414.1.0 ✓
- **`UpdateBeginTimestampTryCountODSU = 14`** — OneDrive Standalone Updater has tried 14 times. Not failing now (current install completed) but historical noise.
- **Storage Provider Sync Roots key empty** — odd but not blocking

### Web research findings

- **Windows 11 24H2 broke OneDrive Personal** for many users (matches the
  4/30 crash storm in this PC's ledger; resolved after the reset). [Microsoft Q&A — OneDrive Personal keeps crashing after Windows 11 24H2 upgrade](https://learn.microsoft.com/en-us/answers/questions/5395229/onedrive-personal-keep-crashing-after-upgrading-to)
- **Session 0 / non-interactive launch breaks sync** (matched my orphan
  OneDrive issue). [Core Technologies — OneDrive 23.48 in Session 0](https://www.coretechnologies.com/blog/alwaysup/onedrive-version-23-48-problem/)
- **Reset OneDrive** is Microsoft's standard fix for WNS-class issues but
  doesn't apply here (the prior reset is implicated). [Microsoft Support — Reset OneDrive](https://support.microsoft.com/en-us/office/reset-onedrive-34701e00-bf7b-42db-b960-84905399050c)
- Process Lasso Below-Normal IO **can** starve a process — confirmed by
  Bitsum docs but not the issue here per measurement.

### Stale-SID task cleanup (Phase 3) — applied
- Deleted: 7 OneDrive tasks owned by SIDs 1002, 1003, 1007, 1009 (mapped
  to UNRESOLVABLE / WsiAccount / DevToolsUser / CodexSandboxOffline).
- Preserved: 3 valid tasks (SYSTEM Per-Machine Update, user-1001 Reporting + Startup).

## Recommendation: 3-step remediation (in order, gated)

### Step A (recommended first): Clear WAM token cache for OneDrive

Less destructive than full re-signin. Forces OneDrive to re-acquire tokens
on next launch:

```powershell
# Stop OneDrive
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 5

# Clear the WAM token cache for this account (8d77a48e98fdb6dc)
$wamRoot = "$env:LOCALAPPDATA\Microsoft\OneAuth"
Get-ChildItem "$wamRoot\accounts\8d77a48e98fdb6dc*" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem "$wamRoot\blobs\8d77a48e98fdb6dc*" -ErrorAction SilentlyContinue | Remove-Item -Force

# Restart via Startup Task
Start-ScheduledTask -TaskName "OneDrive Startup Task-$([System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value)"
Start-Sleep 90
# Re-check WNSPushChannel for HandlerId 2951/2952/3163
```

Expected: OneDrive prompts to confirm sign-in (or auto-signs-in via WAM
fallback to system identity), establishes fresh WNS channel, sync resumes.

### Step B (if A fails): Account unlink + re-link via UI

Microsoft's official next step. Preserves cached files (placeholders
remain on disk).

```
1. Right-click OneDrive tray icon → Settings → Account → Unlink this PC
2. Click sign-in dialog → enter davidmartel07@gmail.com
3. Choose "Use this folder" (DO NOT pick a different location)
4. Skip the tour
```

Expected: full re-link forces fresh WAM identity registration, fresh WNS
channel, sync resumes within 5 min of sign-in.

### Step C (independent optimization, recommended regardless of A/B)
Adjust Process Lasso & Defender for OneDrive performance:

```ini
# In prolasso.ini DefaultPriorities — change from:
onedrive.exe,below normal
# to:
onedrive.exe,normal

# In DefaultIOPriorities — change from:
onedrive.exe,1
# to:
onedrive.exe,2
```

And add Defender exclusions:
```powershell
Add-MpPreference -ExclusionPath "$env:USERPROFILE\OneDrive"
Add-MpPreference -ExclusionProcess @(
  'C:\Program Files\Microsoft OneDrive\OneDrive.exe',
  'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\FileSyncHelper.exe',
  'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\FileCoAuth.exe',
  'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\OneDrive.Sync.Service.exe'
)
```

### Things that would NOT help

- **Another `/reset`**: prior reset (4/30) is implicated; would lose 727 GB of placeholder state and not fix the WNS handshake.
- **Reinstalling OneDrive**: doesn't clear WAM tokens (those live in OneAuth).
- **Restarting WpnService/WpnUserService alone**: already tried, didn't help.
- **Killing competing sync providers**: none are running (only OneDrive is).

## Files
- `findings.md` (initial findings)
- `findings-v2.md` (this file — updated post broader investigation)
- `19-processlasso-audit.txt`, `20-startup-interference.txt`,
  `21-config-audit.txt`, `23-wns-deep.txt`, `24-wpn-channels.txt`,
  `25-wpn-restart.txt` (probe outputs)
- `SyncEngineDatabase.live3.db`, `OCSI.copy.db`, `wpndatabase.copy.db`
- All previous `01-` through `17-` probes from initial investigation
