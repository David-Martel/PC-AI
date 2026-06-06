# OneDrive Triage — Final Verdict (2026-05-09)

## Diagnostic Outcome: Local Diagnostics Exhausted

After WAM cache clear (Step A), Defender + Process Lasso optimizations (Step C), WPN service restart, and AppX/event-log verification (probe-28), the WNS channel registration failure for OneDrive **persists** and is no longer locally remediable.

## What probe-28 told us

### AppX package is healthy
```
Name        : Microsoft.OneDriveSync
Version     : 26070.414.1.0
Status      : Ok
PackageUserInformation : {}     # EMPTY — no per-user registration
```
- `Status: Ok` → package itself is not corrupt; **AppX re-register will not help**.
- `PackageUserInformation: {}` is unusual but not actionable from PowerShell — it suggests the per-user activation state is missing, which is exactly what unlink/relink rebuilds.

### WNS event log — silent for OneDriveSync
- 100 events in last 24h on `Microsoft-Windows-PushNotifications-Platform`.
- **Zero events** mention `Microsoft.OneDriveSync_8wekyb3d8bbwe` or `OneDrive` UWP.
- All `Microsoft.SkyDrive.Desktop` events show `[ErrorCode] The operation completed successfully` — that's the legacy non-UWP shim, succeeding.
- No HRESULT to debug because **the UWP sync engine is never even attempting WNS channel registration**. That's why the `WNSPushChannel` rows are missing — the request never goes out.

### NEW finding — FileSyncHelper service is crashing
```
5/9/2026 3:56:15 PM  7034  Error  Service Control Manager
The FileSyncHelper service terminated unexpectedly. It has done this 3 time(s).
5/9/2026 3:49:47 PM  7034  Error  ... 2 time(s).
5/9/2026 3:29:28 PM  7034  Error  ... 1 time(s).
```
Three crashes today — 3:29 (early triage), 3:49 (post Step A clear), 3:56 (post Step A restart). Every OneDrive restart crashes the helper service. This is plausibly a symptom of the broken per-user state, not a separate root cause.

## Remediations applied this session (kept)

| Action | Result | Reversible |
|---|---|---|
| Stale-SID OneDrive scheduled tasks (7) deleted | ✅ Done | No (already invalid) |
| Defender folder exclusion `~\OneDrive` | ✅ Added | `Remove-MpPreference -ExclusionPath` |
| Defender process exclusions (5 OneDrive .exe) | ✅ Added | `Remove-MpPreference -ExclusionProcess` |
| Process Lasso `onedrive.exe` CPU: BelowNormal → Normal | ✅ Reloaded | `prolasso.ini.bak-stepC-20260509-155608` |
| Process Lasso `onedrive.exe` IO: 1 → 2 | ✅ Reloaded | same backup |

These are independently good hygiene per Microsoft's published guidance and will pay off as soon as sync resumes — but they are **not what's blocking sync**.

## What's left = user-driven (Step B)

The only remaining legitimate action is to **unlink and relink the OneDrive personal account** through the GUI:

1. Right-click OneDrive cloud icon in system tray → Settings → Account
2. Click **Unlink this PC** (under the personal account `davidmartel07@gmail.com`)
3. OneDrive will sign out and show the setup wizard
4. Sign in with `davidmartel07@gmail.com`, accept the existing local OneDrive folder
5. Wait for re-scan (will rebuild the per-user AppX activation state and request a fresh WNS channel)

This rebuilds:
- `PackageUserInformation` for the UWP sync engine
- `WNSPushChannel` row in `wpndatabase.db`
- WAM token graph (already clean post-Step A)

**Risk:** None to local files (OneDrive folder stays in place). 13,348 pending changes will be re-evaluated against the cloud — files newer locally upload, files newer in cloud download. Standard "first-run" reconciliation.

## Confidence summary

- **High confidence** WNS channel is the proximate block: `WNSPushChannel` rows missing for OneDrive UWP NotificationHandlers despite `WpnService` healthy and SkyDrive.Desktop registering fine.
- **High confidence** local automation is exhausted: WAM clear regenerated tokens but didn't trigger channel request; package is `Status: Ok` so re-register is not indicated; service restart didn't help; nothing in event logs to act on.
- **Medium confidence** unlink/relink will fix it: rebuilds the exact state (per-user AppX activation + WNS channel) that's currently malformed.
