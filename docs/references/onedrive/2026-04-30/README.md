# OneDrive Repair References - 2026-04-30

Local HTML copies in this directory were captured during the 2026-04-30
OneDrive/touchpad repair session so future diagnostics can be compared against
the exact Microsoft guidance used at the time.

## Captured Sources

- `microsoft-support-reset-onedrive.html`
  - Source: https://support.microsoft.com/en-gb/office/reset-onedrive-34701e00-bf7b-42db-b960-84905399050c
  - Used for the reset semantics: reset rebuilds OneDrive sync state and causes
    a full sync, while preserving local files.
- `microsoft-support-onedrive-restrictions-limitations.html`
  - Source: https://support.microsoft.com/en-us/office/restrictions-and-limitations-in-onedrive-and-sharepoint-64883a5d-228e-48f5-b3d2-eb39e07630fa
  - Used for sync limits, path/name restrictions, unsupported mapped-drive and
    junction/symlink sync locations, and the 300,000 item performance warning.
- `microsoft-learn-sharepoint-sync-process.html`
  - Source: https://learn.microsoft.com/en-us/sharepoint/sync-process
  - Used for the OneDrive sync model and WNS/realtime sync context.
- `microsoft-learn-onedrive-group-policy.html`
  - Source: https://learn.microsoft.com/en-us/sharepoint/use-group-policy
  - Used to distinguish supported OneDrive policy registry settings from
    generic Windows performance tweaks.
- `microsoft-learn-noremote-recursive-events-context.html`
  - Source: https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/security-setting-changes-not-appear-dfsr-replication
  - Used to scope `NoRemoteChangeNotify` and `NoRemoteRecursiveEvents` as
    remote/DFS-style Explorer notification controls, not local OneDrive
    sync-root performance knobs.

## Installer Evidence

The OneDrive installer was downloaded from Microsoft's official redirect URL:

`https://go.microsoft.com/fwlink/p/?LinkID=2182910`

The downloaded repair artifact was not committed to avoid adding a near-100 MB
vendor binary to the repo. Evidence retained in the repair report:

- Path at capture time:
  `Reports\onedrive-repair\20260430-install-repair\OneDriveSetup.exe`
- SHA256:
  `15ECC74898C2C7B5120EE735879850E89FE9DD58CC80D18289BA601B0FDA5310`
- Size:
  `100860264` bytes
