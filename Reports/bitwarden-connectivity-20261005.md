# Bitwarden and connectivity maintenance — 2026-10-05

The canonical unlock launcher now uses the installed validated machine backend,
reports genuine process failure, and exposes nonmutating Help/DryRun/WhatIf modes.
The DTM-P1GEN7 installed copy matches the reviewed canonical bytes; its destination
DACL was preserved and verified after atomic replacement. The existing scheduled
task already points at this canonical source; its definition was not changed or run.

## Software and runtime evidence

- Actual Pester suite: **49 passed, zero failed/skipped/not-run**, 97.33 seconds.
  All 28 existing archive/bootstrap cases remain unchanged after LF normalization;
  21 additive cases cover the real adapter and nine actual child entry points.
- Both changed PowerShell files: zero parser errors and repository analyzer findings;
  formatter applied, `git diff --check` clean. Independent frozen-source review clear.
- Failure fixtures require native exit 1 for false/null/throwing providers, rollback,
  strict Boolean success and a nonblank validated process session. Diagnostics use
  fixed text, including Quiet and WarningAction Stop. No raw provider output escapes.
- Installed Help/DryRun/WhatIf each exited 0 with no stderr. A separate live bounded
  child established an accepted session: **locked → unlocked**, exit 0, no stderr,
  and User `BW_SESSION` unchanged. Force and User persistence were not requested.
  The backend may restore its protected session cache or write a validated session;
  this proves neither a fresh password unlock nor a parent-shell session repair.
- No archive/export/sync, full secret hydration, task execution or OneDrive profile
  replacement was requested by the canary. Actual User persistence and next-logon
  task execution remain separate acceptance checks.

Frozen raw SHA-256:

| File | SHA-256 |
|---|---|
| `Tools/SystemScripts/Machine/Unlock-BwVault.ps1` | `cf88030d3185b96fb602a891f2fa1b4b8f1b672b9b4ad78894da02e36453d59c` |
| `Tests/LocalMachine/CredentialRepairs.Local.ps1` | `9b9d631c3292f80ca5ac4864f2b9774dd2c0a2c860020aac2673273801bce8cb` |

Failed attempts are retained: the old launcher returned native success for three
provider failures; an intermediate full run passed 48/49, then the whole-suite
harness hit 180 seconds. The final suite budget was 300 seconds to accommodate
inherited integration work; the **20-second child deadline stayed unchanged**.
Native `$PSCmdlet.WriteWarning` avoids optional warning-cmdlet autoload in the
failure path. Different-window timings do not isolate a system-wide speedup.

## Connectivity qualification

- Private `asuspro13-lan` HostName repaired from stale `.79` to verified wired `.2`.
  Effective SSH options changed only that address; pinned host identity, public-key
  authentication, keepalives and compression policy stayed intact. Real strict-key
  hostname validation succeeded. The primary switched fabric route is retained.
- ASUS→Milly Linux-only experiment: three fresh commands median **288.42 ms**,
  three reused commands **12.97 ms**, all six returned the expected hostname.
  The experimental setup ratio is 22.24×; it is not a bandwidth/fleet benchmark.
  Foreground master identity was checked, private socket directory mode 0700,
  and the exact owned master/socket/directory/lease were cleaned up. Opt-in bounded
  reuse needs failure/reconnect controls before durable automation deployment.
  Do not enable unsupported multiplexing in native Windows OpenSSH.
- An owned NUC loopback SSH forward through DTM-WORK completed actual RDP X.224
  negotiation, selecting CredSSP/HYBRID_EX. The exact owned process group and
  loopback listener were removed. This qualifies transport, not desktop login.
  No credential attempt, NLA/firewall change, or desktop transfer was performed.
- A direct NUC UniFi/WireGuard route remains unqualified. Working Tailscale/SSH
  jump access does not prove a configured site VPN. Resolve host-specific owner
  and credential evidence before authenticated desktop qualification; do not
  retry the previously rejected password.

## Profile, mount and client follow-up

The actual Documents path is OneDrive-backed. Existing synced shims were preserved;
keep machine-local implementations and secret backends outside OneDrive, and audit
each host's actual profile/module paths before deployment. Microsoft describes
network/OneDrive profile redirection as a source of module/profile loading issues:
[PowerShell profiles](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles?view=powershell-7).

A transient snapshot showed 398 PowerShell processes / 30.6 GB summed RSS; some
commands reported FileSystem InitializeDefaultDrives errors. Counts later fell
without broad termination. N: was the existing public ASUS NFS mount using soft
timeout 10/retry 1; the maintained mount helper uses timeout 1. Fabric recovered;
no unmount, root-cause attribution or host-wide performance gain is claimed.
Coordinate active share custody before any mount-policy migration.

Current IronRDP source already offers profile import, hidden credential prompting
and opt-in reconnect. Its default TLS backend skips certificate/signature
verification, and profile import ignores clipboard-disable and authentication-level
settings. No current deployed artifact was qualified. Prefer the tested strict SSH
transport and existing Microsoft client until genuine certificate policy and
profile-setting precedence have falsifiable tests and a validated build. Findings
and bounded hardening candidates were sent through the authoritative agent bus.

Private evidence stays under the local `~/.cache/pcai/` connectivity and Bitwarden
maintenance directories. It contains metadata and nonsecret fixtures; credentials
and session values are excluded. Active foreign worktrees, five unrelated files,
ASUS network/driver work and Oto hardware/HIL custody were preserved.
