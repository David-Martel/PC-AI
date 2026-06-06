# Proposal — Robust Git / GPG / GitHub Signing & Integration for Agents & Automation

**Status:** Draft for review · **Author:** claude-opus-pcai (Opus 4.8) · **Date:** 2026-06-06
**Scope:** DTM-P1GEN7 workstation + all David-Martel repos · **Grounding:** live audit (not aspirational)

---

## 0. Executive summary

The workstation already has the *right* signing primitive configured globally — **SSH commit signing**
(`gpg.format=ssh`, `~/.ssh/id_ed25519.pub`, `~/.config/git/allowed_signers`). It is defeated in agent/automation
sessions by three avoidable conflicts: (a) the Claude **agent-home `HOME`/`USERPROFILE` redirect** makes
`git --global` read a non-existent config and makes **GnuPG's keyboxd socket path exceed the OS limit**; (b) some
repos' **local `.git/config` overrides** force OpenPGP+GnuPG, re-introducing that exact failure; (c) **multiple gpg
binaries** create ambiguity.

**Recommendation:** standardize on **SSH signing**, make config resolution **survive the redirect** via
`GIT_CONFIG_GLOBAL` pinning, and add four thin operational layers — a **doctor/preflight**, **multi-agent
ownership**, a **CI/billing health monitor**, and an **upstream-review harvester** (Jules/Copilot/Dependabot →
actionable ledger). Each layer is independently shippable and testable.

**Proof point (validated this session):** with `HOME` redirected, a throwaway repo signed a commit via SSH and
verified `G` (Good signature, `davidmartel07@gmail.com`, ED25519) — **no wrapper, no keyboxd, no prompt**. The same
operation via OpenPGP/GnuPG fails with `keyboxd: socket name … is too long`.

---

## 1. Ground truth (audited 2026-06-06, read-only)

| Area | Finding | Implication |
|---|---|---|
| Global `~/.gitconfig` (real: `C:\Users\david\.gitconfig`) | `gpg.format=ssh`, `user.signingkey=…\.ssh\id_ed25519.pub`, `commit.gpgsign=true`, `gpg.ssh.allowedSignersFile=…\.config\git\allowed_signers` | **SSH signing is the intended standard** — and it works |
| Agent-session `HOME` | Redirected to `…\ClaudeCode\agent-homes\<id>\home` | `git --global` reads the **wrong/empty** `.gitconfig`; GnuPG `socketdir`/`homedir` derive from here |
| PC_AI `.git/config` (local) | `commit.gpgsign=true`, `gpg.program=…\GnuPG\gpg.exe`, **no `gpg.format`** → OpenPGP | Local override **defeats** global SSH signing → forces the brittle GPG path |
| GnuPG `gpgconf --list-dirs` | `socketdir`/`agent-socket` under the **long** redirected path | keyboxd `S.keyboxd` path > limit → `can't connect to keyboxd` → `Couldn't load public key` |
| gpg binaries | system `C:\Program Files\GnuPG\gpg.exe`, Git MSYS `…\Git\usr\bin\gpg.exe`, and `C:\Users\david\bin\gpg.cmd` (passthrough to system) | 3 candidates; ambiguity if `gpg.program` unset |
| SSH key + agent | `id_ed25519` (priv+pub) present; ssh-agent has the `davidmartel07@gmail.com` ED25519 loaded; `allowed_signers` maps it | SSH signing is **agent-backed, passphrase-free, redirect-proof** |
| `gh` CLI | Logged in `David-Martel` (keyring); scopes incl. `admin:public_key`, `admin:gpg_key`, `workflow`, `repo`; `git_protocol=ssh` | Can register signing keys, manage Actions, query alerts |
| OneDrive | `.gitconfig`/`.ssh`/`.gnupg`/`.config` **not** under `%OneDrive%`; but Documents **is** KFM-redirected | Credential dirs safe today; KFM is a latent footgun |
| Secrets tooling | `bw.exe` + `bws.exe` (Bitwarden) present; `.bw` dir absent | Token retrieval should route through Bitwarden, never plaintext |

---

## 2. Conflict taxonomy (what actually clashes)

| ID | Conflict | Today's symptom |
|---|---|---|
| **C1** | Agent-home `HOME`/`USERPROFILE` redirect vs `git --global` + GnuPG socketdir | global config invisible; keyboxd socket too long |
| **C2** | Per-repo `.git/config` override forces OpenPGP over global SSH | `git commit -S` fails; needed an env-reset wrapper |
| **C3** | System GnuPG vs Git MSYS gpg vs `~/bin/gpg.cmd` | wrong binary if `gpg.program` unset/relative |
| **C4** | GnuPG keyboxd Unix-socket path-length limit on Windows + long homedir | `error opening key DB: No Keybox daemon running` |
| **C5** | Multiple agents in one repo (Claude + Codex + Jules) | `index.lock` races, clobbered edits, duplicate commits |
| **C6** | GitHub Actions **billing/quota** blocks runs silently | workflows don't start; no failure surfaced; "CI green" illusion |
| **C7** | Upstream reviewers (Jules/Copilot/Dependabot) findings not actionable | alerts pile up, never become tracked work |
| **C8** | OneDrive KFM of config/credential dirs | sync conflicts / corruption of `.ssh`/`.gitconfig` (latent) |
| **C9** | Secret hygiene in agent flows | 12.9 MB `powershell-profile-files.json` nearly committed this session |

---

## 3. Design — six thin layers (each shippable + testable)

### Layer 0 — Signing standard: **SSH** (with GPG retained only as explicit opt-in)
- **Why SSH over OpenPGP here:** no agent socket (uses ssh-agent named pipe / key file), key is an absolute path
  unaffected by the redirect, GitHub verifies it via an uploaded *SSH signing key*, and it is **proven to work
  under the redirect**. OpenPGP keeps tripping C1/C4.
- **Action:** reconcile each repo to inherit global SSH signing. For PC_AI specifically, replace the local override:
  ```
  git config --local --unset gpg.program           # stop forcing the GnuPG binary
  git config --local gpg.format ssh
  git config --local user.signingkey C:/Users/david/.ssh/id_ed25519.pub
  git config --local gpg.ssh.allowedSignersFile C:/Users/david/.config/git/allowed_signers
  ```
  (Or simply unset the local `gpg.program`/`commit.gpgsign` so the global SSH config applies.)
- **GitHub side:** ensure the SSH **signing** key is uploaded as type *signing* (not just *auth*):
  `gh ssh-key list` → if absent, `gh api user/ssh_signing_keys -f title=… -f key=@~/.ssh/id_ed25519.pub`. This is
  what flips commits to **Verified** on github.com.

### Layer 1 — Config resolution that survives the redirect
- A session bootstrap exports the real global config so the redirect can't hide it:
  ```powershell
  $env:GIT_CONFIG_GLOBAL = 'C:\Users\david\.gitconfig'   # real global, not redirected HOME
  # SSH signing reads keys/agent directly; GnuPG is no longer on the path for signing.
  ```
- Optionally `$env:GNUPGHOME='C:\Users\david\AppData\Roaming\gnupg'` **only** for repos still on GPG (keeps the
  legacy path working) — but the goal is to retire it.
- Packaged as `Initialize-GitSigningEnvironment` in a new **`PCAI.GitOps`** PowerShell module. Idempotent,
  per-process, **no global mutation** (safe in the sandbox).

### Layer 2 — Doctor / preflight (`Test-GitSigning`)
A single read-only command every agent runs before committing (wire into commit-cluster + context-save pre-flight):
1. Resolve effective `gpg.format`, `user.signingkey`, `commit.gpgsign` (warn if OpenPGP + long HOME).
2. ssh-agent has the signing key; `allowed_signers` exists and contains the committer email.
3. **Throwaway signed-commit smoke test** in `%TEMP%` → expect `%G?`=`G` (the test run this session).
4. `gh auth status` valid + required scopes; SSH signing key uploaded to GitHub.
5. No credential dir under `%OneDrive%`. Correct single `gpg.program` if GPG is in use.
Returns structured PASS/FAIL/WARN. (Replaces ad-hoc wrapper hacks; the env-reset GnuPG wrapper becomes a documented
*fallback*, not the default.)

### Layer 3 — Multi-agent concurrency & ownership (agent-bus)
- **Presence + claim before edit:** `set_presence` on entry; `claim_resource`/`post topic=ownership` before
  editing a file; `check_inbox` every few actions (protocol already exists — formalize it as required).
- **Commit serialization:** a repo-scoped advisory lock (`agent-bus claim_resource repo:PC_AI:git-index`) around
  `git add/commit/push` so two agents never race the index. On `index.lock` present: backoff + retry, **never**
  blind-delete (it may be a live commit).
- **Handoff:** post a COMPLETE summary of pushed SHAs (done this session) so peers rebase, not clobber.

### Layer 4 — CI / workflow **billing & health** monitor (`Watch-WorkflowHealth`)
Silent failure (C6) is the dangerous one — billing/quota blocks make runs **never start**, so `conclusion` is
`null`/absent rather than `failure`.
- Poll `gh run list --json status,conclusion,…` AND `gh api /repos/{o}/{r}/actions/runs` for `startup_failure`,
  and `gh api /repos/{o}/{r}/actions/permissions` / billing (`gh api /user/settings/billing/actions`) for
  minutes exhaustion.
- Detect **disabled** workflows and `payment`-class errors; surface to an actionable ledger + agent-bus
  (topic=status) instead of assuming green. (Directly answers the "CI green illusion" friction.)
- Schedule via the existing `/schedule` or a Windows task; output to `Reports/ci-health/`.

### Layer 5 — Upstream-review **harvester** (`Get-UpstreamReviews`)
Turn bot output into tracked work (C7):
- **Dependabot:** `gh api /repos/{o}/{r}/dependabot/alerts --paginate` (the 3 alerts noted on PC_AI pushes).
- **Code scanning:** `gh api /repos/{o}/{r}/code-scanning/alerts`.
- **Copilot/Jules/review bots:** `gh api /repos/{o}/{r}/pulls/{n}/reviews` + `…/comments`, filter bot authors;
  Jules check-runs via `gh api /repos/{o}/{r}/commits/{sha}/check-runs`.
- Dedup + severity-rank → append to a ledger (`Reports/upstream-reviews/` or `*.TODO.md`) so nothing is lost.

### Layer 6 — Secret hygiene & OneDrive guard
- **Pre-commit secret scan** in `lefthook` (gitleaks/trufflehog, or a lightweight regex pass) — would have caught
  the 12.9 MB profile dump (C9). Block on hit.
- **Tokens via Bitwarden only** (`bw`/`bws`); never plaintext; reuse the `bitwarden-gpg-signing` skill pattern for
  unlock. PATs refreshed centrally (note the open DTM token-scope task in the workspace plan).
- **OneDrive guard:** assert `.ssh`/`.gnupg`/`.gitconfig`/`.config\git` are NOT under `%OneDrive%` (C8); a
  `Test-GitSigning` check fails if KFM ever captures them.

---

## 4. Conflict → resolution map

| Conflict | Resolved by | Residual risk |
|---|---|---|
| C1 redirect vs global/socket | L1 `GIT_CONFIG_GLOBAL` pin + L0 SSH (no socket) | none once SSH-standard |
| C2 per-repo override | L0 reconcile repo config | other repos still on GPG until migrated |
| C3 gpg binary ambiguity | L0 (drop gpg.program for SSH) / L2 asserts single binary | legacy GPG repos |
| C4 keyboxd socket length | L0 eliminates GnuPG from signing path | GPG fallback only |
| C5 multi-agent races | L3 ownership + index lock | requires all agents honor protocol |
| C6 silent CI/billing | L4 monitor | needs scheduled execution |
| C7 upstream findings | L5 harvester | needs scheduled execution |
| C8 OneDrive KFM | L6 guard check | user-controlled KFM settings |
| C9 secrets | L6 scan + Bitwarden | scanner coverage |

---

## 5. Phased rollout (each phase has a validation gate)

| Phase | Deliverable | Validation gate |
|---|---|---|
| **0 — Doctor** | `Test-GitSigning` (read-only) | Smoke-test returns `G` on this machine; flags PC_AI's GPG override |
| **1 — SSH standard** | Reconcile PC_AI `.git/config`; `Initialize-GitSigningEnvironment`; upload SSH signing key to GitHub | A real PC_AI commit signs via SSH (no wrapper) and shows **Verified** on github.com |
| **2 — Concurrency** | agent-bus ownership + index-lock helpers in `PCAI.GitOps` | Two concurrent agents commit without `index.lock` corruption (test harness) |
| **3 — CI + upstream** | `Watch-WorkflowHealth` + `Get-UpstreamReviews` (scheduled) | Detects a seeded billing/disabled-workflow state; harvests the 3 Dependabot alerts into a ledger |
| **4 — Hygiene** | lefthook secret scan + OneDrive guard | Scanner blocks a planted secret; guard fails if a cred dir is KFM'd |

---

## 6. Validation & testing for robustness (`Tests/GitOps/GitOps.Tests.ps1`)

| Test | Asserts |
|---|---|
| `signs-under-home-redirect` | throwaway SSH-signed commit verifies `G` with redirected `HOME` |
| `gpg-fallback-wrapper` | env-reset GnuPG wrapper still signs (documented fallback) |
| `gh-auth-and-scopes` | `gh auth status` valid; `admin:public_key` present; SSH signing key uploaded |
| `billing-block-detected` | mocked `startup_failure`/billing response → `Watch-WorkflowHealth` flags it |
| `concurrent-commit-safe` | two processes serialize via the index lock; both commits land |
| `secret-scan-blocks` | planted token in a staged file → lefthook blocks commit |
| `onedrive-guard` | cred dir under `%OneDrive%` → `Test-GitSigning` FAIL |
| `verified-on-github` | pushed commit's `verification.verified == true` via `gh api …/commits/{sha}` |

---

## 7. Immediate quick wins (low risk, do first on approval)
1. **Reconcile PC_AI `.git/config`** to SSH signing (Layer 0) — stops every agent needing the GnuPG wrapper.
2. **Upload the SSH signing key to GitHub** (`gh api user/ssh_signing_keys`) if not already type *signing* — flips
   future commits to Verified.
3. **Ship `Test-GitSigning`** and wire it into commit-cluster + context-save pre-flight.
4. **Bootstrap `GIT_CONFIG_GLOBAL`** in the agent session entry so global SSH config is always visible.

---

## 8. Open questions for review
- **SSH-only, or dual** (SSH default + GPG retained for repos/keys that require OpenPGP, e.g. tag signing on
  specific upstreams)? Recommendation: SSH default, GPG explicit per-repo opt-in.
- Flip **all** David-Martel repos now, or PC_AI first as the pilot?
- Should L4/L5 run as a **scheduled remote agent** (`/schedule`) or a local Windows Task? (Scheduled agent gives
  the actionable-ledger-back-to-chat loop the user asked for.)
- Who owns the shared `PCAI.GitOps` module across repos (single source vs per-repo copy)?

---

### Appendix A — the GnuPG fallback (when a repo *must* use OpenPGP)
Documented in project memory `gpg-keyboxd-socket-wrapper-fix`: an env-reset `gpg.program` wrapper that pins
`USERPROFILE/HOME/GNUPGHOME/LOCALAPPDATA/APPDATA` to real paths before invoking `gpg.exe`, passed via
`git -c gpg.program=<wrapper>`, with the keyid attached to `-S` (since `user.signingkey` is unset for OpenPGP).
This stays as a **fallback only**; SSH signing is the standard.
