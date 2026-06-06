# PC_AI Session Context — 2026-06-06 — GitOps, Signing, Input-Stack Fixes, System Hardening

**Project:** PC_AI · **Branch:** main (PR-only ruleset now enforced) · **Machine:** DTM-P1GEN7 (ThinkPad P1 Gen 7)
**Co-worker:** Codex (active in this repo; coordinated via agent-bus). This was a very long multi-phase session.

## Major arcs completed
1. **commit-cluster** — 9 GPG/SSH-signed commits (workstation evidence + tooling); discovered the agent-home
   `HOME` redirect breaks GnuPG keyboxd (socket too long). Memory: `gpg-keyboxd-socket-wrapper-fix`.
2. **Git/GPG/GitHub robustness** — proposal at `.claude/plans/git-gpg-github-agent-robustness-proposal.md`.
   Standardized on **SSH commit signing** (works under the redirect; proven `verified:true`). PC-AI `.git/config`
   reconciled to SSH signing.
3. **GitOps fleet** (`Tools/GitOps/`) — `Set-RepoRuleset.ps1` applied `dtm-default-protection` to **all 194 eligible
   David-Martel repos**: required signatures + PR-only (0-approval self-merge) + block force-push/deletion +
   conversation resolution; github-actions[bot] (User 41898282) bypass; no human bypass. `Test-GitSigning.ps1`
   doctor. Non-blocking push monitors `Watch-WorkflowHealth.ps1` + `Get-UpstreamReviews.ps1` via lefthook pre-push
   (Win32_Process.Create detach, ~0.8s). Dogfooded: PRs auto-reviewed by gemini/copilot bots → resolve → self-merge.
4. **GitHub account-picker fixed** — routed github https→SSH in real `~/.gitconfig` (`url."git@github.com:".insteadOf`)
   so gh's multi-account credential helper stops prompting.
5. **PowerShell `.bak` shim fixed** — created missing `~/.config/powershell/Microsoft.PowerShell_profile.ps1` (the
   shim's preferred local profile) so it stops dot-sourcing the OneDrive `.bak`; deleted the `.bak`. (Openers were
   NETGEAR A9000 pollers + terminals, NOT codex.)

## Input-stack root causes + fixes APPLIED (reversible; rollbacks in input-stack-investigation-20260606/)
- Keyboard (PS/2 `LEN0071`) and touchpad (**Sensel `SNSL002D`** HID-over-I2C, parent I2C `7E78`) are on DIFFERENT
  buses → two independent causes. Haptic theory REFUTED as primary; Process Lasso/FilterKeys/shared-bus REFUTED.
- **T1** — device power-down OFF on `SNSL002D` + `7E78` (`MSPower_DeviceEnable=False`). Applied.
- **T2** — `WdfDirectedPowerTransitionEnable 1→0` on I2C `7E78` (closes the PEP/PoFx directed-power path T1 missed).
  Applied — **REBOOT to activate**.
- **F3** — MouseKeys `HOTKEYACTIVE` disarmed (Flags `62→58`); prior remediation missed MouseKeys.
- Eval harness: `Tools/InputDiagnostics/Watch-InputGlitch.ps1` (glitches/day before/after). `FINDINGS.md` is canonical.

## System assessment (Reports/system-assessment-20260606/ — 42 files; Reports/input-stack-…/hid-deep-* — 13 files)
- **NVIDIA RTX 2000 Ada (DEV_28B8) Code 31** = driver split vs eGPU RTX 5060 Ti (DEV_2D04). Caused the 62 DWM
  crashes (05-23→26, historical). FIX is a **two-package manual procedure** (Enterprise 610.47 for Ada + per-device
  Have-Disk bind of GRD 610.47 for the 5060 Ti, WU paused) — **NO single package covers both**; must be USER-DRIVEN
  (display-reset risk). Plan: `Reports/system-assessment-20260606/nvidia-driver-plan.md`.
- **RTX 5060 Ti SM_120 gap** → 9 python.exe crashes via nvcuda64.dll → fixed by torch≥2.7+cu128 (applied below).
- Sensel haptic firmware (F8): Lenovo **ds571229** fixes P1 Gen 7 haptic-touchpad freeze — via Vantage.
- Other: WUDFRd 0xC0000365 ×8 boot, Dell TechHub bloatware, login storm, System log 99% benign HttpEvent noise.

## Applied this turn
- **Dependency modernization** (AI-Media): created `pyproject.toml` (requires-python>=3.13, torch>=2.7.0 + cu128 uv
  index, drop abandoned basicsr/realesrgan), `.python-version=3.13`, updated `requirements.txt`. (Manifests only —
  run `uv sync` to install; NEEDS-TEST for the Janus pipeline.) Plan: `…/dependency-modernization-plan.md`.
- **vLLM** `PC_AI-VLLM` + `PC_AI-ToolRouter` → **Manual** (no boot start; explicit `Start-Service`).
- **OneDrive**: confirmed `C:\codedev` is NOT linked (no junction/reparse, not under OneDrive root) — no action needed.
- System restore point created.

## OPEN / next (user-gated or needs test)
- [ ] **NVIDIA driver** two-package install (user-driven, see plan) — clears Code 31 + cursor/DWM risk.
- [ ] **Lenovo Vantage** System Update: touchpad firmware ds571229, BIOS/EC, IR camera (GUI — user runs).
- [ ] **REBOOT** to activate T2; then measure touchpad with `Watch-InputGlitch.ps1`.
- [ ] **Test** the AI-Media dep changes: `cd AI-Media && uv sync` then verify `torch.cuda.get_device_capability()`.
- [ ] **PC-AI CI is RED** (9 failures: CI + Portable CI) + 3 open Dependabot rust-openssl alerts (1 high) — triage.
- [ ] Optional optimizations: power plan (PL reverts Balanced — open decision), pagefile (T: 32-64GB, likely
      intentional), Dell TechHub uninstall, BitLocker on W:/E:, 136 ghost HID cleanup.
- [ ] F2: test Logi Options+ keyboard hook as a Shift-drop vector (quit logioptionsplus_agent during a failure).
