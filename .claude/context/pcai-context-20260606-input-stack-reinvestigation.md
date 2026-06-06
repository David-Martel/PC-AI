# PC_AI Session Context — 2026-06-06 — Input-Stack Re-investigation + Doc Reconciliation

**Project:** PC_AI · **Branch:** main · **Machine:** DTM-P1GEN7 (ThinkPad P1 Gen 7)
**Skills:** commit-cluster → context-save-restore → systematic-debugging
**Agents:** Explore (reports triage), general-purpose ×2 (live evidence — died on Overloaded but wrote 14 evidence files; doc/tooling eval)

## What happened this session

### 1. commit-cluster (earlier) — 9 signed commits pushed (f20b32e..e1cae05)
Workstation evidence + tooling. Key infra fix: **GPG signing was broken** by the Claude agent-home
`USERPROFILE` redirect (keyboxd socket path too long). Fix = env-resetting `gpg.program` wrapper
(`%TEMP%\git-gpg-fix.cmd`) passed via `git -c gpg.program=`. Saved to project memory
(`gpg-keyboxd-socket-wrapper-fix`). `user.signingkey` is unset → must attach keyid to `-S498163FF6E59F96A`.

### 2. Input-stack root-cause investigation (shift key + touchpad lockup)
Evidence: `Reports/input-stack-investigation-20260606/` (`FINDINGS.md` + `_SUMMARY.txt` + H1–H7 raw probes).
- **Load-bearing fact:** Shift = PS/2 `LEN0071`/i8042 (parent PCI `7E02`); Touchpad = **Sensel `SNSL002D`**
  HID-over-I2C (parent `7E78`). **Different buses → two independent root causes** (prior ledger wrongly
  called the touchpad "Synaptics" — that's the fingerprint; ELAN is the TrackPoint. Corrected in ledger.)
- **Verdicts:** H1 Filter/Sticky Keys REFUTED (off + hotkey disarmed now, Flags=2). H3 shared-bus REFUTED.
  H4 Process Lasso REFUTED (it *protects* input). H6 power **SUPPORTS touchpad** — `MSPower_DeviceEnable=True`
  (confirmed live) on BOTH the Sensel touchpad AND its parent I2C controller `7E78` × Modern Standby (506/507)
  = I2C-HID resume-lockup mechanism. Shift has NO OS-layer evidence (i8042 clean 14d) → needs `Test-KeyInput`
  trace to decide software-vs-EC. Anomaly: dwm.exe crashed 62× on 05-26 (dwmcore.dll) → cross-links to open
  NVIDIA Code 31.
- **New actionable fix (T1):** disable device power-down on `SNSL002D` + `7E78` (reversible), then measure.
- **New eval tool:** `Tools/InputDiagnostics/Watch-InputGlitch.ps1` — Gate-C glitch capture + glitches/day
  before/after a fix (the measurement prior fixes lacked). Validated read-only.

### 3. Doc reconciliation + tooling eval
- CLAUDE.md fixed: "89 scripts"→"~81 top-level (245+ recursive)"; added `portable-ci.yml` + `jules-review.yml`
  to the workflows table (11 actual); Litho "not yet generated / wrong binary name" status note.
- `Reports/doc-tooling-evaluation-20260606.md`: validate-doc-accuracy works (caught TOOLS.md drift: schema 31
  tools vs doc 28 — missing 3 media tools); all `Reports/` doc outputs 2–4 months stale; Litho/deepwiki C4 docs
  NEVER generated (`docs/auto/` absent; pipeline looks for `litho` not `deepwiki-rs.exe`).

### 4. Cleanup
Removed: session gpg wrapper, `Tools/__pycache__`, 44 empty Reports dirs. Gitignored: eval runtime output +
incidental `direct-pass-*` dumps (12.9 MB profile JSON — possible secrets). NOT deleted: ~934 MB OneDrive `.db`
forensic copies (evidence for the still-OPEN OneDrive WNS issue) — surfaced for user decision.

## Open / next (see machine-reliability.TODO.md + boot.TODO.md)
- [ ] Run `Test-KeyInput.ps1` during a Shift failure (decides keyboard branch).
- [ ] Apply touchpad fix T1 (power-down off on `SNSL002D`+`7E78`); measure with `Watch-InputGlitch.ps1`.
- [ ] NVIDIA RTX 2000 Ada **Code 31** (CM_PROB_FAILED_ADD) — restore point + Vantage.
- [ ] OneDrive WNS channel (GUI relink) + ~15 h OneDrive CPU churn.
- [ ] Re-run doc pipeline (stale); fix Litho binary-name/path; make validator CI-gateable.
- [ ] Decide: delete ~934 MB OneDrive `.db` evidence copies?
