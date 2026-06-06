# PCAI.GitOps — Robust Git / GPG / GitHub signing & integration for agents

Implements the proposal in [`.claude/plans/git-gpg-github-agent-robustness-proposal.md`](../../.claude/plans/git-gpg-github-agent-robustness-proposal.md).
Standard: **SSH commit signing** (no keyboxd; survives the agent-home HOME redirect — proven) with
GitHub-enforced **required signed commits** and a **PR-only** merge flow.

## Decisions in force (2026-06-06)
- **Signing:** SSH default (`gpg.format=ssh`, `~/.ssh/id_ed25519.pub`, `~/.config/git/allowed_signers`).
  GPG retained only as a documented fallback (env-reset `gpg.program` wrapper; memory `gpg-keyboxd-socket-wrapper-fix`).
- **Branch protection (all 194 eligible David-Martel repos):** ruleset `dtm-default-protection` on the
  default branch — required signatures + PR-only (0 required approvals → owner self-merges after checks) +
  block force-push/deletion + require conversation resolution. **Bypass: GitHub Actions app (15368)** so
  CI auto-commit workflows (changelog/format) aren't rejected; **no human bypass**.

## Scripts
| Script | Purpose | Mutating? |
|---|---|---|
| `Set-RepoRuleset.ps1` | Idempotently apply the ruleset to `-Repos <names>` or `-All` eligible repos; per-repo JSONL log under `Reports/gitops/`; `-DryRun` to preview | GitHub (rulesets) |
| `Test-GitSigning.ps1` | Read-only doctor/preflight: effective signing config, ssh-agent, allowed_signers, throwaway SSH-signed-commit smoke test (`%G?`==G), gh auth, OneDrive-KFM guard. Non-zero exit on FAIL. Wire into commit-cluster / context-save pre-flight | no |

## Quick start
```powershell
pwsh Tools/GitOps/Test-GitSigning.ps1                 # verify you can sign before committing
pwsh Tools/GitOps/Set-RepoRuleset.ps1 -Repos PC-AI -DryRun   # preview
pwsh Tools/GitOps/Set-RepoRuleset.ps1 -All            # flip every eligible repo
```

## Signing under the agent-home redirect
The Claude agent sandbox redirects `HOME`/`USERPROFILE` to a long path, which (a) hides the real global
`~/.gitconfig` and (b) makes GnuPG's keyboxd socket path exceed the OS limit. SSH signing sidesteps both:
```powershell
git -c commit.gpgsign=true -c gpg.format=ssh `
    -c user.signingkey=C:/Users/david/.ssh/id_ed25519.pub `
    -c gpg.ssh.allowedSignersFile=C:/Users/david/.config/git/allowed_signers `
    commit -m "..."
```
(For a permanent fix, reconcile the repo's local `.git/config` to SSH signing — see below.)

## Monitors + non-blocking hooks (directive 3 — shipped)
| Script | Purpose |
|---|---|
| `Watch-WorkflowHealth.ps1` | Detect silent CI failures: `startup_failure` (billing/permissions), failures, disabled workflows, Actions-minutes exhaustion → `Reports/gitops/workflow-health-*.json` |
| `Get-UpstreamReviews.ps1` | Harvest Dependabot + code-scanning alerts + Copilot/Gemini/Jules PR reviews → `Reports/gitops/upstream-reviews-*.json` |
| `Invoke-GitOpsMonitors.ps1` | Runs both, writes a combined snapshot, best-effort posts to the agent-bus (`localhost:8400`). Launched **detached** from the hook so it never blocks. |
| `Install-GitOpsHooks.ps1` | Installs a non-blocking `pre-push` hook for non-lefthook repos; prints the lefthook snippet otherwise. |

Wired into PC_AI via `lefthook.yml` `pre-push` (fires `Invoke-GitOpsMonitors.ps1` detached). Git has no
`post-push` hook, so `pre-push` + a detached worker (with a short delay) captures post-push CI state without
blocking. Outputs land in `Reports/gitops/` (gitignored) + the agent-bus for live agents.

## Roadmap (remaining)
- [ ] **Reconcile-GitSigning.ps1** — flip an arbitrary local clone's `.git/config` to SSH signing (PC-AI done inline).
- [ ] **Tests/GitOps/GitOps.Tests.ps1** — sign-under-redirect, billing-detect, secret-scan, onedrive-guard.
- [ ] Reconcile legacy *classic* branch protection on any repo that still has it (PC-AI done; it conflicted with the ruleset's 0-approval self-merge).

## Coordination
Multiple agents (Claude, Codex, Jules) work these repos. Claim files via the agent-bus `topic=ownership`
before editing shared config; post pushed SHAs so peers rebase, not clobber.
