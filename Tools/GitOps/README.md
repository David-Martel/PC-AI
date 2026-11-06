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

## Repository discovery and report limits

The [detached worker](Invoke-GitOpsMonitors.ps1) discovers the repository only when
`-Repo` is absent. Its GitHub response helper accepts an Object response only when
the parsed root has the concrete `System.Management.Automation.PSCustomObject`
runtime type. Arrays and scalars remain unavailable for Object requests; Array
requests retain their existing behavior.

A discovered `nameWithOwner` must be a string containing only the existing ASCII
repository-name characters, with exactly one owner/name separator. The discovery
guard uses strict `\A` and `\z` anchors to reject a trailing newline. It assigns
the repository only after validation; explicit `-Repo` handling remains unchanged.

The combined report retains `detail.repository` with exactly four fields:
`operation`, `status`, `exit_code`, and controlled `reason`. This repository projection excludes raw process
output, error text, arguments, paths, environment values and the parsed response
object. Successful discovery retains this projection alongside workflow and
upstream details. A nonzero process exit remains unavailable even if its stdout
looks like valid identity JSON. Launch/deadline failures retain exit `-1` and their
generic reason; that result does not identify which cause occurred.

Failed discovery produces overall `Unavailable`, process exit `2`, and
`monitors-latest-unavailable.json`, with no workflow/upstream inspection or bus
publication. This report is evidence of the probe outcome, not a diagnosis of
credentials, working directory, encoding or environment.

### Workflow and billing report contract

The [workflow producer](Watch-WorkflowHealth.ps1) reports
`workflow_inspection_status` from both run and workflow-list inspection after
metadata validation. Billing failure leaves that workflow status independent.
Aggregate inspection status, existing findings/exhaustion comparison, exits and
publication decisions retain their existing behavior.

`billing.status` reflects inspection availability. Successful legacy responses
retain numeric `used` and `included` values after finite-scalar validation.
Strings, Booleans, containers, null and nonfinite values are unavailable.
Unavailable billing exposes `quota_status=Unknown` with exact-null metrics;
it does not infer zero usage or included entitlement. The endpoint, authentication
and request deadlines are unchanged.

### Focused controls and remaining qualification

The [unit fixture](../../Tests/Unit/GitOpsMonitors.Tests.ps1) executes the actual
worker, workflow producer and shared helper with external process/CIM boundaries
stubbed. Run this complete fixture file from the repository root in a fresh shell:

~~~powershell
pwsh -NoProfile -Command "Invoke-Pester -Path Tests/Unit/GitOpsMonitors.Tests.ps1 -Output Detailed"
~~~

The [schema review](../../Tests/Unit/gitops-workflow-schema-review.json) binds
four groups/19 cases to the current producer and whole-test hashes. The
[discovery reaffirmation](../../Tests/Unit/gitops-discovery-review.json) retains
all seven unchanged selector bodies under that new whole-test context and the
updated workflow dependency where exercised. Independent review verified complete
byte extents; the shared checker still has no Pester parser or canonical AST PASS.

Schema RED produced 19 failures, then focused GREEN produced 19 passes. The final
bounded file replay passed all 66 unique cases: original 18, discovery 29 and schema
19, with zero failures, skips or not-run cases. Its owned job elapsed 104.94 seconds,
did not time out, left no owned processes, recorded zero external guard hits and
preserved all 14 bound hashes. Receipt SHA-256:
`92c06e1e7249dfcab16fc4ffdaa5b5bce53ca2be07bf75f264e31817fb2e9829`.
This qualifies the stubbed GitOps fixture file, not the complete project suite,
live GitHub/bus/billing availability, application runtime or deployment.

The [signed first-PR record](https://github.com/David-Martel/PC-AI/blob/2800c09957e9073a079e2321f00149f0ea2d0595/Tests/Unit/gitops-discovery-review.json)
preserves the earlier `dc5da21962ac15b63d39b738eab41ffd15238d90` implementation
binding to the `301943` test revision and its historical 29-pass discovery and
14-pass/four-failure legacy result. That history and the earlier 13-pass/five-failure
baseline remain distinct from this current 66-case candidate. Current review
commit fields remain null until the actual schema implementation is committed.

The [workflow producer](Watch-WorkflowHealth.ps1) still calls the product-specific
`/user/settings/billing/actions` endpoint. [GitHub retired that endpoint family on
September 26, 2025](https://github.blog/changelog/2025-09-26-product-specific-billing-apis-are-closing-down/).
This report-schema correction does not qualify live billing availability or included quota. API migration requires separate validation; missing quota information remains unknown.

## Roadmap (remaining)
- [ ] **Reconcile-GitSigning.ps1** — flip an arbitrary local clone's `.git/config` to SSH signing (PC-AI done inline).
- [ ] **Tests/GitOps/GitOps.Tests.ps1** — sign-under-redirect, billing-detect, secret-scan, onedrive-guard.
- [ ] Reconcile legacy *classic* branch protection on any repo that still has it (PC-AI done; it conflicted with the ruleset's 0-approval self-merge).

## Coordination
Multiple agents (Claude, Codex, Jules) work these repos. Claim files via the agent-bus `topic=ownership`
before editing shared config; post pushed SHAs so peers rebase, not clobber.
