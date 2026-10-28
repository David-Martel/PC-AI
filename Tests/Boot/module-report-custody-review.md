# Captured module reports during publication

The [installer](../../Tools/Install-PcaiDevModules.ps1) preserves installed
`Config/Reports` as captured evidence. Only the exact first relative directory
segment `Reports` receives byte copying. Settings outside that subtree still
require object-root JSON and merge installed values over source defaults.
`Reports2` and `nested/Reports` do not receive captured-evidence treatment.
All existing path and reparse checks run before copying captured files.

The repository has six tracked benchmark snapshots under `Config/Reports`,
introduced by commit `c25ca2d`; their availability files contain JSON arrays.
The installed umbrella on the primary host already has these six snapshots.
The original installer therefore rejected an otherwise valid repeat update
before publication. This correction preserves the reports in source, generated
packages, installed modules and recoverable backups. It does not change the
generator or relax settings validation elsewhere.

The five new groups in the [maintained fixture](DevModuleDeployment.Tests.ps1)
cover eight cases: repeat publication with report and backup identity; late-fault
rollback; four exact-subtree/settings boundary cases; write-free DryRun; and a
captured-report junction refusal. Each group states its rationale and prerequisites.
Run on Windows with PowerShell 7 and Pester 5:

```powershell
Invoke-Pester -Path Tests/Boot/DevModuleDeployment.Tests.ps1
```

The same eight new cases against the old installer produce six passes and two
intended failures: report-bearing publication and the late publication fault
cannot get past array-root report parsing. The corrected installer passes all
46 cases, including the existing 38. No case is skipped or unrun in that complete
run. The [review register](module-report-custody-review.json) binds the five new
groups to LF-normalized source, whole-test and selector hashes. Existing path
review receipts in `module-install-path-review.json` retain their historical
bindings; they are not silently relabeled as reviews of changed source.
An independent Windows fleet reviewer retains all five groups at those exact
hashes and independently executed the complete 46-case fixture with zero
failures, skips or unrun cases.

External acceptance also uses the actual generated umbrella at source
`17fa6efbe1f2ffa1eba2b6dbc593af095c6b6f88`, with the candidate installer. A
synthetic receiver preserves installed settings and installed-only files,
retains an exact old backup, supports raw-identity recovery to old and forward
again, and successfully republishes while preserving all captured report bytes.
A deterministic later target collision then exercises automatic rollback and
retains failed-stage evidence. Process, user and machine module paths stay
unchanged. A fresh profile-free child validates six required exports, real
evaluation classes/dataset/results and CLI discovery from the staged umbrella.
The generated package contains no DLL, shared library or executable payload.

Raw receipts remain under the operator evidence root
`r2/module-staging/`: `reports-baseline.clixml`,
`reports-candidate-final.clixml`, `adapter-qualification.json`,
`qualified-fresh-consumer.json`, and `pinned-stage-manifest.json`.
These are external synthetic-receiver checks. Live fleet installation remains
separate and must regenerate from the accepted source pin, preserve each host's
actual configuration and backup, and validate a fresh consumer afterward.
