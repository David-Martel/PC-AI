# Staged PowerShell syntax safety review

The [syntax helper](../../Tools/Test-PowerShellSyntax.ps1) parses literal staged
source without executing it. The [pre-commit entry](../../.pre-commit-config.yaml)
uses the filename argument boundary. The [maintained tests](PowerShellSyntaxHook.Tests.ps1)
read that configured entry, refuse command evaluation, and launch the actual
helper in bounded child PowerShell processes with inert TestDrive source.

The [review register](powershell-syntax-review.json) binds ten selector groups
(fourteen cases) to normalized selector, whole-test, helper and configuration hashes.
Independent review retains all ten groups at the registered selector and source hashes. Windows passes fourteen cases with zero failures or skips. Linux passes thirteen cases with zero failures and one explicit Windows Hidden-attribute prerequisite skip. The corrected helper also parses actual tracked Tests/.pester.ps1 on Linux. Final Git blobs at source commit c8d06853ebe3c576d10b5feea659e1a27b61d93a were independently executed on ASUS with PowerShell 7.6.9 and Pester 5.7.1. The dot-file and Windows-attribute controls each detect the missing Force option on their applicable platform. The register preserves prior receipts and binds this renewed review. No fleet installation, aggregate repository coverage or native/runtime acceptance follows.

The red controls use the same maintained tests: an always-success replacement
fails eight cases; the original command-evaluating hook configuration fails
before fixture execution; an execution replacement fails the marker-absence
assertion even though the valid source returns normally. The original malformed
YAML and original filename-execution defect were separately reproduced. No gate
threshold, skip or source assertion was weakened.

Run the portable selector suite with PowerShell 7 and Pester 5:

```powershell
Invoke-Pester -Path Tests/Unit/PowerShellSyntaxHook.Tests.ps1 -Tag Portable
```

The [Linux CI allowlist](../linux-ci-pester.txt) includes this suite; its execution
must be confirmed in isolation and with the existing cohort before final review.
The [CI guide](../../CI-CD-GUIDE.md#pre-commit-hooks) describes the hook boundary.
