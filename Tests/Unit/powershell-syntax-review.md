# Staged PowerShell syntax safety review

The [syntax helper](../../Tools/Test-PowerShellSyntax.ps1) parses literal staged
source without executing it. The [pre-commit entry](../../.pre-commit-config.yaml)
uses the filename argument boundary. The [maintained tests](PowerShellSyntaxHook.Tests.ps1)
read that configured entry, refuse command evaluation, and launch the actual
helper in bounded child PowerShell processes with inert TestDrive source.

The [review register](powershell-syntax-review.json) binds eight selector groups
(twelve cases) to normalized selector, whole-test, helper and configuration hashes.
Independent review retains all eight groups at the registered selector and source hashes. Windows and Linux controls pass twelve cases, zero failed or skipped. Final committed Git blobs were independently executed on ASUS Linux x86_64 with PowerShell 7.6.9 and Pester 5.7.1; the register records the source commit and receipt hash. These controls do not establish fleet installation, aggregate repository coverage or native/runtime acceptance.

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
