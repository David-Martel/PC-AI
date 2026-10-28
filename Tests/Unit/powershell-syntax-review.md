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

## Intrinsic-command autoload boundary follow-up

Integrating accepted main into the draft exposed a separate boundary: the canonical runner prepends repository Modules to PSModulePath. A fresh syntax child could then autoload the aggregate while trying to report a parser error through unqualified Write-Error. Five existing syntax controls failed because an unrelated Evaluation producer was imported instead of reporting the original syntax/path diagnostic.

The helper now qualifies its intrinsic file-discovery and output/error commands. Four new controls create an owned module advertising Get-Item, Write-Output and Write-Error; module import writes a marker and throws. All four controls fail against the predecessor helper because the marker appears. With the fix, the same four controls and the existing fourteen syntax cases pass, including original error diagnostics. The combined syntax/empty-manifest run passes 23 tests with no failures or skips on PowerShell7.6.7/Pester5.7.1.

The selector register retains previous reviews as historical bindings. Independent root review accepts all eleven current groups at commit 039d7e8a08160ee81b77ac4ef3618f6568b26b8c and independently reran all eighteen Windows syntax cases with no failures or skips. The four new cases have no current Linux execution receipt. These controls neither change the WindowsCI cohort nor add full-run coverage credit. The original stage-source execution/hidden-file refusal assertions remain intact. No module search paths or installed modules are changed outside the owned children.
