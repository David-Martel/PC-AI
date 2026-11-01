# Receipt ancestor custody review

Install: use Windows PowerShell 7 and Pester 5; no deployment is required.
Build: the fixture compiles the exact [native source](../../Tools/SystemScripts/Storage/PcaiArchiveSource.cs) in its fresh test process.
Test/run: `pwsh -NoLogo -NoProfile -Command "Invoke-Pester ./Tests/Boot/ArchiveSourceCustody.Tests.ps1"` from the repository root.

The [hash-bound register](archive-source-custody-review.json) records independent review of seven receipt lifecycle selectors in the [unchanged fixture](ArchiveSourceCustody.Tests.ps1). It protects durable append and source deletion against held-parent replacement, leaf writers, symlinks, junctions, hardlinks and failed encoding. Tests inspect real native handles and owned content; remote transport remains an inert fixture.

The first ancestor acquired from the physical parent chain is the volume root. The reviewed source permits write sharing on that handle while still denying delete sharing and checking reparse attributes and GUID identity. Every descendant retains write/delete exclusion. This fixes the observed Win32 sharing violation 32 on `C:\` without allowing receipt directory replacement. The root-sharing exception itself was reproduced using the external baseline/candidate probe; it has no dedicated deterministic root-writer selector in this unchanged suite.

The independent fresh-process Windows rerun executed all 63 cases:63 passed, 0 failed, 0 skipped, 0 not run (22.62 seconds). Evidence is `C:/codedev/evidence/pcai-fleet-integration/r2/integration/independent-archive-root.{json,xml,log}`. This is execution evidence for the full suite; semantic review here is limited to the seven registered selectors and changed custody lifecycle. It makes no claim about Linux native compatibility, all 63 selector rationales, full WindowsCI or the 85 percent coverage gate.
