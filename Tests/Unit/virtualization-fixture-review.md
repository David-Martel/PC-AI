# Virtualization fixture provider isolation

Install: PowerShell 7 and Pester 5; no deployment or Defender configuration changes.
Build: none. The [unit fixture](PC-AI.Virtualization.Tests.ps1) imports the selected [module](../../Modules/PC-AI.Virtualization/PC-AI.Virtualization.psd1).
Test/run from the repository root: `& ./Tests/.pester.ps1 -Profile Unit -Path @('Tests/Unit/PC-AI.Virtualization.Tests.ps1','Tests/Unit/PcaiFdSearchContracts.Tests.ps1')`.

Pester resolved `Mock Add-MpPreference` by importing the live ConfigDefender compatibility provider before installing its mock. A fresh process executing the three existing Defender selectors followed by the four unchanged physical-root fd cases reproduced a downstream `RemoteException` when entering the owned custom drive. The command-lookup trace binds the provider import to the actual Mock call. A Drivers-only discovery sequence passed all four fd cases and was rejected as the cause.

The unit fixture now defines an inert module-local Add-MpPreference function before Pester command resolution, alongside its existing Hyper-V boundaries. Its body throws on an unmocked call. Production code and its ShouldProcess behavior are unchanged. The new selector compares loaded provider modules and remoting sessions with their pre-Mock state. Its [hash-bound review register](virtualization-fixture-review.json) records independent root KEEP at the exact source/test hashes.

The actual old fixture produced 6 passing and 1 failing selected cases. With the new assertion but no stub, the same sequence produced 6 passing and 2 failures: ConfigDefender imported and the actual custom drive remained inaccessible. With the stub, all 8 selected cases passed, 0 failed and 0 skipped. Evidence lives at `C:/codedev/evidence/pcai-fleet-integration/r2/integration/{fd-defender-red,defender-isolation-selector-red,fd-defender-green}.log` and sibling NUnit XML directories. These are Windows fixture isolation results, not native model, Linux or 85 percent coverage acceptance.

Root independently reran the complete pair: 50 passed, 0 failed, 0 skipped, 0 not run in 8.98 seconds; evidence is `C:/codedev/evidence/pcai-fleet-integration/r2/integration/virtualization-root-peer.{json,log}`. The three registry fixture setup failures and one full-cohort syntax child timeout remain separate diagnostics.
