# Virtualization fixture provider isolation

Install: PowerShell 7 and Pester 5; no deployment or Defender configuration changes.
Build: none. The [unit fixture](PC-AI.Virtualization.Tests.ps1) imports the selected [module](../../Modules/PC-AI.Virtualization/PC-AI.Virtualization.psd1).
Test/run from the repository root: `Invoke-Pester ./Tests/Unit/PC-AI.Virtualization.Tests.ps1`. The fd fixture used in the integration-draft comparison is absent from accepted main.

Pester resolved `Mock Add-MpPreference` by importing the live ConfigDefender compatibility provider before installing its mock. A fresh process executing the three existing Defender selectors followed by the four unchanged physical-root fd cases reproduced a downstream `RemoteException` when entering the owned custom drive. The command-lookup trace binds the provider import to the actual Mock call. A Drivers-only discovery sequence passed all four fd cases and was rejected as the cause.

The unit fixture now defines an inert module-local Add-MpPreference function before Pester command resolution, alongside its existing Hyper-V boundaries. Its body throws on an unmocked call. Production code and its ShouldProcess behavior are unchanged. The new selector compares loaded provider modules and remoting sessions with their pre-Mock state. Its [hash-bound review register](virtualization-fixture-review.json) records independent root KEEP at the exact source/test hashes.

The actual old fixture produced 6 passing and 1 failing selected cases. With the new assertion but no stub, the same sequence produced 6 passing and 2 failures: ConfigDefender imported and the actual custom drive remained inaccessible. With the stub, all 8 selected cases passed, 0 failed and 0 skipped. Evidence lives at `C:/codedev/evidence/pcai-fleet-integration/r2/integration/{fd-defender-red,defender-isolation-selector-red,fd-defender-green}.log` and sibling NUnit XML directories. These are Windows fixture isolation results, not native model, Linux or 85 percent coverage acceptance.

Root independently reran the integration-draft pair (not the accepted-main producer): 50 passed, 0 failed, 0 skipped, 0 not run in 8.98 seconds; evidence is `C:/codedev/evidence/pcai-fleet-integration/r2/integration/virtualization-root-peer.{json,log}`. The three registry fixture setup failures and one full-cohort syntax child timeout remain separate diagnostics.

On accepted main with the separate runtime-discovery correction and canonical plural Defender export, the actual primary fixture executes 38 cases: 38 passed, 0 failed, 0 skipped, 0 not-run. Evidence: `C:/codedev/evidence/pcai-fleet-integration/r2/integration/public-api-virtualization38-green.log` and `publication-virtualization.json`. The fixture retains its original assertions and adds the reviewed discovery guard and fail-closed unit boundaries so active administrator cases use owned mocks.

Independent root rechecked the exact fixture and new primary producer scope, then reran the combined API6 plus Virtualization38: 44 passed, 0 failed/skipped/not-run in 15.03 seconds. The source/test hashes remain in the registers; receipt `C:/codedev/evidence/pcai-fleet-integration/r2/integration/public-api-root-peer.json` supersedes the earlier draft-pair result for accepted-source execution.
