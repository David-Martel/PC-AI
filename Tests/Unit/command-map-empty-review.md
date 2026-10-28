# Empty module command discovery review

The [cache-key helper](../../Modules/PC-AI.CLI/Private/Get-PcCommandMapCacheKey.ps1) already returns a zero-manifest key, but mandatory parameter binding rejects null and empty collections before that body executes. The [public producer](../../Modules/PC-AI.CLI/Public/Get-PCCommandMap.ps1) receives null from real discovery in an existing empty Modules directory, so help/version discovery fails. AllowNull and AllowEmptyCollection admit those existing fallback inputs.

Run the [five Pester controls](PcaiCommandMapEmptyModules.Tests.ps1) from the repository root in a fresh process:

```powershell
pwsh -NoProfile -Command "Invoke-Pester -Path Tests/Unit/PcaiCommandMapEmptyModules.Tests.ps1 -Output Detailed"
```

They use maintained producers and disposable TestDrive manifests, force the memory cache and restore prior global/environment state. No aggregate module, native library, provider service, model or hardware is required. The nonempty-manifest case checks real mapping and invalidation; an injected cache marker proves a matching entry is served, and caller mutation must not alter stored state.

## Source comparison and observed controls

The source was recovered from the unpublished Gen7 commit e9c5d91149e253f779616f223142f5c1cdd8c843. Only its two parameter attributes and focused fixture are proposed. No other integration-branch source is imported.

The helper exists unchanged at main 06b33e541760ab75402d24235a2ba645a10bb7b3 and draft PR233 head 55201a778a2fed555484bf3538b2013fbaebc703. Their shared-cache implementations differ, so both were exercised as exact source snapshots. On Windows PowerShell7.6.7/Pester5.7.1, the final fixture with rationales produced:

| Source snapshot | Original helper             | Attribute-only candidate      |
| --------------- | --------------------------- | ----------------------------- |
| main 06b33e5    | 1 pass, 4 failures, 0 skips | 5 passes, 0 failures, 0 skips |
| PR233 55201a7   | 1 pass, 4 failures, 0 skips | 5 passes, 0 failures, 0 skips |

The baseline failures explicitly reject null/empty binding and public empty-directory discovery. The nonempty case passes before and after. To reproduce the baseline against the current checkout, write the old helper to a temporary file and pass it through the fixture's source parameter:

```powershell
$baseline = Join-Path ([IO.Path]::GetTempPath()) 'pcai-command-map-baseline.ps1'
git show 06b33e541760ab75402d24235a2ba645a10bb7b3:Modules/PC-AI.CLI/Private/Get-PcCommandMapCacheKey.ps1 | Set-Content -LiteralPath $baseline
$container = New-PesterContainer -Path Tests/Unit/PcaiCommandMapEmptyModules.Tests.ps1 -Data @{ CommandMapCacheKeySourcePath = $baseline }
Invoke-Pester -Container $container -Output Detailed
```

The [selector register](command-map-empty-review.json) binds the final test bytes and main producer sources to the local raw controls. Its disposition remains unreviewed until an independent reviewer assesses those exact bytes. Full Windows suite/85% coverage, Linux execution, installed-consumer admission and performance remain separate boundaries. This fix changes no timing or inference behavior.
