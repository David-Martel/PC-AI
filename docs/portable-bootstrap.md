# Portable cache and managed build bootstrap

Install PowerShell 7 and a .NET SDK supporting net8.0. From the repository root:

```powershell
# Build: Linux x64; Windows x64 uses win-x64 instead.
dotnet build Native/PcaiNative/PcaiNative.csproj -c Release -r linux-x64
# Test and run the bounded bootstrap acceptance (no Redis server required).
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-CacheBootstrap.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-RustToolDiscovery.ps1
```

[Portable bootstrap CI](../.github/workflows/portable-bootstrap.yml) checks the actual
runner OS/architecture against its target RID, runs profile-free cache regression
checks, then builds the managed bridge on Windows and Ubuntu. It does not deploy,
change profiles or invoke hardware. A successful managed build does not qualify the
production native loader, GPU backend, native FFI calls or USB4 Apply behavior.

[Cache configuration](../Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1)
ignores absent or whitespace Windows roots before constructing optional Redis CLI
paths. An explicit configured path retains priority, followed by existing candidates
and application lookup on PATH. This fixes memory-provider bootstrap on Ubuntu
without inventing Windows environment variables or changing provider selection.

[Test selectors](../Tests/Portable/Test-CacheBootstrap.ps1) exercise the real helper:
absent roots, whitespace roots and explicit CLI precedence. Each states its purpose
and prerequisites. The original helper fails the absent-root check before the fix.
The fixture file is never executed, and modified process environment is restored.
Independent review dispositions and hashes live in
[the review register](../Tests/Portable/cache-bootstrap-review.json).

Deployment should stage commit/hash/RID-bound local payloads outside cloud-synced
profile folders and validate their actual native backend before switching a loader
pointer. Existing native loader and performance admission gaps remain separate work;
these bootstrap checks must not be presented as full PC-AI deployment qualification.

[Tool discovery](../Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1)
uses the host PATH separator and executable suffix, skips absent search folders,
and resolves applications without depending on Windows where.exe.
[Its regression selectors](../Tests/Portable/Test-RustToolDiscovery.ps1) check
exact PATH fixture selection and honest missing-tool reporting without execution.
