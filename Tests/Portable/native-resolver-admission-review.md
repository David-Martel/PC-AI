# Resolver admission before public native diagnostics

The [resolver](../../Native/PcaiNative/NativeResolver.cs) checks the existing
[required export inventory](../../Native/PcaiNative/PcaiCore.cs) before publishing
any selected core handle. A rejected handle is released, including when the
verification itself throws. A cleanup failure retains both admission and release
exceptions. Other native modules retain their existing resolution behavior.

The [eight xUnit cases](../../Native/PcaiNative.Tests/NativeResolverPartialCoreAdmissionTests.cs)
run through the existing managed test project and bounded child-process helper:

```powershell
dotnet test Native/PcaiNative.Tests/PcaiNative.Tests.csproj -c Release --filter FullyQualifiedName~NativeResolverPartialCoreAdmissionTests
```

Each case compiles an owned synthetic partial or full-symbol native library and
starts a fresh profile-free, non-interactive PowerShell child. Cleanup resolves
the full fixture path under its captured temporary root and refuses a linked
fixture root before recursive removal. Test and cleanup failures retain both
exceptions. The four call paths are public
CpuCount, GetDiagnostics, GetStatusDescription and the exact production resolver
callback. Partial fixtures must return the existing managed CPU/status fallbacks,
publish no resolver record and report the exact missing-export exception when the
callback is called directly. Windows also verifies the rejected module is unloaded
before the child exits. Full-symbol fixtures retain native sentinel results,
availability and the successful binding. The fixture derives its symbol inventory
from the production list rather than maintaining another required-export list.

Source-bound Windows evidence used accepted main
`e7aa85df7e329c7cc0ec73f7bcf9a9d715b0a8c2` and integration
`6bf800ffb552811c2682b67d7f2a9306e5a244af`. For each unchanged source, the actual
retained xUnit regression executes eight cases: four partial cases fail and four
full-symbol positive cases pass. With the resolver repair each executes eight
passes, zero failures and zero skips. The preceding exception-safe-r2 packet has separate fresh .NET8 console controls
for the byte-identical resolver repair, reproducing the same four paths and
Windows loader observations. Those console results remain predecessor evidence;
the current test-source revision is qualified by the fresh xUnit runs. These
overlapping cohorts are not added as unique tests.

Before repair a rejected partial core returns CPU sentinel 424242 through
GetDiagnostics while reporting IsAvailable=false. After repair this host returns
the managed CPU count 16, records no native binding and unloads the partial module.
The direct resolver reports the selected path, provenance and missing
pcai_free_string export. Raw TRX, logs, exact source snapshots and SHA256 bindings
are identified in the PR alongside the current source-bound run. Source changes
require a fresh run and independent review. Review hashes convert CRLF to LF and
retain any UTF-8 BOM; raw capture hashes remain separate.

The [existing review register](native-library-lifetime-review.json) appends one
independently reviewed disposition per new case, with exact test/resolver bytes and peer source
hashes. Earlier lifetime reviews remain historical. The existing native override
admission workflow discovers this test source automatically; no workflow is added.

Synthetic full-symbol fixtures qualify inventory, magic/string-copy round-trip,
and the called CPU/status behavior only. Uncalled symbols are inert placeholders,
not production C ABI implementations. These runs do not qualify a production Rust
core, GPU/model execution, deployment, performance, exceptional OS unload failures,
or Linux/macOS execution. Existing real-core CI controls retain those distinct
boundaries. The exception-preserving cleanup branch is source-reviewed, not an
injected operating-system failure test.

## Retained override controls

The existing [Python override fixture](test_native_override_admission.py) now
builds the actual production bridge and references it from its probe. The probe
registers the production resolver for its own P/Invoke assembly. It obtains the
complete symbol inventory from PcaiCore.RequiredExports before creating its
owned sentinel libraries, preserving the previous default 7 and explicit 42
selection checks. Its uncalled symbol placeholders establish admission inventory
only. The fixture clears unrelated bundle selection and confines discovery to
its owned root.

All six missing/unloadable override subcases, the default plus three valid
overrides, and three new loadable-partial refusal subcases pass on Windows.
Replaying the updated fixture against the unchanged accepted resolver executes
sentinel 99 in all three partial subcases and fails those assertions. The actual
production-core selector remains an explicit local skip when its artifact is
absent. Existing temporary-admission lifetime controls pass the partial/unload
case and explicitly skip their unavailable real-core positive. The existing
native admission CI still supplies and requires its host-built real core.

The [override review register](native-override-review.json) appends new
source-bound dispositions without changing earlier reviews. Current local
managed execution passes 33 selected resolver, P/Invoke and bounded-process
cases with zero skips. These cohorts overlap with the focused cases above and
are not an additive coverage total. No installed native binary was replaced.

The current review rows bind the exact source commit and selector hashes. Local
execution used the .NET11 preview SDK targeting .NET8 on Windows x64; explicit
real-core skips remain skips after the source-review keep disposition.
