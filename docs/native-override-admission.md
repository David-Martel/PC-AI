# Native override admission

Install PowerShell 7 and the .NET 8 SDK; build the managed bridge with
`dotnet build Native/PcaiNative/PcaiNative.csproj -c Release`. Run the focused
initializer checks with `pwsh -NoProfile -File Tests/Portable/Test-NativeOverrideAdmission.ps1`.
Run resolver import controls with
`uv run --no-project python -m unittest discover -s Tests/Portable -p test_native_override_admission.py -v`.
The resolver controls require Rust and .NET 8; `PCAI_ADMISSION_DOTNET` can select
a private .NET 8 host. Supply `PCAI_ADMISSION_REAL_CORE_PATH` for the real-core
smoke selector; otherwise that selector reports an explicit skip. The dedicated
Windows/Ubuntu workflow builds the locked core and supplies this path.
Run the normal native integration suite separately before deployment.

An explicit core path is operator intent. In precedence order,
`PCAI_CORE_LIB_PATH`, `PCAI_TEST_CORE_DLL`, then `PCAI_NATIVE_DIR` selects the
core library. A missing selected library must stop initialization before
searching other directories, changing PATH, or overwriting the override.
A previously loaded-state flag must not turn that failure into success.
The initializer returns false and clears its availability/provenance state.

The production C# import resolver must also stop when the selected override
is missing or cannot load. Returning a zero handle delegates to the runtime's
default search, which can silently load another library. An explicit loading
exception prevents that fallback. Unconfigured discovery remains available.

The six initializer controls run in independent PowerShell processes and cover
all three override variables, both initially unloaded and stale-loaded states.
They protect authoritative path selection, detect discovery and PATH mutation,
and check that failure clears stale version/hash state. The original initializer
fails the first control by entering fallback discovery. These controls exercise
the production initializer with a discovery tripwire; they do not establish ABI,
required-export completeness, throughput, or physical USB4 readiness.

Native libraries already loaded by a process cannot be safely replaced through
this initializer. Use a fresh process to qualify a new artifact. Current managed
integration tests also require separate review of process-wide environment
mutation and parallel execution; these focused subprocess controls do not certify
the entire test suite.

Implementation: [initializer](../Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1),
[resolver](../Native/PcaiNative/NativeResolver.cs),
[initializer controls](../Tests/Portable/Test-NativeOverrideAdmission.ps1).
Related: [fleet capabilities](windows-fleet-capabilities.md).

[Resolver controls](../Tests/Portable/test_native_override_admission.py) compile
the production resolver into a minimal bridge, execute Rust sentinel libraries
through actual P/Invoke in fresh processes, and distinguish default (7) from
explicit (42) selection. Missing and unloadable overrides must reject even with
a loadable competing default artifact. This verifies resolver behavior, not the
full managed bridge surface. The real-core selector checks the production magic
export; it does not establish all required exports or operation correctness.

[Recorded selector review](../Tests/Portable/native-override-review.md) links the
hash-bound dispositions for these controls.
