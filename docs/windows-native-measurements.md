# Windows native measurement contracts

Hardware JSON must distinguish missing data from measurements. The native core
keeps its C ABI unchanged; consumers must accept nullable measurement fields.

The native workspace requires Rust 1.95 or newer: the locked sysinfo 0.39.6
dependency declares that minimum. The previous workspace declaration of 1.85
was inconsistent with its dependencies. Optional backends may impose additional
requirements; this host's Rust 1.99 successfully checks the core. Run CargoTools'
`Test-BuildEnvironment -Detailed` before diagnosing compiler or linker failures.
The managed bridge uses .NET 8 and a compatible PowerShell SDK; validate P/Invoke
signatures separately from loading a complete optional inference backend.

CPU sampling retains a process-wide sampler. Its first call returns null entries
in `cpu_usage`, null `avg_load`, and `cpu_sample_status: WarmingUp`. A subsequent
call after sysinfo's minimum CPU update interval returns a sampled percentage.
Calls before that interval retain the previous sample. The sampler does not sleep
in production. `cpu_sample_age_milliseconds` describes the retained sample age.

`ComponentTemp.celsius` is nullable. Unavailable or non-finite temperatures are
null rather than zero. ACPI thermal zones do not establish CPU core temperature.

`SMART_GET_VERSION` establishes a capability only. Native disk rows therefore
remain `status: Unknown`, `severity: Unknown`, `smart_status_ok: null`, and
`health_data_available: false`; `health_source` identifies the capability probe.
Physical drive handles request read access only. This is not NVMe SMART or a
measured failure prediction. Use storage reliability counters or an applicable
vendor SMART tool separately, preserving provider errors and unavailable data.

The PowerShell hardware wrapper also rejects passing legacy native rows without
explicit measured-health provenance. A CIM operational status alone does not
establish complete SMART health. Keep that limitation visible to callers.

Source validation is separate from deployment: a successful Rust check or test
does not update the installed DLL. A release requires a matching managed/native
bundle, fresh-process ABI and runtime checks, and a retained previous bundle.
Keep builds bounded by the machine profile; do not build optional inference or
CUDA components as a prerequisite for passive fleet discovery.
