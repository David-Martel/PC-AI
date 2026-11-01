# PC-AI aggregate compatibility

`Modules/PC-AI` preserves the observed PC-AI 0.1.0 aggregate's 164 function names
and GUID as version 0.1.1. It contains only a manifest and loader. Component
implementations remain in the twelve current standalone producers; there is no
copied `Modules` subtree or parent-relative `NestedModules` manifest reference.

The manifest records the named producer map. The loader requires sibling
components under its own admitted module root, validates all required paths and
root script names before importing any component, and refuses an incomplete
export contract. It never fills missing components from another PSModulePath
root. For repository use only, the maintained flat `PcaiInference.psd1` layout is
accepted when the repository CLI and installer markers are present. Installed
standalone modules use their own named directories. Component reparse points are
refused; publish the coherent package set in Copy mode.

Inherited functions retain their producer-bound script blocks and parameter
contracts. `Initialize-PcaiNative` explicitly selects PC-AI.Acceleration. Both
current implementations accept `Force`, and Common delegates to Acceleration;
selection therefore follows the implementation contract rather than import
order. The facade itself does not invoke that initializer.

Two aggregate-owned helpers remain:

- `Import-PcaiComponentModules` reloads the admitted components with Force and
  emits no success output. Missing or foreign components now fail closed instead
  of silently leaving an incomplete aggregate. This is not an atomic runtime
  reload guarantee; coordinate callers before replacing installed packages.
- `Get-PcaiReleaseInfo` retains `ModuleRoot`, `ConfigPath` and `Components`.
  ConfigPath remains `ModuleRoot/Config`. Components now contain paths relative
  to ModuleRoot that address the actual sibling producers. Joining each path to
  ModuleRoot remains valid without embedding component bodies.

Publish through `Tools/Install-PcaiDevModules.ps1` after binding its final source
head and hash. Include the aggregate and all twelve dependencies in the same
reviewed Copy publication, along with other explicitly selected current modules.
The existing installer stages and validates every selected payload before active
renames. Its normal revision custody retains the displaced legacy manifest,
loader, nested bodies and original Config bytes. Active JSON Config preserves
installed values while adding defaults; formatting can change. Opaque Config
files retain their bytes. Do not discard the retained previous installation.
Private configuration and credential settings remain on their original host.

Fresh-process validation must confirm the installed aggregate and all producers
resolve under the admitted root, the 164-name map is complete, and ConfigPath
still addresses retained local settings. Inspect direct standalone imports too:
an older OneDrive module may otherwise shadow a new LocalAppData copy. Keep
PcaiNativeBridge and its paired binary custody separate; this facade does not
replace or retire it. No fleet install is qualified by source tests alone.

`Tests/Unit/PcaiAggregateCompatibility.Tests.ps1` loads the actual facade over
independent inert producers. Nine controls detect name-map drift, lost producer
state/parameter context, initializer ambiguity, helper regressions, foreign or
missing producer fallback, missing exports and old-body/config custody using the
actual installer against owned temporary paths. They execute no models, native
DLLs, USB operations or live providers. Passing them does not establish the
behavior or performance of every delegated diagnostic function.
