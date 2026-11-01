# Evaluation standalone dependency discovery

A detached Evaluation consumer could autoload the compatibility aggregate while
emitting a missing-dependency warning. The aggregate exports a Common warning
function with the same name as the native cmdlet. A genuine offline dependency
warning could therefore become a foreign-producer import error.

The dependency validator now binds its five warning calls to the native
PowerShell utility cmdlet. Messages, warning-stream handling, dependency search
order and offline availability reporting are retained.

Optional runtime and project-root helpers are discovered through the loaded
Function provider. An absent optional helper no longer triggers discovery of a
separate installation. Existing helpers are still invoked; absent helpers retain
explicit-root and source-ancestry behavior.

The copied-consumer fixture includes the actual runtime/cache helper pair. Its
fresh-child result also checks that the compatibility aggregate was not loaded.
The explicit WithoutCommon case remains and still exercises genuine ancestry;
the detached unconfigured consumer still refuses artifact creation.

## Detecting validation

- Original maintained root contracts: 12 passed, 2 failed with warning-triggered
  foreign-producer import errors.
- Native warning binding alone: all 14 original root contracts passed.
- Added aggregate-absence assertion with complete helper pair: 13 passed and the
  WithoutCommon case failed because optional discovery autoloaded the aggregate.
- Loaded-function discovery: all 14 root contracts passed.
- Combined root/runtime consumer, dependency and public-contract suites:
  101 passed, 0 failed, 0 skipped, 0 not run, using Pester 5.7.1 in a fresh
  profile-free PowerShell process with memory cache and native loading disabled.

These checks exercise owned files and mocked provider boundaries. They do not
qualify native inference, deployed consumers, hardware or a performance gain.
The unchanged full Windows coverage threshold remains a separate admission gate.
