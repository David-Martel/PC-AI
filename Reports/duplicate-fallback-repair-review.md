# Duplicate fallback contract repair

The shipped [duplicate scanner](../Modules/PC-AI.Acceleration/Public/Find-DuplicatesFast.ps1)
dropped `Exclude` when fd was unavailable. Its fd-error fallback also dropped
both filters and always recursed. An excluded file could inflate duplicate counts
and savings; a nonrecursive request could admit nested files.

The repair forwards `Exclude` through ordinary enumeration and preserves
`Recurse`, `Include` and `Exclude` when fd fails. Native routing, result projection
and the shipped [.NET hash helper](../Modules/PC-AI.Acceleration/Private/DotNet-Helpers.ps1)
retain their previous behavior.

The [Unit contracts](../Tests/Unit/DuplicateFallbackContracts.Tests.ps1) exercise
actual public scanning, filesystem enumeration and SHA256/SHA1/MD5 hashing on
eight owned tiny files. Only tool discovery is mocked; an inert throwing script
forces the fd-error boundary. Four cases failed against the unchanged predecessor
at `d664fa0`, with seven passing controls. The reviewed candidate passed all eleven.
Normal repository-relative discovery then passed all eleven in 18.16 seconds,
with no failures, skips, unexecuted cases, block/container failures or source drift.
The [review register](../Tests/Fixtures/DuplicateFallback.TestReview.json) binds
each selector and the source/helper hashes to independent review.

The assertions verify genuine byte equality, unequal same-size files, inclusive
size boundaries, duplicate projections and early no-candidate behavior. They
do not qualify the native backend, establish a speedup or supply estimated CI
coverage credit. Nonrecursive provider-specific `Include` semantics, native
filter parity and hash-failure reporting remain separate review items.
