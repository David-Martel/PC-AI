# Machine-secret diagnostic boundary

When the actual Common module was already loaded, its exported error function
used a separate module preference. A SecretsTier caller selecting
`-ErrorAction SilentlyContinue` could therefore receive a terminating error from
a rejected synthetic backup or failed synthetic synchronization.

Seven intended native error emissions now bind directly to the PowerShell
utility cmdlet. Messages, categories, ShouldProcess decisions and return paths
are retained. The registered Common source is unchanged.

The detecting fixture runs the actual Common and SecretsTier modules in a fresh
profile-free child. Synthetic restore/sync boundaries exercise Continue,
SilentlyContinue and Stop, including error capture and emitted categories.
External provider, cache, environment-publication and process tripwires must
remain uncalled.

## Evidence and remaining gates

- Original selected controls passed 5/5 in isolation, but failed 2/5 with Common
  loaded. A pure synthetic diagnostic probe reproduced the preference boundary.
- The final maintained regression failed against an immutable old producer.
- Six selected controls passed with no failures or skips, retaining the existing
  20-second child limits.
- The broader maintained synthetic backup subgroup passed 87 controls with no
  failures or skips. This is not a full-suite acceptance claim.
- Analyzer reported no warnings or errors; independent source review found no
  issues and verified the exact producer, fixture and unchanged Common hashes.

Existing fixtures do not separately cover locked export, empty export, locked
restore or missing-note diagnostic branches. The earlier NUC COMPUTERNAME
child timeout was not reproduced locally; it remains a full-run verification
gap. Full Windows coverage, installed consumer identity and authenticated fleet
operation remain separate gates. These checks used no live credentials or
external providers.
