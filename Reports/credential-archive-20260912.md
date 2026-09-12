# Credential archive maintenance

The canonical archive entrypoint is
`Tools/SystemScripts/Machine/Update-BwArchive.ps1`. The workstation launcher under
`~/.machine` delegates to this source. Machine secret modules and cache files
remain outside Git, as required by the SystemScripts source policy.

The archive verifies private ACLs before export, performs bounded native calls,
checks unlocked session state, and publishes only a nonempty export. Native sync
or export failures preserve the previous latest archive. Retention matches only
timestamped archive names; manual exports and unrelated files are preserved.
Help, dry-run, and WhatIf paths do not access the vault or mutate storage.

The explicit Windows suite in `Tests/LocalMachine/CredentialRepairs.Local.ps1`
exercises the real archive, credential initializer, and private-file logic with
nonsecret external-I/O fixtures. It covers cloud/cache token resolution,
environment persistence, ACL failures, timeouts, empty exports, retention,
locked sessions, and failed candidate-session rollback. No real vault login,
unlock, synchronization, export, or decryption is performed by the suite.

Validation on 2026-09-12: the initial Pester run passed 24 tests with zero failures, skips, or
inconclusive results. The archive passed PowerShell syntax and static analysis;
an explicit integer output contract resolves the analyzer's informational finding.

Review follow-up deduplicates SYSTEM when it is the task identity, and adds two
identity regression cases. The machine-only suite now uses an explicit Pester
container and a `.Local.ps1` suffix, so recursive repository coverage discovery
does not import private workstation dependencies.

The reviewed suite passed 26 tests with zero failures or skips. An actual
discovery-only Pester run over this directory plus an external sentinel found
only the sentinel, confirming that default discovery excludes the local suite.

Plain JSON remains the compatibility default. Encrypted exports require a
separately validated recovery procedure before changing unattended jobs.
