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

The explicit Windows suite in `Tests/LocalMachine/CredentialRepairs.Tests.ps1`
exercises the real archive, credential initializer, and private-file logic with
nonsecret external-I/O fixtures. It covers cloud/cache token resolution,
environment persistence, ACL failures, timeouts, empty exports, retention,
locked sessions, and failed candidate-session rollback. No real vault login,
unlock, synchronization, export, or decryption is performed by the suite.

Validation on 2026-09-12: Pester passed 24 tests with zero failures, skips, or
inconclusive results. The archive passed PowerShell syntax and static analysis;
an explicit integer output contract resolves the analyzer's informational finding.

Plain JSON remains the compatibility default. Encrypted exports require a
separately validated recovery procedure before changing unattended jobs.
