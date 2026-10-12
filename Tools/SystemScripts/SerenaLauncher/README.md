# Serena launcher

The PowerShell entrypoint starts the existing native Serena backend through its
adjacent JavaScript protocol shim. These files preserve the reviewed dtm-p1gen7
launcher; the shim currently resolves the binary from `T:/projects/serena-source`.
Evaluate the destination host's binary location and database configuration before
installing this launcher on another machine.

An unavailable local PostgreSQL endpoint now fails before Docker discovery.
Starting or restarting managed PostgreSQL requires the explicit per-invocation
`-AllowManagedPostgresStartup` switch. An already available local endpoint or an
external database preserves the existing behavior. Windows Serena memory storage
requires PostgreSQL; the launcher does not provide a fallback database.

`-h`, `--help`, `-Help` and `-DryRun` emit static stderr text and return before
paths, environment, database checks, Docker or backend startup. Ordinary MCP
startup uses stdout only for the existing protocol.

The tests in `Tests/Unit/SerenaManagedPostgresStartup.Tests.ps1` exercise complete
prerequisite functions with inert network, Docker and wait providers. Their
seventeen controls cover default refusal, healthy/external delegation, explicit
opt-in and its failures, and complete help/dry-run behavior. They do not qualify
authenticated database readiness, native Docker stderr, effective Rust config,
binary provenance, full backend startup or a responsiveness improvement.

Install both companion files together. Preserve the installed preimage, verify
source and destination hashes, and leave client configuration and running
sessions intact. The opt-in path retains the existing native credential argument
channel; do not publish command lines or credential values.
