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

## Stdio framing and limits

The JavaScript bridge selects framed or newline-delimited JSON separately for
each direction and retains that mode. An incomplete `Content-Length` header
cannot become a JSON-line message. Plain JSON is held until a newline or valid
EOF; framed bodies are held until the declared byte count arrives. UTF-8 and
JSON are validated without reserializing parsed values. Inbound framed messages
retain their original bytes. Outbound framed JSON is compacted into one physical
line by removing only insignificant whitespace outside strings, as required by
the [MCP stdio transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports).
Numeric lexemes, Unicode and all string bytes and escapes remain unchanged.
A truncated or malformed frame/JSON message fails nonzero. Child closure also
refuses an unfinished inbound message, including a JSON tail lacking client EOF;
an idle, open client input is permitted.

Payloads are limited to 8 MiB and complete headers to 16 KiB, including the header
delimiter. These bounds allow substantial tool responses while limiting retained
memory; the queue to each destination is capped at 16 MiB plus 16 KiB. Stream
backpressure pauses the corresponding source. Exceeding a bound refuses the
message; these are explicit transport limits, not measured Serena workload
capacity or an assurance that every tool response will fit.

Child exit does not imply stdout is drained. Completion waits for child stdio
closure and outstanding destination write callbacks, then lets the wrapper's
own pending writes flush naturally. A late write failure remains nonzero. A nonzero
child result remains nonzero. Signal termination is reported numerically as
`128 + signal number` after drainage (for example, SIGTERM is 143 and SIGKILL is
137), rather than terminating the wrapper with that signal. A protocol or pipe
failure closes only the child
started by this wrapper, with a one-second termination grace before a hard kill.
PostgreSQL service, Docker, binary selection and project/database arguments are
unchanged. No extra parser companion file is required at installation.

Run the bounded transport controls with:

```powershell
node --test Tests/Portable/test-serena-stdio.js
```

The tests execute the maintained whole launcher with inert external spawn/file
boundaries, plus owned Node children and a real wrapper subprocess with slow
stdout and open client stdin. They cover fragmented headers/bodies/Unicode,
coalesced messages, JSON-line/EOF handling, exact limits, malformed UTF-8,
backpressure, nonzero exits and drainage. They start no Serena backend, database
or Docker service and do not qualify deployed binary/database behavior or load.
