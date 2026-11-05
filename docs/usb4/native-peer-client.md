# Offline native peer-client plans

The `pcai-usb4-client` binary produces an SSH/SFTP executable and argument array
from explicit declarations. It does not run that command, inspect a local link,
authenticate a peer or change a connection. A completed `OfflinePlan` is not
permission to execute. Use the existing [peer SSH access guide](peer-ssh-access.md)
for the separately reviewed Windows firewall operation.

Build and run the focused contract on a qualified build host:

```text
cargo build --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb --bin pcai-usb4-client
cargo test --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb --test client_plan_contract
pcai-usb4-client --help
pcai-usb4-client plan --policy ABSOLUTE_POLICY_FILE --dry-run
```

Windows build operators use the host's CargoTools workflow. The binary is a new
target in the existing [crate manifest](../../Native/pcai_core/pcai_tb_cli/Cargo.toml).
It has no PowerShell, rclone, controller, agent-bus or Linux observer dependency.
A binary already installed elsewhere is not upgraded or qualified by this source.

## Explicit policy

The file must be an absolute, regular local path, at most 65,536 bytes. The reader
checks ancestors before descendants, refuses symlinks, and checks the opened
handle. POSIX opens additionally use nonblocking/no-follow flags so a FIFO cannot
wait for a writer. Windows refuses network/unknown drives and reparse/offline/
recall attributes. Admission/open is not an atomic traversal transaction and does
not certify transparent remote filesystems or a hard filesystem-latency bound.

This synthetic example declares Windows IPv4 SSH; it contains no credentials:

```json
{
  "schema": 1,
  "client_platform": "windows",
  "protocol": "ssh",
  "client_executable": "C:\\Windows\\System32\\OpenSSH\\ssh.exe",
  "account": "fixture",
  "host_key_alias": "fixture-peer",
  "identity_file": "C:\\Fixture\\identity key",
  "known_hosts_file": "C:\\Fixture\\known hosts",
  "local_interface": {
    "id": "11111111-2222-3333-4444-555555555555",
    "mac": "02-00-00-00-00-01",
    "index": 33
  },
  "source_address": "169.254.12.34",
  "peer_address": "169.254.56.78"
}
```

Unknown/duplicate fields and coercive scalar types are refused. Account and
host-key alias are conservative ASCII names; domain-qualified accounts are not
supported in this slice. Paths are explicit literals. Environment substitutions,
relative/device paths, unsafe Windows components and wildcard substitutions are
refused. Private/link-local IPv4 and ULA/link-local IPv6 are accepted; mismatched
families, identical endpoints, mapped IPv4 and foreign link-local zones refuse.

For SFTP, change `protocol` to `sftp` and the executable leaf to `sftp.exe`.
Also supply `sftp_ssh_executable` as an explicit absolute `ssh.exe` path. The
planner emits it through SFTP's `-S` option rather than relying on a PATH-selected
helper; helper identity remains `NotVerified`. SSH refuses this unused field.
Optional `sftp_batch_file` is an absolute literal batch-file path; it does not
authorize its commands. Omit it for an interactive SFTP argument plan. For a
declared POSIX client use `client_platform: "posix"`, POSIX absolute executable/
credential paths and the declared interface name instead of a Windows GUID.

## Argument and authority boundaries

The [pure builder and parser](../../Native/pcai_core/pcai_tb_cli/src/client_plan.rs)
return an argv array, never a shell command. IPv6 link-local endpoints use the
declared local index. SSH uses a raw scoped destination and `-b` source; SFTP
uses a bracketed IPv6 destination and `-o BindAddress`. SFTP `-b` names its batch
file, not its source. No Linux remote command is appended.

Explicit identity and pin paths are quoted as OpenSSH configuration option
values, with backslashes/quotes escaped for that parser and literal percent
characters doubled. Identity uses `-o IdentityFile` rather than escaped `-i`:
OpenSSH's `-i` path is stat-checked before later token expansion. Spaces and
percent-named directories therefore retain their declared meaning. The program
and SFTP batch-file paths remain literal process arguments.

The plan disables inherited config, proxy/jump routing, multiplex reuse,
agent/X11 forwarding, local/known-host commands and host-key updates. Strict host
checking, explicit host-key alias, identity-only authentication and source binding
remain present. Pin/key contents, executable identity and the interface are not
read or verified. The result labels them `NotCollected` or `NotVerified` and
reports inspection/execution support as false.

Default `plan`, `--dry-run` and `--what-if` have the same offline behavior. Help
does not open a policy. `inspect-local`, `execute` and `--apply` are unavailable
and return nonzero; there is no successful inspection stub. A future executor
requires fresh physical interface/source/direct-route proof and its own process
lifetime controls. Existing topology adjacency is not that proof.

The [actual builder/parser and binary controls](../../Native/pcai_core/pcai_tb_cli/tests/client_plan_contract.rs)
include IPv4/IPv6 grammar, hostile declarations, literal paths, offline capability
labels, bounded input and POSIX FIFO/symlink cases. Their
[review register](../../Native/pcai_core/pcai_tb_cli/tests/client-plan-review.json)
records source hashes and pending review separately from execution evidence.
These controls do not qualify an installed client, a live peer, reconnect,
throughput or interactive process custody. No startup speedup is claimed.

The real no-child sentinel control is POSIX-only: a private executable first
proves it can write its marker, then offline planning must leave that marker
absent. Windows does not compile that selector; its actual CLI controls verify
unqualified output, unavailable actions and no extra owned files. The common
builder contains no process or credential-read provider on either platform.

Primary syntax references: [OpenSSH SFTP](https://man.openbsd.org/sftp.1),
[client configuration tokens](https://man.openbsd.org/ssh_config.5#TOKENS),
[configuration quoting](https://github.com/openssh/openssh-portable/blob/master/misc.c),
and [identity-option handling](https://github.com/openssh/openssh-portable/blob/master/ssh.c).
