# Portable Thunderbolt peer tooling

Install: build the CLI, review its dry-run report, and use `install --apply` only after reviewing existing destination files. This candidate has not been installed as a fleet service.

Build: from `Native/pcai_core`, run `cargo build --locked -p pcai-tb`. Test: `cargo test --locked -p pcai-tb`; lint: `cargo clippy --locked -p pcai-tb --all-targets -- -D warnings`. Run: `pcai-tb status --compact` or `pcai-tb identity --computer fixture --registry /path/to/registry.json`. An explicitly requested registry must exist and pass schema/address validation. Installed defaults are validated when identity, discovery, registration or installation loads them.

This crate imports uncommitted Windows development work for Linux review. Its source custody, exact revisions, observed limitations and isolated Linux QA are recorded in the machine's PC-AI import report. It is not an upstream release or a qualified network provisioning service.

## Failure contracts and tests

`tests/registry_cli.rs` invokes the actual CLI executable. It protects the entrypoint against malformed JSON, unknown schemas, duplicate fleet addresses and missing explicit configuration; these must fail before producing success JSON or installing files. A valid configuration is a positive control. `src/configure.rs` tests failed file copies and failed OS action reports. `src/platform/linux.rs` tests missing, negative, overflowing and malformed speed measurements. The existing integer JSON schema uses zero for unknown speed; zero is not observed throughput.

Unknown link speed must not become an invented 20Gb/s measurement. Failed installation copies propagate an error. Applied registration emits its action report then exits nonzero if an action failed. Installation is not transactional: an error may leave an earlier successfully copied destination; rollback and atomic replacement remain deployment prerequisites.

## Deployment limits

Linux currently reports hard-coded interface metrics; MTU configuration does not set the requested metric. Registration is temporary `ip` configuration, without NetworkManager ownership or reboot persistence. Windows metric changes likewise require review. Do not deploy automatic apply until those contracts, strict CLI argument validation, cancellation/timeouts, identity authentication, route rollback and management-path preservation are implemented and tested.

No throughput, reconnect, DHCP forwarding, bridge-loop protection, Windows compilation or hardware acceptance is established by the unit tests. Discovery hints and SSH/NTLM banners are not authenticated host identity. Benchmark collection is not a link throughput benchmark.

## Source custody and review follow-through

[source-custody.json](source-custody.json) preserves baseline hashes from both Windows development variants. The gen7 variant includes later registry validation work; its handshake consumer needed adapting to the normalized schema. Files in this candidate include subsequent Linux reliability fixes and formatting, so these baseline hashes are provenance, not current-file hashes. Windows checkouts and their work in progress were preserved.

Future addressing work should prioritize explicit IPv6 link-local interface scopes and an allocated fleet ULA prefix, with IPv4 fallback where needed. Do not infer a trusted peer from an unauthenticated hostname, hard-code new addresses, or bridge independent upstream networks without loop/domain admission checks.

Linux QA used exact crate source in a single-member fixture to avoid unrelated workspace Git dependencies. Separate output directories are mandatory for pristine and mutated source fixtures: sharing the same Cargo target and mounted source path can reuse a stale mutated executable. Final verification used a newly created pristine target after both mutation suites, with strict clippy and all 11 unit/3 executable integration tests passing. Windows and full workspace qualification remain outstanding.
