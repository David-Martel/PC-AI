# Enrolled USB4 observer

Build and test from the repository root:

```sh
cargo build --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb --bin pcai-usb4-agent
cargo test --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb
sudo /usr/local/libexec/pcai-usb4-agent observe /etc/pcai-usb4/policy.json
```

The [policy model](../../Native/pcai_core/pcai_tb_cli/src/edge_policy.rs) validates enrolled node/controller UUIDs and derives symmetric per-edge /64 addresses inside an approved locally assigned ULA /48. A full enrolled-pair collision check runs before inventory. Derived addresses are proposals, not installed addresses or fleet-wide IPAM reservations.

The [observer](../../Native/pcai_core/pcai_tb_cli/src/edge_observer.rs) reads root-namespace thunderbolt-net interfaces and actual sysfs controller ancestry. Interface names are presentation, not identity. Enrolled edges remain `ENROLLED_UNVERIFIED`: controller UUIDs are not cryptographic authentication. Explicitly protected or mastered interfaces are `PROTECTED`; unknown identities remain `UNKNOWN`. Missing inventory fails rather than producing an empty success. Listed namespaces are explicitly reported as uninspected; this implementation does not claim complete topology or namespace coverage.

The [CLI](../../Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-agent.rs) accepts only observation. Its policy must be a root-owned regular file without group/world write permission. Install the built binary at `/usr/local/libexec/pcai-usb4-agent`, policy at `/etc/pcai-usb4/policy.json` (root:root, mode600), and the [service](../../Deploy/usb4/pcai-usb4-observer.service) and [timer](../../Deploy/usb4/pcai-usb4-observer.timer) under `/etc/systemd/system/`. Verify units before enabling the timer. Read observations through `journalctl -u pcai-usb4-observer.service`. Stop/disable the timer to withdraw monitoring; no routes or profiles require rollback because observation never changes them.

The service drops all capabilities, restricts address families and makes the filesystem read-only. It intentionally has no SSH credentials or network-configuration authority. A timer refreshes observations after reconnect and resume without subscribing to a particular interface name. This is periodic discovery, not immediate event-driven recovery or failover.

## Verification boundaries and remaining integration

Policy tests protect deterministic reversed-endpoint allocation and reject unknown/local mismatches, unsafe runtime names, duplicate enrollment, invalid pools and actual subnet collisions. Observer fixtures protect actual sysfs ancestry, renamed interfaces, existing master ownership and unknown-peer nonpromotion; the missing-inventory selector protects provider failure semantics. Tests are adjacent to the implementation above. Current live receipts and independent reviews belong in the fleet evidence repository, not embedded deployment credentials.

Pending acceptance gates: persistent exact-profile adoption and rollback; namespace-aware ownership; authenticated pinned-key SSH on the exact source/egress; atomic status records with boot ID, epoch, provider completeness and freshness; event-driven udev enqueue; unplug/resume/profile-interruption tests; Windows provider parity; and integration with vigil-utils/vigil-spark and vigil-ai-tpm traceability. None is implied by a successful observation or service run. No default route, bridge, firewall, forwarding or existing profile is changed here.
