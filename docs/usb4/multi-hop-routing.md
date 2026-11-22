# USB4 routing across hubs and intermediate hosts

`pcai-tb route-plan` plans IPv4 addressing, host routes and transit forwarding for
USB4 hosts that share no cable. It reads one fresh
[fabric observation](fabric-topology.md) and an owner-written route policy, and
prints the commands for the owner to review and run. It changes nothing itself.
The [planner](../../Native/pcai_core/pcai_tb_cli/src/route_plan.rs) and its
[CLI entrypoint](../../Native/pcai_core/pcai_tb_cli/src/main.rs) refuse `--apply`.

For a directly cabled Windows pair, see the separate [USB4 NetworkDirect preview preflight](rdma-preview.md).

## Why hub topologies broke

The fleet's USB4 links run Gen7 to the NUC to Gen9. Gen9's adapter sits behind a
CalDigit Element 5 hub, and the NUC has one adapter per root router. Three
things prevented traffic between Gen7 and Gen9 (observed 2026-11-21):

- **Configuration is tied to the adapter instance.** Windows creates a new
  `USB4(TM) P2P Network Adapter #N`, with a new interface GUID, MAC, ifIndex and
  alias, for every parent router and port the cable passes through. Gen9 has
  seven such instances; only `#12` is present. Addresses that `register --apply`
  put on an earlier instance vanished with it. The NUC kept `172.31.255.22/30`
  while Gen9's `.21` was gone, so the /30 was half-configured.
- **Nothing routes between segments.** `register` adds only on-link addresses and
  routes for a directly connected peer. A host two hops away gets neither a
  route nor a forwarding host, and the NUC had IPv4 forwarding disabled on both
  USB4 interfaces.
- **The collector could not read a real host.** Its neighbor bound counted
  multicast rows, and its 10-second deadline was shorter than the NUC's
  collection time. PR #334 fixes both.

## What the planner does

1. It builds segments with the existing `topology-plan` logic, so endpoints
   pair by reciprocal neighbor MAC and remote address, never by alias or index.
2. It gives each segment the /30 that `compute_commutative_pairwise_ipv4`
   derives from the two registered host names. That address is the same on
   every plug. It is also what `register --assign-pairwise-ipv4` uses: the NUC's
   live `172.31.255.22` matches the derivation for `dtm-p1gen9 <-> dtm-super-nuc`.
3. It finds the shortest host path for each ordered pair. Only hosts listed in
   `transit_hosts` may appear between the two ends; adjacency never makes a
   host a router.
4. For a routed pair, it adds a `/32` host route on each hop that does not share
   the destination's /30, and enables IPv4 forwarding on both USB4 interfaces
   of each transit host. Replies are planned explicitly: the reversed path gets
   routes to the address the source sent from, its end of the first segment.
   A pair's routes are added together or not at all. Windows uses the strong
   host model by default, so a host's own packets, replies included, leave only
   through the interface that owns their source address. If the sending host of
   either direction already routes that destination through another interface,
   the pair is reported `Blocked` with the conflicting route named, rather than
   `Routed` one way. A transit host forwards on any route, so a different next
   hop there is only dropped with a warning.
5. Each host's commands begin with a guard. The guard checks that every planned
   ifIndex still carries the observed MAC and interface GUID, and stops if the
   adapter has been replaced. The plan expires with the oldest observation's TTL.

Addresses already present on an interface are reported as `already_present` and
have no rollback, because the plan did not create them. A different
`172.31.240.0/20` address on a planned interface produces a warning. Two
segments that derive the same /30 stop the plan.

## Run it

```powershell
# On each host, over an independently trusted management channel:
./Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1 -Source management-ssh |
    ConvertTo-Json -Depth 12 -Compress | Set-Content -LiteralPath C:/private/obs-<host>.json
# Merge the hosts arrays into one pcai.usb4-fabric-observation/v1 envelope, then:
pcai-tb route-plan --input C:/private/observation.json --policy C:/private/route-policy.json `
    --registry Config/thunderbolt-peers.json
```

The policy names transit hosts explicitly
([example](../../Config/usb4-route-policy.example.json)):

```json
{ "schema": "pcai.usb4-route-policy/v1", "transit_hosts": ["dtm-super-nuc"], "route_metric": 5 }
```

For each host, run `guard`, then `preflight` (read-only; keep the output), then
each action's `apply`. Each apply checks the host's UTC clock against the plan's
evaluation and expiry times. It checks again just before making a network change.
Regenerate the plan if either check fails. Each apply creates its address, route
or forwarding setting only if it is absent, and records it in a per-plan journal at
`%ProgramData%\pcai-usb4\route-plan\<plan-id>.journal`. State that existed before
apply is reported as "not owned". To undo, run the `rollback` lines in reverse
order. Rollback is available after expiry and requires a matching key in the host's
journal, which preserves state that predates the plan.
[Route expiry test review](../../Native/pcai_core/pcai_tb_cli/tests/route-plan-validity-review.json).

Every apply and rollback line checks the journal's location before accessing it.
`route-plan` must be owned by Administrators or SYSTEM, must
not be a junction or symbolic link, and must grant no other account write, delete or
permission rights. Its parent `pcai-usb4` must be admin-owned and not a reparse
point; it may keep the inherited right of Users to add files. Apply creates missing
directories with a protected ACL; a line that finds anything else stops with
"Journal storage ... review and remove it" before it reads the journal or changes
the network. Apply writes the key before it makes the change, and removes the key
again if the change fails and the state is still absent. Rollback undoes the state
only while it is present and then removes the key, so a second rollback cannot
remove an address, route or forwarding setting that someone else recreated. Rollback also checks that the ifIndex still names the adapter the plan observed (MAC and GUID); after a replug Windows can give that index to another adapter, so a mismatch changes nothing and keeps the key. The journal file must be owned by Administrators, SYSTEM or the elevated caller.

`New-NetIPAddress` disables DHCP on an
interface where it is enabled, and removing the address would not restore it, so
apply refuses a DHCP-enabled interface. The USB4 interfaces observed on 2026-11-21
had `Dhcp Disabled`.

## Limits

- IPv4 only. IPv6 ULA host routes over link-local next hops are not planned.
- Windows commands only: the observation schema describes Windows USB4NET adapters.
- Routing does not change Windows Firewall. A destination's inbound rules must
  admit the remote /30 source, for example the SSH rule on a Public-profile link
  (see [peer SSH access](peer-ssh-access.md)).
- Transit throughput is unmeasured. A plan says nothing about link speed; measure
  one hop before relying on the routed path.
- Parallel segments between the same two hosts use the lowest edge ID; the others
  stay unaddressed and are reported.

Tests: the [planner controls](../../Native/pcai_core/pcai_tb_cli/src/route_plan.rs)
use a sanitized recording of the 2026-11-21 Gen7, NUC and Gen9 chain
([fixture](../../Native/pcai_core/pcai_tb_cli/tests/fixtures/usb4-hub-chain-observation.json)),
and the [CLI controls](../../Native/pcai_core/pcai_tb_cli/tests/registry_cli.rs) run
the built binary on it.
