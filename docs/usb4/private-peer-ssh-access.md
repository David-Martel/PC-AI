# Enrolled private IPv4 peer SSH access

Use the existing [peer SSH helper](../../Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1) with an explicit controller Peer policy object. No build is required. Run both detecting suites before use:

```powershell
pwsh -NoProfile -File Tests/Portable/Test-Usb4PeerSshAccess.ps1
pwsh -NoProfile -File Tests/Portable/Test-Usb4PrivatePeerSsh.ps1
$policy = Get-Content ./explicit-peer-policy.json -Raw | ConvertFrom-Json -AsHashtable
# Pass $policy as -EnrolledPrivatePolicy with the exact host/GUID/alias/MAC,
# ExpectedPeerMac, LocalAddress, RemoteAddress, ProgramPath and ServiceBinding.
# Default invocation plans only. Apply remains an explicit separate operation.
```

The [existing guide](peer-ssh-access.md) defines executable/service binding, rule custody and externally bounded Apply. The [controller policy](peer-networking.md) owns address assignment; this helper never assigns an address or changes routes, ICMP, DNS, MTU, NAT, adapter state, SSH services or service SIDs. Existing link-local admission and rule names remain unchanged when no private enrollment is supplied.

## Exact private enrollment

`EnrolledPrivatePolicy` projects the controller's explicit `schemaVersion:1`, `mode:Peer`, `hostName`, `adapterMac`, `peerMac`, `localIPv4`, `peerIPv4` and integer `prefixLength:30` fields. It must match the separately declared expected local host/MAC and `ExpectedPeerMac`. The local and remote address arguments must contain exactly their respective single IPv4 hosts. Both must be distinct usable RFC1918 addresses in the same /30: network, broadcast, public, broad-CIDR, swapped, ambiguous, wrong-host, wrong-role and mismatched-MAC declarations are refused before any OS provider is called. No wildcard or subnet is ever placed in a firewall address field.

The normalized enrollment participates in the owned scope hash. Default link-local rules retain their previous hash. A private rule still binds one exact interface alias/GUID/MAC, the exact running SSH application (and service when selected), inbound TCP22 and the exact local and peer host addresses. Both default planning and private planning remain offline: the caller supplies the policy object; no policy file or network is discovered implicitly. Planning reports peer identity as declared, never authenticated.

## Fresh Apply evidence

Immediately before creating a rule, the existing preflight verifies the running service image/generation and a positive USB4-network PnP identity for the exact GUID, alias and MAC. Private enrollment additionally requires the exact local IPv4 with a known integer /30 prefix and a current neighbor row for the exact peer IPv4 on that interface. Its MAC must match `ExpectedPeerMac`, and its state must be Reachable (5). Missing, ambiguous, stale, incomplete, permanent or foreign neighbor evidence refuses the operation; the helper does not probe to manufacture evidence. The [Microsoft neighbor definition](https://learn.microsoft.com/en-us/windows/win32/fwp/wmi/nettcpipprov/msft-netneighbor) distinguishes recent reachability from a stale or statically provisioned row.

Obtain fresh ordinary peer reachability through the already enrolled path before an independently reviewed Apply. The helper repeats this preflight immediately before Add under its existing owned mutation lock. A changed peer or interface generation is refused. MAC reachability is link evidence, not cryptographic SSH peer authentication; retain the existing strict key pin separately.

Observed firewall /32 masks,255.255.255.255 masks and equal-address ranges are canonicalized only as exact hosts and compared against the complete expected rule. A broader CIDR, foreign/drifted rule or duplicate name is never overwritten or removed. Remove still requires the identical explicit owned scope, but does not require a connected peer, preserving recovery when a device is absent. Existing link-local rules are not adopted or replaced by a new private rule.

## Detecting controls and boundary

The [new suite](../../Tests/Portable/Test-Usb4PrivatePeerSsh.ps1) covers five selector groups: exact offline enrollment; malformed/mismatched policy; fresh USB4/prefix/peer evidence including second-preflight drift; private add/readback/no-op/remove custody; and real unregistered Windows COM value conversion. It uses inert providers and one unregistered COM value only. The first four groups are portable; the COM group explicitly reports NOT_RUN outside Windows. The [original suite](../../Tests/Portable/Test-Usb4PeerSshAccess.ps1) continues to exercise239 assertions and the original link-local/service safeguards. The baseline refuses the new test at its unknown enrollment parameter; the implementation must pass both suites.

The [hash-bound review](../../Tests/Portable/usb4-private-peer-ssh-review.json) covers the exact source, selectors and [dedicated CI workflow](../../.github/workflows/usb4-peer-ssh-access.yml). Fixture and unregistered-object passes do not constitute permission or evidence for live firewall mutation. Live enrollment, current images, actual addresses and peer reachability still require independent readback on each host before applying its exact reviewed rule.
