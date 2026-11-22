# Private access on a Windows USB4 link

Run the [private access installer](../../Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1)
from an elevated PowerShell 7 session on the receiver. It changes one enrolled
USB4NET connection profile to Private and adds an inbound IPv6 echo-request
rule for the two enrolled link-local addresses on that interface. Windows
uses the Private profile to classify the link and can also activate existing
rules assigned to Private. Inspect those effective rules before applying.
[Microsoft USB4 interdomain testing](https://learn.microsoft.com/windows-hardware/design/component-guidelines/usb4-interdomain-connections#testing)

## Inspect and apply

Collect the adapter GUID, local MAC and preferred local IPv6 address from
the receiver. Select the peer MAC and scoped link-local address from the
same interface's neighbor table. Put the addresses in the command without
their zone suffixes. The installer resolves the live interface index,
requires an on-link route with the intended source, and checks the peer
IP and MAC against Windows' cache. That cache proves observed adjacency.
Use the [direct SSH guide](peer-ssh-access.md) to authenticate the remote
service with a pinned key.

The following values are examples. Replace them with the enrolled link's
identities and select a receipt location in an existing, operator-controlled
directory on a local fixed drive.

```powershell
$enrolledLink = @{
    AdapterGuid = '10000000-0000-0000-0000-000000000001'
    LocalMac = '02-00-00-00-00-41'
    LocalAddress = 'fe80::41'
    PeerMac = '02-00-00-00-00-42'
    PeerAddress = 'fe80::42'
}
$receipt = Join-Path $env:USERPROFILE 'usb4-private-receiver.json'

./Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1 @enrolledLink -DryRun
./Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1 @enrolledLink -Apply -ReceiptPath $receipt
```

A default run, `-DryRun` and `-Apply -WhatIf` leave the host and filesystem
untouched. Windows can associate several interfaces with one network
classification. Inspect `UnapprovedSameNameProfiles` in the plan. If every
listed interface is an enrolled USB4NET link, pass its verified adapter GUID
in `-SharedAdapterGuid`. Apply refuses to change category when a same-name
profile connection remains outside the authorized scope. It captures the
complete category snapshot and verifies the result after the change. Each
run installs only the echo rule for the primary adapter and peer.
[Windows Network List Manager](https://learn.microsoft.com/windows/win32/api/netlistmgr/nn-netlistmgr-inetwork)

A successful Apply reads back the connection profile and the rule in
PersistentStore and ActiveStore. Keep its receipt. Repeat Apply on
the same validated state returns `Unchanged` and preserves the receipt.
An existing rule at the deterministic name with a different owner, scope
or filter blocks the operation.

The added rule covers ICMPv6 echo-request type 128, with Private, exact source
and destination addresses and the enrolled interface alias. It adds zero
TCP or UDP service rules. See the [USB4 hardware test guide](hardware-testing.md)
for scoped connectivity validation and the separate temporary benchmark rules.

## Roll back

Keep the enrolled links attached and reuse the original arguments,
`SharedAdapterGuid` list and receipt. If several installs share a
classification, roll them back in reverse application order.
Rollback removes the rule created by that transaction and restores the prior
connection category when the installer changed it. A mismatched adapter,
profile or rule blocks rollback and leaves the receipt available for recovery.
An interrupted rollback can be resumed with the same receipt. A completed
rollback keeps the receipt as a record. Use a fresh path for a new Apply.
If a remote caller times out, inspect the receipt and live state before retrying.
Windows Firewall and connection-profile calls can complete after the caller exits.

```powershell
./Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1 @enrolledLink -Rollback -ReceiptPath $receipt
```

## Test locally

The [portable fixture](../../Tests/Portable/Test-Usb4PrivateAccess.ps1)
replaces Windows providers with inert substitutes. It exercises read-only
planning, invalid link evidence, shared profile admission and recovery
from collateral category changes. Use the
[hardware test guide](hardware-testing.md) separately to verify packet flow.
The [review record](../../Tests/Portable/usb4-private-access-review.json) binds
the fixtures, installer and CI step.

```powershell
pwsh -NoProfile -File Tests/Portable/Test-Usb4PrivateAccess.ps1
```
