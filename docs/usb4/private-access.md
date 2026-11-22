# Private access on a Windows USB4 link

Run the [private access installer](../../Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1)
from an elevated PowerShell 7 session on the receiver. It changes one enrolled
USB4NET connection profile to Private and adds an inbound IPv6 echo-request
rule for the two enrolled link-local addresses on that interface. Windows
can also activate existing Private-profile inbound rules on the link. Examples
include File and Printer Sharing, Network Discovery, Remote Assistance and
other installed services. A `LocalSubnet` address filter can include a USB4
link-local peer. Inspect the rules before applying and approve the profile
change only when that exposure is suitable for every enrolled link.
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

Older registrations can leave a broad inbound allow rule active for multiple
interfaces and address ranges. Query its exact name in both policy stores and
inspect the configured Private-only inbound allow rules before promotion.

```powershell
$legacy = 'PC-AI-USB4-P2P-Allow'
Get-NetFirewallRule -Name $legacy -PolicyStore PersistentStore -ErrorAction SilentlyContinue
Get-NetFirewallRule -Name $legacy -PolicyStore ActiveStore -ErrorAction SilentlyContinue

Get-NetFirewallRule -PolicyStore ActiveStore -Direction Inbound -Action Allow -Enabled True |
    Where-Object {
        $profiles = [string]$_.Profile -split ',\s*'
        $profiles -contains 'Private' -and $profiles -notcontains 'Public'
    } | Select-Object Name, DisplayName, Profile, PolicyStoreSourceType
```

Use `Get-NetFirewallAddressFilter`, `Get-NetFirewallInterfaceFilter`, and
`Get-NetFirewallPortFilter` on rules from that query to inspect their link,
address and service restrictions. Keeping the legacy rule is an operator choice.
If the operator chooses to retire a locally owned rule, first verify its service
dependencies, then use the exact name from an elevated session. If central
policy owns the rule, only the policy owner should change it at the source.

```powershell
Remove-NetFirewallRule -Name 'PC-AI-USB4-P2P-Allow' -PolicyStore PersistentStore
```

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
untouched. Review `LegacyRule` and `PrivateProfileExposure` before executing
Apply. The exposure summary reports the total and up to 20 exact rule names.
Use the inspection command above for the full list. Apply refuses an enabled
inbound legacy allow rule in either policy store by default. An administrator
who has reviewed the existing exposure can pass `-AcceptLegacyRule` on that
Apply. The receipt records the acceptance. The legacy rule stays in its original
policy store and Rollback handles only the profile and scoped rule it owns.
Windows can associate several interfaces with one network
classification. Inspect `UnapprovedSameNameProfiles` in the plan. If every
listed interface is an enrolled USB4NET link, pass its verified adapter GUID
in `-SharedAdapterGuid`. Apply refuses to change category when a same-name
profile connection remains outside the authorized scope. It captures the
complete category snapshot and verifies the result after the change. Each
run installs one echo rule for the primary adapter and peer. Windows may
assign a newly attached adapter to that same network later. Review adapter
membership after a hardware or topology change.
[Windows Network List Manager](https://learn.microsoft.com/windows/win32/api/netlistmgr/nn-netlistmgr-inetwork)

A successful Apply reads back the connection profile and the rule in
PersistentStore and ActiveStore. Keep its receipt. Repeat Apply on
the same validated state returns `Unchanged` and preserves the receipt.
An existing rule at the deterministic name with a different owner, scope
or filter blocks the operation.

The added rule covers ICMPv6 echo-request type 128, with Private, exact source
and destination addresses and the enrolled interface alias. Existing rules
determine TCP and UDP access after promotion. Verify each required service
under its own host policy. See the [USB4 hardware test guide](hardware-testing.md)
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
