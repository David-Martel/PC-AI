# Persisted USB4 state admission review

The new [state reader](../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1)
refuses unreadable or non-object persisted custody before the controller enters
`InitialObservation`. Missing state still permits bootstrap. Existing JSON `{}`
also remains readable because the controller legitimately publishes it after a
successful rollback to a missing-file preimage. Neither admission nor a readable
backup clears an unfinished transaction.

The independent reviewer, `codex-p1gen9-connectivity-context`, did not author the
production change or [new controls](Test-Usb4StateAdmission.ps1). The
[hash-bound register](usb4-state-admission-review.json) identifies all five selector
groups, exact LF-normalized source and test bytes, review time and base commit.
The review covers an uncommitted working snapshot based on
`96135799f1a0d0e1a80e30aa0ed9a1e008510b0b`; its hashes identify the reviewed change.

## Finding resolved

The initial patch rejected every empty object. Existing rollback code restores
the previous empty state after a fully successful initial rollback, then publishes
that exact object. Rejecting it would strand the next run despite successful
rollback. The author corrected the reader to accept objects, including `{}`, and
added a focused compatibility and byte-preservation control. No other review
finding remains open for this narrow change.

## Independent execution

On Windows x64 dtm-p1gen9, PowerShell 7.6.7:

- [State admission](Test-Usb4StateAdmission.ps1): 45 assertions pass, including
  the bounded real Windows DryRun child and corrupt-file byte preservation.
- [Existing controller controls](Test-Usb4PeerNetwork.ps1): 110 assertions pass,
  including pending rollback, failed commit and publication failures.
- [Existing installer controls](Test-Usb4PeerNetworkInstall.ps1): 14 groups pass.
  Installer and installer-test sources are unchanged from the base commit.
- All three scripts also pass sequentially in one fresh profile-free PowerShell
  child, matching their shared workflow process behavior.

The parent reported a desired-task installer failure in an existing shell. It was
not reproduced in two standalone runs or the combined fresh-child run. Its cause
remains unclassified; these observations do not establish a baseline defect or
justify changing the installer.

The parent's recorded failing-first run used the original actual state-read
assignment, which accepted the zero-byte journal. That historical run was not
independently repeated by this reviewer. The reviewer inspected the detecting
cases and independently executed the corrected production source.

## Limits

Keep all five selector groups for the exact source recorded. This is state-shape
admission and regression coverage. It does not repair installed state, resolve a
pending rollback, authenticate a release, qualify live networking, prove crash
durability or add a state-file size bound. Host/schema/policy ownership validation
remains in reconciliation. The fixtures use owned temporary files; no installed
policy, adapter, task or firewall is modified. Linux execution is separate, and
the Windows child explicitly reports `NOT_RUN` there.

The [operator guide](../../docs/usb4/peer-networking.md) describes recovery custody;
the [USB4 workflow](../../.github/workflows/usb4-lifecycle.yml) runs the new suite
alongside the existing controller and installer controls.
