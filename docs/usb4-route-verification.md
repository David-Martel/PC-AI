# USB4 route deletion verification

Install PowerShell 7. This bounded check needs no Windows CIM server, adapters,
privileges or real route mutation:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4RouteVerification.ps1
```

[The lifecycle implementation](../Modules/PC-AI.Drivers/Public/Invoke-ThunderboltPeerLifecycle.ps1)
uses best-effort CIM lookup for optional inventory. That behavior must not certify
a mutation's postcondition: a denied or failed route query is not proof of absence.
The post-delete query therefore requires successful CIM completion. Failure
propagates into the existing action error handler, preserving `Executed=false`
and the error instead of falsely reporting success.

[The regression](../Tests/Portable/Test-Usb4RouteVerification.ps1) calls the actual
extracted helper with failing, empty and present query fixtures, retains optional
inventory compatibility, and guards strict call-site wiring through the source AST.
It never executes the lifecycle Apply function. This is query/error contract
qualification; route ownership, real deletion, concurrent route creation, driver
recovery and sleep/resume require their own Windows/HIL checks.

[Windows/Ubuntu CI](../.github/workflows/usb4-route-verification.yml) exercises the
same tests. [Independent hash-bound review](../Tests/Portable/usb4-route-query-review.json)
records the selectors and fault controls. Broader fleet topology and namespace
acceptance remains in [the network design](design/usb4-sfp-namespace-vnv-design.md).
