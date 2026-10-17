# USB4 route-query verification independent review

Reviewed the bounded follow-up on `fix/usb4-route-verification`, based on Gemini's lifecycle implementation. Source and foreign work remained untouched. The production diff adds the optional `Required` switch to `Get-SafeCimRows`, rethrows lookup failures when set, and sets it on the post-deletion `MSFT_NetRoute` lookup.

## Result

No critical or high-priority issue found in this bounded change. A failed verification lookup now reaches the surrounding catch, leaving `Executed=false` and a recorded error. Previously the empty fallback could falsely certify successful absence after a failed query. Valid empty results still certify absence; present rows remain visible. Optional inventory retains its existing best-effort fallback.

The four functional/static checks passed on Ubuntu PowerShell: required query failure propagation, optional compatibility, valid empty/present preservation, and required-switch call-site wiring. An original-like mutant removing only `if ($Required) { throw }` failed with `Required query failure was accepted as absence`. The mutant was a private `/tmp` copy; no routes, adapters or live CIM services were modified.

The call-site check is structural rather than end-to-end deletion coverage. Full lifecycle route ownership and mutation behavior, network races, and Windows hardware qualification remain separate evidence. This correction prevents false success reporting; it cannot roll back a deletion that succeeded before a verification failure.

## Reviewed SHA256

- Lifecycle source: `ac0fdab7c1a5932a1bb004b748dd8530604d479707a48ba621dee07252218a6e`
- Route verification test: `a470a40c7de89cc6efd01cb685936a771f96ee5f1344fee10eeea7f47247f0a9`

No production numeric limits, timeouts, routes or deployment configuration changed in this diff. The new switch is backward compatible unless explicitly requested. Root owns subsequent CI/documentation/register additions and Windows test receipts.
