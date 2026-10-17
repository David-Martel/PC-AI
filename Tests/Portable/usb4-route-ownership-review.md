# USB4 route ownership controls

The [six controls](Test-Usb4RouteOwnership.ps1) execute the actual production route-withdrawal foreach AST with bounded fixtures. Mocked deletion and CIM readback record intended calls; no host route API is invoked. The production repair requires positive membership in the current USB4 interface inventory, including when that inventory is empty.

Current controls pass for empty inventory, a foreign interface, a positively inventoried interface with exact prefix, WhatIf, default-route exclusion and non-PC-AI-prefix exclusion. The original d5b5eda loop fails the first selector: it calls the mock deletion on foreign interface999 with an empty USB4 inventory and reports Executed=true. This is a real planner defect; no real network route was deleted. The positive owned-interface case prevents an unconditional no-op repair from receiving a pass; WhatIf checks actual ShouldProcess behavior.

The [register](usb4-route-ownership-review.json) retains unreviewed status until an independent reviewer evaluates the authored controls. These checks isolate the withdrawal loop; they do not qualify adapter discovery, complete lifecycle Apply, racing interface reuse, kernel behavior, physical USB4 availability or route rollback.

Related [query controls](usb4-route-query-review.md) distinguish verification-query failure from legitimate absence after deletion. Together the source guards stop unowned deletion and false success reporting, with actual host mutation still held for separate HIL acceptance.

Root independent test review: all six controls PASS; isolated original d5 source fails the empty-inventory selector as expected. Verified LF-normalized hashes, exact prefix/interface assertions, zero-mutation exclusions and actual ShouldProcess WhatIf behavior. Disposition: keep all six.
