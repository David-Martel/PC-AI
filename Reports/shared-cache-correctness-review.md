# Shared-cache correctness review

Reviewed October 10, 2026 UTC. Owner: Codex integration lane.

Four cache defects broke reuse or crossed namespace boundaries. Directory dependencies
read a nonexistent `Length` property under StrictMode. A dictionary value named `Keys`
shadowed actual key enumeration and lost data during copying. Local namespace clearing
interpreted wildcard characters. Redis clearing supplied an unescaped glob and trusted
all returned scan keys. The repair keeps file-length sensitivity, enumerates dictionary
keys through `PSBase`, compares literal local prefixes, and escapes Redis glob data before
independently filtering every deletion against the literal prefix.

The detecting baseline passed one and failed seven of eight new expanded controls.
The repaired module passed eleven substantive controls, with one explicit live-Redis
skip, no failures and no unexecuted cases. The unavailable-Redis branch now calls
`Set-ItResult -Skipped`; its former early return produced a false pass. Available-provider
hydration assertions remain intact. No Redis server was contacted by this validation;
the deletion-boundary control mocks only transport and provider status. Script analysis
and whitespace checks passed.

[The selector review](../Tests/Fixtures/PcaiSharedCache.TestReview.json) binds all eleven
assertion bodies, twelve expanded rows and the production source. Independent review
is preserved in `.pcai/integration/pcai-shared-cache-correctness-peer-r2/review.json`
(`301D198B26E4DAB7DA623FF1AE6C50344CFDFB95D4D41CB6B8B3136E7A16ED87`).
Actual result/XML/log files are under
`D:/pcai-relocation/pcai-shared-cache-correctness-r2/`; baseline failures remain under
the adjacent `pcai-shared-cache-correctness-r1/`. Four legacy rationale comments were
added after the run, outside exact unchanged reviewed assertion bodies.

This is a correctness repair. Performance, current-head coverage credit and live Redis
acceptance are unmeasured. Existing external-process capture lifecycle and general clone
cardinality gaps remain open; these tests do not qualify them. The fleet's unchanged
85% coverage requirement still blocks PC-AI integration.
