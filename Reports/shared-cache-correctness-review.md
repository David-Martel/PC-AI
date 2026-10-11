# Shared-cache correctness review

Reviewed October 11, 2026, 13:22 UTC. Owner: Codex integration lane.

## Current corrective validation

The earlier review below retains its predecessor scope. Additional controls
against that unchanged helper pass12, fail9 and skip the live Redis case.
Current source repairs array shape and supported mutable copies, setter expiry
across local/external hydration, and stale external reads that could delete a
newer writer's record. Invalid reads now miss without deletion. Legacy entries
without expiry retain their earlier age semantics; arbitrary mutable CLR
objects and cyclic graphs are not qualified.

The public unit suite now expands to22 cases:21 pass, none fail and one live
Redis case is explicitly skipped. Ordinary fixtures force memory before import
and restore six process environment values. Live Redis requires explicit
`EnableRedisIntegration`, bounded probing and its own GUID key prefix; the
tests do not clear ambient keys. Production JSON decoding, cloning, expiry and
getter behavior execute directly; external transport/status alone are mocked.

Root actual evidence is `.pcai/integration/cache-correctness-root-actual-r1/`;
distinct source/test review is
`.pcai/integration/cache-correctness-regression-tests-peer-r2/`. A failed initial
positive container and the nine detecting predecessor failures remain retained.
Current ProcessCapture registers and the portable independent
`Tests/ProcessCapture/source-review.json` bind these22 controls. Current Carbon
qualification passed18 process-capture controls and21 full-module cache cases
with one explicit Redis skip; independent review accepts source ancestry,
physical XML identities and Root-attributed original resource closure. See
`.pcai/integration/maintained-current-cache-execution-actual-peer-r2/review.json`.
The first wrapper's separator admission failure and reviewed-source miss remain
preserved separately; that failure occurred before the canonical cache suite.

These are correctness results. No equivalent-semantics speedup, live Redis,
fleet deployment or new current-head CI acceptance is claimed. Published b1cc1
CI passes2,809 tests with40 skips but fails the unchanged85% floor at65.271716%.

## Earlier namespace and dependency repair

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
