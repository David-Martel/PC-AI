# Tooling modernization review — October 2, 2026

## Review basis

Reviewed Claude ASUS's F2 research proposal and fleet plan, two independent
documentation/build reviews, live repository configuration and Windows command
resolution. The F2 copy matches its remote SHA-256:
`2d0d766a50ab729b2e427f962db29801b7795bcb2f5137a85a8ed046d0dd8a4b`.
The fleet-plan snapshot matches
`677cd74740fefae1939c5845df5d4e5fff1496b4e51856585de816a4f3742bfb`.
Copies and raw review evidence remain outside Git in the October 2 closeout packet.

The local Codex coordination lane supports additive documentation pilots,
existing registry reuse and separate host/HIL decisions. Recommendations were
returned to Claude on `toolchain-proposal-review-20261002`. Package-manager
selection, changed dependency floors and fleet convergence remain owner decisions.

## Decisions and order

| Priority | Work | Decision and next evidence |
|---|---|---|
| 1 | Installed-tool census and git-guard adapter | Continue report-only. Consume the canonical fleet-version checker; preserve missing tools, unknown versions, shadowing, lane errors and incomplete coverage. Record source hash, host, command context and resolved executable. Review actual checker/adapter heads before enforcement. |
| 1 | Sphinx-Needs | Proceed with an isolated generated-view pilot over the existing CSVs. Complete field/link preservation and strict rendering before publishing it as traceability evidence. |
| 1 | Existing Sphinx, MyST, rosdoc2 and Vale | Extend DOC-TOOL-008/009/010/005 and maintained recipes. They already exist here; no duplicate publisher or tool registry is needed. |
| 2 | nextest and pytest-xdist | Pilot one Rust suite and one isolated Python suite. Compare collected cases and outcomes, not timing alone; retain doctests and meaningful failure controls. |
| 2 | Compiler/linker and BuildKit caches | Census first, then disposable configurations with bounded resources, trusted cache writers and cold/warm/invalidation controls. Select each platform separately. |
| Deferred | StrictDoc authoring migration, broad mise replacement, MkDocs retirement | No demonstrated need to replace the canonical CSV workflow or current host managers. An optional generated export can be evaluated separately. The F2 MkDocs EOL claim is unverified. |
| Deferred | labgrid deployment and automatic retry acceptance | First define real adapters, reservations, selected hosts and cleanup faults. Hardware remains outside ordinary quality checks; a later pass does not erase the first failure. |

## Corrections required in F2

The header's corrections must reach its body, tables and install examples. The
review verified Sphinx-Needs 8.5.0 and StrictDoc 0.30.1 package metadata; these are
candidates, not installed or approved targets. rosdoc2's verified PyPI version is
0.2.1; the proposed 8.2.6 number is not established. Replace unbound percentages,
architecture-parity assertions and “no breaking changes” claims with measured
results or explicit hypotheses.

Operational A/R/M/F labels do not establish IEC software safety classes or SOUP
dispositions. Particular documentation frameworks are not established medical
device requirements by this proposal. Its invented TRACE/DOC/HIL/BUILD identifiers
are proposal labels, not registered requirement or test links. Reuse applicable
outcomes and record intended use, failure effects and tool assurance. FDA's current
guidance recommends assurance proportionate to automation risk. [FDA, February 2026](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software).

Specific recipe corrections:

- Cargo does not document `build.rustc-link-arg`. Select a compatible Linux
  compiler driver and target-scoped Rust flags for a mold pilot; preserve Windows
  MSVC settings. [Cargo configuration](https://doc.rust-lang.org/cargo/reference/config.html).
- Replace the invented ccache `[remote]` stanza with the selected release's
  documented storage configuration. Resolve the 6380/6379 conflict. A coordination
  Redis database is not an approved build-cache service; namespaces alone do not
  isolate memory capacity or eviction. [ccache manual](https://ccache.dev/manual/4.14.1.html).
- nextest runs Rust tests in separate processes; it does not directly accelerate
  colcon's Python/C++ packages. Retain a separate doctest route and explicitly
  choose failure-collection behavior. [nextest execution model](https://nexte.st/docs/design/why-process-per-test/).
- Use documented colcon mixins and CMake generator selection, with separate build
  directories. Do not multiply package concurrency by unrestricted pytest workers.
  Session fixtures run per xdist worker. [colcon mixins](https://colcon.readthedocs.io/en/released/reference/mixin-arguments.html),
  [xdist fixtures](https://pytest-xdist.readthedocs.io/en/stable/how-to.html).
- Preserve existing property/fuzz coverage. Do not adopt the speculative 5% flaky
  allowance; diagnostic retries must retain failure and attempt provenance.
- Exact versions need platform-specific artifact verification and a stated
  fallback policy. Mutable download URLs and unpinned installs do not provide it.

## Windows findings and PATH decision

Current PowerShell resolves Ruff 0.16.10 at `C:/Users/david/.local/bin/ruff.exe`,
pre-commit 4.6.2 in the same directory and just 1.58.0 at
`C:/Users/david/bin/just.exe`. Python, CMake and Ninja resolve through existing
`C:/Users/david/bin` wrappers. The git-guard checkout is clean at the verified
signed v0.2.4 tag; this does not establish that every repository uses that hook.
TPM and agent-bus both have repository-local `core.hooksPath` settings.

Keep the global user PATH order unchanged. Evaluate changes in fresh PowerShell,
noninteractive SSH and runner contexts, recording before/after resolution and
versions for all affected tools. Prefer scoped launchers or environments when a
project needs a different compiler/tool version; do not silently change Python,
CMake or Ninja while correcting Ruff resolution.

The claim that every just recipe requires `sh` is contradicted by this repo's
`windows-shell` PowerShell setting and successful `just --dry-run lint` under
1.58.0. Bash-shebang recipes such as setup require their own Git Bash preflight.
Native Git Bash and WSL are separate environments; test the intended launcher.

Claude disclosed that the git-guard update landed after the requested gate freeze.
Preserve that timing in the external receipt. The normal agent-bus push later
passed with the recorded state; no mid-gate rollback or hook bypass occurred.
The earlier routing failures remain preserved and unexplained. New tool changes
must begin after the affected lane has finished its gates.

## Documentation pilot: acceptance before publication

The existing exporter emits directive text, but `docs/sphinx/conf.py` does not
enable Sphinx-Needs. The exporter omits `user_need_id`, foundational/aspirational criteria,
verification/validation methods, owner, source and evidence fields. Its WI, risk
and instrument links also need actual rendered/imported target objects.
Successful formatter tests do not establish a usable traceability site.

- Use the existing `requirements_dag` generator and `just docs-sphinx-strict`
  route. Pin the candidate dependency and prove resolution/build compatibility
  in the actual Windows and Linux documentation environments.
- Preserve the canonical fields above, unknown values and raw evidence states.
  Configure existing hyphenated IDs and explicit fields/links; do not suppress
  dangling links to make a build green. [Sphinx-Needs configuration](https://sphinx-needs.readthedocs.io/en/stable/configuration.html).
- Render a closed, reviewed slice including requirement, WI, risk, test oracle and
  selected source/evidence links. Keep the full-census gap report accessible.
- Run graph, structural traceability and source-pin validation before direct or
  synchronized rendering. The current synchronizer does not itself establish
  these checks; CLI generation must not bypass invariants through early return.
- Negative controls must reject duplicate IDs, missing targets, changed source
  pins and dropped criteria. Repeated exports must leave input bytes unchanged
  and produce equivalent outputs.
- Inspect developer navigation and a stakeholder-facing export: readable tables,
  direct evidence links, visible failed/unknown states and clear review boundaries.
  Shared Drive/Office publication still requires current comment/revision review
  and visual inspection of the final artifact.

Track qualification in [the existing V&V backlog](../../v_v_testing.TODO.md#standards-and-tool-probe-evidence-handoff-proposed-2026-09-27)
and [closed-loop contract](../TRACEABILITY_CLOSED_LOOP_CONTRACT.md). Tool installation
does not advance a requirement result or authorize a new dependency floor.

## Accelerator pilot: acceptance before convergence

Use disposable build directories/images and the same source, toolchain, target,
features and test selection for comparisons. Record cold/warm wall time and
resource use; change a header, compiler flag and target to prove cache invalidation.
Include a planted failing test and a corrupted artifact to prove failure reporting
and artifact rejection. Do not mix CargoTools and native Cargo feature/cache
profiles and call their timing difference a speedup.

For Rust, account for sccache's disabled-incremental prerequisite and linking
crates that cannot be cached. The ccache 4.14.1 built-in Redis backend is
deprecated; select a supported storage design before a remote-cache pilot.
[sccache Rust constraints](https://github.com/mozilla/sccache/blob/main/docs/Rust.md),
[ccache storage backends](https://ccache.dev/manual/4.14.1.html).

Keep untrusted PR writers outside release caches. Require dedicated cache
capacity, unavailable-cache fallback, ownership and credential handling before
remote deployment. Fleet changes follow the existing campaign sequencing and
Ansible/image route. CMake caps and other compatibility constraints need explicit
lane treatment; implementing a recorded minimum and changing a floor are different
decisions. Preserve installed versus launcher-loaded identities and per-host scope.

## Follow-up fleet plan: decisions still needed

Claude's October 2 follow-up adds four proposals. The response sent on the same
coordination thread sets these boundaries:

- Keep the current Spark CMake compatibility seal until a disposable CMake 4
  comparison establishes ROS/OpenCV behavior and migration cost. Compare host
  managers with the actual package, compiler and rollback requirements before
  choosing one.
- Version intelligence may propose updates in report-only mode. Retain the
  canonical policy, source identities and review route; operational tool classes
  do not by themselves authorize automatic policy or safety-related updates.
- Measure worktree/PR queues first. Treat three worktrees and four PRs as proposed
  warning levels, not approved blocking limits. Count ownership, dependent PRs,
  active custody and justified infrastructure exceptions. Mandatory caps or hooks
  need a reviewed rollout that cannot strand preserved work.
- Inventory local model/KG services without loading GPUs reserved for procedure
  or HIL work. Loading and deployment require the affected owners' agreement.

The ASUS owner reports that the host census PR #334 still needs failed-probe,
empty-selection and interpreter-attribution corrections. Its adapter must retain
unknown findings and incomplete coverage even when the checker exits zero. Review
the actual corrected heads and negative controls before linking these outputs as
qualified evidence; the report is not a new enforcement minimum.
